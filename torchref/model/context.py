"""The information half of a :class:`~torchref.model.model.Model`.

:class:`ModelContext` holds what a model *is loaded from* and *sits in* -- the unit
cell, the space group, the atom table, the link records, the provenance and the
hydrogen policy -- as opposed to what is being refined, which stays on the model as
parameter wrappers and per-atom buffers. The geometry restraints belong here too: they
are fixed by the atom set and the dictionaries, and are evaluated against coordinates
the caller passes in.

Atom identity lives on :attr:`ModelContext.topology`, a node-only
:class:`~torchref.topology.Topology`; refinable values never live here. An atom table
(a pandas DataFrame) is read only at construction: :meth:`ModelContext.from_atoms`
settles it -- unusable rows dropped, hydrogens stripped or generated, the crystal built
-- and splits it into the topology and an :class:`AtomValues` bundle of starting values
that the model's parameter wrappers are built from. Every way of making a model --
loading a file, selecting, stripping, hydrogenating, restoring a state dict -- produces
a context and values first, and only then installs wrappers over them.

Splitting it out means the crystallographic context can be passed to code that needs
only that (structure-factor engines, scalers, most targets) without handing over the
refinable state, and it keeps the model's own surface to parameters and behaviour.

Mutable by design; prefer :meth:`ModelContext.copy` over editing in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from torchref.utils.device_mixin import DeviceMixin

if TYPE_CHECKING:
    import pandas

    from torchref.symmetry import Cell, SpaceGroup
    from torchref.topology import Topology
    from torchref.topology.restraints import Restraints

#: Three-letter residue code to one-letter code, modified residues included.
THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    # Common modified residues
    "MSE": "M",
    "CSE": "C",
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
}

#: What to do with the hydrogens of the input atom table.
HYDROGEN_SOURCES = ("keep", "add", "strip")

#: How hydrogen rows are parametrised.
HYDROGEN_MODES = ("atoms", "riding")

#: Fields a derived context inherits from the one it was derived from.
_SETTINGS = (
    "cif_path",
    "verbose",
    "hydrogens",
    "hydrogen_mode",
    "hydrogens_in_xray",
    "input_file",
)


def check_hydrogen_policy(hydrogens: str, hydrogen_mode: str) -> None:
    """Raise ``ValueError`` unless ``(hydrogens, hydrogen_mode)`` is a valid pair.

    Parameters
    ----------
    hydrogens : {"keep", "add", "strip"}
    hydrogen_mode : {"atoms", "riding"}
    """
    if hydrogens not in HYDROGEN_SOURCES:
        raise ValueError(
            f"hydrogens must be one of {HYDROGEN_SOURCES}, got {hydrogens!r}"
        )
    if hydrogen_mode not in HYDROGEN_MODES:
        raise ValueError(
            f"hydrogen_mode must be one of {HYDROGEN_MODES}, got {hydrogen_mode!r}"
        )
    if hydrogens == "strip" and hydrogen_mode == "riding":
        raise ValueError(
            "hydrogen_mode='riding' with hydrogens='strip': you threw the hydrogens "
            "overboard and then asked them to ride. Nothing is left to ride -- use "
            "hydrogens='keep' or hydrogens='add'."
        )


def _copy_links(links):
    """An independent copy of a reader's LINK records (a DataFrame or a sequence)."""
    if links is None:
        return None
    return links.copy() if hasattr(links, "copy") else list(links)


def own_spacegroup(value, dtype: torch.dtype, device) -> Optional["SpaceGroup"]:
    """A space group owned by the caller, on ``device`` and in ``dtype``.

    An incoming :class:`~torchref.symmetry.SpaceGroup` is copied rather than shared,
    because ``.to()`` moves in place and would otherwise relocate the caller's object.

    Parameters
    ----------
    value : SpaceGroup, gemmi.SpaceGroup, str, int or None
        Anything :class:`~torchref.symmetry.SpaceGroup` accepts.
    dtype : torch.dtype
    device : torch.device

    Returns
    -------
    SpaceGroup or None
    """
    from torchref.symmetry import SpaceGroup

    if value is None:
        return None
    if isinstance(value, SpaceGroup):
        return value.copy().to(device=device, dtype=dtype)
    # SpaceGroup falls back to the global default device otherwise, which would plant
    # accelerator-resident matrices on a CPU-pinned model.
    return SpaceGroup(value, dtype=dtype, device=device)


#: Columns of an atom table that are parameter values rather than identity.
_U_COLUMNS = ("u11", "u22", "u33", "u12", "u13", "u23")


@dataclass(eq=False)
class AtomValues:
    """Starting values for the parameter wrappers, one row per atom.

    Read from an atom table at construction and consumed by
    ``Model._install_parameters``; afterwards the wrappers are the only source of these
    values.

    Parameters
    ----------
    xyz : numpy.ndarray
        Cartesian coordinates in Å, shape ``(N, 3)``.
    b : numpy.ndarray
        Isotropic B-factors in Å², shape ``(N,)``.
    u : numpy.ndarray
        Anisotropic U in Å², shape ``(N, 6)`` as ``u11 u22 u33 u12 u13 u23``; NaN for
        isotropic atoms.
    occupancy : numpy.ndarray
        Occupancies, shape ``(N,)``.
    aniso : numpy.ndarray
        True for atoms carrying an ANISOU record, shape ``(N,)``.
    """

    xyz: np.ndarray
    b: np.ndarray
    u: np.ndarray
    occupancy: np.ndarray
    aniso: np.ndarray

    @classmethod
    def from_table(cls, pdb: "pandas.DataFrame") -> "AtomValues":
        """The value columns of an atom table; missing ANISOU columns read as NaN."""
        n = len(pdb)
        u = np.full((n, 6), np.nan)
        for i, column in enumerate(_U_COLUMNS):
            if column in pdb.columns:
                u[:, i] = pdb[column].to_numpy(dtype=np.float64)
        aniso = (
            pdb["anisou_flag"].to_numpy(dtype=bool)
            if "anisou_flag" in pdb.columns
            else np.zeros(n, dtype=bool)
        )
        return cls(
            xyz=pdb[["x", "y", "z"]].to_numpy(dtype=np.float64),
            b=pdb["tempfactor"].to_numpy(dtype=np.float64),
            u=u,
            occupancy=pdb["occupancy"].to_numpy(dtype=np.float64),
            aniso=aniso,
        )

    def __len__(self) -> int:
        return len(self.b)

    def gather(self, rows: np.ndarray) -> "AtomValues":
        """Values for the atoms ``rows`` names, in that order; rows may repeat."""
        rows = np.asarray(rows, dtype=np.int64)
        return AtomValues(
            xyz=self.xyz[rows].copy(),
            b=self.b[rows].copy(),
            u=self.u[rows].copy(),
            occupancy=self.occupancy[rows].copy(),
            aniso=self.aniso[rows].copy(),
        )


@dataclass(eq=False, repr=False)
class ModelContext(DeviceMixin):
    """Crystallographic context, atom bookkeeping and provenance for one model.

    Parameters
    ----------
    cell : Cell or None
        Unit cell, or None before a structure is loaded.
    spacegroup : SpaceGroup or None
        Space group, or None before a structure is loaded.
    topology : Topology or None
        Atom identity -- names, elements, altlocs, residues, chains, record types --
        as a node-only topology (no edges). Replaced, never edited, when the atom set
        changes.
    links : list or None
        Link records from the reader, used to build inter-residue restraints.
    altloc_pairs : list
        Index groups of alternative conformations, one tuple of index tensors per
        residue with more than one conformation; rebuilt by :meth:`register_altlocs`.
    input_file : str or None
        Path the structure was loaded from.
    z_value : int or None
        The CRYST1 Z of the input file, written back unchanged.
    cif_path : str or list of str or None
        Restraint dictionary path(s). Change it with :meth:`set_cif_path`, which drops
        restraints built over the old dictionaries.
    verbose : int, default 1
        Verbosity level.
    hydrogens : {"keep", "add", "strip"}, default "keep"
        What :meth:`from_atoms` does with the input's hydrogens: keep what the file
        has, additionally generate the ones the monomer templates name and the file
        lacks (waters included), or remove them all.
    hydrogen_mode : {"atoms", "riding"}, default "atoms"
        How hydrogen rows are parametrised: as ordinary refinable atoms, or riding on
        their parent heavy atoms (rebuilt from them every forward, not refined).
        ``"riding"`` with ``hydrogens="strip"`` raises ``ValueError``: there is
        nothing left to ride.
    hydrogens_in_xray : bool, default True
        Whether hydrogens enter the structure-factor calculation. Restraints and the
        non-bonded term see them either way; the bulk-solvent mask never does.
    initialized : bool, default False
        Whether a structure has been loaded. ``if model:`` tests this.
    restraints : Restraints or None
        Geometry restraints over ``topology``, or None until :meth:`build_restraints`
        runs.
        Reset to None whenever the atom table or ``cif_path`` changes; read them
        through ``Model.restraints``, which builds on first access.

    Notes
    -----
    Deliberately does **not** carry the device or float dtype. Those are live
    :class:`~torchref.utils.device_mixin.DeviceMixin` trackers that the traversal
    rewrites in place on the object that owns the tensors, so they stay on the model
    rather than becoming a second source of truth here.

    Holds no refinable parameters, so this is a dataclass rather than an
    ``nn.Module``.
    """

    cell: Optional["Cell"] = None
    spacegroup: Optional["SpaceGroup"] = None
    topology: Optional["Topology"] = None
    links: Optional[List[Any]] = None
    altloc_pairs: List[Any] = field(default_factory=list)
    input_file: Optional[str] = None
    z_value: Optional[int] = None
    cif_path: Optional[Any] = None
    verbose: int = 1
    hydrogens: str = "keep"
    hydrogen_mode: str = "atoms"
    hydrogens_in_xray: bool = True
    initialized: bool = False
    restraints: Optional["Restraints"] = None

    def __post_init__(self) -> None:
        check_hydrogen_policy(self.hydrogens, self.hydrogen_mode)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def settings(self) -> Dict[str, Any]:
        """The policy fields a context derived from this one inherits.

        Returns
        -------
        dict
            ``cif_path``, ``verbose``, ``hydrogens``, ``hydrogen_mode``,
            ``hydrogens_in_xray`` and ``input_file``, ready to pass to
            :meth:`from_atoms`.
        """
        return {name: getattr(self, name) for name in _SETTINGS}

    @classmethod
    def from_atoms(
        cls,
        pdb: "pandas.DataFrame",
        cell,
        spacegroup,
        *,
        dtype: torch.dtype,
        device,
        links=None,
        **settings,
    ) -> Tuple["ModelContext", AtomValues]:
        """Settle an atom table and split it into a context and starting values.

        Rows without coordinates, B-factor or occupancy are dropped, the table is split
        into identity (:meth:`Topology.from_table`) and :class:`AtomValues`, the cell
        and space group are built, and the hydrogen policy is applied (see
        :meth:`derive`). This is the only place a model's atoms are read from a table.

        Parameters
        ----------
        pdb : pandas.DataFrame
            Atom table as read. Not modified.
        cell : Cell or array-like
            ``[a, b, c, alpha, beta, gamma]`` in Å and degrees, or a Cell to copy.
        spacegroup : SpaceGroup, gemmi.SpaceGroup, str or int
        dtype : torch.dtype
            Float dtype of the cell and space-group tensors.
        device : torch.device
            Where the cell and space-group tensors live.
        links : list, optional
            Link records from the reader.
        **settings
            Any of the fields :meth:`settings` returns.

        Returns
        -------
        ctx : ModelContext
            Initialized, with no restraints built yet.
        values : AtomValues
            Starting values, row-aligned with ``ctx.topology``.

        Raises
        ------
        ValueError
            For an invalid hydrogen policy, including ``strip`` with ``riding``.
        """
        from torchref.symmetry import Cell
        from torchref.topology import Topology

        z_value = getattr(pdb, "attrs", {}).get("z")
        ctx = cls(links=links, z_value=z_value, **settings)
        pdb = pdb.dropna(subset=["x", "y", "z", "tempfactor", "occupancy"])
        pdb = pdb.reset_index(drop=True)
        ctx.cell = Cell(
            cell.data if isinstance(cell, Cell) else cell, dtype=dtype, device=device
        )
        ctx.spacegroup = own_spacegroup(spacegroup, dtype, device)
        ctx.topology = Topology.from_table(pdb)
        values = ctx._settle(AtomValues.from_table(pdb), dtype)
        return ctx, values

    def derive(
        self, topology: "Topology", values: AtomValues, **overrides
    ) -> Tuple["ModelContext", AtomValues]:
        """A new context over ``topology`` in this one's crystal, with its settings.

        The hydrogen policy is applied to the new atoms: ``"strip"`` removes every
        hydrogen, ``"add"`` generates the ones the monomer templates name and the atoms
        lack (waters included), ``"keep"`` leaves them as they are.

        Parameters
        ----------
        topology : Topology
            Identity of the new atom set, node-only.
        values : AtomValues
            Its starting values, row-aligned with ``topology``.
        **overrides
            Settings to change, e.g. ``hydrogens="strip"``.

        Returns
        -------
        ctx : ModelContext
        values : AtomValues
            Row-aligned with ``ctx.topology``, which differs from ``topology`` when the
            policy added or removed atoms.
        """
        ctx = ModelContext(
            cell=self.cell.clone() if self.cell is not None else None,
            spacegroup=self.spacegroup.copy() if self.spacegroup is not None else None,
            links=_copy_links(self.links),
            topology=topology,
            z_value=self.z_value,
            **{**self.settings(), **overrides},
        )
        return ctx, ctx._settle(values, self.cell.dtype)

    def _settle(self, values: AtomValues, dtype: torch.dtype) -> AtomValues:
        """Apply the hydrogen policy to ``topology`` and ``values``; finish the context."""
        if self.hydrogens == "strip":
            keep = ~self.topology.atoms.is_hydrogen.cpu().numpy()
            if not keep.all():
                rows = np.nonzero(keep)[0]
                self.topology = self.topology.gather(rows)
                values = values.gather(rows)
        if self.hydrogens == "add":
            values = self._add_missing_hydrogens(values, dtype)
        self.register_altlocs()
        self.initialized = True
        return values

    def _add_missing_hydrogens(self, values: AtomValues, dtype: torch.dtype) -> AtomValues:
        """Top up the hydrogens the atoms are missing; returns the extended values.

        Per parent, not per file: a structure deposited with some hydrogens gets the
        rest, because the plan only ever proposes a hydrogen the template names and the
        atoms do not have (1AK5 arrives with 675 of roughly 2500). A new hydrogen takes
        its parent's occupancy and B-factor, and is isotropic.

        Costs a restraint build without the pair list, because the plan needs the
        connected topology; it is discarded afterwards.
        """
        from torchref.topology.hydrogens import optimise_free_torsions, plan_hydrogens

        xyz = torch.tensor(values.xyz, dtype=dtype)
        restraints = self.build_restraints(xyz, nonbonded=False, verbose=0)
        plan = plan_hydrogens(
            restraints.topology, restraints.cif_dict, xyz, verbose=self.verbose
        )
        if self.verbose > 0 and restraints.missing_residues:
            print(
                "No restraint dictionary for "
                f"{sorted(restraints.missing_residues)}: not hydrogenated. Pass one "
                "with cif_path / --cif."
            )
        if plan.n_hydrogens == 0:
            return values
        optimise_free_torsions(plan, restraints.topology, xyz)
        self.topology, source, _, plan_rows = self.topology.with_hydrogens(plan)
        values = values.gather(source)
        values.xyz[plan_rows] = np.asarray(plan.position, dtype=np.float64)
        values.u[plan_rows] = np.nan
        values.aniso[plan_rows] = False
        if self.verbose > 0:
            print(f"Generated {plan.n_hydrogens} hydrogens")
        return values

    @property
    def n_atoms(self) -> int:
        """Number of atoms; 0 before a structure is loaded."""
        return 0 if self.topology is None else self.topology.n_atoms

    def set_cif_path(self, cif_path) -> None:
        """Replace the restraint dictionary path and drop restraints built over the old one.

        Parameters
        ----------
        cif_path : str or list of str or None
            Restraint dictionary file(s).
        """
        self.cif_path = cif_path
        self.restraints = None

    def build_restraints(
        self, xyz: torch.Tensor, *, nonbonded: bool = True, verbose=None
    ) -> "Restraints":
        """Build restraints over :attr:`topology` and store them on :attr:`restraints`.

        Parameters
        ----------
        xyz : torch.Tensor
            Current Cartesian coordinates in Å, shape ``(n_atoms, 3)``, from the
            model's ``xyz`` wrapper. The restraints land on its device.
        nonbonded : bool, default True
            Build the non-bonded pair list. False is for a throwaway build that needs
            only the topology, and is then **not** stored.
        verbose : int, optional
            Defaults to :attr:`verbose`.

        Returns
        -------
        Restraints
        """
        from torchref.topology.restraints import Restraints

        restraints = Restraints(
            topology=self.topology,
            cif_path=self.cif_path,
            xyz=xyz.detach(),
            cell=self.cell,
            spacegroup=self.spacegroup,
            links=self.links,
            verbose=self.verbose if verbose is None else verbose,
            nonbonded=nonbonded,
        )
        if nonbonded:
            self.restraints = restraints
        return restraints

    # ------------------------------------------------------------------
    # Identity queries
    # ------------------------------------------------------------------

    def _residue_groups(self, with_altloc: bool) -> Dict[tuple, List[int]]:
        """Atom rows grouped by ``(resname, resseq, chain[, altloc])``, keys sorted.

        The key order is the one pandas' sorted ``groupby`` gives. Occupancy groups are
        numbered in it, and checkpoints store occupancies in group space, so it must
        not change.
        """
        columns = self.topology.columns()
        altloc = np.where(columns["altloc"] == " ", "", columns["altloc"])
        keys: Dict[tuple, List[int]] = {}
        for row in range(self.topology.n_atoms):
            key = (
                str(columns["resname"][row]),
                int(columns["resseq"][row]),
                str(columns["chain"][row]),
            )
            if with_altloc:
                key = key + (str(altloc[row]),)
            keys.setdefault(key, []).append(row)
        return {key: keys[key] for key in sorted(keys)}

    def _altloc_residues(self) -> List[Tuple[tuple, List[str], Dict[str, List[int]]]]:
        """Residues with more than one altloc: ``(key, sorted altlocs, rows per altloc)``.

        Keys are ``(resname, resseq, chain)``, sorted; blank-altloc atoms are not part
        of any conformer.
        """
        altloc = self.topology.atoms.altloc
        out = []
        for key, rows in self._residue_groups(with_altloc=False).items():
            by_altloc: Dict[str, List[int]] = {}
            for row in rows:
                if altloc[row] != " ":
                    by_altloc.setdefault(str(altloc[row]), []).append(row)
            if len(by_altloc) > 1:
                labels = sorted(by_altloc)
                out.append((key, labels, {a: by_altloc[a] for a in labels}))
        return out

    def occupancy_groups(self, initial_occ):
        """``(sharing_groups, altloc_groups, refinable_mask)`` for an
        :class:`~torchref.model.parameter_wrappers.OccupancyTensor` over these atoms.

        Altloc conformations share one collapsed index each; other residues share
        one only when their occupancies agree to within 0.01, and an occupancy is
        refinable only if it differs from 1.0 by more than that same deadband.
        """
        n_atoms = len(initial_occ)
        altloc_groups = []
        refinable_mask = torch.zeros(n_atoms, dtype=torch.bool)

        sharing_groups_tensor = torch.arange(n_atoms, dtype=torch.long)  # dtype-ok: arange atom indices (sharing groups); index requires long
        collapsed_idx = 0

        # First pass: altlocs. ALL atoms of one conformation must share a collapsed
        # index whatever their individual occupancies, or the sum-to-1
        # normalization in OccupancyTensor.forward() acts on the wrong group.
        altloc_residues = set()
        for key, labels, rows_by_altloc in self._altloc_residues():
            altloc_residues.add(key)
            conformation_atom_lists = []
            for label in labels:
                indices = rows_by_altloc[label]
                sharing_groups_tensor[indices] = collapsed_idx
                for idx in indices:
                    if abs(initial_occ[idx].item() - 1.0) > 0.01:
                        refinable_mask[idx] = True
                conformation_atom_lists.append(indices)
                collapsed_idx += 1
            altloc_groups.append(tuple(conformation_atom_lists))

        # Second pass: non-altloc residues, sharing by occupancy similarity.
        for key, indices in self._residue_groups(with_altloc=True).items():
            if key[:3] in altloc_residues:
                continue

            residue_occs = initial_occ[indices]

            occ_min = residue_occs.min().item()
            occ_max = residue_occs.max().item()
            occ_mean = residue_occs.mean().item()

            if (occ_max - occ_min) <= 0.01:
                sharing_groups_tensor[indices] = collapsed_idx
                collapsed_idx += 1

                if abs(occ_mean - 1.0) > 0.01:
                    for idx in indices:
                        refinable_mask[idx] = True
            else:
                # Occupancies disagree within the residue: keep atoms independent.
                for idx in indices:
                    if abs(initial_occ[idx].item() - 1.0) > 0.01:
                        refinable_mask[idx] = True

        # Compact to contiguous indices 0..n_collapsed-1.
        unique_indices = torch.unique(sharing_groups_tensor, sorted=True)
        index_map = torch.zeros(n_atoms, dtype=torch.long)  # dtype-ok: index_map atom-index remap; indexing requires long
        for new_idx, old_idx in enumerate(unique_indices):
            mask = sharing_groups_tensor == old_idx
            sharing_groups_tensor[mask] = new_idx

        n_collapsed = len(unique_indices)

        if self.verbose > 1:
            n_groups = n_collapsed
            n_independent = n_atoms - n_collapsed
            n_refinable = refinable_mask.sum().item()
            n_altloc_groups = len(altloc_groups)

            print("\nOccupancy Setup:")
            print(f"  Total atoms: {n_atoms}")
            print(f"  Collapsed indices: {n_collapsed}")
            print(f"  Alternative conformation groups: {n_altloc_groups}")
            print(f"  Refinable atoms: {n_refinable}")
            print(f"  Compression ratio: {n_atoms / n_collapsed:.2f}x")

        return sharing_groups_tensor, altloc_groups, refinable_mask

    def register_altlocs(self) -> None:
        """Rebuild :attr:`altloc_pairs` from the topology's altlocs.

        One tuple per residue that has multiple conformations, holding one index tensor
        per conformation (in sorted altloc order), e.g.
        ``[(tensor([100, 101]), tensor([110, 111])), ...]``. Overwrites any previous
        content, so call it after the atom numbering changes.
        """
        self.altloc_pairs = [
            tuple(
                torch.tensor(rows_by_altloc[label], dtype=torch.long)  # dtype-ok: altloc atom indices; indexing requires long
                for label in labels
            )
            for _, labels, rows_by_altloc in self._altloc_residues()
        ]

    @property
    def chain_sequences(self) -> List[Tuple[str, str]]:
        """Per-chain one-letter sequences, ``[(chain_id, sequence), ...]``.

        HETATM records are excluded, numbering gaps become ``?`` and unrecognized
        residues ``X``.
        """
        result = []
        for chain, residues in self._polymer_residues():
            seq_chars = []
            for i, (resseq, resname) in enumerate(residues):
                if i > 0:
                    gap = resseq - residues[i - 1][0] - 1
                    if gap > 0:
                        seq_chars.extend(["?"] * gap)
                seq_chars.append(THREE_TO_ONE.get(resname, "X"))
            result.append((chain, "".join(seq_chars)))
        return result

    def _polymer_residues(self) -> List[Tuple[str, List[Tuple[int, str]]]]:
        """``(chain, [(resseq, resname), ...])`` over ATOM records, chains in file order.

        One entry per ``(resseq, icode)``, sorted by ``resseq`` (stably, so insertion
        codes keep their file order).
        """
        if self.topology is None:
            return []
        residues = self.topology.residues
        first = residues.atom_start.astype(np.int64)
        polymer = ~self.topology.atoms.is_hetatm[first] if len(first) else []
        chains: Dict[str, Dict[tuple, Tuple[int, str]]] = {}
        for r in np.nonzero(polymer)[0]:
            chain, resseq, icode = residues.key(int(r))
            seen = chains.setdefault(chain, {})
            seen.setdefault((resseq, icode), (resseq, str(residues.resname[r])))
        return [
            (chain, sorted(seen.values(), key=lambda item: item[0]))
            for chain, seen in chains.items()
        ]

    @property
    def chain_residues(self) -> List[Tuple[str, List[str]]]:
        """Per-chain residue names as 3-letter codes, ``[(chain_id, [resname, ...])]``.

        Excludes HETATM records. Unlike :attr:`chain_sequences`, the raw 3-letter codes
        without gap filling; used by the IHM and mmCIF writers.
        """
        return [
            (chain, [resname for _, resname in residues])
            for chain, residues in self._polymer_residues()
        ]

    def copy(self) -> "ModelContext":
        """An independent copy.

        The topology is copied, the cell and space group are cloned and built restraints
        are copied, so nothing is shared with the original. Cloning the
        space group matters because it is a mutable dataclass: sharing the reference
        would let an edit through one model's context reach every model copied from it.

        Returns
        -------
        ModelContext
            New context sharing no mutable state with this one.
        """
        duplicate = ModelContext(
            cell=self.cell.clone() if self.cell is not None else None,
            spacegroup=(
                self.spacegroup.copy() if self.spacegroup is not None else None
            ),
            topology=self.topology.copy() if self.topology is not None else None,
            links=_copy_links(self.links),
            altloc_pairs=[
                tuple(t.clone() for t in group) for group in self.altloc_pairs
            ],
            initialized=self.initialized,
            z_value=self.z_value,
            **self.settings(),
        )
        if self.restraints is not None:
            restraints = self.restraints.copy()
            # Point at the copied crystal rather than the deep-copied duplicates, so the
            # new context is its single owner.
            restraints._cell = duplicate.cell
            restraints._spacegroup = duplicate.spacegroup
            duplicate.restraints = restraints
        return duplicate

    def state(self) -> Dict[str, Any]:
        """What :meth:`from_state` needs besides the atom table, as picklable entries.

        Returns
        -------
        dict
            The cell as a CPU tensor, the space group as its extended Hermann-Mauguin
            symbol (``gemmi.SpaceGroup`` is not picklable), the altloc groups and the
            settings. The atom table itself is written by the model, which alone has
            the current values; restraints are not saved, they rebuild.
        """
        return {
            "cell": self.cell.data.cpu() if self.cell is not None else None,
            "spacegroup": self.spacegroup.xhm if self.spacegroup else None,
            "initialized": self.initialized,
            "cif_path": self.cif_path,
            "altloc_pairs": self.altloc_pairs,
            "hydrogens": self.hydrogens,
            "hydrogen_mode": self.hydrogen_mode,
            "hydrogens_in_xray": self.hydrogens_in_xray,
        }

    @classmethod
    def from_state(
        cls, state: Dict[str, Any], *, dtype: torch.dtype, device, verbose: int = 1
    ) -> Tuple["ModelContext", Optional[AtomValues]]:
        """Rebuild a context, and the saved values, from a model state dict.

        Consumes the entries: every key read is popped off ``state``, so what remains is
        for ``load_state_dict``. The saved atom table (``"pdb"``) is split as at
        construction, but taken as saved -- the hydrogen policy is recorded, not
        re-applied. Checkpoints that predate the policy are mapped:
        ``strip_H`` becomes ``hydrogens``, ``"free"`` becomes ``"atoms"``, and a saved
        riding wrapper (an ``xyz.h_row`` entry) implies ``"riding"``.

        Parameters
        ----------
        state : dict
            A model state dict.
        dtype : torch.dtype
            Float dtype for the cell and space group.
        device : torch.device
        verbose : int, default 1

        Returns
        -------
        ctx : ModelContext
        values : AtomValues or None
            None when the state holds no atoms.
        """
        from torchref.symmetry import Cell
        from torchref.topology import Topology

        hydrogens = state.pop("hydrogens", None)
        strip_h = state.pop("strip_H", True)
        state.pop("add_hydrogens", None)
        if hydrogens is None:
            hydrogens = "strip" if strip_h else "keep"
        mode = state.pop("hydrogen_mode", None)
        if mode not in HYDROGEN_MODES:
            mode = "riding" if state.get("xyz.h_row") is not None else "atoms"
        if hydrogens == "strip" and mode == "riding":
            hydrogens = "keep"

        cell = state.pop("cell", None)
        table = state.pop("pdb", None)
        z_value = None if table is None else getattr(table, "attrs", {}).get("z")
        ctx = cls(
            topology=None if table is None else Topology.from_table(table),
            cell=Cell(cell, dtype=dtype, device=device) if cell is not None else None,
            spacegroup=own_spacegroup(state.pop("spacegroup", None), dtype, device),
            initialized=state.pop("initialized", False),
            cif_path=state.pop("cif_path", None),
            altloc_pairs=state.pop("altloc_pairs", []),
            hydrogens=hydrogens,
            hydrogen_mode=mode,
            hydrogens_in_xray=state.pop("hydrogens_in_xray", True),
            verbose=verbose,
            z_value=z_value,
        )
        return ctx, None if table is None else AtomValues.from_table(table)

    @property
    def crystal_key(self):
        """Value identity of the crystal, or None while cell or space group is unset.

        Returns
        -------
        tuple or None
            ``(cell.key, spacegroup.key)``; hashable, so anything derived from the
            crystal alone can be cached against it.
        """
        if self.cell is None or self.spacegroup is None:
            return None
        return (self.cell.key, self.spacegroup.key)

    def __repr__(self) -> str:
        n_atoms = self.n_atoms
        sg = None if self.spacegroup is None else self.spacegroup.name
        return (
            f"ModelContext(spacegroup={sg!r}, n_atoms={n_atoms}, "
            f"hydrogens={self.hydrogens!r}, hydrogen_mode={self.hydrogen_mode!r}, "
            f"initialized={self.initialized})"
        )


__all__ = [
    "ModelContext",
    "AtomValues",
    "HYDROGEN_SOURCES",
    "HYDROGEN_MODES",
    "check_hydrogen_policy",
]
