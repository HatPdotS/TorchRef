"""The information half of a :class:`~torchref.model.model.Model`.

:class:`ModelContext` holds what a model is loaded from and sits in -- cell, space
group, atom identity (:attr:`ModelContext.topology`, node-only), link records,
provenance, hydrogen policy and the geometry restraints -- as opposed to what is
refined, which lives only in the model's parameter wrappers. An atom table is read once,
by :meth:`ModelContext.from_atoms`, which splits it into the topology and the
:class:`AtomValues` the wrappers are built from; every other way of making a model goes
through :meth:`ModelContext.derive`.

Mutable by design; prefer :meth:`ModelContext.copy` over editing in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from torchref.config import get_int_dtype
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
            "hydrogen_mode='riding' requires hydrogens='keep' or 'add': you threw "
            "the hydrogens overboard and then asked them to ride."
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
        :meth:`derive`).

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
            For an invalid hydrogen policy, including ``strip`` with ``riding``, or a
            table whose ``model_num`` column holds more than one model.
        """
        from torchref.symmetry import Cell
        from torchref.topology import Topology

        if "model_num" in pdb.columns:
            models = sorted(int(n) for n in pdb["model_num"].dropna().unique())
            if len(models) > 1:
                raise ValueError(
                    f"The atom table holds model_num {models}, every atom once per "
                    "model, but a Model takes the rows of a single model. Select "
                    "those first, or load an IHM ensemble with "
                    "ModelCollection.from_ihm."
                )
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
        """Apply the hydrogen policy to ``topology`` and ``values``; finish up."""
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

    def _add_missing_hydrogens(
        self, values: AtomValues, dtype: torch.dtype
    ) -> AtomValues:
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
        """Replace the restraint dictionary path and drop restraints built over it.

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

    def _residue_parts(self) -> List[Tuple[int, Dict[Tuple[str, str], List[int]]]]:
        """Each residue's atom rows, split by ``(resname, altloc)``, in a fixed order.

        A residue is a topology residue, ``(chain, resseq, icode)``, the identity the
        restraints use: 100 and 100A are two residues, while alternates with different
        residue names at one position (microheterogeneity) are parts of one. Residues
        are sorted by ``(resname, resseq, chain, icode)`` and parts by
        ``(resname, altloc)``, so occupancy groups are numbered the same whatever
        order the file lists its residues in. A blank altloc is ``" "``.
        """
        residues = self.topology.residues
        resname = self.topology.columns()["resname"]
        altloc = self.topology.atoms.altloc
        order = sorted(
            range(residues.n_residues),
            key=lambda r: (
                str(residues.resname[r]),
                int(residues.resseq[r]),
                str(residues.chain[r]),
                str(residues.icode[r]),
            ),
        )
        out = []
        for residue in order:
            parts: Dict[Tuple[str, str], List[int]] = {}
            for row in residues.atom_rows(residue):
                parts.setdefault((str(resname[row]), str(altloc[row])), []).append(row)
            out.append((residue, {key: parts[key] for key in sorted(parts)}))
        return out

    @staticmethod
    def _conformers(parts: Dict[Tuple[str, str], List[int]]) -> Dict[str, List[int]]:
        """A residue's atom rows per altloc label, or ``{}`` below two labels."""
        rows: Dict[str, List[int]] = {}
        for (_, label), part in parts.items():
            if label != " ":
                rows.setdefault(label, []).extend(part)
        if len(rows) < 2:
            return {}
        return {label: sorted(rows[label]) for label in sorted(rows)}

    def altloc_residues(self) -> List[Tuple[int, List[str], Dict[str, List[int]]]]:
        """The residues that carry more than one conformer.

        Occupancy grouping, :attr:`altloc_pairs` and ``Model.strip_altlocs`` all take
        a residue's conformers from here.

        Returns
        -------
        list of tuple
            ``(residue, labels, rows)`` per topology residue with at least two altloc
            labels, sorted by residue name, then ``resseq``, ``chain`` and ``icode``:
            the residue index in :attr:`topology`, its sorted labels, and each label's
            atom rows in ascending order. A conformer includes every residue name it
            carries; blank-altloc atoms belong to no conformer.
        """
        out = []
        for residue, parts in self._residue_parts():
            rows = self._conformers(parts)
            if rows:
                out.append((residue, list(rows), rows))
        return out

    def occupancy_groups(
        self, initial_occ: torch.Tensor
    ) -> Tuple[torch.Tensor, List[tuple]]:
        """Sharing groups and altloc groups for an
        :class:`~torchref.model.parameter_wrappers.OccupancyTensor` over these atoms.

        Every conformer of a residue with several altlocs is one group, whatever its
        atoms' occupancies, so the sum-to-1 normalization over a residue's conformers
        acts on whole conformers. Every other part of a residue -- its blank-altloc
        atoms, or, in a residue with at most one altloc label, its atoms split by
        residue name and altloc -- is one group when its occupancies agree to within
        0.01 and one group per atom otherwise. No group spans two residues; a starting
        occupancy changes only where the atoms of one group disagree (a conformer's
        atoms, or a part's within the deadband), which collapse to one shared value.
        Which groups are refinable is not decided here but by
        ``Model.set_default_masks`` (occupancy below 0.999).

        Parameters
        ----------
        initial_occ : torch.Tensor
            Occupancies, shape ``(n_atoms,)``.

        Returns
        -------
        sharing_groups : torch.Tensor
            Group index per atom, shape ``(n_atoms,)``, contiguous from 0: the
            conformers first, then the other parts, both in the residue order of
            :meth:`altloc_residues`.
        altloc_groups : list of tuple
            Per residue with several conformers, the atom rows of each conformer.

        Raises
        ------
        ValueError
            If ``initial_occ`` does not hold one value per atom.
        """
        n_atoms = len(initial_occ)
        if n_atoms != self.n_atoms:
            raise ValueError(
                f"initial_occ has {n_atoms} values for a context of {self.n_atoms} atoms"
            )
        sharing_groups = torch.full((n_atoms,), -1, dtype=get_int_dtype())
        altloc_groups = []
        others = []
        for _, parts in self._residue_parts():
            conformers = self._conformers(parts)
            if conformers:
                altloc_groups.append(tuple(conformers.values()))
            others.extend(
                part
                for (_, label), part in parts.items()
                if not conformers or label == " "
            )

        n_groups = 0
        for conformers in altloc_groups:
            for rows in conformers:
                sharing_groups[rows] = n_groups
                n_groups += 1
        for rows in others:
            occ = initial_occ[rows]
            if occ.max().item() - occ.min().item() <= 0.01:
                sharing_groups[rows] = n_groups
                n_groups += 1
            else:
                sharing_groups[rows] = torch.arange(
                    n_groups, n_groups + len(rows), dtype=get_int_dtype()
                )
                n_groups += len(rows)

        if self.verbose > 1:
            print("\nOccupancy Setup:")
            print(f"  Total atoms: {n_atoms}")
            print(f"  Collapsed indices: {n_groups}")
            print(f"  Alternative conformation groups: {len(altloc_groups)}")
            print(f"  Compression ratio: {n_atoms / max(n_groups, 1):.2f}x")

        return sharing_groups, altloc_groups

    def register_altlocs(self) -> None:
        """Rebuild :attr:`altloc_pairs` from the topology's altlocs.

        One tuple per residue that has multiple conformations, holding one index tensor
        per conformation (in sorted altloc order), e.g.
        ``[(tensor([100, 101]), tensor([110, 111])), ...]``. Overwrites any previous
        content, so call it after the atom numbering changes.
        """
        self.altloc_pairs = [
            tuple(
                torch.tensor(rows_by_altloc[label], dtype=get_int_dtype())
                for label in labels
            )
            for _, labels, rows_by_altloc in self.altloc_residues()
        ]

    @property
    def chain_sequences(self) -> List[Tuple[str, str]]:
        """Per-chain one-letter sequences, ``[(chain_id, sequence), ...]``.

        Over the residues :func:`~torchref.topology.residue_graph.polymer_type`
        classes as protein or nucleic acid, whatever their record type; numbering gaps
        become ``?`` and unrecognized residues ``X``.
        """
        result = []
        for chain, residues in self._polymer_residues():
            seq_chars = []
            for i, (resseq, _, resname) in enumerate(residues):
                if i > 0:
                    gap = resseq - residues[i - 1][0] - 1
                    if gap > 0:
                        seq_chars.extend(["?"] * gap)
                seq_chars.append(THREE_TO_ONE.get(resname, "X"))
            result.append((chain, "".join(seq_chars)))
        return result

    def _polymer_residues(self) -> List[Tuple[str, List[Tuple[int, str, str]]]]:
        """``(chain, [(resseq, icode, resname), ...])`` over polymer residues.

        Chains in file order, one entry per ``(resseq, icode)``, sorted by ``resseq``
        (stably, so insertion codes keep their file order).
        """
        from torchref.topology.residue_graph import polymer_type

        if self.topology is None:
            return []
        residues = self.topology.residues
        chains: Dict[str, Dict[tuple, Tuple[int, str, str]]] = {}
        for r in np.nonzero(polymer_type(residues.resname) != "")[0]:
            chain, resseq, icode = residues.key(int(r))
            seen = chains.setdefault(chain, {})
            seen.setdefault((resseq, icode), (resseq, icode, str(residues.resname[r])))
        return [
            (chain, sorted(seen.values(), key=lambda item: item[0]))
            for chain, seen in chains.items()
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
            duplicate.restraints = self.restraints.copy()
        return duplicate

    def state(self) -> Dict[str, Any]:
        """What :meth:`from_state` needs besides the atom table, as picklable entries.

        Returns
        -------
        dict
            The cell as a CPU tensor, the space group as its extended Hermann-Mauguin
            symbol (``gemmi.SpaceGroup`` is not picklable), the altloc groups, a copy
            of the link records and the settings other than ``verbose``. The atom
            table itself is written by the model, which alone has the current values;
            restraints are not saved, they rebuild.
        """
        settings = self.settings()
        del settings["verbose"]
        return {
            "cell": self.cell.data.cpu() if self.cell is not None else None,
            "spacegroup": self.spacegroup.xhm if self.spacegroup else None,
            "initialized": self.initialized,
            "altloc_pairs": self.altloc_pairs,
            "links": _copy_links(self.links),
            **settings,
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
            links=state.pop("links", None),
            input_file=state.pop("input_file", None),
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
