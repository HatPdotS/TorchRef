"""The information half of a :class:`~torchref.model.model.Model`.

:class:`ModelContext` holds what a model *is loaded from* and *sits in* -- the unit
cell, the space group, the atom table, the link records, the provenance and the
hydrogen policy -- as opposed to what is being refined, which stays on the model as
parameter wrappers and per-atom buffers. The geometry restraints belong here too: they
are fixed by the atom set and the dictionaries, and are evaluated against coordinates
the caller passes in.

:meth:`ModelContext.from_atoms` is the one place an atom table is settled: hydrogens
stripped or generated, unusable rows dropped, the crystal built. Every way of making a
model -- loading a file, selecting, stripping, hydrogenating, restoring a state dict --
produces a context first and only then installs parameter wrappers over it.

Splitting it out means the crystallographic context can be passed to code that needs
only that (structure-factor engines, scalers, most targets) without handing over the
refinable state, and it keeps the model's own surface to parameters and behaviour.

Mutable by design; prefer :meth:`ModelContext.copy` over editing in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from torchref.utils.device_mixin import DeviceMixin

if TYPE_CHECKING:
    import pandas

    from torchref.symmetry import Cell, SpaceGroup
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


@dataclass(eq=False, repr=False)
class ModelContext(DeviceMixin):
    """Crystallographic context, atom bookkeeping and provenance for one model.

    Parameters
    ----------
    cell : Cell or None
        Unit cell, or None before a structure is loaded.
    spacegroup : SpaceGroup or None
        Space group, or None before a structure is loaded.
    pdb : pandas.DataFrame or None
        The atom table. Refreshed from the model's tensors only by
        ``Model.update_pdb``, so it is stale between refinement steps by design.
    links : list or None
        Link records from the reader, used to build inter-residue restraints.
    altloc_pairs : list
        Index groups of alternative conformations, one tuple of index tensors per
        residue with more than one conformation; rebuilt by :meth:`register_altlocs`.
    input_file : str or None
        Path the structure was loaded from.
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
        Geometry restraints over ``pdb``, or None until :meth:`build_restraints` runs.
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
    pdb: Optional["pandas.DataFrame"] = None
    links: Optional[List[Any]] = None
    altloc_pairs: List[Any] = field(default_factory=list)
    input_file: Optional[str] = None
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
    ) -> "ModelContext":
        """Settle an atom table and build the context around it.

        In order: strip hydrogens (``hydrogens="strip"``); drop rows without
        coordinates, B-factor or occupancy and renumber the ``index`` column; build the
        cell and space group; generate missing hydrogens (``hydrogens="add"``); record
        the alternative conformations.

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
        ModelContext
            Initialized, with no restraints built yet unless hydrogen generation
            needed them (in which case they were built over the table *before*
            generation and discarded).

        Raises
        ------
        ValueError
            For an invalid hydrogen policy, including ``strip`` with ``riding``.
        """
        from torchref.symmetry import Cell

        ctx = cls(links=links, **settings)
        if ctx.hydrogens == "strip":
            pdb = pdb.loc[pdb["element"].str.strip() != "H"]
        # Renumber before deriving ``index``: every consumer uses it to address length-N
        # per-atom tensors positionally, so a gapped index from the drop sends them past
        # the end (roughly one PDB-REDO entry in six loses rows here).
        pdb = pdb.dropna(subset=["x", "y", "z", "tempfactor", "occupancy"])
        ctx.pdb = cls._renumbered(pdb)
        ctx.cell = Cell(
            cell.data if isinstance(cell, Cell) else cell, dtype=dtype, device=device
        )
        ctx.spacegroup = own_spacegroup(spacegroup, dtype, device)
        if ctx.hydrogens == "add":
            ctx._add_missing_hydrogens(dtype)
        ctx.register_altlocs()
        ctx.initialized = True
        return ctx

    def derive(self, pdb: "pandas.DataFrame", **overrides) -> "ModelContext":
        """A new context over ``pdb`` in this one's crystal, with its settings.

        Parameters
        ----------
        pdb : pandas.DataFrame
            The new atom table; see :meth:`from_atoms`.
        **overrides
            Settings to change, e.g. ``hydrogens="strip"``.

        Returns
        -------
        ModelContext
        """
        settings = {**self.settings(), **overrides}
        return ModelContext.from_atoms(
            pdb,
            self.cell,
            self.spacegroup,
            dtype=self.cell.dtype,
            device=self.cell.device,
            links=self.links,
            **settings,
        )

    @staticmethod
    def _renumbered(pdb: "pandas.DataFrame") -> "pandas.DataFrame":
        pdb = pdb.reset_index(drop=True)
        pdb["index"] = pdb.index.to_numpy(dtype=int)
        return pdb

    def _add_missing_hydrogens(self, dtype: torch.dtype) -> None:
        """Top up the hydrogens the atom table is missing.

        Per parent, not per file: a structure deposited with some hydrogens gets the
        rest, because the plan only ever proposes a hydrogen the template names and the
        table does not have (1AK5 arrives with 675 of roughly 2500).

        Costs a restraint build without the pair list, over the table as loaded,
        because the plan needs its topology; it is discarded afterwards.
        """
        from torchref.topology.hydrogens import (
            augment_atom_table,
            optimise_free_torsions,
            plan_hydrogens,
        )

        xyz = torch.tensor(self.pdb[["x", "y", "z"]].values, dtype=dtype)
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
            return
        optimise_free_torsions(plan, restraints.topology, xyz)
        self.pdb = self._renumbered(
            augment_atom_table(self.pdb, plan, restraints.topology)
        )
        if self.verbose > 0:
            print(f"Generated {plan.n_hydrogens} hydrogens")

    # ------------------------------------------------------------------
    # Restraints
    # ------------------------------------------------------------------

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
        """Build restraints over the atom table and store them on :attr:`restraints`.

        Parameters
        ----------
        xyz : torch.Tensor
            Current Cartesian coordinates in Å, shape ``(n_atoms, 3)``; the atom table's
            own columns are stale during refinement. The restraints land on its device.
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
            pdb=self.pdb,
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
    # Atom-table queries
    # ------------------------------------------------------------------

    def occupancy_groups(self, initial_occ):
        """``(sharing_groups, altloc_groups, refinable_mask)`` for an
        :class:`~torchref.model.parameter_wrappers.OccupancyTensor` over this table.

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
        pdb_with_altlocs = self.pdb[self.pdb["altloc"] != ""]
        altloc_residues = set()

        if len(pdb_with_altlocs) > 0:
            grouped_by_residue = pdb_with_altlocs.groupby(
                ["resname", "resseq", "chainid"]
            )

            for (resname, resseq, chainid), group in grouped_by_residue:
                unique_altlocs = sorted(group["altloc"].unique())

                if len(unique_altlocs) > 1:
                    altloc_residues.add((resname, resseq, chainid))
                    conformation_atom_lists = []

                    for altloc in unique_altlocs:
                        altloc_atoms = group[group["altloc"] == altloc]
                        indices = altloc_atoms["index"].tolist()

                        sharing_groups_tensor[indices] = collapsed_idx

                        for idx in indices:
                            if abs(initial_occ[idx].item() - 1.0) > 0.01:
                                refinable_mask[idx] = True

                        conformation_atom_lists.append(indices)
                        collapsed_idx += 1

                    altloc_groups.append(tuple(conformation_atom_lists))

        # Second pass: non-altloc residues, sharing by occupancy similarity.
        grouped = self.pdb.groupby(["resname", "resseq", "chainid", "altloc"])

        for (resname, resseq, chainid, altloc), group in grouped:
            if (resname, resseq, chainid) in altloc_residues:
                continue

            indices = group["index"].tolist()

            if len(indices) == 0:
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
        """
        Rebuild ``self.altloc_pairs`` from the ``altloc`` column.

        One tuple per residue that has multiple conformations, holding one
        index tensor per conformation (in sorted altloc order), e.g.
        ``[(tensor([100, 101]), tensor([110, 111])), ...]``. Overwrites any
        previous content, so call it after the atom numbering changes.
        """
        self.altloc_pairs = []

        pdb_with_altlocs = self.pdb[self.pdb["altloc"] != ""]

        if len(pdb_with_altlocs) == 0:
            return

        grouped = pdb_with_altlocs.groupby(["resname", "resseq", "chainid"])

        for (resname, resseq, chainid), group in grouped:
            unique_altlocs = sorted(group["altloc"].unique())

            # A lone altloc label is not an alternative conformation.
            if len(unique_altlocs) > 1:
                conformation_tensors = []
                for altloc in unique_altlocs:
                    altloc_atoms = group[group["altloc"] == altloc]
                    indices = torch.tensor(
                        altloc_atoms["index"].tolist(), dtype=torch.long  # dtype-ok: altloc atom indices; indexing requires long
                    )
                    conformation_tensors.append(indices)

                self.altloc_pairs.append(tuple(conformation_tensors))

    @property
    def chain_sequences(self) -> List[Tuple[str, str]]:
        """Per-chain one-letter sequences, ``[(chain_id, sequence), ...]``.

        HETATM records are excluded, numbering gaps become ``?`` and unrecognized
        residues ``X``.
        """
        if self.pdb is None:
            return []

        atom_df = self.pdb[self.pdb["ATOM"] == "ATOM"]
        result = []

        for chain in atom_df["chainid"].unique():
            chain_df = atom_df[atom_df["chainid"] == chain]
            residues = chain_df.drop_duplicates(subset=["resseq", "icode"]).sort_values(
                "resseq"
            )
            resseqs = residues["resseq"].values
            resnames = residues["resname"].values

            seq_chars = []
            for i, (rseq, rname) in enumerate(zip(resseqs, resnames)):
                if i > 0:
                    gap = int(rseq) - int(resseqs[i - 1]) - 1
                    if gap > 0:
                        seq_chars.extend(["?"] * gap)
                code = THREE_TO_ONE.get(str(rname).strip(), "X")
                seq_chars.append(code)

            result.append((str(chain), "".join(seq_chars)))

        return result

    @property
    def chain_residues(self) -> List[Tuple[str, List[str]]]:
        """
        Per-chain residue names as 3-letter codes (for IHM/CIF writing).

        Excludes HETATM records. Unlike :attr:`chain_sequences`, returns
        the raw 3-letter codes without gap filling.

        Returns
        -------
        list of (str, list of str)
            Ordered list of ``(chain_id, [resname, ...])``.
        """
        if self.pdb is None:
            return []

        atom_df = self.pdb[self.pdb["ATOM"] == "ATOM"]
        result = []

        for chain in atom_df["chainid"].unique():
            chain_df = atom_df[atom_df["chainid"] == chain]
            residues = chain_df.drop_duplicates(subset=["resseq", "icode"]).sort_values(
                "resseq"
            )
            resnames = [str(r).strip() for r in residues["resname"].values]
            result.append((str(chain), resnames))

        return result

    # ------------------------------------------------------------------
    # Copying and persistence
    # ------------------------------------------------------------------

    def copy(self) -> "ModelContext":
        """An independent copy.

        The atom table is deep-copied, the cell and space group are cloned and built
        restraints are copied, so nothing is shared with the original. Cloning the
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
            pdb=self.pdb.copy(deep=True) if self.pdb is not None else None,
            links=list(self.links) if self.links is not None else None,
            altloc_pairs=[
                tuple(t.clone() for t in group) for group in self.altloc_pairs
            ],
            initialized=self.initialized,
            **self.settings(),
        )
        if self.restraints is not None:
            restraints = self.restraints.copy()
            # Point at the copied table and crystal rather than the deep-copied
            # duplicates, so the new context is the single owner of both.
            restraints.pdb = duplicate.pdb
            restraints._cell = duplicate.cell
            restraints._spacegroup = duplicate.spacegroup
            duplicate.restraints = restraints
        return duplicate

    def state(self) -> Dict[str, Any]:
        """What :meth:`from_state` needs, as picklable entries for a model state dict.

        Returns
        -------
        dict
            The atom table, the cell as a CPU tensor, the space group as its extended
            Hermann-Mauguin symbol (``gemmi.SpaceGroup`` is not picklable), the
            altloc groups and the settings. Restraints are not saved; they rebuild.
        """
        return {
            "pdb": self.pdb.copy() if self.pdb is not None else None,
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
    ) -> "ModelContext":
        """Rebuild a context from the entries :meth:`state` wrote.

        Consumes them: every key read is popped off ``state``, so what remains is for
        ``load_state_dict``. The atom table is taken as saved -- the hydrogen policy is
        recorded, not re-applied. Checkpoints that predate the policy are mapped:
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
        ModelContext
        """
        from torchref.symmetry import Cell

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
        ctx = cls(
            pdb=state.pop("pdb", None),
            cell=Cell(cell, dtype=dtype, device=device) if cell is not None else None,
            spacegroup=own_spacegroup(state.pop("spacegroup", None), dtype, device),
            initialized=state.pop("initialized", False),
            cif_path=state.pop("cif_path", None),
            altloc_pairs=state.pop("altloc_pairs", []),
            hydrogens=hydrogens,
            hydrogen_mode=mode,
            hydrogens_in_xray=state.pop("hydrogens_in_xray", True),
            verbose=verbose,
        )
        return ctx

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
        n_atoms = 0 if self.pdb is None else len(self.pdb)
        sg = None if self.spacegroup is None else self.spacegroup.name
        return (
            f"ModelContext(spacegroup={sg!r}, n_atoms={n_atoms}, "
            f"hydrogens={self.hydrogens!r}, hydrogen_mode={self.hydrogen_mode!r}, "
            f"initialized={self.initialized})"
        )


__all__ = ["ModelContext", "HYDROGEN_SOURCES", "HYDROGEN_MODES", "check_hydrogen_policy"]
