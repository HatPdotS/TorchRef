"""The information half of a :class:`~torchref.model.model.Model`.

:class:`ModelContext` holds what a model *is loaded from* and *sits in* -- the unit
cell, the space group, the atom table, the link records and the provenance -- as
opposed to what is being refined, which stays on the model as parameter wrappers and
per-atom buffers. The geometry restraints belong here too: they are fixed by the atom
set and the dictionaries, and are evaluated against coordinates the caller passes in.

Splitting it out means the crystallographic context can be passed to code that needs
only that (structure-factor engines, scalers, most targets) without handing over the
refinable state, and it keeps the model's own surface to parameters and behaviour.

Mutable by design; prefer :meth:`ModelContext.copy` over editing in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

from torchref.utils.device_mixin import DeviceMixin

if TYPE_CHECKING:
    import pandas
    import torch

    from torchref.symmetry import Cell, SpaceGroup
    from torchref.topology.restraints import Restraints


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
        Index groups of alternative conformations, rebuilt by
        ``Model.register_alternative_conformations``.
    input_file : str or None
        Path the structure was loaded from.
    cif_path : str or None
        Restraint dictionary path, if one was set.
    verbose : int, default 1
        Verbosity level.
    strip_H : bool, default True
        Whether hydrogens were stripped on load.
    hydrogens_in_xray : bool, default True
        Whether hydrogens enter the structure-factor calculation. Restraints and the
        non-bonded term see them either way; the bulk-solvent mask never does.
    add_hydrogens : bool, default False
        Generate hydrogens on load for residues that arrive without them. Ignored when
        ``strip_H`` is set, which removes them again.
    hydrogen_mode : str, default "free"
        How hydrogen rows are parametrised: ``"riding"`` (positions derived from the
        parent heavy atoms each forward, not refined), ``"free"`` (ordinary refinable
        atoms) or ``"none"`` (the table holds no hydrogens).
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
    cif_path: Optional[str] = None
    verbose: int = 1
    strip_H: bool = True
    hydrogens_in_xray: bool = True
    add_hydrogens: bool = False
    hydrogen_mode: str = "free"
    initialized: bool = False
    restraints: Optional["Restraints"] = None

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
        self, xyz: "torch.Tensor", *, nonbonded: bool = True, verbose=None
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

    def copy(self) -> "ModelContext":
        """An independent copy.

        The atom table is deep-copied, the cell and space group are cloned and built
        restraints are copied, so nothing is shared with the original. Cloning the space group matters now that
        it is a mutable dataclass: sharing the reference would let an edit through one
        model's context reach every model that was copied from it.

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
            input_file=self.input_file,
            cif_path=self.cif_path,
            verbose=self.verbose,
            strip_H=self.strip_H,
            hydrogens_in_xray=self.hydrogens_in_xray,
            add_hydrogens=self.add_hydrogens,
            hydrogen_mode=self.hydrogen_mode,
            initialized=self.initialized,
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
            f"initialized={self.initialized})"
        )


__all__ = ["ModelContext"]
