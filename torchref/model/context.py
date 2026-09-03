"""The information half of a :class:`~torchref.model.model.Model`.

:class:`ModelContext` holds what a model *is loaded from* and *sits in* -- the unit
cell, the space group, the atom table, the link records and the provenance -- as
opposed to what is being refined, which stays on the model as parameter wrappers and
per-atom buffers.

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

    from torchref.symmetry import Cell, SpaceGroup


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
    exclude_H_from_sf : bool, default False
        Whether hydrogens are excluded from structure-factor calculation.
    add_hydrogens : bool, default True
        Generate hydrogens on load for residues that arrive without them. Ignored when
        ``strip_H`` is set, which removes them again.
    initialized : bool, default False
        Whether a structure has been loaded. ``if model:`` tests this.

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
    exclude_H_from_sf: bool = False
    add_hydrogens: bool = True
    initialized: bool = False

    def copy(self) -> "ModelContext":
        """An independent copy.

        The atom table is deep-copied and the cell and space group are cloned, so
        nothing is shared with the original. Cloning the space group matters now that
        it is a mutable dataclass: sharing the reference would let an edit through one
        model's context reach every model that was copied from it.

        Returns
        -------
        ModelContext
            New context sharing no mutable state with this one.
        """
        return ModelContext(
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
            exclude_H_from_sf=self.exclude_H_from_sf,
            add_hydrogens=self.add_hydrogens,
            initialized=self.initialized,
        )

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
