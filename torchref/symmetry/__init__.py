"""Crystallographic symmetry: symmetry groups, space groups and unit cells.

:class:`Symmetry` holds a group as rotation matrices and fractional translations and
owns every verb derivable from the operations alone -- expansion of positions and
Miller indices, translation phases, the reflection predicates, symmetry-compatible grid
sizes, and map symmetrization. Nothing in it is crystallographic, so a group built from
a raw operation list serves non-crystallographic symmetry too.

:class:`SpaceGroup` specialises it with the crystallographic identity (Hermann-Mauguin
naming, number, point group, crystal system) and the CCP4 asymmetric-unit verbs
(``expand_hkl``, ``reduce_hkl``, ``complete_hkl``, ``canonicalize_hkl``). It accepts a
name, a number 1-230, a ``gemmi.SpaceGroup``, another instance, or None for P1.

:class:`Cell` is separate: it wraps the six cell parameters, not a symmetry group.

Map and reciprocal-grid operators are reached through :class:`Symmetry`
(:meth:`~Symmetry.symmetrize_map`, :meth:`~Symmetry.reciprocal_extractor`), which owns
their caching -- the operator classes themselves are private.
"""

from .cell import Cell
from .spacegroup import SpaceGroup, SpaceGroupLike
from .symmetry import Symmetry, find_fft_friendly_size, is_fft_friendly

__all__ = [
    # Unit cell
    "Cell",
    # Symmetry groups
    "Symmetry",
    "SpaceGroup",
    "SpaceGroupLike",
    # Grid sizing helpers (group-independent)
    "is_fft_friendly",
    "find_fft_friendly_size",
]
