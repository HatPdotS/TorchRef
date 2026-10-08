"""Crystallographic symmetry: symmetry groups, space groups and unit cells.

:class:`~.symmetry.Symmetry` holds a group as rotation matrices and fractional
translations and owns every verb derivable from the operations alone -- expansion of
positions and Miller indices, translation phases, the reflection predicates, map
symmetrization and symmetry-compatible grid sizes. Nothing in it is crystallographic,
so a group built from a raw operation list serves non-crystallographic symmetry too.

:class:`~.spacegroup.SpaceGroup` specialises it with the crystallographic identity
(Hermann-Mauguin naming, number, point group, crystal system) and the CCP4 ASU verbs
(``equivalent_hkl``, ``expand_hkl``, ``canonicalize_hkl``). It accepts a name, a number
1-230, a ``gemmi.SpaceGroup``, another instance, or None for P1.

:class:`~.cell.Cell` is separate; it wraps six cell parameters, not a symmetry group.

Map and reciprocal-grid operators are private, reached through ``Symmetry``, which owns
their caching: :meth:`~.symmetry.Symmetry.symmetrize_map` and
:meth:`~.symmetry.Symmetry.reciprocal_extractor`.
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
