"""
DEPRECATED: Use SpaceGroup instead. Symmetry is now an alias for backward compatibility.

The Symmetry class has been merged into SpaceGroup, which provides the same
functionality plus additional features. This module provides the Symmetry alias
for backward compatibility with existing code.

Usage
-----
Old code using Symmetry continues to work::

    from torchref.symmetry import Symmetry
    sym = Symmetry('P21')
    sym.apply(coords)

Preferred new style::

    from torchref.symmetry import SpaceGroup
    sg = SpaceGroup('P21')
    sg.apply(coords)
"""

from torchref.symmetry.spacegroup import SpaceGroup, SpaceGroupLike

# Backward compatibility alias - Symmetry is now SpaceGroup
Symmetry = SpaceGroup

__all__ = ["Symmetry", "SpaceGroupLike"]
