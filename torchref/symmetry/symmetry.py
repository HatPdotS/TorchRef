"""
DEPRECATED: Use SpaceGroup instead. ``Symmetry`` is now a bare alias for
``SpaceGroup``, kept for backward compatibility.

The Symmetry class has been merged into SpaceGroup, which provides the same
functionality plus additional features. This module exposes ``Symmetry`` as a
direct alias (``Symmetry = SpaceGroup``). It is a plain alias and does NOT emit
a DeprecationWarning; the ``warnings`` import is currently unused.

Usage
-----
Old code using Symmetry continues to work unchanged::

    from torchref.symmetry import Symmetry
    sym = Symmetry('P21')
    sym.apply(coords)

Preferred new style::

    from torchref.symmetry import SpaceGroup
    sg = SpaceGroup('P21')
    sg.apply(coords)
"""

import warnings

from torchref.symmetry.spacegroup import SpaceGroup, SpaceGroupLike

# Backward compatibility alias - Symmetry is now SpaceGroup
# Using the class directly so isinstance() checks work.
Symmetry = SpaceGroup

__all__ = ["Symmetry", "SpaceGroupLike"]
