"""DEPRECATED: ``Symmetry`` is a bare alias for :class:`SpaceGroup`.

``Symmetry = SpaceGroup`` literally, so ``isinstance`` checks against either name
succeed and existing calls keep working. No ``DeprecationWarning`` is emitted --
nothing tells a caller to migrate. Prefer ``SpaceGroup`` in new code.
"""

import warnings

from torchref.symmetry.spacegroup import SpaceGroup, SpaceGroupLike

# Backward compatibility alias - Symmetry is now SpaceGroup
# Using the class directly so isinstance() checks work.
Symmetry = SpaceGroup

__all__ = ["Symmetry", "SpaceGroupLike"]
