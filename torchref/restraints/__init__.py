"""Geometry restraints (bonds, angles, torsions, planes, chirals, VDW contacts).

``Restraints`` (an alias of ``RestraintsNew``) builds and holds them all from CIF
dictionaries resolved through :func:`get_library_manager`; ``MONOMER_LIB_PATH``
resolves lazily to that manager's ``monomer_dir``. Ideal values come from the CCP4
Monomer Library (Long et al. 2017, Acta Cryst. D73, 112-122).

The builder classes and the inter-residue builders are used across the package but
deliberately *not* re-exported here -- import them from their defining submodules.
Connectivity itself lives in :mod:`torchref.topology`, which is also where the
riding-hydrogen map moved to.
"""

from torchref.restraints.library import get_library_manager
from torchref.restraints.restraints import RestraintsNew as Restraints


def __getattr__(name):
    """Lazy access to MONOMER_LIB_PATH for backward compatibility."""
    if name == "MONOMER_LIB_PATH":
        return get_library_manager().monomer_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Restraints",
    "MONOMER_LIB_PATH",
    "get_library_manager",
]
