"""
Restraints module for crystallographic refinement.

This module provides classes for building and managing geometry restraints
(bonds, angles, torsions, planes, chirals, VDW contacts) from CIF dictionaries.

Restraint geometry parameters are sourced from the CCP4 Monomer Library,
which derives ideal values from the Cambridge Structural Database.

References
----------
Long, F., et al. (2017). AceDRG: a stereochemical description generator
    for ligands. Acta Cryst. D73, 112-122.
Sherri, L.N., et al. (2018). Updated CCP4 Monomer Library.
    Acta Cryst. D74, 641-655.

Classes
-------

Restraints
    Restraints handler (alias of ``RestraintsNew``); builds and manages all
    geometry restraints from CIF dictionaries.
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
