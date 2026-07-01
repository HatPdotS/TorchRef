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

Classes
-------

Restraints
    Restraints handler (alias of ``RestraintsNew``); builds and manages all
    geometry restraints from CIF dictionaries.

Functions
---------

get_library_manager
    Return the shared :class:`~torchref.restraints.library.MonomerLibraryManager`
    instance used to resolve restraint CIF dictionaries.

Attributes
----------
MONOMER_LIB_PATH
    Lazily-resolved monomer-library root directory (see ``__getattr__``);
    resolves to ``get_library_manager().monomer_dir`` on first access.

Notes
-----
The restraint builder classes and topology helpers
(``build_all_restraints``, ``HydrogenTopology``, ``build_hydrogen_topology``,
the intra-/inter-residue builders) are public-by-use across the package but
are not re-exported at the ``torchref.restraints`` level; import them from
their defining submodules.
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
