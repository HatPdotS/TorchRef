"""Where restraint *data* comes from: the monomer library and the CIF readers.

The dictionaries of ideal geometry, resolved through :func:`get_library_manager`, the
``chem_mod`` records that patch them when a link forms, the Numba matchers that map a
template onto the atoms present, the Ramachandran surfaces, and the spatial search
behind the non-bonded pair list. ``MONOMER_LIB_PATH`` resolves lazily to the manager's
``monomer_dir``. Ideal values come from the CCP4 Monomer Library (Long et al. 2017,
Acta Cryst. D73, 112-122).

What is *built* from that data lives in :mod:`torchref.topology`: the connectivity, the
values layered over its edges, and :class:`~torchref.topology.restraints.Restraints`,
which orchestrates the two. Nothing here is re-exported -- import from the defining
submodule.
"""

from torchref.restraints.library import get_library_manager


def __getattr__(name):
    """Lazy access to MONOMER_LIB_PATH for backward compatibility."""
    if name == "MONOMER_LIB_PATH":
        return get_library_manager().monomer_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MONOMER_LIB_PATH",
    "get_library_manager",
]
