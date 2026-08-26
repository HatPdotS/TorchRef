"""Model topology as a graph: residues over atoms, connectivity over restraints.

:class:`Topology` holds two levels. :class:`ResidueGraph` is the sequence -- residues as
template instances, inter-residue links as edges. :class:`AtomGraph` is the expansion --
atoms as nodes, typed :class:`EdgeBlock` sets over them, and a CSR bond adjacency that
answers ``neighbors(i)``.

The topology is **target-free**: it says what is connected, not what the ideal geometry
is. Ideal values and sigmas belong to a restraint layer keyed to the same edges, so one
connectivity can carry monomer-library targets, force-field parameters, or
ADP-similarity sigmas without duplicating the edges.

Build one with :func:`build_topology`.
"""

from .atom_graph import AtomGraph
from .build import build_topology
from .edges import ORIGIN_ORDER, EdgeBlock
from .residue_graph import ResidueGraph
from .templates import resolve_template_keys
from .topology import Topology

__all__ = [
    "Topology",
    "ResidueGraph",
    "AtomGraph",
    "EdgeBlock",
    "ORIGIN_ORDER",
    "build_topology",
    "resolve_template_keys",
]
