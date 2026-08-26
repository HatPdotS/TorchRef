"""Model topology as a graph: residues over atoms, connectivity over restraints.

:class:`Topology` holds two levels. :class:`ResidueGraph` is the sequence -- residues as
template instances, inter-residue links as edges. :class:`AtomGraph` is the expansion --
atoms as nodes, typed :class:`EdgeBlock` sets over them, and a CSR bond adjacency that
answers ``neighbors(i)``.

The topology is **target-free**: it says what is connected, not what the ideal geometry
is. Ideal values and sigmas belong to a restraint layer keyed to the same edges, so one
connectivity can carry monomer-library targets, force-field parameters, or
ADP-similarity sigmas without duplicating the edges.

Build one with :func:`build_topology`. :func:`plan_hydrogens` uses it to instantiate
monomer templates, which is how hydrogens are generated.
"""

from .atom_graph import AtomGraph
from .build import build_topology, build_topology_with_values
from .edges import ORIGIN_ORDER, EdgeBlock
from .hydrogens import (
    HydrogenPlan,
    augment_atom_table,
    optimise_free_torsions,
    plan_hydrogens,
)
from .residue_graph import ResidueGraph
from .restraint_sets import assemble_entries, max_period
from .templates import resolve_template_keys
from .topology import Topology

__all__ = [
    "Topology",
    "ResidueGraph",
    "AtomGraph",
    "EdgeBlock",
    "ORIGIN_ORDER",
    "build_topology",
    "build_topology_with_values",
    "assemble_entries",
    "max_period",
    "HydrogenPlan",
    "plan_hydrogens",
    "optimise_free_torsions",
    "augment_atom_table",
    "resolve_template_keys",
]
