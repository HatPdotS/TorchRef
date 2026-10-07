"""Model topology as a graph: residues over atoms, connectivity over restraints.

The topology is where a model's atom identity lives -- names, elements, altlocs,
residues, chains -- from the moment its atom table is read
(:meth:`Topology.from_table`, :meth:`Topology.select`). Connectivity is added later,
against the monomer dictionaries.

:class:`Topology` holds two levels. :class:`ResidueGraph` is the sequence -- residues as
template instances, inter-residue links as edges. :class:`AtomGraph` is the expansion --
atoms as nodes, typed :class:`EdgeBlock` sets over them, and a CSR bond adjacency that
answers ``neighbors(i)``.

The topology is **target-free**: it says what is connected, not what the ideal geometry
is. Ideal values and sigmas belong to a restraint layer keyed to the same edges, so one
connectivity can carry monomer-library targets, force-field parameters, or
ADP-similarity sigmas without duplicating the edges.

Build one with :func:`build_topology`.

Hydrogens come in two forms over the same graph. Under ``hydrogens="add"``,
:func:`plan_hydrogens` instantiates the monomer templates to add them as real atoms. A
model without hydrogens (a heavy-only file under the default ``"keep"``, or ``"strip"``)
has :mod:`torchref.topology.riding` reconstruct them from their parents at each
non-bonded evaluation instead, so their sterics still count. Only one applies at a time.
"""

from .atom_graph import AtomGraph
from .build import build_topology, build_topology_with_values
from .edges import ORIGIN_ORDER, EdgeBlock
from .hydrogens import HydrogenPlan, optimise_free_torsions, plan_hydrogens
from .residue_graph import ResidueGraph
from .restraints import Restraints
from .riding import (
    HydrogenTopology,
    build_h_candidate_pairs,
    build_hydrogen_topology,
    place_riding_hydrogens,
)
from .restraint_sets import assemble_entries, max_period
from .templates import resolve_template_keys
from .topology import IDENTITY_COLUMNS, Topology, identity_columns

__all__ = [
    "Topology",
    "identity_columns",
    "IDENTITY_COLUMNS",
    "Restraints",
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
    "HydrogenTopology",
    "build_hydrogen_topology",
    "build_h_candidate_pairs",
    "place_riding_hydrogens",
    "resolve_template_keys",
]
