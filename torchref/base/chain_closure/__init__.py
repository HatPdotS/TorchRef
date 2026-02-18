"""
Chain closure submodule for differentiable analytical junction closure.

Provides backbone identification, junction planning, NeRF-based forward
kinematics, Newton-based closure solving, and IFT-based gradient computation.
"""

from .backbone_utils import (
    identify_backbone_atoms,
    get_chain_residues,
    compute_backbone_torsions,
    estimate_secondary_structure,
    plan_junction_placement,
    get_junction_backbone_indices,
    AA_NAMES,
)

from .closure import (
    backbone_fk_junction,
    closure_residual,
    JunctionClosure,
    JunctionSolver,
)

__all__ = [
    "identify_backbone_atoms",
    "get_chain_residues",
    "compute_backbone_torsions",
    "estimate_secondary_structure",
    "plan_junction_placement",
    "get_junction_backbone_indices",
    "AA_NAMES",
    "backbone_fk_junction",
    "closure_residual",
    "JunctionClosure",
    "JunctionSolver",
]
