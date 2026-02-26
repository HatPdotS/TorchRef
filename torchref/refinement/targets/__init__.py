"""
Refinement target functions for crystallographic structure refinement.

This module provides target (loss) functions for X-ray, geometry, and ADP restraints.
"""

from .base import (
    Target,
    ModelTarget,
    DataTarget,
    gaussian_nll,
    von_mises_nll,
    adp_similarity_nll,
)
from .combined import (
    CombinedTargets,
    TotalADPTarget,
    TotalGeometryTarget,
)
from .xray import (
    XrayTarget,
    GaussianXrayTarget,
    LeastSquaresXrayTarget,
    MaximumLikelihoodXrayTarget,
    create_xray_target,
)
from .geometry import (
    GeometryTarget,
    BondTarget,
    AngleTarget,
    TorsionTarget,
    PlanarityTarget,
    ChiralTarget,
    NonBondedTarget,
    RamachandranTarget,
)
from .adp import (
    ADPTarget,
    ADPSimilarityTarget,
    RigidBondTarget,
    ADPEntropyTarget,
    ADPLocalityTarget,
)
from .difference import (
    DifferenceXrayTarget,
    PhaseInformedDifferenceTarget,
    TaylorCorrectedDifferenceTarget,
)
from .realspace import (
    RealSpaceTarget,
    RealSpaceCorrelationTarget,
    RealSpaceDifferenceTarget,
    RealSpaceExtrapolatedTarget,
)
from .occupancy_floor_diagnostic import (
    OccupancyFloorDiagnostic,
    NegativeDensityPenalty,
    DisplacementRegularizer,
    DifferenceAmplitudeRegularizer,
)
from .sampled_ml_phase_target import (
    SampledMLPhaseTarget,
    SampledMLDifferenceTarget,
    create_sampled_ml_target,
    create_sampled_ml_difference_target,
)

# Force field target (optional dependency)
try:
    from .forcefield_target import ForceFieldTarget
except ImportError:
    ForceFieldTarget = None  # torchmd-net not installed

# AMBER target (optional dependency: openmm, pdbfixer)
try:
    from .amber_target import AmberTarget, AMBER14_STANDARD
except ImportError:
    AmberTarget = None  # openmm / pdbfixer not installed
    AMBER14_STANDARD = None

__all__ = [
    # Base classes
    "Target",
    "ModelTarget",
    "DataTarget",
    # Utility functions
    "gaussian_nll",
    "von_mises_nll",
    "adp_similarity_nll",
    # X-ray targets
    "XrayTarget",
    "GaussianXrayTarget",
    "MaximumLikelihoodXrayTarget",
    "LeastSquaresXrayTarget",
    "create_xray_target",
    # Difference targets
    "DifferenceXrayTarget",
    "PhaseInformedDifferenceTarget",
    "TaylorCorrectedDifferenceTarget",
    # Geometry targets
    "GeometryTarget",
    "BondTarget",
    "AngleTarget",
    "TorsionTarget",
    "PlanarityTarget",
    "ChiralTarget",
    "NonBondedTarget",
    "RamachandranTarget",
    # ADP targets
    "ADPTarget",
    "ADPSimilarityTarget",
    "RigidBondTarget",
    "ADPEntropyTarget",
    "ADPLocalityTarget",
    # Combined targets
    "CombinedTargets",
    "TotalGeometryTarget",
    "TotalADPTarget",
    # Force field
    "ForceFieldTarget",
    # AMBER force field
    "AmberTarget",
    "AMBER14_STANDARD",
    # Occupancy diagnostics
    "OccupancyFloorDiagnostic",
    "NegativeDensityPenalty",
    "DisplacementRegularizer",
    "DifferenceAmplitudeRegularizer",
    # Sampled ML phase targets
    "SampledMLPhaseTarget",
    "SampledMLDifferenceTarget",
    "create_sampled_ml_target",
    "create_sampled_ml_difference_target",
    # Real-space targets
    "RealSpaceTarget",
    "RealSpaceCorrelationTarget",
    "RealSpaceDifferenceTarget",
    "RealSpaceExtrapolatedTarget",
]
