"""
Refinement target functions for crystallographic structure refinement.

This module provides target (loss) functions for X-ray, geometry, and ADP restraints.
"""

from .combined_targets import (
    CombinedTargets,
    TotalADPTarget,
    TotalGeometryTarget,
)
from .targets import (
    ADPEntropyTarget,
    ADPLocalityTarget,
    ADPSimilarityTarget,
    ADPTarget,
    AngleTarget,
    BondTarget,
    ChiralTarget,
    GaussianXrayTarget,
    GeometryTarget,
    LeastSquaresXrayTarget,
    MaximumLikelihoodXrayTarget,
    NonBondedTarget,
    PlanarityTarget,
    RigidBondTarget,
    TorsionTarget,
    XrayTarget,
    Target,
    ModelTarget,
    DataTarget,
)

from .difference_xray_target import DifferenceXrayTarget
from .phase_informed_difference_target import PhaseInformedDifferenceTarget
from .taylor_corrected_difference_target import TaylorCorrectedDifferenceTarget
from .occupancy_floor_diagnostic import (
    OccupancyFloorDiagnostic,
    NegativeDensityPenalty,
    DisplacementRegularizer,
    DifferenceAmplitudeRegularizer,
)

# Force field target (optional dependency)
try:
    from .forcefield_target import ForceFieldTarget
except ImportError:
    ForceFieldTarget = None  # torchmd-net not installed

__all__ = [
    "Target",
    "XrayTarget",
    "GaussianXrayTarget",
    "MaximumLikelihoodXrayTarget",
    "LeastSquaresXrayTarget",
    "DifferenceXrayTarget",
    "PhaseInformedDifferenceTarget",
    "TaylorCorrectedDifferenceTarget",
    "GeometryTarget",
    "BondTarget",
    "AngleTarget",
    "TorsionTarget",
    "PlanarityTarget",
    "ChiralTarget",
    "NonBondedTarget",
    "ADPTarget",
    "ADPSimilarityTarget",
    "RigidBondTarget",
    "ADPEntropyTarget",
    "ADPLocalityTarget",
    "CombinedTargets",
    "TotalGeometryTarget",
    "TotalADPTarget",
    "ForceFieldTarget",
    "OccupancyFloorDiagnostic",
    "NegativeDensityPenalty",
    "DisplacementRegularizer",
    "DifferenceAmplitudeRegularizer",
]
