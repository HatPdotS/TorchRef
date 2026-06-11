"""
Refinement target functions for crystallographic structure refinement.

This module provides target (loss) functions for X-ray, geometry, and ADP restraints.
"""

from .adp import (
    ADPEntropyTarget,
    ADPLocalityTarget,
    ADPSimilarityTarget,
    ADPTarget,
    RigidBondTarget,
)
from .base import (
    DataTarget,
    ModelTarget,
    Target,
    adp_similarity_nll,
    gaussian_nll,
    von_mises_nll,
)
from .collection import (
    CollectionDifferenceTarget,
    CollectionMLTarget,
    CollectionRiceTarget,
    MultiModelADPTarget,
    MultiModelGeometryTarget,
)
from .combined import (
    CombinedTargets,
    TotalADPTarget,
    TotalGeometryTarget,
)
from .difference import (
    DifferenceXrayTarget,
    PhaseInformedDifferenceTarget,
    RiceDifferenceTarget,
    TaylorCorrectedDifferenceTarget,
)
from .geometry import (
    AngleTarget,
    BondTarget,
    ChiralTarget,
    GeometryTarget,
    NonBondedHTarget,
    NonBondedTarget,
    PlanarityTarget,
    RamachandranTarget,
    TorsionTarget,
)
from .occupancy_floor_diagnostic import (
    DifferenceAmplitudeRegularizer,
    DisplacementRegularizer,
    NegativeDensityPenalty,
    OccupancyFloorDiagnostic,
)
from .realspace import (
    RealSpaceCorrelationTarget,
    RealSpaceDifferenceTarget,
    RealSpaceExtrapolatedTarget,
    RealSpaceTarget,
)
from .sampled_ml_phase_target import (
    SampledMLDifferenceTarget,
    SampledMLPhaseTarget,
    create_sampled_ml_difference_target,
    create_sampled_ml_target,
)
from .similarity import CoordinateSimilarityTarget
from .xray import (
    GaussianXrayTarget,
    LeastSquaresXrayTarget,
    MaximumLikelihoodXrayTarget,
    RiceXrayTarget,
    XrayTarget,
    create_xray_target,
)

# Force field target (optional dependency: torchmd-net)
# Note: torchmd-net is imported lazily inside ForceFieldTarget.__init__
# and raises a clear ImportError there if missing.
try:
    from .forcefield_target import ForceFieldTarget
except ImportError:
    ForceFieldTarget = None

# AMBER target (optional dependency: openmm)
# Note: openmm is imported lazily inside AmberTarget methods
# and raises a clear ImportError there if missing.
try:
    from .amber_target import AMBER14_STANDARD, AmberTarget
except ImportError:
    AmberTarget = None
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
    "RiceXrayTarget",
    "MaximumLikelihoodXrayTarget",
    "LeastSquaresXrayTarget",
    "create_xray_target",
    # Collection (multi-dataset) targets
    "CollectionDifferenceTarget",
    "CollectionRiceTarget",
    "CollectionMLTarget",
    "MultiModelGeometryTarget",
    "MultiModelADPTarget",
    # Difference targets
    "DifferenceXrayTarget",
    "PhaseInformedDifferenceTarget",
    "RiceDifferenceTarget",
    "TaylorCorrectedDifferenceTarget",
    # Geometry targets
    "GeometryTarget",
    "BondTarget",
    "AngleTarget",
    "TorsionTarget",
    "PlanarityTarget",
    "ChiralTarget",
    "NonBondedTarget",
    "NonBondedHTarget",
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
    # Similarity restraint
    "CoordinateSimilarityTarget",
]
