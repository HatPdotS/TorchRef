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
from .similarity import CoordinateSimilarityTarget
from .xray import (
    GaussianXrayTarget,
    LeastSquaresXrayTarget,
    MaximumLikelihoodXrayTarget,
    RiceXrayTarget,
    XrayTarget,
    create_xray_target,
)

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
    # Similarity restraint
    "CoordinateSimilarityTarget",
]
# Force-field, real-space, sampled-ML phase, and occupancy-diagnostic
# targets are experimental and live in :mod:`torchref.experimental.targets`.
