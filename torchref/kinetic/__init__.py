"""
Kinetic refinement for time-resolved crystallography.

This submodule provides tools for refining multiple structural models
against multiple time-resolved datasets with kinetics-constrained
occupancy fractions.

Key Classes
-----------
ModelCollection
    Named dictionary of MixedModel instances at different timepoints,
    sharing the same base structural models.
KineticRefinement
    Orchestrator combining DatasetCollection + ModelCollection with
    difference and ML targets, geometry/ADP restraints, and optional
    kinetic prior regularization.
KineticModel
    PyTorch ODE solver for kinetic schemes (matrix exponential).
occupancies_kinetics
    Kinetics-constrained occupancy model wrapping KineticModel.
"""

from torchref.kinetic.kinetics import KineticModel
from torchref.kinetic.occupancies import (
    occupancy_unrestrained,
    occupancies_kinetics,
    occupancies_kinetics_multiexperiment,
)
from torchref.model.model_collection import ModelCollection
from torchref.kinetic.refinement import KineticRefinement
from torchref.kinetic.targets import (
    CollectionDifferenceTarget,
    CollectionMLTarget,
    MultiModelGeometryTarget,
    MultiModelADPTarget,
    KineticPriorTarget,
)

__all__ = [
    "KineticModel",
    "occupancy_unrestrained",
    "occupancies_kinetics",
    "occupancies_kinetics_multiexperiment",
    "ModelCollection",
    "KineticRefinement",
    "CollectionDifferenceTarget",
    "CollectionMLTarget",
    "MultiModelGeometryTarget",
    "MultiModelADPTarget",
    "KineticPriorTarget",
]
