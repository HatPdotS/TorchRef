"""Collection (multi-dataset) refinement targets.

Generic targets that operate on a paired ``DatasetCollection`` +
``ModelCollection`` — the multi-dataset analogues of the single-dataset X-ray /
geometry / ADP targets. Used by kinetic / time-resolved / multi-dataset
refinement. Kinetic-specific targets (e.g. ``KineticPriorTarget``) stay in
:mod:`torchref.experimental.kinetic.targets`.
"""

from ._util import _scale_fcalc
from .base import CollectionXrayTarget
from .multimodel import MultiModelADPTarget, MultiModelGeometryTarget
from .xray import (
    CollectionDifferenceTarget,
    CollectionMLTarget,
    CollectionRiceTarget,
)

__all__ = [
    "CollectionXrayTarget",
    "CollectionDifferenceTarget",
    "CollectionRiceTarget",
    "CollectionMLTarget",
    "MultiModelGeometryTarget",
    "MultiModelADPTarget",
]
