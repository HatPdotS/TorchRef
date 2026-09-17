"""Collection (multi-dataset) refinement targets.

Generic targets that operate on a paired ``DatasetCollection`` +
``ModelCollection`` — the multi-dataset analogues of the single-dataset X-ray /
geometry / ADP targets. Used by kinetic / time-resolved / multi-dataset
refinement. Kinetic-specific targets (e.g. ``KineticPriorTarget``) stay in
:mod:`torchref.experimental.kinetic.targets`.
"""

from ._util import _scale_fcalc
from .base import (
    CollectionLossInputs,
    CollectionSigmaALossInputs,
    CollectionXrayTarget,
)
from .intensity import CollectionTwoMomentIntensityTarget
from .multimodel import MultiModelADPTarget, MultiModelGeometryTarget
from .xray import (
    CollectionDifferenceIntensityTarget,
    CollectionDifferenceTarget,
    CollectionMLTarget,
)
from ._specs import (  # noqa: E402  (imports the rows above)
    COLLECTION_XRAY_TARGETS,
    CollectionXrayTargetSpec,
    CollectionXrayTargetTable,
)

__all__ = [
    "COLLECTION_XRAY_TARGETS",
    "CollectionXrayTargetSpec",
    "CollectionXrayTargetTable",
    "CollectionXrayTarget",
    "CollectionLossInputs",
    "CollectionSigmaALossInputs",
    "CollectionTwoMomentIntensityTarget",
    "CollectionDifferenceTarget",
    "CollectionDifferenceIntensityTarget",
    "CollectionMLTarget",
    "MultiModelGeometryTarget",
    "MultiModelADPTarget",
]
