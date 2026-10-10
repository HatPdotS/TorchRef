"""Atomic models: coordinates, ADPs, occupancies and their structure factors.

:class:`~torchref.model.model.Model` holds the refinable atomic parameters,
:class:`~torchref.model.context.ModelContext` the crystallographic context, atom table
and provenance. :class:`~torchref.model.model_ft.ModelFT` adds structure factors via its
:class:`~torchref.model.sf_fft.SfFFT` engine; :class:`~torchref.model.sf_ds.SfDS` is a
standalone direct-summation engine over the same atom arrays.
:class:`~torchref.model.mixed_model.MixedModel` mixes ModelFT states by population
(e.g. dark/light); :class:`~torchref.model.model_collection.ModelCollection` keys
mixtures by timepoint (``_SharedMixedModel`` is its non-re-registering variant). The
parametrizations :class:`~torchref.model.parameter_wrappers.MixedTensor` (with its
``Positive`` / ``Cholesky`` / ``Occupancy`` subclasses) and
:class:`~torchref.model.rigid_xyz.RigidXYZTensor` decide which parameters are refinable.
"""

from torchref.model.sf_fft import SfFFT
from torchref.model.sf_ds import SfDS
from torchref.model.context import ModelContext
from torchref.model.mixed_model import MixedModel
from torchref.model.model import Model
from torchref.model.model_ft import ModelFT
from torchref.model.parameter_wrappers import (
    CholeskyMixedTensor,
    MixedTensor,
    OccupancyTensor,
    PositiveMixedTensor,
)
from torchref.model.model_collection import ModelCollection, _SharedMixedModel
from torchref.model.rigid_xyz import RigidXYZTensor

__all__ = [
    "SfFFT",
    "SfDS",
    "MixedModel",
    "Model",
    "ModelContext",
    "ModelFT",
    "MixedTensor",
    "PositiveMixedTensor",
    "CholeskyMixedTensor",
    "OccupancyTensor",
    "ModelCollection",
    "_SharedMixedModel",
    "RigidXYZTensor",
]
