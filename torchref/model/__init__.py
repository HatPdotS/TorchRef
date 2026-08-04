"""Atomic models: coordinates, ADPs, occupancies and their structure factors.

:class:`Model` holds the refinable atomic parameters; :class:`ModelFT` adds
structure-factor calculation on top, through :class:`SfFFT` (FFT) or
:class:`SfDS` (direct summation). :class:`MixedModel` combines ModelFT states
by population fraction (e.g. dark/light), and :class:`ModelCollection` keys
mixtures by timepoint (``_SharedMixedModel`` is its non-re-registering variant).
The wrappers from :mod:`torchref.model.parameter_wrappers` --
:class:`MixedTensor` and its ``Positive`` / ``Cholesky`` / ``Occupancy``
subclasses plus :class:`RigidXYZTensor` -- are the parametrizations that decide
which parameters are refinable. ``FFT`` is a deprecated alias for
:class:`SfFFT`.
"""

from torchref.model.sf_fft import SfFFT, FFT
from torchref.model.sf_ds import SfDS
from torchref.model.mixed_model import MixedModel
from torchref.model.model import Model
from torchref.model.model_ft import ModelFT
from torchref.model.parameter_wrappers import (
    CholeskyMixedTensor,
    MixedTensor,
    OccupancyTensor,
    PassThroughTensor,
    PositiveMixedTensor,
)
from torchref.model.model_collection import ModelCollection, _SharedMixedModel
from torchref.model.rigid_xyz import RigidXYZTensor

__all__ = [
    "FFT",
    "SfFFT",
    "SfDS",
    "MixedModel",
    "Model",
    "ModelFT",
    "MixedTensor",
    "PositiveMixedTensor",
    "CholeskyMixedTensor",
    "PassThroughTensor",
    "OccupancyTensor",
    "ModelCollection",
    "_SharedMixedModel",
    "RigidXYZTensor",
]
