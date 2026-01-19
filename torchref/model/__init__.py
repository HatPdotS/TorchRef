"""
Atomic structure model module for TorchRef.

This module provides PyTorch nn.Module-based representations of
crystallographic atomic models, including coordinates, B-factors,
occupancies, and anisotropic displacement parameters.

Classes
-------
Model
    Base atomic model storing xyz coordinates, B-factors, occupancies.
ModelFT
    Fourier Transform model for FFT-based structure factor calculation.
MixedTensor
    Hybrid tensor allowing partial freezing of parameters.
PositiveMixedTensor
    MixedTensor with positivity constraint.
PassThroughTensor
    Direct parameter access wrapper.
OccupancyTensor
    Tensor constrained to [0, 1] range for occupancies.

Example
-------
::

    from torchref.model import Model, ModelFT, MixedTensor

    # Load model from PDB
    model = Model()
    model.load_pdb('structure.pdb')

    # Access coordinates and B-factors
    xyz = model.xyz  # (N, 3) tensor
    b = model.b      # (N,) tensor

    # Use ModelFT for FFT-based structure factors
    model_ft = ModelFT(data, device='cuda')
    F_calc = model_ft.get_F_calc()
"""

from torchref.model.model import Model
from torchref.model.model_ft import ModelFT
from torchref.model.parameter_wrappers import (
    MixedTensor,
    OccupancyTensor,
    PassThroughTensor,
    PositiveMixedTensor,
)

__all__ = [
    "ModelFT",
    "Model",
    "MixedTensor",
    "PositiveMixedTensor",
    "PassThroughTensor",
    "OccupancyTensor",
]
