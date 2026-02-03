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
FFT
    Standalone FFT module for electron density and structure factor calculations.
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

    from torchref.model import Model, ModelFT, MixedTensor, FFT

    # Load model from PDB
    model = Model()
    model.load_pdb('structure.pdb')

    # Access coordinates and B-factors
    xyz = model.xyz  # (N, 3) tensor
    b = model.b      # (N,) tensor

    # Use ModelFT for FFT-based structure factors
    model_ft = ModelFT(data, device='cuda')
    F_calc = model_ft.get_F_calc()

    # Use FFT standalone for custom workflows
    fft = FFT(max_res=1.5)
    fft.setup_grid(cell, spacegroup)
    sf = fft.map_to_structure_factors(density_map, hkl)
"""

from torchref.model.fft import FFT
from torchref.model.internal_coordinates import InternalCoordinateTensor
from torchref.model.mixed_model import MixedModel
from torchref.model.model import Model
from torchref.model.model_ft import ModelFT
from torchref.model.parameter_wrappers import (
    MixedTensor,
    OccupancyTensor,
    PassThroughTensor,
    PositiveMixedTensor,
)

__all__ = [
    "FFT",
    "InternalCoordinateTensor",
    "MixedModel",
    "Model",
    "ModelFT",
    "MixedTensor",
    "PositiveMixedTensor",
    "PassThroughTensor",
    "OccupancyTensor",
]
