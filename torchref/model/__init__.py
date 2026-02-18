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
SfFFT
    Structure Factor calculator using FFT (Fast Fourier Transform).
SfDS
    Structure Factor calculator using Direct Summation.
FFT
    Backward compatibility alias for SfFFT.
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

    from torchref.model import Model, ModelFT, MixedTensor, SfFFT, SfDS

    # Load model from PDB
    model = Model()
    model.load_pdb('structure.pdb')

    # Access coordinates and B-factors
    xyz = model.xyz  # (N, 3) tensor
    b = model.b      # (N,) tensor

    # Use ModelFT for FFT-based structure factors
    model_ft = ModelFT(data, device='cuda')
    F_calc = model_ft.get_F_calc()

    # Use SfFFT standalone for custom workflows
    sf_fft = SfFFT(max_res=1.5)
    sf_fft.setup_grid(cell, spacegroup)
    sf = sf_fft.map_to_structure_factors(density_map, hkl)

    # Use SfDS for direct summation
    sf_ds = SfDS(cell, spacegroup)
    sf, _ = sf_ds.compute_structure_factors(hkl, xyz, adp, occ, A, B)
"""

from torchref.model.sf_fft import SfFFT, FFT
from torchref.model.sf_ds import SfDS
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
from torchref.model.segmented_internal_coordinates import (
    SegmentedInternalCoordinateTensor,
)
from torchref.model.closed_segmented_internal_coordinates import (
    ClosedSegmentedInternalCoordinateTensor,
)

__all__ = [
    "FFT",
    "SfFFT",
    "SfDS",
    "InternalCoordinateTensor",
    "MixedModel",
    "Model",
    "ModelFT",
    "MixedTensor",
    "PositiveMixedTensor",
    "PassThroughTensor",
    "OccupancyTensor",
    "SegmentedInternalCoordinateTensor",
    "ClosedSegmentedInternalCoordinateTensor",
]
