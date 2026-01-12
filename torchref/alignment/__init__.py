"""
Patterson-based alignment module for TorchRef.

Aligns predicted structures to observed diffraction data using
Patterson map vector matching.

Example
-------
>>> from torchref.model import Model
>>> from torchref.io.datasets.reflection_data import ReflectionData
>>> from torchref.alignment import PattersonAligner
>>>
>>> data = ReflectionData().load_mtz('data.mtz')
>>> model = Model().load_pdb('predicted.pdb')
>>>
>>> aligner = PattersonAligner(data, model)
>>> aligned_model, result = aligner.align(model)
>>> aligned_model.write_pdb('aligned.pdb')
"""

from .align import PattersonAligner, AlignmentResult
from .sampling import VectorSampler


__all__ = [
    # Main API
    'PattersonAligner',
    'AlignmentResult',
    # Lower-level utilities (for advanced users)
    'VectorSampler',
    'params_to_matrix',
    'matrix_to_params',
    'random_rotation_params',
    'random_rotation_matrix',
]
