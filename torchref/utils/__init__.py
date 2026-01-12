"""
Utility functions and classes for TorchRef.

This module provides:
- TensorMasks and TensorDict for managing tensor collections
- Debugging utilities and mixins
- Statistics formatting and tracking
- Hyperparameter management
- Gradient norm computation
- PDB/selection parsing utilities

Example
-------
>>> from torchref.utils import TensorMasks, DebugMixin, gradnorm
>>>
>>> # Create tensor masks for parameter selection
>>> masks = TensorMasks()
>>> masks['backbone'] = backbone_mask
>>>
>>> # Use debugging mixin in your class
>>> class MyRefinement(DebugMixin):
...     pass
>>>
>>> # Compute gradient norm
>>> grad_norm = gradnorm(loss, model.parameters())
"""

# Core utilities
from .utils import (
    TensorMasks,
    TensorDict,
    ModuleReference,
    save_map,
    sanitize_pdb_dataframe,
    parse_phenix_selection,
    create_selection_mask,
)

# Debugging
from .debug_utils import DebugMixin, print_module_summary

# Statistics
from .stats import (
    StatEntry,
    stat,
    filter_stats,
    flatten_stats,
    format_stats_table,
)

# Hyperparameters
from .hyperparameters import HyperparameterMixin

# Gradient utilities
from .gradnorm import gradnorm

__all__ = [
    # Core utilities
    'TensorMasks',
    'TensorDict',
    'ModuleReference',
    'save_map',
    'sanitize_pdb_dataframe',
    'parse_phenix_selection',
    'create_selection_mask',
    # Debugging
    'DebugMixin',
    'print_module_summary',
    # Statistics
    'StatEntry',
    'stat',
    'filter_stats',
    'flatten_stats',
    'format_stats_table',
    # Hyperparameters
    'HyperparameterMixin',
    # Gradients
    'gradnorm',
]
