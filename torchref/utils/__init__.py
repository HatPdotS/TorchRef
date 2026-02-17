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
::

    from torchref.utils import TensorMasks, DebugMixin, gradnorm

    # Create tensor masks for parameter selection
    masks = TensorMasks()
    masks['backbone'] = backbone_mask

    # Use debugging mixin in your class
    class MyRefinement(DebugMixin):
        pass

    # Compute gradient norm
    grad_norm = gradnorm(loss, model.parameters())
"""

# Caching
from .caching import CachedForwardMixin, ParameterFingerprint

# Debugging
from .debug_utils import DebugMixin, print_module_summary

# Gradient utilities
from .gradnorm import gradnorm

# Hyperparameters
from .hyperparameters import HyperparameterMixin

# Statistics
from .stats import (
    StatEntry,
    filter_stats,
    flatten_stats,
    format_stats_table,
    stat,
)

# Core utilities
from .utils import (
    ModuleReference,
    TensorDict,
    TensorMasks,
    create_selection_mask,
    parse_phenix_selection,
    sanitize_pdb_dataframe,
    save_map,
)

__all__ = [
    # Caching
    "ParameterFingerprint",
    "CachedForwardMixin",
    # Core utilities
    "TensorMasks",
    "TensorDict",
    "ModuleReference",
    "save_map",
    "sanitize_pdb_dataframe",
    "parse_phenix_selection",
    "create_selection_mask",
    # Debugging
    "DebugMixin",
    "print_module_summary",
    # Statistics
    "StatEntry",
    "stat",
    "filter_stats",
    "flatten_stats",
    "format_stats_table",
    # Hyperparameters
    "HyperparameterMixin",
    # Gradients
    "gradnorm",
]
