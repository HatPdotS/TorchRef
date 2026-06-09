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

# Autograd introspection
from .autograd_introspection import collect_loss_leaves

# Caching
from .caching import CachedForwardMixin, ParameterFingerprint

# Debugging
from .debug_utils import DebugMixin, print_module_summary

# Device movement
from .device_mixin import DeviceMixin, DeviceMovementMixin
from .device_resolution import resolve_device

# Gradient utilities
from .gradnorm import gradnorm

# Loss finiteness validator
from .loss_validation import (
    NonFiniteLossError,
    reset_diagnostic_budget,
    validate_loss,
)

# Statistics
from .stats import (
    StatEntry,
    filter_stats,
    flatten_stats,
    format_stats_table,
    stat,
)

# Serialization
from .serialization import convert_to_serializable

# Triton/eager backend dispatch
from .triton_dispatch import (
    Engine,
    get_engine,
    set_engine,
    should_use_triton,
    triton_available,
    use_engine,
)

# Core utilities
from .utils import (
    ModuleReference,
    TensorDict,
    TensorMasks,
    create_selection_mask,
    parse_phenix_selection,
    sanitize_pdb_dataframe,
)

__all__ = [
    # Caching
    "ParameterFingerprint",
    "CachedForwardMixin",
    # Device movement
    "DeviceMixin",
    "DeviceMovementMixin",
    "resolve_device",
    # Core utilities
    "TensorMasks",
    "TensorDict",
    "ModuleReference",
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
    # Serialization
    "convert_to_serializable",
    # Gradients
    "gradnorm",
    # Loss finiteness validator
    "validate_loss",
    "NonFiniteLossError",
    "reset_diagnostic_budget",
    # Autograd introspection
    "collect_loss_leaves",
    # Triton/eager backend dispatch
    "Engine",
    "get_engine",
    "set_engine",
    "use_engine",
    "triton_available",
    "should_use_triton",
]
