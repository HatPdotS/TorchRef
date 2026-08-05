"""
Utility functions and classes for TorchRef.

Tensor containers (``TensorMasks``, ``TensorDict``), device/dtype movement
(``DeviceMixin``, ``resolve_device``), debugging mixins, statistics formatting, gradient
norms, PDB/selection parsing, loss-finiteness validation, autograd introspection, JSON
serialization and backend dispatch.

``__all__`` is the package's public surface. Submodule-only helpers -- e.g.
``torchref.utils.timing.register_timing`` -- are deliberately not re-exported and must be
imported from their own module. Note that importing ``torchref.utils.stats`` patches the
stdlib ``json`` encoders process-wide; see that module.
"""

# Autograd introspection
from .autograd_introspection import collect_loss_leaves

# Caching
from .caching import CachedForwardMixin, ParameterFingerprint

# Debugging
from .debug_utils import DebugMixin, print_module_summary

# Device movement
from .device_mixin import DeviceMixin, DeviceMovementMixin
from .device_resolution import require_cell_dtype, resolve_device

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

# Backend dispatch
from .backends import (
    force_portable,
    set_force_portable,
    triton_available,
    use_portable,
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
    "require_cell_dtype",
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
    # Backend dispatch
    "force_portable",
    "set_force_portable",
    "use_portable",
    "triton_available",
]
