"""Lazy compilation and availability probe for the Metal (MPS) splat kernels.

The build is a single in-process ``torch.mps.compile_shader`` call -- no ninja, build
directory or file locking, since PyTorch caches the compiled pipeline-state objects
itself. ``mps_kernels_available()`` is what ``torchref.utils.should_use_metal`` consults
and returns False whenever MPS is absent, ``compile_shader`` is missing (torch < 2.9), or
the shader fails to build; the caller then falls back to the portable plain splat and
warns, while ``why_unavailable()`` reports :func:`last_error`.
"""

from __future__ import annotations

import traceback
from typing import Optional, Tuple

import torch

from torchref.base.electron_density.kernels.mps._shaders import MSL_SOURCE

# Memoized compile state (attempted at most once per process).
_lib = None
_lib_failed = False
_lib_error: Optional[Tuple[str, str]] = None  # (message, traceback)


def _mps_shader_supported() -> bool:
    """True iff this torch build can compile+run Metal shaders on this host."""
    return (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "compile_shader")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def _get_lib():
    """The compiled shader library, or None if unavailable.

    Short-circuits both prior outcomes so compilation is attempted exactly once; any failure
    is recorded and turned into None so callers fall back to the plain splat.
    """
    global _lib, _lib_failed, _lib_error
    if _lib is not None:
        return _lib
    if _lib_failed:
        return None
    if not _mps_shader_supported():
        _lib_failed = True
        _lib_error = (
            "torch.mps.compile_shader unavailable or MPS not available",
            "",
        )
        return None
    try:
        _lib = torch.mps.compile_shader(MSL_SOURCE)
    except Exception as e:  # noqa: BLE001 - any build failure -> fall back
        _lib_failed = True
        _lib_error = (f"{type(e).__name__}: {e}", traceback.format_exc())
        return None
    return _lib


def why_unavailable() -> Optional[str]:
    """``None`` if the Metal kernels are usable, else why they are not.

    The single availability probe for this backend, consumed by
    :mod:`torchref.utils.backends`. It returns the *reason* rather than a bool because
    "torch
    has no ``compile_shader``" and "the MSL failed to compile" need different fixes.
    """
    if _get_lib() is not None:
        return None
    reason = _lib_error[0] if _lib_error else "unknown reason"
    return (
        f"the Metal splat kernels are not available ({reason}); see "
        "torchref.base.electron_density.kernels.mps.compile.last_error()"
    )


def mps_kernels_available() -> bool:
    """Whether the Metal splat kernels compiled and are ready to dispatch.

    Derived from :func:`why_unavailable` rather than re-testing, so there is one
    availability check here, not two that can drift.
    """
    return why_unavailable() is None


def warmup() -> bool:
    """Eagerly trigger compilation (e.g. to move the one-time cost off the
    first refinement step). Returns availability."""
    return _get_lib() is not None


def clear_cache() -> None:
    """Forget the compiled library and failure state (recompiled on next use)."""
    global _lib, _lib_failed, _lib_error
    _lib = None
    _lib_failed = False
    _lib_error = None


def last_error() -> Optional[Tuple[str, str]]:
    """The (message, traceback) of the last compile failure, if any."""
    return _lib_error
