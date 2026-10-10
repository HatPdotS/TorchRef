"""Loader and calling convention for the prebuilt ``torchref-kernels`` package.

The CPU kernels ship as a separate, prebuilt Rust extension (``kernels/`` in the source
tree), so nothing is compiled when TorchRef runs. That extension links neither libtorch
nor a particular CPython; its entry points take ``(data_ptr, numel)`` pairs. This module
is the single place that turns tensors into those pairs, so every kernel wrapper gets the
same guarantees: CPU, contiguous, the expected dtype, and storage that outlives the call.

Availability follows the backend-table probe contract (:mod:`torchref.utils.backends`):
:func:`why_unavailable` returns ``None`` or a reason, and a missing, mismatched or debug
build is a reason, never an import error.
"""

from __future__ import annotations

import os
import traceback
from typing import Optional, Tuple

import torch

#: Calling-convention version this TorchRef is written against; must equal the
#: extension's ``ABI_VERSION``.
ABI_VERSION = 1

# Below this, a Rust build is an unoptimised debug build and runs 10-100x slower.
_MIN_OPT_LEVEL = 2

_native = None
_failed = False
_error: Optional[Tuple[str, str]] = None


def _load():
    import torchref_kernels

    info = torchref_kernels.build_info()
    if info["abi_version"] != ABI_VERSION:
        raise ImportError(
            f"torchref-kernels {info['version']} has ABI version {info['abi_version']}, "
            f"but this TorchRef needs {ABI_VERSION}; install a matching torchref-kernels"
        )
    opt = info["opt_level"]
    if not (opt.isdigit() and int(opt) >= _MIN_OPT_LEVEL):
        raise ImportError(
            f"torchref-kernels was built without optimisation (opt-level {opt}); "
            "rebuild with `pip install ./kernels` or `maturin develop --release`"
        )
    return torchref_kernels.native()


def native():
    """The kernel extension module, or ``None`` if it cannot be used.

    Loaded once per process; a failure is remembered and reported by
    :func:`why_unavailable` and :func:`last_error`.
    """
    global _native, _failed, _error
    if _native is not None or _failed:
        return _native
    try:
        _native = _load()
    except Exception as exc:  # noqa: BLE001 - any load failure means "unavailable"
        _failed = True
        _error = (f"{type(exc).__name__}: {exc}", traceback.format_exc())
    return _native


def why_unavailable() -> Optional[str]:
    """``None`` if the kernel extension is usable, else why it is not."""
    if native() is not None:
        return None
    reason = _error[0] if _error else "unknown reason"
    return (
        f"the torchref-kernels extension is not available ({reason}); see "
        "torchref.utils.native.last_error()"
    )


def last_error() -> Optional[Tuple[str, str]]:
    """The ``(message, traceback)`` of the load failure, if any."""
    native()
    return _error


def build_info() -> Optional[dict]:
    """The loaded extension's build metadata, or ``None`` if it is unavailable."""
    if native() is None:
        return None
    import torchref_kernels

    return torchref_kernels.build_info()


def buf(t: torch.Tensor, dtype: torch.dtype, name: str) -> Tuple[int, int]:
    """``(data_ptr, numel)`` of ``t`` for a kernel entry point.

    Parameters
    ----------
    t : torch.Tensor
        Must be a contiguous CPU tensor of ``dtype``; it is not copied or converted, so
        the caller keeps it alive for the duration of the call.
    dtype : torch.dtype
        The kernel's scalar type.
    name : str
        Argument name for error messages.

    Raises
    ------
    ValueError
        If ``t`` is not on the CPU, not contiguous, or not of ``dtype``. The kernels read
        raw memory, so these are checked here rather than trusted.
    """
    if t.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor, got device {t.device}")
    if t.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {t.dtype}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return t.data_ptr(), t.numel()


def num_threads() -> int:
    """Worker count for a kernel call: torch's intra-op thread count."""
    return torch.get_num_threads()


def source_hash(kernels_dir: str) -> str:
    """FNV-1a digest of the Rust sources under ``kernels_dir/src``.

    Matches ``build_info()["source_hash"]`` of an extension built from the same sources
    (see ``kernels/build.rs``), so a development checkout can detect a stale build.
    """
    root = os.path.abspath(kernels_dir)
    files = []
    for dirpath, _, names in os.walk(os.path.join(root, "src")):
        for n in names:
            if n.endswith(".rs"):
                full = os.path.join(dirpath, n)
                files.append((os.path.relpath(full, root).replace(os.sep, "/"), full))
    h = 0xCBF29CE484222325
    mask = (1 << 64) - 1

    def feed(data: bytes) -> None:
        nonlocal h
        for b in data:
            h ^= b
            h = (h * 0x100000001B3) & mask

    for rel, full in sorted(files):
        feed(rel.encode())
        feed(b"\0")
        with open(full, "rb") as f:
            feed(f.read())
        feed(b"\0")
    return f"{h:016x}"
