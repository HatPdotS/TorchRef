"""Shared accelerator/eager backend dispatch for TorchRef.

A single capability-based selector used by every dispatch site (direct
summation, geometry/X-ray targets, FFT electron density). For a given
``(device, dtype)`` there is exactly one best path, so the backend is
*derived* rather than configured:

- CUDA + float32 + Triton available  ->  Triton kernel
- MPS + float32 + shader compiles    ->  native Metal kernel (density splat only)
- everything else (CPU, float64, no Triton)  ->  eager fallback

The :class:`Engine` enum exists only to *force* a path for tests and
benchmarks. There is **no environment-variable dispatch** — selection is via
the engine object (process-wide default + per-call override) only.

Examples
--------
::

    from torchref.utils import Engine, use_engine, should_use_triton

    # force eager for an A/B block (e.g. in a test)
    with use_engine(Engine.EAGER):
        loss = target()

    # inside a dispatch site
    if should_use_triton(xyz):           # reads the global engine
        return _triton_path(...)
    return _eager_path(...)
"""

from __future__ import annotations

import contextlib
import enum
from typing import Iterator, Optional

import torch

__all__ = [
    "Engine",
    "get_engine",
    "set_engine",
    "use_engine",
    "triton_available",
    "should_use_triton",
    "should_use_metal",
]


class Engine(enum.Enum):
    """Backend selector shared by all accelerator dispatch sites.

    ``AUTO`` derives the backend from device/dtype/availability. ``TRITON``,
    ``METAL`` and ``EAGER`` force a path; the two accelerator forces are
    strict and never silently degrade -- ``TRITON`` raises if
    CUDA+float32+availability is not met, ``METAL`` if MPS+float32+a compiled
    shader is not met.

    ``METAL`` selects the native Metal *density splat* only; there are no
    Metal direct-summation or target kernels, so those sites treat it as
    eager. Prefer scoping it with :func:`use_engine` around the density call
    over a process-wide :func:`set_engine`, which would also send every target
    math function down the eager path and skew a benchmark.
    """

    AUTO = "auto"
    TRITON = "triton"
    METAL = "metal"
    EAGER = "eager"


# Process-wide default engine. No environment-variable read — change via
# set_engine()/use_engine() or a per-call engine= override.
_ENGINE: Engine = Engine.AUTO


def get_engine() -> Engine:
    """Return the current process-wide engine."""
    return _ENGINE


def set_engine(engine: Engine) -> None:
    """Set the process-wide engine."""
    global _ENGINE
    _ENGINE = engine


@contextlib.contextmanager
def use_engine(engine: Engine) -> Iterator[None]:
    """Temporarily force an engine for the duration of the ``with`` block.

    Restores the previous engine on exit (including on exception).
    """
    prev = get_engine()
    set_engine(engine)
    try:
        yield
    finally:
        set_engine(prev)


_has_triton: Optional[bool] = None


def triton_available() -> bool:
    """Whether ``triton`` is importable (cheap, cached for the process)."""
    global _has_triton
    if _has_triton is None:
        try:
            import triton  # noqa: F401

            _has_triton = True
        except Exception:
            _has_triton = False
    return _has_triton


def should_use_triton(*tensors: torch.Tensor, engine: Optional[Engine] = None) -> bool:
    """Coarse triton-vs-eager gate shared by every dispatch site.

    Reads the process-wide engine unless an explicit ``engine`` is passed
    (direct summation / ``SfDS`` use the per-call override). Probes the given
    tensors: the Triton path requires every non-None tensor to be CUDA
    float32.

    Parameters
    ----------
    *tensors : torch.Tensor
        Tensors whose device/dtype gate the Triton path. ``None`` entries are
        ignored (a target may pass optional tensors).
    engine : Engine, optional
        Per-call override. Defaults to the process-wide engine.

    Returns
    -------
    bool
        True if the Triton kernel should be used.

    Raises
    ------
    RuntimeError
        If ``engine`` is ``Engine.TRITON`` but the inputs are not CUDA
        float32, or Triton is unavailable.
    ValueError
        If ``engine`` is a member this function does not handle. Deliberately
        loud: the AUTO case used to be an implicit ``else``, so *any* new
        member silently selected Triton.

    Notes
    -----
    ``Engine.METAL`` returns False here rather than raising -- it forces the
    Metal density splat (see :func:`should_use_metal`), and every other
    dispatch site correctly runs eager under it.
    """
    eng = engine if engine is not None else get_engine()
    if eng is Engine.EAGER or eng is Engine.METAL:
        return False

    cuda_f32 = all(
        t is None or (t.is_cuda and t.dtype is torch.float32) for t in tensors
    )

    if eng is Engine.TRITON:
        if not cuda_f32:
            raise RuntimeError(
                "engine=Engine.TRITON requires CUDA float32 inputs"
            )
        if not triton_available():
            raise RuntimeError("Triton is not available")
        return True

    if eng is Engine.AUTO:
        return cuda_f32 and triton_available()

    raise ValueError(f"should_use_triton: unhandled engine {eng!r}")


def should_use_metal(*tensors: torch.Tensor, engine: Optional[Engine] = None) -> bool:
    """Metal-vs-portable gate for the MPS electron-density splat.

    The Metal counterpart of :func:`should_use_triton`, and the sole decision
    point for the Metal path: it probes device, dtype **and** shader
    availability together. Folding availability in here rather than leaving it
    as a nested ``if`` at the dispatch site is what makes ``Engine.METAL``
    genuinely strict -- otherwise an uncompiled shader under a forced engine
    would fall past the gate onto the portable splat, silently degrading the
    very thing the caller asked to force.

    Parameters
    ----------
    *tensors : torch.Tensor
        Tensors whose device/dtype gate the Metal path. ``None`` entries are
        ignored.
    engine : Engine, optional
        Per-call override. Defaults to the process-wide engine.

    Returns
    -------
    bool
        True if the Metal kernels should be used.

    Raises
    ------
    RuntimeError
        If ``engine`` is ``Engine.METAL`` but the inputs are not MPS float32,
        or the shader library is unavailable. The latter message carries the
        recorded compile error, since that is the one failure a user can act on
        (torch < 2.9, or an older Metal rejecting the MSL).
    """
    eng = engine if engine is not None else get_engine()
    # EAGER means portable everywhere. TRITON has already raised at the Triton
    # gate on any MPS input, so reaching here under it means a non-MPS host.
    if eng is Engine.EAGER or eng is Engine.TRITON:
        return False
    if eng is not Engine.AUTO and eng is not Engine.METAL:
        raise ValueError(f"should_use_metal: unhandled engine {eng!r}")

    mps_f32 = all(
        t is None or (t.device.type == "mps" and t.dtype is torch.float32)
        for t in tensors
    )
    if not mps_f32:
        if eng is Engine.METAL:
            raise RuntimeError(
                "engine=Engine.METAL requires MPS float32 inputs"
            )
        return False

    # Imported lazily and only once the cheap checks pass: this module is
    # imported very early via ``torchref.utils``, and pulling in the mps
    # package eagerly would load the MSL source on every platform.
    try:
        from torchref.base.electron_density.kernels.mps.compile import (
            last_error,
            mps_kernels_available,
        )
    except Exception:
        # A stripped install or a torch without ``torch.mps`` must not make a
        # predicate raise under AUTO.
        if eng is Engine.METAL:
            raise
        return False

    if mps_kernels_available():
        return True

    if eng is Engine.METAL:
        err = last_error()
        reason = err[0] if err else "unknown reason"
        raise RuntimeError(
            f"engine=Engine.METAL requested but the Metal splat kernels are "
            f"not available ({reason}). See "
            "torchref.base.electron_density.kernels.mps.compile._lib_error "
            "for the full traceback."
        )
    return False
