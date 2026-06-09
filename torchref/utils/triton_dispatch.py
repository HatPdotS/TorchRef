"""Shared Triton/eager backend dispatch for TorchRef.

A single capability-based selector used by every dispatch site (direct
summation, geometry/X-ray targets, FFT electron density). For a given
``(device, dtype)`` there is exactly one best path, so the backend is
*derived* rather than configured:

- CUDA + float32 + Triton available  ->  Triton kernel
- everything else (CPU, float64, MPS, no Triton)  ->  eager fallback

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
]


class Engine(enum.Enum):
    """Backend selector shared by all Triton dispatch sites.

    ``AUTO`` derives the backend from device/dtype/availability. ``TRITON``
    and ``EAGER`` force a path; ``TRITON`` raises if CUDA+float32+availability
    is not met (force = strict, never silently degrade).
    """

    AUTO = "auto"
    TRITON = "triton"
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
    """
    eng = engine if engine is not None else get_engine()
    if eng is Engine.EAGER:
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

    # AUTO
    return cuda_f32 and triton_available()
