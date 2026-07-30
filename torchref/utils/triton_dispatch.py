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
    """Coarse triton-vs-eager gate.

    Asks the ``triton`` row of ``TARGET_BACKENDS`` whether it would run, so the
    device/dtype/availability criteria are stated once, in that table, rather than a second
    time here. The target-math table is the right one to defer to: it is the *generic*
    Triton question, with no kernel-family-specific requirement attached. The density and
    direct-summation sites have their own tables because their availability probes and probe
    sets genuinely differ.

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
        If ``engine`` is not an ``Engine`` member. Deliberately loud: the AUTO case used to
        be an implicit ``else``, so *any* unrecognised value silently selected Triton.

    Notes
    -----
    ``Engine.METAL`` returns False here rather than raising -- it forces the
    Metal density splat (see :func:`should_use_metal`), and every other
    dispatch site correctly runs eager under it. In the table that is the ``eager`` row
    listing METAL among its engines, which is checked for completeness at import.
    """
    # Function-local: ``torchref.utils`` loads before ``torchref.base``.
    from torchref.base.targets._dispatch import TARGET_BACKENDS
    from torchref.utils.backends import admits

    return admits(TARGET_BACKENDS, "triton", tensors, engine)


def should_use_metal(*tensors: torch.Tensor, engine: Optional[Engine] = None) -> bool:
    """Metal-vs-portable gate for the MPS electron-density splat.

    Retained as public API, but no longer a second statement of the criteria: it asks the
    ``mps_metal`` row of ``DENSITY_BACKENDS`` whether it would run. The device, dtype and
    shader-availability conditions live in that table, so this cannot drift from what
    dispatch actually does.

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
        True if the Metal kernels should be used. ``False`` -- not an error -- for an
        engine that does not admit Metal at all, since it is simply not Metal's turn.

    Raises
    ------
    RuntimeError
        If ``engine`` is ``Engine.METAL`` but the inputs are not MPS float32, or the shader
        library is unavailable. The message carries the recorded compile error, since that
        is the one failure a user can act on (torch < 2.9, or an older Metal rejecting the
        MSL).
    ValueError
        If ``engine`` is not an ``Engine`` member.
    """
    # Imported here, not at module scope: ``torchref.utils`` loads very early and must not
    # reach into ``torchref.base`` while it is still initialising.
    from torchref.base.electron_density._backends import DENSITY_BACKENDS
    from torchref.utils.backends import admits

    return admits(DENSITY_BACKENDS, "mps_metal", tensors, engine)
