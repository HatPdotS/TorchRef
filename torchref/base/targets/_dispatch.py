"""Shared dispatch gate for the target math functions.

When called on a CUDA float32 tensor and the Triton kernels are importable, the math
functions in this package route to their implementations in
:mod:`torchref.base.targets.triton`. CPU tensors, non-float32 tensors, and environments
without a usable Triton fall back to the plain eager implementation.

The criteria live in :data:`TARGET_BACKENDS` rather than in a predicate body, for the same
reason as the density and direct-summation tables: so ``Engine.METAL`` meaning "run eager
here" is a declared fact rather than an early ``return False``, and so a new ``Engine``
member cannot be added without something claiming it.

This is a **gate-only** table -- the twelve call sites each do their own
``from .triton.<mod> import <fn>`` three lines from the ``if``, which is more legible than a
registry lookup for a single-kernel choice. The table's job is the decision, not the
dispatch, so its rows carry no ``kernel``.

To force the eager path for an A/B comparison or to sidestep a flaky Triton install::

    from torchref.utils import use_engine, Engine
    with use_engine(Engine.EAGER):
        ...
"""

from typing import Optional

import torch

from torchref.utils.backends import Backend, BackendTable, admits
from torchref.utils.triton_dispatch import Engine, triton_available

_THIS = "torchref.base.targets._dispatch"


def why_unavailable() -> Optional[str]:
    """``None`` if the target Triton kernels can run, else why they cannot.

    Closes a real gap. ``triton_available()`` answers "is the ``triton`` package
    importable", which is not the same question as "do *these* kernels import" -- and
    nothing asked the second one. Each of the twelve call sites does an unguarded
    function-local import, so a Triton present but skewed against the installed driver or
    LLVM raised straight through a refinement step, where the density and direct-summation
    paths would have degraded. Importing the package is the shared prerequisite for all
    twelve, so one probe covers them.

    Deliberately does *not* re-check ``torch.cuda.is_available()``. Selection is two-phase,
    so the device criterion has already matched by the time a probe runs -- and the presence
    of a CUDA tensor is stronger evidence than the query. Asking again would also make this
    probe disagree with ``triton_available()`` on a host that has Triton installed but no
    GPU, which is an ordinary CI configuration.
    """
    if not triton_available():
        return "triton is not importable"
    try:
        import torchref.base.targets.triton  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - an import failure is a reason, not a crash
        return (
            "the target Triton kernels could not be imported "
            f"({type(exc).__name__}: {exc})"
        )
    return None


TARGET_BACKENDS = BackendTable(
    name="target math",
    backends=(
        Backend(
            name="triton",
            kernel=None,  # gate-only; see the module docstring
            engines=frozenset({Engine.AUTO, Engine.TRITON}),
            device="cuda",
            dtypes=(torch.float32,),
            probe=(_THIS, "why_unavailable"),
            expect_available="cuda",
            # Availability is handled by the probe above, so this governs only a kernel
            # that imported and then threw -- a bug in pure math on already-validated
            # tensors. Degrading would buy a silent ~50x slowdown with subtly different
            # numbers, which is worse than the exception.
            on_failure="raise",
            second_order=False,
        ),
        Backend(
            name="eager",
            kernel=None,
            # METAL is here because there are no Metal target kernels: at this site it has
            # to mean "run eager". TRITON is absent, which is what makes it strict.
            engines=frozenset({Engine.AUTO, Engine.EAGER, Engine.METAL}),
            expect_available="always",
            on_failure="raise",
            second_order=True,
        ),
    ),
)


def use_triton(*tensors: torch.Tensor) -> bool:
    """Decide whether to route a call to the Triton kernel.

    Asks the ``triton`` row of :data:`TARGET_BACKENDS` whether it would run, using the
    process-wide engine. ``None`` entries among ``tensors`` are ignored, so a caller may
    pass optional inputs straight through.
    """
    return admits(TARGET_BACKENDS, "triton", tensors)
