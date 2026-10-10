"""Shared dispatch gate for the target math functions.

On a CUDA float32 tensor with importable Triton kernels, this package's math functions
route to :mod:`torchref.base.targets.triton`; CPU tensors, non-float32 tensors and a
missing or broken Triton fall back to eager. The criteria are data in
:data:`TARGET_BACKENDS`, matching the density and direct-summation tables.

**Gate-only**: rows carry no ``kernel``, because each call site does its own
``from .triton.<mod> import <fn>`` beside the ``if``. To force eager for an A/B
comparison or around a flaky Triton install::

    from torchref.utils import use_portable
    with use_portable():
        ...
"""

import functools
from typing import Callable, Optional

import torch

from torchref.utils.backends import Backend, BackendTable, triton_available, will_use

_THIS = "torchref.base.targets._dispatch"


def why_unavailable() -> Optional[str]:
    """``None`` if the target Triton kernels can run, else why they cannot.

    Distinct from ``triton_available()``, which only answers "is the package
    importable": the call sites import unguarded, so a Triton skewed against the driver
    or LLVM would raise straight through a refinement step instead of degrading.
    Importing the package is the shared prerequisite, so one probe covers all of them.

    Deliberately does *not* re-check ``torch.cuda.is_available()``: selection is
    two-phase, so the device criterion has already matched, and re-asking would make
    this disagree with ``triton_available()`` on the ordinary Triton-but-no-GPU CI host.
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
            device="cuda",
            dtypes=(torch.float32,),  # dtype-ok: backend capability declaration, not an allocation
            probe=(_THIS, "why_unavailable"),
            expect_available="cuda",
            # Not consulted by this gate-only table: a Triton kernel that throws raises.
            on_failure="raise",
            second_order=False,
        ),
        Backend(
            name="eager",
            kernel=None,
            expect_available="always",
            on_failure="raise",
            second_order=True,
        ),
    ),
)


def use_triton(*tensors: torch.Tensor) -> bool:
    """Whether to route this call to the Triton kernel, per the ``triton`` row of
    :data:`TARGET_BACKENDS`. ``None`` entries in ``tensors`` are ignored, so optional
    inputs can be passed straight through.
    """
    return will_use(TARGET_BACKENDS, "triton", tensors)


def first_order_only(backward: Callable) -> Callable:
    """Make a first-order-only autograd ``backward`` refuse ``create_graph=True``.

    Decorate the ``backward`` staticmethod of a :class:`torch.autograd.Function` whose
    gradient comes from a kernel and so carries no graph. Autograd runs ``backward``
    with grad mode on exactly when the caller asked for ``create_graph=True``; the
    wrapper then raises a ``RuntimeError`` naming the Function, because a second
    derivative taken through it would silently miss this term. First-order backward is
    untouched, and the check reads grad mode only, so it never syncs the device.
    ``torch.autograd.function.once_differentiable`` is not enough: it raises only when
    the second pass reaches this node, which a loss linear in the output, or a second
    ``torch.autograd.grad(inputs=x)``, never does.

    Parameters
    ----------
    backward : Callable
        The ``backward(ctx, *grad_outputs)`` to guard.

    Returns
    -------
    Callable
        ``backward`` wrapped with the check.

    Raises
    ------
    RuntimeError
        From the wrapped ``backward``, when it runs with grad mode enabled.
    """

    @functools.wraps(backward)
    def guarded(ctx, *grad_outputs):
        if torch.is_grad_enabled():
            raise RuntimeError(
                f"{backward.__qualname__}: the second derivative is not implemented, "
                "so create_graph=True cannot differentiate through this kernel. Where "
                "its backend table has a second-order portable row, run under "
                "torchref.utils.use_portable() to take that instead."
            )
        return backward(ctx, *grad_outputs)

    return guarded
