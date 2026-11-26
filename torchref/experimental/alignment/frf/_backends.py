"""Dispatch policy for the FRF expansion kernels.

Two tables, one row per kernel, with every criterion for choosing it as a field. See
:mod:`torchref.utils.backends` for what each field means, and
:mod:`torchref.base.electron_density._backends` for the table this follows.

:data:`LEGENDRE_BACKENDS` serves the single-system Legendre-recurrence-and-shell
accumulation stage. Both of its kernels take the same arguments::

    (Tr, Ti, rep_cos, rep_sin, Dr, Di, shell, a_coef, b_coef, sect)

and both accumulate into ``Tr``/``Ti`` in place, which is what lets the dispatch
site be a single call.

:data:`EXPAND_BACKENDS` serves the whole batched expansion,
:func:`~torchref.experimental.alignment.frf.batched.batched_bessel_sh_expand`.
"""

from __future__ import annotations

import torch

from torchref.utils.backends import Backend, BackendTable

_CPU = "torchref.experimental.alignment.frf.kernels.cpu.legendre_shell"
_PORTABLE = "torchref.experimental.alignment.frf.kernels.portable"
_THIS_BACKENDS = "torchref.experimental.alignment.frf._backends"

#: Argument positions carrying the device/dtype contract: the two accumulators and
#: the four per-cluster float arrays. ``shell`` is int64 and the three coefficient
#: tables are built by the caller at the working dtype, so probing them would only
#: restate what the caller already chose.
_FLOAT_ARGS = (0, 1, 2, 3, 4, 5)

LEGENDRE_BACKENDS = BackendTable(
    name="FRF Legendre/shell accumulation",
    backends=(
        Backend(
            name="cpu_fused",
            kernel=(_CPU, "legendre_shell_accumulate", "legendre_shell_accumulate"),
            device="cpu",
            dtypes=(torch.float32,),  # dtype-ok: backend capability declaration
            # The kernel reads every array through a raw `float*`, so a mixed-dtype
            # call would reinterpret the buffer rather than convert it. The gate
            # keeps that from reaching the kernel; the kernel checks it too, since
            # a table row is easier to widen by accident than a TORCH_CHECK.
            require_uniform_dtype=True,
            probes=_FLOAT_ARGS,
            probe=(_CPU, "why_unavailable"),
            # Same package as the CPU splat, so the same expectation.
            expect_available="native",
            # NOT "degrade": the kernel accumulates into Tr/Ti as it goes, so a
            # mid-run failure leaves them partly written and the portable path
            # would add its own contributions on top. There is nothing to fall
            # back to once it has started.
            on_failure="raise",
            second_order=False,
        ),
        Backend(
            name="portable",
            kernel=(_PORTABLE, "legendre_shell_accumulate",
                    "legendre_shell_accumulate"),
            second_order=False,
        ),
    ),
)


_BATCHED = "torchref.experimental.alignment.frf.batched"


def why_expand_unavailable():
    """``None`` if the fused Triton expansion kernels can run, else why they cannot."""
    from torchref.utils.backends import triton_available

    if not triton_available():
        return "triton is not importable"
    try:
        from .kernels import triton_expand  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - an import failure is a reason
        return (
            "the Triton expansion kernels could not be imported "
            f"({type(exc).__name__}: {exc})"
        )
    return None


#: Dispatch for
#: :meth:`~torchref.experimental.alignment.frf.batched.ExpansionPlan.expand`. Both
#: rows take ``(plan, I_list, zsymm)``. Only the first intensity vector is probed: it
#: carries the working precision, while the plan's s-vectors may legitimately be
#: wider (they feed the shell keys).
EXPAND_BACKENDS = BackendTable(
    name="batched FRF Bessel-SH expansion",
    backends=(
        Backend(
            name="triton",
            kernel=(_BATCHED, "_expand_triton", "_expand_triton"),
            device="cuda",
            dtypes=(torch.float32,),  # dtype-ok: backend capability declaration
            probes=(1,),
            probe=(_THIS_BACKENDS, "why_expand_unavailable"),
            expect_available="cuda",
            # The kernels write freshly allocated buffers only, so a failure leaves
            # nothing half-accumulated for the portable path to add to.
            on_failure="degrade",
            second_order=False,
        ),
        Backend(
            name="portable",
            kernel=(_BATCHED, "_expand_portable", "_expand_portable"),
            expect_available="always",
            on_failure="raise",
            second_order=False,
        ),
    ),
)

__all__ = ["EXPAND_BACKENDS", "LEGENDRE_BACKENDS"]
