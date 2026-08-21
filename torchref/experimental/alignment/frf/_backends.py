"""Dispatch policy for the Legendre-recurrence-and-shell-accumulation stage.

One row per kernel, with every criterion for choosing it as a field. See
:mod:`torchref.utils.backends` for what each field means, and
:mod:`torchref.base.electron_density._backends` for the table this follows.

Both kernels take the same arguments::

    (Tr, Ti, rep_cos, rep_sin, Dr, Di, shell, a_coef, b_coef, sect)

and both accumulate into ``Tr``/``Ti`` in place, which is what lets the dispatch
site be a single call.
"""

from __future__ import annotations

import torch

from torchref.utils.backends import Backend, BackendTable

_CPU = "torchref.experimental.alignment.frf.kernels.cpu.legendre_shell"
_PORTABLE = "torchref.experimental.alignment.frf.kernels.portable"

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
            dtypes=(torch.float32,),
            # The kernel reads every array through a raw `float*`, so a mixed-dtype
            # call would reinterpret the buffer rather than convert it. The gate
            # keeps that from reaching the kernel; the kernel checks it too, since
            # a table row is easier to widen by accident than a TORCH_CHECK.
            require_uniform_dtype=True,
            probes=_FLOAT_ARGS,
            probe=(_CPU, "why_unavailable"),
            # A compiler is not guaranteed on every host, and a missing one is a
            # performance problem, not an outage.
            expect_available="never",
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

__all__ = ["LEGENDRE_BACKENDS"]
