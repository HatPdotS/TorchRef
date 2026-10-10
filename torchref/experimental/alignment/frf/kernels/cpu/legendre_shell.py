"""Fused Legendre recurrence and shell accumulation, as one native kernel.

The portable version runs the vertical recurrence as one torch operation per
``l`` and then scatters the row, so every row makes a round trip to memory. At
L=101 over 4.4e5 clusters that is ~108 GB for the recurrence and ~71 GB for the
scatter, both measured at ~50 GB/s -- the stages are bandwidth-bound, and the
arithmetic underneath is a small fraction of the time.

float32 throughout, matching the rest of this codebase's kernels. The radial
Bessel recurrence is a separate stage and runs at the same working precision,
kept in range by its power-of-two rescaling.

Fusing them removes the round trip: one cluster's three rows are 1.2 kB of stack,
so ``cur`` is produced, multiplied and accumulated without ever reaching memory.
Two further things fall out of writing it as a loop nest:

* **Ragged widths are free.** ``bar_P[l, m]`` is zero for m > l, so step ``l``
  needs only columns 0..l -- ``for (m = 0; m <= l; ++m)`` and nothing more. In
  torch the same saving needs narrowed views, and that was measured *slower*,
  because a strided scatter target costs more than the zeros it skips.
* **No atomics.** The clusters arrive sorted by shell, so a thread that owns a
  range of shells owns every write into those shells' rows. Parallelising over
  clusters instead would race on the shared accumulator.

The accumulator rows for one shell are ``n_even * L`` scalars -- 40 kB at L=101 --
so they stay in cache across that shell's clusters, which is the point of
grouping by shell in the first place.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from torchref.utils import native


def why_unavailable() -> Optional[str]:
    """``None`` if the fused kernel is usable, else why it is not.

    The single availability probe for this backend, read by
    :mod:`torchref.utils.backends`.
    """
    reason = native.why_unavailable()
    if reason is None:
        return None
    return f"the fused CPU Legendre/shell kernel is not available: {reason}"


def available() -> bool:
    """Whether the fused kernel is ready to dispatch."""
    return why_unavailable() is None


def last_error() -> Optional[Tuple[str, str]]:
    """``(message, traceback)`` from the kernel load failure, if any."""
    return native.last_error()


def shell_offsets(shell: torch.Tensor, n_shells: int) -> torch.Tensor:
    """Start index of each shell in a shell-sorted cluster array, plus the end.

    ``(n_shells + 1,)`` int64. The kernel needs the ranges rather than the
    per-cluster labels so that a thread can own a set of shells outright and
    write their accumulator rows without atomics.
    """
    counts = torch.bincount(shell, minlength=n_shells)
    # dtype-ok: the kernel reads int64 offsets
    offsets = torch.zeros(n_shells + 1, dtype=torch.int64, device=shell.device)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return offsets


def legendre_shell_accumulate(
    Tr: torch.Tensor,
    Ti: torch.Tensor,
    rep_cos: torch.Tensor,
    rep_sin: torch.Tensor,
    Dr: torch.Tensor,
    Di: torch.Tensor,
    shell: torch.Tensor,
    a_coef: torch.Tensor,
    b_coef: torch.Tensor,
    sect: torch.Tensor,
) -> None:
    """Fused recurrence and accumulation, in place on ``Tr``/``Ti``.

    Same signature and same effect as
    :func:`torchref.experimental.alignment.frf.kernels.portable.legendre_shell_accumulate`.
    ``shell`` must be sorted non-decreasing -- the kernel partitions work by
    shell to avoid atomics, and unsorted input would silently drop
    contributions rather than merely run slowly.
    """
    from ....sh import LEGENDRE_SEED

    module = native.native()
    if module is None:
        raise RuntimeError(why_unavailable())
    # float32 only, by policy: this codebase has no float64 kernels. Checked rather than
    # dispatched on, so a float64 accumulator is a loud error, never a conversion.
    if Tr.dtype != torch.float32:  # dtype-ok: kernel dtype contract, not an allocation
        raise RuntimeError(
            f"legendre_shell_accumulate is float32 only, got {Tr.dtype}"
        )
    f32 = torch.float32  # dtype-ok: kernel dtype contract, not an allocation
    i64 = torch.int64  # dtype-ok: the kernel reads int64 shell labels and offsets
    shell = shell.contiguous()
    if shell.dtype != i64:
        raise RuntimeError(f"shell must be int64, got {shell.dtype}")
    offsets = shell_offsets(shell, Tr.shape[1])
    ins = [rep_cos.contiguous(), rep_sin.contiguous(), Dr.contiguous(),
           Di.contiguous()]
    coefs = [a_coef.contiguous(), b_coef.contiguous(), sect.contiguous()]
    names = ["rep_cos", "rep_sin", "Dr", "Di"]
    n_even, n_shells, L = (int(s) for s in Tr.shape)
    try:
        module.legendre_shell_accumulate_f32(
            native.buf(Tr, f32, "Tr"), native.buf(Ti, f32, "Ti"),
            *(native.buf(t, f32, n) for t, n in zip(ins, names)),
            native.buf(shell, i64, "shell"), native.buf(offsets, i64, "offsets"),
            *(native.buf(t, f32, n) for t, n in zip(coefs, ["a_coef", "b_coef", "sect"])),
            n_even, n_shells, L, float(LEGENDRE_SEED), native.num_threads(),
        )
    except ValueError as exc:
        raise RuntimeError(f"legendre_shell_accumulate: {exc}") from exc


__all__ = [
    "available",
    "last_error",
    "legendre_shell_accumulate",
    "shell_offsets",
    "why_unavailable",
]
