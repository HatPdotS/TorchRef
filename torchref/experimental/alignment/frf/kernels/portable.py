"""Portable Legendre-recurrence-and-shell-accumulation, in plain torch.

The reference the fused kernel is checked against, and the fallback whenever that
kernel cannot be built. One step of the vertical recurrence per ``l``, then the
row is multiplied by the azimuthal sums and scattered into its cluster's shell.

Both stages are memory-bound: at L=101 over 4.4e5 clusters the recurrence moves
about 108 GB and the scatter about 71 GB, and both measure ~50 GB/s, which is
roughly what four threads get from a server memory controller. The arithmetic is
a small fraction of that -- which is the whole reason for a fused kernel, where
the row never leaves cache.
"""

from __future__ import annotations

import torch

from ...sh import LEGENDRE_SEED


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
    """Accumulate ``sum_c barP[c, l, m] * D[c, m]`` into ``Tr``/``Ti``, in place.

    Parameters
    ----------
    Tr, Ti : torch.Tensor
        ``(n_even, n_shells, L)`` real accumulators, added into.
    rep_cos, rep_sin : torch.Tensor
        ``(n_clusters,)`` cos and sin of the polar angle, per cluster.
    Dr, Di : torch.Tensor
        ``(n_clusters, L)`` real and imaginary azimuthal sums.
    shell : torch.Tensor
        ``(n_clusters,)`` int64 shell index of each cluster, into ``Tr``'s middle
        axis.
    a_coef, b_coef : torch.Tensor
        ``(L, L)`` recurrence coefficients, zero for ``m >= l``.
    sect : torch.Tensor
        ``(L,)`` sectoral factors.
    """
    L = Tr.shape[-1]
    cos_e = rep_cos.unsqueeze(-1)
    prev2 = torch.zeros_like(Dr)
    prev1 = torch.zeros_like(Dr)
    prev1[:, 0] = LEGENDRE_SEED                              # bar_P_0^0
    for l in range(1, L):
        # `a_coef` and `b_coef` are zero for m >= l, so this runs at full width.
        # Narrowing it to the l+1 columns that can be non-zero was measured and
        # is slower: the saving is real (the summed width at L=101 falls from
        # 10100 to 6481) but a strided scatter target costs more than it, and the
        # recurrence did not speed up at all -- it is not arithmetic-bound. The
        # fused kernel gets the ragged widths for free, as loop bounds.
        cur = a_coef[l] * cos_e * prev1 - b_coef[l] * prev2
        # The sectoral term must land BEFORE the products below are formed: it is
        # the m = l entry of this very row. Forming `cur * Dr` first silently
        # drops that entry for every even l.
        cur[:, l] = sect[l] * rep_sin * prev1[:, l - 1]
        if l >= 2 and (l % 2 == 0):
            pos = (l - 2) // 2
            Tr[pos].index_add_(0, shell, cur * Dr)
            Ti[pos].index_add_(0, shell, cur * Di)
        prev2, prev1 = prev1, cur


__all__ = ["legendre_shell_accumulate"]
