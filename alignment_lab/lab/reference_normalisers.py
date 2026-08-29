"""E-value normalisers as they were before the convention seam, frozen.

These are not production code and are not imported by it. They are kept so a
future convention can be compared against what the rotation function actually
shipped, rather than against a description of it -- the comparison the seam was
originally validated with, and the one any replacement will want again.

Frozen means frozen: if a production convention changes, these do not follow.
That is the whole point of an oracle.
"""

from __future__ import annotations

import torch

from torchref.experimental.alignment.sh import (
    assign_shells, equal_count_shell_edges,
)


def wilson_normalise(
    F: torch.Tensor,
    s_mag: torch.Tensor,
    n_shells: int = 20,
):
    """Per-shell Wilson normalisation of amplitudes.

    Source: Phaser's ``Feff[r] / SIGMAN.sqrt_epsnSN[r]`` (``DataMR.cc:925``)
    minus French-Wilson + explicit ε (``F`` is assumed anisotropy-corrected
    by the caller).

        E_h = F_h / sqrt(<F²>_p)    where p = shell containing h.

    Returns ``(E_h, sqrt_mean_F2_per_h)``.
    """
    edges, _ = equal_count_shell_edges(s_mag, n_shells)
    shell_idx = assign_shells(s_mag, edges)
    valid = shell_idx >= 0
    F_dtype = F.dtype
    F2 = F * F
    count = torch.zeros(n_shells, dtype=torch.int64, device=F.device)
    sumF2 = torch.zeros(n_shells, dtype=F_dtype, device=F.device)
    F2_v = F2[valid]
    idx_v = shell_idx[valid]
    count.index_add_(0, idx_v, torch.ones_like(idx_v))
    sumF2.index_add_(0, idx_v, F2_v)
    mean_F2 = sumF2 / count.clamp(min=1).to(F_dtype)
    mean_F2 = mean_F2.clamp(min=1e-12)
    sqrt_mean = mean_F2.sqrt()
    per_h = torch.ones_like(F)
    per_h[valid] = sqrt_mean[idx_v]
    E = F / per_h
    return E, per_h
