"""Peak finding on the adaptive SO(3) sample list.

Phaser source: ``SiteListAng::findpeaks`` (referenced from FastRot.cc;
implementation in ``SiteListAng.cc`` ``findpeaks`` and the related NMS
routines). Phaser's strategy is essentially:

  1. Compute mean + std of all samples → z-score per sample.
  2. Sort by descending value.
  3. Greedy non-max suppression on SO(3) by *angular distance* between
     rotations (not by α, β, γ box distance — that would double-count
     near the poles).

We implement the same flow in PyTorch, vectorised where possible. The
SO(3) angular-distance NMS is identical to the v13 ``_so3_greedy_nms``
in ``ball_search.py`` — that part of v13 was correct; the bug was in
the FFT/grid, not the NMS.
"""
from __future__ import annotations

import math
from typing import List

import torch

from .types import AdaptiveRotationFunction, RotationPeak

__all__ = [
    "find_rotation_peaks",
]


def _euler_to_matrix_edmonds_zyz(
    alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor,
) -> torch.Tensor:
    """R = R_z(α) R_y(β) R_z(γ) — Edmonds ZYZ convention.

    Returns shape (*alpha.shape, 3, 3) real.
    """
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)
    # R = Rz(a) * Ry(b) * Rz(g)
    R = torch.stack(
        [
            torch.stack([ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb], dim=-1),
            torch.stack([sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb], dim=-1),
            torch.stack([-sb * cg,                sb * sg,                cb       ], dim=-1),
        ],
        dim=-2,
    )
    return R


def _so3_angular_distance_deg(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Angular distance between two rotation matrices, in degrees.

    R1: (..., 3, 3), R2: (..., 3, 3). Returns (...,) real.
    """
    trace = torch.einsum("...ij,...ij->...", R1, R2)
    cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0, max=1.0)
    return torch.arccos(cos_theta) * (180.0 / math.pi)


def _so3_greedy_nms(
    alphas: torch.Tensor,
    betas: torch.Tensor,
    gammas: torch.Tensor,
    values: torch.Tensor,
    nms_radius_deg: float,
    keep_at_most: int,
) -> torch.Tensor:
    """Return indices (into the input order) of kept peaks after SO(3) NMS.

    Greedy: walk the values in descending order; keep a candidate if its
    angular distance from every already-kept rotation is > nms_radius_deg.
    """
    n = values.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.int64, device=values.device)
    order = torch.argsort(values, descending=True)
    R_all = _euler_to_matrix_edmonds_zyz(alphas, betas, gammas)  # (n, 3, 3)
    R_all = R_all.to(torch.float64)

    kept_idx: List[int] = []
    kept_R: List[torch.Tensor] = []
    for i_t in order.tolist():
        Ri = R_all[i_t]
        if kept_R:
            stack = torch.stack(kept_R, dim=0)  # (k, 3, 3)
            dists = _so3_angular_distance_deg(Ri.unsqueeze(0), stack)
            if dists.min().item() <= nms_radius_deg:
                continue
        kept_R.append(Ri)
        kept_idx.append(i_t)
        if len(kept_idx) >= keep_at_most:
            break
    return torch.tensor(kept_idx, dtype=torch.int64, device=values.device)


def find_rotation_peaks(
    arf: AdaptiveRotationFunction,
    n_peaks: int = 500,
    sigma_threshold: float = -5.0,
    nms_radius_deg: float = 6.0,
) -> List[RotationPeak]:
    """Greedy SO(3) NMS over the adaptive sample list.

    Returns peaks sorted by descending value, capped at ``n_peaks`` and
    filtered by ``sigma >= sigma_threshold``.
    """
    values = arf.values
    if values.numel() == 0:
        return []

    mean = values.mean()
    std = values.std().clamp(min=1e-30)
    sigma = (values - mean) / std

    # Pre-filter by sigma threshold to keep the NMS loop tractable.
    keep_mask = sigma >= sigma_threshold
    if not keep_mask.any():
        return []

    idx_filtered = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
    # Optionally cap candidate set so the O(n_kept · n_cand) NMS stays small.
    candidate_cap = max(n_peaks * 20, 2000)
    if idx_filtered.numel() > candidate_cap:
        top_vals = values[idx_filtered]
        top_keep = torch.topk(top_vals, candidate_cap).indices
        idx_filtered = idx_filtered[top_keep]

    a = arf.alphas[idx_filtered]
    b = arf.betas[idx_filtered]
    g = arf.gammas[idx_filtered]
    v = values[idx_filtered]

    kept = _so3_greedy_nms(
        a, b, g, v,
        nms_radius_deg=nms_radius_deg,
        keep_at_most=n_peaks,
    )

    peaks: List[RotationPeak] = []
    for k in kept.tolist():
        peaks.append(
            RotationPeak(
                alpha=float(a[k].item()),
                beta=float(b[k].item()),
                gamma=float(g[k].item()),
                value=float(v[k].item()),
                sigma=float(((v[k] - mean) / std).item()),
            )
        )
    return peaks
