"""Peak finding on the adaptive SO(3) sample list.

Phaser source: ``SiteListAng::findpeaks`` (referenced from FastRot.cc;
implementation in ``SiteListAng.cc`` ``findpeaks`` and the related NMS
routines). Phaser's strategy is essentially:

  1. Compute mean + std of all samples → z-score per sample.
  2. Sort by descending value.
  3. Greedy non-max suppression on SO(3) by *angular distance* between
     rotations (not by α, β, γ box distance — that would double-count
     near the poles).

We implement the same flow in PyTorch, vectorised where possible.
"""
from __future__ import annotations

import math
from typing import List

import torch

from ....base.alignment.rotation import rotation_matrix_euler_zyz
from .types import AdaptiveRotationFunction, RotationPeak

__all__ = [
    "find_rotation_peaks",
]


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
    # The greedy walk is inherently sequential and latency-bound; on GPU a
    # per-iteration `.item()` sync would dominate. Move the (tiny) candidate
    # rotations to CPU once and run the loop there with no device syncs, a
    # preallocated kept-buffer (no repeated torch.stack), and a cosine threshold
    # (no per-iteration arccos). Result is identical to the original distance test.
    order = torch.argsort(values, descending=True).cpu().tolist()
    # `rotation_matrix_euler_zyz` is the shared implementation, and it is now the
    # only one -- the alignment package's own batch copy was deleted after being
    # measured bit-identical to it over 180k elements in both float32 and
    # float64. (The three-matrix product it used reduces to the same two-term
    # sums, because the rotation factors carry exact zeros and ones.) Rounding
    # matters here beyond tidiness: the NMS threshold below flips for pairs
    # sitting exactly on it.
    R_all = (
        rotation_matrix_euler_zyz(torch.stack([alphas, betas, gammas], dim=-1))
        .to(torch.float64).cpu()
    )  # (n, 3, 3)
    # angle > nms_radius  ⇔  cos(angle) < cos(nms_radius); cos(angle) from trace.
    cos_thresh = math.cos(math.radians(nms_radius_deg))
    kept_idx: List[int] = []
    kept_R = torch.empty((keep_at_most, 3, 3), dtype=torch.float64)
    count = 0
    for i_t in order:
        Ri = R_all[i_t]
        if count > 0:
            trace = torch.einsum("kij,ij->k", kept_R[:count], Ri)
            cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0, max=1.0)
            # Some kept rotation within nms_radius (cos_theta > cos_thresh) → skip.
            if bool((cos_theta > cos_thresh).any()):
                continue
        kept_R[count] = Ri
        kept_idx.append(i_t)
        count += 1
        if count >= keep_at_most:
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

    # Gather kept peaks and move to CPU once (avoids a per-peak device sync).
    a_k = a[kept].cpu().tolist()
    b_k = b[kept].cpu().tolist()
    g_k = g[kept].cpu().tolist()
    v_k = v[kept]
    s_k = ((v_k - mean) / std).cpu().tolist()
    v_k = v_k.cpu().tolist()
    peaks: List[RotationPeak] = [
        RotationPeak(alpha=a_k[i], beta=b_k[i], gamma=g_k[i], score=v_k[i], sigma=s_k[i])
        for i in range(len(a_k))
    ]
    return peaks
