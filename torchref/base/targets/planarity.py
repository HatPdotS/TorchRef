"""Planarity restraint NLL."""

from typing import List, Tuple

import torch

from ._common import LOG_2PI
from ._dispatch import use_triton


def _plane_normals_detached(centered_f64: torch.Tensor) -> torch.Tensor:
    """SVD-derived plane normal (right singular vector at smallest σ).

    Caller must wrap in ``torch.no_grad()``; the result is intentionally
    detached so backward flows only through the deviation projection.
    """
    _U, _S, Vh = torch.linalg.svd(centered_f64, full_matrices=False)
    return Vh[:, -1, :]


def _planarity_math_eager(
    xyz: torch.Tensor,
    plane_groups: List[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    if not plane_groups:
        return torch.zeros((), device=xyz.device, dtype=xyz.dtype)
    all_nlls = []
    for indices, sigmas in plane_groups:
        positions = xyz[indices]
        centroids = positions.mean(dim=1, keepdim=True)
        centered = positions - centroids
        with torch.no_grad():
            normals = _plane_normals_detached(
                centered.detach().to(torch.float64)
            ).to(xyz.dtype)
        deviations = torch.einsum("paj,pj->pa", centered, normals)
        nll = 0.5 * (deviations / sigmas) ** 2 + torch.log(sigmas) + 0.5 * LOG_2PI
        all_nlls.append(nll.flatten())
    return torch.cat(all_nlls).sum()


def planarity_math(
    xyz: torch.Tensor,
    plane_groups: List[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Planarity NLL summed over plane-size buckets.

    Matches ``PlanarityTarget.forward``. Each entry of ``plane_groups`` is
    ``(indices, sigmas)`` where ``indices`` has shape ``(P, n_atoms)``
    with ``n_atoms > 3`` (3-atom planes have zero deviation by
    construction and are skipped by the caller).

    The plane normal is computed in float64 via SVD and detached — gradients
    flow through the deviation projection but not through the
    eigendecomposition.

    Dispatches to
    :func:`torchref.base.targets.triton.planarity.planarity_math_triton`
    on CUDA fp32 (per-bucket fused gather/centroid/project/NLL kernel +
    analytic backward). Falls back to eager otherwise.
    """
    if use_triton(xyz):
        from .triton.planarity import planarity_math_triton
        return planarity_math_triton(xyz, plane_groups)
    return _planarity_math_eager(xyz, plane_groups)
