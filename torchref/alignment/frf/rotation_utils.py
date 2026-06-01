"""Pure-geometry helpers shared by the FRF, the rescore, and tests.

Edmonds active ZYZ convention throughout: a rotation matrix is built as
``R = R_z(α) R_y(β) R_z(γ)``.  ``α, γ ∈ [0, 2π)``, ``β ∈ [0, π]``.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


def rotation_matrix_from_edmonds_euler(
    alpha: float, beta: float, gamma: float, dtype=torch.float64,
) -> torch.Tensor:
    """Build ``R = R_z(α) R_y(β) R_z(γ)`` (Edmonds active ZYZ).

    Equivalent to passing ``[γ, β, α]`` to
    ``torchref.alignment.transform.rotation_matrix_from_euler``.
    """
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    Rz_a = torch.tensor([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    Ry_b = torch.tensor([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]], dtype=dtype)
    Rz_c = torch.tensor([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    return Rz_a @ Ry_b @ Rz_c


def rotation_matrix_from_edmonds_euler_batch(
    alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor,
) -> torch.Tensor:
    """Vectorised Edmonds ZYZ Euler → ``R``.

    ``alpha``, ``beta``, ``gamma``: identically-shaped real tensors. Returns
    ``(..., 3, 3)`` in the same dtype/device as the inputs.
    """
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)
    zero = torch.zeros_like(alpha)
    one = torch.ones_like(alpha)
    Rz_a = torch.stack([
        torch.stack([ca, -sa, zero], dim=-1),
        torch.stack([sa,  ca, zero], dim=-1),
        torch.stack([zero, zero, one], dim=-1),
    ], dim=-2)
    Ry_b = torch.stack([
        torch.stack([cb,  zero, sb], dim=-1),
        torch.stack([zero, one, zero], dim=-1),
        torch.stack([-sb, zero, cb], dim=-1),
    ], dim=-2)
    Rz_c = torch.stack([
        torch.stack([cg, -sg, zero], dim=-1),
        torch.stack([sg,  cg, zero], dim=-1),
        torch.stack([zero, zero, one], dim=-1),
    ], dim=-2)
    return Rz_a @ Ry_b @ Rz_c


def edmonds_euler_from_rotation_matrix(R: torch.Tensor) -> Tuple[float, float, float]:
    """Recover ``(α, β, γ)`` such that ``R = R_z(α) R_y(β) R_z(γ)``.

    Returns angles in radians; ``α, γ ∈ [0, 2π)``, ``β ∈ [0, π]``. Singular
    when ``β = 0`` or ``π`` (only ``α+γ`` is determined); in those cases
    ``γ=0`` is returned.
    """
    R = R.to(torch.float64)
    cos_beta = R[2, 2].clamp(-1.0, 1.0).item()
    beta = math.acos(cos_beta)
    sin_beta = math.sin(beta)
    if abs(sin_beta) < 1e-9:
        alpha = math.atan2(R[1, 0].item(), R[0, 0].item())
        gamma = 0.0
    else:
        alpha = math.atan2(R[1, 2].item(), R[0, 2].item())
        gamma = math.atan2(R[2, 1].item(), -R[2, 0].item())
    alpha = alpha % (2.0 * math.pi)
    gamma = gamma % (2.0 * math.pi)
    return alpha, beta, gamma


def rotation_angular_distance_deg(R1: torch.Tensor, R2: torch.Tensor) -> float:
    """Geodesic distance on SO(3) in degrees: ``arccos((tr(R1 R2^T) − 1)/2)``."""
    R = R1.to(torch.float64) @ R2.to(torch.float64).T
    tr = (R[0, 0] + R[1, 1] + R[2, 2]).clamp(-1.0, 3.0).item()
    cos_a = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    return math.degrees(math.acos(cos_a))
