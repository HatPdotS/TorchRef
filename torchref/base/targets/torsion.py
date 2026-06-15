"""Torsion restraint NLL: unimodal von Mises + omega cis/trans mixture."""

import torch

from ._common import DEG2RAD, LOG_2PI, torsions_from_xyz
from ._dispatch import use_triton


def _torsion_omega_math_eager(
    xyz: torch.Tensor,
    idx: torch.Tensor,
    sigmas_deg: torch.Tensor,
    is_proline: torch.Tensor,
    w_cis_proline: float = 0.05,
    w_cis_general: float = 0.0005,
) -> torch.Tensor:
    omega_deg = torsions_from_xyz(xyz, idx)
    omega_rad = omega_deg * DEG2RAD

    sigmas_rad = sigmas_deg * DEG2RAD
    kappa = torch.clamp(1.0 / (sigmas_rad ** 2), min=1e-3, max=1e4)
    log_i0_kappa = torch.log(torch.special.i0e(kappa)) + kappa
    log_norm = LOG_2PI + log_i0_kappa

    w_cis = torch.where(
        is_proline,
        torch.tensor(w_cis_proline, device=kappa.device, dtype=kappa.dtype),
        torch.tensor(w_cis_general, device=kappa.device, dtype=kappa.dtype),
    )
    w_trans = 1.0 - w_cis

    cos_omega = torch.cos(omega_rad)
    log_p_trans = torch.log(w_trans) - kappa * cos_omega
    log_p_cis = torch.log(w_cis) + kappa * cos_omega

    log_mixture = torch.logsumexp(torch.stack([log_p_trans, log_p_cis]), dim=0)
    return (log_norm - log_mixture).sum()


def torsion_omega_math(
    xyz: torch.Tensor,
    idx: torch.Tensor,
    sigmas_deg: torch.Tensor,
    is_proline: torch.Tensor,
    w_cis_proline: float = 0.05,
    w_cis_general: float = 0.0005,
) -> torch.Tensor:
    """Omega cis/trans mixture NLL.

    Mirrors ``targets.geometry.torsions._omega_mixture_nll`` plus the
    omega-angle computation. Models each ω as a 2-component mixture:
    ``w_trans VM(ω; π, κ) + w_cis VM(ω; 0, κ)``.

    Dispatches to
    :func:`torchref.base.targets.triton.torsion_omega_math_triton` on
    CUDA float32 (~5.6× faster fwd+bw on A100). Falls back to eager
    otherwise.
    """
    if use_triton(xyz):
        from .triton.torsion import torsion_omega_math_triton
        return torsion_omega_math_triton(
            xyz, idx, sigmas_deg, is_proline, w_cis_proline, w_cis_general,
        )
    return _torsion_omega_math_eager(
        xyz, idx, sigmas_deg, is_proline, w_cis_proline, w_cis_general,
    )
