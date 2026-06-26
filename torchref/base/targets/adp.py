"""ADP (B-factor) restraint NLLs: similarity, KL-divergence, locality."""

import math

import torch

from ._common import LOG_2PI
from ._dispatch import use_triton

EIGHT_PI2 = 8.0 * math.pi**2
# 6-vector order: [U11, U22, U33, U12, U13, U23]
_U6_DIAG = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
# off-diagonal U components appear twice in the symmetric 3x3 (Frobenius norm)
_U6_WCOMP = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)


def _adp_simu_math_eager(
    b: torch.Tensor,
    pair_indices: torch.Tensor,
    simu_sigma: torch.Tensor,
) -> torch.Tensor:
    diffs = b[pair_indices[:, 0]] - b[pair_indices[:, 1]]
    nll = (
        0.5 * (diffs / simu_sigma) ** 2
        + torch.log(simu_sigma)
        + 0.5 * LOG_2PI
    )
    return nll.sum()


def adp_simu_math(
    b: torch.Tensor,
    pair_indices: torch.Tensor,
    simu_sigma: torch.Tensor,
) -> torch.Tensor:
    """ADP similarity (SIMU) NLL on bonded-atom B-factor differences.

    Dispatches to
    :func:`torchref.base.targets.triton.adp_simu_math_triton` on CUDA
    float32 (~1.6× faster fwd+bw on A100). Falls back to eager
    otherwise.

    Parameters
    ----------
    b : torch.Tensor
        (N_atoms,) B-factors.
    pair_indices : torch.Tensor
        (N, 2) bonded-atom pairs to compare.
    simu_sigma : torch.Tensor
        Scalar sigma on the difference (a buffer in the target).
    """
    if use_triton(b):
        from .triton.adp_simu import adp_simu_math_triton
        return adp_simu_math_triton(b, pair_indices, simu_sigma)
    return _adp_simu_math_eager(b, pair_indices, simu_sigma)


# ----------------------------------------------------------------------
# Anisotropic ADP restraints (operate on the unified per-atom U6 tensor).
#
# Both restraints split into a magnitude channel on B_eq -- which reduces
# EXACTLY to the isotropic restraint when every atom is isotropic (D == 0,
# B_eq == B) -- and a deviatoric (anisotropy) channel on the traceless part
# of the B-tensor (8*pi^2 * U). For an iso atom the deviatoric part is 0, so
# iso<->aniso pairs are handled natively with no special casing.
# ----------------------------------------------------------------------
def u6_b_eq(u6: torch.Tensor) -> torch.Tensor:
    """Equivalent isotropic B from a unified U6 (== B for iso atoms)."""
    return (EIGHT_PI2 / 3.0) * (u6[..., 0] + u6[..., 1] + u6[..., 2])


def u6_deviatoric(u6: torch.Tensor) -> torch.Tensor:
    """Traceless part of U as a 6-vector (identically 0 for iso atoms)."""
    tr3 = (u6[..., 0] + u6[..., 1] + u6[..., 2]) / 3.0
    d = u6.clone()
    d[..., 0] = u6[..., 0] - tr3
    d[..., 1] = u6[..., 1] - tr3
    d[..., 2] = u6[..., 2] - tr3
    return d


def adp_simu_aniso_math(
    u6: torch.Tensor,
    pair_indices: torch.Tensor,
    simu_sigma: torch.Tensor,
    simu_sigma_aniso: torch.Tensor,
) -> torch.Tensor:
    """Anisotropic SIMU NLL on bonded-atom U tensors.

    Magnitude channel (on B_eq) reduces EXACTLY to :func:`adp_simu_math` when
    all atoms are isotropic; the deviatoric channel restrains tensor shape.
    """
    beq = u6_b_eq(u6)
    dmag = beq[pair_indices[:, 0]] - beq[pair_indices[:, 1]]
    nll = 0.5 * (dmag / simu_sigma) ** 2 + torch.log(simu_sigma) + 0.5 * LOG_2PI
    dev = EIGHT_PI2 * u6_deviatoric(u6)
    ddev = dev[pair_indices[:, 0]] - dev[pair_indices[:, 1]]
    wcomp = u6.new_tensor(_U6_WCOMP)
    nll_dev = 0.5 * (wcomp * (ddev / simu_sigma_aniso) ** 2).sum(dim=-1)
    return nll.sum() + nll_dev.sum()


def adp_locality_aniso_math(
    u6: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_distances: torch.Tensor,
    sigma_aniso: torch.Tensor,
) -> torch.Tensor:
    """Anisotropic locality NLL (k-NN, inverse-distance weighted).

    Magnitude channel (on log B_eq) reproduces the isotropic locality loss when
    all atoms are isotropic. The deviatoric channel restrains tensor shape and
    is kept on the SAME (dimensionless) scale as the magnitude channel: the
    isotropic channel penalises differences of ``log B_eq`` (a relative/ratio
    quantity with a fixed log-sigma of 0.5), so the anisotropy channel
    penalises differences of the *fractional* anisotropy ``dev / B_eq`` rather
    than absolute deviatoric U. ``sigma_aniso`` is therefore dimensionless and
    comparable to that 0.5 log-sigma (not an absolute Å² value).
    """
    beq = u6_b_eq(u6)
    beq_c = beq.clamp(min=1e-3)
    log_b = torch.log(beq_c)
    w = 1.0 / (neighbor_distances + 1e-6)
    diff = log_b.unsqueeze(1) - log_b[neighbor_indices]
    mag = (w * (diff / 0.5) ** 2).sum()
    # Fractional anisotropy: 8*pi^2 * deviatoric(U) / B_eq is the dimensionless
    # analogue of log B_eq -- same scale as the magnitude channel above.
    frac = (EIGHT_PI2 * u6_deviatoric(u6)) / beq_c.unsqueeze(-1)
    dfrac = frac.unsqueeze(1) - frac[neighbor_indices]
    wcomp = u6.new_tensor(_U6_WCOMP)
    dev_term = (w.unsqueeze(-1) * wcomp * (dfrac / sigma_aniso) ** 2).sum()
    return mag + dev_term

