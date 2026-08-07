"""ADP (B-factor) restraint NLLs: similarity (SIMU), locality, rigid-bond
(DELU), and the shifted inverse-gamma distribution prior, with isotropic and
anisotropic variants on the unified per-atom U6 tensor."""

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
    float32. Falls back to eager
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

    Parameters
    ----------
    u6 : torch.Tensor
        (N_atoms, 6) per-atom U tensors in 6-vector order
        ``[U11, U22, U33, U12, U13, U23]``.
    pair_indices : torch.Tensor
        (N, 2) bonded-atom pairs to compare.
    simu_sigma : torch.Tensor
        Scalar sigma on the B_eq (magnitude) difference.
    simu_sigma_aniso : torch.Tensor
        Scalar sigma on the deviatoric (anisotropy) component differences.
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

    Parameters
    ----------
    u6 : torch.Tensor
        (N_atoms, 6) per-atom U tensors in 6-vector order
        ``[U11, U22, U33, U12, U13, U23]``.
    neighbor_indices : torch.Tensor
        (N_atoms, k) k-nearest-neighbour atom indices per atom.
    neighbor_distances : torch.Tensor
        (N_atoms, k) distances to those neighbours; used as inverse-distance
        weights ``1 / (d + 1e-6)``.
    sigma_aniso : torch.Tensor
        Scalar dimensionless sigma on the fractional-anisotropy differences
        (comparable to the fixed 0.5 log-sigma of the magnitude channel).
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


def adp_sigd_math(
    b: torch.Tensor,
    alpha: torch.Tensor,
    b_shift: torch.Tensor,
) -> torch.Tensor:
    """Shifted inverse-gamma (SIGD) prior NLL on the B-factor distribution.

    Masmaliyeva & Murshudov (2019), *Acta Cryst.* D **75**, 505-518, showed that
    macromolecular B values follow a shifted inverse-gamma distribution rather
    than the log-normal that a Gaussian-in-log(B) restraint assumes. For
    ``x = B - B0`` distributed as ``InvGamma(alpha, beta)``::

        -log p(x) = -alpha log(beta) + lgamma(alpha)
                    + (alpha + 1) log(x) + beta / x

    The scale is set from the **detached** mean so that the prior's mean matches
    the data's, ``beta = mean(x).detach() * (alpha - 1)``. That is the direct
    analogue of the detached ``mu_data`` in the log-normal KL term this replaces:
    the restraint cannot drive the overall B level up or down, it only penalises
    departures from the SIGD *shape*.

    The returned per-atom NLL is offset by its value at the distribution mode
    ``x_mode = beta / (alpha + 1)``, so each atom's contribution is ``>= 0``,
    vanishing only for an atom sitting exactly at the mode. The *sum* does not
    reach zero for real data: ``beta`` tracks the data mean, so ``x_mode`` is
    ``(alpha-1)/(alpha+1)`` of it and a uniform B distribution still costs
    ``(alpha+1) log((alpha+1)/(alpha-1)) - 2`` per atom (0.645 at alpha=3.5).
    The offset is a fixed reference, not an attainable floor. Because ``beta``
    is detached, the two
    B-independent terms (``-alpha log beta`` and ``lgamma(alpha)``) cancel
    exactly against that offset, leaving::

        loss_i = (alpha + 1) log(x_i / x_mode) + beta (1/x_i - 1/x_mode)

    which is what is evaluated -- algebraically identical to the offset NLL, with
    no ``lgamma`` call and no large cancelling terms.

    Two properties this form has and the log-normal KL it replaces did not:
    it is finite for a perfectly uniform B distribution (the KL diverged there),
    and it is monotonically increasing in ``std(log B)``, so it can never reward
    spreading the distribution out.

    Parameters
    ----------
    b : torch.Tensor
        (N_atoms,) B-factors (Å²). For anisotropic models pass ``B_eq`` from
        :func:`u6_b_eq`.
    alpha : torch.Tensor
        Scalar shape parameter. Sets the log-width via
        ``std(log B) = sqrt(trigamma(alpha))`` (independent of ``beta``), the
        analogue of sigma in the log-normal form. Larger alpha is a narrower
        reference distribution and so a stronger restraint.
    b_shift : torch.Tensor
        Scalar shift ``B0``. Zero corresponds to unsharpened data; a non-zero
        B0 is the signature of sharpening/blurring, which the same authors
        recommend avoiding during refinement.

    Returns
    -------
    torch.Tensor
        Scalar sum over atoms, ``>= 0``.
    """
    x = (b - b_shift).clamp(min=1e-3)
    beta = (x.mean() * (alpha - 1.0)).detach()
    x_mode = beta / (alpha + 1.0)
    nll = (alpha + 1.0) * torch.log(x / x_mode) + beta * (1.0 / x - 1.0 / x_mode)
    return nll.sum()


def adp_rigid_bond_aniso_math(
    u6: torch.Tensor,
    xyz: torch.Tensor,
    pair_indices: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Anisotropic rigid-bond (Hirshfeld DELU) NLL on the unified U6.

    For each bond the mean-square displacement amplitude along the bond,
    ``z = l^T U l`` (``l`` the unit bond vector), should match for the two
    atoms: ``Δz = z_i - z_j ~ 0``. Reduces EXACTLY to the isotropic
    ``Δz = (B_i - B_j) / 8 pi^2`` form when both atoms are isotropic
    (``l^T (U_iso I) l = U_iso`` for any direction), so iso<->aniso bonds are
    handled natively. Gradient flows to both the U tensors and the coordinates
    (the Hirshfeld test couples ADP and geometry).
    """
    M = u6.new_zeros(u6.shape[0], 3, 3)
    M[:, 0, 0] = u6[:, 0]
    M[:, 1, 1] = u6[:, 1]
    M[:, 2, 2] = u6[:, 2]
    M[:, 0, 1] = M[:, 1, 0] = u6[:, 3]
    M[:, 0, 2] = M[:, 2, 0] = u6[:, 4]
    M[:, 1, 2] = M[:, 2, 1] = u6[:, 5]
    r = xyz[pair_indices[:, 1]] - xyz[pair_indices[:, 0]]
    l = r / torch.sqrt((r * r).sum(-1, keepdim=True) + 1e-8)
    z1 = torch.einsum("bi,bij,bj->b", l, M[pair_indices[:, 0]], l)
    z2 = torch.einsum("bi,bij,bj->b", l, M[pair_indices[:, 1]], l)
    dz = z1 - z2
    return (0.5 * (dz / sigma) ** 2 + math.log(sigma) + 0.5 * LOG_2PI).sum()

