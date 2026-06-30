"""Per-atom analytic electron-density splat radius.

The effective real-space width of atom *i* is ``sigma_eff_i = sqrt((b_form_i +
B_i) / 8pi^2)``, where ``b_form_i`` is the broadest ITC92 Gaussian width of the
atom's element and ``B_i`` its ADP. Truncating at ``r = N_sigma * sigma_eff``
carries the same fractional tail mass for every atom by construction (3.5 sigma
-> ~0.09%, 4 sigma -> ~0.013%), so the structure-wide F-truncation residual is
governed by the single knob ``N_sigma`` (``torchref.sigma_cutoff_ed``) rather
than by the worst aggregate atom -- the failure mode of the old per-structure
scalar radius.

The radius is quantized up to ``round_to`` (0.25 A) and clamped to ``[r_lo,
r_hi]`` so the downstream offset caches stay small. ``round_to``/``r_lo``/``r_hi``
are fixed policy constants here; the only user-facing knob is ``n_sigma``.
"""

from __future__ import annotations

import math

import torch

EIGHT_PI2 = 8.0 * math.pi**2

# Fixed policy constants (not user-exposed; the sigma cutoff is the only knob).
R_LO = 2.0
R_HI = 7.0
ROUND_TO = 0.25


def _ceil_round(x: torch.Tensor, round_to: float = ROUND_TO) -> torch.Tensor:
    """Round ``x`` up to the nearest multiple of ``round_to``."""
    return torch.ceil(x / round_to) * round_to


def _u6_to_u3(u: torch.Tensor) -> torch.Tensor:
    """(n,6) U components [U11,U22,U33,U12,U13,U23] -> (n,3,3) symmetric tensor."""
    n = u.shape[0]
    U3 = u.new_zeros(n, 3, 3)
    U3[:, 0, 0] = u[:, 0]
    U3[:, 1, 1] = u[:, 1]
    U3[:, 2, 2] = u[:, 2]
    U3[:, 0, 1] = U3[:, 1, 0] = u[:, 3]
    U3[:, 0, 2] = U3[:, 2, 0] = u[:, 4]
    U3[:, 1, 2] = U3[:, 2, 1] = u[:, 5]
    return U3


def sigma_eff_iso(adp: torch.Tensor, B_widths: torch.Tensor) -> torch.Tensor:
    """``sigma_eff_i = sqrt((max_k B_widths[i,k] + adp_i) / 8pi^2)``, shape (n,).

    Parameters
    ----------
    adp : torch.Tensor
        Per-atom isotropic ADP (B-factor), shape (n,).
    B_widths : torch.Tensor
        ITC92 Gaussian widths, shape (n, 5).
    """
    b_form = B_widths.detach().max(dim=1).values  # broadest component
    return torch.sqrt(((b_form + adp.detach()).clamp(min=1e-6)) / EIGHT_PI2)


def per_atom_radius_iso(
    adp: torch.Tensor,
    B_widths: torch.Tensor,
    *,
    n_sigma: float,
    r_lo: float = R_LO,
    r_hi: float = R_HI,
    round_to: float = ROUND_TO,
) -> torch.Tensor:
    """Per-atom isotropic splat radius = clamp(ceil_round(n_sigma*sigma_eff), [r_lo,r_hi])."""
    sigma = sigma_eff_iso(adp, B_widths)
    r = _ceil_round(n_sigma * sigma, round_to)
    return r.clamp(min=r_lo, max=r_hi)


def sigma_eff_aniso(B_widths: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Broadest real-space width of an anisotropic atom, shape (n,).

    Along the principal axis with the largest U eigenvalue ``lambda_max``, the
    effective B is ``b_form + 8pi^2*lambda_max`` (the aniso analogue of the iso
    ``b_form + B``), so ``sigma_max^2 = b_form/8pi^2 + lambda_max``. The
    isotropic bounding box must contain this broadest direction.

    Parameters
    ----------
    B_widths : torch.Tensor
        ITC92 Gaussian widths, shape (n, 5).
    u : torch.Tensor
        Anisotropic U parameters [U11,U22,U33,U12,U13,U23], shape (n, 6).
    """
    b_form = B_widths.detach().max(dim=1).values  # broadest ITC92 width
    lam_max = torch.linalg.eigvalsh(_u6_to_u3(u.detach())).max(dim=1).values
    return torch.sqrt((b_form / EIGHT_PI2 + lam_max).clamp(min=1e-6))


def per_atom_radius_aniso(
    B_widths: torch.Tensor,
    u: torch.Tensor,
    *,
    n_sigma: float,
    r_lo: float = R_LO,
    r_hi: float = R_HI,
    round_to: float = ROUND_TO,
) -> torch.Tensor:
    """Per-atom isotropic-box radius for anisotropic atoms (largest principal axis)."""
    sigma = sigma_eff_aniso(B_widths, u)
    r = _ceil_round(n_sigma * sigma, round_to)
    return r.clamp(min=r_lo, max=r_hi)
