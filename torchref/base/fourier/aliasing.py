"""Aliasing control for the FFT structure-factor route: grid period and B-blurring.

A density sampled on a grid of ``N_i`` points per axis has a periodic spectrum: the
FFT returns at reflection ``h`` the sum of the true transform at every ``h + m * N``.
In Cartesian reciprocal space those images sit on the lattice spanned by ``N_i a*_i``,
whose shortest vector has length ``sigma`` (Å⁻¹, :func:`alias_period`). One Gaussian
of total width ``g`` (Å², form-factor width plus atomic B) then carries, at
``|s| = s``, a nearest image at ``sigma - s`` whose amplitude relative to the true
term is::

    exp(-g * ((sigma - s)**2 - s**2) / 4) = exp(-g * kappa(s)),
    kappa(s) = sigma * (sigma - 2 s) / 4                              (Å⁻²)

so aliasing is set by the *sharpest* density on the grid, not by the resolution alone.
Every ITC92 atom has a zero-width constant term, so the sharpest Gaussian of an atom is
its ADP itself and ``exp(-B_atom * kappa)`` bounds that atom's relative error.

Blurring every atom by ``B_add`` and multiplying ``F`` by ``exp(B_add s**2 / 4)``
afterwards leaves ``F`` unchanged in exact arithmetic and scales every alias term by
``exp(-B_add * kappa)``. Hence the blur that brings the worst atom to a relative error
``rtol`` at the cutoff ``s_max`` is::

    B_add = max(0, ln(1 / rtol) / kappa(s_max) - B_min)

(:func:`blur_b`), and the grid's job is only to keep that blur representable by the
splat (:func:`grid_oversampling`). ``rtol`` is :func:`torchref.config.get_aliasing_tolerance`.

The three pieces split by what they depend on: :func:`grid_oversampling` reads only the
resolution, so the grid never changes during refinement; :func:`alias_kappa` reads the
grid; :func:`blur_b` reads the current ADPs through :func:`sharpest_b` and is cheap
enough to evaluate on every structure-factor call.
"""

import itertools
import math
from typing import Optional, Sequence

import torch

from torchref.base.electron_density.radius_policy import (
    EIGHT_PI2,
    R_HI,
    _min_eig_sym3,
    _u6_to_u3,
)

# Broadest ITC92 Gaussian width of the elements a macromolecular model normally holds
# (Fe is the widest through Zn at 76.9 Å²). The grid guard subtracts it from the widest
# Gaussian the splat can hold untruncated, so the blurred sharpest atom keeps its full
# sigma cutoff.
_B_FORM_BROADEST = 80.0


def alias_period(inv_frac_matrix: torch.Tensor, grid_shape: Sequence[int]) -> float:
    """Length of the shortest alias vector of a grid, ``sigma`` in Å⁻¹.

    Parameters
    ----------
    inv_frac_matrix : torch.Tensor
        Cartesian-to-fractional matrix ``(3, 3)``, whose rows are ``a*, b*, c*`` in Å⁻¹.
    grid_shape : sequence of int
        Grid dimensions ``(nx, ny, nz)``.

    Returns
    -------
    float
        ``min |sum_i m_i N_i a*_i|`` over nonzero integer ``m`` with ``|m_i| <= 2``;
        for an orthogonal cell this is ``min_i N_i / a_i``, the inverse of the coarsest
        grid spacing. Combinations reach the true minimum for every reduced cell.
    """
    rows = inv_frac_matrix.detach().cpu()
    counts = torch.tensor([float(n) for n in grid_shape], dtype=rows.dtype)
    m = torch.tensor(
        [c for c in itertools.product(range(-2, 3), repeat=3) if any(c)],
        dtype=rows.dtype,
    )
    period = rows * counts[:, None]
    return float((m @ period).norm(dim=1).min())


def alias_kappa(period: float, s_max: float) -> Optional[float]:
    """Alias decay rate ``kappa = sigma (sigma - 2 s_max) / 4`` in Å⁻².

    Parameters
    ----------
    period : float
        Shortest alias vector ``sigma`` in Å⁻¹, from :func:`alias_period`.
    s_max : float
        Highest ``|s| = 1 / d`` to protect, in Å⁻¹.

    Returns
    -------
    float or None
        ``kappa``, or None when the grid does not reach Nyquist for ``s_max``
        (``sigma <= 2 s_max``): no blur can then separate a reflection from its image.
    """
    kappa = period * (period - 2.0 * s_max) / 4.0
    return kappa if kappa > 0.0 else None


def widest_splat_b(n_sigma: float) -> float:
    """Widest Gaussian (Å²) the splat holds out to ``n_sigma`` before the radius clamp.

    The splat radius is clamped to ``R_HI`` Å, so a Gaussian of width
    ``8 pi^2 (R_HI / n_sigma)^2`` or more is truncated at fewer sigmas than configured.
    """
    return EIGHT_PI2 * (R_HI / n_sigma) ** 2


def grid_oversampling(
    d_min: float, rtol: float, n_sigma: float, oversampling: float
) -> float:
    """Oversampling factor (grid period over ``1 / d_min``) for a resolution.

    ``oversampling`` unless the blur the worst case (an atom with ``B = 0``) needs
    would push the broadest form factor past :func:`widest_splat_b`; the grid is then
    refined until it does not. Depends only on the resolution and the settings, never
    on the model, so the grid stays fixed while ADPs refine.

    Parameters
    ----------
    d_min : float
        Resolution cutoff in Å.
    rtol : float
        Aliasing tolerance, in ``(0, 1)``.
    n_sigma : float
        Splat sigma cutoff.
    oversampling : float
        Baseline factor, :data:`torchref.config.NYQUIST_OVERSAMPLING`.

    Returns
    -------
    float
        ``max(oversampling, 1 + sqrt(1 + 4 ln(1/rtol) d_min^2 / B_cap))`` with
        ``B_cap = widest_splat_b(n_sigma) - 80``; from ``kappa = ln(1/rtol) / B_cap``.
        With the defaults the guard takes over below about 7.5 Å.
    """
    b_cap = widest_splat_b(n_sigma) - _B_FORM_BROADEST
    if b_cap <= 0.0:
        return oversampling
    needed = 1.0 + math.sqrt(1.0 + 4.0 * math.log(1.0 / rtol) * d_min**2 / b_cap)
    return max(oversampling, needed)


def sharpest_b(
    adp_iso: Optional[torch.Tensor], u_aniso: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Smallest atomic B (Å²) over isotropic and anisotropic atoms, as a 0-d tensor.

    Parameters
    ----------
    adp_iso : torch.Tensor or None
        Isotropic B-factors in Å², any shape (batched atoms included).
    u_aniso : torch.Tensor or None
        Cartesian ``[U11, U22, U33, U12, U13, U23]`` in Å², shape ``(..., 6)``; each
        atom counts with ``8 pi^2`` times its smallest principal U, its sharpest
        direction.

    Returns
    -------
    torch.Tensor or None
        Detached 0-d tensor on the inputs' device, or None without atoms. Negative
        values (a non-positive-definite U) are clamped to zero. No host sync.
    """
    parts = []
    if adp_iso is not None and adp_iso.numel() > 0:
        parts.append(adp_iso.detach().reshape(-1).min())
    if u_aniso is not None and u_aniso.numel() > 0:
        u = u_aniso.detach().reshape(-1, 6)
        parts.append(EIGHT_PI2 * _min_eig_sym3(_u6_to_u3(u)).min())
    if not parts:
        return None
    return torch.stack(parts).min().clamp(min=0.0)


def blur_b(
    b_min: torch.Tensor, kappa: float, rtol: float, b_max: float
) -> torch.Tensor:
    """Isotropic blur (Å²) that bounds every atom's alias error at ``rtol``.

    Parameters
    ----------
    b_min : torch.Tensor
        Sharpest atomic B in Å², 0-d, from :func:`sharpest_b`.
    kappa : float
        Alias decay rate at the cutoff in Å⁻², from :func:`alias_kappa`.
    rtol : float
        Aliasing tolerance, in ``(0, 1)``.
    b_max : float
        Upper bound on the returned blur in Å², normally :func:`widest_splat_b`.

    Returns
    -------
    torch.Tensor
        ``ceil(max(0, ln(1/rtol)/kappa - b_min))`` clamped to ``b_max``, 0-d. Rounded
        up to whole Å² so the blur, and with it ``F``, is piecewise constant in the
        ADPs instead of shifting by alias-sized amounts on every parameter update.
    """
    raw = math.log(1.0 / rtol) / kappa - b_min
    return torch.ceil(raw).clamp(min=0.0, max=b_max)
