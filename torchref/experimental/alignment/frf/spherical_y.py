"""Spherical harmonics ``Y_l^m(θ, φ)`` table.

Phaser source: ``phaser/lib/sphericalY.h`` (Condon-Shortley phase, the
standard physics convention).

We build the table once per (θ, φ) batch and reuse across all
``(l, m)`` indices needed.
"""
from __future__ import annotations

import math

import torch


def _normalised_associated_legendre(
    cos_theta: torch.Tensor, L: int
) -> torch.Tensor:
    """Compute the *normalised* associated Legendre polynomials.

    Returns ``P[l, m, ...] = sqrt((2l+1)/(4π) · (l-m)!/(l+m)!) · P_l^m(cos θ)``
    for l ∈ [0, L), m ∈ [0, l], with the standard recurrence (no Condon-Shortley
    sign — that's applied at the Y_lm level).

    Out-of-range entries (m > l) are zero.
    """
    device = cos_theta.device
    x = cos_theta.to(torch.float64)
    sin_t = torch.sqrt(torch.clamp(1.0 - x * x, min=0.0))

    plm = torch.zeros((L, L, *x.shape), dtype=torch.float64, device=device)
    # m = 0 sector: standard Legendre recurrence with normalisation built in.
    plm[0, 0] = math.sqrt(1.0 / (4.0 * math.pi))
    if L > 1:
        plm[1, 0] = math.sqrt(3.0 / (4.0 * math.pi)) * x
    for l in range(2, L):
        a = math.sqrt((2 * l + 1) * (2 * l - 1)) / l
        b = math.sqrt((2 * l + 1) / (2 * l - 3)) * (l - 1) / l
        plm[l, 0] = a * x * plm[l - 1, 0] - b * plm[l - 2, 0]

    # m > 0 sector: build P_l^l from the previous diagonal, then recur up in l.
    for m in range(1, L):
        plm[m, m] = -math.sqrt((2 * m + 1) / (2 * m)) * sin_t * plm[m - 1, m - 1]
        if m + 1 < L:
            plm[m + 1, m] = math.sqrt(2 * m + 3) * x * plm[m, m]
        for l in range(m + 2, L):
            a = math.sqrt((2 * l + 1) * (2 * l - 1) / ((l - m) * (l + m)))
            b = math.sqrt(
                (2 * l + 1) * (l + m - 1) * (l - m - 1)
                / ((l - m) * (l + m) * (2 * l - 3))
            )
            plm[l, m] = a * x * plm[l - 1, m] - b * plm[l - 2, m]

    return plm


def ylm_table(
    theta: torch.Tensor, phi: torch.Tensor, L: int
) -> torch.Tensor:
    """Compute ``Y_l^m(θ, φ)`` for all l ∈ [0, L), m ∈ [-l, l].

    Convention: Condon-Shortley phase (the ``(-1)^m`` factor is in
    ``Y_l^m`` for m > 0). The normalised associated Legendre polynomial
    is real; the φ dependence is ``exp(i m φ)``.

    Parameters
    ----------
    theta, phi : torch.Tensor (real, same shape)
        Polar (θ ∈ [0, π]) and azimuthal (φ ∈ [0, 2π)) angles.

    Returns
    -------
    Y : torch.Tensor (complex128), shape (L, 2L-1, *theta.shape)
        ``Y[l, m + L - 1, ...] = Y_l^m(θ, φ)`` for |m| ≤ l, else 0.
    """
    if theta.shape != phi.shape:
        raise ValueError(
            f"theta {tuple(theta.shape)} and phi {tuple(phi.shape)} must agree"
        )
    device = theta.device
    plm = _normalised_associated_legendre(torch.cos(theta), L)   # (L, L, *)
    Y = torch.zeros(
        (L, 2 * L - 1, *theta.shape), dtype=torch.complex128, device=device
    )
    phi64 = phi.to(torch.float64)
    # m = 0
    Y[:, L - 1, ...] = plm[:, 0, ...].to(torch.complex128)
    for m in range(1, L):
        e_pos = torch.complex(torch.cos(m * phi64), torch.sin(m * phi64))
        e_neg = e_pos.conj()
        # Y_l^{+m} = (-1)^m * sqrt(...) P_l^m(cosθ) * exp(i m φ), already
        # includes the (-1)^m if we apply it here.
        sign = (-1) ** m
        Y[:, L - 1 + m, ...] = sign * plm[:, m, ...].to(torch.complex128) * e_pos
        Y[:, L - 1 - m, ...] = plm[:, m, ...].to(torch.complex128) * e_neg
    return Y
