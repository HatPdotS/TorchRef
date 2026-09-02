"""Fast translation search: where in the cell does an oriented model sit?

A translation shifts phase, ``F(h, t) = F(h) exp(2 pi i h.t)``, so scoring every
``t`` on a grid is a Fourier transform rather than a scan. The Crowther-Blow
form used here accumulates the pair coefficients
``sum_h c(h) G_i*(h) G_j(h)`` onto a reciprocal grid at
``(h R_j - h R_i) mod G`` and takes one inverse FFT, which replaces ``G^3`` grid
evaluations with a single transform.

Both sides of that sum are **normalised**. The observed side is the rotation
search's own LERF1 intensity, ``cw (E_obs^2 - 1) w sigma_A^2``, built from the
run's one Wilson fit; the calculated side is the oriented model's transform
divided by its own Wilson curve, so ``<|E_calc(h, t)|^2> = 1`` per shell for
every candidate. The score is then a covariance of two normalised intensities
and every resolution shell carries the weight the model error gives it. The
previous form divided raw ``|F_calc|^2`` by its own sum, which is not a
correlation: on 2DQ6 it was 0.665 at a position 41 A from the deposited pose
and 0.350 at the pose itself, and the search followed it there.

The grid is sized to the resolution of the translation set, one FFT per
candidate, and the best few peaks are re-scored with the full Rice/Woolfson
likelihood at fixed ``sigma_A``. That likelihood is also what ranks the
candidates against each other.

The observed side is prepared **once** per run, by :class:`TranslationObs`,
and reused for every orientation. Normalisation, weighting and model error are
properties of the observations and the search model, which do not change when
the model moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

from torchref.base.targets.xray_likelihoods import rice_per_refl
from torchref.config import get_complex_dtype, get_default_device, get_float_dtype
from torchref.scaling import WilsonNormaliser
from torchref.scaling.weighting import (inverse_variance_weight,
                                        normalise_weight, snr_from_amplitude)
from torchref.symmetry.symmetry import find_fft_friendly_size

from .frf.preprocessing import build_lerf1_intensity, eterm_sigma_a

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...model.model_ft import ModelFT


#: Chebyshev order of the Wilson fit. Matches the rotation function's
#: ``frf.api.WILSON_N_COEFF``: the two stages score the same observations and a
#: different order on each would be two normalisations again.
WILSON_N_COEFF = 6

#: Largest FFT grid per axis. 256^3 complex64 is 134 MB, which bounds the
#: translation map for an uncut high-resolution set on a long cell; at the
#: default window the grid never reaches it.
MAX_GRID_PER_AXIS = 256


@dataclass
class TranslationObs:
    """The observed side of a translation search, normalised and weighted once.

    Everything here is a property of the observations and of the search model's
    expected error, so none of it changes when the model rotates or moves.

    Attributes
    ----------
    F_obs, hkl, s_mag, centric, eps
        The masked observations and their crystallographic bookkeeping.
    E_obs : torch.Tensor
        ``F / sqrt(eps Sigma(s))``, with ``Sigma`` the shared Wilson fit, so
        ``<E^2> = 1`` as an identity of that fit.
    weight : torch.Tensor
        Mean-1 inverse-variance weight, from measurement error and model error
        in one denominator. Uniform when the data carry no sigmas.
    sigma_a : torch.Tensor
        The Luzzati fall-off ``exp(-(2 pi^2 / 3) s^2 vrms^2)`` for the search
        model's expected coordinate error -- the same term the rotation
        function weights with. It is the ``D`` of the likelihood and the
        calc-side weight of the fast search. A prior, not a fit: nothing can be
        fitted before the model is placed.
    coeff : torch.Tensor
        The fast search's per-reflection coefficient,
        ``cw (E_obs^2 - 1) weight sigma_A^2`` -- the rotation function's LERF1
        intensity with its calc-side ``sigma_A^2`` folded in. Centred, so a
        placement that puts calculated intensity everywhere gains nothing.
    fit : WilsonNormaliser
        Kept, not discarded. Anything comparing an observed curve against a
        calculated one needs the curve itself.
    """

    F_obs: torch.Tensor
    hkl: torch.Tensor
    s_mag: torch.Tensor
    centric: torch.Tensor
    eps: torch.Tensor
    E_obs: torch.Tensor
    weight: torch.Tensor
    sigma_a: torch.Tensor
    coeff: torch.Tensor
    fit: "WilsonNormaliser"

    @classmethod
    def build(
        cls,
        F_obs: torch.Tensor,
        hkl: torch.Tensor,
        spacegroup,
        real_cell,
        *,
        sig_F: Optional[torch.Tensor] = None,
        delta_vrms_A: float = 1.0,
        n_coeff: int = WILSON_N_COEFF,
        device=None,
    ) -> "TranslationObs":
        """Normalise and weight one set of observations.

        Parameters
        ----------
        F_obs : torch.Tensor
            ``(N,)`` observed amplitudes; complex input is coerced to ``|.|``.
        hkl : torch.Tensor
            ``(N, 3)`` integer Miller indices, matching ``F_obs`` row for row.
        spacegroup, real_cell
            Supply multiplicity, centricity and the reciprocal basis.
        sig_F : torch.Tensor, optional
            ``(N,)`` measurement errors. Without them the weight is uniform,
            which is the honest fallback: the varying part of the weight *is*
            the measurement term, and inventing one would be worse than not
            having it.
        delta_vrms_A : float
            R.m.s. coordinate error of the search model, which sets the model
            half of the variance budget and the likelihood's ``sigma_A``.
        """
        dev = get_default_device() if device is None else device
        real = get_float_dtype()
        F = F_obs.detach().to(dev)
        F = (F.abs() if F.is_complex() else F).to(real)
        hkl_i = hkl.detach().to(dev)

        rec_basis = real_cell.reciprocal_basis_matrix.to(dev).to(real)
        s_mag = (hkl_i.to(real) @ rec_basis).norm(dim=-1)

        hkl_l = hkl_i.round().to(torch.int64)
        # friedel=False: Wilson's <I> = eps*Sigma counts the operations mapping
        # h to itself, which add coherently and set the mean. The Friedel-folded
        # branch changes the distribution instead, and that is centricity --
        # which enters separately, as the Gamma shape.
        eps = spacegroup.epsilon(hkl_l, friedel=False).to(real).clamp(min=1.0)
        centric = spacegroup.is_centric(hkl_l).to(torch.bool)

        fit = WilsonNormaliser(
            F * F, s_mag, eps=eps, centric=centric, n_coeff=n_coeff,
        )
        sigma_a = eterm_sigma_a(s_mag, float(delta_vrms_A)).to(real)

        if sig_F is None:
            weight = torch.ones_like(F)
        else:
            sig = sig_F.detach().to(dev).to(real).abs()
            weight = normalise_weight(inverse_variance_weight(
                snr_from_amplitude(F, sig), sigma_a, eps=eps,
            ))

        E_obs = fit.E.to(real)
        coeff = build_lerf1_intensity(E_obs, centric, weight=weight) * sigma_a ** 2

        return cls(
            F_obs=F, hkl=hkl_i, s_mag=s_mag, centric=centric, eps=eps,
            E_obs=E_obs, weight=weight, sigma_a=sigma_a, coeff=coeff, fit=fit,
        )


class DirectModelEvaluator:
    """Returns ``F_p1(hkl)`` of a P1-spacegroup model at integer HKL.

    The translation search asks its evaluator for ``F`` at a list of rotated
    Miller indices. The rotation is already baked into the model's coordinates
    by the time this is built, so ``R`` is ignored and every call is a direct
    structure-factor evaluation rather than an interpolation.

    Answers on the **configured default device**, whatever device the model
    happens to sit on. Reading the device off the model instead is how the
    translation stage ends up split across two devices when a caller builds a
    CPU model on a host with an accelerator: the model answers on the CPU while
    everything derived from config answers on the GPU.
    """

    def __init__(self, m: "ModelFT") -> None:
        self._m = m
        self.device = get_default_device()

    def evaluate(self, R, hkl, real_cell, return_amplitude=False):
        hkl_int = hkl.round().to(torch.int64).to(self._m.xyz().device)
        with torch.no_grad():
            f = self._m(hkl_int)
        f = f.to(self.device)
        return f.abs() if return_amplitude else f


@dataclass
class CandidateTransform:
    """One oriented model's transform at the symmetry-rotated indices.

    Attributes
    ----------
    G : torch.Tensor
        ``(S, N)`` complex. ``G_i(h) = F_p1(h R_i) exp(2 pi i h.t_i) / norm(h)``,
        so ``E_calc(h, t) = |sum_i G_i(h) exp(2 pi i (h R_i).t)|`` is the
        **normalised** calculated amplitude: ``<E_calc^2> = 1`` per shell.
    h_R : torch.Tensor
        ``(S, N, 3)`` the rotated indices ``h R_i``.
    norm : torch.Tensor
        ``(N,)`` ``sqrt(eps n_ops Sigma_P(s))``: the raw amplitude is
        ``E_calc * norm``.
    """

    G: torch.Tensor
    h_R: torch.Tensor
    norm: torch.Tensor

    def e_calc(self, t: torch.Tensor) -> torch.Tensor:
        """``E_calc(h, t)`` for ``t`` of shape ``(3,)`` or ``(K, 3)``: ``(N,)`` or ``(K, N)``."""
        single = t.ndim == 1
        tt = t.reshape(-1, 3).to(self.h_R.device).to(self.h_R.dtype)
        phase_arg = torch.einsum("ind,kd->kin", self.h_R, tt)
        phase = torch.exp((2j * math.pi) * phase_arg.to(self.G.dtype))
        E = (self.G.unsqueeze(0) * phase).sum(dim=1).abs()
        return E[0] if single else E


def prepare_candidate(
    evaluator,
    obs: TranslationObs,
    spacegroup,
    real_cell,
) -> CandidateTransform:
    """Evaluate one orientation's transform and normalise it.

    The only per-candidate model evaluation: ``F_p1`` at all ``S x N`` rotated
    indices in one call. The normalising curve ``Sigma_P(s)`` is the same
    Wilson fit the observed side uses, on the same abscissa, fitted to the
    transform's mean intensity over the ``S`` copies -- which is what the crystal
    sum averages to over a shell, since the cross terms between symmetry copies
    have zero mean over ``h``. The crystal's ``<|F_calc|^2>`` is then
    ``eps n_ops Sigma_P``, and dividing by it is what puts every candidate's
    ``E_calc`` on one footing with ``E_obs`` and with each other.
    """
    device = get_default_device()
    real = get_float_dtype()
    cplx = get_complex_dtype()

    hkl = obs.hkl.to(device).to(real)
    sym_R = spacegroup.matrices.detach().to(device).to(real)
    sym_t = spacegroup.translations.detach().to(device).to(real)
    S = int(sym_R.shape[0])
    N = int(hkl.shape[0])

    # h_R[i, n, d] = sum_e hkl[n, e] sym_R[i, e, d]: the h.S convention.
    h_R = torch.einsum("ne,ied->ind", hkl, sym_R)
    phase = torch.exp((2j * math.pi) * torch.einsum("ne,ie->in", hkl, sym_t).to(cplx))
    eye3 = torch.eye(3, dtype=real, device=device)
    F_all = evaluator.evaluate(
        eye3, h_R.reshape(-1, 3), real_cell, return_amplitude=False,
    ).reshape(S, N).to(cplx)
    G_raw = F_all * phase

    I_P = (G_raw.abs() ** 2).mean(dim=0).to(real)
    s_mag = obs.s_mag.to(device).to(real)
    fit_P = WilsonNormaliser(
        I_P, s_mag, n_coeff=WILSON_N_COEFF,
        s_lo=float(s_mag.min()), s_hi=float(s_mag.max()),
    )
    Sigma_c = S * fit_P.evaluate(s_mag).to(real)
    norm = (obs.eps.to(device).to(real) * Sigma_c).clamp(min=1e-30).sqrt()
    return CandidateTransform(G=G_raw / norm.to(cplx), h_R=h_R, norm=norm)


@dataclass
class TranslationPeak:
    """A peak of the fast translation function.

    Attributes
    ----------
    translation : np.ndarray
        Fractional coordinates (3,), refined to sub-grid precision.
    score : float
        The fast search's score at the grid maximum.
    sigma : float
        Standard deviations above the map mean.
    """
    translation: np.ndarray
    score: float
    sigma: float


def _grid_sizes(real_cell, grid_spacing_A: float) -> Tuple[int, int, int]:
    """FFT-friendly grid, at most ``grid_spacing_A`` apart along each axis."""
    sizes = []
    for length in (real_cell.a, real_cell.b, real_cell.c):
        n = int(math.ceil(float(length) / float(grid_spacing_A)))
        n = find_fft_friendly_size(max(n, 4))
        sizes.append(min(n, MAX_GRID_PER_AXIS))
    return sizes[0], sizes[1], sizes[2]


def _parabolic_offset(fm: float, f0: float, fp: float) -> float:
    """Sub-grid offset of a maximum from its three samples, in grid units."""
    denom = fm - 2.0 * f0 + fp
    if denom >= 0.0:
        return 0.0
    return float(min(0.5, max(-0.5, 0.5 * (fm - fp) / denom)))


def _find_peaks(
    score: torch.Tensor,
    n_peaks: int,
    radii_frac: Tuple[float, float, float],
) -> List[TranslationPeak]:
    """Greedy non-maximum suppression on the periodic map, then sub-grid refinement."""
    nx, ny, nz = score.shape
    flat = score.reshape(-1)
    mean = float(flat.mean())
    std = float(flat.std().clamp(min=1e-30))
    n_take = min(flat.numel(), max(50, 20 * n_peaks))
    vals, idx = torch.topk(flat, n_take)
    vals = vals.cpu().numpy()
    idx = idx.cpu().numpy()
    grid = np.array([nx, ny, nz], dtype=np.float64)
    radii = np.asarray(radii_frac, dtype=np.float64)
    score_np = score.cpu().numpy()

    kept: List[TranslationPeak] = []
    kept_t: List[np.ndarray] = []
    for v, i in zip(vals, idx):
        ijk = np.array(np.unravel_index(int(i), (nx, ny, nz)), dtype=np.int64)
        t_grid = ijk / grid
        is_new = True
        for prev in kept_t:
            d = np.abs(t_grid - prev)
            d = np.minimum(d, 1.0 - d)
            if np.all(d < radii):
                is_new = False
                break
        if not is_new:
            continue
        # Parabolic refinement along each axis from the periodic neighbours.
        offs = np.zeros(3)
        for d, n in enumerate((nx, ny, nz)):
            lo = ijk.copy(); lo[d] = (ijk[d] - 1) % n
            hi = ijk.copy(); hi[d] = (ijk[d] + 1) % n
            offs[d] = _parabolic_offset(
                float(score_np[tuple(lo)]), float(v), float(score_np[tuple(hi)]),
            )
        kept.append(TranslationPeak(
            translation=(ijk + offs) / grid, score=float(v),
            sigma=(float(v) - mean) / std,
        ))
        kept_t.append(t_grid)
        if len(kept) >= n_peaks:
            break
    return kept


def fast_translation_function(
    obs: TranslationObs,
    cand: CandidateTransform,
    real_cell,
    *,
    grid_spacing_A: float,
    n_peaks: int = 3,
    cluster_radius_A: float = 4.0,
) -> Tuple[torch.Tensor, List[TranslationPeak]]:
    """The Crowther-Blow map of ``sum_h coeff(h) |E_calc(h, t)|^2`` and its peaks.

    ``coeff`` is :attr:`TranslationObs.coeff` and ``E_calc`` is normalised per
    candidate by :func:`prepare_candidate`, so the map is the covariance of
    two unit-mean intensities weighted by the model's expected reliability --
    the rotation function's own score equation, for translations. Expanding
    ``|sum_i G_i exp(2 pi i (h R_i).t)|^2`` gives pair terms at frequency
    ``h R_j - h R_i``; accumulating them onto a reciprocal grid and inverting
    evaluates every grid translation in one FFT.

    Parameters
    ----------
    grid_spacing_A : float
        Target spacing of the translation grid along each axis. A third of the
        translation set's resolution samples the peak densely enough for the
        parabolic refinement to land within a fraction of a grid step.
    n_peaks : int
        How many distinct peaks to return, best first.
    cluster_radius_A : float
        Peaks closer than this (per axis, periodic) are one peak.

    Returns
    -------
    score : torch.Tensor
        The ``(nx, ny, nz)`` map, fractional grid ``t = (i/nx, j/ny, k/nz)``.
    peaks : list of TranslationPeak
    """
    device = get_default_device()
    real = get_float_dtype()
    cplx = get_complex_dtype()

    nx, ny, nz = _grid_sizes(real_cell, grid_spacing_A)
    G = cand.G.to(device).to(cplx)
    S, N = G.shape
    coeff = obs.coeff.to(device).to(cplx)
    h_R_int = cand.h_R.round().to(torch.int64)

    # The pair (j, i) is the conjugate of (i, j) at -dh, so the map is twice
    # the real part of the upper triangle's transform plus the diagonal, which
    # carries no t and is a constant. Half the scatter, which is the cost here.
    W = torch.zeros(nx * ny * nz, dtype=cplx, device=device)
    for i in range(S - 1):
        pair = G[i].conj().view(1, -1) * G[i + 1:]                  # (S-i-1, N)
        dh = h_R_int[i + 1:] - h_R_int[i:i + 1]                     # (S-i-1, N, 3)
        flat = ((dh[..., 0] % nx) * ny + (dh[..., 1] % ny)) * nz + (dh[..., 2] % nz)
        W.index_add_(0, flat.reshape(-1), (coeff.view(1, -1) * pair).reshape(-1))
    diag = (obs.coeff.to(device).to(real) * (G.abs() ** 2).sum(dim=0).to(real)).sum()
    score = (2.0 * torch.fft.ifftn(W.view(nx, ny, nz), dim=(0, 1, 2)).real
             * float(nx * ny * nz)).to(real) + diag

    radii = tuple(float(cluster_radius_A) / float(L)
                  for L in (real_cell.a, real_cell.b, real_cell.c))
    peaks = _find_peaks(score, n_peaks, radii)
    return score, peaks


def translation_score_at(obs: TranslationObs, cand: CandidateTransform,
                         t: torch.Tensor) -> float:
    """The fast search's score at one translation, without the FFT."""
    E2 = cand.e_calc(t) ** 2
    return float((obs.coeff.to(E2.device).to(E2.dtype) * E2).sum())


def llg_at_translations(
    obs: TranslationObs,
    cand: CandidateTransform,
    t_candidates: torch.Tensor,
) -> torch.Tensor:
    """Rice/Woolfson log-likelihood gain at each of ``K`` translations.

    ``LLG(t) = sum_h [LL(E_obs; sigma_A E_calc(h, t), 1 - sigma_A^2)
    - LL(E_obs; 0, 1)]`` with the complex-variance convention of
    :func:`~torchref.base.targets.xray_likelihoods.rice_per_refl`, which
    derives the centric case from the same ``Sigma``. ``sigma_A`` is the
    Luzzati prior carried by ``obs`` -- the same for every candidate, so the
    values are comparable across orientations as well as across translations,
    and no candidate is scored against a likelihood tuned to itself.

    Returns ``(K,)``.
    """
    E_calc = cand.e_calc(t_candidates)                               # (K, N)
    K, N = E_calc.shape
    dev, real = E_calc.device, E_calc.dtype
    E_obs = obs.E_obs.to(dev).to(real).view(1, N).expand(K, N)
    D = obs.sigma_a.to(dev).to(real).view(1, N)
    Sigma = (1.0 - D * D).clamp(min=1e-3).expand(K, N)
    cent = obs.centric.to(dev).view(1, N).expand(K, N)
    ll = -rice_per_refl(E_obs, D * E_calc, Sigma, cent)              # (K, N)
    ll_wil = -rice_per_refl(
        E_obs[0], torch.zeros(N, dtype=real, device=dev),
        torch.ones(N, dtype=real, device=dev), cent[0],
    ).sum()
    return ll.sum(dim=1) - ll_wil


def analytic_r_at(obs: TranslationObs, cand: CandidateTransform,
                  t: torch.Tensor) -> float:
    """``R = sum ||F_obs| - k |F_calc(t)|| / sum |F_obs|`` with one global scale.

    On raw amplitudes, because that is what a crystallographer reads; not the
    number a full Scaler would return, since there is no bulk solvent and no
    B-factor scaling behind ``k``.
    """
    F_c = cand.e_calc(t) * cand.norm.to(cand.G.device)
    F_o = obs.F_obs.to(F_c.device).to(F_c.dtype)
    k = (F_o * F_c).sum() / (F_c * F_c).sum().clamp(min=1e-30)
    return float((F_o - k * F_c).abs().sum() / F_o.sum().clamp(min=1e-30))
