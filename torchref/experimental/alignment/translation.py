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

#: How far down the translation map the peak finder will look for distinct
#: maxima. Past this many grid values a map has no peaks left worth the
#: likelihood's time -- the half-model site that needed digging was within the
#: first few thousand -- and on a polar group, where every maximum along the
#: polar axis is one site, an uncapped search for 32 distinct sites walked the
#: whole map (1DAW: 4.3 s against 0.8).
MAX_PEAK_POOL = 8192


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

        hkl_l = hkl_i.round().to(torch.int64)  # dtype-ok: Miller indices are integers
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

    def f_calc(self, t: torch.Tensor) -> torch.Tensor:
        """The normalised complex transform ``sum_i G_i exp(2 pi i (h R_i).t)``
        for ``t`` of shape ``(3,)`` or ``(K, 3)``: ``(N,)`` or ``(K, N)`` complex."""
        single = t.ndim == 1
        tt = t.reshape(-1, 3).to(self.h_R.device).to(self.h_R.dtype)
        phase_arg = torch.einsum("ind,kd->kin", self.h_R, tt)
        phase = torch.exp((2j * math.pi) * phase_arg.to(self.G.dtype))
        F = (self.G.unsqueeze(0) * phase).sum(dim=1)
        return F[0] if single else F

    def e_calc(self, t: torch.Tensor) -> torch.Tensor:
        """``|E_calc(h, t)|`` for ``t`` of shape ``(3,)`` or ``(K, 3)``: ``(N,)`` or ``(K, N)``."""
        return self.f_calc(t).abs()


def prepare_candidate(
    model_p1: "ModelFT",
    obs: TranslationObs,
    spacegroup,
    real_cell,
) -> CandidateTransform:
    """Evaluate one orientation's transform and normalise it.

    ``model_p1`` is an ordinary :class:`~torchref.model.ModelFT` in P1 with the
    crystal's cell, already in the candidate orientation; its grid is whatever
    its ``max_res`` implies. The only per-candidate model evaluation: ``F_p1``
    at all ``S x N`` rotated indices in one call. The result is moved to the
    configured default device, whatever device the model sits on, so the
    translation stage never straddles two devices. The normalising curve ``Sigma_P(s)`` is the same
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
    hkl_SN = h_R.reshape(-1, 3).round().to(torch.int64).to(model_p1.xyz().device)  # dtype-ok: Miller indices are integers
    with torch.no_grad():
        F_all = model_p1(hkl_SN).to(device).reshape(S, N).to(cplx)
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


def _figure_of_merit(E_obs, F_mean, V, centric):
    """Rice/Woolfson figure of merit: ``I1(X)/I0(X)`` acentric, ``tanh(X/2)`` centric,
    ``X = 2 E_obs F_mean / V``."""
    X = (2.0 * E_obs * F_mean / V).clamp(max=1e6)
    m_acen = torch.special.i1e(X) / torch.special.i0e(X).clamp(min=1e-30)
    m_cen = torch.tanh(0.5 * X)
    return torch.where(centric, m_cen, m_acen)


@dataclass
class FixedComponent:
    """The chains already placed, as the likelihood sees them.

    Everything here is the Rice/Woolfson likelihood ``rice_per_refl`` evaluated
    around the fixed structure's contribution ``D_f E_f`` with variance
    ``V = 1 - D_f^2``. The moving model is then a perturbation of that point,
    and the two coefficients below are the likelihood's first and second
    derivatives there, so the fast translation map is the second-order
    expansion of the very likelihood the peaks are scored with:

    * ``c_quad = 2 w sigma_A^2 dLL/dSigma`` multiplies ``|E_m(h, t)|^2`` --
      the Crowther-Blow pair terms. With nothing fixed it is bit for bit the
      LERF1 coefficient ``cw (E_obs^2 - 1) w sigma_A^2`` the rotation function
      expands; with a fixed part it also removes the fixed-moving cross term
      from the Patterson, which is why it is what the difference rotation
      function uses too.
    * ``c_lin = 2 w sigma_A dLL/d|F_c| exp(i phi_f)`` multiplies
      ``Re(conj(c_lin) E_m(h, t))`` -- the phased translation function, Read's
      figure-of-merit-weighted difference-map coefficient. Linear in the
      moving model, so a small fragment keeps its contrast.

    Attributes
    ----------
    E_f : torch.Tensor
        ``(N,)`` complex, ``F_f / sqrt(eps Sigma_f)`` with ``Sigma_f`` the fixed
        structure's own Wilson fit on the observed side's abscissa.
    D_f, V, m : torch.Tensor
        ``(N,)`` real: the fixed part's reliability, the conditional variance
        and the figure of merit. ``D_f`` is the Luzzati prior for the fixed
        chains' coordinate error times one scale fitted to the data: a placed
        half of the asymmetric unit accounts for half the scattering, and the
        complete-model prior would call every placement a mismatch.
    c_quad, c_lin : torch.Tensor
        ``(N,)`` real and complex, above.
    F_f_raw : torch.Tensor
        ``(N,)`` complex, the fixed structure factors on the model's scale, for
        the analytical R.
    ll_ref : torch.Tensor
        The fixed-only log-likelihood, the reference the gain is measured from.
    """

    E_f: torch.Tensor
    D_f: torch.Tensor
    V: torch.Tensor
    m: torch.Tensor
    c_quad: torch.Tensor
    c_lin: torch.Tensor
    F_f_raw: torch.Tensor
    ll_ref: torch.Tensor

    #: Ceiling on the fixed part's D: V = 1 - D^2 floors at about 0.1, which
    #: keeps 1/V^2 in the coefficients from amplifying a well-fitted fixed part
    #: into a weight nothing else can compete with.
    D_CAP = 0.95

    @classmethod
    def build(
        cls,
        obs: TranslationObs,
        F_fixed: torch.Tensor,
        *,
        err_fixed_A: float,
        scales: Tuple[float, ...] = None,
    ) -> "FixedComponent":
        device = get_default_device()
        real = get_float_dtype()
        cplx = get_complex_dtype()
        F_f = F_fixed.detach().to(device).to(cplx)
        s_mag = obs.s_mag.to(device).to(real)
        eps = obs.eps.to(device).to(real)
        cent = obs.centric.to(device)
        E_obs = obs.E_obs.to(device).to(real)
        w = obs.weight.to(device).to(real)
        sig_a = obs.sigma_a.to(device).to(real)

        I_f = (F_f.abs() ** 2).to(real)
        if float(I_f.max()) <= 0.0:
            # Nothing fixed after all: the coefficients must be the LERF1 ones.
            E_f = torch.zeros_like(F_f)
        else:
            fit_f = WilsonNormaliser(
                I_f, s_mag, eps=eps, centric=cent,
                n_coeff=WILSON_N_COEFF, s_lo=float(s_mag.min()), s_hi=float(s_mag.max()),
            )
            E_f = F_f / (eps * fit_f.evaluate(s_mag).to(real)).clamp(min=1e-30).sqrt().to(cplx)
        E_f_abs = E_f.abs()

        # One scale on the Luzzati prior for the fixed chains, fitted to the
        # data: a profile over the same grid the moving part's likelihood uses.
        prior = eterm_sigma_a(s_mag, float(err_fixed_A)).to(real)
        if float(E_f_abs.max()) <= 0.0:
            # A fixed part that explains nothing has no reliability either:
            # V = 1 and the coefficients are the LERF1 ones exactly.
            D_f = torch.zeros_like(prior)
        else:
            grid = SIGMA_A_SCALES if scales is None else scales
            best_c, best_ll = 1.0, -float("inf")
            for c in grid:
                D = (prior * float(c)).clamp(max=cls.D_CAP)
                ll = float(-rice_per_refl(E_obs, D * E_f_abs, (1.0 - D * D), cent).sum())
                if ll > best_ll:
                    best_c, best_ll = float(c), ll
            D_f = (prior * best_c).clamp(max=cls.D_CAP)
        V = 1.0 - D_f * D_f
        F_mean = D_f * E_f_abs
        m = _figure_of_merit(E_obs, F_mean, V, cent)

        # dLL/dSigma at (F_mean, V): acentric [E^2 + F^2 - 2 m E F - V] / V^2,
        # centric half of it -- see _rice_body. Twice the derivative is the
        # LERF1 convention: with F = 0, V = 1 it is cw (E^2 - 1), cw = 2 and 1.
        dll_dsigma = (E_obs ** 2 + F_mean ** 2 - 2.0 * m * E_obs * F_mean - V) / (V * V)
        dll_dsigma = torch.where(cent, 0.5 * dll_dsigma, dll_dsigma)
        # dLL/d|F_c| at the same point: acentric 2 (m E - F) / V, centric (m E - F) / V.
        dll_dfc = (m * E_obs - F_mean) / V
        dll_dfc = torch.where(cent, dll_dfc, 2.0 * dll_dfc)
        phase_f = torch.exp(1j * torch.angle(F_f).to(real)).to(cplx)

        c_quad = 2.0 * w * sig_a ** 2 * dll_dsigma
        c_lin = (2.0 * w * sig_a * dll_dfc).to(cplx) * phase_f
        ll_ref = -rice_per_refl(E_obs, F_mean, V, cent).sum()
        return cls(E_f=E_f, D_f=D_f, V=V, m=m, c_quad=c_quad, c_lin=c_lin,
                   F_f_raw=F_f, ll_ref=ll_ref)


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
    shifts: np.ndarray,
    polar: np.ndarray,
) -> List[TranslationPeak]:
    """Greedy non-maximum suppression on the periodic map, then sub-grid refinement.

    Two maxima are one peak when they coincide modulo the lattice, an allowed
    origin shift, or any displacement along a polar axis -- those are the same
    molecular-replacement solution, and returning them as the "top three" left
    the likelihood nothing to choose between. In P2(1) that was every peak the
    search returned: the same site shifted along ``b`` and by ``(0, 0, 1/2)``.

    The pool of candidate maxima grows until ``n_peaks`` distinct ones are found
    or ``MAX_PEAK_POOL`` values have been examined, so fewer than ``n_peaks``
    may come back. A fixed pool of a few dozen values is not enough on a weak
    map, where they all belong to one broad maximum: a search asked for three
    peaks returned one, 7 A from a true site that out-scored it.
    """
    nx, ny, nz = score.shape
    flat = score.reshape(-1)
    mean = float(flat.mean())
    std = float(flat.std().clamp(min=1e-30))
    grid = np.array([nx, ny, nz], dtype=np.float64)
    radii = np.asarray(radii_frac, dtype=np.float64)
    score_np = score.cpu().numpy()
    n_total = flat.numel()

    def same_site(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Which of ``pos`` (P, 3) coincide with ``t`` modulo the origin freedom."""
        d = (pos - t)[:, None, :] - shifts[None, :, :]                 # (P, m, 3)
        d = d - np.round(d)
        if polar.shape[1]:
            d = d - (d @ polar) @ polar.T
        return np.any(np.all(np.abs(d) < radii, axis=2), axis=1)

    kept: List[TranslationPeak] = []
    limit = min(n_total, MAX_PEAK_POOL)
    pool = min(limit, max(64, 20 * n_peaks))
    while True:
        idx = torch.topk(flat, pool).indices.cpu().numpy()
        vals = flat[torch.as_tensor(idx, device=flat.device)].cpu().numpy()
        ijk_all = np.stack(np.unravel_index(idx, (nx, ny, nz)), axis=1).astype(np.int64)
        pos_all = ijk_all / grid
        alive = np.ones(pool, dtype=bool)
        kept = []
        while len(kept) < n_peaks:
            live = np.flatnonzero(alive)
            if live.size == 0:
                break
            i = int(live[0])
            ijk, v = ijk_all[i], float(vals[i])
            alive &= ~same_site(pos_all, pos_all[i])
            # Parabolic refinement along each axis from the periodic neighbours.
            offs = np.zeros(3)
            for dim, n in enumerate((nx, ny, nz)):
                lo = ijk.copy(); lo[dim] = (ijk[dim] - 1) % n
                hi = ijk.copy(); hi[dim] = (ijk[dim] + 1) % n
                offs[dim] = _parabolic_offset(
                    float(score_np[tuple(lo)]), v, float(score_np[tuple(hi)]),
                )
            kept.append(TranslationPeak(
                translation=(ijk + offs) / grid, score=v, sigma=(v - mean) / std,
            ))
        if len(kept) >= n_peaks or pool >= limit:
            return kept
        pool = min(limit, pool * 8)


def fast_translation_function(
    obs: TranslationObs,
    cand: CandidateTransform,
    spacegroup,
    real_cell,
    *,
    grid_spacing_A: float,
    n_peaks: int = 3,
    cluster_radius_A: float = 4.0,
    fixed: Optional[FixedComponent] = None,
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
        How many distinct peaks to return, best first. Distinct modulo the
        group's origin freedom: an allowed origin shift or a displacement along
        a polar axis does not make a new peak.
    cluster_radius_A : float
        Peaks closer than this (per axis, periodic) are one peak.
    fixed : FixedComponent, optional
        Chains already placed. The pair terms then take ``fixed.c_quad`` for
        their coefficient and the map gains the phased term
        ``Re(conj(c_lin) E_m(h, t))``, one more scatter per operation on the
        same grid; and the origin freedom is gone, so peaks are distinct only
        modulo the lattice.

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
    coeff_real = (obs.coeff if fixed is None else fixed.c_quad).to(device).to(real)
    coeff = coeff_real.to(cplx)
    h_R_int = cand.h_R.round().to(torch.int64)  # dtype-ok: Miller indices are integers

    # The pair (j, i) is the conjugate of (i, j) at -dh, so the map is twice
    # the real part of the upper triangle's transform plus the diagonal, which
    # carries no t and is a constant. Half the scatter, which is the cost here.
    W = torch.zeros(nx * ny * nz, dtype=cplx, device=device)
    for i in range(S - 1):
        pair = G[i].conj().view(1, -1) * G[i + 1:]                  # (S-i-1, N)
        dh = h_R_int[i + 1:] - h_R_int[i:i + 1]                     # (S-i-1, N, 3)
        flat = ((dh[..., 0] % nx) * ny + (dh[..., 1] % ny)) * nz + (dh[..., 2] % nz)
        W.index_add_(0, flat.reshape(-1), (coeff.view(1, -1) * pair).reshape(-1))
    if fixed is not None:
        # The phased term Re(conj(c_lin) sum_i G_i e^{2 pi i (h R_i).t}) lands at
        # frequency h R_i; half of it, since the 2 Re below doubles it.
        half_lin = (0.5 * fixed.c_lin.conj().to(device).to(cplx)).view(1, -1) * G   # (S, N)
        flat_i = ((h_R_int[..., 0] % nx) * ny + (h_R_int[..., 1] % ny)) * nz + (h_R_int[..., 2] % nz)
        W.index_add_(0, flat_i.reshape(-1), half_lin.reshape(-1))
    diag = (coeff_real * (G.abs() ** 2).sum(dim=0).to(real)).sum()
    score = (2.0 * torch.fft.ifftn(W.view(nx, ny, nz), dim=(0, 1, 2)).real
             * float(nx * ny * nz)).to(real) + diag

    radii = tuple(float(cluster_radius_A) / float(L)
                  for L in (real_cell.a, real_cell.b, real_cell.c))
    if fixed is None:
        shifts, polar = spacegroup.origin_shifts()
        shifts, polar = shifts.numpy(), polar.numpy()
    else:
        # A fixed component pins the origin: only the lattice remains.
        shifts, polar = np.zeros((1, 3)), np.zeros((3, 0))
    peaks = _find_peaks(score, n_peaks, radii, shifts, polar)
    return score, peaks


def translation_score_at(obs: TranslationObs, cand: CandidateTransform,
                         t: torch.Tensor, fixed: Optional[FixedComponent] = None) -> float:
    """The fast search's score at one translation, without the FFT.

    ``sum_h [Re(conj(c_lin) E_m) + c_quad |E_m|^2]`` with a fixed component,
    ``sum_h coeff |E_m|^2`` without; the same functional the map evaluates on
    its grid, up to the map's t-independent constant.
    """
    F = cand.f_calc(t)
    E2 = F.abs() ** 2
    if fixed is None:
        return float((obs.coeff.to(E2.device).to(E2.dtype) * E2).sum())
    quad = (fixed.c_quad.to(E2.device).to(E2.dtype) * E2).sum()
    lin = (fixed.c_lin.to(F.device).conj() * F).real.sum()
    return float(quad + lin)


#: Multipliers on the Luzzati ``sigma_A`` the likelihood may choose from, per
#: translation. The prior assumes a complete model; a search model that is half
#: the asymmetric unit accounts for roughly half the scattering, and against
#: the full prior every placement scores as a gross mismatch. Letting each
#: placement take the scale that explains it best is what Phaser's per-solution
#: sigma_A refinement does; here it is a profile over a grid, one (K, N) Rice
#: evaluation per point.
SIGMA_A_SCALES = tuple(float(c) for c in np.linspace(0.1, 1.0, 19))


def llg_at_translations(
    obs: TranslationObs,
    cand: CandidateTransform,
    t_candidates: torch.Tensor,
    *,
    scales: Tuple[float, ...] = SIGMA_A_SCALES,
    fixed: Optional[FixedComponent] = None,
) -> torch.Tensor:
    """Rice/Woolfson log-likelihood gain at each of ``K`` translations.

    ``LLG(t) = max_c sum_h [LL(E_obs; c sigma_A E_calc(h, t), 1 - c^2 sigma_A^2)
    - LL(E_obs; 0, 1)]`` with the complex-variance convention of
    :func:`~torchref.base.targets.xray_likelihoods.rice_per_refl`, which
    derives the centric case from the same ``Sigma``. ``sigma_A`` is the
    Luzzati prior carried by ``obs``; ``c`` is the placement's own scale on it,
    chosen from ``scales``.

    The maximum over ``c`` is what makes the values comparable across
    placements of a model that does not account for all the scattering. With
    the scale fixed at one, a 48% model's true site scored -15900 and a site
    14 A away -17900: both gross mismatches, ordered by noise. With the scale
    free the true site takes ``c`` near its completeness and scores positive,
    while a wrong site takes the smallest ``c`` and scores near zero -- the
    Wilson reference, which is exactly what a placement that explains nothing
    should score. For a complete model the maximum sits at ``c = 1`` and
    nothing changes.

    With ``fixed``, the mean is ``D_f E_f + c sigma_A E_m(t)`` as a complex
    sum -- the moving model's phase relative to the fixed structure is what a
    placement determines -- with variance ``1 - D_f^2 - c^2 sigma_A^2``, and
    the gain is measured from the fixed-only likelihood rather than Wilson's.

    Returns ``(K,)``.
    """
    F_calc = cand.f_calc(t_candidates)                               # (K, N) complex
    K, N = F_calc.shape
    dev = F_calc.device
    real = F_calc.real.dtype
    E_obs = obs.E_obs.to(dev).to(real).view(1, N).expand(K, N)
    D0 = obs.sigma_a.to(dev).to(real).view(1, N)
    cent = obs.centric.to(dev).view(1, N).expand(K, N)
    if fixed is None:
        E_calc = F_calc.abs()
        ll_ref = -rice_per_refl(
            E_obs[0], torch.zeros(N, dtype=real, device=dev),
            torch.ones(N, dtype=real, device=dev), cent[0],
        ).sum()
        base_mean = None
        var0 = torch.ones(1, N, dtype=real, device=dev)
    else:
        E_calc = None
        ll_ref = fixed.ll_ref.to(real)
        base_mean = (fixed.D_f.to(real) * fixed.E_f).to(F_calc.dtype).view(1, N)
        var0 = fixed.V.to(dev).to(real).view(1, N)
    best = torch.full((K,), -float("inf"), dtype=real, device=dev)
    for c in scales:
        D = D0 * float(c)
        Sigma = (var0 - D * D).clamp(min=1e-3).expand(K, N)
        if base_mean is None:
            mean = D * E_calc
        else:
            mean = (base_mean + D.to(F_calc.dtype) * F_calc).abs()
        ll = -rice_per_refl(E_obs, mean, Sigma, cent).sum(dim=1) - ll_ref
        best = torch.maximum(best, ll)
    return best


def analytic_r_at(obs: TranslationObs, cand: CandidateTransform,
                  t: torch.Tensor, fixed: Optional[FixedComponent] = None) -> float:
    """``R = sum ||F_obs| - k |F_calc(t)|| / sum |F_obs|`` with one global scale.

    On raw amplitudes, because that is what a crystallographer reads; not the
    number a full Scaler would return, since there is no bulk solvent and no
    B-factor scaling behind ``k``.
    """
    F_m = cand.f_calc(t) * cand.norm.to(cand.G.device).to(cand.G.dtype)
    if fixed is not None:
        F_m = F_m + fixed.F_f_raw.to(F_m.device).to(F_m.dtype)
    F_c = F_m.abs()
    F_o = obs.F_obs.to(F_c.device).to(F_c.dtype)
    k = (F_o * F_c).sum() / (F_c * F_c).sum().clamp(min=1e-30)
    return float((F_o - k * F_c).abs().sum() / F_o.sum().clamp(min=1e-30))
