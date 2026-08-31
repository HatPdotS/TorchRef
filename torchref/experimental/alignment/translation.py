"""Fast translation search: where in the cell does an oriented model sit?

A translation shifts phase, ``F(h, t) = F(h) exp(2 pi i h.t)``, so scoring every
``t`` on a grid is a Fourier transform rather than a scan. The Crowther-Blow
form used here accumulates the pair coefficients
``sum_h w(h) G_i*(h) G_j(h)`` onto a reciprocal grid at
``(h R_j - h R_i) mod G`` and takes one inverse FFT, which replaces ``G^3`` grid
evaluations with a single transform.

This is where the discrimination happens. The rotation function upstream is a
shortlist generator -- over 30 seeded cells it puts truth at rank 0 six times;
the correlation here does it 24 times and the likelihood 27. Rotation ghosts are
morphologically identical to truth in a Patterson by construction, and stop
being identical as soon as the crystal lattice is involved.

The observed side is prepared **once** per run, by
:class:`TranslationObs`, and reused for every orientation and every candidate
translation. That is not only an optimisation: normalisation and weighting are
properties of the observations, which do not change when the model moves, and
three separate answers to "what is the mean intensity here" used to live in this
module and its caller.
"""

import numpy as np
import torch

from torchref.scaling import WilsonNormaliser
from torchref.scaling.weighting import (inverse_variance_weight,
                                        normalise_weight, snr_from_amplitude)

from .distributions import rice_log_likelihood, woolfson_log_likelihood
from .sh import assign_shells, equal_count_shell_edges
from dataclasses import dataclass
from typing import List, Optional, Tuple


#: Chebyshev order of the Wilson fit. Matches the rotation function's
#: ``frf.api.WILSON_N_COEFF``: the two stages score the same observations and a
#: different order on each would be two normalisations again.
WILSON_N_COEFF = 6


@dataclass
class TranslationObs:
    """The observed side of a translation search, normalised and weighted once.

    Everything here is a property of the observations alone, so none of it
    changes when the model rotates or moves. Building it per orientation -- which
    is what the module used to do -- refits a Gamma GLM for every candidate to
    get the same answer back, and worse, it made "what is ``E_obs``" a question
    with three different answers depending on which function you asked.

    Attributes
    ----------
    F_obs, hkl, s_mag, centric, eps
        The masked observations and their crystallographic bookkeeping.
    E_obs : torch.Tensor
        ``F / sqrt(eps Sigma(s))``, with ``Sigma`` the shared Wilson fit, so
        ``<E^2> = 1`` as an identity of that fit rather than as a separate
        normalisation step.
    weight : torch.Tensor
        Mean-1 inverse-variance weight, from measurement error and model error
        in one denominator. **This is the half that does not cancel.** A
        per-resolution *scaling* is gauge in a correlation -- twelve conventions
        moved the rotation function's truth rank by nothing -- but a weight that
        varies within a shell is not, and until now the translation search had
        none at all: every reflection counted the same.
    shell_idx, n_shells
        Equal-count binning in ``|s|``, shared by the sigma_A fit and the
        likelihood so the two cannot disagree about which reflection is where.
    fit : WilsonNormaliser
        Kept, not discarded. Anything comparing an observed curve against a
        calculated one needs the curve itself, not the per-reflection values it
        produced.
    """

    F_obs: torch.Tensor
    hkl: torch.Tensor
    s_mag: torch.Tensor
    centric: torch.Tensor
    eps: torch.Tensor
    E_obs: torch.Tensor
    weight: torch.Tensor
    shell_idx: torch.Tensor
    n_shells: int
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
        n_shells: int = 20,
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
            half of the variance budget through the Luzzati falloff. The same
            number the rotation function weights with.
        """
        dev = device if device is not None else F_obs.device
        real = torch.float64
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

        if sig_F is None:
            weight = torch.ones_like(F)
        else:
            sig = sig_F.detach().to(dev).to(real).abs()
            # eterm_sigma_a is the rotation function's own model-error term;
            # importing it rather than restating the exponent is the point.
            from .frf.preprocessing import eterm_sigma_a
            weight = normalise_weight(inverse_variance_weight(
                snr_from_amplitude(F, sig),
                eterm_sigma_a(s_mag, float(delta_vrms_A)).to(real),
                eps=eps,
            ))

        edges, _ = equal_count_shell_edges(s_mag, n_shells)
        shell_idx = assign_shells(s_mag, edges).clamp(min=0)

        return cls(
            F_obs=F, hkl=hkl_i, s_mag=s_mag, centric=centric, eps=eps,
            E_obs=fit.E.to(real), weight=weight,
            shell_idx=shell_idx, n_shells=int(n_shells), fit=fit,
        )


@dataclass
class TranslationPeak:
    """
    Translation search peak.

    Attributes
    ----------
    translation : np.ndarray
        Fractional coordinates (3,).
    score : float
        Correlation score.
    sigma : float
        Z-score above mean.
    """
    translation: np.ndarray
    score: float
    sigma: float


def find_translation_peaks(
    correlation_map: np.ndarray,
    n_peaks: int = 10,
    cluster_radius: float = 0.05,
) -> List[TranslationPeak]:
    """
    Extract and cluster peaks from translation function.

    Parameters
    ----------
    correlation_map : np.ndarray
        Translation function values, shape (Nx, Ny, Nz).
    n_peaks : int
        Maximum number of peaks to return.
    cluster_radius : float
        Minimum fractional distance between peaks (periodic).

    Returns
    -------
    peaks : list
        List of TranslationPeak objects sorted by score.
    """
    Nx, Ny, Nz = correlation_map.shape
    mean_val = correlation_map.mean()
    std_val = correlation_map.std()

    if std_val < 1e-10:
        return []

    flat = correlation_map.flatten()
    sorted_idx = np.argsort(flat)[::-1]

    peaks = []
    used = []

    for idx in sorted_idx:
        if len(peaks) >= n_peaks:
            break

        pos_3d = np.unravel_index(idx, correlation_map.shape)
        trans = np.array([pos_3d[0] / Nx, pos_3d[1] / Ny, pos_3d[2] / Nz])
        score = flat[idx]
        sigma = (score - mean_val) / std_val

        # Check clustering - skip if too close to existing peak
        is_new = True
        for prev in used:
            diff = np.abs(trans - prev)
            diff = np.minimum(diff, 1 - diff)  # Periodic boundary
            if np.linalg.norm(diff) < cluster_radius:
                is_new = False
                break

        if is_new:
            peaks.append(TranslationPeak(trans, score, sigma))
            used.append(trans)

    return peaks


def amplitude_translation_search(
    obs: TranslationObs,
    interpolator,
    R_rotation: torch.Tensor,
    spacegroup,
    real_cell,
    grid_steps: int = 16,
    n_peaks: int = 20,
    cluster_radius: float = 0.05,
    batch_size: int = 256,
    precomputed_G: Optional[torch.Tensor] = None,
    precomputed_h_R: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray, List[TranslationPeak]]:
    """
    Coarse-grid translation search via |F|²-correlation.

    For each candidate fractional translation `t` on a `grid_steps`³ grid in
    `[0, 1)³`, scores the model at the current rotation translated by `t`
    against the observed amplitudes by Pearson correlation of `|F_obs|²` and
    `|F_calc(h, t)|²`. The structure-factor sum uses the spacegroup symmetry
    expansion

        F_calc(h, t) = Σ_i G_i(h) · exp(2πi (h R_i) · t)
        G_i(h) = exp(2πi h · t_i) · F_p1(h R_i)

    with `F_p1(h R_i)` looked up via the supplied interpolator at the rotation
    already applied to the model. The `G_i` factors are computed once; only the
    phase exponential changes per candidate, so the scan is efficient.

    Parameters
    ----------
    obs : TranslationObs
        The observations, normalised and weighted once for the whole run.
    interpolator : object
        Anything providing ``evaluate(R, hkl, real_cell, return_amplitude=False)``
        -- in the pipeline, ``align._DirectModelEvaluator``.
    R_rotation : torch.Tensor, shape (3, 3)
        Rotation that has been applied to the model coordinates.
    spacegroup : SpaceGroup
        Provides `matrices` and `translations`.
    real_cell : Cell
        Real crystal cell.
    grid_steps : int, default 16
        Per-axis grid resolution. Total candidates = grid_steps³.
    n_peaks : int, default 20
        Number of peaks returned (after clustering).
    cluster_radius : float, default 0.05
        Minimum fractional separation between returned peaks.
    batch_size : int, default 256
        Number of candidate translations evaluated per inner batch.

    Returns
    -------
    correlation_map : np.ndarray, shape (grid_steps, grid_steps, grid_steps)
        Pearson correlation of |F_obs|² and |F_calc(t)|² at each grid point.
    best_translation : np.ndarray, shape (3,)
        Top-scoring fractional translation.
    peaks : list of TranslationPeak
        Top-`n_peaks` peaks sorted by descending correlation.
    """
    device = getattr(interpolator, "device", obs.hkl.device)
    real_dtype = torch.float64
    complex_dtype = torch.complex128

    hkl = obs.hkl
    E_obs = obs.E_obs.to(device).to(real_dtype)
    w = obs.weight.to(device).to(real_dtype)

    # Correlating E^2 rather than F^2 is what makes this robust to the
    # resolution envelope: the model has no bulk solvent and the wrong overall
    # B, and that mismatch is multiplicative, so subtracting a mean does not
    # remove it but dividing by Sigma(s) does.
    F_obs2 = E_obs * E_obs
    # Centred at the WEIGHTED mean, which is what the weighted correlation
    # below is a numerator for. With uniform weight this is the plain mean.
    F_obs2_centered = F_obs2 - (w * F_obs2).sum() / w.sum().clamp(min=1e-30)

    # Pre-compute G_i(h) = exp(2πi h·t_i) · F_p1(h R_i)  (or reuse caller's)
    two_pi_i = 2j * torch.pi
    if precomputed_G is not None and precomputed_h_R is not None:
        G = precomputed_G.to(device).to(complex_dtype)
        h_R = precomputed_h_R.to(device).to(real_dtype)
    else:
        G, h_R = precompute_G_for_rotation(
            interpolator, R_rotation, hkl, spacegroup, real_cell, device=device,
        )

    # Crowther–Blow FFT translation function (Acta Cryst. B23 (1967) 544).
    # The grid-evaluated score
    #     num(t) = Σ_h F_obs²_centered(h) · |F_calc(h, t)|²
    # expands as
    #     num(t) = Σ_{i,j} [Σ_h F_obs²_c(h) · G_i*(h) · G_j(h)]
    #                       · exp(2πi · (h·R_j − h·R_i) · t)
    # and on a regular fractional t-grid t = (jx, jy, jz) / G this is exactly
    # an inverse DFT of the bracketed coefficients accumulated onto a 3-D
    # reciprocal grid at integer indices (h·R_j − h·R_i) mod G.
    #
    # We accumulate two such reciprocal grids in one sym-op pass:
    #   W_num : weight per h = w(h)·E_obs²_centered(h) → num(t)
    #   W_den : weight per h = w(h)                    → Σ_h w|F_calc(h,t)|²
    # Score(t) = num(t) / Σ_h|F_calc(h,t)|²  — a per-t scale-normalised
    # Pearson proxy (Phaser's TF uses the full Pearson denominator; ours
    # uses the same scaling that the previous separable-phase code applied
    # via explicit per-t centering, achieved here without materialising
    # |F_calc(h,t)|² per t-point).
    #
    # One IFFT pair replaces G³ grid evaluations — for our defaults this is
    # ~5000× less arithmetic than the separable-phase scoring it supersedes,
    # and orders of magnitude less than the original explicit grid loop.
    S_eff, N_eff = G.shape
    h_R_int = h_R.round().to(torch.int64)                        # (S, N, 3)
    # Both grids carry the same per-reflection weight, so the ratio below is a
    # weighted correlation rather than an unweighted one with a weighted
    # numerator. Uniform w reproduces the previous scores exactly.
    F_obs2_c_complex = (w * F_obs2_centered).to(complex_dtype)   # (N,)
    w_complex = w.to(complex_dtype)                              # (N,)

    W_num_flat = torch.zeros(
        grid_steps ** 3, dtype=complex_dtype, device=device,
    )
    W_den_flat = torch.zeros(
        grid_steps ** 3, dtype=complex_dtype, device=device,
    )
    G_stride_xy = grid_steps * grid_steps
    for i in range(S_eff):
        Gi_conj = G[i].conj()                                    # (N,)
        pair = Gi_conj.view(1, -1) * G                           # (S, N)
        coeff_num = F_obs2_c_complex.view(1, -1) * pair          # (S, N)
        coeff_den = w_complex.view(1, -1) * pair                 # (S, N)
        dh = (h_R_int - h_R_int[i:i + 1]) % grid_steps           # (S, N, 3)
        flat = (dh[..., 0] * G_stride_xy
                + dh[..., 1] * grid_steps + dh[..., 2])          # (S, N)
        flat_flat = flat.reshape(-1)
        W_num_flat.index_add_(0, flat_flat, coeff_num.reshape(-1))
        W_den_flat.index_add_(0, flat_flat, coeff_den.reshape(-1))

    W_num = W_num_flat.view(grid_steps, grid_steps, grid_steps)
    W_den = W_den_flat.view(grid_steps, grid_steps, grid_steps)
    # IFFT scales by 1/G³; undo so values are raw integrals.
    num_t = (torch.fft.ifftn(W_num, dim=(0, 1, 2)).real
             * (grid_steps ** 3)).to(real_dtype)
    den_t = (torch.fft.ifftn(W_den, dim=(0, 1, 2)).real
             * (grid_steps ** 3)).to(real_dtype)
    corr_map = num_t / den_t.clamp(min=1e-30)
    corr_map_np = corr_map.detach().cpu().numpy().astype(np.float32)
    peaks = find_translation_peaks(corr_map_np, n_peaks=n_peaks,
                                    cluster_radius=cluster_radius)
    best = peaks[0].translation if peaks else np.zeros(3)
    return corr_map_np, best, peaks


def fit_sigma_a_per_shell(
    E_obs: torch.Tensor,
    E_calc: torch.Tensor,
    centric: torch.Tensor,
    shell_idx: torch.Tensor,
    n_shells: int,
    n_grid: int = 81,
    interp_var: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-shell sigma_A = D, fitted by a grid scan of the shell likelihood.

    Scans ``D`` in ``[0, 0.99]`` and returns the per-shell maximum. At
    ``n_grid=81`` the resolution is ~0.012, which is finer than the difference
    between adjacent shells on any real falloff.

    This is the *fitted* half of a pair. :func:`torchref.scaling.weighting.empirical_sigma_a`
    is the other: it measures model reliability from the ratio of two Wilson
    curves and works **before** the molecule is placed, which is what the
    rotation function needs. Once a translation exists the residual is
    per-reflection rather than per-shell-average, so a direct fit against the
    placed model is available and strictly better informed. The two answer the
    same question at two different points in the search.

    Returns
    -------
    sigma_a : torch.Tensor, shape (n_shells,)
    """
    device = E_obs.device
    dtype = E_obs.dtype
    N = E_obs.numel()
    D_grid = torch.linspace(0.0, 0.99, n_grid, device=device, dtype=dtype)  # (G,)
    F_mean = D_grid.view(-1, 1) * E_calc.view(1, -1)                        # (G, N)
    var_d = (1.0 - D_grid * D_grid).clamp(min=1e-4)                         # (G,)
    if interp_var is None:
        var_full = var_d.view(-1, 1).expand(n_grid, N)
    else:
        var_full = (var_d.view(-1, 1) + interp_var.view(1, -1)).clamp(min=1e-4)
    E_obs_full = E_obs.view(1, -1).expand(n_grid, N)

    ll_acent = rice_log_likelihood(E_obs_full, F_mean, var_full)
    ll_cent = woolfson_log_likelihood(E_obs_full, F_mean, var_full)
    cent_full = centric.view(1, -1)
    ll = torch.where(cent_full, ll_cent, ll_acent)                          # (G, N)

    # Sum per shell, take argmax over the D-grid.
    shell_idx_gn = shell_idx.view(1, -1).expand(n_grid, N)
    ll_per_shell = torch.zeros((n_grid, n_shells), dtype=dtype, device=device)
    ll_per_shell.scatter_add_(1, shell_idx_gn, ll)                          # (G, n_shells)
    best_idx = ll_per_shell.argmax(dim=0)                                   # (n_shells,)
    return D_grid[best_idx]


def llg_translation_rescore(
    obs: TranslationObs,
    G: torch.Tensor,
    h_R: torch.Tensor,
    t_candidates: torch.Tensor,
    sigma_a: torch.Tensor,
    interp_var: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-translation Rice / Woolfson log-likelihood over candidate positions.

    For each candidate t::

        F_calc(h, t) = sum_i G_i(h) exp(2 pi i (h R_i).t)
        E_calc(h, t) = |F_calc(h, t)| / sqrt(<F^2>_shell)
        LLG(t)       = sum_h [LL(E_obs, D E_calc, var) - LL_Wilson(E_obs)]

    with ``var = (1 - D^2) + interp_var``. The Rice branch is used for acentric
    reflections and Woolfson for centric.

    The scoring rule the amplitude correlation is a pre-filter for. At rank
    level it is the strongest discriminator measured -- truth at rank 0 in 27 of
    30 seeded cells against the correlation's 24 and the rotation function's 6 --
    but re-ranking translation peaks by it does **not** improve end-to-end pose
    recovery (28/30 against 27/30 the other way, one discordant cell), which is
    why ``use_llg_tf`` defaults off.

    ``E_calc`` is normalised per shell **per candidate**, and that is not a
    fourth answer to what the observations' normalisation is: it is a per-``t``
    scale, and forcing it to unit shell variance for every candidate is what
    makes the K likelihoods comparable. What discriminates is the pattern across
    reflections, not the scale.

    Parameters
    ----------
    obs : TranslationObs
        Supplies ``E_obs``, centricity and the shell binning -- the same binning
        ``sigma_a`` was fitted on, which is why it is not re-derived here.
    G : (S, N) complex
        Per-sym ``F_p1`` contributions x per-sym translation phase, from
        :func:`precompute_G_for_rotation`.
    h_R : (S, N, 3)
        Per-sym rotated reciprocal indices.
    t_candidates : (K, 3)
        Fractional translations to score.
    sigma_a : (n_shells,)
        Fixed per-shell sigma_A, shared across candidates.
    interp_var : (N,), optional
        Per-reflection variance inflation.

    Returns
    -------
    llg : (K,) torch.Tensor — log-likelihood gain per candidate.
    """
    device = G.device
    real_dtype = torch.float64
    complex_dtype = G.dtype

    shell_idx = obs.shell_idx
    n_shells = obs.n_shells
    centric = obs.centric

    K = t_candidates.shape[0]
    S, N = G.shape

    t_cand = t_candidates.to(device).to(real_dtype)               # (K, 3)
    # Phase factor for each (k, i, n): exp(2πi · (h_R[i, n] · t[k]))
    phase_arg = torch.einsum("ind,kd->kin", h_R.to(real_dtype), t_cand)
    phase = torch.exp(2j * torch.pi * phase_arg.to(complex_dtype))  # (K, S, N)
    # F_calc(k, n) = Σ_i G[i, n] · phase[k, i, n]
    Fc_complex = (G.view(1, S, N) * phase).sum(dim=1)             # (K, N)
    F_calc = Fc_complex.abs().to(real_dtype)                       # (K, N)

    # Per-shell E normalisation of F_calc across the K-batch.
    shell_idx_l = shell_idx.to(device).long()
    cnt = torch.bincount(shell_idx_l, minlength=n_shells).to(real_dtype)
    shell_idx_k = shell_idx_l.view(1, -1).expand(K, N)
    F2 = F_calc * F_calc
    sum_per_shell = torch.zeros((K, n_shells), dtype=real_dtype, device=device)
    sum_per_shell.scatter_add_(1, shell_idx_k, F2)
    mean_per_shell = (sum_per_shell / cnt.clamp(min=1.0).unsqueeze(0)).clamp(min=1e-30)
    norm_per_refl = mean_per_shell.sqrt().gather(1, shell_idx_k)  # (K, N)
    E_calc = F_calc / norm_per_refl                                 # (K, N)

    E_obs = obs.E_obs.to(device).to(real_dtype)

    sigma_a_d = sigma_a.to(device).to(real_dtype)                  # (n_shells,)
    D_per_refl = sigma_a_d.index_select(0, shell_idx_l)            # (N,)
    var_d = (1.0 - D_per_refl * D_per_refl).clamp(min=1e-4)        # (N,)
    if interp_var is not None:
        var_per_refl = (var_d + interp_var.to(device).to(real_dtype)).clamp(min=1e-4)
    else:
        var_per_refl = var_d

    F_mean = D_per_refl.view(1, N) * E_calc                        # (K, N)
    var_full = var_per_refl.view(1, N).expand(K, N)
    E_obs_full = E_obs.view(1, N).expand(K, N)
    cent_full = centric.to(device).to(torch.bool).view(1, N)

    ll_acent = rice_log_likelihood(E_obs_full, F_mean, var_full)
    ll_cent = woolfson_log_likelihood(E_obs_full, F_mean, var_full)
    ll = torch.where(cent_full, ll_cent, ll_acent)                 # (K, N)

    # Wilson reference (data only): F_mean = 0, var = 1.
    var0 = torch.ones_like(E_obs)
    F_mean0 = torch.zeros_like(E_obs)
    ll_wil_acent = rice_log_likelihood(E_obs, F_mean0, var0)
    ll_wil_cent = woolfson_log_likelihood(E_obs, F_mean0, var0)
    ll_wil_per_refl = torch.where(centric.to(device).to(torch.bool),
                                   ll_wil_cent, ll_wil_acent)
    ll_wil_total = ll_wil_per_refl.sum()

    return ll.sum(dim=1) - ll_wil_total                            # (K,)


def precompute_G_for_rotation(
    interpolator,
    R_rotation: torch.Tensor,
    hkl: torch.Tensor,
    spacegroup,
    real_cell,
    device=None,
):
    """
    Pre-compute per-symmetry F_asu contributions `G_i(h)` for a fixed rotation.

    These are the only inputs that depend on `R_rotation` (and therefore on
    expensive interpolator/model-forward evaluations). Passing the result
    into `amplitude_translation_search` and `local_translation_refine` lets
    them share a single set of (n_sym) model evaluations across the coarse
    TF and the fine refinement, instead of recomputing each call.

    Returns
    -------
    G : (S, N) complex128
    h_R : (S, N, 3) float64
    """
    real_dtype = torch.float64
    complex_dtype = torch.complex128
    if device is None:
        device = getattr(interpolator, "device", hkl.device)

    hkl_t = hkl.detach().to(device).to(real_dtype)
    sym_R = spacegroup.matrices.detach().to(device).to(real_dtype)
    sym_t = spacegroup.translations.detach().to(device).to(real_dtype)
    S = sym_R.shape[0]
    N = hkl_t.shape[0]
    R_rot = R_rotation.detach().to(device).to(real_dtype)

    # Batched: h_R[i, n, d] = Σ_e hkl[n, e] · sym_R[i, e, d]
    h_R = torch.einsum("ne,ied->ind", hkl_t, sym_R)              # (S, N, 3)
    # Per-sym-op translation phase: exp(2πi · h · t_i)
    two_pi_i = 2j * torch.pi
    phase_arg = torch.einsum("ne,ie->in", hkl_t, sym_t)          # (S, N)
    phase = torch.exp(two_pi_i * phase_arg.to(complex_dtype))    # (S, N)

    # One interpolator.evaluate over all (S × N) rotated indices: lets the
    # backend do a single grid_sample instead of S sequential ones.
    h_R_flat = h_R.reshape(-1, 3)                                # (S·N, 3)
    F_flat = interpolator.evaluate(
        R_rot, h_R_flat, real_cell, return_amplitude=False,
    )
    F_all = F_flat.reshape(S, N).to(complex_dtype)               # (S, N)

    G = F_all * phase
    return G, h_R


def local_translation_refine(
    obs: TranslationObs,
    interpolator,
    R_rotation: torch.Tensor,
    spacegroup,
    real_cell,
    t_init: torch.Tensor,
    radius: float = 0.06,
    grid_steps: int = 13,
    n_refinement_passes: int = 2,
    batch_size: int = 1024,
    precomputed_G: Optional[torch.Tensor] = None,
    precomputed_h_R: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Fine-grid Patterson translation refinement around ``t_init``.

    Locates the peak on a fine grid of half-width ``radius`` around ``t_init``
    by the same weighted ``E^2`` correlation the coarse search maximises, then
    reports the analytical-scale R-factor there::

        R(t) = sum ||F_obs| - k|F_calc(t)|| / sum |F_obs|
        k(t) = sum |F_obs||F_calc(t)| / sum |F_calc(t)|^2

    The two halves answer different questions and use different quantities on
    purpose. The *search* runs on normalised, weighted ``E^2``, because that is
    what the coarse stage optimised and refining against a different objective
    would walk away from the peak it was handed. The *reported number* is an
    R-factor on raw amplitudes, because that is what ranks candidates and what a
    crystallographer reads.

    That R uses one global scale, so it is not the number a full Scaler returns.
    It is used as a ranking key, and for that its minimum's *location* is what
    matters. The winner gets a solvent-aware Scaler refit.

    Returns ``(t, R)`` at the minimum. ``n_refinement_passes`` is accepted and
    ignored -- the FFT evaluates the whole fine grid at once, so there is
    nothing for a second zoom pass to buy.
    """
    device = getattr(interpolator, "device", obs.hkl.device)
    real_dtype = torch.float64
    complex_dtype = torch.complex128

    hkl = obs.hkl
    F_obs_t = obs.F_obs.to(device).to(real_dtype)
    F_obs_sum = F_obs_t.sum().clamp(min=1e-30)
    E_obs = obs.E_obs.to(device).to(real_dtype)
    w = obs.weight.to(device).to(real_dtype)

    two_pi_i = 2j * torch.pi
    if precomputed_G is not None and precomputed_h_R is not None:
        G = precomputed_G.to(device).to(complex_dtype)
        h_R = precomputed_h_R.to(device).to(real_dtype)
    else:
        G, h_R = precompute_G_for_rotation(
            interpolator, R_rotation, hkl, spacegroup, real_cell, device=device,
        )

    # Adapt batch_size to keep the largest inner einsum tensor under ~250 MB
    # of complex128. The (S, B, N) phase tensor is the offender:
    # S × B × N × 16 bytes ≤ 2.5e8 → B ≤ 2.5e8 / (16 × S × N).
    S_eff, N_eff = G.shape
    safe_b = max(8, int(2.5e8 / (16.0 * max(S_eff, 1) * max(N_eff, 1))))
    batch_size = min(batch_size, safe_b)

    # Crowther–Blow FFT refinement on a fine grid around t_init.
    # Bake t_init into G as a per-h_R phase factor, then the IFFT trick from
    # `amplitude_translation_search` works on the offset grid Δt with the
    # same (num/den) Pearson-proxy scoring. The previous nested-grid Python
    # evaluation paid O(G³ · N · S) Bessel/exp/einsum per call; this pays
    # one IFFT pair on G_fft³ + an O(N · S) F_calc evaluation at the final t.
    S, N = G.shape
    t_init_t = torch.as_tensor(t_init, dtype=real_dtype, device=device)
    h_R_int = h_R.round().to(torch.int64)                                # (S, N, 3)

    # Bake t_init into G:
    phase_init = torch.exp(
        two_pi_i * torch.einsum("snd,d->sn", h_R, t_init_t).to(complex_dtype)
    )                                                                    # (S, N)
    G_shifted = G * phase_init                                           # (S, N)

    # Pick G_fft so the IFFT spacing matches the requested fine grid:
    # spacing = 2·radius / (grid_steps − 1), G_fft = round(1 / spacing).
    # Cap at 128 to bound memory (128³ complex128 ≈ 32 MB).
    desired_spacing = max(2.0 * float(radius) / max(grid_steps - 1, 1), 1e-6)
    G_fft = max(grid_steps, int(round(1.0 / desired_spacing)))
    G_fft = min(G_fft, 128)
    half_window = max(1, int(round(float(radius) * G_fft)))

    # The same weighted E^2 correlation as the coarse search. It used to be a
    # raw |F|^2 correlation here, which made the fine grid optimise a different
    # objective from the one that chose the peak it is centred on.
    F_obs2 = E_obs * E_obs
    F_obs2_centered = F_obs2 - (w * F_obs2).sum() / w.sum().clamp(min=1e-30)
    F_obs2_c_complex = (w * F_obs2_centered).to(complex_dtype)
    w_complex = w.to(complex_dtype)

    W_num_flat = torch.zeros(G_fft ** 3, dtype=complex_dtype, device=device)
    W_den_flat = torch.zeros(G_fft ** 3, dtype=complex_dtype, device=device)
    G_stride_xy = G_fft * G_fft
    for i in range(S):
        Gi_conj = G_shifted[i].conj()
        pair = Gi_conj.view(1, -1) * G_shifted                           # (S, N)
        coeff_num = F_obs2_c_complex.view(1, -1) * pair
        coeff_den = w_complex.view(1, -1) * pair
        dh = (h_R_int - h_R_int[i:i + 1]) % G_fft                        # (S, N, 3)
        flat = (dh[..., 0] * G_stride_xy
                + dh[..., 1] * G_fft + dh[..., 2])                       # (S, N)
        flat_flat = flat.reshape(-1)
        W_num_flat.index_add_(0, flat_flat, coeff_num.reshape(-1))
        W_den_flat.index_add_(0, flat_flat, coeff_den.reshape(-1))

    W_num = W_num_flat.view(G_fft, G_fft, G_fft)
    W_den = W_den_flat.view(G_fft, G_fft, G_fft)
    num_t = torch.fft.ifftn(W_num, dim=(0, 1, 2)).real * (G_fft ** 3)
    den_t = torch.fft.ifftn(W_den, dim=(0, 1, 2)).real * (G_fft ** 3)
    score = num_t / den_t.clamp(min=1e-30)

    # Roll so the (Δt = 0) cell sits in the centre of a (2·half_window+1)
    # window, then look for the maximum within the radius-sphere.
    score_rolled = torch.roll(
        score, shifts=(half_window, half_window, half_window), dims=(0, 1, 2),
    )
    w = 2 * half_window + 1
    score_window = score_rolled[:w, :w, :w]
    idx_flat = int(score_window.argmax().item())
    jx = idx_flat // (w * w)
    rem = idx_flat % (w * w)
    jy = rem // w
    jz = rem % w
    Delta_t = torch.tensor(
        [(jx - half_window) / G_fft,
         (jy - half_window) / G_fft,
         (jz - half_window) / G_fft],
        dtype=real_dtype, device=device,
    )
    best_t = t_init_t + Delta_t

    # Compute the analytical-scale R-factor at best_t (one t evaluation),
    # which is what the caller uses to rank rotation × translation
    # candidates. The local-refine grid search above optimised the
    # FFT-scored Pearson proxy; analytical R is monotonically related on
    # this neighbourhood so the choice of which fine-grid maximum to
    # commit to is preserved.
    phase_best = torch.exp(
        two_pi_i * torch.einsum("snd,d->sn", h_R, best_t).to(complex_dtype)
    )                                                                    # (S, N)
    F_calc_best = (G * phase_best).sum(dim=0)                            # (N,) complex
    F_c_abs = F_calc_best.abs().to(real_dtype)
    num_a = (F_obs_t * F_c_abs).sum()
    den_a = (F_c_abs ** 2).sum().clamp(min=1e-30)
    k = num_a / den_a
    best_R = float(
        ((F_obs_t - k * F_c_abs).abs().sum() / F_obs_sum).item()
    )
    # Unused: `n_refinement_passes`, `batch_size` kept in signature for
    # back-compat with callers passing them.
    _ = n_refinement_passes
    _ = batch_size

    return best_t.cpu(), best_R


