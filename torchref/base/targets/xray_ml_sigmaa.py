"""Maximum-likelihood (Read MLF) X-ray math with a per-shell model-error
variance ``beta``.

PyTorch port of the Phenix/cctbx maximum-likelihood sigma_A treatment with the
Luzzati ``alpha`` (mean-coupling) term fixed at 1: the per-reflection model mean
is ``|F_calc|`` and the conditional variance is ``epsilon * beta`` (the
experimental sigma is not added). ``beta`` is the absolute model-error variance
in F^2 units and is the overfit-controlling ingredient.

The estimator (:func:`estimate_beta`) fits ``beta`` by maximum likelihood per
resolution shell on the FREE set (weighted moments plus a root-find for the ML
parameter ``topt``), 3-point smooths ``topt``, and interpolates ``beta`` to
every reflection.

Acentric (with ``eb = epsilon * beta``)::

    L = -log(2 F_o / eb) + F_o**2/eb + F_c**2/eb - log I0(2 F_o F_c / eb)

Centric::

    L = -0.5 log(2/(pi eb)) + F_o**2/(2eb) + F_c**2/(2eb) - log cosh(F_o F_c / eb)

With ``beta=1, epsilon=1`` (``eb=1``) this reduces to the unit-variance MLF
(used as a reduction test) — exactly, up to the ``+1e-12`` floors inside the
``log``/denominator terms, which are numerical guards and not part of the
analytic target. Numerical-stability tricks (``i0e`` exp-scaled Bessel,
log-cosh shifted form, clamps) match :mod:`torchref.base.targets.xray_ml`.

This module is the live machinery for the X-ray mode named ``ml``; the name
``ml_sigmaa`` (and this file's basename) is a retired alias kept for
backward compatibility.
"""

import math

import numpy as np
import torch

from torchref.config import get_float_dtype

# =====================================================================
# Loss math (mean = |Fc|, variance = epsilon*beta)
# =====================================================================


def _ml_beta_nll_per_refl(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    beta: torch.Tensor,
    centric_flags: torch.Tensor,
    epsilon: torch.Tensor = None,
) -> torch.Tensor:
    """Per-reflection NLL (NOT masked/summed). ``Sigma = epsilon * beta``."""
    if centric_flags is None:
        centric_flags = torch.zeros_like(F_obs, dtype=torch.bool)
    if epsilon is None:
        epsilon = torch.ones_like(F_obs)

    F_calc_amp = torch.abs(F_calc)
    beta = torch.clamp(beta, min=1e-10)

    Sigma = torch.clamp(epsilon * beta, min=1e-10)
    Fc = F_calc_amp

    # --- acentric -----------------------------------------------------------
    term1 = -torch.log(2 * F_obs / Sigma + 1e-12)
    term2 = (F_obs**2) / Sigma
    term3 = Fc**2 / Sigma
    arg_bessel = torch.clamp(2 * Fc * F_obs / Sigma, max=1e6)
    term4 = -(torch.log(torch.special.i0e(arg_bessel) + 1e-12) + arg_bessel)
    loss_acentric = term1 + term2 + term3 + term4

    # --- centric ------------------------------------------------------------
    term1_c = -0.5 * torch.log(2 / (np.pi * Sigma) + 1e-12)
    term2_c = (F_obs**2) / (2 * Sigma)
    term3_c = Fc**2 / (2 * Sigma)
    term4_c = -(Fc * F_obs) / Sigma
    arg_exp = torch.clamp(-2 * Fc * F_obs / Sigma, min=-80.0, max=80.0)
    term5_c = -torch.log((1 + torch.exp(arg_exp)) / 2 + 1e-12)
    loss_centric = term1_c + term2_c + term3_c + term4_c + term5_c

    return torch.where(centric_flags, loss_centric, loss_acentric)


def ml_xray_loss_beta_math(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    beta: torch.Tensor,
    centric_flags: torch.Tensor,
    mask: torch.Tensor = None,
    epsilon: torch.Tensor = None,
) -> torch.Tensor:
    """Masked-sum Read-MLF loss; mean ``|Fc|``, variance ``epsilon*beta``.

    ``mask`` defaults to all reflections (``None``); compact inputs need no mask.
    """
    if mask is None:
        mask = torch.ones(F_obs.shape[0], dtype=torch.bool, device=F_obs.device)
    loss = _ml_beta_nll_per_refl(F_obs, F_calc, beta, centric_flags, epsilon)
    loss = torch.where(torch.isfinite(loss), loss, torch.full_like(loss, 1e6))
    return (loss * mask).sum()


# =====================================================================
# Per-reflection epsilon (multiplicity)
# =====================================================================


def epsilon_from_hkl(hkl: torch.Tensor, spacegroup) -> torch.Tensor:
    """Per-reflection epsilon: number of rotation symops mapping h -> +/-h.

    Mirrors ``ReciprocalSymmetry.get_epsilon`` (Friedel-aware) but works directly
    on the scattered HKL list. Returns ones if ``spacegroup`` is None or lacks
    ``apply_to_hkl``.

    Always returns on ``hkl.device``, whatever device the space group's symmetry
    matrices live on: the caller multiplies this against per-reflection data
    sitting beside ``hkl``.
    """
    n = hkl.shape[0]
    float_dtype = get_float_dtype()
    if spacegroup is None or not hasattr(spacegroup, "apply_to_hkl"):
        return torch.ones(n, device=hkl.device, dtype=float_dtype)

    with torch.no_grad():
        # Configured float dtype, not float64: MPS has no float64 and casting
        # there raises. Symmetry arithmetic on Miller indices is exact in
        # float32 (integer-valued rotation matrices, small indices), so the
        # exact `==` comparisons below remain valid.
        #
        # ``apply_to_hkl`` moves its input onto the matrices' device, so build
        # ``h`` there too -- otherwise ``Hs`` and ``h0`` land on different
        # devices and the comparisons below raise. The space group wins for the
        # arithmetic; the result is handed back on the caller's device.
        sym_device = getattr(spacegroup, "matrices", hkl).device
        h = hkl.to(device=sym_device, dtype=float_dtype)
        Hs = spacegroup.apply_to_hkl(h)  # (N,3,ops)
        h0 = h.unsqueeze(-1)  # (N,3,1)
        same = (Hs == h0).all(dim=1)
        friedel = (Hs == -h0).all(dim=1)
        eps = (same | friedel).sum(dim=1).clamp(min=1).to(float_dtype)
    return eps.to(hkl.device)


# =====================================================================
# Phenix-style ML alpha/beta estimator (Lunin-Skovoroda)
# =====================================================================


def _fom_term(arg: torch.Tensor, centric: torch.Tensor) -> torch.Tensor:
    """Rice figure of merit: I1/I0(2*arg) acentric, tanh(arg) centric.

    Uses exp-scaled Bessel (i1e/i0e) so the exp factors cancel -> stable for
    large arg. ``arg = topt_bin * b_j >= 0``.
    """
    z = torch.clamp(2.0 * arg, min=0.0, max=1e6)
    rac = torch.special.i1e(z) / (torch.special.i0e(z) + 1e-30)
    return torch.where(centric, torch.tanh(arg), rac)


def estimate_beta(
    F_obs: torch.Tensor,
    F_calc: torch.Tensor,
    centric: torch.Tensor,
    epsilon: torch.Tensor,
    d_star_sq: torch.Tensor,
    free_mask: torch.Tensor,
    per_bin: int = 140,
    n_iter: int = 60,
    sigma_a_max: float = 0.99,
    min_bins: int = 5,
    min_per_bin: int = 40,
):
    """Maximum-likelihood per-reflection model-error variance ``beta``.

    Estimates ``beta`` on the FREE set in equal-count resolution shells via the
    Lunin-Skovoroda ML root-find for ``topt`` (``beta_bin = 2*hbeta``), 3-point
    smooths ``topt``, then interpolates linearly in ``d_star_sq`` to all
    reflections. Intended to run under ``torch.no_grad()``.

    Parameters
    ----------
    F_obs, F_calc, centric, epsilon, d_star_sq, free_mask : torch.Tensor
        1-D length-N tensors. ``F_calc`` is the scaled amplitude.
    per_bin : int, optional
        Target reflections per resolution shell. Default 140.
    n_iter : int, optional
        Iterations for the bracket and regula-falsi root-find. Default 60.
    sigma_a_max : float, optional
        Floors ``beta`` at ``(1 - sigma_a_max**2) * B`` so it cannot collapse to
        ~0 in saturated bins (a near-zero variance gives a near-infinite
        per-reflection weight). Default 0.99.
    min_bins, min_per_bin : int, optional
        Floor the bin count (down to ``min_per_bin`` reflections/bin) so sparse
        free sets still get enough shells for ``beta`` to vary smoothly.
        Defaults 5 and 40.

    Returns
    -------
    tuple of torch.Tensor
        ``(beta_per_refl, beta_per_bin, bin_dss)``. ``beta_per_refl`` is the
        per-bin ``beta`` interpolated to every reflection, floored at
        ``(1 - sigma_a_max**2) * B`` (and clamped away from 0; see
        ``sigma_a_max``). The latter two are ``None`` for a degenerate free set
        (fewer than 2 free reflections).

    Notes
    -----
    The per-bin moments are not smoothed: they must stay mutually consistent
    (``C**2 <= A*B`` etc.) for the ML root-find, so only the fitted ``topt`` is
    3-point smoothed afterwards (matching Phenix).
    """
    device = F_obs.device
    dtype = F_obs.dtype
    fo_all = F_obs.reshape(-1)
    fc_all = torch.abs(F_calc).reshape(-1)
    cen_all = (
        centric.reshape(-1).to(torch.bool)
        if centric is not None
        else torch.zeros_like(fo_all, dtype=torch.bool)
    )
    eps_all = (
        epsilon.reshape(-1).to(dtype)
        if epsilon is not None
        else torch.ones_like(fo_all)
    )
    dss_all = d_star_sq.reshape(-1).to(dtype)

    free_idx = torch.nonzero(free_mask.reshape(-1), as_tuple=True)[0]
    n_free = int(free_idx.numel())
    if n_free < 2:
        # degenerate: no usable free set -> beta=<Fo^2>
        valid = torch.isfinite(fo_all)
        b = (
            (fo_all[valid] ** 2).mean()
            if valid.any()
            else torch.ones((), device=device, dtype=dtype)
        )
        beta = torch.full_like(fo_all, float(b))
        return beta, None, None

    # --- sort free reflections by resolution, equal-count chunks ----------
    # stable=True so tied d_star_sq break deterministically and identically on
    # CPU/GPU (a non-stable CUDA argsort shuffles ties per process, which
    # reshuffles shell membership and makes beta non-deterministic).
    order = torch.argsort(dss_all[free_idx], stable=True)
    fo = fo_all[free_idx][order]
    fc = fc_all[free_idx][order]
    cen = cen_all[free_idx][order]
    eps = eps_all[free_idx][order]
    dss = dss_all[free_idx][order]

    # Adaptive bin count. Aim for ~per_bin reflections/bin, but for sparse free
    # sets ~140/bin can give only 2 bins -> a dead high-res half. Floor the bin
    # count at min_bins (down to min_per_bin reflections/bin) so beta can decay
    # smoothly across enough shells.
    n_by_count = max(1, n_free // per_bin)
    n_cap = max(1, n_free // min_per_bin)
    n_bins = max(n_by_count, min(min_bins, n_cap))
    seg = (torch.arange(n_free, device=device) * n_bins) // n_free  # (n_free,)

    # Segments are contiguous (data is sorted by resolution and ``seg`` is a
    # non-decreasing ramp), so the per-shell reductions are contiguous segment
    # sums. Use ``segment_reduce`` (atomic-free, one program per segment) rather
    # than ``scatter_add`` (CUDA atomicAdd, non-deterministic accumulation
    # order): bit-stable run-to-run and identical CPU/GPU to float32 rounding.
    seg_lengths = torch.bincount(seg, minlength=n_bins)

    def segsum(x):
        return torch.segment_reduce(x, "sum", lengths=seg_lengths, unsafe=True)

    w = torch.where(cen, torch.ones_like(fo), 2.0 * torch.ones_like(fo))
    SUMw = segsum(w).clamp(min=1e-30)
    fm2e = fc * fc / eps
    fo2e = fo * fo / eps
    bj = fo * fc / eps  # per-reflection "b_j"

    A = segsum(w * fm2e) / SUMw
    B = segsum(w * fo2e) / SUMw
    C = segsum(w * bj) / SUMw
    AB = (A * B).clamp(min=1e-30)

    # --- wi/AB = 1 - rho^2, computed float32-stably ---
    # The naive ``wi = A*B - C**2`` is a catastrophic cancellation when the
    # shell correlation rho -> 1 (it subtracts two nearly-equal large numbers;
    # in float32 it is 6-59% wrong right where the saturated gate lives). Write
    # it instead as the squared orthogonal residual of the weighted amplitude
    # vectors u = sqrt(w/eps)*Fc, v = sqrt(w/eps)*Fo: with unit vectors
    # u_hat, v_hat and rho = <u_hat, v_hat>,
    #     1 - rho^2 = sum_i (u_hat_i - rho * v_hat_i)^2,
    # a sum of non-negative terms (no cancellation, accurate to ~1e-5 in f32).
    sc_w = torch.sqrt((w / eps).clamp(min=0.0))
    u = sc_w * fc
    v = sc_w * fo
    nu = torch.sqrt((A * SUMw).clamp(min=1e-30))  # ||u|| per shell
    nv = torch.sqrt((B * SUMw).clamp(min=1e-30))  # ||v|| per shell
    rho = (C / torch.sqrt(AB)).clamp(min=-1.0, max=1.0)  # per shell
    resid = u / nu[seg] - rho[seg] * (v / nv[seg])  # per reflection
    wiAB = segsum(resid * resid).clamp(min=0.0)  # = 1 - rho^2, per shell
    wi = wiAB * AB

    # --- OMEGA = Pearson correlation of X=Fc^2/eps and Y=Fo^2/eps, computed
    # float32-stably via CENTERED moments (the naive
    # (D - A*B)/sqrt((p - A^2)(q - B^2)) cancels the same way). Cov/Var are
    # accumulated about the per-shell means A=<X>, B=<Y>. ---
    dX = fm2e - A[seg]
    dY = fo2e - B[seg]
    VarX = segsum(w * dX * dX) / SUMw
    VarY = segsum(w * dY * dY) / SUMw
    Cov = segsum(w * dX * dY) / SUMw
    rr = VarX * VarY
    OMEGA = torch.where(
        rr > 0, Cov / torch.sqrt(rr.clamp(min=1e-30)), torch.zeros_like(A)
    )

    trivial = OMEGA <= 0.0
    saturated = wiAB <= 3.0e-7
    need = (~trivial) & (~saturated)

    bin_centric = cen  # per-free-reflection; FOM uses per-reflection centric

    def blamm(t_bin):
        arg = t_bin[seg] * bj
        return segsum(w * bj * _fom_term(arg, bin_centric)) / SUMw

    def funcgm(t_bin):
        return (
            torch.sqrt(1.0 + 4.0 * A * B * t_bin * t_bin)
            - 1.0
            - 2.0 * t_bin * blamm(t_bin)
        )

    # --- bracket + regula-falsi root-find for topt (vectorized over bins) ---
    # funcgm(0)=0 and funcgm(t)>0 for large t (== at t=C/wi); the wanted root is
    # the positive crossing in (0, C/wi) where funcgm dips negative. Mirror
    # Phenix ``solvm``: start at t_hi=C/wi (funcgm>0), halve until funcgm<0 to
    # bracket [t_lo (neg), t_up (pos)], then regula-falsi.
    t_hi = (C / wi.clamp(min=1e-30)).clamp(min=1e-30)
    t_lo = t_hi.clone()
    t_up = t_hi.clone()
    found = torch.zeros(n_bins, device=device, dtype=torch.bool)
    for _ in range(n_iter):
        still = need & (~found)
        t_new = torch.where(still, t_lo * 0.5, t_lo)
        f_new = funcgm(t_new)
        now = still & (f_new < 0.0)
        t_up = torch.where(now, t_lo, t_up)  # last positive point
        t_lo = torch.where(still, t_new, t_lo)
        found = found | now
    no_root = need & (~found)  # funcgm never went negative -> topt=0

    a = t_lo  # negative side
    b = t_up  # positive side
    topt_solved = b.clone()
    for _ in range(n_iter):
        fa = funcgm(a)
        fb = funcgm(b)
        denom = fb - fa
        denom = torch.where(denom.abs() < 1e-30, torch.full_like(denom, 1e-30), denom)
        t = (a * fb - b * fa) / denom
        ft = funcgm(t)
        pos = ft > 0.0
        b = torch.where(found & pos, t, b)
        a = torch.where(found & (~pos), t, a)
        topt_solved = torch.where(found, t, topt_solved)

    topt = torch.zeros(n_bins, device=device, dtype=dtype)
    topt = torch.where(found, topt_solved, topt)  # found roots
    topt = torch.where(no_root, torch.zeros_like(topt), topt)
    topt = torch.where(saturated, torch.full_like(topt, 1e10), topt)
    topt = torch.where(trivial, torch.zeros_like(topt), topt)

    # 3-point smooth on topt (Phenix smooths topt before deriving beta)
    topt = _smooth3(topt)

    # --- beta from topt (Phenix alpha_beta_in_zones; alpha readout dropped) --
    tt = 2.0 * topt
    ww = torch.sqrt(1.0 + A * B * tt * tt)
    hbeta = B / (ww + 1.0)
    beta_bin = 2.0 * hbeta
    # trivial bins (genuinely uncorrelated shell): full variance beta=B.
    beta_bin = torch.where(topt <= 0.0, B, beta_bin)

    # Physical model-error floor on beta: the conditional variance eps*beta is
    # never ~0 (sigma_A < 1). beta >= (1 - sigma_a_max^2) * B caps the
    # per-reflection weight and stops the (near-)saturated-bin blow-ups.
    beta_floor = float(1.0 - sigma_a_max * sigma_a_max) * B
    beta_bin = torch.maximum(beta_bin, beta_floor)
    beta_bin = torch.nan_to_num(beta_bin, nan=1.0).clamp(min=1e-10)

    # bin-center resolution for interpolation
    counts = segsum(torch.ones_like(fo)).clamp(min=1.0)
    bin_dss = segsum(dss) / counts

    beta_refl = _interp_in_dss(dss_all, bin_dss, beta_bin)
    return beta_refl, beta_bin, bin_dss


def _smooth3(v: torch.Tensor) -> torch.Tensor:
    """3-point moving average across bins (edge-padded)."""
    if v.numel() < 3:
        return v
    pad = torch.cat([v[:1], v, v[-1:]])
    return (pad[:-2] + pad[1:-1] + pad[2:]) / 3.0


def _interp_in_dss(dss_all, bin_dss, vals):
    """Linear interpolation of per-bin ``vals`` (at ``bin_dss``) to all
    reflections by their ``d_star_sq``; clamp-to-edge outside the range."""
    n_bins = bin_dss.numel()
    if n_bins == 1:
        return torch.full_like(dss_all, float(vals[0]))
    idx = torch.searchsorted(bin_dss, dss_all).clamp(1, n_bins - 1)
    x0 = bin_dss[idx - 1]
    x1 = bin_dss[idx]
    wlin = ((dss_all - x0) / (x1 - x0).clamp(min=1e-30)).clamp(0.0, 1.0)
    return (1 - wlin) * vals[idx - 1] + wlin * vals[idx]


# =====================================================================
# Stateful estimator (owned by the consuming target, not the scaler)
# =====================================================================


class SigmaAEstimator:
    """Lazy, cached free-set model-error variance ``beta`` (Luzzati σ_A).

    Thin stateful wrapper around :func:`estimate_beta`: it caches the detached
    ``(beta, epsilon)`` from the last estimate and re-estimates only after
    :meth:`reset`. The owning target calls :meth:`reset` from its
    ``maintenance()`` hook so ``beta`` refreshes once per optimizer-step block
    (the same cadence the scaler used previously).

    Ownership note
    --------------
    ``beta`` (the conditional variance ``epsilon*beta``) is the only
    overfit-controlling ingredient; the Luzzati ``alpha`` mean-coupling is
    gauge-absorbed by the scaler and intentionally not produced (see the module
    docstring). This estimator therefore belongs to the *target* that consumes
    the likelihood, not to the scaler (which now owns scaling only). Plain
    tensor in/out — no ``ReflectionData``/``Scaler`` coupling — so it is usable
    from both ``scaling`` and ``refinement.targets`` without an import cycle.
    """

    def __init__(self):
        self._cache = None  # (beta_per_refl, epsilon) detached
        self._beta_per_bin = None  # diagnostics

    def reset(self) -> None:
        """Invalidate the cache so the next :meth:`get` re-estimates ``beta``."""
        self._cache = None

    @property
    def beta_per_bin(self):
        """Last-estimated per-bin ``beta`` (diagnostics); ``None`` until first call."""
        return self._beta_per_bin

    def get(
        self,
        F_obs: torch.Tensor,
        F_calc_scaled: torch.Tensor,
        centric: torch.Tensor,
        epsilon: torch.Tensor,
        d_star_sq: torch.Tensor,
        free_mask: torch.Tensor,
        out_epsilon: torch.Tensor = None,
        target_dss: torch.Tensor = None,
    ):
        """Return cached-or-recomputed ``(beta, epsilon)``, both detached.

        Estimated on the **free** set under ``no_grad``; gradients never flow
        through ``beta``.

        Parameters
        ----------
        F_obs, F_calc_scaled, centric, epsilon, d_star_sq, free_mask
            Length-N tensors passed straight to :func:`estimate_beta`.
            ``F_calc_scaled`` must already carry the scaler's scaling.
        out_epsilon : torch.Tensor, optional
            Multiplicity to return/cache, if it differs from the estimation
            ``epsilon`` (e.g. the collection case pools several datasets for the
            fit but applies a single common ``epsilon``). Defaults to
            ``epsilon``.
        target_dss : torch.Tensor, optional
            If given, the per-bin ``beta`` is interpolated onto this
            ``d_star_sq`` grid instead of using the per-reflection estimate
            (used to map a pooled multi-dataset fit back onto the common HKL).
        """
        if self._cache is not None:
            return self._cache
        with torch.no_grad():
            beta_refl, bbin, bin_dss = estimate_beta(
                F_obs, F_calc_scaled, centric, epsilon, d_star_sq, free_mask
            )
            self._beta_per_bin = bbin

            if target_dss is not None:
                if bbin is None or bin_dss is None:
                    finite = torch.isfinite(F_obs)
                    mean_fo2 = (
                        (F_obs[finite] ** 2).mean()
                        if finite.any()
                        else F_obs.new_ones(())
                    )
                    beta = torch.full_like(target_dss, float(mean_fo2))
                else:
                    beta = _interp_in_dss(target_dss, bin_dss, bbin)
            else:
                beta = beta_refl

            eps_ret = out_epsilon if out_epsilon is not None else epsilon
            eps_ret = eps_ret.detach() if torch.is_tensor(eps_ret) else eps_ret
            self._cache = (beta.detach(), eps_ret)
        return self._cache
