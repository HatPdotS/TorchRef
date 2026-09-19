"""Difference-driven error estimation: the expected difference power ``sigma_D``.

A light-minus-dark difference coefficient ``dF_obs = dF_true + noise`` carries a true
signal whose power ``S = E[dF_true**2]`` varies with resolution and with the dark
amplitude, and a measurement noise ``sigma_diff**2`` the merge reports. The best linear
estimate of ``dF_true`` from ``dF_obs`` is ``w * dF_obs`` with the Wiener weight
``w = S / (S + sigma_diff**2)``, so a difference map needs ``S`` per reflection. Inverse
variance alone weights by precision and treats every reflection as carrying the same
expected difference, which suppresses the strong reflections whose difference power is
ten to seventy times that of weak ones.

``S`` needs no half datasets. Per resolution shell the second moment of the observed
differences is ``B = S + S2`` with ``S2`` the mean measurement variance, so
``Sigma_N = B - S2`` is the expected true difference power, the same identity
:mod:`.sigma_a` uses for amplitudes. Within a shell the power follows the dark amplitude
as ``(F_dark / <F_dark>)**gamma`` with one fitted exponent, carried by the per-reflection
multiplier ``epsilon`` exactly as the reflection multiplicity is. With a difference model
``dF_calc`` the shell moments also give the Gaussian coupling ``alpha = <dF_obs dF_calc> /
<dF_calc**2>`` and the unexplained power ``beta_model = Sigma_N - alpha**2 Sigma_P``, the
extra variance a difference likelihood adds to ``sigma_diff**2``.

Differences are signed and small, so the statistics are Gaussian throughout: there is no
Rice branch and no centric distinction. Plain tensors in and out, no ``ReflectionData``
or ``Scaler`` coupling, so :mod:`torchref.maps` and :mod:`torchref.cli` can import this
module without closing an import cycle. Every result lives on the device of its inputs.
"""

import math
from dataclasses import dataclass

import torch

from torchref.config import get_float_dtype

from ._shells import equal_count_shells, interp_in_dss, segsum
from .sigma_a import SHRINK_ENABLED

# --- sigma_D estimator constants -------------------------------------------------
#: Shell construction, matching the ``estimate_beta`` defaults so a sigma_A and a sigma_D
#: fit on the same reflections use the same shells.
PER_BIN = 140
MIN_BINS = 5
MIN_PER_BIN = 40
#: Exponent of the dark-amplitude power law when it cannot be fitted. The fitted value on
#: the small-molecule, TD1 and bacteriorhodopsin A/B campaigns was 0.8-1.2, so 1.0 (power
#: proportional to the amplitude) is the informed default; 0.0 would be the pure shell model.
GAMMA_DEFAULT = 1.0
#: Bounds on the fitted exponent. Outside [0, 2] the class means are dominated by one
#: amplitude decile and the regression is on noise; 2 is proportionality to the intensity.
GAMMA_BOUNDS = (0.0, 2.0)
#: Dark-amplitude quantile classes per shell for the exponent regression. Four keeps at
#: least 35 reflections per class at the default shell size; more classes did not move
#: the fitted exponent on the campaign data.
N_F_CLASSES = 4
#: Minimum number of usable (shell, class) cells before the exponent is fitted at all.
GAMMA_MIN_CLASSES = 8
#: Minimum reflections in a class for its moment to enter the exponent regression.
MIN_PER_CLASS = 5
#: Largest standard error at which a fitted exponent is used. Above it the class
#: moments are noise (a null dataset gives se ~ 1 or more), and the default is safer than
#: a random exponent that would redistribute weight between amplitude classes.
GAMMA_SE_MAX = 0.5
#: Gauss-Newton iterations for the decaying-curve fit; the problem is two-parameter and
#: well conditioned, so this is far more than it needs.
CURVE_ITERS = 60
#: Floor on ``F_dark`` relative to its shell mean before the power law is evaluated, so a
#: zero or near-zero dark amplitude cannot delete the expected difference power.
F_FLOOR_FRAC = 0.05
#: Positive floor used where a logarithm of a clamped-to-zero power is needed.
_TINY = 1e-30


@dataclass(frozen=True)
class SigmaDConfig:
    """The estimator's knobs, as one value.

    ``gamma=None`` fits the dark-amplitude exponent; a float fixes it. ``shrink=None``
    means the module default shared with sigma_A, normalised here so consumers never
    handle ``None``. Frozen, so two consumers sharing a config cannot drift apart.
    """

    gamma: float | None = None
    shrink: bool | None = None

    def __post_init__(self):
        if self.gamma is not None:
            g = float(self.gamma)
            lo, hi = GAMMA_BOUNDS
            if not (lo <= g <= hi):
                raise ValueError(f"gamma must lie in {GAMMA_BOUNDS}, got {g}")
            object.__setattr__(self, "gamma", g)
        object.__setattr__(
            self, "shrink", bool(SHRINK_ENABLED if self.shrink is None else self.shrink)
        )


@dataclass(frozen=True)
class SigmaDShells:
    """Per-shell output of :func:`estimate_sigma_d`.

    All power quantities are ``epsilon``-reduced and in ``F**2`` units of the input.

    Attributes
    ----------
    B, S2
        Raw second moment of the observed differences and mean measurement variance.
    Sigma_N_raw, Sigma_N
        Expected true difference power ``(B - S2)`` clamped at zero, before and after the
        shrinkage of the signed value toward a decaying curve ``exp(a + b d*^2)`` fitted
        to every shell. ``Sigma_N`` is what the weights use.
    Sigma_P, C, alpha, beta_model
        Model power, cross moment, Gaussian coupling ``C / Sigma_P`` and unexplained power
        ``(Sigma_N - alpha**2 Sigma_P)`` clamped at zero. Without a model ``Sigma_P`` and
        ``C`` are zero, ``alpha`` one and ``beta_model == Sigma_N``.
    counts, bin_dss
        Reflections per shell and its mean ``d*^2`` in A^-2, the interpolation abscissa.
    bin_log_fbar, bin_log_z
        Log of the shell-mean dark amplitude and of the shell mean of
        ``(F / Fbar)**gamma``, so the per-reflection multiplier
        ``(F / Fbar)**gamma / Z`` has shell mean one and ``Sigma_N`` stays the shell mean
        of the per-reflection power. Zero when no dark amplitude was supplied.
    shrink_w, tau, curve_a, curve_b
        Shrinkage weight per shell, the between-shell sd about the fitted curve and its
        coefficients ``exp(a + b d*^2)`` (NaN when no curve was fitted; the curve is then
        zero everywhere).
    gamma, gamma_fitted
        The exponent used and whether it was fitted rather than fixed or defaulted.
    has_model, degenerate, all_zero
        Whether ``dF_calc`` was supplied, whether fewer than two usable reflections
        existed, and whether every shell's ``Sigma_N`` is zero (weights would vanish).
    diagnostics
        Counters: ``n_dropped, n_fit, n_shell, n_s2_clamped, n_beta_clamped,
        n_f_floored, n_class_dropped, n_class_used, gamma_se, gamma_at_bound,
        gamma_reason``.
    """

    B: torch.Tensor
    S2: torch.Tensor
    Sigma_N_raw: torch.Tensor
    Sigma_N: torch.Tensor
    Sigma_P: torch.Tensor
    C: torch.Tensor
    alpha: torch.Tensor
    beta_model: torch.Tensor
    counts: torch.Tensor
    bin_dss: torch.Tensor
    bin_log_fbar: torch.Tensor
    bin_log_z: torch.Tensor
    shrink_w: torch.Tensor
    tau: float
    curve_a: float
    curve_b: float
    gamma: float
    gamma_fitted: bool
    has_model: bool
    degenerate: bool
    all_zero: bool
    diagnostics: dict


@dataclass(frozen=True)
class SigmaDEstimate:
    """Everything a consumer needs from one estimate, per reflection and detached.

    Attributes
    ----------
    S
        Expected true difference power ``epsilon * Sigma_N(d*^2) * g(F_dark)``.
    sigma_sq
        The measurement variance the weight was formed with (``sigma_diff**2``).
    w
        Wiener weight ``S / (S + sigma_sq)`` in ``[0, 1)``, not normalised.
    alpha, beta_model
        Coupling and unexplained power, interpolated per shell; ``beta_model`` carries
        the same ``epsilon * g`` multiplier as ``S``.
    epsilon
        The multiplicity actually applied.
    shells
        The :class:`SigmaDShells` this was interpolated from.
    """

    S: torch.Tensor
    sigma_sq: torch.Tensor
    w: torch.Tensor
    alpha: torch.Tensor
    beta_model: torch.Tensor
    epsilon: torch.Tensor
    shells: SigmaDShells


def _working_dtype(t: torch.Tensor) -> torch.dtype:
    dtype = torch.promote_types(get_float_dtype(), t.dtype)
    # dtype-ok: MPS capability guard, not an allocation
    if dtype == torch.float64 and t.device.type == "mps":
        raise RuntimeError(
            "MPS has no float64; set the defaults float dtype to float32 or use CPU"
        )
    return dtype


def _degenerate(
    delta_obs: torch.Tensor, gamma: float, has_model: bool, diagnostics: dict, out_dtype
) -> SigmaDShells:
    """One conservative shell: the mean squared difference as the power, alpha one."""
    ok = torch.isfinite(delta_obs)
    b = (delta_obs[ok] ** 2).mean() if bool(ok.any()) else delta_obs.new_ones(())
    one = torch.ones(1, device=delta_obs.device, dtype=out_dtype)
    zero = torch.zeros(1, device=delta_obs.device, dtype=out_dtype)
    b1 = (one * b).to(out_dtype)
    return SigmaDShells(
        B=b1,
        S2=zero,
        Sigma_N_raw=b1,
        Sigma_N=b1,
        Sigma_P=zero,
        C=zero,
        alpha=one,
        beta_model=b1,
        counts=zero,
        bin_dss=zero,
        bin_log_fbar=zero,
        bin_log_z=zero,
        shrink_w=zero,
        tau=0.0,
        curve_a=float("nan"),
        curve_b=float("nan"),
        gamma=gamma,
        gamma_fitted=False,
        has_model=has_model,
        degenerate=True,
        all_zero=False,
        diagnostics=diagnostics,
    )


def _fit_gamma(
    d2e: torch.Tensor,
    s2e: torch.Tensor,
    log_ratio: torch.Tensor,
    seg: torch.Tensor,
    n_bins: int,
) -> tuple[float, float, bool, int, int, str]:
    """Fit the dark-amplitude exponent from within-shell amplitude classes.

    Each shell is split into ``N_F_CLASSES`` quantile classes of the dark amplitude. A
    class contributes ``log(<dF**2/eps> - <sigma**2/eps>)`` against its mean log amplitude
    ratio when that difference power is positive. One slope is fitted across all shells
    with the shell means removed (fixed effects), weighted by ``n_c / 2``: the log of a
    mean of ``n`` squared Gaussians has variance ``2 / n``.

    Returns ``(gamma, gamma_se, at_bound, n_used, n_dropped, reason)``; ``reason`` is
    ``"fitted"`` or names why the default was taken.
    """
    xs, ys, ws, shell_id = [], [], [], []
    n_dropped = 0
    for k in range(n_bins):
        in_shell = torch.nonzero(seg == k, as_tuple=True)[0]
        n_k = int(in_shell.numel())
        if n_k < N_F_CLASSES * MIN_PER_CLASS:
            n_dropped += N_F_CLASSES
            continue
        order = torch.argsort(log_ratio[in_shell], stable=True)
        idx = in_shell[order]
        cls = (
            torch.arange(n_k, device=seg.device) * N_F_CLASSES
        ) // n_k  # dtype-ok: bincount input; PyTorch requires int64
        lengths = torch.bincount(cls, minlength=N_F_CLASSES).to(d2e.dtype)
        m = (segsum(d2e[idx], lengths) - segsum(s2e[idx], lengths)) / lengths
        xc = segsum(log_ratio[idx], lengths) / lengths
        keep = (m > 0) & (lengths >= MIN_PER_CLASS)
        n_dropped += int((~keep).sum())
        if int(keep.sum()) < 2:
            continue
        xs.append(xc[keep])
        ys.append(torch.log(m[keep]))
        ws.append(lengths[keep] / 2.0)
        shell_id.append(torch.full_like(xc[keep], float(k)))
    if not xs:
        return GAMMA_DEFAULT, float("nan"), False, 0, n_dropped, "too_few_classes"
    x = torch.cat(xs)
    y = torch.cat(ys)
    w = torch.cat(ws)
    sid = torch.cat(shell_id)
    n_used = int(x.numel())
    if n_used < GAMMA_MIN_CLASSES:
        return GAMMA_DEFAULT, float("nan"), False, n_used, n_dropped, "too_few_classes"
    # Remove each shell's weighted mean from x and y: the slope is then estimated from
    # within-shell contrasts only, so shell-to-shell differences in power cannot leak in.
    xc = x.clone()
    yc = y.clone()
    n_shells_used = 0
    for k in torch.unique(sid):
        s = sid == k
        n_shells_used += 1
        wk = w[s]
        xc[s] = x[s] - (wk * x[s]).sum() / wk.sum()
        yc[s] = y[s] - (wk * y[s]).sum() / wk.sum()
    sxx = (w * xc * xc).sum()
    if float(sxx) <= 0.0:
        return (
            GAMMA_DEFAULT,
            float("nan"),
            False,
            n_used,
            n_dropped,
            "no_amplitude_spread",
        )
    gamma = float((w * xc * yc).sum() / sxx)
    dof = n_used - n_shells_used - 1
    if dof > 0:
        resid = yc - gamma * xc
        s2 = float((w * resid * resid).sum() / dof)
        gamma_se = math.sqrt(max(s2, 0.0) / float(sxx))
    else:
        gamma_se = float("nan")
    if not math.isfinite(gamma_se) or gamma_se > GAMMA_SE_MAX:
        return GAMMA_DEFAULT, gamma_se, False, n_used, n_dropped, "too_uncertain"
    lo, hi = GAMMA_BOUNDS
    clamped = min(max(gamma, lo), hi)
    return clamped, gamma_se, clamped != gamma, n_used, n_dropped, "fitted"


def _fit_decay(y: torch.Tensor, var: torch.Tensor, x: torch.Tensor):
    """Weighted fit of ``exp(a + b x)``, ``b <= 0``, to signed per-shell power.

    Works on the signed ``B - S2`` of every shell, so shells whose power is zero or
    negative by sampling noise pull the curve down instead of being ignored: on a null
    dataset the curve goes to zero rather than to the winner's curse of the positive
    shells. Gauss-Newton with step halving on the weighted least squares; the fit is
    two-parameter and well conditioned.

    Returns ``(curve, a, b)``; ``curve`` is zeros with NaN coefficients when fewer than
    four shells are usable or the weighted mean power is not positive.
    """
    nan = float("nan")
    usable = torch.isfinite(y) & torch.isfinite(var) & (var > 0)
    if int(usable.sum()) < 4:
        return torch.zeros_like(y), nan, nan
    w = torch.where(usable, 1.0 / var.clamp(min=_TINY), torch.zeros_like(var))
    yz = torch.where(usable, y, torch.zeros_like(y))
    mean = float((w * yz).sum() / w.sum())
    if mean <= 0.0:
        return torch.zeros_like(y), nan, nan
    a = torch.tensor(math.log(mean), dtype=y.dtype, device=y.device)
    b = torch.zeros((), dtype=y.dtype, device=y.device)

    def loss(a_, b_):
        r = yz - torch.exp(a_ + b_ * x)
        return float((w * r * r).sum())

    current = loss(a, b)
    for _ in range(CURVE_ITERS):
        f = torch.exp(a + b * x)
        r = yz - f
        # Jacobian of f with respect to (a, b): f and f*x.
        j_a, j_b = f, f * x
        g = torch.stack([(w * r * j_a).sum(), (w * r * j_b).sum()])
        h = torch.stack(
            [
                torch.stack([(w * j_a * j_a).sum(), (w * j_a * j_b).sum()]),
                torch.stack([(w * j_a * j_b).sum(), (w * j_b * j_b).sum()]),
            ]
        )
        h = (
            h
            + 1e-12 * torch.eye(2, dtype=h.dtype, device=h.device) * h.diagonal().max()
        )
        step = torch.linalg.solve(h, g)
        scale = 1.0
        improved = False
        for _ in range(12):
            a_new = a + scale * step[0]
            b_new = (b + scale * step[1]).clamp(max=0.0)
            new = loss(a_new, b_new)
            if new < current:
                a, b, current, improved = a_new, b_new, new, True
                break
            scale *= 0.5
        if not improved or float(step.abs().max()) < 1e-9:
            break
    return torch.exp(a + b * x), float(a), float(b)


def estimate_sigma_d(
    delta_obs: torch.Tensor,
    sigma_diff: torch.Tensor,
    epsilon: torch.Tensor | None,
    d_star_sq: torch.Tensor,
    f_dark: torch.Tensor | None,
    fit_mask: torch.Tensor,
    *,
    delta_calc: torch.Tensor | None = None,
    gamma: float | None = None,
    shrink: bool | None = None,
    per_bin: int = PER_BIN,
    min_bins: int = MIN_BINS,
    min_per_bin: int = MIN_PER_BIN,
) -> SigmaDShells:
    """Per-shell expected difference power, with the dark-amplitude exponent and,
    given a difference model, its coupling and unexplained power.

    Runs under ``torch.no_grad()``. The working dtype is the wider of the configured
    float dtype and ``delta_obs.dtype``; results are cast back to ``delta_obs.dtype``.

    Parameters
    ----------
    delta_obs : torch.Tensor
        Signed observed differences ``F_light - F_dark``, shape ``(N,)``, on one common
        amplitude scale.
    sigma_diff : torch.Tensor
        Propagated uncertainty of ``delta_obs``, shape ``(N,)``, same units.
    epsilon : torch.Tensor or None
        Reflection multiplicity, shape ``(N,)``; ``None`` means ones.
    d_star_sq : torch.Tensor
        ``1/d**2`` per reflection, shape ``(N,)``, in A^-2.
    f_dark : torch.Tensor or None
        Dark amplitude for the power law, shape ``(N,)``. ``None`` disables the amplitude
        dependence (``gamma`` reported as the default with reason ``"no_f_dark"``).
    fit_mask : torch.Tensor
        Boolean ``(N,)``: which reflections enter the fit.
    delta_calc : torch.Tensor, optional
        Model differences ``|F_calc_light| - |F_calc_dark|``, shape ``(N,)``, on the
        observed scale. Enables ``alpha`` and ``beta_model``.
    gamma : float, optional
        Fix the exponent instead of fitting it.
    shrink : bool, optional
        Shrink the signed shell power toward a decaying curve in ``d*^2``; default the
        module setting.
    per_bin, min_bins, min_per_bin : int, optional
        Shell construction, see :func:`~._shells.equal_count_shells`.

    Returns
    -------
    SigmaDShells
        One frozen record of per-shell quantities plus counters.
    """
    device = delta_obs.device
    out_dtype = delta_obs.dtype
    dtype = _working_dtype(delta_obs)
    shrink = bool(SHRINK_ENABLED if shrink is None else shrink)
    if gamma is not None:
        lo, hi = GAMMA_BOUNDS
        if not (lo <= float(gamma) <= hi):
            raise ValueError(f"gamma must lie in {GAMMA_BOUNDS}, got {gamma}")

    with torch.no_grad():
        d_all = delta_obs.reshape(-1).to(dtype)
        s_all = sigma_diff.reshape(-1).to(dtype)
        x_all = d_star_sq.reshape(-1).to(dtype)
        e_all = (
            epsilon.reshape(-1).to(dtype)
            if epsilon is not None
            else torch.ones_like(d_all)
        )
        has_f = f_dark is not None
        f_all = f_dark.reshape(-1).to(dtype) if has_f else None
        has_model = delta_calc is not None
        c_all = delta_calc.reshape(-1).to(dtype) if has_model else None

        finite = (
            torch.isfinite(d_all)
            & torch.isfinite(s_all)
            & torch.isfinite(x_all)
            & torch.isfinite(e_all)
            & (s_all >= 0.0)
            & (e_all > 0.0)
        )
        if has_f:
            finite &= torch.isfinite(f_all)
        if has_model:
            finite &= torch.isfinite(c_all)
        fit = fit_mask.reshape(-1).to(torch.bool)
        usable = fit & finite
        n_dropped = int((fit & ~finite).sum())
        idx = torch.nonzero(usable, as_tuple=True)[0]
        n_fit = int(idx.numel())

        diagnostics = {
            "n_dropped": n_dropped,
            "n_fit": n_fit,
            "n_shell": 0,
            "n_s2_clamped": 0,
            "n_beta_clamped": 0,
            "n_f_floored": 0,
            "n_class_dropped": 0,
            "n_class_used": 0,
            "gamma_se": float("nan"),
            "gamma_at_bound": False,
            "gamma_reason": "degenerate",
        }
        if n_fit < 2:
            g = float(gamma) if gamma is not None else GAMMA_DEFAULT
            return _degenerate(d_all, g, has_model, diagnostics, out_dtype)

        order, seg, seg_lengths, n_bins = equal_count_shells(
            x_all[idx], per_bin=per_bin, min_bins=min_bins, min_per_bin=min_per_bin
        )
        sel = idx[order]
        d, s, e, x = d_all[sel], s_all[sel], e_all[sel], x_all[sel]
        counts = seg_lengths.to(dtype)
        d2e = d * d / e
        s2e = s * s / e

        B = segsum(d2e, seg_lengths) / counts
        S2 = segsum(s2e, seg_lengths) / counts
        Sigma_N_raw = (B - S2).clamp(min=0.0)
        n_s2_clamped = int((S2 >= B).sum())
        bin_dss = segsum(x, seg_lengths) / counts

        # --- dark-amplitude power law -------------------------------------------
        if has_f:
            f = f_all[sel]
            fbar = (segsum(f, seg_lengths) / counts).clamp(min=_TINY)
            fbar_h = fbar[seg]
            floor = F_FLOOR_FRAC * fbar_h
            n_f_floored = int((f < floor).sum())
            f_fl = torch.maximum(f, floor)
            log_ratio = torch.log(f_fl) - torch.log(fbar_h)
            bin_log_fbar = torch.log(fbar)
        else:
            n_f_floored = 0
            log_ratio = torch.zeros_like(d)
            bin_log_fbar = torch.zeros_like(bin_dss)

        if gamma is not None:
            g_used, g_se, at_bound, n_used, n_cls_dropped, reason = (
                float(gamma),
                float("nan"),
                False,
                0,
                0,
                "fixed",
            )
            fitted = False
        elif not has_f:
            g_used, g_se, at_bound, n_used, n_cls_dropped, reason = (
                GAMMA_DEFAULT,
                float("nan"),
                False,
                0,
                0,
                "no_f_dark",
            )
            fitted = False
        else:
            g_used, g_se, at_bound, n_used, n_cls_dropped, reason = _fit_gamma(
                d2e, s2e, log_ratio, seg, n_bins
            )
            fitted = reason == "fitted"

        if has_f:
            g_raw = torch.exp(g_used * log_ratio)
            Z = (segsum(g_raw, seg_lengths) / counts).clamp(min=_TINY)
            bin_log_z = torch.log(Z)
        else:
            bin_log_z = torch.zeros_like(bin_dss)

        # --- difference model ------------------------------------------------------
        if has_model:
            c = c_all[sel]
            Sigma_P = segsum(c * c / e, seg_lengths) / counts
            C = segsum(d * c / e, seg_lengths) / counts
            alpha = C / Sigma_P.clamp(min=_TINY)
        else:
            Sigma_P = torch.zeros_like(B)
            C = torch.zeros_like(B)
            alpha = torch.ones_like(B)

        # --- stability shrinkage of the signed power toward a decaying curve ---------
        # The signed B - S2 keeps every shell as evidence: a shell below zero by noise
        # says the power there is small, and it must count. var(B) is 2 B**2 / n for
        # Gaussian differences, so that is the sampling variance of each shell's value.
        signed = B - S2
        var_s = 2.0 * B * B / counts
        if shrink:
            curve, curve_a, curve_b = _fit_decay(signed, var_s, bin_dss)
            resid = signed - curve
            prec = 1.0 / var_s.clamp(min=_TINY)
            Q = (prec * resid * resid).sum()
            dof = float(max(int(signed.numel()) - 2, 1))
            c = (prec.sum() - (prec * prec).sum() / prec.sum().clamp(min=_TINY)).clamp(
                min=_TINY
            )
            # Q < dof means the shells scatter no more than their noise: take the curve.
            tau_sq = ((Q - dof) / c).clamp(min=0.0)
            shrink_w = var_s / (var_s + tau_sq).clamp(min=_TINY)
            Sigma_N = ((1.0 - shrink_w) * signed + shrink_w * curve).clamp(min=0.0)
        else:
            Sigma_N = Sigma_N_raw
            shrink_w, tau_sq = torch.zeros_like(B), B.new_zeros(())
            curve_a = curve_b = float("nan")
        Sigma_N = torch.where(torch.isfinite(Sigma_N), Sigma_N, torch.zeros_like(B))

        beta_model = (Sigma_N - alpha * alpha * Sigma_P).clamp(min=0.0)
        n_beta_clamped = int(((Sigma_N - alpha * alpha * Sigma_P) < 0.0).sum())
        all_zero = bool((Sigma_N <= 0.0).all())

        diagnostics.update(
            n_shell=int(n_bins),
            n_s2_clamped=n_s2_clamped,
            n_beta_clamped=n_beta_clamped,
            n_f_floored=n_f_floored,
            n_class_dropped=n_cls_dropped,
            n_class_used=n_used,
            gamma_se=g_se,
            gamma_at_bound=at_bound,
            gamma_reason=reason,
        )

        to = lambda t: t.to(out_dtype)
        return SigmaDShells(
            B=to(B),
            S2=to(S2),
            Sigma_N_raw=to(Sigma_N_raw),
            Sigma_N=to(Sigma_N),
            Sigma_P=to(Sigma_P),
            C=to(C),
            alpha=to(alpha),
            beta_model=to(beta_model),
            counts=to(counts),
            bin_dss=to(bin_dss),
            bin_log_fbar=to(bin_log_fbar),
            bin_log_z=to(bin_log_z),
            shrink_w=to(shrink_w),
            tau=float(tau_sq.clamp(min=0.0).sqrt()),
            curve_a=curve_a,
            curve_b=curve_b,
            gamma=float(g_used),
            gamma_fitted=fitted,
            has_model=has_model,
            degenerate=False,
            all_zero=all_zero,
            diagnostics=diagnostics,
        )


def sigma_d_per_reflection(
    shells: SigmaDShells,
    d_star_sq: torch.Tensor,
    epsilon: torch.Tensor | None,
    f_dark: torch.Tensor | None,
    sigma_diff: torch.Tensor,
) -> SigmaDEstimate:
    """Interpolate a shell estimate onto reflections and form the Wiener weights.

    Parameters
    ----------
    shells : SigmaDShells
        The shell estimate.
    d_star_sq : torch.Tensor
        ``1/d**2`` of the output reflections, shape ``(M,)``, in A^-2.
    epsilon : torch.Tensor or None
        Multiplicity of the output reflections, shape ``(M,)``; ``None`` means ones.
    f_dark : torch.Tensor or None
        Dark amplitude of the output reflections for the power law; reflections with a
        missing or non-finite value get a multiplier of one.
    sigma_diff : torch.Tensor
        Propagated uncertainty of the output differences, shape ``(M,)``. Non-finite
        entries give a weight of zero.

    Returns
    -------
    SigmaDEstimate
        Per-reflection, detached fields all of length ``M``.
    """
    with torch.no_grad():
        dtype = shells.Sigma_N.dtype
        grid = d_star_sq.reshape(-1).to(dtype)
        eps = (
            epsilon.reshape(-1).to(dtype)
            if epsilon is not None
            else torch.ones_like(grid)
        )
        sig = sigma_diff.reshape(-1).to(dtype)
        if shells.degenerate or shells.bin_dss.numel() == 0:
            sigma_n = torch.full_like(grid, float(shells.Sigma_N[0]))
            alpha = torch.full_like(grid, float(shells.alpha[0]))
            beta_model = torch.full_like(grid, float(shells.beta_model[0]))
            g = torch.ones_like(grid)
        else:
            log_sn = interp_in_dss(
                grid, shells.bin_dss, torch.log(shells.Sigma_N.clamp(min=_TINY))
            )
            sigma_n = torch.exp(log_sn)
            sigma_n = torch.where(
                sigma_n > 10.0 * _TINY, sigma_n, torch.zeros_like(sigma_n)
            )
            alpha = interp_in_dss(grid, shells.bin_dss, shells.alpha)
            log_bm = interp_in_dss(
                grid, shells.bin_dss, torch.log(shells.beta_model.clamp(min=_TINY))
            )
            beta_model = torch.exp(log_bm)
            beta_model = torch.where(
                beta_model > 10.0 * _TINY, beta_model, torch.zeros_like(beta_model)
            )
            if f_dark is not None and shells.gamma != 0.0:
                f = f_dark.reshape(-1).to(dtype)
                log_fbar = interp_in_dss(grid, shells.bin_dss, shells.bin_log_fbar)
                log_z = interp_in_dss(grid, shells.bin_dss, shells.bin_log_z)
                fbar = torch.exp(log_fbar)
                f_fl = torch.maximum(f, F_FLOOR_FRAC * fbar)
                g = torch.exp(shells.gamma * (torch.log(f_fl) - log_fbar) - log_z)
                g = torch.where(torch.isfinite(f) & (fbar > 0), g, torch.ones_like(g))
            else:
                g = torch.ones_like(grid)
        S = eps * sigma_n * g
        sigma_sq = sig * sig
        w = torch.where(
            torch.isfinite(sigma_sq),
            S / (S + sigma_sq).clamp(min=_TINY),
            torch.zeros_like(S),
        )
        return SigmaDEstimate(
            S=S.detach(),
            sigma_sq=sigma_sq.detach(),
            w=w.detach(),
            alpha=alpha.detach(),
            beta_model=(eps * beta_model * g).detach(),
            epsilon=eps.detach(),
            shells=shells,
        )


class SigmaDEstimator:
    """Lazy, cached difference-power estimate.

    Thin stateful wrapper around :func:`estimate_sigma_d` and
    :func:`sigma_d_per_reflection`: caches the detached estimate and re-estimates only
    after :meth:`reset`. **The owning target must call :meth:`reset` from its
    ``maintenance()`` hook**, otherwise the estimate is frozen for the whole run. Holds
    no tensors of its own beyond the cache, so it has no device to move.

    Parameters
    ----------
    config : SigmaDConfig, optional
        Exponent and shrinkage settings; the module defaults when omitted.
    """

    def __init__(self, config: SigmaDConfig | None = None):
        self.config = config if config is not None else SigmaDConfig()
        self._cache: SigmaDEstimate | None = None
        self._shells: SigmaDShells | None = None

    def reset(self) -> None:
        """Invalidate the cache so the next :meth:`get` re-estimates."""
        self._cache = None

    @property
    def shells(self) -> SigmaDShells | None:
        """Last shell estimate, for diagnostics; ``None`` until the first call."""
        return self._shells

    def get(
        self,
        delta_obs: torch.Tensor,
        sigma_diff: torch.Tensor,
        epsilon: torch.Tensor | None,
        d_star_sq: torch.Tensor,
        f_dark: torch.Tensor | None,
        fit_mask: torch.Tensor,
        *,
        delta_calc: torch.Tensor | None = None,
        target_dss: torch.Tensor | None = None,
        out_epsilon: torch.Tensor | None = None,
        out_f_dark: torch.Tensor | None = None,
        out_sigma_diff: torch.Tensor | None = None,
    ) -> SigmaDEstimate:
        """Return the cached-or-recomputed :class:`SigmaDEstimate`.

        The fit inputs may be a pooled, flattened set (several datasets end to end); the
        ``target_*`` / ``out_*`` arguments map the result onto another reflection list,
        defaulting to the fit inputs themselves.
        """
        if self._cache is not None:
            return self._cache
        shells = estimate_sigma_d(
            delta_obs,
            sigma_diff,
            epsilon,
            d_star_sq,
            f_dark,
            fit_mask,
            delta_calc=delta_calc,
            gamma=self.config.gamma,
            shrink=self.config.shrink,
        )
        self._shells = shells
        self._cache = sigma_d_per_reflection(
            shells,
            d_star_sq if target_dss is None else target_dss,
            epsilon if out_epsilon is None else out_epsilon,
            f_dark if out_f_dark is None else out_f_dark,
            sigma_diff if out_sigma_diff is None else out_sigma_diff,
        )
        return self._cache
