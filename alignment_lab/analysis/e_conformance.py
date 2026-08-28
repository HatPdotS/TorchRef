"""Does an E convention do what we want E to do?

Phaser is no longer the specification for this part of the code, which means
comparison-debugging against a reference implementation is gone. What replaces it
is invariants: Wilson statistics, scale invariance, shrinkage monotonicity and
epsilon-correctness are true regardless of whose code computes them. This module
is that safety net, and it exists so conventions can be changed with something
other than an argument deciding whether the change was right.

Eight checks, of which two are the ones nothing in the tree currently makes:

* **absolute** unit mean, not merely a flat trend -- the rotation function is a
  correlation and does not care, but the rescore's LLG compares an observation
  against a predicted distribution and there is no free scale to cancel;
* **obs and calc on a common footing** -- the measured symptom of getting this
  wrong is an expected moving-model intensity with mean 2.14 where ~1 belongs.

The strongest check is not the mean but the **distribution**. Wilson statistics
predict ``|E|**2 ~ Exp(1)`` acentric and ``~ chi2_1`` centric at *every*
resolution, which catches a normaliser that is right on average and wrong in
shape. A mean cannot.

Deliberately a report rather than a gate: a convention may fail a property and
still rank truth better, and in that case the property tells us what the winner
is trading away rather than vetoing it.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

#: Wilson moment ratios <E^4>/<E^2>^2. Departures upward are the standard
#: twinning / tNCS diagnostic -- 2DQ6 reads about 5.5 acentric, which is how its
#: tNCS was originally identified.
IDEAL_MOMENT_RATIO = {"acentric": 2.0, "centric": 3.0}


def _deciles(s_mag: torch.Tensor, n: int = 10) -> torch.Tensor:
    """Equal-count resolution deciles, as an index per reflection."""
    order = torch.argsort(s_mag)
    out = torch.zeros_like(s_mag, dtype=torch.long)
    chunk = max(1, s_mag.numel() // n)
    for k in range(n):
        lo = k * chunk
        hi = (k + 1) * chunk if k < n - 1 else s_mag.numel()
        out[order[lo:hi]] = k
    return out


def _ks_uniform(sorted_u: torch.Tensor) -> float:
    """One-sample KS statistic of ``sorted_u`` against Uniform(0, 1)."""
    n = sorted_u.numel()
    if n < 2:
        return float("nan")
    i = torch.arange(1, n + 1, dtype=torch.float64, device=sorted_u.device)
    d_plus = (i / n - sorted_u).max()
    d_minus = (sorted_u - (i - 1) / n).max()
    return float(torch.maximum(d_plus, d_minus))


def _wilson_ks(E2: torch.Tensor, centric: torch.Tensor) -> dict:
    """KS of E**2 against its Wilson distribution, by centric class.

    Acentric ``E**2 ~ Exp(1)``, so ``1 - exp(-E**2)`` is uniform. Centric
    ``E**2 ~ chi2_1``, so ``erf(sqrt(E**2 / 2))`` is uniform. Both CDFs are
    closed-form, so no scipy dependency and no interpolation error.
    """
    out = {}
    for name, mask, cdf in (
        ("acentric", ~centric, lambda x: 1.0 - torch.exp(-x)),
        ("centric", centric, lambda x: torch.erf(torch.sqrt(x * 0.5))),
    ):
        v = E2[mask].to(torch.float64)
        if v.numel() < 50:
            out[name] = float("nan")
            continue
        u = cdf(v.clamp(min=0.0)).clamp(0.0, 1.0)
        out[name] = _ks_uniform(torch.sort(u).values)
    return out


def check_e_convention(
    cls,
    F: torch.Tensor,
    s_mag: torch.Tensor,
    centric: torch.Tensor,
    *,
    sig_F: Optional[torch.Tensor] = None,
    eps: Optional[torch.Tensor] = None,
    n_shells: int = 20,
    F_calc: Optional[torch.Tensor] = None,
    n_deciles: int = 10,
) -> dict:
    """Run every applicable property check on one convention.

    Takes the **class**, not an instance, because three of the checks need to
    construct it again on perturbed inputs -- rescaled F, raised sigmas -- and a
    convention that has already normalised its data cannot be asked those
    questions.
    """
    conv = cls(F, s_mag, centric, sig_F=sig_F, eps=eps, n_shells=n_shells)
    E = conv.E.to(torch.float64)
    E2 = E * E
    cen = conv.centric
    dec = _deciles(s_mag, n_deciles)
    rep: dict = {"name": cls.__name__ if hasattr(cls, "__name__") else str(cls)}

    # (1) stationarity + (2) absolute unit mean.
    rep["mean_E2"] = float(E2.mean())
    per_dec = torch.stack([
        E2[dec == k].mean() if bool((dec == k).any()) else torch.tensor(float("nan"))
        for k in range(n_deciles)
    ])
    rep["decile_mean_E2"] = [round(float(v), 4) for v in per_dec]
    rep["max_decile_dev"] = float((per_dec - 1.0).abs().max())
    # A trend, not just scatter: correlation of the decile mean with resolution.
    finite = torch.isfinite(per_dec)
    if int(finite.sum()) > 2:
        x = torch.arange(n_deciles, dtype=torch.float64)[finite]
        y = per_dec[finite].to(torch.float64)
        xc, yc = x - x.mean(), y - y.mean()
        denom = (xc.norm() * yc.norm()).clamp(min=1e-30)
        rep["decile_trend_r"] = float((xc * yc).sum() / denom)
    else:
        rep["decile_trend_r"] = float("nan")

    # (3) Wilson distribution, globally and worst-decile.
    rep["ks"] = _wilson_ks(E2, cen)
    worst = 0.0
    for k in range(n_deciles):
        m = dec == k
        if int(m.sum()) < 100:
            continue
        ks_k = _wilson_ks(E2[m], cen[m])
        for v in ks_k.values():
            if not math.isnan(v):
                worst = max(worst, v)
    rep["ks_worst_decile"] = worst

    # (4) moment ratios.
    rep["moment_ratio"] = {}
    for name, mask in (("acentric", ~cen), ("centric", cen)):
        v = E2[mask]
        if v.numel() < 50:
            rep["moment_ratio"][name] = float("nan")
            continue
        rep["moment_ratio"][name] = float(
            (v * v).mean() / v.mean().clamp(min=1e-30) ** 2
        )

    # (5) scale invariance: E must not depend on the units of F.
    devs = []
    for c in (1e-3, 1e3):
        E_c = cls(F * c, s_mag, centric,
                  sig_F=None if sig_F is None else sig_F * c,
                  eps=eps, n_shells=n_shells).E.to(torch.float64)
        scale = E.abs().max().clamp(min=1e-30)
        devs.append(float((E_c - E).abs().max() / scale))
    rep["scale_invariance_dev"] = max(devs)

    # (6) epsilon-correctness: axial reflections must not sit systematically
    #     above general ones. Only meaningful when eps was supplied and varies.
    if eps is not None:
        axial = eps > 1.0
        rep["eps_frac_gt1"] = float(axial.to(torch.float64).mean())
        if bool(axial.any()) and bool((~axial).any()):
            rep["eps_ratio"] = float(
                E2[axial].mean() / E2[~axial].mean().clamp(min=1e-30)
            )
        else:
            # All or none: no contrast to measure. All-axial means epsilon is
            # counting something it should not -- on a centred lattice the
            # rotation part repeats per centring coset, so `h.W == h` matches
            # once per coset for EVERY reflection.
            rep["eps_ratio"] = float("nan")
    else:
        rep["eps_frac_gt1"] = float("nan")
        rep["eps_ratio"] = float("nan")

    # (7) shrinkage monotonicity: raising sigma_F at fixed F must move E toward
    #     the shell mean, never away. Only conventions that read sig_F can.
    if getattr(cls, "uses_sigma_f", False) and sig_F is not None:
        loud = cls(F, s_mag, centric, sig_F=sig_F * 4.0, eps=eps,
                   n_shells=n_shells).E.to(torch.float64)
        ref = conv.sigma.sqrt().to(torch.float64)     # the shell scale E sits on
        moved_closer = (loud - ref).abs() <= (E - ref).abs() + 1e-9
        rep["shrinkage_frac_ok"] = float(moved_closer.to(torch.float64).mean())
    else:
        rep["shrinkage_frac_ok"] = float("nan")

    # (8) obs/calc common footing: normalise a calc set with the same convention
    #     and compare mean E**2. Both should sit at 1 if the convention puts them
    #     on a common scale; a mismatch is the eImove defect in miniature.
    if F_calc is not None:
        # A sigma_F-consuming convention cannot normalise a calc set -- there is
        # no measurement error to shrink -- so ask it which companion to use.
        calc_cls = cls.for_calc() if hasattr(cls, "for_calc") else cls
        rep["calc_via"] = calc_cls.__name__
        E_calc = calc_cls(F_calc, s_mag, centric, sig_F=None, eps=eps,
                          n_shells=n_shells).E.to(torch.float64)
        rep["mean_E2_calc"] = float((E_calc * E_calc).mean())
        rep["obs_calc_ratio"] = rep["mean_E2_calc"] / max(rep["mean_E2"], 1e-30)
    else:
        rep["calc_via"] = "-"
        rep["obs_calc_ratio"] = float("nan")
    return rep


def format_table(reports) -> str:
    """One row per convention, the columns that decide things."""
    head = (f"{'convention':18s} {'<E2>':>7s} {'maxdec':>7s} {'trend':>6s} "
            f"{'KS ac':>6s} {'KS cen':>7s} {'KSdec':>6s} {'m2 ac':>6s} "
            f"{'m2 cen':>7s} {'scale':>8s} {'e>1':>6s} {'eps':>6s} "
            f"{'shrink':>7s} {'o/c':>6s} {'calc via':>16s}")
    lines = [head, "-" * len(head)]
    for r in reports:
        lines.append(
            f"{r['name']:18s} {r['mean_E2']:>7.4f} {r['max_decile_dev']:>7.4f} "
            f"{r['decile_trend_r']:>+6.2f} "
            f"{r['ks']['acentric']:>6.4f} {r['ks']['centric']:>7.4f} "
            f"{r['ks_worst_decile']:>6.4f} "
            f"{r['moment_ratio']['acentric']:>6.3f} "
            f"{r['moment_ratio']['centric']:>7.3f} "
            f"{r['scale_invariance_dev']:>8.1e} "
            f"{r.get('eps_frac_gt1', float('nan')):>6.3f} "
            f"{r['eps_ratio']:>6.3f} "
            f"{r['shrinkage_frac_ok']:>7.3f} {r['obs_calc_ratio']:>6.3f} "
            f"{r.get('calc_via', '-'):>16s}"
        )
    lines.append("")
    lines.append("ideal: <E2>=1  maxdec=0  trend=0  KS small  m2 ac=2 cen=3  "
                 "scale=0  eps=1  shrink=1  o/c=1")
    lines.append("e>1 = fraction with epsilon>1; ~1.0 means epsilon is counting "
                 "centring cosets, not point-group stabilisers")
    return "\n".join(lines)
