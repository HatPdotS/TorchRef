#!/usr/bin/env python
"""What does the data-to-data scale fit cost, and which objective generalises?

``DatasetCollection.scale()`` puts one dataset onto another. There is no model on either
side, so there is no model error for a sigma_A or Rice likelihood to account for, and the
only real choices are the weighting and which reflections the fit is allowed to see.

The question it answers: **the leak.** The fit used to mask with
``ReflectionData.masks()`` -- validity only, with no work/free notion -- so the free
reflections went into the scale parameters, upstream of every target and therefore
upstream of every free-set number the pipeline reports. How much did that buy it, and
does removing it move the fitted scale?

Scored on reflections the fit never saw, under two yardsticks applied identically to
every arm, so the comparison is not circular:

    R_data = sum|F - F_ref| / sum F_ref          scale-free and interpretable
    chi2   = mean[(F - F_ref)**2 / (s**2 + s_ref**2)]   is the disagreement within error?

``chi2`` is the one that can separate them: ``ls`` is entitled to win on ``R_data``, which
is what it optimises up to a constant.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch


def build(dark_sf, light_sf, d_min, device):
    """The figure-4 pair, loaded exactly as the difference CLI loads it, unscaled."""
    from torchref.cli.collection_difference_refine import setup_dataset_collection

    return setup_dataset_collection(dark_sf, light_sf, d_min, device)


def _reset(dc):
    """Zero every fitted scale parameter and drop the corrected caches."""
    for _, ds in dc:
        with torch.no_grad():
            if getattr(ds, "log_scale", None) is not None:
                ds.log_scale.zero_()
            if getattr(ds, "U_aniso", None) is not None:
                ds.U_aniso.zero_()
        ds._corrected_fp = None
        ds._corrected_cache = None
        ds._corrected_I_fp = None
        ds._corrected_I_cache = None


def legacy_scale(dc):
    """The pre-fix fit, copied verbatim for comparison: validity masks, unnormalised.

    A deliberate duplicate rather than a flag on the library method -- the point is to
    measure what the old behaviour did, not to keep it selectable.
    """
    ref_ds = dc._datasets[dc._reference_dataset]
    to_scale = [ds for name, ds in dc if name != dc._reference_dataset]
    params = [p for data in to_scale for p in data.parameters()]
    [p.requires_grad_(True) for p in params]
    opt = torch.optim.LBFGS(params, max_iter=100, line_search_fn="strong_wolfe")
    ref_mask = ref_ds.masks()
    ds_masks = [ds.masks() for ds in to_scale]

    def closure():
        opt.zero_grad()
        loss = 0.0
        ref_F, _ = ref_ds.get_corrected_data()
        for ds, m in zip(to_scale, ds_masks):
            F, _ = ds.get_corrected_data()
            cm = m & ref_mask
            loss = loss + torch.sum((F[cm] - ref_F[cm]) ** 2)
        loss.backward()
        return loss

    for _ in range(10):
        opt.step(closure)
    [p.requires_grad_(False) for p in params]


def per_refl_chi2(dc, subset):
    """Per-reflection chi-square contributions on ``subset``, in a fixed order.

    Returned rather than reduced so arms can be compared **paired**. The mask depends
    only on the flags and the validity masks, never on the fitted scale, so the same
    reflection sits at the same index in every arm -- which is what makes the pairing
    valid. Unpaired means cannot resolve this comparison: the two datasets are different
    structures, so most of chi2 is real difference and it cancels only when paired.
    """
    from torchref.base.targets.xray_likelihoods import floor_sigma_obs

    ref_name = dc._reference_dataset
    other = [n for n, _ in dc if n != ref_name][0]
    ref_ds, ds = dc[ref_name], dc[other]
    with torch.no_grad():
        ref_F, ref_s = ref_ds.get_corrected_data()
        F, s = ds.get_corrected_data()
        m = getattr(ds, subset).mask & getattr(ref_ds, subset).mask
        so, sr = floor_sigma_obs(s[m]), floor_sigma_obs(ref_s[m])
        return (((F[m] - ref_F[m]) ** 2) / (so**2 + sr**2)).cpu()


def paired_ci(a, b, n_boot=4000, seed=0):
    """Bootstrap CI on ``mean(a - b)`` over reflections. Positive favours ``b``."""
    d = (a - b).numpy()
    g = torch.Generator().manual_seed(seed)
    n = len(d)
    idx = torch.randint(0, n, (n_boot, n), generator=g).numpy()
    means = d[idx].mean(axis=1)
    lo, hi = sorted(means)[int(0.025 * n_boot)], sorted(means)[int(0.975 * n_boot)]
    return float(d.mean()), float(lo), float(hi)


def score(dc, subset):
    """``(R_data, chi2, n)`` between the two datasets on one held-out subset."""
    from torchref.base.targets.xray_likelihoods import floor_sigma_obs

    names = [n for n, _ in dc]
    ref_name = dc._reference_dataset
    other = [n for n in names if n != ref_name][0]
    ref_ds, ds = dc[ref_name], dc[other]

    with torch.no_grad():
        ref_F, ref_s = ref_ds.get_corrected_data()
        F, s = ds.get_corrected_data()
        m = getattr(ds, subset).mask & getattr(ref_ds, subset).mask
        fo, fr = F[m], ref_F[m]
        so, sr = floor_sigma_obs(s[m]), floor_sigma_obs(ref_s[m])
        r = float((fo - fr).abs().sum() / fr.abs().sum().clamp(min=1e-30))
        chi2 = float((((fo - fr) ** 2) / (so**2 + sr**2)).mean())
    return r, chi2, int(m.sum())


def fitted(dc):
    other = [n for n, _ in dc if n != dc._reference_dataset][0]
    ds = dc[other]
    ls = float(ds.log_scale.detach().reshape(-1)[0])
    u = ds.U_aniso.detach().reshape(-1).tolist()
    return ls, u


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    fig4 = Path(__file__).resolve().parent / "figure4_difference_refinement"
    ap.add_argument("--dark-sf", default=str(fig4 / "data/8QL2-sf.cif"))
    ap.add_argument("--light-sf", default=str(fig4 / "data/7YYZ-light.mtz"))
    ap.add_argument("--dmin", type=float, default=2.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device)
    dc = build(args.dark_sf, args.light_sf, args.dmin, dev)

    # A sigma-weighted arm was measured here and removed: it scored slightly better on
    # held-out reflections for this one pair, but inverse-variance weighting collapses on
    # a scale fit -- down-weighting the weak shells is what lets the scale run away in
    # them -- and that failure mode was found across a panel. Unit weights stand.
    arms = [
        ("legacy (validity mask, unnormalised)", lambda: legacy_scale(dc)),
        ("ls     (work set, normalised)", lambda: dc.scale()),
    ]

    rows = []
    per_refl = {}
    for label, fit in arms:
        _reset(dc)
        fit()
        ls, u = fitted(dc)
        rw, cw, nw = score(dc, "work")
        rf, cf, nf = score(dc, "free")
        per_refl[label] = {
            "free": per_refl_chi2(dc, "free"),
            "work": per_refl_chi2(dc, "work"),
        }
        rows.append(
            dict(arm=label, log_scale=ls, U_aniso=u,
                 R_work=rw, chi2_work=cw, n_work=nw,
                 R_free=rf, chi2_free=cf, n_free=nf)
        )

    print()
    print("Data-to-data scale fit: fitted on WORK, scored on the held-out FREE set")
    print("=" * 86)
    print(f"{'arm':38s} {'log_scale':>10s} {'R_work':>8s} {'R_free':>8s} "
          f"{'chi2_work':>10s} {'chi2_free':>10s}")
    print("-" * 86)
    for r in rows:
        print(f"{r['arm']:38s} {r['log_scale']:10.5f} {r['R_work']:8.5f} "
              f"{r['R_free']:8.5f} {r['chi2_work']:10.3f} {r['chi2_free']:10.3f}")
    print("-" * 86)
    print(f"n_work={rows[0]['n_work']}  n_free={rows[0]['n_free']}")
    print()
    base = rows[0]
    for r in rows[1:]:
        d = r["log_scale"] - base["log_scale"]
        print(f"{r['arm']:38s} d(log_scale) vs legacy = {d:+.6f} "
              f"({100*(math.exp(d)-1):+.3f}% in scale)")

    # --- the comparison that can actually resolve this: paired, per reflection ------
    labels = [lbl for lbl, _ in arms]
    print()
    print("Paired per-reflection chi2 difference (positive => the SECOND arm is better)")
    print("=" * 86)
    print(f"{'comparison':52s} {'set':6s} {'mean d':>10s} {'95% CI':>22s}")
    print("-" * 86)
    pairs = [(labels[0], labels[1])]
    paired = []
    for a, b in pairs:
        for subset in ("work", "free"):
            m, lo, hi = paired_ci(per_refl[a][subset], per_refl[b][subset])
            sig = "" if lo <= 0.0 <= hi else "  <-- CI excludes 0"
            name = f"{a.split('(')[0].strip()} vs {b.split('(')[0].strip()}"
            print(f"{name:52s} {subset:6s} {m:+10.5f} [{lo:+.5f}, {hi:+.5f}]{sig}")
            paired.append(dict(a=a, b=b, subset=subset, mean=m, lo=lo, hi=hi))
    print("-" * 86)

    if args.out:
        Path(args.out).write_text(json.dumps({"arms": rows, "paired": paired}, indent=2))
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
