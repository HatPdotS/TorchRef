#!/usr/bin/env python
"""Is the DED correlation a fair way to compare an amplitude and an intensity target?

``torchref.validate-ded`` correlates ``WDFo`` against ``WDFc``, and ``WDFc`` is
``(|F_mixed| - |F_dark|) * w`` -- a weighted **amplitude** difference. That is, up to the
weight and the Fourier transform, exactly the residual ``CollectionDifferenceTarget``
minimises. Scoring an amplitude target on it is close to scoring it on its own objective,
so a win there is not evidence.

This computes the same correlation in **both** spaces, on the same reflections and the
same models:

    amplitude    obs  Fo_light - Fo_dark      calc  |Fc_light| - |Fc_dark|
    intensity    obs  Io_light - Io_dark      calc  |Fc_light|^2 - |Fc_dark|^2

If the ranking flips between the two, neither is decisive and the comparison has to be
made on something neither target optimises.

**READ THIS BEFORE USING THE FREE-SET NUMBERS.** They are reported for completeness and
they cannot answer the question. A difference feature is compact in real space and
therefore spread over ALL of reciprocal space, so a held-out subset of reflections does
not contain a reduced-precision version of it -- it does not contain it. Measured on this
pair: the same models score 0.52 on all reflections and 0.21 on the free 3.5%, while
restricting in the other domain goes the other way, 0.53 over the full cell to 0.85 on the
0.12% of voxels around the ligand. Localising helps in real space and destroys the signal
in reciprocal space.

So a reflection-wise hold-out validates a global scalar (R-free) and nothing local. To
cross-validate a local difference feature, hold out in the domain the feature lives in --
an omit refinement -- or use an independent dataset, or ground truth.

``F_calc`` is taken from each run's own results MTZ -- the scaled, mixed amplitudes that
run produced -- so no model is re-scaled here and each arm is scored on what it actually
built. Observed intensities come from one collection build shared by every arm.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import reciprocalspaceship as rs
import torch


def cc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def observed_intensities(dark_sf, light_sf, d_min, device):
    """Scaled ``(hkl, I_dark, I_light)`` from one collection build."""
    from torchref.cli.collection_difference_refine import setup_dataset_collection

    dc = setup_dataset_collection(dark_sf, light_sf, d_min, device)
    I_d, _ = dc["dark"].get_corrected_intensities()
    I_l, _ = dc["light"].get_corrected_intensities()
    hkl = dc.hkl.cpu().numpy()
    return hkl, I_d.cpu().numpy(), I_l.cpu().numpy()


_PAIRED = []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    fig4 = Path(__file__).resolve().parent / "figure4_difference_refinement"
    ap.add_argument("mtz", nargs="+", help="one results MTZ per arm (label=path accepted)")
    ap.add_argument("--dark-sf", default=str(fig4 / "data/8QL2-sf.cif"))
    ap.add_argument("--light-sf", default=str(fig4 / "data/7YYZ-light.mtz"))
    ap.add_argument("--dmin", type=float, default=2.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    hkl, Id_full, Il_full = observed_intensities(
        args.dark_sf, args.light_sf, args.dmin, torch.device(args.device))
    key = {tuple(h): i for i, h in enumerate(hkl)}

    rows = []
    for spec in args.mtz:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(spec).parent.name, spec
        ds = rs.read_mtz(path).reset_index()
        H = ds[["H", "K", "L"]].to_numpy()
        idx = np.array([key.get(tuple(h), -1) for h in H])
        ok = idx >= 0

        Fo_d = ds["Fo_dark"].to_numpy(float)
        Fo_l = ds["Fo_light"].to_numpy(float)
        Fc_d = ds["Fc_dark"].to_numpy(float)
        Fc_l = ds["Fc_light"].to_numpy(float)
        free = ds["FreeR_flag_light"].to_numpy() == 0

        Id = np.full(len(H), np.nan); Il = np.full(len(H), np.nan)
        Id[ok] = Id_full[idx[ok]]; Il[ok] = Il_full[idx[ok]]

        dFo, dFc = Fo_l - Fo_d, Fc_l - Fc_d          # amplitude difference
        dIo, dIc = Il - Id, Fc_l**2 - Fc_d**2        # intensity difference

        r = {"arm": label}
        sel_map = {"work": ~free & ok, "free": free & ok}
        for name, sel in sel_map.items():
            r[f"amp_{name}"] = cc(dFo[sel], dFc[sel])
            r[f"int_{name}"] = cc(dIo[sel], dIc[sel])
            r[f"n_{name}"] = int(sel.sum())
        rows.append(r)
        if len(_PAIRED) < 2:
            _PAIRED.append({"arm": label, "amp": (dFo, dFc), "int": (dIo, dIc),
                            "sel": sel_map})

    print()
    print("Difference-signal correlation, same reflections and models, two spaces")
    print("=" * 78)
    print(f"{'arm':16s} {'amp work':>10s} {'amp free':>10s} {'int work':>10s} "
          f"{'int free':>10s} {'n free':>8s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['arm']:16s} {r['amp_work']:10.4f} {r['amp_free']:10.4f} "
              f"{r['int_work']:10.4f} {r['int_free']:10.4f} {r['n_free']:8d}")
    print("-" * 78)
    # Paired bootstrap over reflections. A correlation is not a mean, so the
    # difference of two CCs has no closed-form error; resampling the SAME reflections
    # for both arms keeps the comparison paired, which matters because most of the
    # scatter is shared signal that cancels.
    if len(rows) >= 2 and _PAIRED:
        print()
        print("Paired bootstrap on the CC difference (4000 resamples over reflections)")
        print("-" * 78)
        a, b = _PAIRED[0], _PAIRED[1]
        rng = np.random.default_rng(0)
        for sp, (oa, ca, ob, cb) in (("amp", a["amp"] + b["amp"]),
                                     ("int", a["int"] + b["int"])):
            for st in ("work", "free"):
                m = a["sel"][st]
                ia, ja = oa[m], ca[m]
                ib, jb = ob[m], cb[m]
                keep = np.isfinite(ia) & np.isfinite(ja) & np.isfinite(ib) & np.isfinite(jb)
                ia, ja, ib, jb = ia[keep], ja[keep], ib[keep], jb[keep]
                n = len(ia)
                d = np.corrcoef(ia, ja)[0, 1] - np.corrcoef(ib, jb)[0, 1]
                boot = np.empty(4000)
                for k in range(4000):
                    s_ = rng.integers(0, n, n)
                    boot[k] = (np.corrcoef(ia[s_], ja[s_])[0, 1]
                               - np.corrcoef(ib[s_], jb[s_])[0, 1])
                lo, hi = np.percentile(boot, [2.5, 97.5])
                flag = "" if lo <= 0 <= hi else "   <-- CI excludes 0"
                print(f"  {sp}_{st:4s}  d = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                      f"  n={n}{flag}")
        print(f"\n  positive => {_PAIRED[0]['arm']} predicts the difference better")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
