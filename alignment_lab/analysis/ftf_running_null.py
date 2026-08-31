"""Early stopping needs a cross-candidate contrast, so build the null as you go.

The per-candidate Z-scores fail as stopping criteria, and the separability table
says why: pooled over 750 placements, a wrong orientation's TFZ is *higher* than
truth's (median 3.15 vs 2.99). TFZ asks "is this translation better than the
other translations for this orientation" -- and a wrong orientation still has a
best translation that stands out of its own map. The contrast that carries the
signal is "is this orientation better than the other orientations", which no
single placement can answer.

But a sequential loop does not need all 25 placements to answer it -- only
enough of them to know what a wrong answer looks like on this structure. So:
place ``--burn-in`` candidates unconditionally, estimate the null from them with
a median/MAD (robust, because truth is often among the first few and a mean
would be dragged by it), then commit to the first candidate -- burn-in included
-- that sits ``--thr`` robust sigmas above it.

Sweeps threshold and burn-in over the recorded placements, in the FRF order a
live loop would walk. A cell that never crosses falls back to the argmax, which
is the no-early-stop behaviour, so an over-strict threshold costs placements
rather than answers.
"""

from __future__ import annotations

import argparse
import statistics as st
from collections import defaultdict


def load(paths):
    cells = defaultdict(list)
    for path in paths:
        for line in open(path):
            if not line.startswith("CAND"):
                continue
            d = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
            cells[(d["pdb"], int(d["trial"]))].append(
                dict(k=int(d["k"]), ang=float(d["ang"]),
                     truth=d["is_truth"] == "1", tf_corr=float(d["tf_corr"]),
                     tf_llg=float(d["tf_llg"]), tfz=float(d["tfz"]),
                     r=float(d["r"])))
    for v in cells.values():
        v.sort(key=lambda c: c["k"])
    return cells


def robust_z(x, ref, higher_is_better=True):
    """``x`` in MAD-sigmas above the centre of ``ref``. Sign-normalised."""
    med = st.median(ref)
    mad = st.median([abs(v - med) for v in ref])
    scale = 1.4826 * mad
    if scale < 1e-30:
        return float("inf") if x != med else 0.0
    z = (x - med) / scale
    return z if higher_is_better else -z


def simulate(cells, key, thr, burn, hi=True):
    hit = miss = exh_hit = exh_miss = 0
    placed, miss_ang = [], []
    for cs in cells.values():
        b = min(burn, len(cs))
        ref = [c[key] for c in cs[:b]]
        stopped = None
        # The burn-in candidates are tested too: truth is often among the first
        # few, and a loop that could not commit to one it had already placed
        # would pay the whole list on exactly the easy cases.
        for i, c in enumerate(cs):
            n_paid = max(b, i + 1)
            if i >= b:
                ref = [q[key] for q in cs[:i]]
            if robust_z(c[key], ref, hi) >= thr:
                stopped = (i, n_paid, c)
                break
        if stopped is None:
            placed.append(len(cs))
            best = (max if hi else min)(cs, key=lambda c: c[key])
            if best["truth"]:
                exh_hit += 1
            else:
                exh_miss += 1
                miss_ang.append(best["ang"])
        else:
            _, n_paid, c = stopped
            placed.append(n_paid)
            if c["truth"]:
                hit += 1
            else:
                miss += 1
                miss_ang.append(c["ang"])
    n = len(cells)
    return dict(n=n, hit=hit, miss=miss, exh_hit=exh_hit, exh_miss=exh_miss,
                ok=hit + exh_hit, med=st.median(placed),
                mean=sum(placed) / n, miss_ang=(st.median(miss_ang)
                                                if miss_ang else float("nan")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--key", default="tf_llg",
                    choices=["tf_llg", "tf_corr", "r", "tfz"])
    ap.add_argument("--burn-ins", default="3,5,8")
    ap.add_argument("--thresholds", default="3,5,8,10,15,20,30")
    args = ap.parse_args()
    hi = args.key != "r"

    cells = load(args.logs)
    print(f"# {len(cells)} cells x {st.median([len(v) for v in cells.values()]):.0f} "
          f"candidates, key={args.key} ({'higher' if hi else 'lower'} is better)")
    print(f"# OK = committed to a candidate within 8 deg; placements counted "
          f"include the burn-in\n")
    print(f"{'burn':>4s} {'thr':>4s} {'OK':>7s} {'stop_hit':>8s} {'stop_miss':>9s} "
          f"{'exh_hit':>7s} {'exh_miss':>8s} {'med_n':>5s} {'mean_n':>6s} "
          f"{'miss_ang':>8s}")
    best = None
    for burn in [int(x) for x in args.burn_ins.split(",")]:
        for thr in [float(x) for x in args.thresholds.split(",")]:
            r = simulate(cells, args.key, thr, burn, hi)
            print(f"{burn:>4d} {thr:>4g} {r['ok']:>4d}/{r['n']:<2d} "
                  f"{r['hit']:>8d} {r['miss']:>9d} {r['exh_hit']:>7d} "
                  f"{r['exh_miss']:>8d} {r['med']:>5.0f} {r['mean']:>6.1f} "
                  f"{r['miss_ang']:>8.1f}")
            if best is None or (r["ok"], -r["mean"]) > (best[0]["ok"], -best[0]["mean"]):
                best = (r, burn, thr)
    r, burn, thr = best
    print(f"\nper structure at burn={burn} thr={thr:g}:")
    for p in sorted({k[0] for k in cells}):
        sub = {k: v for k, v in cells.items() if k[0] == p}
        s = simulate(sub, args.key, thr, burn, hi)
        print(f"  {p:6s} OK {s['ok']}/{s['n']}  placed med={s['med']:.0f} "
              f"mean={s['mean']:.1f}"
              + ("" if s["miss_ang"] != s["miss_ang"]
                 else f"  miss_ang={s['miss_ang']:.0f} deg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
