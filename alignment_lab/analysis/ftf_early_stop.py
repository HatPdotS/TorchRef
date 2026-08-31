"""Would a sequential TFZ-gated search stop on the right orientation?

The discrimination panel ranks candidates against *each other*, which needs all
of them placed. A guidance loop wants the opposite: walk the FRF peaks in
descending order, place one at a time, and stop as soon as one is convincing --
paying for the whole list only on the cases that need it.

That needs an **absolute** criterion, computed from a single candidate. Two are
available per placement and both are already in the code:

``tfz``   the top translation peak measured against the spread of that
          orientation's own translation map (``TranslationPeak.sigma``, Phaser's
          TFZ).
``llgz``  the same for the LLG re-rank: best translation against the other 19.

Simulates the loop over the recorded placements at a range of thresholds. A run
that never crosses the threshold falls back to the argmax over all candidates
placed, which is the no-early-stop behaviour -- so the cost of a threshold set
too high is wasted work, not a failure, and the two are reported separately.
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
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
                     truth=d["is_truth"] == "1",
                     tfz=float(d["tfz"]), llgz=float(d["llgz"]),
                     tf_llg=float(d["tf_llg"]), r=float(d["r"])))
    for v in cells.values():
        v.sort(key=lambda c: c["k"])          # FRF descending order
    return cells


def simulate(cells, key, thr):
    """Walk each cell in FRF order, stop at the first candidate over ``thr``."""
    hit = miss = exh_hit = exh_miss = 0
    placed, miss_ang = [], []
    for cs in cells.values():
        for i, c in enumerate(cs):
            if c[key] >= thr:
                placed.append(i + 1)
                if c["truth"]:
                    hit += 1
                else:
                    miss += 1
                    miss_ang.append(c["ang"])
                break
        else:                                  # never convinced: rank them all
            placed.append(len(cs))
            best = max(cs, key=lambda c: c[key])
            if best["truth"]:
                exh_hit += 1
            else:
                exh_miss += 1
                miss_ang.append(best["ang"])
    n = len(cells)
    return dict(thr=thr, n=n, hit=hit, miss=miss, exh_hit=exh_hit,
                exh_miss=exh_miss, ok=hit + exh_hit,
                med_placed=st.median(placed), mean_placed=sum(placed) / n,
                med_miss_ang=st.median(miss_ang) if miss_ang else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--key", default="tfz", choices=["tfz", "llgz"])
    ap.add_argument("--thresholds", default="0,3,4,5,6,7,8,9,10,12,15,20,1e9")
    args = ap.parse_args()

    cells = load(args.logs)
    if not cells:
        print("no CAND lines found", file=sys.stderr)
        return 1
    n_cand = st.median([len(v) for v in cells.values()])
    print(f"# {len(cells)} cells, {n_cand:.0f} candidates each, key={args.key}")
    print(f"# a cell is OK if the loop commits to a candidate within 8 deg\n")
    print(f"{'thr':>6s} {'OK':>7s} {'stop_hit':>8s} {'stop_miss':>9s} "
          f"{'exh_hit':>7s} {'exh_miss':>8s} {'med_n':>6s} {'mean_n':>7s} "
          f"{'miss_ang':>8s}")
    for t in [float(x) for x in args.thresholds.split(",")]:
        r = simulate(cells, args.key, t)
        label = "none" if t > 1e8 else f"{t:g}"
        print(f"{label:>6s} {r['ok']:>4d}/{r['n']:<2d} {r['hit']:>8d} "
              f"{r['miss']:>9d} {r['exh_hit']:>7d} {r['exh_miss']:>8d} "
              f"{r['med_placed']:>6.0f} {r['mean_placed']:>7.1f} "
              f"{r['med_miss_ang']:>8.1f}")

    print("\nper structure at the best threshold by (OK, then fewest placed):")
    best = max((simulate(cells, args.key, t)
                for t in [float(x) for x in args.thresholds.split(",")]),
               key=lambda r: (r["ok"], -r["mean_placed"]))
    print(f"  thr={best['thr']:g}")
    for p in sorted({k[0] for k in cells}):
        sub = {k: v for k, v in cells.items() if k[0] == p}
        r = simulate(sub, args.key, best["thr"])
        print(f"  {p:6s} OK {r['ok']}/{r['n']}  placed med={r['med_placed']:.0f} "
              f"mean={r['mean_placed']:.1f}"
              + ("" if r['med_miss_ang'] != r['med_miss_ang']
                 else f"  miss_ang={r['med_miss_ang']:.1f} deg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
