#!/usr/bin/env python
"""Mine the REFMAC 0-cycle logs for what systematically differs between the
torchref_norb and phenix_norb models (i.e. what tracks the residual R-free gap).

REFMAC re-scales every model identically, so any field it reports differently
for the two models reflects a real difference in the model, not in scaling.

Per log we parse:
  - Free R / Overall R               (REFMAC's own R-factors)
  - Overall   : scale, B             (overall B REFMAC must apply to the model)
  - Partial structure 1: scale, B    (REFMAC's own bulk-solvent fit)
  - M./S. chain bond B values rms    (local B roughness between bonded atoms)
  - low-res work R   (mean Rf_used over the 3 lowest-resolution shells)
  - high-res work R  (mean Rf_used over the 3 highest-resolution shells)

Then: medians per arm + correlation of (torchref - phenix) delta with the
per-structure free-R gap.
"""
import csv
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "runs/refmac_logs"
OUT = ROOT / "runs/metrics"

RE_FREE = re.compile(r"Free R factor\s*=\s*([\d.]+)")
RE_RWORK = re.compile(r"Overall R factor\s*=\s*([\d.]+)")
RE_OVERALL = re.compile(r"Overall\s*:\s*scale\s*=\s*([-\d.]+),\s*B\s*=\s*([-\d.]+)")
RE_PARTIAL = re.compile(r"Partial structure\s*1:\s*scale\s*=\s*([-\d.]+),\s*B\s*=\s*([-\d.]+)")
RE_MBOND = re.compile(r"M\. chain bond B values: refined atoms\s+\d+\s+([\d.]+)")
RE_SBOND = re.compile(r"S\. chain bond B values: refined atoms\s+\d+\s+([\d.]+)")


def parse(path):
    txt = path.read_text(errors="ignore")
    d = {}
    for key, rx in [("rfree", RE_FREE), ("rwork", RE_RWORK)]:
        m = rx.findall(txt)
        d[key] = float(m[-1]) if m else np.nan
    for key, rx in [("ov", RE_OVERALL), ("sol", RE_PARTIAL)]:
        m = rx.search(txt)
        d[f"{key}_scale"] = float(m.group(1)) if m else np.nan
        d[f"{key}_B"] = float(m.group(2)) if m else np.nan
    for key, rx in [("mbond", RE_MBOND), ("sbond", RE_SBOND)]:
        m = rx.search(txt)
        d[key] = float(m.group(1)) if m else np.nan
    # resolution-shell work R: rows after the "M(4SSQ/LL) NR_used" header,
    # columns: 4ssq NR %obs Fo Fc Rf_used WR ... ; take col index 5 (Rf_used).
    shells = []
    in_tab = False
    for line in txt.splitlines():
        if "M(4SSQ/LL) NR_used" in line:
            in_tab = True
            continue
        if in_tab:
            s = line.strip()
            if s.startswith("$$") or s.startswith("NR_free"):
                if shells:  # second $$ closes table
                    if s.startswith("$$"):
                        break
                continue
            parts = s.split()
            if len(parts) >= 6:
                try:
                    shells.append((float(parts[0]), float(parts[5])))
                except ValueError:
                    pass
    if shells:
        shells.sort()
        lo = np.mean([r for _, r in shells[:3]])   # lowest 4ssq = low resolution
        hi = np.mean([r for _, r in shells[-3:]])  # highest 4ssq = high resolution
        d["lowres_R"], d["hires_R"] = lo, hi
    else:
        d["lowres_R"] = d["hires_R"] = np.nan
    return d


def main():
    codes = sorted({p.name[:-len("_torchref_norb.log")]
                    for p in LOGS.glob("*_torchref_norb.log")})
    rows = []
    for c in codes:
        tl = LOGS / f"{c}_torchref_norb.log"
        pl = LOGS / f"{c}_phenix_norb.log"
        if not pl.exists():
            continue
        t, p = parse(tl), parse(pl)
        if np.isnan(t["rfree"]) or np.isnan(p["rfree"]):
            continue
        row = {"code": c, "gap": t["rfree"] - p["rfree"]}
        for k in ("rfree", "rwork", "ov_B", "sol_scale", "sol_B",
                  "mbond", "sbond", "lowres_R", "hires_R"):
            row[f"tr_{k}"] = t[k]
            row[f"px_{k}"] = p[k]
            row[f"d_{k}"] = t[k] - p[k]
        rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / "refmac_discrepancy.csv"
    with open(fn, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    arr = lambda k: np.array([r[k] for r in rows], float)
    gap = arr("gap")
    print(f"n={len(rows)} paired structures\n")
    print(f"median free-R gap (tr - px): {np.median(gap):+.4f}\n")
    print(f"{'field':12}{'tr median':>12}{'px median':>12}{'Δ median':>12}"
          f"{'corr(Δ,gap)':>14}")
    for k in ("ov_B", "sol_scale", "sol_B", "mbond", "sbond",
              "lowres_R", "hires_R"):
        t, p, d = arr(f"tr_{k}"), arr(f"px_{k}"), arr(f"d_{k}")
        m = ~(np.isnan(d) | np.isnan(gap))
        cc = np.corrcoef(d[m], gap[m])[0, 1] if m.sum() > 10 else np.nan
        print(f"{k:12}{np.nanmedian(t):12.3f}{np.nanmedian(p):12.3f}"
              f"{np.nanmedian(d):12.3f}{cc:14.3f}")
    print(f"\nwrote {fn}")


if __name__ == "__main__":
    main()
