#!/usr/bin/env python3
"""Aggregate the 10x10 log weight grid (submit_weight_grid.py).

For each wgrid arm (from wgrid_manifest.csv) parse every code's REFMAC validate.log
for R-factors and geometry RMS/sigma, and report per-structure:
  r_work, r_free, bond_rmsz, angle_rmsz, mc_b_rmsz (main-chain bond B-value RMSZ).
Writes runs/metrics/weight_grid.csv and prints per-cell coverage.

Usage:
    ./.dev/bin/python analysis/aggregate_weight_grid.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_af_pipeline as P  # noqa: E402
from aggregate_figure_metrics import RE_RWORK, RE_RFREE, parse_geometry  # noqa: E402

MANIFEST = P.RUNS / "metrics" / "wgrid_manifest.csv"
OUT = P.RUNS / "metrics" / "weight_grid.csv"


def ratio(num, den):
    return (num / den) if (num is not None and den) else None


def main():
    if not MANIFEST.exists():
        sys.exit(f"missing {MANIFEST}; run submit_weight_grid.py first")
    manifest = list(csv.DictReader(open(MANIFEST)))

    rows = []
    for m in manifest:
        arm = m["arm"]
        arm_dir = P.RUNS / arm
        if not arm_dir.is_dir():
            continue
        for code_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
            vlog = code_dir / "validate.log"
            if not vlog.exists():
                continue
            txt = vlog.read_text(errors="replace")
            mw, mf = RE_RWORK.search(txt), RE_RFREE.search(txt)
            geom = parse_geometry(vlog)
            if not (mw and mf and geom):
                continue
            rows.append({
                "arm": arm, "gi": m["gi"], "ai": m["ai"],
                "geometry": m["geometry"], "adp": m["adp"], "code": code_dir.name,
                "r_work": float(mw.group(1)), "r_free": float(mf.group(1)),
                "bond_rmsz": ratio(geom.get("rmsBOND"), geom.get("sigBOND")),
                "angle_rmsz": ratio(geom.get("rmsANGL"), geom.get("sigANGL")),
                "mc_b_rmsz": ratio(geom.get("rmsB_mc_bond"), geom.get("sigB_mc_bond")),
            })

    cols = ["arm", "gi", "ai", "geometry", "adp", "code", "r_work", "r_free",
            "bond_rmsz", "angle_rmsz", "mc_b_rmsz"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    per_cell = {}
    for r in rows:
        per_cell.setdefault(r["arm"], 0)
        per_cell[r["arm"]] += 1
    ncells = len(per_cell)
    counts = sorted(per_cell.values())
    print(f"wrote {OUT}  ({len(rows)} rows, {ncells}/{len(manifest)} cells populated)")
    if counts:
        print(f"  per-cell coverage: min={counts[0]} median={counts[len(counts)//2]} "
              f"max={counts[-1]}")
    have_mcb = sum(1 for r in rows if r["mc_b_rmsz"] is not None)
    print(f"  rows with MC-B RMSZ: {have_mcb}/{len(rows)}")


if __name__ == "__main__":
    main()
