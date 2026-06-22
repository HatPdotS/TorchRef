#!/usr/bin/env python3
"""Collect resolution + R-factor-gap data for Extended Figure 2.

Per structure, the R-factor gap between the TorchRef model (locked default
arm, xray 1 / geometry 0.2 / adp 0.02) and each reference program (PHENIX and
REFMAC), all judged by the SAME independent scorer (phenix.model_vs_data) so the
comparison is bias-free, versus resolution.

  ΔR-free vs PHENIX = R-free(TorchRef) − R-free(PHENIX)   [percentage points]
  ΔR-free vs REFMAC = R-free(TorchRef) − R-free(REFMAC)
  (and likewise for R-work)

Source: figure2_alphafold_start/runs/metrics/fig_crossscore.csv
        (analysis/aggregate_crossscore.py); d_min from the TorchRef refined MTZ.

Output: data/exF2_rfactor_by_resolution.csv

Usage:
    ./.dev/bin/python paper/extended_figures/exF2/collect_exF2_data.py
"""
import sys
from pathlib import Path

import gemmi
import pandas as pd

BASE = Path(__file__).resolve().parent
AF_ROOT = BASE.parent.parent / "figure2_alphafold_start"
CROSSSCORE = AF_ROOT / "runs" / "metrics" / "fig_crossscore.csv"
TORCHREF_ARM = AF_ROOT / "runs" / "torchref_g0p2_a0p02"
SCORER = "phenix"                       # common independent scorer (main-figure scorer)
TORCHREF_ENGINE = "torchref_g0p2_a0p02"
PHENIX_ENGINE = "phenix"
REFMAC_ENGINE = "refmac"
OUT_CSV = BASE / "data" / "exF2_rfactor_by_resolution.csv"


def main():
    if not CROSSSCORE.exists():
        sys.exit(f"missing {CROSSSCORE}; run analysis/aggregate_crossscore.py first")
    df = pd.read_csv(CROSSSCORE)
    df = df[df["scorer"] == SCORER]

    def eng(name, suffix):
        return df[df.model_engine == name][["code", "r_work", "r_free"]].rename(
            columns={"r_work": f"rwork_{suffix}", "r_free": f"rfree_{suffix}"})

    tr = eng(TORCHREF_ENGINE, "torchref")
    ph = eng(PHENIX_ENGINE, "phenix")
    rm = eng(REFMAC_ENGINE, "refmac")
    merged = tr.merge(ph, on="code", how="inner").merge(rm, on="code", how="inner")
    print(f"Structures with TorchRef + PHENIX + REFMAC ({SCORER}-scored): {len(merged)}")

    # resolution from the TorchRef refined MTZ
    resolutions = {}
    missing = []
    for code in merged["code"]:
        mtz = TORCHREF_ARM / code / "refined.mtz"
        if not mtz.exists():
            missing.append(code)
            continue
        try:
            resolutions[code] = gemmi.read_mtz_file(str(mtz)).resolution_high()
        except Exception as e:
            print(f"  Warning: failed to read {code}.mtz: {e}", file=sys.stderr)
            missing.append(code)
    if missing:
        print(f"  Missing/failed MTZ: {len(missing)} ({missing[:5]}...)")

    merged["d_min"] = merged["code"].map(resolutions)
    merged = merged.dropna(subset=["d_min"])

    # ΔR (TorchRef − reference), in percentage points, for each reference program
    for ref in ("phenix", "refmac"):
        merged[f"delta_rwork_{ref}"] = (
            merged["rwork_torchref"] - merged[f"rwork_{ref}"]) * 100
        merged[f"delta_rfree_{ref}"] = (
            merged["rfree_torchref"] - merged[f"rfree_{ref}"]) * 100
    merged = merged.sort_values("d_min").reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False, float_format="%.6f")
    print(f"\nSaved {len(merged)} structures to {OUT_CSV}")
    print(f"  Resolution range: {merged['d_min'].min():.2f} – {merged['d_min'].max():.2f} Å")
    for ref in ("phenix", "refmac"):
        print(f"  vs {ref.upper():7s} median ΔR-work {merged[f'delta_rwork_{ref}'].median():+.2f} pp"
              f"   median ΔR-free {merged[f'delta_rfree_{ref}'].median():+.2f} pp")


if __name__ == "__main__":
    main()
