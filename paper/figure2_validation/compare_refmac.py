"""Paired, all-REFMAC comparison of fig2_ml_sigmaa_w10 vs other targets + Phenix.

Everything is REFMAC-recomputed (r_work/r_free/rmsBOND/rmsANGL) so the comparison
is independent of TorchRef's internal numbers. Pairs on the common set of PDB
codes present in every experiment.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
EXP = BASE / "experiments"

import sys

MAIN = sys.argv[1] if len(sys.argv) > 1 else "fig2_ml_sigmaa_default"
OTHERS = ["fig2_ml_noof", "fig2_bhatt_noof", "fig2_ml_sigmaa_noof", "fig2_ml_sigmaa_w10"]


def load(name, variant="default"):
    f = EXP / name / "metrics" / "refmac_metrics.csv"
    if not f.exists():
        print(f"  (missing {f})")
        return None
    df = pd.read_csv(f)
    df = df[df.variant == variant][["code", "r_work", "r_free", "rmsBOND", "rmsANGL"]]
    return df.set_index("code")


main = load(MAIN)
phenix = load(MAIN, variant="phenix")
others = {n: load(n) for n in OTHERS}
others = {n: d for n, d in others.items() if d is not None}

print(f"\n=== {MAIN}: {len(main)} structures (REFMAC-recomputed) ===")
print(f"median r_free={main.r_free.median():.4f}  r_work={main.r_work.median():.4f}  "
      f"rmsBOND={main.rmsBOND.median():.4f}  rmsANGL={main.rmsANGL.median():.3f}")

print(f"\n{'comparison':<28}{'N':>5}{'  med Rfree (this/other)':>26}"
      f"{'  ΔRfree':>9}{'  win%':>7}{'  med rmsBOND (this/other)':>28}")

def compare(other_name, other_df):
    common = main.index.intersection(other_df.index)
    a = main.loc[common]
    b = other_df.loc[common]
    d = a.r_free - b.r_free                      # negative = we're better
    win = float((d < 0).mean()) * 100
    print(f"{other_name:<28}{len(common):>5}"
          f"{a.r_free.median():>12.4f}/{b.r_free.median():.4f}"
          f"{d.median():>+9.4f}{win:>6.0f}%"
          f"{a.rmsBOND.median():>16.4f}/{b.rmsBOND.median():.4f}")

for n, d in others.items():
    compare(n, d)
if phenix is not None and len(phenix):
    compare("phenix", phenix)
