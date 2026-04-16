#!/usr/bin/env python3
"""Collect resolution + R-factor data for Extended Figure 2.

Reads d_min from MTZ headers via gemmi, merges with REFMAC5-validated
R-factors from the 1,000-structure benchmark.

Output: data/exF2_rfactor_by_resolution.csv
"""

import json
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PAPER_ROOT = BASE.parent.parent                              # paper/
VALIDATION = PAPER_ROOT / "figure2_validation"
REFMAC_CSV = VALIDATION / "data" / "refmac_metrics.csv"
STRUCTURES_JSON = VALIDATION / "structures.json"
DATA_DIR = PAPER_ROOT / "data"                              # symlink → scientific_testing/data
OUT_CSV = BASE / "data" / "exF2_rfactor_by_resolution.csv"


def main():
    with open(STRUCTURES_JSON) as f:
        codes = json.load(f)
    print(f"Structures in benchmark: {len(codes)}")

    df = pd.read_csv(REFMAC_CSV)

    # Pivot to get torchref and phenix side by side
    default = df[df["variant"] == "default"][["code", "r_work", "r_free"]].rename(
        columns={"r_work": "rwork_torchref", "r_free": "rfree_torchref"}
    )
    phenix = df[df["variant"] == "phenix"][["code", "r_work", "r_free"]].rename(
        columns={"r_work": "rwork_phenix", "r_free": "rfree_phenix"}
    )
    merged = default.merge(phenix, on="code", how="inner")
    print(f"Structures with both TorchRef + Phenix R-factors: {len(merged)}")

    # Read resolution from MTZ headers
    resolutions = {}
    n_atoms_map = {}
    missing = []
    for code in merged["code"]:
        mtz_path = DATA_DIR / code / f"{code}.mtz"
        pdb_path = DATA_DIR / code / f"{code}_shaken.pdb"
        if not mtz_path.exists():
            missing.append(code)
            continue
        try:
            mtz = gemmi.read_mtz_file(str(mtz_path))
            resolutions[code] = mtz.resolution_high()
        except Exception as e:
            print(f"  Warning: failed to read {code}.mtz: {e}", file=sys.stderr)
            missing.append(code)

        # Also grab n_atoms from PDB if available
        if pdb_path.exists():
            try:
                st = gemmi.read_structure(str(pdb_path))
                n_atoms_map[code] = sum(
                    1 for model in st for chain in model for res in chain for _ in res
                )
            except Exception:
                pass

    if missing:
        print(f"  Missing/failed MTZ: {len(missing)} ({missing[:5]}...)")

    merged["d_min"] = merged["code"].map(resolutions)
    merged["n_atoms"] = merged["code"].map(n_atoms_map)
    merged = merged.dropna(subset=["d_min"])

    # Compute deltas (TorchRef minus Phenix, in percentage points)
    merged["delta_rwork"] = (merged["rwork_torchref"] - merged["rwork_phenix"]) * 100
    merged["delta_rfree"] = (merged["rfree_torchref"] - merged["rfree_phenix"]) * 100

    merged = merged.sort_values("d_min").reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False, float_format="%.6f")
    print(f"\nSaved {len(merged)} structures to {OUT_CSV}")
    print(f"  Resolution range: {merged['d_min'].min():.2f} – {merged['d_min'].max():.2f} Å")
    print(f"  Median ΔR-work: {merged['delta_rwork'].median():.2f} pp")
    print(f"  Median ΔR-free: {merged['delta_rfree'].median():.2f} pp")


if __name__ == "__main__":
    main()
