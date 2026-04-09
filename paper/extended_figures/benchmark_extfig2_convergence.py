#!/usr/bin/env python3
"""Collect refinement convergence data for Extended Figure 2.

Extracts per-cycle R-work/R-free from existing TorchRef refinement_history.json
files (produced by run_pipeline.py) and Phenix refinement logs.

Output: data/extfig2_convergence.json

Usage:
    # Auto-select structures and extract from existing pipeline results:
    python benchmark_extfig2_convergence.py

    # Specify structures manually:
    python benchmark_extfig2_convergence.py --codes 138L 1A4E 3K7M

    # Point to a specific experiment directory:
    python benchmark_extfig2_convergence.py --experiment paper_v1
"""

import argparse
import json
import re
import sys
from pathlib import Path

import gemmi
import numpy as np

BASE = Path(__file__).resolve().parent
PAPER_ROOT = BASE.parent                                    # paper/
DATA_DIR = PAPER_ROOT / "data"                              # symlink → scientific_testing/data
PHENIX_DIR = PAPER_ROOT / "phenix_refinements"              # symlink → scientific_testing/.../refinements
EXPERIMENTS_DIR = PAPER_ROOT / "figure2_validation" / "experiments"
EXTFIG1_CSV = BASE / "data" / "extfig1_rfactor_by_resolution.csv"
OUT_JSON = BASE / "data" / "extfig2_convergence.json"

MACRO_CYCLES = 10


def auto_select_structures(n=5):
    """Pick structures spanning resolution and ΔR range."""
    import pandas as pd

    df = pd.read_csv(EXTFIG1_CSV)

    # Sort by resolution
    df = df.sort_values("d_min").reset_index(drop=True)

    picks = {}

    # High resolution (~1.2–1.5 Å)
    hi = df[(df["d_min"] >= 1.2) & (df["d_min"] <= 1.6)]
    if len(hi) > 0:
        # Pick one near median ΔR-free
        idx = (hi["delta_rfree"] - hi["delta_rfree"].median()).abs().idxmin()
        picks["high_res"] = df.loc[idx, "code"]

    # Medium resolution (~1.8–2.2 Å)
    med = df[(df["d_min"] >= 1.8) & (df["d_min"] <= 2.2)]
    if len(med) > 0:
        idx = (med["delta_rfree"] - med["delta_rfree"].median()).abs().idxmin()
        picks["med_res"] = df.loc[idx, "code"]

    # Low resolution (~2.6–3.0 Å)
    lo = df[(df["d_min"] >= 2.6) & (df["d_min"] <= 3.0)]
    if len(lo) > 0:
        idx = (lo["delta_rfree"] - lo["delta_rfree"].median()).abs().idxmin()
        picks["low_res"] = df.loc[idx, "code"]

    # TorchRef best (most negative ΔR-free)
    best = df.nsmallest(10, "delta_rfree")
    for _, row in best.iterrows():
        if row["code"] not in picks.values():
            picks["torchref_wins"] = row["code"]
            break

    # Phenix best (most positive ΔR-free)
    worst = df.nlargest(10, "delta_rfree")
    for _, row in worst.iterrows():
        if row["code"] not in picks.values():
            picks["phenix_wins"] = row["code"]
            break

    print("Auto-selected structures:")
    for label, code in picks.items():
        row = df[df["code"] == code].iloc[0]
        print(
            f"  {label:>15}: {code}  "
            f"(d_min={row['d_min']:.2f} Å, ΔR-free={row['delta_rfree']:+.2f} pp)"
        )

    return list(picks.values())


def extract_torchref_history(code, exp_dir):
    """Extract per-cycle R-factors from an existing refinement_history.json."""
    hist_path = exp_dir / "results" / code / "default" / "refinement_history.json"
    if not hist_path.exists():
        print(f"  History not found: {hist_path}")
        return None

    with open(hist_path) as f:
        h = json.load(f)

    history = h.get("history", {})
    # Find the refinement key (refinement_1, refinement_everything_1, etc.)
    ref_keys = [k for k in history if k.startswith("refinement")]
    if not ref_keys:
        print(f"  No refinement key found in history")
        return None

    cycles = history[ref_keys[-1]]

    # Initial R-factors from after_scaling of cycle 1
    init = cycles[0].get("after_scaling", {})
    rwork_trace = [float(init["rwork"])]
    rfree_trace = [float(init["rfree"])]

    for c in cycles:
        # End of cycle: try adp.after first, then after_refinement (older format)
        end = c.get("adp", {}).get("after", {})
        if "rwork" not in end:
            end = c.get("after_refinement", {})
        rw = end.get("rwork")
        rf = end.get("rfree")
        if rw is not None and rf is not None:
            rwork_trace.append(float(rw))
            rfree_trace.append(float(rf))

    # Get n_atoms and d_min from PDB/MTZ
    n_atoms = None
    d_min = None
    pdb_path = DATA_DIR / code / f"{code}_shaken.pdb"
    mtz_path = DATA_DIR / code / f"{code}.mtz"
    if pdb_path.exists():
        st = gemmi.read_structure(str(pdb_path))
        n_atoms = sum(
            1 for model in st for chain in model for res in chain for _ in res
        )
    if mtz_path.exists():
        d_min = gemmi.read_mtz_file(str(mtz_path)).resolution_high()

    return {
        "rwork": rwork_trace,
        "rfree": rfree_trace,
        "n_atoms": n_atoms,
        "d_min": d_min,
    }


def parse_phenix_log(code):
    """Parse per-cycle R-factors from Phenix refinement log.

    Strategy: the ``| r_work= ... |`` lines appear at the START of each
    macro cycle (reporting the state coming in). So the initial R-factors
    are on the first such line, and the end-of-cycle-N values are on the
    start-of-cycle-(N+1) line. The final cycle's result comes from the
    last such line in the file.
    """
    log_path = PHENIX_DIR / code / f"{code}_refined_001.log"
    if not log_path.exists():
        print(f"  Phenix log not found: {log_path}")
        return None

    text = log_path.read_text()

    # Match the boxed status lines: | r_work= 0.XXXX r_free= 0.XXXX ... |
    status_re = re.compile(
        r"\| r_work=\s*([\d.]+)\s+r_free=\s*([\d.]+)\s+coordinate error"
    )

    matches = status_re.findall(text)
    if len(matches) < 2:
        print(f"  Could not parse R-factors from Phenix log ({len(matches)} status lines)")
        return None

    # Each match is (rwork, rfree). First = initial, rest = start of each cycle
    # = end of previous cycle. Deduplicate consecutive identical values.
    rwork_trace = [float(matches[0][0])]
    rfree_trace = [float(matches[0][1])]

    for rw_s, rf_s in matches[1:]:
        rw, rf = float(rw_s), float(rf_s)
        if rw != rwork_trace[-1] or rf != rfree_trace[-1]:
            rwork_trace.append(rw)
            rfree_trace.append(rf)

    if len(rwork_trace) < 2:
        print(f"  Could not parse R-factors from Phenix log (no changes)")
        return None

    return {"rwork": rwork_trace, "rfree": rfree_trace}


def main():
    parser = argparse.ArgumentParser(description="Convergence data for ExtFig 2")
    parser.add_argument("--codes", nargs="+", help="PDB codes to use")
    parser.add_argument("--experiment", default="paper_v1",
                        help="Experiment name under figure2_validation/experiments/")
    args = parser.parse_args()

    exp_dir = EXPERIMENTS_DIR / args.experiment
    if not exp_dir.exists():
        print(f"Error: experiment directory not found: {exp_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Using experiment: {exp_dir}")

    if args.codes:
        codes = args.codes
    else:
        codes = auto_select_structures()

    results = {}

    for code in codes:
        print(f"\n{'='*60}")
        print(f"Structure: {code}")
        print(f"{'='*60}")

        entry = {"code": code}

        # TorchRef — extract from existing history JSON
        print("  Extracting TorchRef history...")
        tr = extract_torchref_history(code, exp_dir)
        if tr is not None:
            entry["torchref"] = tr
            entry["n_atoms"] = tr["n_atoms"]
            entry["d_min"] = tr["d_min"]
            print(
                f"  TorchRef: {len(tr['rwork'])} points, "
                f"R-work {tr['rwork'][0]:.4f} → {tr['rwork'][-1]:.4f}, "
                f"R-free {tr['rfree'][0]:.4f} → {tr['rfree'][-1]:.4f}"
            )

        # Phenix — parse from existing log
        print("  Parsing Phenix log...")
        ph = parse_phenix_log(code)
        if ph is not None:
            entry["phenix"] = ph
            print(
                f"  Phenix: {len(ph['rwork'])} points, "
                f"R-work {ph['rwork'][0]:.4f} → {ph['rwork'][-1]:.4f}, "
                f"R-free {ph['rfree'][0]:.4f} → {ph['rfree'][-1]:.4f}"
            )

        results[code] = entry

    output = {
        "benchmark": "extfig2_convergence",
        "experiment": args.experiment,
        "structures": results,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
