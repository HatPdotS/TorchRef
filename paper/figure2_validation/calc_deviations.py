#!/das/work/p17/p17490/CONDA/torchref/bin/python -u
"""
Compute per-structure coordinate and ADP deviations for a validation2 experiment.

Compares the *original* (unshaken, converted) PDB to both the TorchRef-refined
and PHENIX-refined structures.  Outputs a CSV with columns matching what
make_publication_figure.py expects.

Usage
-----
    python calc_deviations.py <experiment_name>

Output is written to experiments/<name>/metrics/deviations.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from torchref.io.pdb import load_as_dataframe

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
EXPERIMENTS = BASE / "experiments"
SCIENTIFIC_TESTING = BASE.parent
DATA = SCIENTIFIC_TESTING / "data"
PHENIX = SCIENTIFIC_TESTING / "phenix_refinement" / "refinements"


def original_pdb_path(code):
    return DATA / code / f"{code}_converted.pdb"


def phenix_pdb_path(code):
    return PHENIX / code / f"{code}_refined_001.pdb"


def torchref_pdb_path(exp_dir, code, variant):
    return exp_dir / "results" / code / variant / "refined.pdb"


NAN_RESULT_KEYS = [
    "rmsd_torchref_phenix_mean", "rmsd_torchref_phenix_std",
    "rmsd_original_phenix_mean", "rmsd_original_phenix_std",
    "rmsd_original_torchref_mean", "rmsd_original_torchref_std",
    "adp_torchref_phenix_mean", "adp_torchref_phenix_std",
    "adp_original_phenix_mean", "adp_original_phenix_std",
    "adp_original_torchref_mean", "adp_original_torchref_std",
]


def _nan_row(code):
    return {"code": code, **{k: np.nan for k in NAN_RESULT_KEYS}}


def calc_deviations_for_code(code, exp_dir, variant):
    idx_cols = ["chainid", "resname", "resseq", "name", "altloc"]

    original_path = original_pdb_path(code)
    phenix_path = phenix_pdb_path(code)
    torchref_path = torchref_pdb_path(exp_dir, code, variant)

    for p in [original_path, phenix_path, torchref_path]:
        if not p.exists():
            return _nan_row(code)

    try:
        original = load_as_dataframe(str(original_path))
        original = original.loc[original["resname"] != "HOH"].set_index(idx_cols)

        phenix = load_as_dataframe(str(phenix_path))
        phenix = phenix.loc[phenix["resname"] != "HOH"].set_index(idx_cols)

        torchref = load_as_dataframe(str(torchref_path))
        torchref = torchref.loc[torchref["resname"] != "HOH"].set_index(idx_cols)

        joint = original.index.intersection(phenix.index).intersection(torchref.index)
        original = original.loc[joint]
        phenix = phenix.loc[joint]
        torchref = torchref.loc[joint]
    except Exception as e:
        print(f"  {code}: error loading PDBs: {e}")
        return _nan_row(code)

    try:
        xyz_orig = original[["x", "y", "z"]].values
        xyz_phen = phenix[["x", "y", "z"]].values
        xyz_tr = torchref[["x", "y", "z"]].values

        dist_tr_ph = np.sqrt(((xyz_tr - xyz_phen) ** 2).sum(axis=1))
        dist_orig_ph = np.sqrt(((xyz_orig - xyz_phen) ** 2).sum(axis=1))
        dist_orig_tr = np.sqrt(((xyz_orig - xyz_tr) ** 2).sum(axis=1))

        b_orig = np.log(np.clip(original["tempfactor"].values, 1, None))
        b_phen = np.log(np.clip(phenix["tempfactor"].values, 1, None))
        b_tr = np.log(np.clip(torchref["tempfactor"].values, 1, None))

        return {
            "code": code,
            "rmsd_torchref_phenix_mean": dist_tr_ph.mean(),
            "rmsd_torchref_phenix_std": dist_tr_ph.std(),
            "rmsd_original_phenix_mean": dist_orig_ph.mean(),
            "rmsd_original_phenix_std": dist_orig_ph.std(),
            "rmsd_original_torchref_mean": dist_orig_tr.mean(),
            "rmsd_original_torchref_std": dist_orig_tr.std(),
            "adp_torchref_phenix_mean": (b_tr - b_phen).mean(),
            "adp_torchref_phenix_std": (b_tr - b_phen).std(),
            "adp_original_phenix_mean": (b_orig - b_phen).mean(),
            "adp_original_phenix_std": (b_orig - b_phen).std(),
            "adp_original_torchref_mean": (b_orig - b_tr).mean(),
            "adp_original_torchref_std": (b_orig - b_tr).std(),
        }
    except Exception as e:
        print(f"  {code}: error computing deviations: {e}")
        return _nan_row(code)


def main():
    parser = argparse.ArgumentParser(description="Compute deviations for a validation2 experiment.")
    parser.add_argument("experiment", help="Experiment name (e.g. default_0226_1851)")
    args = parser.parse_args()

    exp_dir = EXPERIMENTS / args.experiment
    if not exp_dir.exists():
        sys.exit(f"Experiment directory not found: {exp_dir}")

    with open(exp_dir / "experiment.json") as f:
        meta = json.load(f)
    variant = meta["variant_labels"][0]
    structures = meta["structures"]

    print(f"Computing deviations for {len(structures)} structures "
          f"(experiment={args.experiment}, variant={variant})")

    results = []
    for i, code in enumerate(structures):
        row = calc_deviations_for_code(code, exp_dir, variant)
        results.append(row)
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(structures)}")

    df = pd.DataFrame(results)
    n_ok = df["rmsd_original_torchref_mean"].notna().sum()
    print(f"Done: {n_ok}/{len(structures)} structures with valid deviations")

    out_path = exp_dir / "metrics" / "deviations.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
