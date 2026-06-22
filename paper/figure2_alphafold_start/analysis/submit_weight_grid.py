#!/usr/bin/env python3
"""Fine log-spaced loss-weight screen on AF-start: xray fixed=1, vary geometry & adp.

Default is a 10x10 log grid of geometry and adp, each in [1e-3, 1] (everything
expressed relative to the x-ray data term, which is held at 1), on the first 50
solved structures. Each structure is refined exactly like the canonical arm
(``--xray-mode ml -n 10 --mode separate``) and validated by a REFMAC 0-cycle run,
so each ``validate.log`` carries R-factors + geometry RMS/sigma (incl. main-chain
B-value restraints) needed for the validation-landscape heatmaps.

Arms are named by GRID INDEX (``wgrid_g{gi}_a{ai}``) to avoid float round-trip;
the exact (geometry, adp) per arm is written to
``runs/metrics/wgrid_manifest.csv``.

Usage
-----
    ./.dev/bin/python analysis/submit_weight_grid.py --dry-run
    ./.dev/bin/python analysis/submit_weight_grid.py            # submit
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_af_pipeline as P  # noqa: E402

MANIFEST = P.RUNS / "metrics" / "wgrid_manifest.csv"


def build_script(arm, code, pdb, mtz, outdir, n_cycles, mem, weights):
    wjson = json.dumps(weights)
    return f"""#!/bin/bash
#SBATCH --job-name=af_{arm}_{code}
#SBATCH --partition=hour
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem={mem}
#SBATCH --output={outdir / 'out.log'}
#SBATCH --error={outdir / 'error.log'}

set -e
export TORCHREF_NUM_THREADS=4

{P.PYTHON} -u {P.REFINE_SCRIPT} \\
    -m {pdb} -sf {mtz} -o {outdir} \\
    -n {n_cycles} \\
    --mode separate \\
    --xray-mode ml \\
    --sigma-m-scale 1.0 \\
    --weights '{wjson}'

# REFMAC 0-cycle validation (validate.log -> R-factors + geometry RMS/sigma)
source {P.CCP4_SETUP}
TEMP_DIR=/tmp/validate_{arm}_{code}_${{SLURM_JOB_ID}}
mkdir -p $TEMP_DIR && cd $TEMP_DIR
export CCP4_SCR=$TEMP_DIR
cp {outdir / 'refined.pdb'} input.pdb
cp {mtz} input.mtz
refmac5 HKLIN input.mtz HKLOUT output.mtz XYZIN input.pdb XYZOUT output.pdb \\
    > {outdir / 'validate.log'} 2>&1 << EOF
NCYCLES 0
MAKE HYDR NO
END
EOF
cd / && rm -rf $TEMP_DIR
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-geom", type=int, default=10)
    ap.add_argument("--n-adp", type=int, default=10)
    ap.add_argument("--geom-range", type=float, nargs=2, default=[1e-3, 1.0])
    ap.add_argument("--adp-range", type=float, nargs=2, default=[1e-3, 1.0])
    ap.add_argument("--limit", type=int, default=50,
                    help="Use the first N solved codes (default 50).")
    ap.add_argument("--codes", nargs="+", default=None)
    ap.add_argument("--n-cycles", type=int, default=10)
    ap.add_argument("--mem", default="8G")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    geoms = np.logspace(np.log10(args.geom_range[0]), np.log10(args.geom_range[1]),
                        args.n_geom)
    adps = np.logspace(np.log10(args.adp_range[0]), np.log10(args.adp_range[1]),
                       args.n_adp)

    codes = args.codes if args.codes else P.load_solved_codes()
    if args.limit:
        codes = codes[:args.limit]

    # Manifest: arm -> (gi, ai, geometry, adp), exact weights for aggregation.
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "gi", "ai", "geometry", "adp"])
        for gi, g in enumerate(geoms):
            for ai, a in enumerate(adps):
                w.writerow([f"wgrid_g{gi}_a{ai}", gi, ai, f"{g:.6g}", f"{a:.6g}"])

    print(f"Log weight grid (xray fixed = 1):")
    print(f"  geometry [{args.n_geom}]: {', '.join(f'{g:.4g}' for g in geoms)}")
    print(f"  adp      [{args.n_adp}]: {', '.join(f'{a:.4g}' for a in adps)}")
    print(f"  {args.n_geom * args.n_adp} combos x {len(codes)} structures = "
          f"{args.n_geom * args.n_adp * len(codes)} jobs max")
    print(f"  manifest: {MANIFEST}\n")

    first, total_sub = True, 0
    for gi, g in enumerate(geoms):
        for ai, a in enumerate(adps):
            arm = f"wgrid_g{gi}_a{ai}"
            arm_dir = P.RUNS / arm
            tmp = arm_dir / "tmp_scripts"
            weights = {"xray": 1, "geometry": float(g), "adp": float(a)}
            sub = skip = miss = 0
            for code in codes:
                pdb, mtz = P.PLACED / f"{code}_af.pdb", P._mtz(code)
                if not pdb.exists() or not mtz.exists():
                    miss += 1
                    continue
                outdir = arm_dir / code
                if P._refmac_complete(outdir / "validate.log") and not args.force:
                    skip += 1
                    continue
                if not args.dry_run:
                    outdir.mkdir(parents=True, exist_ok=True)
                script = build_script(arm, code, pdb, mtz, outdir, args.n_cycles,
                                      args.mem, weights)
                if first and args.dry_run:
                    print("── example sbatch script ──")
                    print(script)
                    print("───────────────────────────")
                    first = False
                if P._sbatch(script, tmp / f"ref_{code}.sh", args.dry_run) or args.dry_run:
                    sub += 1
            total_sub += sub
    print(f"TOTAL submitted={total_sub}  (over {args.n_geom * args.n_adp} arms)")


if __name__ == "__main__":
    main()
