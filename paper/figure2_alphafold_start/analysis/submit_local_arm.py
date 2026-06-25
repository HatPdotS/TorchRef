#!/usr/bin/env python3
"""Re-run the canonical AF-start TorchRef arm with THIS (dev) build, in-tree.

Reproduces the canonical default arm exactly: ``-n 10 --mode separate
--xray-mode ml`` with NO ``--weights`` override, so the group weights stay at the
locked ``DEFAULT_GROUP_WEIGHTS`` (xray 1 / geometry 0.2 / adp 0.02); separate
scaler (corefine off, the LBFGSRefinement default), no rigid body.

Because this lives under the dev worktree, ``run_af_pipeline``'s ``PYTHON`` /
``REFINE_SCRIPT`` already point at the dev build — no override needed.

Each SLURM job does refine + REFMAC 0-cycle validation in one shot so the
``validate.log`` (the apples-to-apples R-factor) lands next to ``refined.pdb``.

Baseline (review build, REFMAC-0-cycle validated; see baseline/fig_rfactors.csv):
    torchref  R_work 0.2682  R_free 0.3235  (n=760)
    phenix    R_work 0.2846  R_free 0.3243
    refmac    R_work 0.2784  R_free 0.3216

Usage
-----
    ./.dev/bin/python analysis/submit_local_arm.py --dry-run --limit 1
    ./.dev/bin/python analysis/submit_local_arm.py --limit 1   # one real job
    ./.dev/bin/python analysis/submit_local_arm.py            # all solved
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_af_pipeline as P  # noqa: E402


def build_script(arm, code, pdb, mtz, outdir, n_cycles, mem, weights=None):
    # Group weights default to DEFAULT_GROUP_WEIGHTS (xray=1 / geom=0.2 /
    # adp=0.02) — no flag needed. Pass an explicit {xray, geometry, adp} dict
    # to override and record the exact weights in the job script (merged onto
    # the defaults by refine.py).
    weights_line = (
        "" if not weights
        else f" \\\n    --weights '{json.dumps(weights)}'"
    )
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
    --sigma-m-scale 1.0{weights_line}

# REFMAC 0-cycle validation on the refined model (apples-to-apples R-factors)
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
    ap.add_argument("--arm", default="torchref_devbuild",
                    help="Output arm name under runs/ (default torchref_devbuild).")
    ap.add_argument("--n-cycles", type=int, default=10)
    ap.add_argument("--xray-weight", type=float, default=None,
                    help="Override the xray group weight (default 1).")
    ap.add_argument("--geometry-weight", type=float, default=None,
                    help="Override the geometry group weight (default 0.2).")
    ap.add_argument("--adp-weight", type=float, default=None,
                    help="Override the adp group weight (default 0.02).")
    ap.add_argument("--no-ramachandran", action="store_true",
                    help="Disable the Ramachandran restraint by setting the "
                         "component weight 'geometry/ramachandran' to 0 (the "
                         "other geometry components stay at their group weight). "
                         "Use a distinct --arm (e.g. *_norama) so the run lands "
                         "in its own directory for the with/without comparison.")
    ap.add_argument("--mem", default="8G",
                    help="SLURM --mem per job (e.g. 16G for large structures).")
    ap.add_argument("--codes", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-submit even if validate.log already complete.")
    args = ap.parse_args()

    codes = args.codes if args.codes else P.load_solved_codes()
    if args.limit:
        codes = codes[:args.limit]

    arm_dir = P.RUNS / args.arm
    tmp = arm_dir / "tmp_scripts"
    weights = {}
    if args.xray_weight is not None:
        weights["xray"] = args.xray_weight
    if args.geometry_weight is not None:
        weights["geometry"] = args.geometry_weight
    if args.adp_weight is not None:
        weights["adp"] = args.adp_weight
    if args.no_ramachandran:
        # Zero the Ramachandran component (group weight still multiplies in, but
        # 0.2 * 0 = 0). refine.py merges this onto DEFAULT_GROUP_WEIGHTS, so the
        # remaining geometry components keep their default group weight.
        weights["geometry/ramachandran"] = 0.0
    weights = weights or None
    wdesc = "DEFAULT_GROUP_WEIGHTS (1/0.2/0.02)" if weights is None else json.dumps(weights)
    print(f"Arm: {args.arm}  build=DEV(in-tree)  n_cycles={args.n_cycles}  "
          f"xray-mode=ml  weights={wdesc}")
    print(f"Python: {P.PYTHON}")
    print(f"Output: {arm_dir}\n{len(codes)} solved structures\n")

    submitted = skipped = missing = 0
    for code in codes:
        pdb, mtz = P.PLACED / f"{code}_af.pdb", P._mtz(code)
        if not pdb.exists() or not mtz.exists():
            missing += 1
            continue
        outdir = arm_dir / code
        if P._refmac_complete(outdir / "validate.log") and not args.force:
            skipped += 1
            continue
        if not args.dry_run:
            outdir.mkdir(parents=True, exist_ok=True)
        script = build_script(args.arm, code, pdb, mtz, outdir, args.n_cycles,
                              args.mem, weights=weights)
        if args.dry_run and submitted == 0:
            print("── example sbatch script ──")
            print(script)
            print("───────────────────────────")
        if P._sbatch(script, tmp / f"ref_{code}.sh", args.dry_run) or args.dry_run:
            submitted += 1

    print(f"\nsubmitted={submitted}, skipped={skipped}, missing={missing}")


if __name__ == "__main__":
    main()
