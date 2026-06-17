#!/usr/bin/env python3
"""Re-run the canonical AF-start TorchRef arm with THIS (dev) build, in-tree.

Reproduces the validated default arm ``torchref_scalerfix_nocoref_n10`` exactly:
``-n 10 --mode separate --xray-mode ml --adp-weight 0.1`` with NO ``--weights``
override (xray group weight stays the default 10 from ``DEFAULT_GROUP_WEIGHTS``),
separate scaler (corefine off, the LBFGSRefinement default), no rigid body.

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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_af_pipeline as P  # noqa: E402


def build_script(arm, code, pdb, mtz, outdir, n_cycles, mem, xray_weight=None):
    # Default weights (xray=10 / geom=1 / adp=0.1) come from
    # DEFAULT_GROUP_WEIGHTS — no flag needed. --xray-weight only overrides the
    # data-term weight (merged onto the defaults by refine.py), e.g. 5 to
    # down-weight the data and let the geometry prior regularize more.
    weights_line = (
        "" if xray_weight is None
        else f" \\\n    --weights '{{\"xray\": {xray_weight}}}'"
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
                    help="Override the xray group weight (default 10). e.g. 5 "
                         "to down-weight the data term; geom=1/adp=0.1 unchanged.")
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
    xw = "10 (default)" if args.xray_weight is None else f"{args.xray_weight}"
    print(f"Arm: {args.arm}  build=DEV(in-tree)  n_cycles={args.n_cycles}  "
          f"xray-mode=ml  xray-weight={xw}  geom=1/adp=0.1")
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
                              args.mem, xray_weight=args.xray_weight)
        if args.dry_run and submitted == 0:
            print("── example sbatch script ──")
            print(script)
            print("───────────────────────────")
        if P._sbatch(script, tmp / f"ref_{code}.sh", args.dry_run) or args.dry_run:
            submitted += 1

    print(f"\nsubmitted={submitted}, skipped={skipped}, missing={missing}")


if __name__ == "__main__":
    main()
