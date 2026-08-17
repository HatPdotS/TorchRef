#!/usr/bin/env python3
"""Submit the 3-arm seeded-optimizer benchmark over a fixed 100-structure subset.

For each code in ``subset_100.txt`` this submits three SLURM jobs, each running
refinement (from the Phaser-placed AF start model) + a REFMAC 0-cycle validation
in one shot so the apples-to-apples R-factors land in ``validate.log``:

    bench_sep_lbfgs     -n 10 --mode separate  --xray-mode ml     (production standard)
    bench_joint_lbfgs   -n 10 --mode everything --xray-mode ml    (joint control)
    bench_joint_seeded  -n 10 --mode everything --optimizer seeded (+ diagonal seed)

`sep→joint` isolates separate-vs-joint; `joint_lbfgs→joint_seeded` isolates the
diagonal-Hessian seed; `sep→seeded` compares the new approach to the shipped default.

Usage
-----
    ./.dev/bin/python analysis/submit_seeded_benchmark.py --dry-run --limit 1
    ./.dev/bin/python analysis/submit_seeded_benchmark.py --limit 1      # one real triple
    ./.dev/bin/python analysis/submit_seeded_benchmark.py               # all 100 x 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_af_pipeline as P  # noqa: E402

# arm name -> extra refine.py flags (everything else identical across arms)
ARMS = {
    "bench_sep_lbfgs": "--mode separate --xray-mode ml",
    "bench_joint_lbfgs": "--mode everything --xray-mode ml",
    "bench_joint_seeded": "--mode everything --optimizer seeded --xray-mode ml",
    # scaler co-refinement arms (scaler folded into the joint step)
    "bench_joint_seeded_coref": "--mode everything --optimizer seeded --corefine-scaler --xray-mode ml",
    "bench_joint_lbfgs_coref": "--mode everything --corefine-scaler --xray-mode ml",
}


def build_script(arm, code, pdb, mtz, outdir, n_cycles, mem, refine_flags):
    return f"""#!/bin/bash
#SBATCH --job-name=sb_{arm}_{code}
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem={mem}
#SBATCH --output={outdir / 'out.log'}
#SBATCH --error={outdir / 'error.log'}

set -e
export TORCHREF_NUM_THREADS=4

{P.PYTHON} -u {P.REFINE_SCRIPT} \\
    -m {pdb} -sf {mtz} -o {outdir} \\
    -n {n_cycles} \\
    {refine_flags}

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
    ap.add_argument("--codes-file", default=str(HERE / "subset_100.txt"))
    ap.add_argument("--n-cycles", type=int, default=10)
    ap.add_argument("--mem", default="16G",
                    help="SLURM --mem per job (16G recovers the large structures "
                         "that OOM at the 8G canonical default).")
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    choices=list(ARMS))
    ap.add_argument("--codes", nargs="+", default=None,
                    help="Override the subset with explicit codes.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-submit even if validate.log already complete.")
    args = ap.parse_args()

    if args.codes:
        codes = args.codes
    else:
        codes = [c.strip() for c in Path(args.codes_file).read_text().split()
                 if c.strip()]
    if args.limit:
        codes = codes[:args.limit]

    print(f"Subset: {len(codes)} codes x {len(args.arms)} arms  n_cycles={args.n_cycles}"
          f"  mem={args.mem}")
    print(f"Python: {P.PYTHON}\nArms: {args.arms}\n")

    submitted = skipped = missing = 0
    for code in codes:
        pdb, mtz = P.PLACED / f"{code}_af.pdb", P._mtz(code)
        if not pdb.exists() or not mtz.exists():
            missing += 1
            continue
        for arm in args.arms:
            outdir = P.RUNS / arm / code
            if P._refmac_complete(outdir / "validate.log") and not args.force:
                skipped += 1
                continue
            if not args.dry_run:
                outdir.mkdir(parents=True, exist_ok=True)
            script = build_script(arm, code, pdb, mtz, outdir, args.n_cycles,
                                  args.mem, ARMS[arm])
            if args.dry_run and submitted == 0:
                print(f"── example sbatch script ({arm}) ──")
                print(script)
                print("───────────────────────────")
            tmp = P.RUNS / arm / "tmp_scripts"
            if P._sbatch(script, tmp / f"ref_{code}.sh", args.dry_run) or args.dry_run:
                submitted += 1

    print(f"\nsubmitted={submitted}, skipped={skipped}, missing={missing}")


if __name__ == "__main__":
    main()
