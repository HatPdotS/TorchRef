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

Stage it rather than firing all 5000 tasks blind: run the locked default cell on its own,
check it against the previous grid's numbers for the same cell, and release the rest only if
they agree. Indices and arm names are unaffected by ``--only-cells``, so both halves
aggregate together.

    ./.dev/bin/python analysis/submit_weight_grid.py --array --only-cells 7,4
    # ... compare, then:
    ./.dev/bin/python analysis/submit_weight_grid.py --array

NB ``--dry-run`` still (re)writes the manifest -- pre-existing behaviour, and the reason a
dry run with a *different* ``--n-geom``/``--n-adp`` would overwrite the real grid's manifest.
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


def build_array_script(name, tasks_file, n_tasks, logdir, n_cycles, mem, throttle):
    """One SLURM job array: task i reads line i of the (tab-separated) tasks file
    `outdir<TAB>pdb<TAB>mtz<TAB>weights_json` and runs refine + REFMAC validation."""
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=hour
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem={mem}
#SBATCH --array=0-{n_tasks - 1}%{throttle}
#SBATCH --output={logdir}/task_%a.out
#SBATCH --error={logdir}/task_%a.err

set -e
export TORCHREF_NUM_THREADS=4

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{tasks_file}")
OUTDIR=$(printf '%s' "$LINE" | cut -f1)
PDB=$(printf '%s' "$LINE" | cut -f2)
MTZ=$(printf '%s' "$LINE" | cut -f3)
WEIGHTS=$(printf '%s' "$LINE" | cut -f4)
mkdir -p "$OUTDIR"

{P.PYTHON} -u {P.REFINE_SCRIPT} \\
    -m "$PDB" -sf "$MTZ" -o "$OUTDIR" \\
    -n {n_cycles} \\
    --mode separate \\
    --xray-mode ml \\
    --weights "$WEIGHTS"

# REFMAC 0-cycle validation (validate.log -> R-factors + geometry RMS/sigma)
source {P.CCP4_SETUP}
TEMP_DIR=/tmp/validate_{name}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
mkdir -p $TEMP_DIR && cd $TEMP_DIR
export CCP4_SCR=$TEMP_DIR
cp "$OUTDIR/refined.pdb" input.pdb
cp "$MTZ" input.mtz
refmac5 HKLIN input.mtz HKLOUT output.mtz XYZIN input.pdb XYZOUT output.pdb \\
    > "$OUTDIR/validate.log" 2>&1 << EOF
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
    ap.add_argument("--tag", default="",
                    help="Namespace suffix for an ablation grid, e.g. 'nosimu'. "
                         "Arms become wgrid_<tag>_g{gi}_a{ai} with their own "
                         "manifest so they don't collide with the baseline grid.")
    ap.add_argument("--disable", nargs="+", default=[],
                    help="Restraint component weights to force to 0 for this "
                         "grid, e.g. --disable adp/simu adp/KL.")
    ap.add_argument("--out-root", default=None,
                    help="Base directory for this grid's arm output + manifest "
                         "(default: figure2 runs/). Use a dedicated dir for "
                         "ablation grids to keep runs/ uncluttered.")
    ap.add_argument("--only-cells", nargs="+", default=None, metavar="GI,AI",
                    help="Submit only these grid cells, e.g. --only-cells 7,4. The axes are "
                         "built in full first, so indices, arm names and the manifest are "
                         "IDENTICAL to a full run and the remaining cells can be released "
                         "later into the same aggregation. Use this to stage the grid: run "
                         "the locked default cell alone, check it against the previous "
                         "grid's numbers, and only then release the other 99 -- 50 jobs "
                         "instead of 5000 before you know whether the build agrees with "
                         "history. An out-of-range index is an error, not an empty "
                         "submission.")
    ap.add_argument("--array", action="store_true",
                    help="Submit the whole grid as ONE SLURM job array instead "
                         "of one job per (cell, structure). Much lighter on the "
                         "scheduler.")
    ap.add_argument("--array-throttle", type=int, default=200,
                    help="Max concurrently-running array tasks (sbatch %%N). "
                         "At 4 cpus/task, 200 ~ 800 cpus (per-user cap ~1167).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve() if args.out_root else P.RUNS
    suffix = f"_{args.tag}" if args.tag else ""
    manifest_path = out_root / "metrics" / f"wgrid{suffix}_manifest.csv"
    disabled = {k: 0 for k in args.disable}

    geoms = np.logspace(np.log10(args.geom_range[0]), np.log10(args.geom_range[1]),
                        args.n_geom)
    adps = np.logspace(np.log10(args.adp_range[0]), np.log10(args.adp_range[1]),
                       args.n_adp)

    codes = args.codes if args.codes else P.load_solved_codes()
    if args.limit:
        codes = codes[:args.limit]

    # Parsed AFTER the axes exist, so the indices mean the same thing they would in a full
    # run and can be validated against the real extent. A typo'd cell must fail here rather
    # than submit an empty work list and look like "everything was already complete".
    only = None
    if args.only_cells:
        only = set()
        for spec in args.only_cells:
            try:
                gi, ai = (int(x) for x in spec.split(","))
            except ValueError:
                raise SystemExit(f"--only-cells: expected GI,AI, got {spec!r}")
            if not (0 <= gi < args.n_geom and 0 <= ai < args.n_adp):
                raise SystemExit(
                    f"--only-cells {spec}: out of range for a "
                    f"{args.n_geom}x{args.n_adp} grid (gi 0..{args.n_geom - 1}, "
                    f"ai 0..{args.n_adp - 1})")
            only.add((gi, ai))

    # Manifest: arm -> (gi, ai, geometry, adp), exact weights for aggregation. Written for
    # EVERY cell even under --only-cells, so a staged run and the later full release share
    # one manifest and the aggregator needs no knowledge of the staging.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "gi", "ai", "geometry", "adp"])
        for gi, g in enumerate(geoms):
            for ai, a in enumerate(adps):
                w.writerow([f"wgrid{suffix}_g{gi}_a{ai}", gi, ai,
                            f"{g:.6g}", f"{a:.6g}"])

    print(f"Log weight grid (xray fixed = 1)  tag={args.tag or '(baseline)'}:")
    print(f"  geometry [{args.n_geom}]: {', '.join(f'{g:.4g}' for g in geoms)}")
    print(f"  adp      [{args.n_adp}]: {', '.join(f'{a:.4g}' for a in adps)}")
    if disabled:
        print(f"  DISABLED components (weight=0): {', '.join(disabled)}")
    n_cells = len(only) if only else args.n_geom * args.n_adp
    if only:
        print("  STAGED — only these cells:")
        for gi, ai in sorted(only):
            print(f"    g{gi}_a{ai}: geometry={geoms[gi]:.4g}  adp={adps[ai]:.4g}")
    print(f"  {n_cells} combo(s) x {len(codes)} structures = "
          f"{n_cells * len(codes)} jobs max")
    print(f"  manifest: {manifest_path} (all "
          f"{args.n_geom * args.n_adp} cells)\n")

    # Collect the (cell x structure) work list, skipping already-complete cells.
    tasks = []  # each: (outdir, pdb, mtz, weights_json)
    skip = miss = 0
    for gi, g in enumerate(geoms):
        for ai, a in enumerate(adps):
            if only is not None and (gi, ai) not in only:
                continue
            arm = f"wgrid{suffix}_g{gi}_a{ai}"
            arm_dir = out_root / arm
            weights = {"xray": 1, "geometry": float(g), "adp": float(a),
                       **disabled}
            wjson = json.dumps(weights)
            for code in codes:
                pdb, mtz = P.PLACED / f"{code}_af.pdb", P._mtz(code)
                if not pdb.exists() or not mtz.exists():
                    miss += 1
                    continue
                outdir = arm_dir / code
                if P._refmac_complete(outdir / "validate.log") and not args.force:
                    skip += 1
                    continue
                tasks.append((str(outdir), str(pdb), str(mtz), wjson))
    print(f"work list: {len(tasks)} tasks  (skip={skip} complete, miss={miss})")

    if not tasks:
        print("nothing to do.")
        return

    if args.array:
        # One job array for the whole grid: task i <- line i of the tasks file.
        tmp = out_root / "tmp_scripts"
        logdir = tmp / f"logs{suffix}"
        tasks_file = tmp / f"tasks{suffix}.tsv"
        array_script = tmp / f"array{suffix}.sh"
        name = f"af_wgrid{suffix or '_base'}"
        script = build_array_script(name, tasks_file, len(tasks), logdir,
                                    args.n_cycles, args.mem, args.array_throttle)
        if args.dry_run:
            print(f"\n[DRY-RUN] would write {len(tasks)} tasks -> {tasks_file}")
            print(f"[DRY-RUN] would submit array 0-{len(tasks) - 1}%"
                  f"{args.array_throttle}\n── array script ──\n{script}")
            return
        tmp.mkdir(parents=True, exist_ok=True)
        logdir.mkdir(parents=True, exist_ok=True)
        with open(tasks_file, "w") as f:
            for outdir, pdb, mtz, wjson in tasks:
                f.write(f"{outdir}\t{pdb}\t{mtz}\t{wjson}\n")
        array_script.write_text(script)
        jid = P._sbatch(script, array_script, dry_run=False)
        print(f"submitted array job {jid}: {len(tasks)} tasks "
              f"(throttle %{args.array_throttle})")
        return

    # Per-job fallback (one sbatch per task).
    first = 0
    for outdir, pdb, mtz, wjson in tasks:
        outdir_p = Path(outdir)
        arm = outdir_p.parent.name
        code = outdir_p.name
        if not args.dry_run:
            outdir_p.mkdir(parents=True, exist_ok=True)
        script = build_script(arm, code, pdb, mtz, outdir_p, args.n_cycles,
                              args.mem, json.loads(wjson))
        if args.dry_run and first == 0:
            print("── example sbatch script ──")
            print(script)
            print("───────────────────────────")
        first += 1
        P._sbatch(script, outdir_p.parent / "tmp_scripts" / f"ref_{code}.sh",
                  args.dry_run)
    print(f"TOTAL submitted={len(tasks)}  (over {args.n_geom * args.n_adp} arms)")


if __name__ == "__main__":
    main()
