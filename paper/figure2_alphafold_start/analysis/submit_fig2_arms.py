#!/usr/bin/env python3
"""Submit all three Figure-2 refinement arms, STRUCTURE-major.

Figure 2 compares TorchRef, REFMAC and phenix.refine on the same Phaser-placed
AlphaFold models. Panel C plots their wall-clock, so *how* the jobs are submitted
is part of the measurement:

* **Structure-major, not engine-major.** A full sweep spans hours during which
  cluster load changes. Submitting all of one engine before the next times each
  engine in its own load window, so engine identity ends up confounded with time
  — measured 2026-08-04 on the exF4 sweep: two programs whose code had not
  changed moved by -43% and +31% against the previous sweep, with per-structure
  IQRs spanning ~2x. Emitting a structure's three arms adjacently puts them under
  near-identical conditions.
* **One CPU model.** `hour`/`day` span Xeon 6152/6230/6230R and EPYC 7453/9335,
  and EPYC 9335 is 2.2-2.7x faster than 7453 on identical code. Without a
  constraint each engine draws a different hardware mix, which is a systematic
  per-engine bias rather than noise that n=767 averages away.

Both properties are what `extended_figures/exF4/submit_singlecore.py` already
does for the 1-core re-timing; this makes the 4-core figure match.

Each arm's cell is built by the code that owns that arm — TorchRef by
``submit_local_arm.build_script`` (locked DEFAULT_GROUP_WEIGHTS, in-job REFMAC
0-cycle validation), REFMAC and phenix by ``run_af_pipeline.script_refmac`` /
``script_phenix`` — so there is one definition of each arm, not two.

Resume-safe: a cell whose output already exists is skipped (TorchRef keys on a
complete ``validate.log`` because it validates in-job; the other two key on their
final PDB, which ``run_af_pipeline.py validate`` scores afterwards).

Usage
-----
    ./.dev/bin/python analysis/submit_fig2_arms.py --dry-run --limit 1
    ./.dev/bin/python analysis/submit_fig2_arms.py --limit 2      # smoke test
    ./.dev/bin/python analysis/submit_fig2_arms.py                # all solved
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import run_af_pipeline as P  # noqa: E402
import submit_local_arm  # noqa: E402

ENGINES = ["torchref", "refmac", "phenix"]
ARM_DIR = {"torchref": "torchref", "refmac": "refmac", "phenix": "phenix_norb"}


def _cell(engine, code, pdb, mtz, outdir, n_cycles, constraint, mem):
    """(script_text, script_path) for one (engine, code) cell."""
    if engine == "torchref":
        return (submit_local_arm.build_script(
                    "torchref", code, pdb, mtz, outdir, n_cycles, mem,
                    constraint=constraint),
                outdir.parent / "tmp_scripts" / f"ref_{code}.sh")
    build = P.script_refmac if engine == "refmac" else P.script_phenix
    return (build(code, pdb, mtz, outdir, cycles=n_cycles, constraint=constraint),
            outdir.parent / "tmp_scripts" / f"{engine}_{code}.sh")


def _done(engine, code, outdir):
    """Already-complete cells are skipped so a partial sweep can be resumed."""
    if engine == "torchref":
        # Validates in-job, so a refined.pdb without a complete validate.log is a
        # half-finished cell and must be redone.
        return P._refmac_complete(outdir / "validate.log")
    return P.model_path(code, ARM_DIR[engine]).exists()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--engines", nargs="+", default=ENGINES, choices=ENGINES,
                    help="Subset of arms to submit (default: all three). Note a "
                         "subset re-times only those engines; Panel C needs all "
                         "three from one sweep.")
    ap.add_argument("--n-cycles", type=int, default=10,
                    help="Macro cycles for every engine (default 10, the "
                         "canonical benchmark count that exF4 re-times).")
    ap.add_argument("--constraint", default=P.CPU_MODEL,
                    help=f"SLURM --constraint for every cell (default "
                         f"{P.CPU_MODEL}).")
    ap.add_argument("--torchref-mem", default="8G",
                    help="SLURM --mem for the TorchRef cells (8G is the "
                         "benchmark default; the largest structures OOM there "
                         "and are reported as failures).")
    ap.add_argument("--codes", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-submit cells that already have output.")
    args = ap.parse_args()

    codes = args.codes if args.codes else P.load_solved_codes()
    if args.limit:
        codes = codes[:args.limit]

    print(f"Figure-2 arms: {', '.join(args.engines)}   build=DEV(in-tree)")
    print(f"n_cycles={args.n_cycles}  constraint={args.constraint}  "
          f"order=STRUCTURE-major")
    print(f"Python: {P.PYTHON}")
    print(f"{len(codes)} solved structures x {len(args.engines)} arms = "
          f"{len(codes) * len(args.engines)} cells\n")

    submitted = {e: 0 for e in args.engines}
    skipped = {e: 0 for e in args.engines}
    missing = 0
    shown = False

    for code in codes:
        pdb, mtz = P.PLACED / f"{code}_af.pdb", P._mtz(code)
        if not pdb.exists() or not mtz.exists():
            missing += 1
            continue
        for engine in args.engines:
            outdir = P.RUNS / ARM_DIR[engine] / code
            if _done(engine, code, outdir) and not args.force:
                skipped[engine] += 1
                continue
            if not args.dry_run:
                outdir.mkdir(parents=True, exist_ok=True)
            script, path = _cell(engine, code, pdb, mtz, outdir, args.n_cycles,
                                 args.constraint, args.torchref_mem)
            if args.dry_run and not shown:
                print(f"── example sbatch script ({engine}) ──")
                print(script)
                print("──────────────────────────────────────")
                shown = True
            if P._sbatch(script, path, args.dry_run) or args.dry_run:
                submitted[engine] += 1

    print(f"missing inputs: {missing} structures")
    for e in args.engines:
        print(f"  {e:<9} submitted={submitted[e]:>4}  skipped={skipped[e]:>4}")
    print(f"\ntotal submitted={sum(submitted.values())}")


if __name__ == "__main__":
    main()
