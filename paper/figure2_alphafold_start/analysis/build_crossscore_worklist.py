#!/usr/bin/env python
"""Build the PHENIX-vs-REFMAC cross-scoring worklist.

For every final model (refmac / phenix / torchref) that has input data, emit one
tab-separated task line

    engine <TAB> code <TAB> model_pdb <TAB> mtz <TAB> out_log

to ``runs/crossscore/worklist.txt`` for consumption by ``crossscore_array.sh``
(which runs ``phenix.model_vs_data``). Tasks whose ``out_log`` already contains a
computed ``r_free:`` are skipped, so the script is resume-safe: re-run it after a
partial array to shrink the worklist to the remaining work.

Usage
-----
    ./.dev/bin/python paper/figure2_alphafold_start/analysis/build_crossscore_worklist.py

    # ExtFig 5 target-mode arms, PHENIX-scored so their cells are directly comparable to
    # Figure 2's (whose headline R-factors come from this same scorer)
    ... build_crossscore_worklist.py --arm ml_noalpha=mode_ml_noalpha \\
                                     --arm ml_full=mode_ml_full \\
                                     --arm nll_beta=mode_nll_beta
"""

import argparse
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent      # paper/figure2_alphafold_start
DATA = BASE.parent / "data"
RUNS = BASE / "runs"
XDIR = RUNS / "crossscore"
WORKLIST = XDIR / "worklist.txt"

# model_engine -> (arm subdir [for the phenix_validate.log], final-PDB path
# relative to RUNS). The prediction model lives in ../placed, not under its arm.
MODELS = {
    "refmac": ("refmac", "refmac/{code}/refined.pdb"),
    "phenix": ("phenix_norb", "phenix_norb/{code}/{code}_refined_001.pdb"),
    # Canonical arm: -n 10 separate ml, locked weights 1/0.2/0.02.
    "torchref": ("torchref", "torchref/{code}/refined.pdb"),
    "prediction": ("af_initial", "../placed/{code}_af.pdb"),
}
RE_DONE = re.compile(r"^\s*r_free:\s*[\d.]+", re.M)


def done(log: Path) -> bool:
    return log.exists() and bool(RE_DONE.search(log.read_text(errors="replace")))


def parse_arms(specs, models):
    """Add ``NAME=DIR`` arms to `models`, in place.

    An extra arm is always a torchref-kind arm, so its final model is
    ``<DIR>/<code>/refined.pdb`` -- the layout ``submit_local_arm.py`` writes. Mirrors the
    ``--arm`` option on ``aggregate_figure_metrics.py`` and on
    ``build_torchref_score_worklist.py`` so one spelling serves all three.
    """
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"--arm expects NAME=DIR, got {spec!r}")
        name, sub = spec.split("=", 1)
        if name in models:
            raise SystemExit(f"--arm {spec}: {name!r} already exists as "
                             f"{models[name][0]!r}; pick another name")
        models[name] = (sub, sub + "/{code}/refined.pdb")
    return models


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=None, metavar="NAME=DIR",
                    help="Extra torchref-kind arm to score, e.g. "
                         "--arm ml_full=mode_ml_full. Repeatable.")
    ap.add_argument("--force", action="store_true",
                    help="Re-score even where a phenix_validate.log already holds an r_free. "
                         "The default resume-skip is what makes this script safe to re-run "
                         "after a partial array, so only use this when the MODELS changed "
                         "under a name that already has logs.")
    args = ap.parse_args()

    models = parse_arms(args.arm, dict(MODELS))

    XDIR.mkdir(parents=True, exist_ok=True)
    (XDIR / "slurm").mkdir(exist_ok=True)  # SLURM won't create the log dir itself

    codes = sorted({d.name for sub, _ in models.values()
                    for d in (RUNS / sub).iterdir() if (RUNS / sub).is_dir()
                    and d.is_dir()})

    lines, skipped, nomtz, nomodel = [], 0, 0, 0
    for code in codes:
        mtz = DATA / code / f"{code}.mtz"
        if not mtz.exists():
            nomtz += 1
            continue
        for eng, (sub, patt) in models.items():
            pdb = (RUNS / patt.format(code=code)).resolve()
            if not pdb.exists():
                nomodel += 1
                continue
            log = RUNS / sub / code / "phenix_validate.log"
            if not log.parent.is_dir():
                nomodel += 1
                continue
            if done(log) and not args.force:
                skipped += 1
                continue
            lines.append(f"{eng}\t{code}\t{pdb}\t{mtz}\t{log}")

    WORKLIST.write_text("\n".join(lines) + ("\n" if lines else ""))
    # Report the arm set, not just the totals: "tasks=767" looks the same whether four arms
    # contributed or one did, and an --arm typo shows up here as a missing row rather than as
    # a silently short worklist.
    print("arms: " + "  ".join(f"{e}->{sub}" for e, (sub, _) in models.items()))
    per_arm = {e: sum(1 for ln in lines if ln.split("\t")[0] == e) for e in models}
    print("tasks per arm: " + "  ".join(f"{e}={n}" for e, n in per_arm.items()))
    print(f"codes={len(codes)}  tasks={len(lines)}  already_done={skipped}  "
          f"missing_mtz={nomtz}  missing_model={nomodel}"
          + ("  [--force]" if args.force else ""))
    print(f"worklist: {WORKLIST}")
    if lines:
        print(f"submit:  sbatch --array=1-{len(lines)}%60 "
              f"{BASE}/analysis/crossscore_array.sh")


if __name__ == "__main__":
    main()
