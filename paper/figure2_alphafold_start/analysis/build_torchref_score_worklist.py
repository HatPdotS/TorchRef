#!/usr/bin/env python
"""Build the TorchRef 0-cycle scoring worklist (all arms, all codes).

For every final model (prediction / refmac / phenix / torchref) with input data,
emit a task line

    engine <TAB> code <TAB> model_pdb <TAB> mtz <TAB> out_json

to ``runs/crossscore/torchref_worklist.txt`` for ``torchref_score_array.sh``
(which runs ``analysis/torchref_score.py``). Resume-safe: tasks whose out_json
already holds an ``r_free`` are skipped.

Usage
-----
    ./.dev/bin/python paper/figure2_alphafold_start/analysis/build_torchref_score_worklist.py

    # ExtFig 5 target-mode arms, scored alongside the four Figure-2 engines
    ... build_torchref_score_worklist.py --arm ml_noalpha=mode_ml_noalpha \\
                                        --arm ml_full=mode_ml_full

    # Re-score everything: the TorchRef scorer itself changed
    ... build_torchref_score_worklist.py --force

``--force`` matters more here than it looks. This scorer runs ``LBFGSRefinement.get_scales()``,
which fits scales through ``ScalerBase.refine_lbfgs(scale_target=...)`` -- so when the
``scale_target`` default changed, **every** stored ``torchref_validate.json`` became a number
from a different scorer, including those of the reference arms nobody re-refined. Without
``--force`` the resume-skip keeps them and ExtFig 3 silently mixes two scorer versions.
Snapshot the old JSONs before forcing: they are the only record of the previous scorer.
"""

import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE.parent / "data"
RUNS = BASE / "runs"
XDIR = RUNS / "crossscore"
WORKLIST = XDIR / "torchref_worklist.txt"

# engine -> (arm subdir [for the out_json], final-PDB path relative to RUNS)
MODELS = {
    "refmac": ("refmac", "refmac/{code}/refined.pdb"),
    "phenix": ("phenix_norb", "phenix_norb/{code}/{code}_refined_001.pdb"),
    "torchref": ("torchref", "torchref/{code}/refined.pdb"),
    "prediction": ("af_initial", "../placed/{code}_af.pdb"),
}


def done(js: Path) -> bool:
    if not js.exists():
        return False
    try:
        return "r_free" in json.loads(js.read_text())
    except (OSError, ValueError):
        return False


def parse_arms(specs, models):
    """Add ``NAME=DIR`` arms to `models`, in place.

    An extra arm is always a torchref-kind arm, so its final model is
    ``<DIR>/<code>/refined.pdb`` -- the layout ``submit_local_arm.py`` writes. Mirrors the
    ``--arm`` option on ``aggregate_figure_metrics.py`` so one spelling serves both.
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
                    help="Re-score even where the out-name already holds a "
                         "result. Needed whenever the SCORER changed (see the module "
                         "docstring) -- the default resume-skip would keep stale numbers.")
    ap.add_argument("--out-name", default="torchref_validate.json",
                    help="Filename written inside each code directory. Point a re-score "
                         "at a NEW name rather than --force-ing over the old one when the "
                         "old numbers still back a figure: two scorer versions then sit "
                         "side by side and stay comparable.")
    ap.add_argument("--only", nargs="*", default=None, metavar="ENGINE",
                    help="Restrict to these engines (default: all).")
    args = ap.parse_args()

    models = parse_arms(args.arm, dict(MODELS))
    if args.only:
        unknown = set(args.only) - set(models)
        if unknown:
            raise SystemExit(f"--only: unknown engine(s) {sorted(unknown)}; "
                             f"have {sorted(models)}")
        models = {k: v for k, v in models.items() if k in args.only}

    XDIR.mkdir(parents=True, exist_ok=True)
    (XDIR / "slurm_tr").mkdir(exist_ok=True)

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
            out = RUNS / sub / code / args.out_name
            if not pdb.exists() or not out.parent.is_dir():
                nomodel += 1
                continue
            if done(out) and not args.force:
                skipped += 1
                continue
            lines.append(f"{eng}\t{code}\t{pdb}\t{mtz}\t{out}")

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
              f"{BASE}/analysis/torchref_score_array.sh")


if __name__ == "__main__":
    main()
