#!/usr/bin/env python3
"""Self-contained pipeline for the AlphaFold-start benchmark arm.

Refines the Phaser-placed AlphaFold models (figure2_alphafold_start/placed/) with
both TorchRef (with/without rigid body) and Phenix, validates every model with a
REFMAC5 zero-cycle run for a fair R-factor comparison, and collects the metrics.

The structure set is the MR-solved subset (mr_status.csv solved==1 / placed/*).

Subcommands
-----------
  refine   --refiner {torchref,phenix} [--rigid-body] [--rigid-body-iter N]
  validate                     REFMAC 0-cycle on af_initial + each refined arm
                               (validate.log co-located in each arm's run dir)
  analyze                      collect R-factors -> metrics/comparison.csv
  migrate                      adopt in-flight runs + relocate old flat logs
  status                       queue + output summary

REFMAC submit/parse logic mirrors figure2_validation/run_pipeline.py
(submit_refmac_jobs / _parse_refmac_log); copied here to keep this self-contained.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent                 # figure2_alphafold_start/
PAPER = HERE.parent                                     # paper/
REPO = PAPER.parent                                     # review/
DATA = PAPER / "data"
PLACED = HERE / "placed"
RUNS = HERE / "runs"
REFMAC_LOGS = RUNS / "refmac_logs"
METRICS = RUNS / "metrics"
MANIFEST = HERE / "manifest.json"
MR_STATUS = HERE / "mr_status.csv"

PYTHON = str(REPO / ".dev" / "bin" / "python")
REFINE_SCRIPT = str(REPO / "torchref" / "cli" / "refine.py")
PHENIX_REFINE = str(PAPER / "figure2_validation" / "phenix_refinement" / "phenix_refine.sh")
CCP4_SETUP = "/afs/psi.ch/sys/psi.ra/MX/ccp4/7.1/ccp4-7.1/bin/ccp4.setup-sh"

# Refinement arms (af_initial = the placed search model, before refinement).
ARMS = ["af_initial", "torchref_norb", "torchref_rb", "phenix_norb", "phenix_rb",
        "refmac"]


def _mtz(code):
    return DATA / code / f"{code}.mtz"


def model_path(code, arm):
    """Resolve the PDB for a given code+arm (may not exist yet)."""
    if arm == "af_initial":
        return PLACED / f"{code}_af.pdb"
    if arm == "torchref_norb":
        return RUNS / "torchref_norb" / "results" / code / "default" / "refined.pdb"
    if arm == "torchref_rb":
        return RUNS / "torchref_rb" / "results" / code / "default" / "refined.pdb"
    if arm in ("phenix_norb", "phenix_rb"):
        return RUNS / arm / code / f"{code}_refined_001.pdb"
    if arm == "refmac":
        return RUNS / "refmac" / code / "refined.pdb"
    raise ValueError(arm)


def validate_log(code, arm):
    """Co-located REFMAC 0-cycle validation log for code+arm.

    Lives next to the model it validates (``model_path().parent``); named
    ``validate.log`` to stay distinct from the refmac arm's refinement log
    (``runs/refmac/{code}/refmac.log``). ``af_initial`` has no run directory, so
    it gets a minimal one under ``runs/af_initial/{code}/``.
    """
    if arm == "af_initial":
        return RUNS / "af_initial" / code / "validate.log"
    return model_path(code, arm).parent / "validate.log"


def load_solved_codes():
    """MR-solved codes that have both a placed model and an MTZ."""
    codes = []
    if MR_STATUS.exists():
        for r in csv.DictReader(open(MR_STATUS)):
            if r["solved"] == "1":
                codes.append(r["code"])
    else:  # fall back to whatever is placed
        codes = [p.name[:-len("_af.pdb")] for p in PLACED.glob("*_af.pdb")]
    return [c for c in sorted(set(codes))
            if (PLACED / f"{c}_af.pdb").exists() and _mtz(c).exists()]


def _select(codes_arg, limit):
    codes = codes_arg if codes_arg else load_solved_codes()
    return codes[:limit] if limit else codes


def _sbatch(script_text, script_path, dry_run):
    if dry_run:
        print(f"  [DRY-RUN] would sbatch -> {script_path}")
        return None
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_text)
    try:
        out = subprocess.run(["sbatch", "--parsable", str(script_path)],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip().split(";")[0]
    except subprocess.CalledProcessError as e:
        print(f"  FAILED sbatch {script_path.name}: {e.stderr.strip()}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# refine
# ──────────────────────────────────────────────────────────────────────────────

def cmd_refine(args):
    codes = _select(args.codes, args.limit)
    submitted = skipped = missing = 0

    if args.refiner == "torchref":
        arm = "torchref_rb" if args.rigid_body else "torchref_norb"
        rb_line = ""
        if args.rigid_body:
            rb_line = f" \\\n    --with-rigid-body --rigid-body-iter {args.rigid_body_iter}"
        tmp = RUNS / arm / "tmp_scripts"
        for code in codes:
            pdb, mtz = PLACED / f"{code}_af.pdb", _mtz(code)
            if not pdb.exists() or not mtz.exists():
                missing += 1
                continue
            outdir = RUNS / arm / "results" / code / "default"
            if (outdir / "refined.pdb").exists() and not args.force:
                skipped += 1
                continue
            if not args.dry_run:
                outdir.mkdir(parents=True, exist_ok=True)
            script = f"""#!/bin/bash
#SBATCH --job-name=af_{arm[9:]}_{code}
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --output={outdir / 'out.log'}
#SBATCH --error={outdir / 'error.log'}

{PYTHON} -u {REFINE_SCRIPT} \\
    -m {pdb} -sf {mtz} -o {outdir} \\
    -n {args.n_cycles} \\
    --mode separate \\
    --xray-mode {args.xray_mode} \\
    --adp-weight {args.adp_weight} \\
    --sigma-m-scale {args.sigma_m_scale}{rb_line}
"""
            if _sbatch(script, tmp / f"ref_{code}.sh", args.dry_run) or args.dry_run:
                submitted += 1

    elif args.refiner == "refmac":
        # Restrained REFMAC refinement from the placed AF model. Same engine /
        # CCP4 setup / MTZ labels as the 0-cycle validation (cmd_validate); the
        # only differences are NCYCLES>0 and capturing XYZOUT. AF models are
        # protein-only, so REFMAC's built-in monomer library suffices (no
        # ligand dictionaries needed). REFMAC picks up FreeR_flag automatically,
        # holding out the same free set the validation scores against.
        arm = "refmac"
        tmp = RUNS / arm / "tmp_scripts"
        for code in codes:
            pdb, mtz = PLACED / f"{code}_af.pdb", _mtz(code)
            if not pdb.exists() or not mtz.exists():
                missing += 1
                continue
            outdir = RUNS / arm / code
            out = outdir / "refined.pdb"
            if out.exists() and not args.force:
                skipped += 1
                continue
            if not args.dry_run:
                outdir.mkdir(parents=True, exist_ok=True)
            script = f"""#!/bin/bash
#SBATCH --job-name=af_refmac_{code}
#SBATCH --partition=hour
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=2G
#SBATCH --output={outdir / 'refmac.log'}
#SBATCH --error={outdir / 'refmac.log'}

source {CCP4_SETUP}
TEMP_DIR=/tmp/refmac_refine_{code}_${{SLURM_JOB_ID}}
mkdir -p $TEMP_DIR && cd $TEMP_DIR
export CCP4_SCR=$TEMP_DIR
cp {pdb} input.pdb
cp {mtz} input.mtz
refmac5 HKLIN input.mtz HKLOUT output.mtz XYZIN input.pdb XYZOUT output.pdb << EOF
NCYCLES {args.refmac_cycles}
MAKE HYDR NO
END
EOF
cp output.pdb {out}
cd / && rm -rf $TEMP_DIR
"""
            if _sbatch(script, tmp / f"refmac_{code}.sh", args.dry_run) or args.dry_run:
                submitted += 1

    else:  # phenix
        rigid = "rb" if args.rigid_body else "norb"
        arm = f"phenix_{rigid}"
        for code in codes:
            if not (PLACED / f"{code}_af.pdb").exists() or not _mtz(code).exists():
                missing += 1
                continue
            out = RUNS / arm / code / f"{code}_refined_001.pdb"
            if out.exists() and not args.force:
                skipped += 1
                continue
            if args.dry_run:
                print(f"  [DRY-RUN] sbatch {PHENIX_REFINE} {code} af {rigid}")
                submitted += 1
                continue
            try:
                subprocess.run(["sbatch", "--job-name", f"phenix_{rigid}_{code}",
                                PHENIX_REFINE, code, "af", rigid], check=True,
                               capture_output=True, text=True)
                submitted += 1
            except subprocess.CalledProcessError as e:
                print(f"  FAILED {code}: {e.stderr.strip()}")

    print(f"\n{args.refiner} refine: submitted={submitted}, "
          f"skipped={skipped}, missing={missing}")


# ──────────────────────────────────────────────────────────────────────────────
# validate (REFMAC 0-cycle)
# ──────────────────────────────────────────────────────────────────────────────

def _refmac_complete(log):
    try:
        return "Overall R factor" in log.read_text()
    except Exception:
        return False


def cmd_validate(args):
    codes = _select(args.codes, args.limit)
    submitted = skipped = 0

    for code in codes:
        mtz = _mtz(code)
        if not mtz.exists():
            continue
        for arm in ARMS:
            pdb = model_path(code, arm)
            if not pdb.exists():
                continue
            label = f"{code}_{arm}"
            log = validate_log(code, arm)
            if _refmac_complete(log) and not args.force:
                skipped += 1
                continue
            if not args.dry_run:
                log.parent.mkdir(parents=True, exist_ok=True)
            script = f"""#!/bin/bash
#SBATCH --job-name=rm_{label[:18]}
#SBATCH --output={log}
#SBATCH --error={log}
#SBATCH --time=00:10:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

source {CCP4_SETUP}
TEMP_DIR=/tmp/refmac_{label}_${{SLURM_JOB_ID}}
mkdir -p $TEMP_DIR && cd $TEMP_DIR
export CCP4_SCR=$TEMP_DIR
cp {pdb} input.pdb
cp {mtz} input.mtz
refmac5 HKLIN input.mtz HKLOUT output.mtz XYZIN input.pdb XYZOUT output.pdb << EOF
NCYCLES 0
MAKE HYDR NO
END
EOF
cd / && rm -rf $TEMP_DIR
"""
            if _sbatch(script, log.parent / "validate.sh", args.dry_run) or args.dry_run:
                submitted += 1
    print(f"\nREFMAC validate: submitted={submitted}, skipped={skipped}")


# ──────────────────────────────────────────────────────────────────────────────
# analyze
# ──────────────────────────────────────────────────────────────────────────────

def _parse_refmac(log):
    """R-work / R-free from a REFMAC 0-cycle log (mirrors run_pipeline.py)."""
    try:
        content = log.read_text()
    except Exception:
        return None, None
    mw = re.search(r"Overall R factor\s+=\s+([\d.]+)", content)
    mf = re.search(r"Free R factor\s+=\s+([\d.]+)", content)
    return (float(mw.group(1)) if mw else np.nan,
            float(mf.group(1)) if mf else np.nan)


def cmd_analyze(args):
    codes = _select(args.codes, args.limit)
    status = {r["code"]: r for r in csv.DictReader(open(MR_STATUS))} if MR_STATUS.exists() else {}
    METRICS.mkdir(parents=True, exist_ok=True)

    rows = []
    for code in codes:
        row = {"code": code}
        st = status.get(code, {})
        row["tfz"] = st.get("tfz", "")
        row["llg"] = st.get("llg", "")
        try:
            row["low_tfz"] = int(float(st["tfz"]) < 8) if st.get("tfz") else ""
        except ValueError:
            row["low_tfz"] = ""
        for arm in ARMS:
            log = validate_log(code, arm)
            rw, rf = _parse_refmac(log) if log.exists() else (np.nan, np.nan)
            row[f"rwork_{arm}"] = rw
            row[f"rfree_{arm}"] = rf
        rows.append(row)

    fields = (["code", "tfz", "llg", "low_tfz"]
              + [f"{m}_{arm}" for arm in ARMS for m in ("rwork", "rfree")])
    comp = METRICS / "comparison.csv"
    with open(comp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Median R-free per arm, all vs high-confidence (TFZ>=8) placements.
    def medians(subset):
        out = {}
        for arm in ARMS:
            vals = [r[f"rfree_{arm}"] for r in subset
                    if isinstance(r[f"rfree_{arm}"], float) and not np.isnan(r[f"rfree_{arm}"])]
            out[arm] = (len(vals), float(np.median(vals)) if vals else np.nan)
        return out

    hi = [r for r in rows if r["low_tfz"] == 0]
    summ = METRICS / "summary.csv"
    with open(summ, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "arm", "n_with_rfree", "median_rfree"])
        for name, subset in [("all", rows), ("tfz>=8", hi)]:
            for arm, (n, med) in medians(subset).items():
                w.writerow([name, arm, n, f"{med:.4f}" if med == med else ""])

    print(f"comparison: {comp}\nsummary:    {summ}")
    print("\nMedian R-free (all solved):")
    for arm, (n, med) in medians(rows).items():
        print(f"  {arm:14s} {med:.4f}  (n={n})" if med == med else f"  {arm:14s}   -")


# ──────────────────────────────────────────────────────────────────────────────
# migrate (adopt in-flight runs)
# ──────────────────────────────────────────────────────────────────────────────

MIGRATIONS = [
    (PAPER / "figure2_validation" / "experiments" / "af_start_v1", RUNS / "torchref_norb"),
    (PAPER / "figure2_validation" / "experiments" / "af_start_rb", RUNS / "torchref_rb"),
    (PAPER / "phenix_refinements_af", RUNS / "phenix_rb"),
]


def cmd_migrate(args):
    RUNS.mkdir(parents=True, exist_ok=True)
    for src, dst in MIGRATIONS:
        if dst.exists():
            print(f"SKIP {dst.name}: destination already exists")
            continue
        if not src.exists():
            print(f"SKIP {dst.name}: source {src} not found")
            continue
        print(f"MOVE {src}  ->  {dst}")
        if not args.dry_run:
            shutil.move(str(src), str(dst))

    # Relocate the old flat validation logs (refmac_logs/{code}_{arm}.log) into
    # their co-located run dirs so the already-computed validations are reused
    # rather than re-submitted. Arm names contain underscores and PDB codes do
    # not, so match the longest arm suffix to split the stem unambiguously.
    moved = unmatched = collided = 0
    arms_by_len = sorted(ARMS, key=len, reverse=True)
    if REFMAC_LOGS.exists():
        for old in sorted(REFMAC_LOGS.glob("*.log")):
            stem = old.stem
            arm = next((a for a in arms_by_len if stem.endswith("_" + a)), None)
            if arm is None:
                unmatched += 1
                continue
            code = stem[: -(len(arm) + 1)]
            dst = validate_log(code, arm)
            if dst.exists():
                collided += 1
                continue
            if not args.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), str(dst))
            moved += 1
        print(f"validation logs: {moved} relocated, {collided} already present, "
              f"{unmatched} unmatched")

    if args.dry_run:
        print("(dry-run; nothing moved)")


# ──────────────────────────────────────────────────────────────────────────────
# status
# ──────────────────────────────────────────────────────────────────────────────

def cmd_status(args):
    codes = load_solved_codes()
    print(f"Solved AF arm: {len(codes)} structures\n")
    for arm in ARMS:
        n = sum(1 for c in codes if model_path(c, arm).exists())
        done = sum(1 for c in codes if _refmac_complete(validate_log(c, arm)))
        print(f"  {arm:14s} models present: {n}/{len(codes)}"
              f"   validated: {done}/{len(codes)}")
    try:
        q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
                           capture_output=True, text=True).stdout
        n = sum(1 for j in q.split() if j.startswith(("af_", "phenix_", "rm_")))
        print(f"  SLURM jobs in queue (af_/phenix_/rm_): {n}")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--codes", nargs="+", default=None)
        p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("refine")
    common(p)
    p.add_argument("--refiner", choices=["torchref", "phenix", "refmac"],
                   required=True)
    p.add_argument("--refmac-cycles", type=int, default=10,
                   help="REFMAC restrained-refinement cycles (refiner=refmac).")
    p.add_argument("--rigid-body", action="store_true")
    p.add_argument("--rigid-body-iter", type=int, default=100)
    # 25 (was 10): with the ADP locality restraint relaxed, B-factors move in
    # small per-cycle increments and only reach their data-supported spread
    # around cycle ~22; 10 cycles under-converges B (the "B-factor momentum"
    # behind the AF-start R-free deficit).
    p.add_argument("--n-cycles", type=int, default=25)
    p.add_argument("--adp-weight", type=float, default=0.1,
                   help="Group weight on the entire ADP loss (default 0.1).")
    p.add_argument("--xray-mode", default="ml")
    p.add_argument("--sigma-m-scale", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_refine)

    p = sub.add_parser("validate")
    common(p)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("analyze")
    common(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("migrate")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("status")
    p.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
