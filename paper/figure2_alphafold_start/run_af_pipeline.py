#!/usr/bin/env python3
"""Self-contained pipeline for the AlphaFold-start benchmark arm.

Refines the Phaser-placed AlphaFold models (figure2_alphafold_start/placed/) with
TorchRef, REFMAC and phenix.refine, validates every model with a REFMAC5
zero-cycle run for a fair R-factor comparison, and collects the metrics.

The structure set is the MR-solved subset (mr_status.csv solved==1 / placed/*).

Subcommands
-----------
  refine   --refiner {torchref,phenix,refmac} [--rigid-body] [--rigid-body-iter N]
  validate                     REFMAC 0-cycle on af_initial + each refined arm
                               (validate.log co-located in each arm's run dir)
  analyze                      collect R-factors -> metrics/comparison.csv
  status                       queue + output summary

For Figure 2 itself, submit the three arms through
``analysis/submit_fig2_arms.py`` rather than calling ``refine`` once per engine:
it emits each structure's three arms adjacently, which Panel C's cross-engine
wall-clock comparison depends on. ``--refiner torchref`` here is the
weights/rigid-body variant used for side experiments; the canonical Figure-2
TorchRef arm is ``analysis/submit_local_arm.py``.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
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
METRICS = RUNS / "metrics"
MANIFEST = HERE / "manifest.json"
MR_STATUS = HERE / "mr_status.csv"

PYTHON = str(REPO / ".dev" / "bin" / "python")
REFINE_SCRIPT = str(REPO / "torchref" / "cli" / "refine.py")
# CCP4 8.0 from the node-local /opt tree. The AFS-hosted 7.1 this used to point at
# is gone: /afs is mounted only on the `afs-week` nodes now, so every refmac call
# (the REFMAC arm *and* the 0-cycle validation every arm is scored by) died with
# "refmac5: command not found". Verified present on epyc9335 (refmac5 reports
# "library version 8.0.003"). NB this is a REFMAC version change from the runs
# before 2026-08-11, which moves the validated R-factors and the geometry RMSZ
# table Figure 2b reads — not only TorchRef changed between those figures.
CCP4_SETUP = "/opt/psi/MX/ccp4/8.0/ccp4-8.0/bin/ccp4.setup-sh"

# Source the phenix env directly rather than `module load`: the modulefile uses an
# unsupported `module-url` command and there is no `module` on the batch nodes at
# all, so `module load phenix/...` fails with "Unable to locate a modulefile".
# Same approach (and same version) as analysis/crossscore_array.sh, so the
# refinement and the scoring run one phenix.
PHENIX_ENV = "/opt/psi/MX/phenix/1.21.1-5286/phenix-1.21.1-5286/phenix_env.sh"

# One CPU model for every refinement arm. Figure 2c compares wall-clock ACROSS
# engines, and `hour`/`day` span Xeon 6152/6230/6230R and EPYC 7453/9335, so
# without a constraint each engine draws a different hardware mix — a systematic
# per-engine bias, not noise that n=767 averages away. The same epyc9335 nodes
# belong to hour/day/week, so constraining does not change the partition each arm
# uses. Matches extended_figures/exF4/submit_singlecore.py, which re-times the
# same refinements at 1 core.
CPU_MODEL = "cpu_epyc9335"

# Refinement arms (af_initial = the placed search model, before refinement).
ARMS = ["af_initial", "refmac", "phenix_norb", "torchref"]


def _mtz(code):
    return DATA / code / f"{code}.mtz"


def model_path(code, arm):
    """Resolve the PDB for a given code+arm (may not exist yet)."""
    if arm == "af_initial":
        return PLACED / f"{code}_af.pdb"
    if arm == "torchref":
        return RUNS / "torchref" / code / "refined.pdb"
    if arm == "phenix_norb":
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
# refine — per-arm sbatch script builders
#
# Module-level so analysis/submit_fig2_arms.py can emit a structure's arms
# adjacently (see its docstring for why interleaving matters for Figure 2c)
# without re-deriving the flag sets.
# ──────────────────────────────────────────────────────────────────────────────

def script_refmac(code, pdb, mtz, outdir, cycles=10, constraint=CPU_MODEL):
    """Restrained REFMAC refinement from the placed AF model.

    Same engine / CCP4 setup / MTZ labels as the 0-cycle validation
    (:func:`cmd_validate`); the only differences are NCYCLES>0 and capturing
    XYZOUT. AF models are protein-only, so REFMAC's built-in monomer library
    suffices (no ligand dictionaries needed). REFMAC picks up FreeR_flag
    automatically, holding out the same free set the validation scores against.

    stdout lands in ``refmac.log``, which is also where the runtime ("Elapsed:")
    and the per-cycle R-factor table are parsed from.
    """
    out = outdir / "refined.pdb"
    return f"""#!/bin/bash
#SBATCH --job-name=af_refmac_{code}
#SBATCH --partition=hour
#SBATCH --constraint={constraint}
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
NCYCLES {cycles}
MAKE HYDR NO
END
EOF
cp output.pdb {out}
cd / && rm -rf $TEMP_DIR
"""


def script_phenix(code, pdb, mtz, outdir, cycles=10, constraint=CPU_MODEL):
    """phenix.refine from the placed AF model, no rigid body.

    The flag set is fixed by what the comparison has to be: automatic
    target-weight optimization is OFF because phenix's default runs a
    per-macrocycle weight grid-search (~2.4x slower) that no other engine does,
    and the Ramachandran restraint is OFF to match. AF models are protein-only,
    so no ligand restraints are needed.

    Runs in ``outdir`` because phenix writes beside its cwd and
    ``aggregate_figure_metrics.runtime_phenix`` reads the epoch stamps out of
    ``{{code}}_refined_001.log`` there. CRYST1 "None" in the Z field is sed-fixed
    first, or phenix rejects the file.

    Identical flags and phenix version to
    ``extended_figures/exF4/submit_singlecore.py`` apart from ``nproc``, so exF4 is
    a like-for-like re-timing of this run at 1 core.

    Every ``--`` flag precedes the positional arguments. phenix.refine's CLI takes
    two variadic positional groups (``files`` then ``phil``) and cannot split them
    across an option, so a flag in the middle makes it reject every argument after
    the model and data files.
    """
    return f"""#!/bin/bash
#SBATCH --job-name=af_phenix_{code}
#SBATCH --partition=day
#SBATCH --constraint={constraint}
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --output={outdir / 'phenix_refine.log'}
#SBATCH --error={outdir / 'phenix_refine.log'}

source {PHENIX_ENV}
cd {outdir}
if grep -q "^CRYST1.*None" {pdb}; then
    sed 's/\\(CRYST1.*P [0-9]* *[0-9]* *[0-9]* *\\)None/\\1   12/' {pdb} > input.pdb
else
    cp {pdb} input.pdb
fi
phenix.refine --overwrite --quiet \\
    input.pdb {mtz} \\
    output.prefix={code}_refined \\
    refinement.main.number_of_macro_cycles={cycles} \\
    refinement.main.nproc=4 \\
    refinement.refine.strategy=individual_sites+individual_adp+occupancies \\
    refinement.main.simulated_annealing=false \\
    refinement.target_weights.optimize_xyz_weight=false \\
    refinement.target_weights.optimize_adp_weight=false \\
    refinement.main.bulk_solvent_and_scale=true \\
    refinement.main.ordered_solvent=false \\
    refinement.ordered_solvent.mode=every_macro_cycle \\
    refinement.pdb_interpretation.ramachandran_plot_restraints.enabled=false \\
    write_def_file=false write_eff_file=false write_geo_file=false --quiet
"""


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
#SBATCH --mem=16G
#SBATCH --output={outdir / 'out.log'}
#SBATCH --error={outdir / 'error.log'}

{PYTHON} -u {REFINE_SCRIPT} \\
    -m {pdb} -sf {mtz} -o {outdir} \\
    -n {args.n_cycles} \\
    --mode separate \\
    --xray-mode {args.xray_mode} \\
    --weights '{{"adp": {args.adp_weight}}}'{rb_line}
"""
            if _sbatch(script, tmp / f"ref_{code}.sh", args.dry_run) or args.dry_run:
                submitted += 1

    else:  # refmac / phenix
        arm = "refmac" if args.refiner == "refmac" else "phenix_norb"
        build = script_refmac if args.refiner == "refmac" else script_phenix
        cycles = args.refmac_cycles if args.refiner == "refmac" else args.n_cycles
        tmp = RUNS / arm / "tmp_scripts"
        for code in codes:
            pdb, mtz = PLACED / f"{code}_af.pdb", _mtz(code)
            if not pdb.exists() or not mtz.exists():
                missing += 1
                continue
            outdir = RUNS / arm / code
            if model_path(code, arm).exists() and not args.force:
                skipped += 1
                continue
            if not args.dry_run:
                outdir.mkdir(parents=True, exist_ok=True)
            script = build(code, pdb, mtz, outdir, cycles=cycles,
                           constraint=args.constraint)
            if args.dry_run and submitted == 0:
                print("── example sbatch script ──")
                print(script)
                print("───────────────────────────")
            if _sbatch(script, tmp / f"{args.refiner}_{code}.sh", args.dry_run) or args.dry_run:
                submitted += 1

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
    # 10 is the canonical macrocycle count for the benchmark: it matches the
    # single-core runtime comparison (extended_figures/exF4) so Fig2c (4-core)
    # and exF4 (1-core) time the SAME refinement. NB: with the ADP locality
    # restraint relaxed, B-factors keep moving past cycle 10 (full data-supported
    # spread ~cycle 22), so raising this trades a small R-free gain for runtime;
    # the benchmark deliberately fixes it at 10.
    p.add_argument("--n-cycles", type=int, default=10)
    p.add_argument("--adp-weight", type=float, default=0.02,
                   help="Group weight on the entire ADP loss (locked default 0.02; "
                        "see DEFAULT_GROUP_WEIGHTS).")
    p.add_argument("--xray-mode", default="ml")
    p.add_argument("--constraint", default=CPU_MODEL,
                   help="SLURM --constraint for the refmac/phenix arms. One CPU "
                        "model across engines is what makes Figure 2c's "
                        "cross-engine wall-clock comparison meaningful; see "
                        "CPU_MODEL.")
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

    p = sub.add_parser("status")
    p.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
