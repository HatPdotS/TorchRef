#!/usr/bin/env python
"""
Geometry weight screening pipeline for Extended Figure 1.

Runs 50 structures at each geometry weight (1–10) using ``torchref.refine``,
validates with REFMAC5 zero-cycle refinement, and collects metrics.

Subcommands
-----------
refine    Submit TorchRef refinement jobs for all (structure, weight) pairs.
validate  Submit REFMAC 0-cycle validation jobs.
collect   Collect metrics into CSV.
plot      Generate the extended figure.
auto      Full pipeline: refine → wait → validate → wait → collect → plot.
status    Check SLURM job status.
list      List existing experiments.

Examples
--------
::

    python run_pipeline.py auto --name sweep
    python run_pipeline.py refine --name sweep --dry-run
    python run_pipeline.py collect --name sweep
    python run_pipeline.py plot --name sweep
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent                     # exF1/
PAPER_ROOT = BASE.parent.parent                            # paper/
REPO_ROOT = PAPER_ROOT.parent                              # torchref repo root
EXPERIMENTS = BASE / "experiments"
STRUCTURES_FILE = BASE / "structures.json"
DATA = PAPER_ROOT / "data"

PYTHON = sys.executable
REFINE_SCRIPT = str(REPO_ROOT / "torchref" / "cli" / "refine.py")

import shutil
REFMAC5 = (
    shutil.which("refmac5")
    or "/afs/psi.ch/sys/psi.ra/MX/ccp4/7.1/ccp4-7.1/bin/refmac5"
)

WEIGHT_VALUES = list(range(1, 11))
N_CYCLES_DEFAULT = 10


def _initial_pdb(code):
    return DATA / code / f"{code}_shaken.pdb"


def _mtz_file(code):
    return DATA / code / f"{code}.mtz"


# ──────────────────────────────────────────────────────────────────────────────
# Experiment Setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_experiment(name, structures, n_cycles):
    exp_dir = EXPERIMENTS / name
    if (exp_dir / "experiment.json").exists():
        print(f"Experiment '{name}' already exists. Loading existing config.")
        return exp_dir

    exp_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": name,
        "created": datetime.now().isoformat(),
        "structures": structures,
        "n_cycles": n_cycles,
        "weight_values": WEIGHT_VALUES,
        "mode": "everything",
        "xray_mode": "bhattacharyya",
    }
    with open(exp_dir / "experiment.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Created experiment '{name}' at {exp_dir}")
    print(f"  Structures: {len(structures)}")
    print(f"  Weight values: {WEIGHT_VALUES}")
    print(f"  Total jobs: {len(structures) * len(WEIGHT_VALUES)}")
    return exp_dir


def load_experiment(name):
    exp_dir = EXPERIMENTS / name
    meta_path = exp_dir / "experiment.json"
    if not meta_path.exists():
        print(f"Error: Experiment '{name}' not found at {exp_dir}", file=sys.stderr)
        sys.exit(1)
    with open(meta_path) as f:
        exp = json.load(f)
    return exp, exp_dir


# ──────────────────────────────────────────────────────────────────────────────
# SLURM Submission — Refinement
# ──────────────────────────────────────────────────────────────────────────────

def submit_refinement_jobs(exp_dir, dry_run=False, force=False):
    with open(exp_dir / "experiment.json") as f:
        exp = json.load(f)

    weights = exp.get("weight_values", WEIGHT_VALUES)
    xray_mode = exp.get("xray_mode", "bhattacharyya")
    job_ids = {}
    submitted, skipped, missing = 0, 0, 0

    tmp_dir = exp_dir / "tmp_scripts"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for weight in weights:
        for code in exp["structures"]:
            pdb = _initial_pdb(code)
            mtz = _mtz_file(code)

            if not pdb.exists() or not mtz.exists():
                missing += 1
                continue

            outdir = exp_dir / "results" / code / f"w{weight:02d}"

            if (outdir / "refined.pdb").exists() and not force:
                skipped += 1
                continue

            outdir.mkdir(parents=True, exist_ok=True)

            script_content = f"""#!/bin/bash
#SBATCH --job-name=ref_{code}_w{weight:02d}
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --output={outdir / 'out.log'}
#SBATCH --error={outdir / 'error.log'}

{PYTHON} -u {REFINE_SCRIPT} \\
    -m {pdb} -sf {mtz} -o {outdir} \\
    -n {exp['n_cycles']} \\
    --mode {exp['mode']} \\
    --xray-mode {xray_mode} \\
    --weights '{{"geometry": {weight}}}'
"""
            label = f"{code}_w{weight:02d}"
            script_file = tmp_dir / f"ref_{label}.sh"
            with open(script_file, "w") as f:
                f.write(script_content)

            if dry_run:
                print(f"  [DRY-RUN] sbatch {script_file}")
            else:
                try:
                    result = subprocess.run(
                        ["sbatch", str(script_file)],
                        capture_output=True, text=True, check=True,
                    )
                    job_id = result.stdout.strip().split()[-1]
                    job_ids[label] = job_id
                    submitted += 1
                except subprocess.CalledProcessError as e:
                    print(f"  FAILED {label}: {e.stderr.strip()}")

            if script_file.exists() and not dry_run:
                script_file.unlink()

    print(f"\nRefinement: submitted={submitted}, skipped={skipped}, missing={missing}")

    if job_ids:
        with open(exp_dir / "refine_job_ids.json", "w") as f:
            json.dump(job_ids, f, indent=2)

    return job_ids


# ──────────────────────────────────────────────────────────────────────────────
# SLURM Submission — REFMAC 0-cycle Validation
# ──────────────────────────────────────────────────────────────────────────────

def _refmac_log_complete(log_path):
    if not log_path.exists():
        return False
    try:
        with open(log_path) as f:
            return "Overall R factor" in f.read()
    except Exception:
        return False


def submit_refmac_jobs(exp_dir, dry_run=False, force=False):
    with open(exp_dir / "experiment.json") as f:
        exp = json.load(f)

    weights = exp.get("weight_values", WEIGHT_VALUES)
    log_dir = exp_dir / "refmac_logs"
    tmp_dir = log_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    job_ids = {}
    submitted, skipped = 0, 0

    tasks = []
    for code in exp["structures"]:
        mtz = _mtz_file(code)
        if not mtz.exists():
            continue

        for weight in weights:
            pdb_path = exp_dir / "results" / code / f"w{weight:02d}" / "refined.pdb"
            if pdb_path.exists():
                tasks.append((str(pdb_path), str(mtz), f"{code}_w{weight:02d}"))

        init_pdb = _initial_pdb(code)
        if init_pdb.exists():
            tasks.append((str(init_pdb), str(mtz), f"{code}_initial"))

    for pdb_path, mtz_path, label in tasks:
        log_file = log_dir / f"{label}.log"

        if _refmac_log_complete(log_file) and not force:
            skipped += 1
            continue

        script_content = f"""#!/bin/bash
#SBATCH --job-name=rm_{label[:20]}
#SBATCH --output={log_file}
#SBATCH --error={log_file}
#SBATCH --time=00:10:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

source /afs/psi.ch/sys/psi.ra/MX/ccp4/7.1/ccp4-7.1/bin/ccp4.setup-sh
TEMP_DIR=/tmp/refmac_{label}_${{SLURM_JOB_ID}}
mkdir -p $TEMP_DIR && cd $TEMP_DIR
export CCP4_SCR=$TEMP_DIR
cp {pdb_path} input.pdb
cp {mtz_path} input.mtz
refmac5 HKLIN input.mtz HKLOUT output.mtz XYZIN input.pdb XYZOUT output.pdb << EOF
NCYCLES 0
MAKE HYDR NO
END
EOF
cd / && rm -rf $TEMP_DIR
"""
        script_file = tmp_dir / f"rm_{label}.sh"
        with open(script_file, "w") as f:
            f.write(script_content)

        if dry_run:
            print(f"  [DRY-RUN] sbatch {script_file}")
        else:
            try:
                result = subprocess.run(
                    ["sbatch", str(script_file)],
                    capture_output=True, text=True, check=True,
                )
                job_id = result.stdout.strip().split()[-1]
                job_ids[label] = job_id
                submitted += 1
            except subprocess.CalledProcessError as e:
                print(f"  FAILED {label}: {e.stderr.strip()}")

        if script_file.exists() and not dry_run:
            script_file.unlink()

    print(f"REFMAC: submitted={submitted}, skipped={skipped}")

    if job_ids:
        with open(exp_dir / "refmac_job_ids.json", "w") as f:
            json.dump(job_ids, f, indent=2)

    return job_ids


# ──────────────────────────────────────────────────────────────────────────────
# Job Monitoring
# ──────────────────────────────────────────────────────────────────────────────

def get_user_jobs():
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i"],
            capture_output=True, text=True, check=True,
        )
        return set(result.stdout.strip().split())
    except Exception:
        return set()


def wait_for_jobs(job_ids, poll_interval=60, timeout=14400, label="jobs"):
    if not job_ids:
        return

    all_ids = set(job_ids.values())
    total = len(all_ids)
    start = time.time()

    print(f"Waiting for {total} {label} (poll every {poll_interval}s, "
          f"timeout {timeout // 60}min)...")
    sys.stdout.flush()

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"  TIMEOUT after {elapsed / 60:.0f} min")
            break

        running = get_user_jobs()
        remaining = all_ids & running
        done = total - len(remaining)

        print(f"  [{elapsed / 60:.0f}min] {done}/{total} complete, "
              f"{len(remaining)} running/pending")
        sys.stdout.flush()

        if not remaining:
            print(f"  All {total} {label} complete.")
            break

        time.sleep(poll_interval)


def _detect_failure(error_log_path):
    if not error_log_path.exists():
        return None
    try:
        content = error_log_path.read_text()
    except Exception:
        return None
    if not content.strip():
        return None
    if "Traceback (most recent call last)" in content:
        lines = content.strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith("File ") and not line.startswith("Traceback"):
                return line[:120]
        return "Python traceback"
    if "Illegal instruction" in content:
        return "Illegal instruction (core dumped)"
    if "oom-kill" in content.lower() or "Out of memory" in content:
        return "Out of memory"
    if "CANCELLED" in content:
        return "SLURM job cancelled"
    if "DUE TO TIME LIMIT" in content:
        return "SLURM time limit"
    return None


def scan_failures(exp_dir):
    with open(exp_dir / "experiment.json") as f:
        exp = json.load(f)

    weights = exp.get("weight_values", WEIGHT_VALUES)
    failures = {}
    success, pending = 0, 0

    for code in exp["structures"]:
        for weight in weights:
            outdir = exp_dir / "results" / code / f"w{weight:02d}"
            label = f"{code}_w{weight:02d}"

            if (outdir / "refined.pdb").exists():
                success += 1
                continue

            reason = _detect_failure(outdir / "error.log")
            if reason is None:
                reason = _detect_failure(outdir / "out.log")
            if reason is None:
                if (outdir / "error.log").exists() or (outdir / "out.log").exists():
                    reason = "No refined.pdb produced (unknown error)"
                else:
                    pending += 1
                    continue

            failures[label] = reason

    return failures, success, pending


def print_failure_report(failures, success, pending, label="Refinement"):
    total = success + len(failures) + pending
    print(f"\n{label} results: {success}/{total} succeeded, "
          f"{len(failures)} failed, {pending} not started")

    if failures:
        by_reason = {}
        for code, reason in failures.items():
            by_reason.setdefault(reason, []).append(code)

        print(f"\nFailures by type:")
        for reason, items in sorted(by_reason.items(), key=lambda x: -len(x[1])):
            print(f"  [{len(items)}x] {reason}")
            for item in items[:5]:
                print(f"        {item}")
            if len(items) > 5:
                print(f"        ... and {len(items) - 5} more")


def check_experiment_status(exp_dir):
    running_jobs = get_user_jobs()

    print("Refinement status:")
    failures, success, pending = scan_failures(exp_dir)
    print_failure_report(failures, success, pending, "Refinement")

    log_dir = exp_dir / "refmac_logs"
    if log_dir.exists():
        logs = list(log_dir.glob("*.log"))
        complete = sum(1 for l in logs if _refmac_log_complete(l))
        print(f"\nREFMAC logs: {complete}/{len(logs)} complete")

    for jf_name in ["refine_job_ids.json", "refmac_job_ids.json"]:
        jf = exp_dir / jf_name
        if jf.exists():
            with open(jf) as f:
                jids = json.load(f)
            in_queue = sum(1 for jid in jids.values() if jid in running_jobs)
            if in_queue:
                print(f"  {jf_name}: {in_queue} jobs still in SLURM queue")


# ──────────────────────────────────────────────────────────────────────────────
# Metric Collection
# ──────────────────────────────────────────────────────────────────────────────

def _parse_refmac_log(log_file):
    try:
        with open(log_file) as f:
            content = f.read()
    except Exception:
        return None

    result = {}

    m = re.search(r"Overall R factor\s+=\s+([\d.]+)", content)
    result["r_work"] = float(m.group(1)) if m else np.nan
    m = re.search(r"Free R factor\s+=\s+([\d.]+)", content)
    result["r_free"] = float(m.group(1)) if m else np.nan

    geo_patterns = {
        "BOND":   r"Bond distances:\s*refined atoms\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
        "ANGL":   r"Bond angles\s*:\s*refined atoms\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
        "CHIRAL": r"Chiral centres:\s*refined atoms\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
        "PLANE":  r"Planar groups:\s*refined atoms\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
    }
    for key, pat in geo_patterns.items():
        m = re.search(pat, content)
        result[f"rms{key}"] = float(m.group(2)) if m else np.nan
        result[f"sig{key}"] = float(m.group(3)) if m else np.nan

    b_patterns = {
        "B_mc_bond":   r"M\. chain bond B values:\s*refined atoms\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
        "B_mc_angle":  r"M\. chain angle B values:\s*refined atoms\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
        "B_sc_bond":   r"S\. chain bond B values:\s*refined atoms\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
        "B_sc_angle":  r"S\. chain angle B values:\s*refined atoms\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
        "B_longrange": r"Long range B values:\s*refined atoms\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
    }
    for key, pat in b_patterns.items():
        m = re.search(pat, content)
        result[f"rms{key}"] = float(m.group(2)) if m else np.nan
        result[f"sig{key}"] = float(m.group(3)) if m else np.nan

    return result


def collect_metrics(exp_dir):
    with open(exp_dir / "experiment.json") as f:
        exp = json.load(f)

    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    log_dir = exp_dir / "refmac_logs"
    rows = []
    if log_dir.exists():
        for log_file in sorted(log_dir.glob("*.log")):
            stem = log_file.stem
            parsed = _parse_refmac_log(str(log_file))
            if parsed is None or np.isnan(parsed.get("r_work", np.nan)):
                continue

            # Parse label: {code}_w{weight:02d} or {code}_initial
            m = re.match(r"^(.+?)_(w\d+|initial)$", stem)
            if not m:
                continue

            parsed["code"] = m.group(1)
            variant = m.group(2)
            if variant == "initial":
                parsed["weight"] = 0
            else:
                parsed["weight"] = int(variant[1:])
            rows.append(parsed)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(metrics_dir / "refmac_metrics.csv", index=False)
    print(f"  {len(df)} rows -> refmac_metrics.csv")

    # Summary: median per weight
    if not df.empty:
        summary_rows = []
        for w in sorted(df["weight"].unique()):
            sub = df[df["weight"] == w]
            row = {"weight": w, "n": len(sub)}
            for col in ["r_work", "r_free", "rmsBOND", "rmsANGL",
                        "rmsCHIRAL", "rmsPLANE", "rmsB_mc_bond",
                        "rmsB_sc_bond", "rmsB_longrange"]:
                if col in sub.columns:
                    row[f"{col}_median"] = sub[col].median()
            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(metrics_dir / "summary.csv", index=False)
        _print_summary(df_summary)

    return df


def _print_summary(df_summary):
    print("\n" + "=" * 90)
    print("SUMMARY: Median across structures per geometry weight")
    print("=" * 90)
    header = (f"{'weight':>8s}  {'n':>4s}  {'R-work':>8s}  {'R-free':>8s}  "
              f"{'Bond':>8s}  {'Angle':>8s}  {'Chiral':>8s}")
    print(header)
    print("-" * 90)

    for _, row in df_summary.iterrows():
        w = int(row["weight"])
        label = f"w={w}" if w > 0 else "initial"
        n = int(row["n"])
        line = f"{label:>8s}  {n:4d}  "
        for col in ["r_work_median", "r_free_median"]:
            v = row.get(col, np.nan)
            line += f"{v:8.4f}  " if pd.notna(v) else f"{'—':>8s}  "
        for col in ["rmsBOND_median", "rmsANGL_median", "rmsCHIRAL_median"]:
            v = row.get(col, np.nan)
            line += f"{v:8.4f}  " if pd.notna(v) else f"{'—':>8s}  "
        print(line)

    print("=" * 90)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def make_plots(exp_dir, df):
    if df.empty:
        print("  No data to plot.")
        return

    from plot_exF1 import plot_weight_sweep

    metrics_csv = exp_dir / "metrics" / "refmac_metrics.csv"
    output_dir = BASE / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_weight_sweep(str(metrics_csv), str(output_dir / "exF1.png"))


# ──────────────────────────────────────────────────────────────────────────────
# Auto Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_auto(exp_dir, poll_interval=60, timeout=14400, dry_run=False):
    print("\n[Step 1/6] Submitting refinement jobs...")
    refine_jobs = submit_refinement_jobs(exp_dir, dry_run=dry_run)
    if dry_run:
        print("Dry run complete.")
        return

    if refine_jobs:
        print(f"\n[Step 2/6] Waiting for {len(refine_jobs)} refinement jobs...")
        wait_for_jobs(refine_jobs, poll_interval=poll_interval, timeout=timeout,
                      label="refinement jobs")

    failures, success, pending = scan_failures(exp_dir)
    print_failure_report(failures, success, pending, "Refinement")

    print("\n[Step 3/6] Submitting REFMAC validation jobs...")
    refmac_jobs = submit_refmac_jobs(exp_dir)

    if refmac_jobs:
        print(f"\n[Step 4/6] Waiting for {len(refmac_jobs)} REFMAC jobs...")
        wait_for_jobs(refmac_jobs, poll_interval=30, timeout=1800,
                      label="REFMAC jobs")

    print("\n[Step 5/6] Collecting metrics...")
    df = collect_metrics(exp_dir)

    print("\n[Step 6/6] Generating plots...")
    make_plots(exp_dir, df)

    if failures:
        failures_file = exp_dir / "failures.json"
        with open(failures_file, "w") as f:
            json.dump(failures, f, indent=2)

    print(f"\nDone! Results at: {exp_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _load_structures():
    if STRUCTURES_FILE.exists():
        with open(STRUCTURES_FILE) as f:
            return json.load(f)
    print(f"Error: {STRUCTURES_FILE} not found", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Geometry weight screening pipeline (Extended Figure 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline command")

    def add_common(p):
        p.add_argument("--name", type=str, required=True, help="Experiment name")
        p.add_argument("--n-cycles", type=int, default=N_CYCLES_DEFAULT,
                       help=f"Refinement macro cycles (default: {N_CYCLES_DEFAULT})")
        p.add_argument("--dry-run", action="store_true",
                       help="Print commands without submitting")
        p.add_argument("--force", action="store_true",
                       help="Re-run even if results exist")

    p_refine = subparsers.add_parser("refine", help="Submit refinement jobs")
    add_common(p_refine)

    p_validate = subparsers.add_parser("validate", help="Submit REFMAC validation")
    p_validate.add_argument("--name", type=str, required=True)
    p_validate.add_argument("--dry-run", action="store_true")
    p_validate.add_argument("--force", action="store_true")

    p_collect = subparsers.add_parser("collect", help="Collect metrics")
    p_collect.add_argument("--name", type=str, required=True)

    p_plot = subparsers.add_parser("plot", help="Generate figure")
    p_plot.add_argument("--name", type=str, required=True)

    p_auto = subparsers.add_parser("auto", help="Full pipeline")
    add_common(p_auto)
    p_auto.add_argument("--poll-interval", type=int, default=60)
    p_auto.add_argument("--timeout", type=int, default=14400)

    p_status = subparsers.add_parser("status", help="Check job status")
    p_status.add_argument("--name", type=str, required=True)

    subparsers.add_parser("list", help="List experiments")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "list":
        if not EXPERIMENTS.exists():
            print("No experiments found.")
            return
        for d in sorted(EXPERIMENTS.iterdir()):
            if (d / "experiment.json").exists():
                with open(d / "experiment.json") as f:
                    meta = json.load(f)
                n = len(meta.get("structures", []))
                w = len(meta.get("weight_values", []))
                print(f"  {d.name:30s}  {n} structures × {w} weights  "
                      f"({meta.get('created', '?')[:10]})")
        return

    if args.command == "status":
        _, exp_dir = load_experiment(args.name)
        check_experiment_status(exp_dir)
        return

    if args.command == "validate":
        _, exp_dir = load_experiment(args.name)
        submit_refmac_jobs(exp_dir, dry_run=args.dry_run, force=args.force)
        return

    if args.command == "collect":
        _, exp_dir = load_experiment(args.name)
        collect_metrics(exp_dir)
        return

    if args.command == "plot":
        _, exp_dir = load_experiment(args.name)
        metrics_csv = exp_dir / "metrics" / "refmac_metrics.csv"
        if not metrics_csv.exists():
            print("No metrics found. Run 'collect' first.", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(metrics_csv)
        make_plots(exp_dir, df)
        return

    # refine / auto
    structures = _load_structures()
    print(f"Using {len(structures)} structures")

    exp_dir = setup_experiment(
        name=args.name, structures=structures,
        n_cycles=getattr(args, "n_cycles", N_CYCLES_DEFAULT),
    )

    if args.command == "refine":
        submit_refinement_jobs(exp_dir, dry_run=args.dry_run, force=args.force)
    elif args.command == "auto":
        run_auto(exp_dir, poll_interval=args.poll_interval,
                 timeout=args.timeout, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
