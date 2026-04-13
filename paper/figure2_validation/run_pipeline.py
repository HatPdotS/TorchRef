#!/usr/bin/env python -u
"""
Validation pipeline for TorchRef refinement vs PHENIX.

Automates: submit TorchRef refinement SLURM jobs → wait → submit REFMAC 0-cycle
validation → wait → collect metrics → plot.

Subcommands
-----------
refine    Submit TorchRef refinement jobs for all structures.
validate  Submit REFMAC 0-cycle validation jobs.
analyze   Collect metrics and generate plots (no job submission).
auto      Full pipeline: refine + wait + validate + wait + analyze.
status    Check SLURM job status.

Examples
--------
::

    # Full automated pipeline (all 1000 structures, default hyperparameters)
    python run_pipeline.py auto --name my_run

    # Just submit refinement jobs
    python run_pipeline.py refine --name my_run

    # Re-run analysis and plots only
    python run_pipeline.py analyze --name my_run
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

# Force unbuffered output for SLURM
(
    sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stdout, "reconfigure")
    else None
)
os.environ["PYTHONUNBUFFERED"] = "1"

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent                     # figure2_validation/
PAPER_ROOT = BASE.parent                                   # paper/
REPO_ROOT = PAPER_ROOT.parent                              # torchref repo root
EXPERIMENTS = BASE / "experiments"
STRUCTURES_FILE = BASE / "structures.json"                 # 1000 PDB codes used in the paper
DATA = PAPER_ROOT / "data"                                 # symlink → scientific_testing/data
PHENIX = PAPER_ROOT / "phenix_refinements"                 # symlink → scientific_testing/.../refinements

PYTHON = sys.executable
REFINE_SCRIPT = str(REPO_ROOT / "torchref" / "cli" / "refine.py")

import shutil
REFMAC5 = shutil.which("refmac5") or "/afs/psi.ch/sys/psi.ra/MX/ccp4/7.1/ccp4-7.1/bin/refmac5"


def _phenix_pdb(code):
    return PHENIX / code / f"{code}_refined_001.pdb"


def _initial_pdb(code):
    return DATA / code / f"{code}_shaken.pdb"


def _mtz_file(code):
    return DATA / code / f"{code}.mtz"


# ──────────────────────────────────────────────────────────────────────────────
# Experiment Setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_experiment(name, structures, n_cycles):
    """Create experiment directory."""
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
        "mode": "refine",
        "xray_mode": "bhattacharyya",
        "sigma_m_scale": 1.0,
    }
    with open(exp_dir / "experiment.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Created experiment '{name}' at {exp_dir}")
    print(f"  Structures: {len(structures)}")
    print(f"  Macro cycles: {n_cycles}")
    return exp_dir


def load_experiment(name):
    """Load experiment metadata."""
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
    """Submit refinement SLURM jobs for all structures."""
    with open(exp_dir / "experiment.json") as f:
        exp = json.load(f)

    xray_mode = exp.get("xray_mode", "bhattacharyya")
    sigma_m_scale = exp.get("sigma_m_scale", 1.0)
    job_ids = {}
    submitted, skipped, missing = 0, 0, 0

    for code in exp["structures"]:
        pdb = _initial_pdb(code)
        mtz = _mtz_file(code)

        if not pdb.exists() or not mtz.exists():
            print(f"  SKIP {code}: missing input files")
            missing += 1
            continue

        outdir = exp_dir / "results" / code / "default"

        if (outdir / "refined.pdb").exists() and not force:
            skipped += 1
            continue

        outdir.mkdir(parents=True, exist_ok=True)

        cmd = (
            f"sbatch -p day -c 4 -t 04:00:00 --mem=8G "
            f"-o {outdir / 'out.log'} -e {outdir / 'error.log'} "
            f"-J ref_{code} "
            f"--wrap='"
            f"{PYTHON} -u {REFINE_SCRIPT} "
            f"-m {pdb} -sf {mtz} -o {outdir} "
            f"-n {exp['n_cycles']} "
            f"--mode {exp['mode']} "
            f"--xray-mode {xray_mode} "
            f"--sigma-m-scale {sigma_m_scale}"
            f"'"
        )

        if dry_run:
            print(f"  [DRY-RUN] {cmd}")
        else:
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, check=True
                )
                job_id = result.stdout.strip().split()[-1]
                job_ids[code] = job_id
                print(f"  Submitted {code}: Job {job_id}")
                submitted += 1
            except subprocess.CalledProcessError as e:
                print(f"  FAILED {code}: {e.stderr.strip()}")

    print(f"\nRefinement: submitted={submitted}, skipped={skipped}, missing={missing}")

    if job_ids:
        jf = exp_dir / "refine_job_ids.json"
        with open(jf, "w") as f:
            json.dump(job_ids, f, indent=2)

    return job_ids


# ──────────────────────────────────────────────────────────────────────────────
# SLURM Submission — REFMAC 0-cycle Validation
# ──────────────────────────────────────────────────────────────────────────────

def _refmac_log_complete(log_path):
    """Check if a REFMAC log contains results."""
    if not log_path.exists():
        return False
    try:
        with open(log_path) as f:
            return "Overall R factor" in f.read()
    except Exception:
        return False


def submit_refmac_jobs(exp_dir, dry_run=False, force=False):
    """Submit REFMAC 0-cycle validation jobs for TorchRef, initial, and PHENIX."""
    with open(exp_dir / "experiment.json") as f:
        exp = json.load(f)

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

        # TorchRef result
        pdb_path = exp_dir / "results" / code / "default" / "refined.pdb"
        if pdb_path.exists():
            tasks.append((str(pdb_path), str(mtz), f"{code}_default"))

        # Initial (shaken) PDB
        init_pdb = _initial_pdb(code)
        if init_pdb.exists():
            tasks.append((str(init_pdb), str(mtz), f"{code}_initial"))

        # Phenix reference
        phenix_pdb = _phenix_pdb(code)
        if phenix_pdb.exists():
            tasks.append((str(phenix_pdb), str(mtz), f"{code}_phenix"))

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
        jf = exp_dir / "refmac_job_ids.json"
        with open(jf, "w") as f:
            json.dump(job_ids, f, indent=2)

    return job_ids


# ──────────────────────────────────────────────────────────────────────────────
# Job Monitoring
# ──────────────────────────────────────────────────────────────────────────────

def get_user_jobs():
    """Get set of running/pending SLURM job IDs for current user."""
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i"],
            capture_output=True, text=True, check=True,
        )
        return set(result.stdout.strip().split())
    except Exception:
        return set()


def wait_for_jobs(job_ids, poll_interval=60, timeout=14400, label="jobs"):
    """Poll squeue until all jobs complete or timeout."""
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
    """Check an error log and return a short failure reason, or None if OK."""
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
    """Scan experiment for failed refinement jobs. Returns dict of failures."""
    with open(exp_dir / "experiment.json") as f:
        exp = json.load(f)

    failures = {}
    success, pending = 0, 0

    for code in exp["structures"]:
        outdir = exp_dir / "results" / code / "default"
        if (outdir / "refined.pdb").exists():
            success += 1
            continue

        error_log = outdir / "error.log"
        out_log = outdir / "out.log"
        reason = _detect_failure(error_log)
        if reason is None:
            reason = _detect_failure(out_log)
        if reason is None:
            if error_log.exists() or out_log.exists():
                reason = "No refined.pdb produced (unknown error)"
            else:
                pending += 1
                continue

        failures[code] = reason

    return failures, success, pending


def print_failure_report(failures, success, pending, label="Refinement"):
    """Print a summary of failures."""
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
            if len(items) <= 10:
                for item in items:
                    print(f"        {item}")
            else:
                for item in items[:5]:
                    print(f"        {item}")
                print(f"        ... and {len(items) - 5} more")


def check_experiment_status(exp_dir):
    """Print status of all jobs in an experiment."""
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
    """Parse a single REFMAC5 log for R-factors, geometry, and B-factor metrics."""
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
        "BOND":  r"Bond distances:\s*refined atoms\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
        "ANGL":  r"Bond angles\s*:\s*refined atoms\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
        "CHIRAL": r"Chiral centres:\s*refined atoms\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
        "PLANE": r"Planar groups:\s*refined atoms\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
    }
    for key, pat in geo_patterns.items():
        m = re.search(pat, content)
        result[f"rms{key}"] = float(m.group(2)) if m else np.nan

    b_patterns = {
        "B_mc_bond":   r"M\. chain bond B values:\s*refined atoms\s+(\d+)\s+([\d.]+)",
        "B_mc_angle":  r"M\. chain angle B values:\s*refined atoms\s+(\d+)\s+([\d.]+)",
        "B_sc_bond":   r"S\. chain bond B values:\s*refined atoms\s+(\d+)\s+([\d.]+)",
        "B_sc_angle":  r"S\. chain angle B values:\s*refined atoms\s+(\d+)\s+([\d.]+)",
        "B_longrange": r"Long range B values:\s*refined atoms\s+(\d+)\s+([\d.]+)",
    }
    for key, pat in b_patterns.items():
        m = re.search(pat, content)
        result[f"rms{key}"] = float(m.group(2)) if m else np.nan

    m = re.search(r"Overall\s+:\s+scale\s*=\s*([\d.]+),\s*B\s*=\s*([\d.-]+)", content)
    result["overall_B"] = float(m.group(2)) if m else np.nan

    return result


def _bfactor_stats_from_pdb(pdb_path):
    """Read B-factor stats from PDB ATOM records."""
    bfactors = []
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith(("ATOM  ", "HETATM")):
                    try:
                        b = float(line[60:66].strip())
                        bfactors.append(b)
                    except (ValueError, IndexError):
                        pass
    except Exception:
        return {}

    if not bfactors:
        return {}

    b = np.array(bfactors)
    log_b = np.log(np.clip(b, 1e-3, None))
    return {
        "b_mean": float(np.mean(b)),
        "b_std": float(np.std(b)),
        "b_min": float(np.min(b)),
        "b_max": float(np.max(b)),
        "log_b_std": float(np.std(log_b)),
    }


def collect_metrics(exp_dir):
    """Collect all metrics into CSVs."""
    with open(exp_dir / "experiment.json") as f:
        exp = json.load(f)

    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Internal metrics (from refinement_history.json + PDB B-factors) ──
    rows = []
    for code in exp["structures"]:
        # TorchRef result
        result_dir = exp_dir / "results" / code / "default"
        hist_file = result_dir / "refinement_history.json"
        pdb_file = result_dir / "refined.pdb"

        row = {"code": code, "variant": "default"}
        if hist_file.exists():
            with open(hist_file) as f:
                hist = json.load(f)
            stats = hist.get("final_statistics", {})
            row["r_work"] = stats.get("R_work", np.nan)
            row["r_free"] = stats.get("R_free", np.nan)
        if pdb_file.exists():
            row.update(_bfactor_stats_from_pdb(str(pdb_file)))
        rows.append(row)

        # Initial
        init_pdb = _initial_pdb(code)
        if init_pdb.exists():
            row = {"code": code, "variant": "initial"}
            row.update(_bfactor_stats_from_pdb(str(init_pdb)))
            rows.append(row)

        # Phenix
        phenix_pdb = _phenix_pdb(code)
        if phenix_pdb.exists():
            row = {"code": code, "variant": "phenix"}
            row.update(_bfactor_stats_from_pdb(str(phenix_pdb)))
            rows.append(row)

    df_internal = pd.DataFrame(rows)
    df_internal.to_csv(metrics_dir / "internal_metrics.csv", index=False)
    print(f"  {len(df_internal)} rows -> internal_metrics.csv")

    # ── REFMAC metrics ──
    log_dir = exp_dir / "refmac_logs"
    rows = []
    if log_dir.exists():
        for log_file in sorted(log_dir.glob("*.log")):
            stem = log_file.stem
            parsed = _parse_refmac_log(str(log_file))
            if parsed is None or np.isnan(parsed.get("r_work", np.nan)):
                continue

            parts = stem.split("_", 1)
            if len(parts) == 2:
                parsed["code"] = parts[0]
                parsed["variant"] = parts[1]
            else:
                parsed["code"] = stem
                parsed["variant"] = "unknown"
            rows.append(parsed)

    df_refmac = pd.DataFrame(rows)
    if not df_refmac.empty:
        df_refmac.to_csv(metrics_dir / "refmac_metrics.csv", index=False)
    print(f"  {len(df_refmac)} rows -> refmac_metrics.csv")

    # ── Summary table ──
    _build_summary(df_internal, df_refmac, metrics_dir)

    return df_internal, df_refmac


def collect_runtimes(exp_dir):
    """Collect wall-clock runtimes for TorchRef and Phenix into a CSV."""
    with open(exp_dir / "experiment.json") as f:
        exp = json.load(f)

    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for code in exp["structures"]:
        row = {"code": code}

        # TorchRef
        log = exp_dir / "results" / code / "default" / "out.log"
        if log.exists():
            text = log.read_text()
            m = re.search(r"Timing:\s+([\d.]+)s\s+wall", text)
            if m:
                row["wall_s_default"] = float(m.group(1))

        # Phenix
        phenix_log = PHENIX / code / f"{code}_refined_001.log"
        if phenix_log.exists():
            text = phenix_log.read_text()
            m = re.search(r"wall clock time:\s+([\d.]+)\s+s", text)
            if m:
                row["wall_s_phenix"] = float(m.group(1))

        if len(row) > 1:
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = metrics_dir / "runtimes.csv"
    df.to_csv(out_path, index=False)
    print(f"  {len(df)} rows -> runtimes.csv")
    return df


def _build_summary(df_internal, df_refmac, metrics_dir):
    """Build and print summary table of median metrics."""
    variants = ["default", "initial", "phenix"]

    summary_rows = []
    for variant in variants:
        row = {"variant": variant}

        mask = df_internal["variant"] == variant
        sub = df_internal.loc[mask]
        if "r_work" in sub.columns:
            row["R-work"] = sub["r_work"].median()
            row["R-free"] = sub["r_free"].median()
        if "log_b_std" in sub.columns:
            row["log(B) std"] = sub["log_b_std"].median()
        if "b_min" in sub.columns:
            row["B min"] = sub["b_min"].median()
            row["B max"] = sub["b_max"].median()

        if not df_refmac.empty:
            rmask = df_refmac["variant"] == variant
            rsub = df_refmac.loc[rmask]
            for col in ["rmsBOND", "rmsANGL", "rmsCHIRAL",
                        "rmsB_mc_bond", "rmsB_sc_bond", "rmsB_longrange"]:
                if col in rsub.columns:
                    row[col] = rsub[col].median()

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(metrics_dir / "summary.csv", index=False)

    print("\n" + "=" * 100)
    print("SUMMARY: Median across test structures")
    print("=" * 100)

    header = f"{'variant':>15s}  {'R-work':>8s}  {'R-free':>8s}  "
    header += f"{'MC-B':>8s}  {'SC-B':>8s}  {'LR-B':>8s}  "
    header += f"{'Bond':>8s}  {'Angle':>8s}  {'log(B)':>8s}"
    print(header)
    print("-" * 100)

    for _, row in df_summary.iterrows():
        line = f"{row['variant']:>15s}  "
        for col in ["R-work", "R-free"]:
            v = row.get(col, np.nan)
            line += f"{v:8.4f}  " if pd.notna(v) else f"{'—':>8s}  "
        for col in ["rmsB_mc_bond", "rmsB_sc_bond", "rmsB_longrange"]:
            v = row.get(col, np.nan)
            line += f"{v:8.2f}  " if pd.notna(v) else f"{'—':>8s}  "
        for col in ["rmsBOND", "rmsANGL"]:
            v = row.get(col, np.nan)
            line += f"{v:8.4f}  " if pd.notna(v) else f"{'—':>8s}  "
        v = row.get("log(B) std", np.nan)
        line += f"{v:8.4f}" if pd.notna(v) else f"{'—':>8s}"
        print(line)

    print("=" * 100)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _setup_matplotlib():
    plt.rcParams.update({
        "font.size": 13,
        "font.family": "sans-serif",
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })


def make_plots(exp_dir, df_internal, df_refmac):
    """Generate comparison plots (boxplots for TorchRef vs initial vs PHENIX)."""
    _setup_matplotlib()

    plot_dir = exp_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_variants = ["initial", "phenix", "default"]

    if not df_refmac.empty:
        _plot_rfactor_boxplots(df_refmac, all_variants, plot_dir)
        _plot_geometry_boxplots(df_refmac, all_variants, plot_dir)
        _plot_bfactor_rmsd_boxplots(df_refmac, all_variants, plot_dir)

    _plot_bfactor_spread(df_internal, all_variants, plot_dir)


def _plot_rfactor_boxplots(df_refmac, variants, plot_dir):
    """Box-and-whisker R-factor comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"initial": "lightyellow", "phenix": "lightblue", "default": "lightcoral"}

    for ax, col, title in [(ax1, "r_work", "R-work"), (ax2, "r_free", "R-free")]:
        data, labels = [], []
        for v in variants:
            vals = df_refmac.loc[df_refmac["variant"] == v, col].dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(v if v != "default" else "TorchRef")

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, v in zip(bp["boxes"], variants[:len(data)]):
                patch.set_facecolor(colors.get(v, "lightgray"))
            ax.set_title(title)

    fig.tight_layout()
    fig.savefig(plot_dir / "rfactor_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved rfactor_boxplot.png")


def _plot_geometry_boxplots(df_refmac, variants, plot_dir):
    """Bond, Angle, Chiral RMSD boxplots."""
    metrics = [("rmsBOND", "Bond RMSD (A)"),
               ("rmsANGL", "Angle RMSD (deg)"),
               ("rmsCHIRAL", "Chiral RMSD")]

    colors = {"initial": "lightyellow", "phenix": "lightblue", "default": "lightcoral"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (col, title) in zip(axes, metrics):
        if col not in df_refmac.columns:
            continue
        data, labels = [], []
        for v in variants:
            vals = df_refmac.loc[df_refmac["variant"] == v, col].dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(v if v != "default" else "TorchRef")
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, v in zip(bp["boxes"], variants[:len(data)]):
                patch.set_facecolor(colors.get(v, "lightgray"))
            ax.set_title(title)

    fig.tight_layout()
    fig.savefig(plot_dir / "geometry_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved geometry_boxplot.png")


def _plot_bfactor_rmsd_boxplots(df_refmac, variants, plot_dir):
    """MC bond B, SC bond B, Long-range B RMSD boxplots."""
    metrics = [("rmsB_mc_bond", "MC bond B RMSD"),
               ("rmsB_sc_bond", "SC bond B RMSD"),
               ("rmsB_longrange", "Long-range B RMSD")]

    colors = {"initial": "lightyellow", "phenix": "lightblue", "default": "lightcoral"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (col, title) in zip(axes, metrics):
        if col not in df_refmac.columns:
            continue
        data, labels = [], []
        for v in variants:
            vals = df_refmac.loc[df_refmac["variant"] == v, col].dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(v if v != "default" else "TorchRef")
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, v in zip(bp["boxes"], variants[:len(data)]):
                patch.set_facecolor(colors.get(v, "lightgray"))
            ax.set_title(title)

    fig.tight_layout()
    fig.savefig(plot_dir / "bfactor_rmsd_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved bfactor_rmsd_boxplot.png")


def _plot_bfactor_spread(df_internal, variants, plot_dir):
    """B-factor spread (std of log B) boxplot."""
    if "log_b_std" not in df_internal.columns:
        return

    colors = {"initial": "lightyellow", "phenix": "lightblue", "default": "lightcoral"}

    fig, ax = plt.subplots(figsize=(6, 5))
    data, labels = [], []
    for v in variants:
        vals = df_internal.loc[df_internal["variant"] == v, "log_b_std"].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(v if v != "default" else "TorchRef")

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, v in zip(bp["boxes"], variants[:len(data)]):
            patch.set_facecolor(colors.get(v, "lightgray"))
        ax.set_title("B-factor spread")
        ax.set_ylabel("std of log(B)")

    fig.tight_layout()
    fig.savefig(plot_dir / "logb_std_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved logb_std_boxplot.png")


# ──────────────────────────────────────────────────────────────────────────────
# Auto Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_analysis(exp_dir):
    """Collect metrics and generate plots (no job submission)."""
    print("\n[Collect] Collecting metrics...")
    df_internal, df_refmac = collect_metrics(exp_dir)

    print("\n[Collect] Collecting runtimes...")
    collect_runtimes(exp_dir)

    print("\n[Plot] Generating plots...")
    make_plots(exp_dir, df_internal, df_refmac)


def run_auto(exp_dir, poll_interval=60, timeout=14400, dry_run=False):
    """Full automated pipeline."""
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
    df_internal, df_refmac = collect_metrics(exp_dir)

    print("\n[Step 6/6] Generating plots...")
    make_plots(exp_dir, df_internal, df_refmac)

    if failures:
        failures_file = exp_dir / "failures.json"
        with open(failures_file, "w") as f:
            json.dump(failures, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"WARNING: {len(failures)} refinement jobs failed")
        print(f"Plots and metrics are based on {success} successful structures.")
        print(f"Details saved to: {failures_file}")
        print(f"{'=' * 60}")

    print(f"\nDone! Results at: {exp_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _load_default_structures():
    """Load the 997 PDB codes used in the paper from structures.json."""
    if STRUCTURES_FILE.exists():
        with open(STRUCTURES_FILE) as f:
            return json.load(f)
    print(f"Warning: {STRUCTURES_FILE} not found, scanning data directory...")
    codes = []
    if not DATA.exists():
        return codes
    for d in sorted(DATA.iterdir()):
        if not d.is_dir():
            continue
        code = d.name
        if (d / f"{code}_shaken.pdb").exists() and (d / f"{code}.mtz").exists():
            codes.append(code)
    return codes


def main():
    parser = argparse.ArgumentParser(
        description="Validation pipeline for TorchRef refinement vs PHENIX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline command")

    # Common args helper
    def add_common(p):
        p.add_argument("--name", type=str, required=True, help="Experiment name")
        p.add_argument("--structures", nargs="+", default=None,
                        help="PDB codes (default: all 997 from structures.json)")
        p.add_argument("--n-cycles", type=int, default=10,
                        help="Refinement macro cycles (default: 10)")
        p.add_argument("--dry-run", action="store_true",
                        help="Print commands without submitting")
        p.add_argument("--force", action="store_true",
                        help="Re-run even if results exist")

    # refine
    p_refine = subparsers.add_parser("refine", help="Submit TorchRef refinement jobs")
    add_common(p_refine)

    # validate
    p_validate = subparsers.add_parser("validate", help="Submit REFMAC validation jobs")
    p_validate.add_argument("--name", type=str, required=True, help="Experiment name")
    p_validate.add_argument("--dry-run", action="store_true")
    p_validate.add_argument("--force", action="store_true")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Collect metrics and plot")
    p_analyze.add_argument("--name", type=str, required=True, help="Experiment name")

    # auto
    p_auto = subparsers.add_parser("auto", help="Full pipeline: refine + validate + analyze")
    add_common(p_auto)
    p_auto.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between SLURM checks (default: 60)")
    p_auto.add_argument("--timeout", type=int, default=14400,
                        help="Max wait time in seconds (default: 14400 = 4h)")

    # status
    p_status = subparsers.add_parser("status", help="Check experiment job status")
    p_status.add_argument("--name", type=str, required=True, help="Experiment name")

    # list
    subparsers.add_parser("list", help="List existing experiments")

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
                n_structs = len(meta.get("structures", []))
                print(f"  {d.name:30s}  {n_structs} structures  "
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

    if args.command == "analyze":
        _, exp_dir = load_experiment(args.name)
        run_analysis(exp_dir)
        return

    # refine / auto — need experiment setup
    if args.structures:
        structures = args.structures
    else:
        structures = _load_default_structures()
        print(f"Using {len(structures)} structures from structures.json")

    exp_dir = setup_experiment(
        name=args.name, structures=structures, n_cycles=args.n_cycles,
    )

    if args.command == "refine":
        submit_refinement_jobs(exp_dir, dry_run=args.dry_run, force=args.force)

    elif args.command == "auto":
        run_auto(exp_dir, poll_interval=args.poll_interval, timeout=args.timeout,
                 dry_run=args.dry_run)


if __name__ == "__main__":
    main()
