#!/usr/bin/env python
"""
CPU Thread Scaling + GPU Benchmark for TorchRef Refinement Cycle.

Measures how the full refinement loss evaluation (x-ray + geometry + ADP)
scales with the number of CPU threads, and optionally benchmarks GPU.

Designed to run with:
    srun -c 16 --gres=gpu:1 python benchmark_thread_scaling.py

Usage:
    python benchmark_thread_scaling.py
    python benchmark_thread_scaling.py --threads 1 2 4 8 16
    python benchmark_thread_scaling.py --max_threads 16 --n_iterations 20
    python benchmark_thread_scaling.py --no_gpu

Results are saved as individual JSON files per thread count, a GPU JSON, a
combined CSV summary, and PNG plots in the output directory.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def _find_repo_root() -> Path:
    """Walk up from this file to find the repo root (marker: pyproject.toml).

    `__file__` can be relative or staged under sbatch; resolving against a
    stable on-disk marker avoids those edge cases. Falls back to the
    TORCHREF_REPO_ROOT env var if the marker isn't found (e.g. installed
    package without a source tree).
    """
    env_root = os.environ.get("TORCHREF_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    p = Path(os.path.abspath(__file__)).parent
    for ancestor in [p, *p.parents]:
        if (ancestor / "pyproject.toml").is_file():
            return ancestor
    raise RuntimeError(
        "Could not locate repo root (no pyproject.toml found walking up "
        f"from {p}). Set TORCHREF_REPO_ROOT explicitly."
    )


REPO_ROOT = _find_repo_root()
SCRIPT_DIR = REPO_ROOT / "paper" / "figure3_performance" / "refinement_cycle_benchmark"
WORKER_SCRIPT = SCRIPT_DIR / "benchmark_worker.py"
PLOT_SCRIPT = SCRIPT_DIR.parent / "plot_figure3b.py"
PYTHON = sys.executable


def _clear_stale_extension_locks():
    """Remove stale lock files from torch C++ extension cache."""
    cache_dir = Path.home() / ".cache" / "torch_extensions"
    if not cache_dir.exists():
        return
    for lock_file in cache_dir.rglob("lock"):
        lock_file.unlink()
        print(f"Removed stale lock: {lock_file}")


def check_gpu_available() -> bool:
    """Check if CUDA is available without importing torch in the main process."""
    try:
        result = subprocess.run(
            [PYTHON, "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip() == "True"
    except Exception:
        return False


def get_default_max_threads() -> int:
    """Auto-detect available CPU count from SLURM or OS."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 16


def run_worker(
    n_threads: int,
    n_iterations: int,
    n_warmup: int,
    output_file: Path,
    device: str = "cpu",
) -> dict | None:
    """Launch a benchmark worker subprocess with the given thread count."""
    env = os.environ.copy()
    env["TORCHREF_NUM_THREADS"] = str(n_threads)
    env["OMP_NUM_THREADS"] = str(n_threads)
    env["MKL_NUM_THREADS"] = str(n_threads)
    env["OPENBLAS_NUM_THREADS"] = str(n_threads)

    cmd = [
        PYTHON,
        str(WORKER_SCRIPT),
        "--n_iterations", str(n_iterations),
        "--n_warmup", str(n_warmup),
        "--device", device,
        "--output", str(output_file),
    ]

    try:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=None,  # pass stderr through to terminal
            text=True,
            timeout=900,  # 15 min timeout (refinement init is slower)
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.returncode != 0:
            print(f"  ERROR (returncode={result.returncode})")
            return None

        with open(output_file) as f:
            return json.load(f)

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: exceeded 900s")
        return None
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return None


def _group_per_target(per_target: dict, mode: str = "fwd_bwd") -> dict:
    """Sum per-target mean times by group (xray, geometry, adp).

    Parameters
    ----------
    per_target : dict
        Per-target results with nested forward/backward/fwd_bwd dicts.
    mode : str
        Which timing mode to sum: 'forward', 'backward', or 'fwd_bwd'.
    """
    groups = {}
    for name, stats in per_target.items():
        parts = name.split("/")
        group = parts[0]
        groups[group] = groups.get(group, 0.0) + stats[mode]["mean_time"]
    return groups


def write_summary_csv(
    cpu_results: list[dict], gpu_result: dict | None, output_path: Path
):
    """Write combined results to a CSV file."""
    fieldnames = [
        "device", "n_threads",
        "agg_fwd_no_grad_mean", "agg_fwd_no_grad_min", "agg_fwd_no_grad_max",
        "agg_fwd_no_grad_speedup",
        "agg_fwd_graph_mean", "agg_fwd_graph_min", "agg_fwd_graph_max",
        "agg_fwd_graph_speedup",
        "agg_bwd_only_mean", "agg_bwd_only_min", "agg_bwd_only_max",
        "agg_bwd_only_speedup",
        "agg_fwd_bwd_mean", "agg_fwd_bwd_min", "agg_fwd_bwd_max",
        "agg_fwd_bwd_speedup",
        "target_xray_mean", "target_geometry_total_mean", "target_adp_total_mean",
        "n_iterations", "n_atoms", "n_reflections", "d_min",
    ]

    # Find single-thread baselines
    baselines = {}
    for r in cpu_results:
        if r["n_threads"] == 1:
            for key in ["aggregate_fwd_no_grad", "aggregate_fwd_graph",
                        "aggregate_bwd_only", "aggregate_fwd_bwd"]:
                baselines[key] = r[key]["mean_time"]
            break

    def _speedup(key, mean_time):
        base = baselines.get(key)
        if base and mean_time > 0:
            return base / mean_time
        return float("nan")

    rows = []
    all_results = sorted(cpu_results, key=lambda x: x["n_threads"])
    if gpu_result:
        all_results.append(gpu_result)

    for r in all_results:
        is_gpu = r["device"] != "cpu"
        nt = 0 if is_gpu else r["n_threads"]
        dev = "gpu" if is_gpu else "cpu"

        per_target_groups = _group_per_target(r.get("per_target", {}))

        row = {
            "device": dev,
            "n_threads": nt,
        }
        for short, full in [
            ("agg_fwd_no_grad", "aggregate_fwd_no_grad"),
            ("agg_fwd_graph", "aggregate_fwd_graph"),
            ("agg_bwd_only", "aggregate_bwd_only"),
            ("agg_fwd_bwd", "aggregate_fwd_bwd"),
        ]:
            stats = r[full]
            row[f"{short}_mean"] = f"{stats['mean_time']:.6f}"
            row[f"{short}_min"] = f"{stats['min_time']:.6f}"
            row[f"{short}_max"] = f"{stats['max_time']:.6f}"
            row[f"{short}_speedup"] = f"{_speedup(full, stats['mean_time']):.3f}"

        row["target_xray_mean"] = f"{per_target_groups.get('xray', 0):.6f}"
        row["target_geometry_total_mean"] = f"{per_target_groups.get('geometry', 0):.6f}"
        row["target_adp_total_mean"] = f"{per_target_groups.get('adp', 0):.6f}"
        row["n_iterations"] = r["n_iterations"]
        row["n_atoms"] = r["n_atoms"]
        row["n_reflections"] = r["n_reflections"]
        row["d_min"] = r["d_min"]
        rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    default_max = get_default_max_threads()

    parser = argparse.ArgumentParser(
        description="CPU thread scaling + GPU benchmark for refinement cycle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--threads", type=int, nargs="+", default=None,
        help="Specific thread counts to test (e.g., 1 2 4 8 16). "
             f"Default: powers of 2 up to --max_threads.",
    )
    parser.add_argument(
        "--max_threads", type=int, default=default_max,
        help=f"Maximum thread count (default: {default_max}, auto-detected).",
    )
    parser.add_argument(
        "--n_iterations", type=int, default=10,
        help="Number of timed iterations per configuration (default: 10).",
    )
    parser.add_argument(
        "--n_warmup", type=int, default=3,
        help="Number of warmup iterations per configuration (default: 3).",
    )
    parser.add_argument(
        "--no_gpu", action="store_true",
        help="Skip GPU benchmark even if a GPU is available.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for results. Default: results_<timestamp>/",
    )

    args = parser.parse_args()

    # Determine thread counts
    if args.threads:
        thread_counts = sorted(set(args.threads))
    else:
        thread_counts = []
        t = 1
        while t <= args.max_threads:
            thread_counts.append(t)
            t *= 2
        if thread_counts[-1] != args.max_threads:
            thread_counts.append(args.max_threads)
        thread_counts = sorted(set(thread_counts))

    _clear_stale_extension_locks()

    has_gpu = False if args.no_gpu else check_gpu_available()

    # Setup output directory — default to the canonical results location
    # under paper/figure3_performance/data/refinement_cycle/ so runs are
    # collected in one place regardless of which CWD the job ran from.
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            REPO_ROOT / "paper" / "figure3_performance" / "data"
            / "refinement_cycle" / f"results_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TorchRef — Refinement Cycle Benchmark")
    print("=" * 78)
    print(f"CPU threads:    {thread_counts}")
    print(f"GPU:            {'yes' if has_gpu else 'no'}")
    print(f"Iterations:     {args.n_iterations} (+ {args.n_warmup} warmup)")
    print(f"Output dir:     {output_dir}")
    print(f"Python:         {PYTHON}")
    print("=" * 78)
    print()

    total_runs = len(thread_counts) + (1 if has_gpu else 0)

    # --- CPU thread scaling ---
    cpu_results = []
    for i, nt in enumerate(thread_counts):
        progress = f"[{i + 1}/{total_runs}]"
        print(f"{progress} CPU with {nt} thread(s)...", flush=True)

        output_file = output_dir / f"threads_{nt:02d}.json"
        result = run_worker(nt, args.n_iterations, args.n_warmup, output_file)

        if result:
            cpu_results.append(result)
        print()

    # --- GPU benchmark ---
    gpu_result = None
    if has_gpu:
        progress = f"[{total_runs}/{total_runs}]"
        print(f"{progress} GPU benchmark...", flush=True)

        output_file = output_dir / "gpu.json"
        gpu_result = run_worker(
            1, args.n_iterations, args.n_warmup, output_file, device="cuda"
        )
        print()

    if not cpu_results and not gpu_result:
        print("No successful runs. Exiting.")
        sys.exit(1)

    # Write combined summary
    csv_path = output_dir / "summary.csv"
    write_summary_csv(cpu_results, gpu_result, csv_path)

    # Find baselines for display
    baselines = {}
    for r in cpu_results:
        if r["n_threads"] == 1:
            baselines["fwd"] = r["aggregate_fwd_no_grad"]["mean_time"]
            baselines["fg"] = r["aggregate_fwd_graph"]["mean_time"]
            baselines["bwd"] = r["aggregate_bwd_only"]["mean_time"]
            baselines["fb"] = r["aggregate_fwd_bwd"]["mean_time"]
            break

    # Print summary table
    print("=" * 110)
    print("Summary")
    print("=" * 110)
    header = (
        f"{'Device':>10s} {'Threads':>8s} | "
        f"{'Fwd':>10s} {'Sp':>6s} | "
        f"{'Fwd(graph)':>10s} {'Sp':>6s} | "
        f"{'Bwd':>10s} {'Sp':>6s} | "
        f"{'Fwd+Bwd':>10s} {'Sp':>6s}"
    )
    print(header)
    print("-" * len(header))

    def _sp(key, val):
        base = baselines.get(key)
        if base and val > 0:
            return base / val
        return float("nan")

    for r in sorted(cpu_results, key=lambda x: x["n_threads"]):
        fwd = r["aggregate_fwd_no_grad"]["mean_time"]
        fg = r["aggregate_fwd_graph"]["mean_time"]
        bwd = r["aggregate_bwd_only"]["mean_time"]
        fb = r["aggregate_fwd_bwd"]["mean_time"]
        nt = r["n_threads"]
        print(
            f"{'CPU':>10s} {nt:>8d} | "
            f"{fwd:>9.4f}s {_sp('fwd', fwd):>5.1f}x | "
            f"{fg:>9.4f}s {_sp('fg', fg):>5.1f}x | "
            f"{bwd:>9.4f}s {_sp('bwd', bwd):>5.1f}x | "
            f"{fb:>9.4f}s {_sp('fb', fb):>5.1f}x"
        )

    if gpu_result:
        fwd = gpu_result["aggregate_fwd_no_grad"]["mean_time"]
        fg = gpu_result["aggregate_fwd_graph"]["mean_time"]
        bwd = gpu_result["aggregate_bwd_only"]["mean_time"]
        fb = gpu_result["aggregate_fwd_bwd"]["mean_time"]
        gpu_name = gpu_result.get("gpu_name", "GPU")
        print(
            f"{gpu_name:>10s} {'':>8s} | "
            f"{fwd:>9.4f}s {_sp('fwd', fwd):>5.1f}x | "
            f"{fg:>9.4f}s {_sp('fg', fg):>5.1f}x | "
            f"{bwd:>9.4f}s {_sp('bwd', bwd):>5.1f}x | "
            f"{fb:>9.4f}s {_sp('fb', fb):>5.1f}x"
        )

    print()
    print(f"Results saved to: {csv_path}")

    # Auto-generate plots
    print()
    print("Generating plots...")
    try:
        subprocess.run(
            [PYTHON, str(PLOT_SCRIPT), "--results-dir", str(output_dir)],
            timeout=60,
        )
    except Exception as e:
        print(f"Plot generation failed: {e}")


if __name__ == "__main__":
    main()
