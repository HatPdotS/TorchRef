#!/usr/bin/env python
"""
CPU Thread Scaling + GPU Benchmark for TorchRef vs cctbx Structure Factor Calculation.

Measures how Fcalc performance scales with the number of CPU threads for both
TorchRef and cctbx, and optionally benchmarks GPU performance.

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

SCRIPT_DIR = Path(__file__).parent.resolve()
WORKER_SCRIPT = SCRIPT_DIR / "benchmark_worker.py"
PLOT_SCRIPT = SCRIPT_DIR / "plot_thread_scaling.py"
PYTHON = sys.executable


def _clear_stale_extension_locks():
    """Remove stale lock files from torch C++ extension cache.

    torch.utils.cpp_extension.load_inline uses file locks to prevent
    concurrent compilation. If a process is killed mid-load, the lock
    stays behind and all future imports block forever. Safe to remove
    when no compilation is actively running (i.e., at script start).
    """
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


def run_worker(n_threads: int, n_iterations: int, n_warmup: int,
               output_file: Path, device: str = "cpu") -> dict | None:
    """Launch a benchmark worker subprocess with the given thread count."""
    env = os.environ.copy()
    env["TORCHREF_NUM_THREADS"] = str(n_threads)
    # Also set low-level threading env vars to be thorough
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
            stderr=None,  # pass stderr through to terminal for live progress
            text=True,
            timeout=600,  # 10 min timeout per run
        )
        # Print worker stdout (final summary)
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.returncode != 0:
            print(f"  ERROR (returncode={result.returncode})")
            return None

        with open(output_file) as f:
            return json.load(f)

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: exceeded 600s")
        return None
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return None


def write_summary_csv(cpu_results: list[dict], gpu_result: dict | None,
                      output_path: Path):
    """Write combined results to a CSV file."""
    fieldnames = [
        "device", "n_threads",
        "torchref_mean", "torchref_min", "torchref_max",
        "torchref_speedup", "torchref_efficiency",
        "torchref_fwd_graph_mean", "torchref_fwd_graph_min", "torchref_fwd_graph_max",
        "torchref_fwd_graph_speedup",
        "torchref_bwd_only_mean", "torchref_bwd_only_min", "torchref_bwd_only_max",
        "torchref_bwd_only_speedup",
        "torchref_fwd_bwd_mean", "torchref_fwd_bwd_min", "torchref_fwd_bwd_max",
        "torchref_fwd_bwd_speedup",
        "cctbx_mean", "cctbx_min", "cctbx_max",
        "cctbx_speedup", "cctbx_efficiency",
        "n_iterations", "n_atoms", "n_reflections", "d_min",
    ]
    # Find single-thread baselines
    tr_baseline = cc_baseline = fb_baseline = fg_baseline = bo_baseline = None
    for r in cpu_results:
        if r["n_threads"] == 1:
            tr_baseline = r["torchref"]["mean_time"]
            fg_baseline = r["torchref_fwd_graph"]["mean_time"]
            bo_baseline = r["torchref_bwd_only"]["mean_time"]
            fb_baseline = r["torchref_fwd_bwd"]["mean_time"]
            cc_baseline = r["cctbx"]["mean_time"]
            break

    rows = []
    for r in sorted(cpu_results, key=lambda x: x["n_threads"]):
        tr = r["torchref"]
        fg = r["torchref_fwd_graph"]
        bo = r["torchref_bwd_only"]
        fb = r["torchref_fwd_bwd"]
        cc = r["cctbx"]
        nt = r["n_threads"]
        tr_speedup = tr_baseline / tr["mean_time"] if tr_baseline else float("nan")
        fg_speedup = fg_baseline / fg["mean_time"] if fg_baseline else float("nan")
        bo_speedup = bo_baseline / bo["mean_time"] if bo_baseline else float("nan")
        fb_speedup = fb_baseline / fb["mean_time"] if fb_baseline else float("nan")
        cc_speedup = cc_baseline / cc["mean_time"] if cc_baseline else float("nan")
        rows.append({
            "device": "cpu",
            "n_threads": nt,
            "torchref_mean": f"{tr['mean_time']:.6f}",
            "torchref_min": f"{tr['min_time']:.6f}",
            "torchref_max": f"{tr['max_time']:.6f}",
            "torchref_speedup": f"{tr_speedup:.3f}",
            "torchref_efficiency": f"{tr_speedup / nt:.3f}",
            "torchref_fwd_graph_mean": f"{fg['mean_time']:.6f}",
            "torchref_fwd_graph_min": f"{fg['min_time']:.6f}",
            "torchref_fwd_graph_max": f"{fg['max_time']:.6f}",
            "torchref_fwd_graph_speedup": f"{fg_speedup:.3f}",
            "torchref_bwd_only_mean": f"{bo['mean_time']:.6f}",
            "torchref_bwd_only_min": f"{bo['min_time']:.6f}",
            "torchref_bwd_only_max": f"{bo['max_time']:.6f}",
            "torchref_bwd_only_speedup": f"{bo_speedup:.3f}",
            "torchref_fwd_bwd_mean": f"{fb['mean_time']:.6f}",
            "torchref_fwd_bwd_min": f"{fb['min_time']:.6f}",
            "torchref_fwd_bwd_max": f"{fb['max_time']:.6f}",
            "torchref_fwd_bwd_speedup": f"{fb_speedup:.3f}",
            "cctbx_mean": f"{cc['mean_time']:.6f}",
            "cctbx_min": f"{cc['min_time']:.6f}",
            "cctbx_max": f"{cc['max_time']:.6f}",
            "cctbx_speedup": f"{cc_speedup:.3f}",
            "cctbx_efficiency": f"{cc_speedup / nt:.3f}",
            "n_iterations": r["n_iterations"],
            "n_atoms": r["n_atoms"],
            "n_reflections": r["n_reflections"],
            "d_min": r["d_min"],
        })

    if gpu_result:
        tr = gpu_result["torchref"]
        fg = gpu_result["torchref_fwd_graph"]
        bo = gpu_result["torchref_bwd_only"]
        fb = gpu_result["torchref_fwd_bwd"]
        tr_speedup = tr_baseline / tr["mean_time"] if tr_baseline else float("nan")
        fg_speedup = fg_baseline / fg["mean_time"] if fg_baseline else float("nan")
        bo_speedup = bo_baseline / bo["mean_time"] if bo_baseline else float("nan")
        fb_speedup = fb_baseline / fb["mean_time"] if fb_baseline else float("nan")
        rows.append({
            "device": "gpu",
            "n_threads": 0,
            "torchref_mean": f"{tr['mean_time']:.6f}",
            "torchref_min": f"{tr['min_time']:.6f}",
            "torchref_max": f"{tr['max_time']:.6f}",
            "torchref_speedup": f"{tr_speedup:.3f}",
            "torchref_efficiency": "",
            "torchref_fwd_graph_mean": f"{fg['mean_time']:.6f}",
            "torchref_fwd_graph_min": f"{fg['min_time']:.6f}",
            "torchref_fwd_graph_max": f"{fg['max_time']:.6f}",
            "torchref_fwd_graph_speedup": f"{fg_speedup:.3f}",
            "torchref_bwd_only_mean": f"{bo['mean_time']:.6f}",
            "torchref_bwd_only_min": f"{bo['min_time']:.6f}",
            "torchref_bwd_only_max": f"{bo['max_time']:.6f}",
            "torchref_bwd_only_speedup": f"{bo_speedup:.3f}",
            "torchref_fwd_bwd_mean": f"{fb['mean_time']:.6f}",
            "torchref_fwd_bwd_min": f"{fb['min_time']:.6f}",
            "torchref_fwd_bwd_max": f"{fb['max_time']:.6f}",
            "torchref_fwd_bwd_speedup": f"{fb_speedup:.3f}",
            "cctbx_mean": "",
            "cctbx_min": "",
            "cctbx_max": "",
            "cctbx_speedup": "",
            "cctbx_efficiency": "",
            "n_iterations": gpu_result["n_iterations"],
            "n_atoms": gpu_result["n_atoms"],
            "n_reflections": gpu_result["n_reflections"],
            "d_min": gpu_result["d_min"],
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    default_max = get_default_max_threads()

    parser = argparse.ArgumentParser(
        description="CPU thread scaling + GPU benchmark for TorchRef vs cctbx Fcalc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=None,
        help="Specific thread counts to test (e.g., 1 2 4 8 16). "
             f"Default: all integers from 1 to --max_threads.",
    )
    parser.add_argument(
        "--max_threads",
        type=int,
        default=default_max,
        help=f"Maximum thread count (default: {default_max}, auto-detected).",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=10,
        help="Number of timed iterations per configuration (default: 10).",
    )
    parser.add_argument(
        "--n_warmup",
        type=int,
        default=3,
        help="Number of warmup iterations per configuration (default: 3).",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Skip GPU benchmark even if a GPU is available.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for results. Default: results_<timestamp>/",
    )

    args = parser.parse_args()

    # Determine thread counts to test
    if args.threads:
        thread_counts = sorted(set(args.threads))
    else:
        # Default: powers of 2 up to max_threads (+ max_threads itself)
        thread_counts = []
        t = 1
        while t <= args.max_threads:
            thread_counts.append(t)
            t *= 2
        if thread_counts[-1] != args.max_threads:
            thread_counts.append(args.max_threads)
        thread_counts = sorted(set(thread_counts))

    # Clear stale C++ extension lock files (left behind by killed processes)
    _clear_stale_extension_locks()

    # Check GPU availability
    has_gpu = False if args.no_gpu else check_gpu_available()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = SCRIPT_DIR / f"results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TorchRef vs cctbx — Fcalc Benchmark")
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
    for i, n_threads in enumerate(thread_counts):
        progress = f"[{i + 1}/{total_runs}]"
        print(f"{progress} CPU with {n_threads} thread(s)...", flush=True)

        output_file = output_dir / f"threads_{n_threads:02d}.json"
        result = run_worker(n_threads, args.n_iterations, args.n_warmup, output_file)

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
    tr_baseline = cc_baseline = fb_baseline = fg_baseline = bo_baseline = None
    for r in cpu_results:
        if r["n_threads"] == 1:
            tr_baseline = r["torchref"]["mean_time"]
            fg_baseline = r["torchref_fwd_graph"]["mean_time"]
            bo_baseline = r["torchref_bwd_only"]["mean_time"]
            fb_baseline = r["torchref_fwd_bwd"]["mean_time"]
            cc_baseline = r["cctbx"]["mean_time"]
            break

    # Print summary table
    print("=" * 120)
    print("Summary")
    print("=" * 120)
    header = (
        f"{'Device':>10s} {'Threads':>8s} | "
        f"{'Fwd':>10s} {'Sp':>6s} | "
        f"{'Fwd(graph)':>10s} {'Sp':>6s} | "
        f"{'Bwd':>10s} {'Sp':>6s} | "
        f"{'Fwd+Bwd':>10s} {'Sp':>6s} | "
        f"{'cctbx':>10s} {'Sp':>6s}"
    )
    print(header)
    print("-" * len(header))

    for r in sorted(cpu_results, key=lambda x: x["n_threads"]):
        tr = r["torchref"]
        fg = r["torchref_fwd_graph"]
        bo = r["torchref_bwd_only"]
        fb = r["torchref_fwd_bwd"]
        cc = r["cctbx"]
        nt = r["n_threads"]
        tr_sp = tr_baseline / tr["mean_time"] if tr_baseline else float("nan")
        fg_sp = fg_baseline / fg["mean_time"] if fg_baseline else float("nan")
        bo_sp = bo_baseline / bo["mean_time"] if bo_baseline else float("nan")
        fb_sp = fb_baseline / fb["mean_time"] if fb_baseline else float("nan")
        cc_sp = cc_baseline / cc["mean_time"] if cc_baseline else float("nan")
        print(
            f"{'CPU':>10s} {nt:>8d} | "
            f"{tr['mean_time']:>9.4f}s {tr_sp:>5.1f}x | "
            f"{fg['mean_time']:>9.4f}s {fg_sp:>5.1f}x | "
            f"{bo['mean_time']:>9.4f}s {bo_sp:>5.1f}x | "
            f"{fb['mean_time']:>9.4f}s {fb_sp:>5.1f}x | "
            f"{cc['mean_time']:>9.4f}s {cc_sp:>5.1f}x"
        )

    if gpu_result:
        tr = gpu_result["torchref"]
        fg = gpu_result["torchref_fwd_graph"]
        bo = gpu_result["torchref_bwd_only"]
        fb = gpu_result["torchref_fwd_bwd"]
        gpu_name = gpu_result.get("gpu_name", "GPU")
        tr_sp = tr_baseline / tr["mean_time"] if tr_baseline else float("nan")
        fg_sp = fg_baseline / fg["mean_time"] if fg_baseline else float("nan")
        bo_sp = bo_baseline / bo["mean_time"] if bo_baseline else float("nan")
        fb_sp = fb_baseline / fb["mean_time"] if fb_baseline else float("nan")
        print(
            f"{gpu_name:>10s} {'':>8s} | "
            f"{tr['mean_time']:>9.4f}s {tr_sp:>5.1f}x | "
            f"{fg['mean_time']:>9.4f}s {fg_sp:>5.1f}x | "
            f"{bo['mean_time']:>9.4f}s {bo_sp:>5.1f}x | "
            f"{fb['mean_time']:>9.4f}s {fb_sp:>5.1f}x | "
            f"{'':>10s} {'':>6s}"
        )

    print()
    print(f"Results saved to: {csv_path}")

    # Auto-generate plots
    print()
    print("Generating plots...")
    try:
        subprocess.run(
            [PYTHON, str(PLOT_SCRIPT), str(output_dir)],
            timeout=60,
        )
    except Exception as e:
        print(f"Plot generation failed: {e}")


if __name__ == "__main__":
    main()
