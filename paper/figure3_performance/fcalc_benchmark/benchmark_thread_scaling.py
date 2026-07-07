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
PLOT_SCRIPT = SCRIPT_DIR.parent / "plot_figure3a.py"
PYTHON = sys.executable


def _find_repo_root() -> Path:
    """Locate the repo root (marker: pyproject.toml); honor TORCHREF_REPO_ROOT."""
    env_root = os.environ.get("TORCHREF_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    for ancestor in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (ancestor / "pyproject.toml").is_file():
            return ancestor
    raise RuntimeError("Could not locate repo root (no pyproject.toml found).")


def discover_structures() -> list[str]:
    """All test structures with a matching tests/files/{pdb,mtz}/{ID} pair.

    Mirrors tests/conftest.py::all_structure_pairs — intersection of PDB and
    MTZ stems. Naturally excludes 1AK5 (pdb is 1AK5_with_H) and 7L84 (no mtz).
    """
    files_dir = _find_repo_root() / "tests" / "files"
    pdb_ids = {p.stem for p in (files_dir / "pdb").glob("*.pdb")}
    mtz_ids = {p.stem for p in (files_dir / "mtz").glob("*.mtz")}
    return sorted(pdb_ids & mtz_ids)


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
               output_file: Path, structure: str = "1DAW",
               device: str = "cpu", timeout: int = 600) -> dict | None:
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
        "--structure", structure,
        "--output", str(output_file),
    ]

    try:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=None,  # pass stderr through to terminal for live progress
            text=True,
            timeout=timeout,
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
        print(f"  TIMEOUT: exceeded {timeout}s")
        return None
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return None


FIELDNAMES = [
    "structure", "device", "n_threads",
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


def _row_for(r: dict, baselines: dict) -> dict:
    """Build one CSV row from a result dict, using per-structure baselines.

    `baselines` maps metric prefix -> single-thread mean_time for this
    structure (None if no 1-thread CPU run was recorded). Speedups are
    relative to that baseline. GPU rows use n_threads=0 and omit cctbx.
    """
    is_gpu = r["device"] != "cpu"
    nt = 0 if is_gpu else r["n_threads"]

    def _speedup(prefix):
        base = baselines.get(prefix)
        return base / r[prefix]["mean_time"] if base else float("nan")

    row = {
        "structure": r["structure"],
        "device": "gpu" if is_gpu else "cpu",
        "n_threads": nt,
        "n_iterations": r["n_iterations"],
        "n_atoms": r["n_atoms"],
        "n_reflections": r["n_reflections"],
        "d_min": r["d_min"],
    }
    for prefix in ("torchref", "torchref_fwd_graph",
                   "torchref_bwd_only", "torchref_fwd_bwd"):
        stats = r[prefix]
        sp = _speedup(prefix)
        row[f"{prefix}_mean"] = f"{stats['mean_time']:.6f}"
        row[f"{prefix}_min"] = f"{stats['min_time']:.6f}"
        row[f"{prefix}_max"] = f"{stats['max_time']:.6f}"
        row[f"{prefix}_speedup"] = f"{sp:.3f}"
    # Efficiency only meaningful for the no_grad CPU forward.
    tr_sp = _speedup("torchref")
    row["torchref_efficiency"] = f"{tr_sp / nt:.3f}" if not is_gpu else ""

    # cctbx is CPU-only.
    if not is_gpu and "cctbx" in r:
        cc = r["cctbx"]
        cc_base = baselines.get("cctbx")
        cc_sp = cc_base / cc["mean_time"] if cc_base else float("nan")
        row["cctbx_mean"] = f"{cc['mean_time']:.6f}"
        row["cctbx_min"] = f"{cc['min_time']:.6f}"
        row["cctbx_max"] = f"{cc['max_time']:.6f}"
        row["cctbx_speedup"] = f"{cc_sp:.3f}"
        row["cctbx_efficiency"] = f"{cc_sp / nt:.3f}"
    else:
        for k in ("cctbx_mean", "cctbx_min", "cctbx_max",
                  "cctbx_speedup", "cctbx_efficiency"):
            row[k] = ""
    return row


def _structure_baselines(cpu_results: list[dict]) -> dict:
    """Per-structure 1-thread baselines: {structure: {prefix: mean_time}}."""
    baselines = {}
    for r in cpu_results:
        if r["n_threads"] == 1:
            baselines[r["structure"]] = {
                prefix: r[prefix]["mean_time"]
                for prefix in ("torchref", "torchref_fwd_graph",
                               "torchref_bwd_only", "torchref_fwd_bwd", "cctbx")
                if prefix in r
            }
    return baselines


def _aggregate_and_write(output_dir: Path):
    """Rebuild the combined summary.csv from every per-structure JSON in output_dir.

    Used after per-structure / gpu-only shard jobs (run on separate nodes) have
    each written their flat ``{structure}_threads_NN.json`` / ``{structure}_gpu.json``
    files; this gathers them all into one canonical summary.csv.
    """
    cpu_results, gpu_results = [], []
    for p in sorted(output_dir.glob("*_threads_*.json")):
        try:
            with open(p) as f:
                cpu_results.append(json.load(f))
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    for p in sorted(output_dir.glob("*_gpu.json")):
        try:
            with open(p) as f:
                gpu_results.append(json.load(f))
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    csv_path = output_dir / "summary.csv"
    write_summary_csv(cpu_results, gpu_results, csv_path)
    n_struct = len({r["structure"] for r in cpu_results + gpu_results})
    print(f"Aggregated {len(cpu_results)} CPU + {len(gpu_results)} GPU JSONs "
          f"({n_struct} structures) -> {csv_path}")


def write_summary_csv(cpu_results: list[dict], gpu_results: list[dict],
                      output_path: Path):
    """Write combined multi-structure results to a CSV file.

    `cpu_results` / `gpu_results` are flat lists across all structures; each
    dict carries its own "structure". Speedups are computed per structure
    against that structure's single-thread CPU run.
    """
    baselines = _structure_baselines(cpu_results)

    rows = []
    for r in sorted(cpu_results, key=lambda x: (x["structure"], x["n_threads"])):
        rows.append(_row_for(r, baselines.get(r["structure"], {})))
    for r in sorted(gpu_results, key=lambda x: x["structure"]):
        rows.append(_row_for(r, baselines.get(r["structure"], {})))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
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
        "--structures",
        type=str,
        nargs="+",
        default=None,
        help="Structures to benchmark (resolved from tests/files/{pdb,mtz}/). "
             "Default: all matching test-data pairs.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-worker subprocess timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for results. Default: results_<timestamp>/",
    )
    parser.add_argument(
        "--gpu_only",
        action="store_true",
        help="Only run the GPU point per structure (skip the CPU thread sweep). "
             "For running the GPU measurement on a GPU node while the CPU "
             "thread-scaling runs as separate exclusive-node jobs.",
    )
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Skip writing summary.csv and auto-plotting at the end (for shard "
             "jobs that only emit per-structure JSONs; a later --aggregate job "
             "builds the combined summary).",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Do not benchmark: load every per-structure JSON in --output_dir, "
             "rebuild the combined summary.csv, then exit.",
    )

    args = parser.parse_args()

    if args.aggregate:
        if not args.output_dir:
            print("--aggregate requires --output_dir")
            sys.exit(1)
        _aggregate_and_write(Path(args.output_dir))
        return

    structures = args.structures if args.structures else discover_structures()

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
    print(f"Structures:     {structures}")
    print(f"CPU threads:    {thread_counts}")
    print(f"GPU:            {'yes' if has_gpu else 'no'}")
    print(f"Iterations:     {args.n_iterations} (+ {args.n_warmup} warmup)")
    print(f"Output dir:     {output_dir}")
    print(f"Python:         {PYTHON}")
    print("=" * 78)
    print()

    runs_per_struct = len(thread_counts) + (1 if has_gpu else 0)
    total_runs = runs_per_struct * len(structures)
    run_idx = 0

    cpu_results = []   # flat across structures
    gpu_results = []   # flat across structures

    for structure in structures:
        print(f"### Structure {structure} ###", flush=True)

        # --- CPU thread scaling ---
        for n_threads in ([] if args.gpu_only else thread_counts):
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] {structure} CPU with "
                  f"{n_threads} thread(s)...", flush=True)
            output_file = output_dir / f"{structure}_threads_{n_threads:02d}.json"
            result = run_worker(n_threads, args.n_iterations, args.n_warmup,
                                output_file, structure=structure,
                                timeout=args.timeout)
            if result:
                cpu_results.append(result)
            print()

        # --- GPU benchmark ---
        if has_gpu:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] {structure} GPU benchmark...",
                  flush=True)
            output_file = output_dir / f"{structure}_gpu.json"
            gpu_result = run_worker(1, args.n_iterations, args.n_warmup,
                                    output_file, structure=structure,
                                    device="cuda", timeout=args.timeout)
            if gpu_result:
                gpu_results.append(gpu_result)
            print()

    if not cpu_results and not gpu_results:
        print("No successful runs. Exiting.")
        sys.exit(1)

    if args.no_summary:
        print(f"\n--no_summary: wrote per-structure JSONs to {output_dir} "
              f"(run --aggregate to build summary.csv).")
        return

    # Write combined summary
    csv_path = output_dir / "summary.csv"
    write_summary_csv(cpu_results, gpu_results, csv_path)

    # Print a compact per-structure summary table
    baselines = _structure_baselines(cpu_results)
    print("=" * 120)
    print("Summary")
    print("=" * 120)
    header = (
        f"{'Structure':>10s} {'Device':>8s} {'Thr':>5s} | "
        f"{'Fwd':>10s} {'Sp':>6s} | "
        f"{'Fwd(graph)':>10s} {'Sp':>6s} | "
        f"{'Bwd':>10s} {'Sp':>6s} | "
        f"{'Fwd+Bwd':>10s} {'Sp':>6s} | "
        f"{'cctbx':>10s} {'Sp':>6s}"
    )
    print(header)
    print("-" * len(header))

    def _sp(r, prefix, struct):
        base = baselines.get(struct, {}).get(prefix)
        return base / r[prefix]["mean_time"] if base else float("nan")

    all_rows = sorted(cpu_results, key=lambda x: (x["structure"], x["n_threads"]))
    all_rows += sorted(gpu_results, key=lambda x: x["structure"])
    for r in all_rows:
        s = r["structure"]
        is_gpu = r["device"] != "cpu"
        dev = r.get("gpu_name", "GPU") if is_gpu else "CPU"
        nt = "-" if is_gpu else str(r["n_threads"])
        tr, fg = r["torchref"], r["torchref_fwd_graph"]
        bo, fb = r["torchref_bwd_only"], r["torchref_fwd_bwd"]
        cc = r.get("cctbx")
        cc_str = (f"{cc['mean_time']:>9.4f}s {_sp(r, 'cctbx', s):>5.1f}x"
                  if cc else f"{'':>10s} {'':>6s}")
        print(
            f"{s:>10s} {dev:>8s} {nt:>5s} | "
            f"{tr['mean_time']:>9.4f}s {_sp(r, 'torchref', s):>5.1f}x | "
            f"{fg['mean_time']:>9.4f}s {_sp(r, 'torchref_fwd_graph', s):>5.1f}x | "
            f"{bo['mean_time']:>9.4f}s {_sp(r, 'torchref_bwd_only', s):>5.1f}x | "
            f"{fb['mean_time']:>9.4f}s {_sp(r, 'torchref_fwd_bwd', s):>5.1f}x | "
            f"{cc_str}"
        )

    print()
    print(f"Results saved to: {csv_path}")

    # Auto-generate plots
    print()
    print("Generating plots...")
    try:
        subprocess.run(
            [PYTHON, str(PLOT_SCRIPT), "--results-dir", str(output_dir)],
            timeout=120,
        )
    except Exception as e:
        print(f"Plot generation failed: {e}")


if __name__ == "__main__":
    main()
