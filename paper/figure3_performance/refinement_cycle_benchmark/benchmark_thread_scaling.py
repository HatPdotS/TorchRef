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


def discover_structures() -> list[str]:
    """All test structures with a matching tests/files/{pdb,mtz}/{ID} pair.

    Mirrors tests/conftest.py::all_structure_pairs — intersection of PDB and
    MTZ stems. Naturally excludes 1AK5 (pdb is 1AK5_with_H) and 7L84 (no mtz).
    """
    files_dir = REPO_ROOT / "tests" / "files"
    pdb_ids = {p.stem for p in (files_dir / "pdb").glob("*.pdb")}
    mtz_ids = {p.stem for p in (files_dir / "mtz").glob("*.mtz")}
    return sorted(pdb_ids & mtz_ids)


def run_worker(
    n_threads: int,
    n_iterations: int,
    n_warmup: int,
    output_file: Path,
    structure: str = "1DAW",
    device: str = "cpu",
    timeout: int = 900,
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
        "--structure", structure,
        "--output", str(output_file),
    ]

    try:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=None,  # pass stderr through to terminal
            text=True,
            timeout=timeout,
        )
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


FIELDNAMES = [
    "structure", "device", "n_threads",
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

_AGG_KEYS = [
    ("agg_fwd_no_grad", "aggregate_fwd_no_grad"),
    ("agg_fwd_graph", "aggregate_fwd_graph"),
    ("agg_bwd_only", "aggregate_bwd_only"),
    ("agg_fwd_bwd", "aggregate_fwd_bwd"),
]


def _structure_baselines(cpu_results: list[dict]) -> dict:
    """Per-structure 1-thread baselines: {structure: {full_key: mean_time}}."""
    baselines = {}
    for r in cpu_results:
        if r["n_threads"] == 1:
            baselines[r["structure"]] = {
                full: r[full]["mean_time"] for _, full in _AGG_KEYS
            }
    return baselines


def _aggregate_and_write(output_dir: Path):
    """Rebuild the combined summary.csv from every per-structure JSON in output_dir.

    Gathers the flat ``{structure}_threads_NN.json`` / ``{structure}_gpu.json``
    files written by per-structure / gpu-only shard jobs (run on separate nodes).
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


def write_summary_csv(
    cpu_results: list[dict], gpu_results: list[dict], output_path: Path
):
    """Write combined multi-structure results to a CSV file.

    `cpu_results` / `gpu_results` are flat lists across all structures; each
    dict carries its own "structure". Speedups are computed per structure
    against that structure's single-thread CPU run.
    """
    baselines = _structure_baselines(cpu_results)

    def _speedup(structure, key, mean_time):
        base = baselines.get(structure, {}).get(key)
        if base and mean_time > 0:
            return base / mean_time
        return float("nan")

    rows = []
    all_results = sorted(cpu_results, key=lambda x: (x["structure"], x["n_threads"]))
    all_results += sorted(gpu_results, key=lambda x: x["structure"])

    for r in all_results:
        is_gpu = r["device"] != "cpu"
        nt = 0 if is_gpu else r["n_threads"]
        structure = r["structure"]

        per_target_groups = _group_per_target(r.get("per_target", {}))

        row = {
            "structure": structure,
            "device": "gpu" if is_gpu else "cpu",
            "n_threads": nt,
        }
        for short, full in _AGG_KEYS:
            stats = r[full]
            row[f"{short}_mean"] = f"{stats['mean_time']:.6f}"
            row[f"{short}_min"] = f"{stats['min_time']:.6f}"
            row[f"{short}_max"] = f"{stats['max_time']:.6f}"
            row[f"{short}_speedup"] = (
                f"{_speedup(structure, full, stats['mean_time']):.3f}"
            )

        row["target_xray_mean"] = f"{per_target_groups.get('xray', 0):.6f}"
        row["target_geometry_total_mean"] = f"{per_target_groups.get('geometry', 0):.6f}"
        row["target_adp_total_mean"] = f"{per_target_groups.get('adp', 0):.6f}"
        row["n_iterations"] = r["n_iterations"]
        row["n_atoms"] = r["n_atoms"]
        row["n_reflections"] = r["n_reflections"]
        row["d_min"] = r["d_min"]
        rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
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
        "--structures", type=str, nargs="+", default=None,
        help="Structures to benchmark (resolved from tests/files/{pdb,mtz}/). "
             "Default: all matching test-data pairs.",
    )
    parser.add_argument(
        "--timeout", type=int, default=900,
        help="Per-worker subprocess timeout in seconds (default: 900). "
             "Large structures + per-target breakdown can be slow.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for results. Default: results_<timestamp>/",
    )
    parser.add_argument(
        "--gpu_only", action="store_true",
        help="Only run the GPU point per structure (skip the CPU thread sweep).",
    )
    parser.add_argument(
        "--no_summary", action="store_true",
        help="Skip writing summary.csv and auto-plotting (shard jobs; a later "
             "--aggregate job builds the combined summary).",
    )
    parser.add_argument(
        "--aggregate", action="store_true",
        help="Do not benchmark: load every per-structure JSON in --output_dir, "
             "rebuild the combined summary.csv, then exit.",
    )

    args = parser.parse_args()

    if args.aggregate:
        if not args.output_dir:
            print("--aggregate requires --output_dir")
            sys.exit(1)
        _aggregate_and_write(Path(args.output_dir).resolve())
        return

    structures = args.structures if args.structures else discover_structures()

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
        for nt in ([] if args.gpu_only else thread_counts):
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] {structure} CPU with "
                  f"{nt} thread(s)...", flush=True)
            output_file = output_dir / f"{structure}_threads_{nt:02d}.json"
            result = run_worker(nt, args.n_iterations, args.n_warmup,
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
    print("=" * 110)
    print("Summary")
    print("=" * 110)
    header = (
        f"{'Structure':>10s} {'Device':>8s} {'Thr':>5s} | "
        f"{'Fwd':>10s} {'Sp':>6s} | "
        f"{'Fwd(graph)':>10s} {'Sp':>6s} | "
        f"{'Bwd':>10s} {'Sp':>6s} | "
        f"{'Fwd+Bwd':>10s} {'Sp':>6s}"
    )
    print(header)
    print("-" * len(header))

    def _sp(r, full):
        base = baselines.get(r["structure"], {}).get(full)
        val = r[full]["mean_time"]
        return base / val if (base and val > 0) else float("nan")

    all_rows = sorted(cpu_results, key=lambda x: (x["structure"], x["n_threads"]))
    all_rows += sorted(gpu_results, key=lambda x: x["structure"])
    for r in all_rows:
        is_gpu = r["device"] != "cpu"
        dev = r.get("gpu_name", "GPU") if is_gpu else "CPU"
        nt = "-" if is_gpu else str(r["n_threads"])
        print(
            f"{r['structure']:>10s} {dev:>8s} {nt:>5s} | "
            f"{r['aggregate_fwd_no_grad']['mean_time']:>9.4f}s "
            f"{_sp(r, 'aggregate_fwd_no_grad'):>5.1f}x | "
            f"{r['aggregate_fwd_graph']['mean_time']:>9.4f}s "
            f"{_sp(r, 'aggregate_fwd_graph'):>5.1f}x | "
            f"{r['aggregate_bwd_only']['mean_time']:>9.4f}s "
            f"{_sp(r, 'aggregate_bwd_only'):>5.1f}x | "
            f"{r['aggregate_fwd_bwd']['mean_time']:>9.4f}s "
            f"{_sp(r, 'aggregate_fwd_bwd'):>5.1f}x"
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
