#!/usr/bin/env python
"""
Orchestrator for the SFcalculator vs TorchRef (SF_DS / SF_FFT) benchmark.

Launches one benchmark_worker.py subprocess per (structure x method) so each
method's peak memory is measured in a clean process. Collects per-run JSONs and
writes a combined summary.csv.

Run via SLURM (see submit_cpu.sbatch / submit_gpu.sbatch):
    python run_benchmarks.py --device cpu
    python run_benchmarks.py --device cuda

cctbx is benchmarked on CPU only.
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
WORKER = SCRIPT_DIR / "benchmark_worker.py"
PLOT_SCRIPT = SCRIPT_DIR / "plot_results.py"
PYTHON = sys.executable

# Mixed size series: 3GR5/1DAW are isotropic; 6G9X/5BOV carry anisotropic
# (ANISOU) atoms to exercise the direct-summation aniso fast paths. Spread spans
# ~1.3k -> 11.4k atoms.
DEFAULT_STRUCTURES = ["3GR5", "1DAW", "6G9X", "5BOV"]
# Order matters only for the summary; "heavier" methods last.
TORCH_METHODS = ["sf_fft", "sf_ds", "sfcalc"]


def _clear_stale_extension_locks():
    """Remove stale torch C++ extension lock files left by killed processes."""
    cache_dir = Path.home() / ".cache" / "torch_extensions"
    if not cache_dir.exists():
        return
    for lock_file in cache_dir.rglob("lock"):
        try:
            lock_file.unlink()
            print(f"Removed stale lock: {lock_file}")
        except OSError:
            pass


def run_worker(method, structure, device, n_iterations, n_warmup, output_file,
               timeout, mode):
    """Launch one worker subprocess for one metric-mode. Returns parsed dict."""
    cmd = [
        PYTHON, str(WORKER),
        "--method", method,
        "--structure", structure,
        "--device", device,
        "--mode", mode,
        "--n-iterations", str(n_iterations),
        "--n-warmup", str(n_warmup),
        "--output", str(output_file),
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=None, text=True, timeout=timeout
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.returncode != 0:
            # Worker crashed without writing JSON (e.g. killed/segfault).
            print(f"  CRASH ({method}/{structure}/{mode}): rc={result.returncode}")
            return {"status": "crash", "mode": mode}
        with open(output_file) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT ({method}/{structure}/{mode}): exceeded {timeout}s")
        return {"status": "timeout", "mode": mode}
    except Exception as e:
        print(f"  EXCEPTION ({method}/{structure}/{mode}): {e}")
        return {"status": "error", "mode": mode}


def run_cell(method, structure, device, n_iterations, n_warmup, out_dir, timeout):
    """Run all metric-modes for one (method, structure) and merge into one dict.

    Each metric runs in its own worker process (cold allocator) so timing and
    the two memory figures are measured independently.
    """
    # Split metrics so one OOM (e.g. backward) doesn't lose the others.
    modes = ["time_fwd", "time_fwd_bwd", "mem_fwd", "mem_fwd_bwd"]
    if method == "cctbx":
        modes = ["time_fwd"]  # classical reference: forward time only

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    merged = {"method": method, "structure": structure, "device": device}
    statuses = {}
    for mode in modes:
        out_file = raw_dir / f"{structure}_{method}_{mode}.json"
        r = run_worker(method, structure, device, n_iterations, n_warmup,
                       out_file, timeout, mode)
        statuses[mode] = (r or {}).get("status", "missing")
        if not r:
            continue
        # Copy metadata (first non-empty wins) and the metric this mode produced.
        for k in ("n_atoms", "n_reflections", "d_min", "dtype", "gpu_name",
                  "torch_threads", "n_iterations", "n_warmup"):
            if k in r and k not in merged:
                merged[k] = r[k]
        for k in ("fwd", "fwd_bwd", "mem_fwd_bytes", "mem_fwd_bwd_bytes"):
            if k in r:
                merged[k] = r[k]
    merged["status"] = statuses
    return merged


def _cell(stats):
    if not stats:
        return ("", "", "")
    return (
        f"{stats['mean_time']:.6f}",
        f"{stats['min_time']:.6f}",
        f"{stats['max_time']:.6f}",
    )


def _fmt_status(status):
    """Compact per-mode status, e.g. 'ok' or 'time=ok;mem_fwd=oom'."""
    if isinstance(status, dict):
        if all(v == "ok" for v in status.values()):
            return "ok"
        return ";".join(f"{k}={v}" for k, v in status.items() if v != "ok")
    return str(status)


def write_summary_csv(results, output_path):
    fieldnames = [
        "device", "structure", "n_atoms", "n_reflections", "d_min", "method",
        "dtype", "torch_threads", "status",
        "fwd_mean", "fwd_min", "fwd_max",
        "fwd_bwd_mean", "fwd_bwd_min", "fwd_bwd_max",
        "mem_fwd_mb", "mem_fwd_bwd_mb",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            fwd_m, fwd_lo, fwd_hi = _cell(r.get("fwd"))
            fb_m, fb_lo, fb_hi = _cell(r.get("fwd_bwd"))
            mf = r.get("mem_fwd_bytes")
            mfb = r.get("mem_fwd_bwd_bytes")
            d_min = r.get("d_min")
            writer.writerow({
                "device": r["device"],
                "structure": r["structure"],
                "n_atoms": r.get("n_atoms", ""),
                "n_reflections": r.get("n_reflections", ""),
                "d_min": f"{d_min:.4f}" if isinstance(d_min, (int, float)) else "",
                "method": r["method"],
                "dtype": r.get("dtype", ""),
                "torch_threads": r.get("torch_threads", ""),
                "status": _fmt_status(r.get("status", "")),
                "fwd_mean": fwd_m, "fwd_min": fwd_lo, "fwd_max": fwd_hi,
                "fwd_bwd_mean": fb_m, "fwd_bwd_min": fb_lo, "fwd_bwd_max": fb_hi,
                "mem_fwd_mb": f"{mf / 1e6:.1f}" if mf is not None else "",
                "mem_fwd_bwd_mb": f"{mfb / 1e6:.1f}" if mfb is not None else "",
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--structures", nargs="+", default=DEFAULT_STRUCTURES)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument(
        "--timeout", type=int, default=1800,
        help="Per-worker timeout in seconds (default 1800). SF_DS on large "
             "structures can be slow; timed-out runs are skipped, not fatal.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Default: results_<timestamp>/<device>/ under this folder.",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Override methods to run (default: sf_fft sf_ds sfcalc, + cctbx on cpu).",
    )
    args = parser.parse_args()

    _clear_stale_extension_locks()

    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = SCRIPT_DIR / f"results_{ts}"
    out_dir = out_root / args.device
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.methods:
        methods = list(args.methods)
    else:
        methods = list(TORCH_METHODS)
        if args.device == "cpu":
            methods.append("cctbx")

    print("=" * 78)
    print("SFcalculator vs TorchRef SF_DS / SF_FFT — benchmark")
    print("=" * 78)
    print(f"Device:      {args.device}")
    print(f"Structures:  {args.structures}")
    print(f"Methods:     {methods}")
    print(f"Iterations:  {args.n_iterations} (+ {args.n_warmup} warmup)")
    print(f"Output dir:  {out_dir}")
    print(f"Python:      {PYTHON}")
    print("=" * 78, flush=True)

    results = []
    total = len(args.structures) * len(methods)
    i = 0
    for structure in args.structures:
        for method in methods:
            i += 1
            print(f"[{i}/{total}] {method} / {structure} ...", flush=True)
            r = run_cell(
                method, structure, args.device,
                args.n_iterations, args.n_warmup, out_dir, args.timeout,
            )
            results.append(r)
            # Persist the merged per-cell result too.
            with open(out_dir / f"{structure}_{method}.json", "w") as f:
                json.dump(r, f, indent=2)
            print(flush=True)

    if not results:
        print("No successful runs. Exiting.")
        sys.exit(1)

    csv_path = out_dir / "summary.csv"
    write_summary_csv(results, csv_path)
    print(f"Summary written to: {csv_path}")

    if not args.no_plot:
        print("\nGenerating plots...")
        try:
            subprocess.run(
                [PYTHON, str(PLOT_SCRIPT), "--results-dir", str(out_root)],
                timeout=120,
            )
        except Exception as e:
            print(f"Plot generation failed (run plot_results.py manually): {e}")


if __name__ == "__main__":
    main()
