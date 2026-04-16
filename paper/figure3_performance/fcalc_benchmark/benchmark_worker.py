#!/usr/bin/env python
"""
Worker script for Fcalc benchmark.

Runs a single benchmark with a specific thread count and device.
TORCHREF_NUM_THREADS must be set in the environment BEFORE this script runs.

Usage (called by benchmark_thread_scaling.py, not directly):
    TORCHREF_NUM_THREADS=4 python benchmark_worker.py --n_iterations 10 --output results.json
    python benchmark_worker.py --device cuda --n_iterations 10 --output gpu.json
"""

import argparse
import json
import os
import sys
import time
import warnings

# Verify TORCHREF_NUM_THREADS is set before any imports
n_threads = int(os.environ.get("TORCHREF_NUM_THREADS", 1))

import torch
from torchref import ModelFT, ReflectionData

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MTZ_FILE = os.path.join(DATA_DIR, "1DAW.mtz")
PDB_FILE = os.path.join(DATA_DIR, "1DAW.pdb")


def _time_iterations(func, n_iterations: int) -> list[float]:
    """Time repeated calls to func on CPU."""
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def _time_iterations_gpu(func, n_iterations: int) -> list[float]:
    """Time repeated calls to func on GPU with proper synchronization."""
    times = []
    for _ in range(n_iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        func()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def _summarize_times(times: list[float]) -> dict:
    """Compute summary statistics from a list of iteration times."""
    total = sum(times)
    return {
        "iteration_times": times,
        "total_time": total,
        "mean_time": total / len(times),
        "min_time": min(times),
        "max_time": max(times),
    }


def run_benchmark(n_iterations: int, n_warmup: int, device_str: str = "cpu") -> dict:
    """Run structure factor calculation benchmark and return timing results."""
    device = torch.device(device_str)
    is_gpu = device.type == "cuda"
    timer = _time_iterations_gpu if is_gpu else _time_iterations

    # Load data
    data = ReflectionData(device=device).load_mtz(MTZ_FILE)
    d_min = data.d_min
    M = ModelFT(max_res=d_min, device=device, radius_angstrom=3.0).load_pdb(PDB_FILE)
    hkl, _, _, _ = data()

    n_atoms = M.xyz().shape[0]
    n_reflections = hkl.shape[0]

    iso_ref = M.get_iso()        # (xyz, adp, occ, A, B)
    aniso_ref = M.get_aniso()    # (xyz, u, occ, A, B)  — tensors may be None
    iso = tuple(t.clone().detach().requires_grad_(True) for t in iso_ref)
    aniso = tuple(
            t.clone().detach().requires_grad_(True) if t is not None else None
            for t in aniso_ref
        )
    
    def _forward():
        sf, _ed = M.fft.compute_structure_factors(hkl, *iso, *aniso)

    # --- TorchRef forward-only benchmark ---
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _forward()
        if is_gpu:
            torch.cuda.synchronize()

    # Timed runs
    with torch.no_grad():
        torchref_times = timer(lambda: _forward(), n_iterations)

    # --- TorchRef forward-with-graph benchmark (no backward) ---
    # Warmup
    for _ in range(n_warmup):
        _forward()
    if is_gpu:
        torch.cuda.synchronize()

    # Timed runs
    torchref_fwd_graph_times = timer(_forward, n_iterations)

    # --- TorchRef backward-only benchmark ---
    # Run forward (untimed) to build graph, then time only backward.
    def _backward_only():
        sf, _ed = M.fft.compute_structure_factors(hkl, *iso, *aniso)
        loss = sf.abs().sum()
        if is_gpu:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        if is_gpu:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return t1 - t0

    # Warmup
    _backward_only()

    # Timed runs
    torchref_bwd_only_times = [_backward_only() for _ in range(n_iterations)]

    # --- TorchRef forward+backward benchmark ---
    def _forward_backward():
        sf, _ed = M.fft.compute_structure_factors(hkl, *iso, *aniso)
        loss = sf.abs().sum()
        loss.backward()

    # Warmup
    _forward_backward()
    if is_gpu:
        torch.cuda.synchronize()

    # Timed runs
    torchref_fwd_bwd_times = timer(_forward_backward, n_iterations)

    result = {
        "device": device_str,
        "n_threads": n_threads,
        "n_iterations": n_iterations,
        "n_warmup": n_warmup,
        "n_atoms": n_atoms,
        "n_reflections": n_reflections,
        "d_min": float(d_min),
        "torch_threads": torch.get_num_threads(),
        "torchref": _summarize_times(torchref_times),
        "torchref_fwd_graph": _summarize_times(torchref_fwd_graph_times),
        "torchref_bwd_only": _summarize_times(torchref_bwd_only_times),
        "torchref_fwd_bwd": _summarize_times(torchref_fwd_bwd_times),
    }

    if is_gpu:
        result["gpu_name"] = torch.cuda.get_device_name(0)

    # --- cctbx benchmark (CPU only) ---
    if not is_gpu:
        from iotbx import pdb as iotbx_pdb

        pdb_input = iotbx_pdb.input(file_name=PDB_FILE)
        xray_structure = pdb_input.xray_structure_simple()

        # Warmup
        for _ in range(n_warmup):
            xray_structure.structure_factors(d_min=d_min).f_calc()

        # Timed runs
        cctbx_times = _time_iterations(
            lambda: xray_structure.structure_factors(d_min=d_min).f_calc(), n_iterations
        )
        result["cctbx"] = _summarize_times(cctbx_times)

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark worker")
    parser.add_argument(
        "--n_iterations", type=int, default=10, help="Number of timed iterations"
    )
    parser.add_argument(
        "--n_warmup", type=int, default=3, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device to benchmark on (default: cpu)",
    )
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    args = parser.parse_args()

    # Suppress torchref threading warnings in worker
    warnings.filterwarnings("ignore", message="TorchRef.*threads")

    # Redirect stdout during model loading to suppress verbose library output
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        results = run_benchmark(args.n_iterations, args.n_warmup, args.device)
    finally:
        sys.stdout = old_stdout

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary to stdout for live monitoring
    tr = results["torchref"]
    fg = results["torchref_fwd_graph"]
    bo = results["torchref_bwd_only"]
    fb = results["torchref_fwd_bwd"]
    device_label = results.get("gpu_name", f"CPU x{results['n_threads']}")
    msg = (
        f"  {device_label}  "
        f"fwd: {tr['mean_time']:.4f}s  "
        f"fwd(graph): {fg['mean_time']:.4f}s  "
        f"bwd: {bo['mean_time']:.4f}s  "
        f"fwd+bwd: {fb['mean_time']:.4f}s"
    )
    if "cctbx" in results:
        cc = results["cctbx"]
        msg += f"  |  cctbx: mean={cc['mean_time']:.4f}s"
    print(msg)


if __name__ == "__main__":
    main()
