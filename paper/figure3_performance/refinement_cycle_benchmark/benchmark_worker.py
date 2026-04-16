#!/usr/bin/env python
"""
Worker script for refinement cycle benchmark.

Runs a single benchmark with a specific thread count and device, timing
the full loss-state evaluation (x-ray + geometry + ADP targets) for
forward, backward, and combined passes.

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
from torchref.refinement import LBFGSRefinement

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
    """Run refinement cycle benchmark and return timing results."""
    device = torch.device(device_str)
    is_gpu = device.type == "cuda"
    timer = _time_iterations_gpu if is_gpu else _time_iterations

    # ---- SETUP (untimed) ----
    # Pass device at construction so all tensors (model, scaler, restraints)
    # are created on the correct device from the start.
    refinement = LBFGSRefinement(
        data_file=MTZ_FILE,
        pdb=PDB_FILE,
        device=device,
        target_mode="bhattacharyya",
    )

    # Create and configure loss state with default weights
    loss_state = refinement.complete_loss_state()

    # Collect metadata
    n_atoms = len(refinement.model.pdb)
    hkl, _, _, _ = refinement.reflection_data()
    n_reflections = hkl.shape[0]
    d_min = float(refinement.reflection_data.d_min)
    target_names = list(loss_state.targets.keys())

    # Helper: zero all parameter gradients
    def _zero_grads():
        for p in refinement.parameters():
            if p.grad is not None:
                p.grad.zero_()

    # ---- MODE 1: Forward-only (no_grad) ----
    def _forward_no_grad():
        refinement.model.reset_cache()
        with torch.no_grad():
            loss_state.aggregate()

    for _ in range(n_warmup):
        _forward_no_grad()
    if is_gpu:
        torch.cuda.synchronize()

    fwd_no_grad_times = timer(_forward_no_grad, n_iterations)

    # ---- MODE 2: Forward with graph ----
    def _forward_graph():
        refinement.model.reset_cache()
        _zero_grads()
        loss_state.aggregate()

    for _ in range(n_warmup):
        _forward_graph()
        _zero_grads()
    if is_gpu:
        torch.cuda.synchronize()

    fwd_graph_times = timer(_forward_graph, n_iterations)

    # ---- MODE 3: Backward-only ----
    # Forward untimed (builds graph), then time only backward
    def _backward_only():
        refinement.model.reset_cache()
        _zero_grads()
        loss = loss_state.aggregate()
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

    bwd_only_times = [_backward_only() for _ in range(n_iterations)]

    # ---- MODE 4: Forward + Backward combined ----
    def _forward_backward():
        refinement.model.reset_cache()
        _zero_grads()
        loss = loss_state.aggregate()
        loss.backward()

    for _ in range(n_warmup):
        _forward_backward()
    if is_gpu:
        torch.cuda.synchronize()

    fwd_bwd_times = timer(_forward_backward, n_iterations)

    # ---- MODE 5: Per-target breakdown (forward, backward, fwd+bwd) ----
    per_target = {}
    for name, target in loss_state.targets.items():
        # Forward only (no_grad)
        def _fwd(t=target):
            refinement.model.reset_cache()
            with torch.no_grad():
                t()

        for _ in range(n_warmup):
            _fwd()
        if is_gpu:
            torch.cuda.synchronize()
        fwd_times = timer(_fwd, n_iterations)

        # Forward + backward combined
        def _fwd_bwd(t=target):
            refinement.model.reset_cache()
            _zero_grads()
            loss = t()
            loss.backward()

        for _ in range(n_warmup):
            _fwd_bwd()
        if is_gpu:
            torch.cuda.synchronize()
        fwd_bwd_times_t = timer(_fwd_bwd, n_iterations)

        # Backward only (forward untimed, backward timed)
        def _bwd(t=target):
            refinement.model.reset_cache()
            _zero_grads()
            loss = t()
            if is_gpu:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss.backward()
            if is_gpu:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            return t1 - t0

        _bwd()  # warmup
        bwd_times = [_bwd() for _ in range(n_iterations)]

        per_target[name] = {
            "forward": _summarize_times(fwd_times),
            "backward": _summarize_times(bwd_times),
            "fwd_bwd": _summarize_times(fwd_bwd_times_t),
        }

    # ---- Assemble results ----
    result = {
        "device": device_str,
        "n_threads": n_threads,
        "n_iterations": n_iterations,
        "n_warmup": n_warmup,
        "n_atoms": n_atoms,
        "n_reflections": n_reflections,
        "d_min": d_min,
        "torch_threads": torch.get_num_threads(),
        "xray_mode": refinement.target_mode,
        "target_names": target_names,
        "aggregate_fwd_no_grad": _summarize_times(fwd_no_grad_times),
        "aggregate_fwd_graph": _summarize_times(fwd_graph_times),
        "aggregate_bwd_only": _summarize_times(bwd_only_times),
        "aggregate_fwd_bwd": _summarize_times(fwd_bwd_times),
        "per_target": per_target,
    }

    if is_gpu:
        result["gpu_name"] = torch.cuda.get_device_name(0)

    return result


def main():
    parser = argparse.ArgumentParser(description="Refinement cycle benchmark worker")
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
    device_label = results.get("gpu_name", f"CPU x{results['n_threads']}")
    fwd = results["aggregate_fwd_no_grad"]
    bwd = results["aggregate_bwd_only"]
    fb = results["aggregate_fwd_bwd"]
    msg = (
        f"  {device_label}  "
        f"fwd: {fwd['mean_time']:.4f}s  "
        f"bwd: {bwd['mean_time']:.4f}s  "
        f"fwd+bwd: {fb['mean_time']:.4f}s"
    )
    # Append per-target top-3 breakdown (by fwd+bwd time)
    sorted_targets = sorted(
        results["per_target"].items(),
        key=lambda kv: kv[1]["fwd_bwd"]["mean_time"], reverse=True,
    )
    top3 = ", ".join(
        f"{name}={v['fwd_bwd']['mean_time']*1000:.1f}ms"
        for name, v in sorted_targets[:3]
    )
    msg += f"  | top: {top3}"
    print(msg)


if __name__ == "__main__":
    main()
