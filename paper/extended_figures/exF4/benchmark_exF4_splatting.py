#!/usr/bin/env python3
"""Benchmark structure factor splatting approaches for Extended Figure 4.

Compares 5 configurations of the Fcalc pipeline on the 1DAW test structure:
  1. CPU original (1 thread)     — trivial splatting
  2. CPU separable (1 thread)    — decomposed splatting
  3. CPU separable (4 threads)   — decomposed + threading
  4. GPU fused_triton            — trivial GPU kernel
  5. GPU separable_triton        — decomposed GPU kernel

For each, times the total Fcalc AND breaks it into 3 stages:
  A. Electron density splatting
  B. FFT
  C. Symmetry extraction

Output: data/exF4_splatting.json

Usage:
    # CPU-only (no GPU required):
    python benchmark_exF4_splatting.py --cpu-only

    # All (needs GPU):
    python benchmark_exF4_splatting.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Set single thread for consistent CPU baselines
os.environ.setdefault("TORCHREF_NUM_THREADS", "1")

import torch
import torchref.base.electron_density.main as ed_module
from torchref import ModelFT, ReflectionData
from torchref.base.fourier import ifft
from torchref.base.reciprocal import ReciprocalSymmetryExtractor

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "figure3_performance" / "data"
PDB_FILE = DATA_DIR / "1DAW.pdb"
MTZ_FILE = DATA_DIR / "1DAW.mtz"
OUT_JSON = Path(__file__).resolve().parent / "exF4_splatting.json"

N_WARMUP = 5
N_ITERATIONS = 25


def _time_cpu(func, n):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def _time_gpu(func, n):
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        func()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def _stats(times):
    import numpy as np
    a = np.array(times)
    return {
        "mean_ms": float(a.mean() * 1e3),
        "std_ms": float(a.std() * 1e3),
        "min_ms": float(a.min() * 1e3),
        "max_ms": float(a.max() * 1e3),
        "times_ms": [float(t * 1e3) for t in times],
    }


def benchmark_config(label, engine_key, engine_val, device_str, n_threads=1,
                     n_warmup=N_WARMUP, n_iterations=N_ITERATIONS):
    """Benchmark one configuration."""
    print(f"\n{'='*60}")
    print(f"Config: {label}")
    print(f"  Engine: {engine_key}={engine_val}, device={device_str}, threads={n_threads}")
    print(f"{'='*60}")

    # Set engine and thread count
    os.environ["TORCHREF_NUM_THREADS"] = str(n_threads)
    torch.set_num_threads(n_threads)
    if "GPU" in engine_key:
        ed_module.ISO_MAP_ENGINE_GPU = engine_val
    else:
        ed_module.ISO_MAP_ENGINE_CPU = engine_val

    device = torch.device(device_str)
    is_gpu = device.type == "cuda"
    timer = _time_gpu if is_gpu else _time_cpu

    # Load model and data
    data = ReflectionData(device=device).load_mtz(str(MTZ_FILE))
    M = ModelFT(max_res=data.d_min, device=device, radius_angstrom=3.0).load_pdb(
        str(PDB_FILE)
    )
    hkl, _, _, _ = data()

    iso = M.get_iso()
    aniso = M.get_aniso()

    # Pre-build the symmetry extractor so it doesn't pollute timing
    grid_shape = tuple(int(x) for x in M.fft.gridsize)
    sym_extractor = ReciprocalSymmetryExtractor(
        hkl, M.fft.spacegroup, grid_shape, device=device
    )

    # === Stage functions ===
    def stage_a():
        return M.fft.build_density_map(
            *iso, *aniso, apply_symmetry=False
        )

    def stage_b(density_map):
        return ifft(density_map, M.fft.cell.volume)

    def stage_c(recip_grid):
        return sym_extractor.extract_from_grid(recip_grid)

    def full_pipeline():
        return M.fft.compute_structure_factors(hkl, *iso, *aniso)

    # === Warmup ===
    print("  Warming up...")
    with torch.no_grad():
        for _ in range(n_warmup):
            full_pipeline()
        if is_gpu:
            torch.cuda.synchronize()

    # === Time total pipeline ===
    print("  Timing total pipeline...")
    with torch.no_grad():
        total_times = timer(lambda: full_pipeline(), n_iterations)

    # === Time individual stages ===
    print("  Timing Stage A (splatting)...")
    with torch.no_grad():
        stage_a_times = timer(stage_a, n_iterations)

    # Get a density map for stage B/C timing
    with torch.no_grad():
        density_map = stage_a()

    print("  Timing Stage B (FFT)...")
    with torch.no_grad():
        stage_b_times = timer(lambda: stage_b(density_map), n_iterations)

    # Get a reciprocal grid for stage C timing
    with torch.no_grad():
        recip_grid = stage_b(density_map)

    print("  Timing Stage C (extraction)...")
    with torch.no_grad():
        stage_c_times = timer(lambda: stage_c(recip_grid), n_iterations)

    result = {
        "label": label,
        "engine": engine_val,
        "device": device_str,
        "n_threads": n_threads,
        "n_atoms": int(M.xyz().shape[0]),
        "n_reflections": int(hkl.shape[0]),
        "d_min": float(data.d_min),
        "total": _stats(total_times),
        "stage_a_splatting": _stats(stage_a_times),
        "stage_b_fft": _stats(stage_b_times),
        "stage_c_extraction": _stats(stage_c_times),
    }

    print(f"  Total:      {result['total']['mean_ms']:8.2f} ± {result['total']['std_ms']:.2f} ms")
    print(f"  Splatting:  {result['stage_a_splatting']['mean_ms']:8.2f} ± {result['stage_a_splatting']['std_ms']:.2f} ms")
    print(f"  FFT:        {result['stage_b_fft']['mean_ms']:8.2f} ± {result['stage_b_fft']['std_ms']:.2f} ms")
    print(f"  Extraction: {result['stage_c_extraction']['mean_ms']:8.2f} ± {result['stage_c_extraction']['std_ms']:.2f} ms")

    return result


def main():
    parser = argparse.ArgumentParser(description="Splatting benchmark for ExF 5")
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU benchmarks")
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP)
    parser.add_argument("--n-iterations", type=int, default=N_ITERATIONS)
    args = parser.parse_args()

    n_warmup = args.n_warmup
    n_iterations = args.n_iterations

    configs = [
        ("CPU trivial (1 thread)", "ISO_MAP_ENGINE_CPU", "original", "cpu", 1),
        ("CPU decomposed (1 thread)", "ISO_MAP_ENGINE_CPU", "separable", "cpu", 1),
        ("CPU decomposed (4 threads)", "ISO_MAP_ENGINE_CPU", "separable", "cpu", 4),
    ]

    has_gpu = torch.cuda.is_available() and not args.cpu_only
    if has_gpu:
        configs.extend([
            ("GPU trivial (fused Triton)", "ISO_MAP_ENGINE_GPU", "fused_triton", "cuda", 1),
            ("GPU decomposed (separable Triton)", "ISO_MAP_ENGINE_GPU", "separable_triton", "cuda", 1),
        ])
    elif not args.cpu_only:
        print("Warning: CUDA not available, skipping GPU benchmarks")

    results = []
    for label, engine_key, engine_val, device_str, n_threads in configs:
        r = benchmark_config(label, engine_key, engine_val, device_str, n_threads,
                             n_warmup=n_warmup, n_iterations=n_iterations)
        results.append(r)

    # Add speedup relative to CPU trivial baseline
    baseline = results[0]["total"]["mean_ms"]
    for r in results:
        r["speedup_vs_baseline"] = baseline / r["total"]["mean_ms"]

    output = {
        "benchmark": "exF4_splatting",
        "structure": "1DAW",
        "n_warmup": n_warmup,
        "n_iterations": n_iterations,
        "results": results,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
