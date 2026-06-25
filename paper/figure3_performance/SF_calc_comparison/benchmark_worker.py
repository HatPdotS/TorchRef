#!/usr/bin/env python
"""
Worker for the SFcalculator vs TorchRef (SF_DS / SF_FFT) structure-factor benchmark.

Benchmarks ONE (method, structure) on ONE device, in its own process so peak
memory is cleanly attributable. Measures:
  - forward time           (structure-factor calc only, under no_grad)
  - forward+backward time  (gradient w.r.t. xyz + B-factors)
  - peak memory            (forward and forward+backward, measured separately)

All three differentiable methods compute the BARE PROTEIN structure factor (no
bulk solvent, no scaling) over the reflections to d_min, with full crystal
symmetry, in float32 / complex64. cctbx is a classical double-precision CPU
reference (forward only).

TORCHREF_NUM_THREADS must be set in the environment BEFORE this script runs.

Usage (called by run_benchmarks.py, not usually directly):
    TORCHREF_NUM_THREADS=16 python benchmark_worker.py \
        --method sf_fft --structure 1DAW --device cpu \
        --n-iterations 10 --n-warmup 3 --output out.json
"""

import argparse
import io
import json
import os
import sys
import threading
import time
import warnings

# Read thread count before importing torch (mirrors fcalc_benchmark worker).
N_THREADS = int(os.environ.get("TORCHREF_NUM_THREADS", 1))

# Repo root = .../comparison_SFcalc  (this file is paper/figure3_performance/SF_calc_comparison/)
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
PDB_DIR = os.path.join(REPO_ROOT, "tests", "files", "pdb")
MTZ_DIR = os.path.join(REPO_ROOT, "tests", "files", "mtz")

METHODS = ("sf_fft", "sf_ds", "sfcalc", "cctbx")


# --------------------------------------------------------------------------- #
# Timing helpers (reused verbatim from fcalc_benchmark/benchmark_worker.py)
# --------------------------------------------------------------------------- #
def _time_iterations(func, n_iterations):
    """Time repeated calls to func on CPU."""
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def _time_iterations_gpu(func, n_iterations):
    """Time repeated calls to func on GPU with proper synchronization."""
    import torch

    times = []
    for _ in range(n_iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        func()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def _summarize_times(times):
    """Compute summary statistics from a list of iteration times."""
    total = sum(times)
    return {
        "iteration_times": times,
        "total_time": total,
        "mean_time": total / len(times),
        "min_time": min(times),
        "max_time": max(times),
    }


# --------------------------------------------------------------------------- #
# Memory helpers
# --------------------------------------------------------------------------- #
class _RSSPeakSampler:
    """Background thread sampling process RSS to estimate CPU peak memory.

    Reports peak RSS observed minus a baseline captured at start(), so the
    figure reflects the memory the timed call itself adds (Python + torch +
    loaded model are already resident at baseline).
    """

    def __init__(self, interval=0.005):
        import psutil

        self._proc = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.baseline = 0
        self.peak = 0

    def __enter__(self):
        self.baseline = self._proc.memory_info().rss
        self.peak = self.baseline
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss
            time.sleep(self._interval)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        # One final reading in case the peak occurred between samples.
        rss = self._proc.memory_info().rss
        if rss > self.peak:
            self.peak = rss

    @property
    def delta_bytes(self):
        return max(0, self.peak - self.baseline)


def _trim_cpu_allocator():
    """Best effort: collect garbage and return freed heap to the OS.

    Without this, freed CPU tensors stay cached by the allocator and inflate
    the RSS baseline, contaminating the peak-minus-baseline measurement.
    """
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _measure_peak_memory(func, device, is_gpu, n_warmup):
    """Return peak memory (bytes) for func, measured AFTER warmup.

    Runs in a dedicated process (one metric per process) so the measurement is
    independent. Kernels are warmed first (Triton/compiled kernels and lazy
    buffers must be built before measuring — a cold first call would misreport).

    - GPU: warm up, then reset the allocator high-water mark and measure one
      steady-state call via ``max_memory_allocated`` (excludes the CUDA context
      and cached kernels, includes the call's tensors).
    - CPU: capture the RSS baseline after model build, then take the peak RSS
      reached across warmup + the measured call (robust to torch's CPU caching
      allocator, which does not return freed blocks to the OS).
    """
    import torch

    if is_gpu:
        for _ in range(n_warmup):
            func()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        func()
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated(device))
    else:
        _trim_cpu_allocator()
        with _RSSPeakSampler() as sampler:  # baseline = RSS before warmup
            for _ in range(n_warmup):
                func()
            func()
        return int(sampler.delta_bytes)


# --------------------------------------------------------------------------- #
# Common data loading
# --------------------------------------------------------------------------- #
def _resolve_files(structure):
    pdb = os.path.join(PDB_DIR, f"{structure}.pdb")
    mtz = os.path.join(MTZ_DIR, f"{structure}.mtz")
    for p in (pdb, mtz):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    return pdb, mtz


# --------------------------------------------------------------------------- #
# Per-method benchmark builders
#
# Each returns: (fwd, fwd_bwd, n_atoms, n_reflections, d_min, dtype_str)
#   fwd      -> callable computing the SF (no grad needed; caller wraps no_grad)
#   fwd_bwd  -> callable computing SF, loss = |F|.sum(), loss.backward()
# --------------------------------------------------------------------------- #
def _build_torchref(method, pdb, mtz, device):
    """Build fwd / fwd_bwd closures for sf_fft or sf_ds."""
    import torch
    from torchref import ModelFT, ReflectionData

    data = ReflectionData(device=device).load_mtz(mtz)
    d_min = float(data.d_min)
    M = ModelFT(max_res=d_min, device=device, radius_angstrom=3.0).load_pdb(pdb)
    hkl, _, _, _ = data()

    n_atoms = int(M.xyz().shape[0])
    n_reflections = int(hkl.shape[0])

    xyz_i, adp_i, occ_i, A_i, B_i = M.get_iso()
    aniso_ref = M.get_aniso()  # (xyz, u, occ, A, B); tensors may be None

    # Only xyz + ADP carry gradients (matches SFcalc: positions + B-factors).
    xyz_i = xyz_i.clone().detach().requires_grad_(True)
    adp_i = adp_i.clone().detach().requires_grad_(True)
    occ_i = occ_i.clone().detach()
    A_i = A_i.clone().detach()
    B_i = B_i.clone().detach()
    iso = (xyz_i, adp_i, occ_i, A_i, B_i)

    def _clone_aniso(t, grad=False):
        if t is None:
            return None
        t = t.clone().detach()
        return t.requires_grad_(True) if grad else t

    aniso = tuple(
        _clone_aniso(t, grad=(idx in (0, 1)))  # xyz_aniso, u_aniso
        for idx, t in enumerate(aniso_ref)
    )

    if method == "sf_fft":
        engine = M.fft
    else:  # sf_ds
        from torchref.model import SfDS

        engine = SfDS(M.cell, M.spacegroup, device=device)

    dtype_str = str(iso[0].dtype)

    def fwd():
        sf, _ = engine.compute_structure_factors(hkl, *iso, *aniso)
        return sf

    def fwd_bwd():
        for t in (xyz_i, adp_i):
            if t.grad is not None:
                t.grad = None
        sf, _ = engine.compute_structure_factors(hkl, *iso, *aniso)
        loss = sf.abs().sum()
        loss.backward()
        return loss

    return fwd, fwd_bwd, n_atoms, n_reflections, d_min, dtype_str


def _build_sfcalc(pdb, mtz, device):
    """Build fwd / fwd_bwd closures for SFcalculator (SFC_Torch)."""
    import torch
    from SFC_Torch import SFcalculator

    sfc = SFcalculator(
        pdb, mtz, device=torch.device(device), set_experiment=False
    )
    # Trigger any lazy form-factor / symmetry setup with one bare call.
    sfc.calc_fprotein()

    # Gradients on positions + isotropic B-factors (matches the torch methods).
    sfc.atom_pos_orth.requires_grad_(True)
    sfc.atom_b_iso.requires_grad_(True)

    n_atoms = int(sfc.atom_pos_orth.shape[0])
    # SFcalc computes over the ASU reflection list (Hasu_array), then maps to
    # the MTZ HKL set; report the count it actually computes.
    n_reflections = int(sfc.Hasu_array.shape[0])
    d_min = float(sfc.dHKL.min()) if hasattr(sfc, "dHKL") else float("nan")
    dtype_str = str(sfc.atom_pos_orth.dtype)

    def fwd():
        return sfc.calc_fprotein(Return=True)

    def fwd_bwd():
        if sfc.atom_pos_orth.grad is not None:
            sfc.atom_pos_orth.grad = None
        if sfc.atom_b_iso.grad is not None:
            sfc.atom_b_iso.grad = None
        Fp = sfc.calc_fprotein(Return=True)
        loss = Fp.abs().sum()
        loss.backward()
        return loss

    return fwd, fwd_bwd, n_atoms, n_reflections, d_min, dtype_str


def _build_cctbx(pdb, mtz):
    """Build fwd closure for cctbx (classical reference, CPU, forward only)."""
    from iotbx import pdb as iotbx_pdb
    from torchref import ReflectionData

    # Use the same d_min as the torch methods for a comparable workload.
    data = ReflectionData().load_mtz(mtz)
    d_min = float(data.d_min)
    n_reflections = int(data()[0].shape[0])

    pdb_input = iotbx_pdb.input(file_name=pdb)
    xray_structure = pdb_input.xray_structure_simple()
    n_atoms = int(xray_structure.scatterers().size())

    def fwd():
        return xray_structure.structure_factors(d_min=d_min).f_calc()

    return fwd, n_atoms, n_reflections, d_min, "float64(cctbx)"


# --------------------------------------------------------------------------- #
# Main benchmark
#
# One METRIC per process so each is measured independently from a cold
# allocator (critical for honest CPU memory numbers):
#   mode "time"        -> forward + forward/backward timing
#   mode "mem_fwd"     -> peak memory of a single forward call
#   mode "mem_fwd_bwd" -> peak memory of a single forward+backward call
# --------------------------------------------------------------------------- #
# One metric per process. time_fwd / time_fwd_bwd are split so a backward OOM
# does not also lose the forward timing. ("time" kept as a combined alias.)
MODES = ("time", "time_fwd", "time_fwd_bwd", "mem_fwd", "mem_fwd_bwd")


def _fwd_no_grad(fwd):
    import torch

    with torch.no_grad():
        fwd()


def _build(method, structure, device):
    """Return (fwd, fwd_bwd_or_None, n_atoms, n_refl, d_min, dtype)."""
    pdb, mtz = _resolve_files(structure)
    if method == "cctbx":
        fwd, n_atoms, n_refl, d_min, dtype_str = _build_cctbx(pdb, mtz)
        return fwd, None, n_atoms, n_refl, d_min, dtype_str
    if method in ("sf_fft", "sf_ds"):
        return _build_torchref(method, pdb, mtz, device)
    if method == "sfcalc":
        return _build_sfcalc(pdb, mtz, device)
    raise ValueError(f"Unknown method: {method}")


def run_benchmark(method, structure, device_str, n_iterations, n_warmup, mode):
    import torch

    device = torch.device(device_str)
    is_gpu = device.type == "cuda"
    timer = _time_iterations_gpu if is_gpu else _time_iterations

    result = {
        "method": method,
        "structure": structure,
        "device": device_str,
        "mode": mode,
        "n_threads": N_THREADS,
        "n_iterations": n_iterations,
        "n_warmup": n_warmup,
        "torch_threads": torch.get_num_threads(),
        "status": "ok",
    }
    if is_gpu:
        result["gpu_name"] = torch.cuda.get_device_name(0)
    if method == "cctbx" and is_gpu:
        raise ValueError("cctbx benchmark is CPU only")

    fwd, fwd_bwd, n_atoms, n_refl, d_min, dtype_str = _build(
        method, structure, device
    )
    result.update(
        {"n_atoms": n_atoms, "n_reflections": n_refl, "d_min": d_min,
         "dtype": dtype_str}
    )

    if mode in ("time", "time_fwd"):
        with torch.no_grad():
            for _ in range(n_warmup):
                fwd()
            if is_gpu:
                torch.cuda.synchronize()
            result["fwd"] = _summarize_times(timer(fwd, n_iterations))

    if mode in ("time", "time_fwd_bwd"):
        if fwd_bwd is not None:
            fwd_bwd()  # warmup (also builds caches)
            for _ in range(max(0, n_warmup - 1)):
                fwd_bwd()
            if is_gpu:
                torch.cuda.synchronize()
            result["fwd_bwd"] = _summarize_times(timer(fwd_bwd, n_iterations))
        else:
            result["fwd_bwd"] = None

    if mode == "mem_fwd":
        result["mem_fwd_bytes"] = _measure_peak_memory(
            lambda: _fwd_no_grad(fwd), device, is_gpu, n_warmup
        )

    elif mode == "mem_fwd_bwd":
        if fwd_bwd is None:
            result["mem_fwd_bwd_bytes"] = None  # classical: no autodiff
        else:
            result["mem_fwd_bwd_bytes"] = _measure_peak_memory(
                fwd_bwd, device, is_gpu, n_warmup
            )

    return result


def _classify_error(exc):
    """Return 'oom' for out-of-memory errors, else 'error'."""
    import torch

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return "oom"
    msg = str(exc).lower()
    if "out of memory" in msg or "can't allocate" in msg or "cannot allocate" in msg:
        return "oom"
    return "error"


def main():
    parser = argparse.ArgumentParser(description="SF comparison benchmark worker")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--mode", default="time", choices=MODES)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message="TorchRef.*threads")
    warnings.filterwarnings("ignore", message=".*deprecated.*")

    # Suppress verbose library loading chatter on stdout (keep stderr live).
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    status = "ok"
    try:
        result = run_benchmark(
            args.method, args.structure, args.device,
            args.n_iterations, args.n_warmup, args.mode,
        )
    except Exception as exc:  # capture (incl. OOM) rather than silently dropping
        sys.stdout = old_stdout
        status = _classify_error(exc)
        result = {
            "method": args.method, "structure": args.structure,
            "device": args.device, "mode": args.mode, "status": status,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        sys.stdout = old_stdout

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    if status != "ok":
        print(f"  {args.method:7s} {args.structure} [{args.device}] "
              f"{args.mode}: {status.upper()}")
        return

    parts = []
    if "fwd" in result:
        parts.append(f"fwd={result['fwd']['mean_time']:.4f}s")
        if result.get("fwd_bwd"):
            parts.append(f"fwd+bwd={result['fwd_bwd']['mean_time']:.4f}s")
    if "mem_fwd_bytes" in result:
        parts.append(f"mem(fwd)={result['mem_fwd_bytes'] / 1e6:.0f}MB")
    if "mem_fwd_bwd_bytes" in result:
        v = result["mem_fwd_bwd_bytes"]
        parts.append(f"mem(fwd+bwd)={v / 1e6:.0f}MB" if v is not None else "mem(fwd+bwd)=n/a")
    print(f"  {args.method:7s} {args.structure} [{args.device}] "
          f"{args.mode}: {'  '.join(parts)}")


if __name__ == "__main__":
    main()
