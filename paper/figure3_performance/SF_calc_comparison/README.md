# SFcalculator vs TorchRef SF_DS / SF_FFT — performance benchmark

Direct performance comparison (requested in paper review) between three
differentiable structure-factor calculators, plus a classical reference:

| Method | What it is |
|--------|------------|
| **TorchRef SF_FFT** | TorchRef FFT-based structure factors (`SfFFT`) |
| **TorchRef SF_DS**  | TorchRef direct-summation structure factors (`SfDS`) |
| **SFcalculator**    | `SFC_Torch.SFcalculator` (Hekstra lab), `calc_fprotein` |
| **cctbx** (ref)     | `iotbx` `xray_structure.structure_factors().f_calc()`, CPU only |

Metrics, per structure, on **GPU** and **CPU (16 threads)**:
- **forward** time (SF calculation only)
- **forward + backward** time (gradient w.r.t. atom xyz + isotropic B-factors)
- **peak memory** (forward and forward+backward, measured separately)

## What is compared (fairness)

- **Bare protein structure factor only** — no bulk-solvent mask, no scaling.
  SFcalc `calc_fprotein` is the matching quantity to TorchRef's model SF.
- **float32 / complex64** for all three differentiable methods (TorchRef default
  == SFcalc default). cctbx is double precision — treat it as a separate-precision
  classical reference, not a like-for-like point.
- **Same reflections to `d_min`** with full crystal symmetry and Cromer-Mann /
  ITC92 atomic form factors. TorchRef computes over the MTZ reflection list;
  SFcalc computes over its ASU list (`Hasu_array`) — each run records its own
  `n_reflections` in the JSON / `summary.csv` for transparency.
- **Gradients** flow through atom positions + isotropic B-factors for all three
  (`loss = |F|.sum()`).

### Measurement methodology
- **One metric per process.** Each `(method, structure)` is benchmarked by
  separate worker processes — one for timing, one for forward peak memory, one
  for forward+backward peak memory — so the three figures are independent and a
  warmed allocator from one never contaminates another.
- **Always warmed.** Every process runs `--n-warmup` calls before measuring, so
  compiled/Triton kernels and lazy buffers are built first (a cold first call
  would misreport both time and memory).
- **Memory.** GPU: warm up, then `reset_peak_memory_stats` + one steady-state
  call → `max_memory_allocated` (excludes CUDA context, includes the call's
  tensors). CPU: baseline RSS after model build, then peak RSS across
  warmup+measured via `psutil` (robust to torch's CPU caching allocator, which
  does not return freed blocks to the OS).
- **Failures are recorded, not dropped.** Each cell carries a `status`
  (`ok` / `oom` / `error` / `timeout`) per metric in the JSON and `summary.csv`,
  so e.g. SFcalc running out of GPU memory on large structures shows up
  explicitly rather than as a silent gap.

### Caveats
- **SF_DS auto-batches** to a memory budget (`SfDS(max_memory_gb=...)`, default
  2 GB). Numbers reflect that as-shipped behavior: its time is for batched
  execution. This is how a user actually runs it.
- **cctbx `f_calc` is single-threaded** regardless of `OMP_NUM_THREADS`; its CPU
  number is effectively single-core, and it is forward-only (no autodiff/memory).
- **SFcalc** has no internal batching and builds dense `(N_atoms, N_HKL)`
  tensors; it can exhaust GPU memory on large structures (recorded as `oom`).

## Structures (mixed size series, from `tests/files/`)

3GR5 (1.3k atoms, iso) · 1DAW (3k, iso) · 6G9X (5.9k, **aniso**) · 5BOV (11.4k,
**aniso**). Override with `--structures`.

The series mixes isotropic and anisotropic (ANISOU) structures so the
direct-summation **anisotropic fast paths** (and the Triton DS kernel) are
exercised, not just the isotropic path.

## Running (SLURM)

```bash
cd paper/figure3_performance/SF_calc_comparison
bash submit_all.sh
```

This submits a CPU job (`--partition=hour`, 16 cores), a GPU job
(`--partition=gpu-day --gres=gpu:1`), and a plotting job that runs after both,
all writing into a shared `results_<timestamp>/` directory. Watch with
`squeue -u $USER`.

Individual pieces:
```bash
sbatch submit_cpu.sbatch        # CPU only
sbatch submit_gpu.sbatch        # GPU only
.dev/bin/python run_benchmarks.py --device cpu     # run directly (no SLURM)
.dev/bin/python plot_results.py --results-dir results_<ts>
```

## Outputs

```
results_<timestamp>/
  cpu/  {STRUCT}_{method}.json   summary.csv
  gpu/  {STRUCT}_{method}.json   summary.csv
  figure3_sf_comparison_cpu.png
  figure3_sf_comparison_gpu.png
```

`summary.csv` columns: `device, structure, n_atoms, n_reflections, d_min,
method, dtype, torch_threads, fwd_{mean,min,max}, fwd_bwd_{mean,min,max},
mem_fwd_mb, mem_fwd_bwd_mb`.

## Files

- `benchmark_worker.py` — benchmarks one `(method, structure)` in its own process.
- `run_benchmarks.py` — orchestrates all method×structure runs for one device.
- `submit_cpu.sbatch`, `submit_gpu.sbatch`, `submit_all.sh` — SLURM submission.
- `plot_results.py` — grouped bar charts per device.
