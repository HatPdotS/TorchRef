#!/bin/bash
# Per-cap stage breakdown (the shipped config is cap64; earlier tables medianed
# cap64 and cap100 together), plus a breakdown INSIDE dense_calc_via_box.
#SBATCH --job-name=frf_dense
#SBATCH --output=alignment_lab/slurm/%x_%j.out
#SBATCH --error=alignment_lab/slurm/%x_%j.err
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
mkdir -p alignment_lab/slurm
echo "host=$(hostname)"
FILT="Loaded|LINK|Wilson|found non|FrenchWilson|Reflections:|Resolution:|Space group|Centric:|✓|Parametrization|Warning|warn|  from |^ *$"
for pdb in 3K7M 1DAW; do
  for arm in cap64 cap100; do
    "$PY" -u -m alignment_lab.diagnostics.frf_benchmark \
        --pdb "$pdb" --arms "$arm" --trials 2 2>&1 | grep -vE "$FILT"
  done
done
echo "=== inside dense_calc_via_box ==="
"$PY" -u -c "
import math, time, torch
torch.set_grad_enabled(False)
from alignment_lab.lab.benchmark import BENCHMARK, load_case
from torchref.symmetry.cell import Cell
from torchref.experimental.alignment.frf.dense_calc import model_sf_abs
from torchref.experimental.alignment.frf.sitelist_ang import phaser_lmax_resolution

for name in ('3K7M', '1DAW'):
    case = load_case(name)
    model, data = case.model, case.data
    d_min_data = 1.0 / data.hkl_to_s(data.hkl).norm(dim=-1).max().item()
    d_max = 100.0
    r = float((model.xyz() - model.xyz().mean(0)).norm(dim=-1).mean().item())
    for cap in (64, 100):
        L, d_min = phaser_lmax_resolution(r, d_min_data, cap)
        t = {}
        t0 = time.perf_counter()
        m = model.copy()
        coords = m.xyz(); dev = coords.device
        extent = (coords - coords.mean(0)).norm(dim=-1).max().item()
        a = float(2.0 * 2.0 * extent)
        t['copy'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        m.max_res = float(d_min); m.spacegroup = 'P 1'
        m.cell = Cell([a, a, a, 90., 90., 90.], device=dev)
        t['cell+grid setup'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        nmax = int(math.ceil(a / d_min))
        idx = torch.arange(-nmax, nmax + 1, device=dev)
        H, K, Lg = torch.meshgrid(idx, idx, idx, indexing='ij')
        hkl = torch.stack([H.reshape(-1), K.reshape(-1), Lg.reshape(-1)], -1).to(torch.long)
        smag = hkl.to(torch.float64).norm(dim=-1) / a
        hkl = hkl[(smag >= 1.0/d_max) & (smag <= 1.0/d_min)].contiguous()
        t['hkl enumerate'] = time.perf_counter() - t0
        model_sf_abs(m, hkl)                      # warm
        t0 = time.perf_counter(); model_sf_abs(m, hkl); t['model_sf_abs'] = time.perf_counter()-t0
        grid = m.cell.compute_grid_size(m.max_res)
        print(f'{name} cap{cap}: d_min={d_min:.2f}A box={a:.0f}A grid={grid} '
              f'spacing={a/grid[0]:.2f}A  n_hkl={hkl.shape[0]} '
              f'(enumerated {(2*nmax+1)**3})')
        tot = sum(t.values())
        for k, v in t.items():
            print(f'    {k:18s} {v*1e3:8.1f} ms  {100*v/tot:5.1f}%')
" 2>&1 | grep -vE "$FILT"
echo "exit_code=$?"
