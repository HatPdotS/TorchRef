#!/bin/bash
# dense_calc_via_box is ~50% of a shipped (cap64) rotation search. Where inside
# it does the time go, and does it actually track the FFT grid fineness? If the
# cost is insensitive to the grid, a coarser grid buys nothing and the
# artificial-B route is pointless.
#SBATCH --job-name=frf_dint
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
"$PY" -u -c "
import math, time, torch
torch.set_grad_enabled(False)
from alignment_lab.lab.benchmark import load_case
from torchref.symmetry.cell import Cell
from torchref.experimental.alignment.frf.dense_calc import model_sf_abs
from torchref.experimental.alignment.frf.api import phaser_lmax_resolution

def timeit(fn, n=3):
    fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); out.append(time.perf_counter() - t0)
    return min(out)

for name in ('3K7M', '1DAW'):
    model, data = load_case(name)
    rb = data.cell.reciprocal_basis_matrix.to(torch.float64)
    d_min_data = 1.0 / (data.hkl.to(torch.float64) @ rb).norm(dim=-1).max().item()
    r = float((model.xyz() - model.xyz().mean(0)).norm(dim=-1).mean().item())
    for cap in (64,):
        L, d_min = phaser_lmax_resolution(r, d_min_data, cap)
        t = {}
        t0 = time.perf_counter()
        m = model.copy()
        t['model.copy()'] = time.perf_counter() - t0
        coords = m.xyz(); dev = coords.device
        extent = (coords - coords.mean(0)).norm(dim=-1).max().item()
        a = float(2.0 * 2.0 * extent)
        t0 = time.perf_counter()
        m.max_res = float(d_min); m.spacegroup = 'P 1'
        m.cell = Cell([a, a, a, 90., 90., 90.], device=dev)
        t['cell/grid setup'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        nmax = int(math.ceil(a / d_min))
        idx = torch.arange(-nmax, nmax + 1, device=dev)
        H, K, Lg = torch.meshgrid(idx, idx, idx, indexing='ij')
        hkl = torch.stack([H.reshape(-1), K.reshape(-1), Lg.reshape(-1)], -1).to(torch.long)
        smag = hkl.to(torch.float64).norm(dim=-1) / a
        hkl = hkl[(smag >= 1.0/100.0) & (smag <= 1.0/d_min)].contiguous()
        t['hkl enumerate'] = time.perf_counter() - t0
        t['model_sf_abs (1st)'] = timeit(lambda: model_sf_abs(m, hkl), n=1)
        t['model_sf_abs (warm)'] = timeit(lambda: model_sf_abs(m, hkl))
        g = m.cell.compute_grid_size(m.max_res)
        print(f'--- {name} cap{cap}: d_min={d_min:.2f}A  box={a:.0f}A  '
              f'grid={g[0]}^3  spacing={a/g[0]:.2f}A  n_hkl={hkl.shape[0]} '
              f'of {(2*nmax+1)**3} enumerated')
        for k, v in t.items():
            print(f'      {k:22s} {v*1e3:8.1f} ms')

        # Does the cost track the grid at all? Same reflections, different grid.
        print('      grid sensitivity (same hkl list, max_res only):')
        for fac in (0.5, 1.0, 1.5, 2.0):
            mm = model.copy()
            mm.max_res = float(d_min / fac)   # fac>1 => finer grid
            mm.spacegroup = 'P 1'
            mm.cell = Cell([a, a, a, 90., 90., 90.], device=dev)
            gg = mm.cell.compute_grid_size(mm.max_res)
            dt = timeit(lambda: model_sf_abs(mm, hkl))
            print(f'        x{fac:<4} grid={gg[0]:4d}^3 spacing={a/gg[0]:5.2f}A  {dt*1e3:8.1f} ms')
" 2>&1 | grep -vE "Loaded|LINK|Wilson|found non|FrenchWilson|Reflections:|Resolution:|Space group|Centric:|✓|Parametrization|Warning|warn|  from |^ *$"
echo "exit_code=$?"
