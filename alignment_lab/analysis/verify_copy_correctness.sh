#!/bin/bash
# Correctness half of the build_grid=False change. No timings here: this runs
# wherever there is a free slot, and runtime numbers are only comparable on the
# pinned EPYC 9335. What matters is that |F_calc| is bit-identical.
#SBATCH --job-name=frf_vcorr
#SBATCH --output=alignment_lab/slurm/%x_%j.out
#SBATCH --error=alignment_lab/slurm/%x_%j.err
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
mkdir -p alignment_lab/slurm
FILT="Loaded|LINK|Wilson|found non|FrenchWilson|Reflections:|Resolution:|Space group|Centric:|✓|Parametrization|Warning|warn|  from |^ *$"
echo "host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"

echo "=== |F_calc| from the box path, grid-building vs skipped ==="
"$PY" -u -c "
import math, torch
torch.set_grad_enabled(False)
from alignment_lab.lab.benchmark import load_case
from torchref.symmetry.cell import Cell
from torchref.experimental.alignment.frf.dense_calc import model_sf_abs
from torchref.experimental.alignment.frf.api import phaser_lmax_resolution

def box_path(model, d_min, d_max, build_grid):
    m = model.copy(build_grid=build_grid)
    coords = m.xyz(); dev = coords.device
    extent = (coords - coords.mean(0)).norm(dim=-1).max().item()
    a = float(2.0 * 2.0 * extent)
    m.max_res = float(d_min); m.spacegroup = 'P 1'
    m.cell = Cell([a, a, a, 90., 90., 90.], device=dev)
    nmax = int(math.ceil(a / d_min))
    idx = torch.arange(-nmax, nmax + 1, device=dev)
    H, K, Lg = torch.meshgrid(idx, idx, idx, indexing='ij')
    hkl = torch.stack([H.reshape(-1), K.reshape(-1), Lg.reshape(-1)], -1).to(torch.long)
    smag = hkl.to(torch.float64).norm(dim=-1) / a
    hkl = hkl[(smag >= 1.0/d_max) & (smag <= 1.0/d_min)].contiguous()
    return model_sf_abs(m, hkl)

bad = 0
for name in ('3K7M', '1DAW'):
    model, data = load_case(name); model.verbose = 0
    rb = data.cell.reciprocal_basis_matrix.to(torch.float64)
    d_min_data = 1.0 / (data.hkl.to(torch.float64) @ rb).norm(dim=-1).max().item()
    r = float((model.xyz() - model.xyz().mean(0)).norm(dim=-1).mean().item())
    for cap in (64, 100):
        L, d_min = phaser_lmax_resolution(r, d_min_data, cap)
        a = box_path(model, d_min, 100.0, False)
        b = box_path(model, d_min, 100.0, True)
        ok = torch.equal(a, b)
        bad += 0 if ok else 1
        md = 0.0 if ok else (a-b).abs().max().item()
        print(f'  {name} cap{cap}: bit-identical={ok}  n={a.numel()}  max|dF|={md:.3e}')
print('MISMATCHES:', bad)
" 2>&1 | grep -vE "$FILT"

echo "=== the caller must not be mutated ==="
"$PY" -u -c "
import torch
torch.set_grad_enabled(False)
from alignment_lab.lab.benchmark import load_case
from torchref.experimental.alignment.frf.dense_calc import dense_calc_via_box
model, data = load_case('1DAW'); model.verbose = 0
before = (str(model.spacegroup), model.max_res, model.cell.data.clone(),
          model.xyz().clone())
dense_calc_via_box(model, 100.0, 4.0, pad=2.0)
after = (str(model.spacegroup), model.max_res, model.cell.data.clone(),
         model.xyz().clone())
print('  spacegroup preserved:', before[0] == after[0], before[0])
print('  max_res preserved   :', before[1] == after[1], before[1])
print('  cell preserved      :', torch.equal(before[2], after[2]))
print('  coords preserved    :', torch.equal(before[3], after[3]))
print('  grid still present  :', model._fft.real_space_grid is not None)
" 2>&1 | grep -vE "$FILT"

echo "=== tests ==="
"$PY" -u -m pytest -q tests/unit/model tests/unit/alignment tests/unit/frf_separate 2>&1 | tail -12
echo "exit_code=$?"
