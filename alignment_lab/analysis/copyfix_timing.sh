#!/bin/bash
# Payoff of build_grid=False. NOTE: whatever node this lands on, the absolute
# ms are NOT comparable with the EPYC 9335 tables -- only the before/after
# ratio measured inside this one job is.
#SBATCH --job-name=frf_ctime
#SBATCH --output=alignment_lab/slurm/%x_%j.out
#SBATCH --error=alignment_lab/slurm/%x_%j.err
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
mkdir -p alignment_lab/slurm
FILT="Loaded|LINK|Wilson|found non|FrenchWilson|Reflections:|Resolution:|Space group|Centric:|✓|Parametrization|Warning|warn|  from |^ *$|No CUDA"
echo "host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
echo "=== the copy itself, and the whole dense-calc stage ==="
"$PY" -u -c "
import math, time, torch
torch.set_grad_enabled(False)
from alignment_lab.lab.benchmark import load_case
from torchref.symmetry.cell import Cell
from torchref.experimental.alignment.frf.dense_calc import dense_calc_via_box, model_sf_abs
from torchref.experimental.alignment.frf.api import phaser_lmax_resolution

def box_path(model, d_min, d_max, build_grid):
    m = model.copy(build_grid=build_grid)
    coords = m.xyz(); dev = coords.device
    a = float(4.0 * (coords - coords.mean(0)).norm(dim=-1).max().item())
    m.max_res = float(d_min); m.spacegroup = 'P 1'
    m.cell = Cell([a, a, a, 90., 90., 90.], device=dev)
    nmax = int(math.ceil(a / d_min))
    idx = torch.arange(-nmax, nmax + 1, device=dev)
    H, K, Lg = torch.meshgrid(idx, idx, idx, indexing='ij')
    hkl = torch.stack([H.reshape(-1), K.reshape(-1), Lg.reshape(-1)], -1).to(torch.long)
    smag = hkl.to(torch.float64).norm(dim=-1) / a
    hkl = hkl[(smag >= 1.0/d_max) & (smag <= 1.0/d_min)].contiguous()
    return model_sf_abs(m, hkl)

def best(fn, n=3):
    fn()
    return min((lambda: (lambda t0: (fn(), time.perf_counter()-t0)[1])(time.perf_counter()))() for _ in range(n))

for name in ('3K7M', '1DAW'):
    model, data = load_case(name); model.verbose = 0
    rb = data.cell.reciprocal_basis_matrix.to(torch.float64)
    d_min_data = 1.0 / (data.hkl.to(torch.float64) @ rb).norm(dim=-1).max().item()
    r = float((model.xyz() - model.xyz().mean(0)).norm(dim=-1).mean().item())
    L, d_min = phaser_lmax_resolution(r, d_min_data, 64)
    tc_on  = best(lambda: model.copy(build_grid=True))
    tc_off = best(lambda: model.copy(build_grid=False))
    t_on   = best(lambda: box_path(model, d_min, 100.0, True))
    t_off  = best(lambda: box_path(model, d_min, 100.0, False))
    print(f'  {name} (cap64, d_min={d_min:.2f}A)')
    print(f'      model.copy()          grid {tc_on*1e3:8.1f} -> no grid {tc_off*1e3:7.1f} ms  ({tc_on/tc_off:5.1f}x)')
    print(f'      whole box path        grid {t_on*1e3:8.1f} -> no grid {t_off*1e3:7.1f} ms  ({t_on/t_off:5.1f}x)')
    print(f'      dense_calc_via_box now     {best(lambda: dense_calc_via_box(model, 100.0, d_min, pad=2.0))*1e3:8.1f} ms')
" 2>&1 | grep -vE "$FILT"
echo "=== whole rotation search, per cap ==="
for pdb in 3K7M 1DAW; do
  for arm in cap64 cap100; do
    "$PY" -u -m alignment_lab.diagnostics.frf_benchmark \
        --pdb "$pdb" --arms "$arm" --trials 2 2>&1 | grep -vE "$FILT"
  done
done
echo "done"
