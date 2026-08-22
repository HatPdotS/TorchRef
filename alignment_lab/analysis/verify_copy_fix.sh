#!/bin/bash
# Verify the build_grid=False change in dense_calc_via_box:
#   1. |F_calc| must be bit-identical to the grid-building path.
#   2. Truth ranks must not move.
#   3. Report the new stage table, per cap (cap64 is what ships).
#SBATCH --job-name=frf_vcopy
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

echo "=== 1. |F_calc| identical, and the copy cost ==="
"$PY" -u -c "
import math, time, torch
torch.set_grad_enabled(False)
from alignment_lab.lab.benchmark import load_case
from torchref.symmetry.cell import Cell
from torchref.experimental.alignment.frf.dense_calc import dense_calc_via_box, model_sf_abs
from torchref.experimental.alignment.frf.api import phaser_lmax_resolution

def box_path(model, d_min, d_max, build_grid):
    '''dense_calc_via_box, with the grid build under our control.'''
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

def timed(fn, n=3):
    fn()
    return min((lambda: (lambda t0: (fn(), time.perf_counter()-t0)[1])(time.perf_counter()))() for _ in range(n))

for name in ('3K7M', '1DAW'):
    model, data = load_case(name)
    model.verbose = 0
    rb = data.cell.reciprocal_basis_matrix.to(torch.float64)
    d_min_data = 1.0 / (data.hkl.to(torch.float64) @ rb).norm(dim=-1).max().item()
    r = float((model.xyz() - model.xyz().mean(0)).norm(dim=-1).mean().item())
    L, d_min = phaser_lmax_resolution(r, d_min_data, 64)
    f_lean = box_path(model, d_min, 100.0, False)
    f_full = box_path(model, d_min, 100.0, True)
    same = torch.equal(f_lean, f_full)
    md = (f_lean - f_full).abs().max().item() if not same else 0.0
    t_lean = timed(lambda: model.copy(build_grid=False))
    t_full = timed(lambda: model.copy(build_grid=True))
    t_stage = timed(lambda: dense_calc_via_box(model, 100.0, d_min, pad=2.0))
    print(f'  {name}: bit-identical={same} (max|dF|={md:.3e}, n={f_lean.numel()})')
    print(f'      model.copy(build_grid=True)  {t_full*1e3:8.1f} ms')
    print(f'      model.copy(build_grid=False) {t_lean*1e3:8.1f} ms')
    print(f'      dense_calc_via_box now       {t_stage*1e3:8.1f} ms')
" 2>&1 | grep -vE "$FILT"

echo "=== 2/3. ranks and the new stage table, per cap ==="
for pdb in 3K7M 1DAW; do
  for arm in cap64 cap100; do
    "$PY" -u -m alignment_lab.diagnostics.frf_benchmark \
        --pdb "$pdb" --arms "$arm" --trials 2 2>&1 | grep -vE "$FILT"
  done
done

echo "=== 4. tests ==="
"$PY" -u -m pytest -q tests/unit/model/test_copy.py tests/unit/alignment \
    tests/unit/frf_separate 2>&1 | tail -15
echo "exit_code=$?"
