#!/bin/bash
# model.copy() is ~97% of dense_calc_via_box, which is ~50% of a shipped
# rotation search. What inside it costs the second, and is it steady state or
# first-call lazy initialisation?
#SBATCH --job-name=frf_copy
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
import copy as copy_module, time, torch
torch.set_grad_enabled(False)
from alignment_lab.lab.benchmark import load_case

def reps(fn, n=5):
    fn()
    return [ (lambda t0: (fn(), time.perf_counter()-t0)[1])(time.perf_counter()) for _ in range(n) ]

for name in ('3K7M', '1DAW'):
    model, data = load_case(name)
    model.verbose = 0
    g = model._fft.real_space_grid if model._fft is not None else None
    print(f'--- {name}: {len(model.pdb)} atoms  cell={[round(x,1) for x in model.cell.parameters_list()] if hasattr(model.cell,\"parameters_list\") else \"?\"}  '
          f'max_res={model.max_res}  grid={None if g is None else tuple(g.shape)}')
    ts = reps(lambda: model.copy())
    print(f'      model.copy() x5:  ' + '  '.join(f'{t*1e3:.0f}' for t in ts) + ' ms')

    # The pieces, each timed on its own.
    print(f'      pdb.copy(deep=True)        {min(reps(lambda: model.pdb.copy(deep=True)))*1e3:8.1f} ms')
    if model._parametrization is not None:
        print(f'      deepcopy(_parametrization) {min(reps(lambda: copy_module.deepcopy(model._parametrization)))*1e3:8.1f} ms')
    if model._fft is not None:
        print(f'      _fft.copy()                {min(reps(lambda: model._fft.copy()))*1e3:8.1f} ms')
        mc = model.copy()
        print(f'      setup_grid(max_res)        {min(reps(lambda: mc.setup_grid(max_res=model.max_res)))*1e3:8.1f} ms')
    print(f'      _rebuild_sf_indices()      {min(reps(lambda: model._rebuild_sf_indices()))*1e3:8.1f} ms')
    print(f'      cell.clone()               {min(reps(lambda: model.cell.clone()))*1e3:8.1f} ms')

    # What the FRF actually needs the copy for: it mutates max_res, spacegroup, cell.
    # Would a copy taken AFTER shrinking max_res be cheaper?
    print(f'      copy() with max_res=4.15   {min(reps(lambda: (lambda m: m)(model.copy())))*1e3:8.1f} ms  (baseline)')
" 2>&1 | grep -vE "Loaded|LINK|Wilson|found non|FrenchWilson|Reflections:|Resolution:|Space group|Centric:|✓|Parametrization|Warning|warn|  from |^ *$"
echo "exit_code=$?"
