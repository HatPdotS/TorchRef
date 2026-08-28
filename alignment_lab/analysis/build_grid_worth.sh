#!/bin/bash
# Is ModelFT.copy(build_grid=False) still worth anything after the merge?
#
# It existed because setup_grid also built MapSymmetry -- one (nx,ny,nz,3)
# sampling array per symmetry operation -- which cost 957.8 ms of a 2.06 s
# search. dev moved that out: setup_grid now asks `spacegroup.can_index_directly`
# instead of building an operator. What remains is compute_real_space_grid, an
# (nx,ny,nz,3) coordinate array, so the question is whether that alone is worth a
# parameter.
#SBATCH --job-name=gridworth
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclusive
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "host=$(hostname)"
"$PY" -u - <<'PYEOF' 2>&1 | grep -E "^GRID|Error|Traceback"
import statistics, sys, time
from pathlib import Path
sys.path.insert(0, "alignment_lab")
import torch
torch.set_grad_enabled(False)
from lab import load_case

for pdb in ("1DAW", "3K7M"):
    model, data = load_case(pdb)[:2]
    model.cell = data.cell
    model.spacegroup = str(data.spacegroup.hm)
    model.max_res = 2.0
    model.setup_grid(max_res=2.0)
    g = model._fft.real_space_grid
    print(f"GRID {pdb}: grid {'None' if g is None else tuple(g.shape)}  "
          f"n_ops={data.spacegroup.n_ops}")
    model.copy(); model.copy(build_grid=False)          # warm
    t = {True: [], False: []}
    for r in range(6):
        for bg in ((True, False) if r % 2 == 0 else (False, True)):
            t0 = time.perf_counter(); c = model.copy(build_grid=bg)
            t[bg].append(time.perf_counter() - t0)
            del c
    med_t, med_f = statistics.median(t[True]), statistics.median(t[False])
    pd = sorted(t[True][i] - t[False][i] for i in range(len(t[True])))
    print(f"GRID {pdb}: copy(build_grid=True)  median {1000*med_t:8.2f} ms")
    print(f"GRID {pdb}: copy(build_grid=False) median {1000*med_f:8.2f} ms")
    print(f"GRID {pdb}: paired median saving {1000*statistics.median(pd):8.2f} ms "
          f"({100*statistics.median(pd)/med_t:+.1f}% of the copy)")
PYEOF
echo "rc=$?"
