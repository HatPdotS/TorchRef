#!/bin/bash
# A RAM memo of the d-blocks only pays off if the table is wanted more than
# once. How many times does one rotation_search ask for it, and at what (L,
# n_beta)? An earlier note claimed the obs and calc sides each build it.
#SBATCH --job-name=frf_wcnt
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
import torch
torch.set_grad_enabled(False)
import torchref.experimental.alignment.frf.wigner_d as wd
from alignment_lab.lab.benchmark import load_case
from torchref.experimental.alignment.rotation_search import rotation_search

calls = []
orig = wd.wigner_contraction_per_beta
def counting(xi, betas):
    calls.append((int(xi.shape[0]), int(betas.shape[0])))
    return orig(xi, betas)
wd.wigner_contraction_per_beta = counting
# the caller imported it by name, so rebind there too
import torchref.experimental.alignment.frf.sitelist_ang as sa
for mod in (sa,):
    if getattr(mod, 'wigner_contraction_per_beta', None) is orig:
        mod.wigner_contraction_per_beta = counting
import sys
for name, mod in list(sys.modules.items()):
    if name.startswith('torchref.experimental.alignment') and \
       getattr(mod, 'wigner_contraction_per_beta', None) is orig:
        mod.wigner_contraction_per_beta = counting
        print('  rebound in', name)

model, data = load_case('1DAW'); model.verbose = 0
for run in (1, 2):
    calls.clear()
    rotation_search(model, data, 0.8, n_peaks=50)
    print(f'  search {run}: {len(calls)} call(s) to wigner_contraction_per_beta -> {calls}')
" 2>&1 | grep -vE "Loaded|LINK|Wilson|found non|FrenchWilson|Reflections:|Resolution:|Space group|Centric:|✓|Parametrization|Warning|warn|  from |^ *$|No CUDA"
echo "done"
