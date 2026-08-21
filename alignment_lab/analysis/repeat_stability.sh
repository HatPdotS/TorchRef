#!/bin/bash
# Does calling the rotation function repeatedly on one model give the same
# answer? The compiled/eager A/B reported a different truth rank from every
# earlier run, and the only thing it did differently was reuse the model across
# many calls -- so either something accumulates on the model, or the rank is
# less stable than measured.
#SBATCH --job-name=frf_repeat
#SBATCH --output=alignment_lab/slurm/%x_%j.out
#SBATCH --error=alignment_lab/slurm/%x_%j.err
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
mkdir -p alignment_lab/slurm
echo "host=$(hostname)"
"$PY" -u -c "
import sys
sys.path.insert(0,'alignment_lab')
import torch; torch.set_grad_enabled(False)
from lab import FRFConfig, orbit_rank, rotated_case, run_frf, seed_for

def rank_of(r, d, R):
    k, a = orbit_rank(r.peaks, R, d.spacegroup.matrices.to(torch.float64).cpu(),
        reciprocal_basis=d.cell.reciprocal_basis_matrix.to(torch.float64).cpu(),
        side='left', frame='cart')
    return k, a

for pdb in ('1DAW', '3K7M'):
    cfg = FRFConfig(lmax_cap=64, n_peaks=200)
    # A: one model, six calls.
    m, d, R = rotated_case(pdb, seed_for(pdb, 0))
    reused = []
    for i in range(6):
        reused.append(rank_of(run_frf(m, d, cfg, capture_arf=False), d, R))
    # B: a freshly built model for each call -- the control.
    fresh = []
    for i in range(6):
        m2, d2, R2 = rotated_case(pdb, seed_for(pdb, 0))
        fresh.append(rank_of(run_frf(m2, d2, cfg, capture_arf=False), d2, R2))
    print(f'{pdb} model REUSED : ranks {[k for k,_ in reused]} '
          f'angles {[round(a,3) for _,a in reused]}')
    print(f'{pdb} model FRESH  : ranks {[k for k,_ in fresh]} '
          f'angles {[round(a,3) for _,a in fresh]}')
"
echo "exit_code=$?"
