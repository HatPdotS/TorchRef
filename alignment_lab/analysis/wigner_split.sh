#!/bin/bash
# Inside wigner_contraction_per_beta, how much is data-INdependent (so
# cacheable) and how much is the contraction against xi (so not)?
#SBATCH --job-name=frf_wigner
#SBATCH --output=alignment_lab/slurm/%x_%j.out
#SBATCH --error=alignment_lab/slurm/%x_%j.err
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
mkdir -p alignment_lab/slurm
echo "host=$(hostname)"
"$PY" -u -c "
import time
import torch; torch.set_grad_enabled(False)
from torchref.experimental.alignment.frf.wigner_d import (
    wigner_contraction_per_beta, _wigner_eig_table)
from torchref.experimental.alignment.frf.sitelist_ang import _SAMPLE_LIST_CACHE

for cap, sampling in ((64, 3.0), (100, 3.0)):
    L = cap + 1
    n_beta = int(__import__('math').ceil(180.0 / sampling))
    betas = torch.arange(n_beta, dtype=torch.float64) * sampling * (3.141592653589793/180)
    xi = torch.randn(L, 2*L-1, 2*L-1, dtype=torch.complex128) * 1e-3

    t0 = time.perf_counter(); _wigner_eig_table(L, torch.device('cpu'))
    cold_eig = time.perf_counter() - t0
    t0 = time.perf_counter(); _wigner_eig_table(L, torch.device('cpu'))
    warm_eig = time.perf_counter() - t0

    wigner_contraction_per_beta(xi, betas)            # warm everything
    t0 = time.perf_counter()
    wigner_contraction_per_beta(xi, betas)
    total = time.perf_counter() - t0

    # Size of the d-table if it were precomputed and stored, float32 real.
    entries = sum((2*l+1)**2 for l in range(L)) * n_beta
    print(f'L={L:4d} n_beta={n_beta:3d} | eig cold {cold_eig*1e3:7.1f} ms  '
          f'warm {warm_eig*1e3:5.2f} ms | contraction total {total*1e3:7.1f} ms '
          f'| d-table would be {entries*4/1e6:7.1f} MB float32')
"
echo "exit_code=$?"
