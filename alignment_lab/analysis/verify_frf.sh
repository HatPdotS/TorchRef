#!/bin/bash
# Verify a change to the rotation function: tests, then the numbers that decide
# whether the change was worth it. Runs on a compute node with the CPU pinned --
# the login node is contended enough that a profile taken there sent one earlier
# optimisation after the wrong stage.
#
#   sbatch --partition=hour --time=00:55:00 --exclusive --mem=0 \
#          --constraint=cpu_epyc9335 alignment_lab/analysis/verify_frf.sh
#SBATCH --job-name=frf_verify
#SBATCH --output=alignment_lab/slurm/%x_%j.out
#SBATCH --error=alignment_lab/slurm/%x_%j.err
set -uo pipefail
REPO="${FRF_VERIFY_REPO:-/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement}"
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
mkdir -p alignment_lab/slurm
echo "host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
echo "sha=$(git -C "$REPO" rev-parse --short HEAD)"

echo "=== unit tests (the expansion's own invariants included) ==="
"$PY" -m pytest tests/unit/alignment tests/unit/frf_separate alignment_lab/tests -q
rc=$?

echo "=== stage profile and truth rank, cap 64 and cap 100, steady state ==="
FRF_PROFILE=1 "$PY" -u -c "
import sys; sys.path.insert(0,'alignment_lab')
import torch; torch.set_grad_enabled(False)
from lab import FRFConfig, orbit_rank, rotated_case, run_frf, seed_for
for cap in (64, 100):
    for pdb in ('1DAW','3K7M'):
        m,d,R = rotated_case(pdb, seed_for(pdb,0))
        cfg = FRFConfig(lmax_cap=cap, n_peaks=200)
        run_frf(m, d, cfg, capture_arf=False)              # warm up
        r = run_frf(m, d, cfg, capture_arf=False)
        rank, ang = orbit_rank(r.peaks, R, d.spacegroup.matrices.to(torch.float64).cpu(),
            reciprocal_basis=d.cell.reciprocal_basis_matrix.to(torch.float64).cpu(),
            side='left', frame='cart')
        print(f'>>> cap{cap} {pdb} {r.seconds:.2f}s  truth rank {rank} at {ang:.3f} deg')
" 2>&1 | grep -E "FRF_PROFILE|>>>"
echo "exit_tests=$rc"
exit "$rc"
