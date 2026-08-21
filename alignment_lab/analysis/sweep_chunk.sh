#!/bin/bash
# Sweep the SH-Bessel expansion's chunk width. The loop body changed -- it is now
# elementwise plus a scatter, with no GEMM -- so an earlier conclusion that
# narrow chunks hurt no longer applies and the width has to be re-measured.
# Narrow chunks keep the recurrence's three rolling rows in cache; wide ones cut
# Python and dispatch overhead.
#
#   sbatch --partition=hour --time=00:55:00 --exclusive --mem=0 \
#          --constraint=cpu_epyc9335 alignment_lab/analysis/sweep_chunk.sh
#SBATCH --job-name=frf_chunk
#SBATCH --output=alignment_lab/slurm/%x_%j.out
#SBATCH --error=alignment_lab/slurm/%x_%j.err
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
mkdir -p alignment_lab/slurm
echo "host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
"$PY" -u -c "
import sys, time
sys.path.insert(0,'alignment_lab')
import torch; torch.set_grad_enabled(False)
import torchref.experimental.alignment.frf.data_mr as dm
from lab import FRFConfig, orbit_rank, rotated_case, run_frf, seed_for

BUDGETS_MB = [2, 8, 32, 128, 256, 1024]
cases = {p: rotated_case(p, seed_for(p, 0)) for p in ('1DAW', '3K7M')}
for cap in (64, 100):
    cfg = FRFConfig(lmax_cap=cap, n_peaks=200)
    for m, d, _ in cases.values():
        run_frf(m, d, cfg, capture_arf=False)                 # warm up
    res = {b: {p: [] for p in cases} for b in BUDGETS_MB}
    rank = {}
    for rep in range(2):
        for b in BUDGETS_MB:                                  # interleaved
            dm.CLUSTER_CHUNK_BYTES = b * 1_000_000
            for p, (m, d, R) in cases.items():
                t0 = time.perf_counter()
                r = run_frf(m, d, cfg, capture_arf=False)
                res[b][p].append(time.perf_counter() - t0)
                k, _ = orbit_rank(r.peaks, R,
                    d.spacegroup.matrices.to(torch.float64).cpu(),
                    reciprocal_basis=d.cell.reciprocal_basis_matrix.to(torch.float64).cpu(),
                    side='left', frame='cart')
                rank[(b, p)] = k
    print(f'--- cap{cap} (best of 2, seconds; rank in brackets) ---')
    print('  ' + 'chunk MB'.rjust(9) + ''.join(f'{p:>18}' for p in cases))
    for b in BUDGETS_MB:
        cells = ''.join(f'{min(res[b][p]):12.2f} [{rank[(b,p)]:>3}]' for p in cases)
        print(f'  {b:9d}{cells}')
"
echo "exit_code=$?"
