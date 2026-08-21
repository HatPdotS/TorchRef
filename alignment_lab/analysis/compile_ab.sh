#!/bin/bash
# A/B the compiled Legendre step against the eager one. Interleaved repeats, so
# any drift on the node hits both arms alike, and the truth rank is reported
# beside each timing -- a faster build that changes the answer is not faster.
#
#   sbatch --partition=hour --time=00:55:00 --exclusive --mem=0 \
#          --constraint=cpu_epyc9335 alignment_lab/analysis/compile_ab.sh
#SBATCH --job-name=frf_compile_ab
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

cases = {p: rotated_case(p, seed_for(p, 0)) for p in ('1DAW', '3K7M')}
for cap in (64, 100):
    cfg = FRFConfig(lmax_cap=cap, n_peaks=200)
    # Warm up BOTH arms: the compiled one pays its build on first call, and
    # charging that to the measurement would answer a different question.
    for compiled in (False, True):
        dm.COMPILE_LEGENDRE_STEP = compiled
        for m, d, _ in cases.values():
            run_frf(m, d, cfg, capture_arf=False)
    res = {a: {p: [] for p in cases} for a in ('eager', 'compiled')}
    rank = {}
    for rep in range(3):
        for arm, compiled in (('eager', False), ('compiled', True)):
            dm.COMPILE_LEGENDRE_STEP = compiled
            for p, (m, d, R) in cases.items():
                t0 = time.perf_counter()
                r = run_frf(m, d, cfg, capture_arf=False)
                res[arm][p].append(time.perf_counter() - t0)
                k, _ = orbit_rank(r.peaks, R,
                    d.spacegroup.matrices.to(torch.float64).cpu(),
                    reciprocal_basis=d.cell.reciprocal_basis_matrix.to(torch.float64).cpu(),
                    side='left', frame='cart')
                rank[(arm, p)] = k
    print(f'--- cap{cap} (best of 3, seconds) ---')
    for p in cases:
        e, c = min(res['eager'][p]), min(res['compiled'][p])
        print(f'  {p:6s} eager {e:7.2f} [rank {rank[(\"eager\",p)]:>3}]   '
              f'compiled {c:7.2f} [rank {rank[(\"compiled\",p)]:>3}]   '
              f'speedup {e/max(c,1e-9):5.2f}x')
"
echo "exit_code=$?"
