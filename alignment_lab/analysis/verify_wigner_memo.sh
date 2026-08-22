#!/bin/bash
# The memoised small-d blocks: same answer, and how much the second search in a
# process saves. Also the peak-memory cost of holding the blocks.
#SBATCH --job-name=frf_wmemo
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

echo "=== contraction: cold vs memoised, and identity ==="
"$PY" -u -c "
import math, resource, time, torch
torch.set_grad_enabled(False)
from torchref.experimental.alignment.frf.wigner_d import (
    wigner_contraction_per_beta, clear_wigner_d_cache)

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

for cap in (64, 100):
    L = cap + 1
    n_beta = int(math.ceil(180.0 / 3.0))
    betas = torch.arange(n_beta, dtype=torch.float64) * 3.0 * (math.pi/180.0)
    g = torch.Generator().manual_seed(0)
    xi = torch.randn(L, 2*L-1, 2*L-1, generator=g, dtype=torch.float64).to(torch.complex128)
    clear_wigner_d_cache()
    r0 = rss_mb()
    t0 = time.perf_counter(); a = wigner_contraction_per_beta(xi, betas); cold = time.perf_counter()-t0
    r1 = rss_mb()
    warm = min([(lambda t: (wigner_contraction_per_beta(xi, betas), time.perf_counter()-t)[1])(time.perf_counter()) for _ in range(3)])
    b = wigner_contraction_per_beta(xi, betas)
    print(f'  cap{cap} L={L}: cold {cold*1e3:7.1f} ms   memoised {warm*1e3:6.1f} ms  '
          f'({cold/warm:.1f}x)   identical={torch.equal(a, b)}   '
          f'RSS +{r1-r0:.0f} MB')
    clear_wigner_d_cache()
" 2>&1 | grep -vE "$FILT"

echo "=== two searches in one process ==="
"$PY" -u -c "
import time, torch
torch.set_grad_enabled(False)
from alignment_lab.lab.benchmark import load_case
from torchref.experimental.alignment.rotation_search import rotation_search
model, data = load_case('1DAW'); model.verbose = 0
rotation_search(model, data, 0.8, n_peaks=50)          # prewarm the process
prev = None
for i in (1, 2, 3):
    t0 = time.perf_counter(); sol = rotation_search(model, data, 0.8, n_peaks=50)
    dt = time.perf_counter() - t0
    fp = float(sol.scores[:20].double().sum())
    tag = '' if prev is None else ('  same top-20 sum' if fp == prev else '  DIFFERS')
    prev = fp
    print(f'  search {i}: {dt:.3f} s{tag}')
" 2>&1 | grep -vE "$FILT"

echo "=== tests ==="
"$PY" -m pytest -q --no-header -p no:cacheprovider \
    tests/unit/alignment tests/unit/frf_separate tests/unit/model \
    > alignment_lab/slurm/_t_wmemo.log 2>&1
rc=$?
tail -2 alignment_lab/slurm/_t_wmemo.log
echo "rc=$rc"
grep -E "^(FAILED|ERROR)" alignment_lab/slurm/_t_wmemo.log || echo "  no failures"
echo "done"
