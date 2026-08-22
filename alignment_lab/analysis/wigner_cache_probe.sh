#!/bin/bash
# Two questions about caching the Wigner d-table:
#   1. What fraction of wigner_contraction_per_beta is the data-INdependent
#      d-block build (so, cacheable) vs the contraction against xi (not)?
#   2. Is loading that table off GPFS actually faster than recomputing it?
#SBATCH --job-name=frf_wcache
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
SCRATCH=alignment_lab/runs/wigner_cache_probe
mkdir -p "$SCRATCH"
"$PY" -u -c "
import math, os, time
import torch; torch.set_grad_enabled(False)
from torchref.experimental.alignment.frf.wigner_d import (
    wigner_contraction_per_beta, _wigner_eig_table)

scratch = '$SCRATCH'
dev = torch.device('cpu')

def build_table(L, betas):
    '''The data-independent half: every d^l(beta) block, packed, float32.'''
    eig = _wigner_eig_table(L, dev)
    out = []
    for l in range(1, L):
        w, V = eig[l - 1]
        phase = torch.exp(-1j * betas.unsqueeze(1) * w.unsqueeze(0))
        VP = V.unsqueeze(0) * phase.unsqueeze(1)
        out.append((VP @ V.conj().transpose(-1, -2)).real.to(torch.float32))
    return out

def contract_only(table, xi, L, n_beta):
    '''The data-dependent half, given a prebuilt table.'''
    dim = 2 * L - 1; c = L - 1
    S = torch.zeros((n_beta, dim, dim), dtype=torch.complex128)
    S[:, c, c] += xi[0, c, c]
    for l in range(1, L):
        lo, hi = c - l, c + l + 1
        S[:, lo:hi, lo:hi] += xi[l, lo:hi, lo:hi].unsqueeze(0) * table[l-1].to(torch.complex128)
    return S

def best(fn, n=3):
    fn()
    return min((lambda: (lambda t0: (fn(), time.perf_counter()-t0)[1])(time.perf_counter()))() for _ in range(n))

for cap in (64, 100):
    L = cap + 1
    n_beta = int(math.ceil(180.0 / 3.0))
    betas = torch.arange(n_beta, dtype=torch.float64) * 3.0 * (math.pi/180)
    xi = torch.randn(L, 2*L-1, 2*L-1, dtype=torch.complex128) * 1e-3

    t_full  = best(lambda: wigner_contraction_per_beta(xi, betas))
    t_build = best(lambda: build_table(L, betas))
    table   = build_table(L, betas)
    t_contr = best(lambda: contract_only(table, xi, L, n_beta))

    # Correctness of the split, in float32 storage.
    ref = wigner_contraction_per_beta(xi, betas)
    got = contract_only(table, xi, L, n_beta)
    rel = ((got - ref).abs().max() / ref.abs().max()).item()

    # Round-trip through GPFS, as one packed flat tensor.
    flat = torch.cat([t.reshape(-1) for t in table])
    path = os.path.join(scratch, f'dtable_L{L}.pt')
    t0 = time.perf_counter(); torch.save(flat, path); t_save = time.perf_counter()-t0
    mb = os.path.getsize(path)/1e6
    os.system('sync')
    loads = []
    for _ in range(3):
        t0 = time.perf_counter(); torch.load(path, map_location='cpu'); loads.append(time.perf_counter()-t0)
    t_load_warm = min(loads)
    print(f'--- cap{cap}  L={L}  n_beta={n_beta} ---')
    print(f'  full contraction now      {t_full*1e3:8.1f} ms')
    print(f'    of which d-block build  {t_build*1e3:8.1f} ms   (cacheable)')
    print(f'    of which xi contraction {t_contr*1e3:8.1f} ms   (not)')
    print(f'  float32 table rel.err     {rel:8.2e}')
    print(f'  on disk                   {mb:8.1f} MB   save {t_save*1e3:.0f} ms  '
          f'load(page-cached) {t_load_warm*1e3:.0f} ms')
    os.remove(path)
"
echo "exit_code=$?"
