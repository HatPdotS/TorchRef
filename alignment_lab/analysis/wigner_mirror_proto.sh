#!/bin/bash
# d^l(pi - beta) is d^l(beta) up to an m-flip and a sign, and the beta grid
# (0, 3, ..., 177 deg) is closed under beta -> pi - beta. So the batched matmul
# only needs 31 of the 60 beta values. No data file, no kernel.
#SBATCH --job-name=frf_wmir
#SBATCH --output=alignment_lab/slurm/%x_%j.out
#SBATCH --error=alignment_lab/slurm/%x_%j.err
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
mkdir -p alignment_lab/slurm
echo "host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
"$PY" -u -c "
import math, time, torch
torch.set_grad_enabled(False)
from torchref.experimental.alignment.frf.wigner_d import (
    wigner_contraction_per_beta, _wigner_eig_table)
dev = torch.device('cpu')

def d_block(w, V, betas):
    phase = torch.exp(-1j * betas.unsqueeze(1) * w.unsqueeze(0))
    return ((V.unsqueeze(0) * phase.unsqueeze(1)) @ V.conj().transpose(-1, -2)).real

# Which mirror identity holds? Test rather than trust.
L = 9
eig = _wigner_eig_table(L, dev)
for l in (1, 3, 6, 8):
    w, V = eig[l - 1]
    b = torch.tensor([0.37, 1.11, 2.05], dtype=torch.float64)
    lhs = d_block(w, V, math.pi - b)
    base = d_block(w, V, b)
    m = torch.arange(-l, l + 1, dtype=torch.float64)
    s2 = ((-1.0) ** (l + m)).reshape(1, 1, -1)
    s1 = ((-1.0) ** (l + m)).reshape(1, -1, 1)
    f2 = s2 * base.flip(-1)      # (-1)^(l+m2) d[m1, -m2]
    f1 = s1 * base.flip(-2)      # (-1)^(l+m1) d[-m1, m2]
    print(f'  l={l}: form(m2-flip) err={float((lhs-f2).abs().max()):.2e}   '
          f'form(m1-flip) err={float((lhs-f1).abs().max()):.2e}')

def contraction_mirror(xi, betas):
    L = xi.shape[0]; dim = 2 * L - 1; c = L - 1
    n_beta = betas.shape[0]
    half = n_beta // 2 + 1                       # 0..30 for n_beta=60
    src = n_beta - torch.arange(half, n_beta)    # beta_j -> pi - beta_j
    S = torch.zeros((n_beta, dim, dim), dtype=torch.complex128)
    S[:, c, c] += xi[0, c, c]
    eig = _wigner_eig_table(L, dev)
    for l in range(1, L):
        w, V = eig[l - 1]
        d_h = d_block(w, V, betas[:half])
        m = torch.arange(-l, l + 1, dtype=d_h.dtype)
        sgn = ((-1.0) ** (l + m)).reshape(1, 1, -1)
        d_l = torch.cat([d_h, sgn * d_h[src].flip(-1)], dim=0)
        lo, hi = c - l, c + l + 1
        S[:, lo:hi, lo:hi] += xi[l, lo:hi, lo:hi].unsqueeze(0) * d_l.to(torch.complex128)
    return S

def best(fn, n=3):
    fn()
    return min((lambda: (lambda t0: (fn(), time.perf_counter()-t0)[1])(time.perf_counter()))() for _ in range(n))

for cap in (64, 100):
    L = cap + 1
    n_beta = int(math.ceil(180.0 / 3.0))
    betas = torch.arange(n_beta, dtype=torch.float64) * 3.0 * (math.pi / 180.0)
    xi = torch.randn(L, 2*L-1, 2*L-1, dtype=torch.complex128) * 1e-3
    ref = wigner_contraction_per_beta(xi, betas)
    got = contraction_mirror(xi, betas)
    rel = float((got - ref).abs().max() / ref.abs().max())
    t_ref = best(lambda: wigner_contraction_per_beta(xi, betas))
    t_new = best(lambda: contraction_mirror(xi, betas))
    print(f'--- cap{cap} L={L}: rel diff {rel:.2e}   '
          f'current {t_ref*1e3:7.1f} ms   mirrored {t_new*1e3:7.1f} ms  '
          f'({t_ref/t_new:.2f}x)')
" 2>&1 | grep -vE "Warning|warn|  from |^ *$|No CUDA"
echo "exit_code=$?"
