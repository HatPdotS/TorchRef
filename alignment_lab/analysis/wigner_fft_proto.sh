#!/bin/bash
# The J_y eigenvalues are the integers -l..l, so S(beta) is a trigonometric
# polynomial in beta: accumulate its Fourier coefficients once (no beta axis)
# and get every beta from one FFT. Prototype + correctness + timing, against
# the current per-beta matmul.
#SBATCH --job-name=frf_wfft
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
import math, time, torch
torch.set_grad_enabled(False)
from torchref.experimental.alignment.frf.wigner_d import (
    wigner_contraction_per_beta, _wigner_eig_table)

dev = torch.device('cpu')

def contraction_fft(xi, betas):
    L = xi.shape[0]; dim = 2 * L - 1; c = L - 1
    n_beta = betas.shape[0]
    N = 2 * n_beta                       # betas must be j * 2*pi/N
    C = torch.zeros((dim, dim, N), dtype=torch.complex128)
    C[c, c, 0] += 2.0 * xi[0, c, c]      # l=0: d^0 = 1, the halving is undone below
    eig = _wigner_eig_table(L, dev)
    for l in range(1, L):
        w, V = eig[l - 1]
        lo, hi = c - l, c + l + 1
        xi_l = xi[l, lo:hi, lo:hi]
        k = (torch.round(w).to(torch.long)) % N
        G = V.unsqueeze(1) * V.conj().unsqueeze(0)        # (sz, sz, sz) over k
        blk = C[lo:hi, lo:hi]
        blk.index_add_(2, k, xi_l.unsqueeze(-1) * G)
        blk.index_add_(2, (-k) % N, xi_l.unsqueeze(-1) * G.transpose(0, 1))
    full = torch.fft.fft(C, n=N, dim=-1)
    return 0.5 * full[..., :n_beta].permute(2, 0, 1).contiguous()

def best(fn, n=3):
    fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); out.append(time.perf_counter() - t0)
    return min(out)

for cap in (64, 100):
    L = cap + 1
    n_beta = int(math.ceil(180.0 / 3.0))
    betas = torch.arange(n_beta, dtype=torch.float64) * 3.0 * (math.pi / 180.0)
    xi = torch.randn(L, 2*L-1, 2*L-1, dtype=torch.complex128) * 1e-3

    # Are the eigenvalues really integers? The whole method rests on it.
    eig = _wigner_eig_table(L, dev)
    dev_max = max(float((w - torch.round(w)).abs().max()) for w, _ in eig)

    ref = wigner_contraction_per_beta(xi, betas)
    got = contraction_fft(xi, betas)
    rel = float((got - ref).abs().max() / ref.abs().max())

    t_ref = best(lambda: wigner_contraction_per_beta(xi, betas))
    t_new = best(lambda: contraction_fft(xi, betas))
    print(f'--- cap{cap} L={L} n_beta={n_beta}')
    print(f'    max |w - round(w)|   {dev_max:.2e}')
    print(f'    rel. difference      {rel:.2e}')
    print(f'    per-beta matmul      {t_ref*1e3:8.1f} ms')
    print(f'    Fourier + one FFT    {t_new*1e3:8.1f} ms   ({t_ref/t_new:.1f}x)')
" 2>&1 | grep -vE "Warning|warn|  from |^ *$"
echo "exit_code=$?"
