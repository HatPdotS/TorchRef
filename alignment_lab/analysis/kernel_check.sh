#!/bin/bash
# Does the fused kernel build, refuse float64, and agree with the portable
# reference? Correctness only, so it does not need an exclusive node -- the
# timing A/B does.
#SBATCH --job-name=frf_kernel_check
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
import torch; torch.set_grad_enabled(False)
from torchref.experimental.alignment.frf.kernels.cpu import legendre_shell as K
from torchref.experimental.alignment.frf.kernels import portable as P
from torchref.experimental.alignment.sh import legendre_recurrence_coefficients

print('available:', K.available())
if not K.available():
    print('why:', K.why_unavailable())
    e = K.last_error()
    if e: print(e[1][-4000:])
    raise SystemExit(1)

try:
    z = torch.zeros(2, 3, 5, dtype=torch.float64)
    K.legendre_shell_accumulate(z, z.clone(),
        torch.zeros(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64),
        torch.zeros(4, 5, dtype=torch.float64), torch.zeros(4, 5, dtype=torch.float64),
        torch.zeros(4, dtype=torch.long), torch.zeros(5, 5, dtype=torch.float64),
        torch.zeros(5, 5, dtype=torch.float64), torch.zeros(5, dtype=torch.float64))
    print('PROBLEM: float64 accepted')
except Exception as e:
    msg = str(e)
    ok = ('float32 only' in msg) or ('dtype' in msg)
    print(('float64 refused: ' if ok else 'WRONG ERROR (not a dtype refusal): ')
          + msg.splitlines()[0][:120])
    if not ok:
        raise SystemExit(1)

g = torch.Generator().manual_seed(4)
for L, n_c, n_sh in ((13, 500, 40), (65, 4000, 300), (101, 3000, 250)):
    n_even = (L - 1 if (L-1) % 2 == 0 else L - 2) // 2
    ct = (2*torch.rand(n_c, generator=g, dtype=torch.float32)-1)
    st = (1-ct*ct).clamp(min=0).sqrt()
    Dr = torch.randn(n_c, L, generator=g, dtype=torch.float32)
    Di = torch.randn(n_c, L, generator=g, dtype=torch.float32)
    sh = torch.sort(torch.randint(0, n_sh, (n_c,), generator=g))[0]
    a, b, se = legendre_recurrence_coefficients(L, torch.float32, torch.device('cpu'))
    ref_r = torch.zeros(n_even, n_sh, L, dtype=torch.float32); ref_i = torch.zeros_like(ref_r)
    got_r = torch.zeros_like(ref_r); got_i = torch.zeros_like(ref_r)
    P.legendre_shell_accumulate(ref_r, ref_i, ct, st, Dr, Di, sh, a, b, se)
    K.legendre_shell_accumulate(got_r, got_i, ct, st, Dr, Di, sh, a, b, se)
    sr = max(ref_r.abs().max().item(), 1e-30); si = max(ref_i.abs().max().item(), 1e-30)
    print(f'  L={L:3d}: rel err  re {(got_r-ref_r).abs().max().item()/sr:.2e}'
          f'  im {(got_i-ref_i).abs().max().item()/si:.2e}')
"
echo "exit_code=$?"
