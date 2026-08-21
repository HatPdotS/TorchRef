#!/bin/bash
# Fused C++ kernel against the portable torch reference: correctness first, then
# speed. Interleaved repeats and the truth rank beside each timing.
#SBATCH --job-name=frf_kernel_ab
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
from torchref.experimental.alignment.frf.kernels.cpu import legendre_shell as K
from torchref.experimental.alignment.frf.kernels import portable as P
from torchref.utils.backends import set_force_portable

print('kernel available:', K.available())
print('float64 must be refused, not reinterpreted:')
try:
    import torch as _t
    z = _t.zeros(2, 3, 5, dtype=_t.float64)
    K.legendre_shell_accumulate(
        z, z.clone(), _t.zeros(4, dtype=_t.float64), _t.zeros(4, dtype=_t.float64),
        _t.zeros(4, 5, dtype=_t.float64), _t.zeros(4, 5, dtype=_t.float64),
        _t.zeros(4, dtype=_t.long), _t.zeros(5, 5, dtype=_t.float64),
        _t.zeros(5, 5, dtype=_t.float64), _t.zeros(5, dtype=_t.float64))
    print('  PROBLEM: float64 was accepted')
except Exception as e:
    msg = str(e)
    ok = ('float32 only' in msg) or ('dtype' in msg)
    print(('  refused: ' if ok else '  WRONG ERROR (not a dtype refusal): ')
          + f'{type(e).__name__}: {msg.splitlines()[0][:110]}')
    if not ok:
        raise SystemExit(1)
if not K.available():
    print('why:', K.why_unavailable())
    err = K.last_error()
    if err: print(err[1][:3000])
    raise SystemExit(1)

# --- correctness: fused vs portable on random shell-sorted input -------------
from torchref.experimental.alignment.sh import legendre_recurrence_coefficients
g = torch.Generator().manual_seed(4)
for L, n_c, n_sh, dt in ((13, 500, 40, torch.float32),
                         (65, 4000, 300, torch.float32),
                         (101, 3000, 250, torch.float32),
                         (65, 2000, 150, torch.float32)):
    n_even = (L - 1 if (L-1) % 2 == 0 else L - 2) // 2
    ct = (2*torch.rand(n_c, generator=g, dtype=dt)-1)
    st = (1-ct*ct).clamp(min=0).sqrt()
    Dr = torch.randn(n_c, L, generator=g, dtype=dt)
    Di = torch.randn(n_c, L, generator=g, dtype=dt)
    sh = torch.sort(torch.randint(0, n_sh, (n_c,), generator=g))[0]
    a, b, se = legendre_recurrence_coefficients(L, dt, torch.device('cpu'))
    ref_r = torch.zeros(n_even, n_sh, L, dtype=dt); ref_i = torch.zeros_like(ref_r)
    got_r = torch.zeros_like(ref_r); got_i = torch.zeros_like(ref_r)
    P.legendre_shell_accumulate(ref_r, ref_i, ct, st, Dr, Di, sh, a, b, se)
    K.legendre_shell_accumulate(got_r, got_i, ct, st, Dr, Di, sh, a, b, se)
    sc = max(ref_r.abs().max().item(), 1e-300)
    er = (got_r-ref_r).abs().max().item()/sc
    ei = (got_i-ref_i).abs().max().item()/max(ref_i.abs().max().item(),1e-300)
    print(f'  L={L:3d} n_c={n_c:5d} {str(dt):15s} rel err  re {er:.2e}  im {ei:.2e}')

# --- what single precision costs, against an ungrouped float64 reference ----
import torchref.experimental.alignment.frf.data_mr as dm
g2 = torch.Generator().manual_seed(77)
for L, hs in ((65, 64.0), (101, 100.0)):
    sv = torch.randn(6000, 3, generator=g2, dtype=torch.float64)
    sv = sv / sv.norm(dim=-1, keepdim=True) * (
        0.07 + 0.18*torch.rand(6000, 1, generator=g2, dtype=torch.float64))
    I = torch.randn(6000, generator=g2, dtype=torch.float64)
    ks, kc = dm._GROUP_SCALE_S, dm._GROUP_SCALE_COS
    dm._GROUP_SCALE_S = dm._GROUP_SCALE_COS = 10**16
    exact = dm.bessel_sh_expand(sv, I, L=L, bessel_h_scale=hs).coeffs
    dm._GROUP_SCALE_S, dm._GROUP_SCALE_COS = ks, kc
    sc = max(exact.abs().max().item(), 1e-300)
    f64 = dm.bessel_sh_expand(sv, I, L=L, bessel_h_scale=hs).coeffs
    f32 = dm.bessel_sh_expand(sv, I, L=L, bessel_h_scale=hs,
                              compute_dtype=torch.complex64).coeffs
    print(f'  L={L:3d} vs exact: float64 angular {((f64-exact).abs().max()/sc):.2e}'
          f'   float32 angular {((f32-exact).abs().max()/sc):.2e}')

# --- speed on the real thing -------------------------------------------------
from lab import FRFConfig, orbit_rank, rotated_case, run_frf, seed_for
cases = {p: rotated_case(p, seed_for(p, 0)) for p in ('1DAW', '3K7M')}
for cap in (64, 100):
    cfg = FRFConfig(lmax_cap=cap, n_peaks=200)
    for forced in (True, False):
        set_force_portable(forced)
        for m, d, _ in cases.values():
            run_frf(m, d, cfg, capture_arf=False)
    res, rank = {}, {}
    for rep in range(3):
        for arm, forced in (('portable', True), ('fused', False)):
            set_force_portable(forced)
            for p, (m, d, R) in cases.items():
                t0 = time.perf_counter()
                r = run_frf(m, d, cfg, capture_arf=False)
                res.setdefault((arm,p), []).append(time.perf_counter()-t0)
                k, _ = orbit_rank(r.peaks, R,
                    d.spacegroup.matrices.to(torch.float64).cpu(),
                    reciprocal_basis=d.cell.reciprocal_basis_matrix.to(torch.float64).cpu(),
                    side='left', frame='cart')
                rank[(arm,p)] = k
    set_force_portable(None)
    print(f'--- cap{cap} (best of 3, seconds) ---')
    for p in cases:
        e, c = min(res[('portable',p)]), min(res[('fused',p)])
        print(f'  {p:6s} portable {e:7.2f} [rank {rank[(\"portable\",p)]:>3}]   '
              f'fused {c:7.2f} [rank {rank[(\"fused\",p)]:>3}]   {e/max(c,1e-9):5.2f}x')
"
echo "exit_code=$?"
