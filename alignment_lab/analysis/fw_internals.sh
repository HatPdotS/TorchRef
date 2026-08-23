#!/bin/bash
# What inside french_wilson_preprocess costs, at the unrolled reflection count?
# The binning, the parabolic-cylinder posterior, or the Halley D-factor solve?
# Also: are the n_ops symmetry copies really carrying identical values?
#SBATCH --job-name=frf_fw
#SBATCH --output=alignment_lab/slurm/%x_%j.out
#SBATCH --error=alignment_lab/slurm/%x_%j.err
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
mkdir -p alignment_lab/slurm
FILT="Loaded|LINK|Wilson outlier|found non|FrenchWilson|Reflections:|Resolution:|Space group|Centric:|✓|Parametrization|Warning|warn|  from |^ *$|No CUDA"
echo "host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
"$PY" -u -c "
import time
import numpy as np, torch
torch.set_grad_enabled(False)
from alignment_lab.lab.benchmark import load_case
from torchref.experimental.alignment.frf.french_wilson import (
    french_wilson_preprocess, _french_wilson_posterior, _get_dfactor_vectorised)

def best(fn, n=3):
    fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); out.append(time.perf_counter()-t0)
    return min(out)

for name in ('3K7M', '1DAW'):
    model, data = load_case(name); model.verbose = 0
    rb = data.cell.reciprocal_basis_matrix.to(torch.float64)
    s_all = (data.hkl.to(torch.float64) @ rb).norm(dim=-1)
    d_min = 1.0 / s_all.max().item()
    keep = (s_all >= 1.0/100.0) & (s_all <= 1.0/d_min)
    n_ops = int(data.spacegroup.matrices.shape[0])
    F = data.F.abs().to(torch.float64)[keep]
    sig = data.F_sigma.to(torch.float64)[keep]
    s = s_all[keep]
    from torchref.experimental.alignment.frf.preprocessing import compute_epsilon
    cen = torch.zeros_like(F, dtype=torch.bool)
    try:
        cen = data.centric.to(torch.bool)[keep]
    except Exception:
        pass
    nu = F.numel()

    # the unrolled arrays, exactly as rotation_search builds them
    Fu = F.repeat(n_ops); sigu = sig.repeat(n_ops)
    su = s.repeat(n_ops); cenu = cen.repeat(n_ops)

    t_unrolled = best(lambda: french_wilson_preprocess(Fu, sigu, su, cenu, n_wilson_shells=20))
    t_unique   = best(lambda: french_wilson_preprocess(F, sig, s, cen, n_wilson_shells=20))
    print(f'--- {name}: {nu} unique x {n_ops} ops = {nu*n_ops} unrolled')
    print(f'      whole function, unrolled   {t_unrolled*1e3:8.1f} ms')
    print(f'      whole function, unique     {t_unique*1e3:8.1f} ms   ({t_unrolled/t_unique:.1f}x)')

    # Are the n_ops copies identical? If so the unrolled call is pure repetition.
    fu = french_wilson_preprocess(Fu, sigu, su, cenu, n_wilson_shells=20)
    fq = french_wilson_preprocess(F, sig, s, cen, n_wilson_shells=20)
    blocks = fu['eEobs'].reshape(n_ops, nu)
    same_across_ops = bool(torch.equal(blocks, blocks[0:1].expand(n_ops, -1)))
    matches_unique = bool(torch.equal(blocks[0], fq['eEobs']))
    print(f'      n_ops copies identical     {same_across_ops}')
    print(f'      block 0 == unique-set run  {matches_unique}')
    if not matches_unique:
        d = (blocks[0] - fq['eEobs']).abs()
        print(f'        max|d eEobs| {d.max().item():.3e}  n_differing={int((d>0).sum())}')

    # Stage split, on the unrolled arrays.
    F_np = Fu.numpy(); sig_np = sigu.numpy(); s_np = su.numpy(); cen_np = cenu.numpy()
    def binning():
        idx = np.argsort(s_np)
        e = np.linspace(0, len(s_np)-1, 21).round().astype(np.int64)
        edges = s_np[idx][e]; edges[0] -= 1e-6; edges[-1] += 1e-6
        sh = np.clip(np.searchsorted(edges, s_np, side='right')-1, 0, 19)
        m = np.zeros(20); c = np.zeros(20, dtype=np.int64)
        np.add.at(m, sh, F_np*F_np); np.add.at(c, sh, 1)
        return m/np.maximum(c,1), sh
    t_bin = best(binning)
    mF2, sh = binning()
    eosq = F_np*F_np/mF2[sh]; sigesq = np.maximum(2.0*F_np*sig_np/mF2[sh], 0.0)
    t_post = best(lambda: _french_wilson_posterior(eosq, sigesq, cen_np))
    ee, eesq = _french_wilson_posterior(eosq, sigesq, cen_np)
    bad = eesq < ee*ee; eesq[bad] = ee[bad]**2 + 1e-12
    t_dfac = best(lambda: _get_dfactor_vectorised(ee, eesq, cen_np))
    print(f'      of which: shells + <F2>    {t_bin*1e3:8.1f} ms')
    print(f'                FW posterior     {t_post*1e3:8.1f} ms')
    print(f'                DFAC Halley      {t_dfac*1e3:8.1f} ms')
" 2>&1 | grep -vE "$FILT"
echo "done"
