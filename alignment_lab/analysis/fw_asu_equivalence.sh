#!/bin/bash
# Is ASU-then-broadcast equivalent to the unrolled computation? The previous run
# reported max|d eEobs| = 0.22 with "54504 differing", but counted ANY nonzero
# float difference, so that count says nothing about size. Characterise the
# distribution, and find where the outlier sits.
#SBATCH --job-name=frf_fwasu
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
echo "host=$(hostname)"
"$PY" -u -c "
import numpy as np, torch
torch.set_grad_enabled(False)
from alignment_lab.lab.benchmark import load_case
from torchref.experimental.alignment.frf.french_wilson import french_wilson_preprocess

for name in ('3K7M', '1DAW'):
    model, data = load_case(name); model.verbose = 0
    rb = data.cell.reciprocal_basis_matrix.to(torch.float64)
    s_all = (data.hkl.to(torch.float64) @ rb).norm(dim=-1)
    d_min = 1.0 / s_all.max().item()
    keep = (s_all >= 1.0/100.0) & (s_all <= 1.0/d_min)
    n_ops = int(data.spacegroup.matrices.shape[0])
    F = data.F.abs().to(torch.float64)[keep]; sig = data.F_sigma.to(torch.float64)[keep]
    s = s_all[keep]; cen = torch.zeros_like(F, dtype=torch.bool)
    M = F.numel()

    unrolled = french_wilson_preprocess(F.repeat(n_ops), sig.repeat(n_ops),
                                        s.repeat(n_ops), cen.repeat(n_ops), n_wilson_shells=20)
    unique = french_wilson_preprocess(F, sig, s, cen, n_wilson_shells=20)
    a = unrolled['eEobs'].reshape(n_ops, M)[0].numpy()
    b = unique['eEobs'].numpy()
    d = np.abs(a - b)
    rel = d / np.maximum(np.abs(a), 1e-12)
    print(f'--- {name}: {M} unique x {n_ops} ops')
    for q in (50, 90, 99, 99.9, 100):
        print(f'      |d eEobs| p{q:<5} {np.percentile(d, q):.3e}   rel {np.percentile(rel, q):.3e}')
    print(f'      above 1e-9 : {int((d > 1e-9).sum())} of {M}')
    print(f'      above 1e-3 : {int((d > 1e-3).sum())} of {M}')
    # Do the shell edges actually differ, and by how many reflections?
    def edges_and_shells(sv):
        sn = sv.numpy(); idx = np.argsort(sn)
        e = np.linspace(0, len(sn)-1, 21).round().astype(np.int64)
        ed = sn[idx][e].copy(); ed[0] -= 1e-6; ed[-1] += 1e-6
        return ed, np.clip(np.searchsorted(ed, sn, side='right')-1, 0, 19)
    eu, shu = edges_and_shells(s)
    er, shr = edges_and_shells(s.repeat(n_ops))
    print(f'      max |d shell edge|        {np.abs(eu - er).max():.3e}')
    print(f'      reflections changing shell {int((shu != shr[:M]).sum())} of {M}')
" 2>&1 | grep -vE "$FILT"
echo "done"
