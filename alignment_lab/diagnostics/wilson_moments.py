"""Second moment of E^2 per structure, through the shared Wilson fit.

``<(E^2-1)^2>`` is the standard indicator for translational NCS and related
intensity modulations: 1.0 for ideal acentric Wilson data, larger when whole
classes of reflections reinforce or cancel together. 2DQ6 was recorded at 5.528
against 1.0-1.2 for every other benchmark structure, and that number is the sole
evidence for calling it a tNCS case.

It is worth recomputing, because tNCS needs at least two copies related by a
pure translation and 2DQ6 deposits ONE chain in the asymmetric unit, and because
the number was measured when the package had five disagreeing answers to what E
means. A second moment is a property of the normalisation as much as of the
data: normalise by a curve that is too flat and the resolution trend leaks
straight into the moment.
"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)
from lab import BENCH_PDBS, load_case  # noqa: E402
from torchref.scaling import WilsonNormaliser  # noqa: E402

print(f"{'pdb':6s} {'sg':12s} {'N':>7s} {'<(E2-1)^2>':>11s} {'<E2>':>7s} "
      f"{'<|E|>':>7s} {'shell-norm':>11s}")
for pdb in BENCH_PDBS:
    model, data = load_case(pdb)
    mask = data.get_valid_mask()
    F = data.F[mask].abs().to(torch.float64)
    hkl = data.hkl[mask]
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64).to(hkl.device)
    s = (hkl.to(torch.float64) @ rec).norm(dim=-1)
    hkl_l = hkl.round().to(torch.int64)
    eps = data.spacegroup.epsilon(hkl_l, friedel=False).to(torch.float64).clamp(min=1.0)
    cen = data.spacegroup.is_centric(hkl_l).to(torch.bool)
    acen = ~cen

    w = WilsonNormaliser(F * F, s, eps=eps, centric=cen, n_coeff=6)
    E2 = w.E_squared.to(torch.float64)[acen]
    m2 = float(((E2 - 1.0) ** 2).mean())

    # The same moment under a 20-shell mean, which is what the older estimate
    # would have used -- to separate "the data are odd" from "the curve was".
    order = torch.argsort(s)
    sh = torch.zeros_like(s, dtype=torch.long)
    chunk = s.numel() // 20
    for k in range(20):
        a = k * chunk
        b = (k + 1) * chunk if k < 19 else s.numel()
        sh[order[a:b]] = k
    I = F * F / eps
    tot = torch.zeros(20, dtype=torch.float64).scatter_add_(0, sh, I)
    cnt = torch.bincount(sh, minlength=20).to(torch.float64).clamp(min=1)
    E2s = I / (tot / cnt).clamp(min=1e-30).index_select(0, sh)
    m2s = float(((E2s[acen] - 1.0) ** 2).mean())

    print(f"{pdb:6s} {str(data.spacegroup.hm):12s} {int(acen.sum()):7d} "
          f"{m2:11.3f} {float(E2.mean()):7.3f} "
          f"{float(E2.clamp(min=0).sqrt().mean()):7.3f} {m2s:11.3f}", flush=True)
