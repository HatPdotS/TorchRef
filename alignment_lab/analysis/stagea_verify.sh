#!/bin/bash
# Stage A verification: every swap is claimed bit-identical, so check each one by
# computing BOTH forms in one process and comparing exactly. Stronger and far
# cheaper than diffing an end-to-end fingerprint against a baseline worktree.
#
# Also answers the two "verify, then delete" questions: do the calc-side
# resolution mask and the near-no-op obs mask drop any reflection at all?
#SBATCH --job-name=stagea
#SBATCH --output=alignment_lab/slurm/%x_%j.out
#SBATCH --error=alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
mkdir -p alignment_lab/slurm
echo "host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"

"$PY" -u - <<'PYEOF'
import math
import torch
torch.manual_seed(0)

ok = True
def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")

print("=== 1. Edmonds ZYZ: base primitive vs the deleted local copy ===")
from torchref.base.alignment.rotation import rotation_matrix_euler_zyz
a = torch.rand(5000, dtype=torch.float64) * 2 * math.pi
b = torch.rand(5000, dtype=torch.float64) * math.pi
g = torch.rand(5000, dtype=torch.float64) * 2 * math.pi

def old_zyz(alpha, beta, gamma):
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)
    return torch.stack([
        torch.stack([ca*cb*cg - sa*sg, -ca*cb*sg - sa*cg, ca*sb], dim=-1),
        torch.stack([sa*cb*cg + ca*sg, -sa*cb*sg + ca*cg, sa*sb], dim=-1),
        torch.stack([-sb*cg,            sb*sg,            cb   ], dim=-1),
    ], dim=-2)

R_old = old_zyz(a, b, g)
R_new = rotation_matrix_euler_zyz(torch.stack([a, b, g], dim=-1))
check("bitwise equal over 5000 random triples", torch.equal(R_old, R_new),
      f"max|d|={ (R_old-R_new).abs().max().item():.3e}")

print("=== 2. Rodrigues: rotation_utils vs the deleted align._rodrigues ===")
from torchref.experimental.alignment.frf.rotation_utils import axis_angle_to_matrix

def old_rodrigues(omega):
    if omega.dtype != torch.float64:
        omega = omega.to(torch.float64)
    single = omega.dim() == 1
    if single:
        omega = omega.unsqueeze(0)
    th = omega.norm(dim=-1, keepdim=True)
    axis = omega / th.clamp(min=1e-30)
    zeros = torch.zeros_like(axis[..., 0])
    K = torch.stack([
        torch.stack([zeros, -axis[..., 2], axis[..., 1]], dim=-1),
        torch.stack([axis[..., 2], zeros, -axis[..., 0]], dim=-1),
        torch.stack([-axis[..., 1], axis[..., 0], zeros], dim=-1),
    ], dim=-2)
    th_b = th.unsqueeze(-1)
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(*omega.shape[:-1], 3, 3)
    R = eye + torch.sin(th_b) * K + (1.0 - torch.cos(th_b)) * torch.matmul(K, K)
    return R.squeeze(0) if single else R

# The production caller builds omegas as a float64 meshgrid, so replicate that.
c = torch.linspace(-0.1, 0.1, 11, dtype=torch.float64)
wx, wy, wz = torch.meshgrid(c, c, c, indexing="ij")
om = torch.stack([wx.flatten(), wy.flatten(), wz.flatten()], dim=-1)
check("bitwise equal on the pipeline's float64 perturbation grid",
      torch.equal(old_rodrigues(om), axis_angle_to_matrix(om)))

print("=== 3. Symop unroll: apply_to_hkl vs the deleted einsum ===")
from torchref.symmetry.spacegroup import SpaceGroup
for sg_name in ("P 1", "C 2", "P 21 21 21", "P 31 2 1", "P 65 2 2", "P 43 32", "P 4 3 2"):
    sg = SpaceGroup(sg_name)
    hkl = torch.randint(-40, 41, (4000, 3))
    rec = torch.eye(3, dtype=torch.float64) * 0.0137 + 0.0011   # arbitrary non-diagonal basis
    old = torch.einsum(
        "kji,nj->kni", sg.matrices.to(torch.float64), hkl.to(torch.float64)
    ).reshape(-1, 3)
    new = sg.apply_to_hkl(hkl).permute(2, 0, 1).reshape(-1, 3).to(torch.float64)
    same_rows = torch.equal(old, new)
    same_s = torch.equal(old @ rec, new @ rec)
    check(f"{sg_name:12s} n_ops={sg.n_ops:2d} rows and s_obs bitwise equal",
          same_rows and same_s,
          "" if same_rows else f"row mismatch {int((old != new).any(-1).sum())}")

print("=== 4. (L, d_min) pass-through replaces the second auto_lmax call ===")
from torchref.experimental.alignment.frf.api import phaser_lmax_resolution
from torchref.experimental.alignment.rotation_search import LMAX_CAP
for r, dmin in ((10.0, 1.8), (15.0, 2.05), (25.0, 1.6), (4.0, 3.0)):
    L1, d1 = phaser_lmax_resolution(r, dmin, LMAX_CAP)
    L2, d2 = phaser_lmax_resolution(r, dmin, LMAX_CAP)   # the call that used to be inside
    check(f"radius {r:5.1f} d_min {dmin:.2f} -> L={L1} d_min_eff={d1:.4f}",
          (L1, d1) == (L2, d2))

print("=== 5. Do the two 'verify then delete' masks drop anything? ===")
from alignment_lab.lab.benchmark import load_case
from torchref.experimental.alignment.frf.dense_calc import dense_calc_via_box
from torchref.experimental.alignment.rotation_search import (
    LOW_RESOLUTION_CUTOFF_A, DENSE_CALC_PAD,
)
for pdb in ("3K7M", "1DAW"):
    model, data = load_case(pdb)[:2]
    rec_basis = data.cell.reciprocal_basis_matrix.to(torch.float64)
    s_mag_all = (data.hkl.to(torch.float64) @ rec_basis).norm(dim=-1)
    d_min_data = float(1.0 / s_mag_all.max().item())
    d_max = float(LOW_RESOLUTION_CUTOFF_A)

    # (a) the obs mask at rotation_search.py:239 -- upper bound is max <= max
    keep = (s_mag_all >= 1.0 / d_max) & (s_mag_all <= 1.0 / d_min_data)
    n_lo = int((s_mag_all < 1.0 / d_max).sum())
    n_hi = int((s_mag_all > 1.0 / d_min_data).sum())
    print(f"  {pdb}: obs mask keeps {int(keep.sum())}/{len(s_mag_all)} "
          f"(dropped {n_lo} below {d_max:.0f} A, {n_hi} above d_min via the "
          f"reciprocal round-trip)")

    # (b) the calc-side mask in score_model, against dense_calc's own window
    model_radius_A = float((model.xyz() - model.xyz().mean(0)).norm(dim=-1).mean().item())
    L, d_min_eff = phaser_lmax_resolution(model_radius_A, d_min_data, LMAX_CAP)
    s_calc, F_calc = dense_calc_via_box(model, d_max, d_min_eff, pad=DENSE_CALC_PAD)
    smag_calc = s_calc.norm(dim=-1)
    keep_c = (smag_calc >= 1.0 / d_max) & (smag_calc <= 1.0 / d_min_eff)
    dropped = int(len(smag_calc) - keep_c.sum())
    print(f"  {pdb}: calc re-mask drops {dropped}/{len(smag_calc)} "
          f"(L={L}, d_min_eff={d_min_eff:.3f} A)")

print()
print("OVERALL:", "PASS" if ok else "FAIL")
PYEOF
rc=$?
echo "python_exit=$rc"
