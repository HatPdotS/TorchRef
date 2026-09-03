"""Place a model given explicit MTZ/PDB paths: rotate it at random, run the
pipeline, report the top solution's pose error against the starting
coordinates (mates and origin shifts allowed) and the likelihood gain."""
import sys, time, argparse
sys.path.insert(0, "/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab")
import torch
from lab import pose_error, random_rotation
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT
from torchref.experimental.alignment import MolecularReplacementPipeline
ap = argparse.ArgumentParser(); ap.add_argument("mtz"); ap.add_argument("pdb"); ap.add_argument("--chain", default=None)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-cand", type=int, default=10); ap.add_argument("--tag", default="")
ap.add_argument("--select", action="store_true", help="pass the loaded model through select('all') first")
ap.add_argument("--set-b", type=float, default=None, help="overwrite every B with this value")
ap.add_argument("--d-min", type=float, default=4.0)
ap.add_argument("--tf-d-min", type=float, default=None, help="translation window high-res limit; default = --d-min")
ap.add_argument("--noise", type=float, default=0.0, help="rms Gaussian coordinate noise in A, seeded")
ap.add_argument("--shift", default=None, help="'center' puts the centroid at the origin; or 'x,y,z' in A")
a = ap.parse_args()
data = ReflectionData().load_mtz(a.mtz)
m = ModelFT().load_pdb(a.pdb)
if a.chain: m = m.select(f"chain {a.chain}")
if a.select: m = m.select("all")
if a.set_b is not None:
    m.pdb["tempfactor"] = a.set_b
    if hasattr(m, "b_iso"): m.b_iso.data[:] = a.set_b
m.pdb["chainid"] = "A"; m.spacegroup = "P 1"; m.cell = data.cell.clone()
if a.shift == "center":
    m.translate(-m.xyz().detach().mean(0))
elif a.shift:
    m.translate(torch.tensor([float(v) for v in a.shift.split(",")], dtype=m.dtype_float))
if a.noise > 0:
    g = torch.Generator().manual_seed(1234 + a.seed)
    m.translate(torch.zeros(3, dtype=m.dtype_float))  # no-op, keeps the write path warm
    with torch.no_grad():
        m.xyz_tensor.data.add_(torch.randn(m.xyz().shape, generator=g).to(m.dtype_float) * a.noise / 3 ** 0.5) if hasattr(m, "xyz_tensor") else None
canon = m.xyz().detach().clone()
print(f"centroid {canon.mean(0).tolist()}", flush=True)
R = random_rotation(a.seed).to(m.dtype_float)
search = m.copy().rotate(R, center=canon.mean(0))
t0 = time.time()
pipe = MolecularReplacementPipeline(data, search, d_min=a.d_min, d_max=15.0, n_shells=20, n_rotation_peaks=200, n_rotation_candidates=a.n_cand, tf_d_min=a.tf_d_min)
sols = pipe.run(do_translation=True)
placed = pipe.place(sols[0])
r, t = pose_error(placed.xyz().detach(), canon, data.cell, data.spacegroup, allow_origin_freedom=True)
# rank of the first candidate whose placement is right
first = -1
for s in sorted(sols, key=lambda s: s.candidate_index):
    rr, tt = pose_error(pipe.place(s).xyz().detach(), canon, data.cell, data.spacegroup, allow_origin_freedom=True)
    if rr < 5 and tt < 2: first = s.candidate_index; break
import gemmi
_st = gemmi.Structure(); _st.cell = gemmi.UnitCell(*[float(x) for x in data.cell.data.tolist()]); _st.spacegroup_hm = data.spacegroup.hm; _st.setup_cell_images(); gc = _st.cell
P = placed.xyz().detach().cpu().numpy(); C = canon.cpu().numpy()
import numpy as np
dn = np.array([gc.find_nearest_image(gemmi.Position(*P[i]), gemmi.Position(*C[i]), gemmi.Asu.Any).dist() for i in range(0, P.shape[0], 5)])
print(f"  nearest-image per-atom rmsd={np.sqrt((dn**2).mean()):.2f} A  r_factor={sols[0].r_factor:.3f}", flush=True)
if np.sqrt((dn**2).mean()) > 2.0:
    Bm = data.cell.inv_fractional_matrix.detach().cpu().numpy() if hasattr(data.cell, "inv_fractional_matrix") else None
    frac2cart = np.array(gc.orth.mat.tolist()) if hasattr(gc, "orth") else None
    for sh in [(0,0,.5),(.5,0,0),(0,.5,0),(.5,.5,0),(.5,0,.5),(0,.5,.5),(.5,.5,.5),(1/3,2/3,0),(2/3,1/3,0),(1/3,2/3,.5),(2/3,1/3,.5)]:
        dv = gc.orthogonalize(gemmi.Fractional(*sh))
        dd = np.array([gc.find_nearest_image(gemmi.Position(P[i][0]+dv.x, P[i][1]+dv.y, P[i][2]+dv.z), gemmi.Position(*C[i]), gemmi.Asu.Any).dist() for i in range(0, P.shape[0], 5)])
        print(f"    origin shift {sh}: rmsd={np.sqrt((dd**2).mean()):.2f}", flush=True)
print(f"PLACE {a.tag} sg='{data.spacegroup.hm}' n_atoms={canon.shape[0]} seed={a.seed} rot={r:.1f} trans={t:.2f} llg={sols[0].llg_score:.0f} llg1={sols[1].llg_score:.0f} first_true_k={first} n_peaks={len(pipe.rotation_candidates)} s={time.time()-t0:.1f}", flush=True)
