"""Do TorchRef's structure factors see the AlphaFold model and the deposited
chain as the same molecule? In-place R against the data for both, and the
correlation of their dense-box transforms after superposition."""
import sys
sys.path.insert(0, "/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab")
import torch, numpy as np, gemmi
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT
from torchref.experimental.alignment.frf.dense_calc import dense_calc_via_box
code = sys.argv[1]; D = "/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper"
data = ReflectionData().load_mtz(f"{D}/data/{code}/{code}.mtz")
dep = ModelFT().load_pdb(f"{D}/data/{code}/{code}.pdb").select("chain A")
af = ModelFT().load_pdb(f"{D}/figure2_alphafold_start/placed/{code}_af.pdb").select("chain A")
hkl = data.hkl
rec = data.cell.reciprocal_basis_matrix.to(torch.float64)
d_all = 1.0 / (hkl.to(torch.float64) @ rec).norm(dim=-1)
def in_place_r(m, tag):
    m.spacegroup = data.spacegroup.hm; m.cell = data.cell.clone()
    with torch.no_grad(): F = m(hkl).abs()
    fo = data.F.abs() if torch.is_complex(data.F) else data.F
    fo = fo.to(F.dtype); d = d_all.to(F.device)
    sel = torch.isfinite(fo) & torch.isfinite(F) & (d >= 4.0) & (d <= 15.0)
    k = (fo[sel] * F[sel]).sum() / (F[sel] ** 2).sum()
    r = (fo[sel] - k * F[sel]).abs().sum() / fo[sel].sum()
    print(f"INPLACE {tag} n={int(sel.sum())} R(15-4A)={float(r):.3f} n_atoms={m.xyz().shape[0]} B_mean={float(m.adp().mean()) if hasattr(m,'adp') else float('nan'):.1f}", flush=True)
in_place_r(dep, "deposited_chainA"); in_place_r(af, "phaser_af")
# superpose the AF model onto the deposited (CA Kabsch) and compare dense-box transforms
def ca(m):
    df = m.pdb; sel = df["name"].str.strip() == "CA"
    return torch.as_tensor(df.loc[sel, ["x", "y", "z"]].to_numpy(), dtype=torch.float64), df.loc[sel, "resseq"].to_numpy()
A, ra = ca(dep); B, rb = ca(af)
common = np.intersect1d(ra, rb); ia = [list(ra).index(r) for r in common]; ib = [list(rb).index(r) for r in common]
A, B = A[ia], B[ib]; ca_, cb_ = A.mean(0), B.mean(0)
U, S, Vt = torch.linalg.svd((B - cb_).T @ (A - ca_)); d = torch.sign(torch.linalg.det(Vt.T @ U.T))
R = Vt.T @ torch.diag(torch.tensor([1, 1, float(d)], dtype=torch.float64)) @ U.T
print(f"superpose rmsd={float((((B - cb_) @ R.T - (A - ca_)) ** 2).sum(1).mean().sqrt()):.2f}", flush=True)
af2 = af.copy(); af2.spacegroup = "P 1"; af2.rotate(R.to(af2.dtype_float), center=cb_.to(af2.dtype_float)); af2.translate((ca_ - cb_).to(af2.dtype_float))
# (a) the superposed AF model in the crystal, at the deposited position
af_cr = af2.copy(); in_place_r(af_cr, "af_superposed_on_deposited")
with torch.no_grad():
    dep_cr = dep.copy(); dep_cr.spacegroup = data.spacegroup.hm; dep_cr.cell = data.cell.clone()
    Fd = dep_cr(hkl).abs(); Fa = af_cr(hkl).abs()
sel = (d_all.to(Fd.device) >= 4.0) & (d_all.to(Fd.device) <= 15.0)
print(f"CRYSTAL corr(|F| dep vs af_superposed, 15-4A)={float(torch.corrcoef(torch.stack([Fd[sel], Fa[sel]]))[0,1]):.4f}", flush=True)
# (b) all-atom rmsd on common atoms
da = {(int(r), n.strip()): i for i, (r, n) in enumerate(zip(dep.pdb["resseq"], dep.pdb["name"]))}
db = {(int(r), n.strip()): i for i, (r, n) in enumerate(zip(af2.pdb["resseq"], af2.pdb["name"]))}
keys = [k for k in da if k in db and k[1][0] != "H"]
xa = dep.xyz().detach()[[da[k] for k in keys]].to(torch.float64); xb = af2.xyz().detach()[[db[k] for k in keys]].to(torch.float64)
print(f"ALLATOM common={len(keys)} of dep={len(da)} af={len(db)} rmsd={float(((xa - xb) ** 2).sum(1).mean().sqrt()):.2f} A", flush=True)
# (c) dense transforms in one shared cubic P1 box
from torchref.symmetry.cell import Cell
ext = max(float((m.xyz().detach() - m.xyz().detach().mean(0)).norm(dim=-1).max()) for m in (dep, af2))
a_box = 2.0 * 2.0 * ext
n = int(a_box / 4.0) + 1
g = torch.arange(-n, n + 1)
hb = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3).to(torch.float64)
inv_d = hb.norm(dim=-1) / a_box; keep = (inv_d >= 1 / 15.0) & (inv_d <= 1 / 4.0); hb = hb[keep]
out = {}
for tag, m in (("dep", dep), ("af", af2)):
    mb = m.copy(); mb.spacegroup = "P 1"; mb.cell = Cell([a_box, a_box, a_box, 90.0, 90.0, 90.0])
    with torch.no_grad(): out[tag] = mb(hb.to(torch.int64)).abs().cpu()
x, y = out["dep"], out["af"]
print(f"DENSE box={a_box:.0f}A n={hb.shape[0]} corr(|F|)={float(torch.corrcoef(torch.stack([x, y]))[0,1]):.4f} corr(|F|^2)={float(torch.corrcoef(torch.stack([x**2, y**2]))[0,1]):.4f} mean|F| dep={float(x.mean()):.1f} af={float(y.mean()):.1f}", flush=True)
