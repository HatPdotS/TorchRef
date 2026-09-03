"""Why does an AlphaFold search model fail to place? Rank of Phaser's
orientation among our rotation peaks, and an oracle arm using Phaser's own
placed copy as the search model."""
import sys, time, json
sys.path.insert(0, "/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab")
sys.path.insert(0, "/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/diagnostics")
import torch, gemmi
from lab import cartesian_symops
from af_placement import ca_xyz_by_chain, REVIEW, DEV
from chain_pose_recovery import first_true_index
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT
from torchref.experimental.alignment import MolecularReplacementPipeline
from torchref.experimental.alignment.frf.rotation_utils import (
    rotation_matrix_from_edmonds_euler, rotation_angular_distance_deg)

code, n_cand = sys.argv[1], int(sys.argv[2])
arms = sys.argv[3].split(",") if len(sys.argv) > 3 else ["af_search", "phaser_copy0"]
from lab import pose_error
comp = json.load(open(REVIEW / "search_models" / code / "components.json"))[0]
data = ReflectionData().load_mtz(str(DEV / "data" / code / f"{code}.mtz"))
phaser_pdb = DEV / "figure2_alphafold_start" / "placed" / f"{code}_af.pdb"
search = ModelFT().load_pdb(comp["processed_pdb"]); search.spacegroup = "P 1"; search.pdb["chainid"] = "A"

# copy 0 of Phaser's solution = its first chains whose CA count adds to the search model's
s_ca = torch.tensor([[a.pos.x, a.pos.y, a.pos.z] for r in gemmi.read_structure(comp["processed_pdb"])[0][0] for a in r if a.name == "CA"], dtype=torch.float64)
theirs = ca_xyz_by_chain(phaser_pdb); acc, j = 0, 0
while acc < s_ca.shape[0]: acc += theirs[j][1].shape[0]; j += 1
p_ca = torch.cat([t[1] for t in theirs[:j]]).to(torch.float64); names = [t[0] for t in theirs[:j]]
assert p_ca.shape == s_ca.shape, (p_ca.shape, s_ca.shape)
# Kabsch: p = R_app (s - cs) + cp
A = s_ca - s_ca.mean(0); B = p_ca - p_ca.mean(0)
U, S, Vt = torch.linalg.svd(A.T @ B); d = torch.sign(torch.linalg.det(Vt.T @ U.T))
R_app = Vt.T @ torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=torch.float64)) @ U.T
rms = ((A @ R_app.T - B) ** 2).sum(1).mean().sqrt()
print(f"TRUTH {code} phaser_copy0_chains={names} kabsch_rms={rms:.2f} A", flush=True)
R_rec_true = R_app.T
sym = cartesian_symops(data.spacegroup, data.cell)

def report(tag, model, n_cand):
    t0 = time.time()
    pipe = MolecularReplacementPipeline(data, model, d_min=4.0, d_max=15.0, n_shells=20, n_rotation_peaks=200, n_rotation_candidates=n_cand)
    sols = pipe.run(do_translation=True)
    dists = []
    for pk in pipe.rotation_candidates:
        R = rotation_matrix_from_edmonds_euler(pk.alpha, pk.beta, pk.gamma).to(torch.float64)
        dists.append(min(rotation_angular_distance_deg(R.T @ R_rec_true, sym[k]) for k in range(sym.shape[0])))
    best_k = min(range(len(dists)), key=lambda k: dists[k])
    k_first = first_true_index(sols, R_rec_true, sym)
    top = sorted(sols, key=lambda s: -s.llg_score)
    print(f"ARM {tag} n_peaks={len(dists)} truth_best_peak={best_k} ({dists[best_k]:.1f} deg) first_true_k={k_first} "
          f"top_llg={top[0].llg_score:.0f} (cand {top[0].candidate_index}, {dists[top[0].candidate_index]:.1f} deg from truth) "
          f"llg1={top[1].llg_score:.0f} s={time.time()-t0:.1f}", flush=True)
    print("  peak->truth deg, first 12:", " ".join(f"{d:.0f}" for d in dists[:12]), flush=True)
    # coordinate check of the top solution against Phaser's copy 0, mates and origin shifts allowed
    placed = pipe.place(top[0])
    tmp_p = f"alignment_lab/runs/af_mr/{code}/debug_top_{tag}.pdb"
    placed.write_pdb(tmp_p)
    ca = torch.cat([t[1] for t in ca_xyz_by_chain(tmp_p)]).to(torch.float64)[: p_ca.shape[0]]
    print(f"  ca shapes ours={tuple(ca.shape)} phaser={tuple(p_ca.shape)}", flush=True)
    if ca.shape == p_ca.shape:
        r, t = pose_error(ca, p_ca, data.cell, data.spacegroup, allow_origin_freedom=True)
        print(f"  top solution vs phaser copy0 by coordinates: rot={r:.1f} deg trans={t:.2f} A", flush=True)
    # alternate operator convention
    B = data.cell.inv_fractional_matrix.detach().cpu().to(torch.float64) if hasattr(data.cell, "inv_fractional_matrix") else None
    print(f"  n_sym={sym.shape[0]} orth_err={max(float((S @ S.T - torch.eye(3, dtype=S.dtype)).abs().max()) for S in sym):.2e}", flush=True)
    return pipe, sols

if "af_search" in arms: report("af_search", search, n_cand)
# oracle: Phaser's placed copy 0 as the search model (already in the crystal cell)
st = gemmi.read_structure(str(phaser_pdb)); st.setup_entities()
for ch in list(st[0]):
    if ch.name not in names: st[0].remove_chain(ch.name)
import os; os.makedirs(f"alignment_lab/runs/af_mr/{code}", exist_ok=True)
tmp = f"alignment_lab/runs/af_mr/{code}/phaser_copy0.pdb"
st.write_pdb(tmp)
oracle = ModelFT().load_pdb(tmp); oracle.spacegroup = "P 1"; oracle.pdb["chainid"] = "A"
if "phaser_copy0" in arms: report("phaser_copy0", oracle, n_cand)
