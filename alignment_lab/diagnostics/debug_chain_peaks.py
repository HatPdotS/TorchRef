"""For one chain cell: the true-orientation candidate's translation peaks, each
placed and measured against the deposited chain, and the likelihood at the true
translation of that orientation (t = 0, since the template is re-oriented about
the search model's centroid, which the benchmark rotation preserved)."""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import load_case, pose_error, random_rotation, seed_for  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain_pose_recovery import chain_selection  # noqa: E402
from torchref.experimental.alignment import MolecularReplacementPipeline, MRSolution  # noqa: E402
from torchref.experimental.alignment.translation import (  # noqa: E402
    fast_translation_function, llg_at_translations, prepare_candidate)

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", required=True); ap.add_argument("--chain", required=True)
ap.add_argument("--trial", type=int, default=0)
args = ap.parse_args()
model, data = load_case(args.pdb)
ch = model.select(chain_selection(model, args.chain))
canonical = ch.xyz().clone()
R_true = random_rotation(seed_for(args.pdb, args.trial) + 7919 * (ord(args.chain[0]) % 26))
search = ch.copy(); search.spacegroup = "P 1"
search = search.copy().rotate(R_true.to(model.dtype_float), center=canonical.mean(0))
pipe = MolecularReplacementPipeline(data, search, d_min=4.0, d_max=15.0, n_shells=20,
                                    n_rotation_peaks=200, n_rotation_candidates=10)
sols = pipe.run()
win = sols[0]
print(f"WINNER k={win.candidate_index} llg={win.llg_score:.1f} pose={tuple(round(v,2) for v in pose_error(win.model.xyz(), canonical, data.cell, data.spacegroup))}")
# The candidate at the true orientation (smallest rotation error), its peaks.
best_k, best_d = None, 1e9
for s in sols:
    m = pipe.place(s)
    d, _ = pose_error(m.xyz(), canonical, data.cell, data.spacegroup)
    if d < best_d:
        best_d, best_k, best_sol = d, s.candidate_index, s
print(f"true-orientation candidate k={best_k} rot_err={best_d:.2f}")
R_rec = torch.as_tensor(best_sol.rotation, dtype=torch.float64)
pipe._orient_template(R_rec)
obs = pipe._obs
cand = prepare_candidate(pipe._p1, obs, data.spacegroup, data.cell)
d_min_set = 1.0 / float(obs.s_mag.max())
_, peaks = fast_translation_function(obs, cand, data.spacegroup, data.cell,
                                     grid_spacing_A=d_min_set / 3.0, n_peaks=8, cluster_radius_A=d_min_set)
t_cands = torch.as_tensor(np.stack([p.translation for p in peaks]), dtype=torch.float64)
llg = llg_at_translations(obs, cand, t_cands)
for i, p in enumerate(peaks):
    sol = MRSolution(rotation=best_sol.rotation, translation=p.translation, rotation_score=0, translation_score=p.score, r_factor=0)
    m = pipe.place(sol)
    print(f"PEAK {i} t={np.round(p.translation,3).tolist()} tf={p.score:.1f} z={p.sigma:.2f} llg={float(llg[i]):.1f} "
          f"pose={tuple(round(v,2) for v in pose_error(m.xyz(), canonical, data.cell, data.spacegroup))}")
t0 = torch.zeros(1, 3, dtype=torch.float64)
print(f"TRUE-T llg={float(llg_at_translations(obs, cand, t0)[0]):.1f} at t=0 for this orientation")
