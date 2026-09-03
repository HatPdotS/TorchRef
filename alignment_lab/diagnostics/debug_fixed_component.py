"""Why does a chain placed with another fixed land in the wrong place?

Three checks on one cell (chain X with the other protein chains fixed at their
deposited positions):

1. Convention: the P1 template's symmetry sum at t = 0 must equal the crystal-
   group ModelFT's structure factor of the same chain. If the two disagree in
   phase, the fixed and moving parts add incoherently and the likelihood is
   scoring nonsense.
2. The fixed component's fitted D_f and the size of c_lin against c_quad.
3. For the true-orientation candidate: every translation peak with its fast
   score, likelihood and clash fraction, plus the likelihood and clash at the
   true translation (t = 0 for the template, which the benchmark rotation about
   the centroid preserves).
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import load_case, pose_error, random_rotation, seed_for  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain_pose_recovery import chain_selection, protein_chains  # noqa: E402
from torchref.experimental.alignment import MolecularReplacementPipeline, MRSolution  # noqa: E402
from torchref.experimental.alignment.translation import (  # noqa: E402
    fast_translation_function, llg_at_translations, prepare_candidate)

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", required=True); ap.add_argument("--chain", required=True)
ap.add_argument("--trial", type=int, default=0)
args = ap.parse_args()
model, data = load_case(args.pdb)
ch = model.select(chain_selection(model, args.chain))
others = [c for c, _ in protein_chains(model) if c != args.chain]
fixed = [model.select(chain_selection(model, c)) for c in others]
canonical = ch.xyz().clone()
seed = seed_for(args.pdb, args.trial) + 7919 * (ord(args.chain[0]) % 26)
R_true = random_rotation(seed)
search = ch.copy(); search.spacegroup = "P 1"
search = search.copy().rotate(R_true.to(model.dtype_float), center=canonical.mean(0))
pipe = MolecularReplacementPipeline(data, search, d_min=4.0, d_max=15.0, n_shells=20,
                                    n_rotation_peaks=200, n_rotation_candidates=10, fixed=fixed)
sols = pipe.run()
obs, fx = pipe._obs, pipe._fixed
print(f"FIXED D_f median={float(fx.D_f.median()):.3f} min={float(fx.D_f.min()):.3f} max={float(fx.D_f.max()):.3f} "
      f"|c_lin| median={float(fx.c_lin.abs().median()):.3g} |c_quad| median={float(fx.c_quad.abs().median()):.3g} "
      f"m median={float(fx.m.median()):.3f}")

# 1. Convention check: deposited chain X through the P1 template at t=0 vs the crystal-group forward.
dep = ch.copy()
if pipe.tf_d_min > 0.0:
    dep.max_res = pipe.tf_d_min / 1.5
dep.spacegroup = "P 1"
cand_dep = prepare_candidate(dep, obs, data.spacegroup, data.cell)
F_sum = cand_dep.f_calc(torch.zeros(3, dtype=torch.float64)) * cand_dep.norm.to(cand_dep.G.dtype)
with torch.no_grad():
    F_cryst = ch(obs.hkl.to(ch.xyz().device)).to(F_sum.device).to(F_sum.dtype)
coh = (F_sum.conj() * F_cryst).sum().abs() / (F_sum.abs().norm() * F_cryst.abs().norm())
amp = (F_sum.abs() * F_cryst.abs()).sum() / (F_sum.abs().norm() * F_cryst.abs().norm())
print(f"CONVENTION complex coherence={float(coh):.4f} amplitude correlation={float(amp):.4f} "
      f"(1.0 = same phases; amplitude-only agreement with low coherence = phase convention mismatch)")
# Also the fixed part: sum of fixed models' F vs F_fixed in the pipeline.
print(f"F_fixed |.| median={float(pipe._F_fixed.abs().median()):.3g}, F_X |.| median={float(F_cryst.abs().median()):.3g}")

# 3. True-orientation candidate peaks.
best = min(sols, key=lambda s: pose_error(pipe.place(s).xyz(), canonical, data.cell, data.spacegroup)[0])
pipe._orient_template(torch.as_tensor(best.rotation, dtype=torch.float64))
cand = prepare_candidate(pipe._p1, obs, data.spacegroup, data.cell)
d_min_set = 1.0 / float(obs.s_mag.max())
_, peaks = fast_translation_function(obs, cand, data.spacegroup, data.cell, grid_spacing_A=d_min_set / 3.0,
                                     n_peaks=12, cluster_radius_A=d_min_set, fixed=fx)
t_c = torch.as_tensor(np.stack([p.translation for p in peaks]), dtype=torch.float64)
llg = llg_at_translations(obs, cand, t_c, fixed=fx)
print(f"true-orientation k={best.candidate_index} winner k={sols[0].candidate_index} winner pose vs own="
      f"{tuple(round(v,2) for v in pose_error(sols[0].model.xyz(), canonical, data.cell, data.spacegroup, allow_origin_freedom=False))}")
for i, p in enumerate(peaks):
    sol = MRSolution(rotation=best.rotation, translation=p.translation, rotation_score=0, translation_score=p.score, r_factor=0)
    m = pipe.place(sol)
    pe = pose_error(m.xyz(), canonical, data.cell, data.spacegroup, allow_origin_freedom=False)
    print(f"PEAK {i} t={np.round(p.translation,3).tolist()} tf={p.score:.1f} z={p.sigma:.2f} llg={float(llg[i]):.1f} "
          f"clash={pipe._clash_fraction_at(t_c[i]):.2f} pose={tuple(round(v,2) for v in pe)}")
t0 = torch.zeros(1, 3, dtype=torch.float64)
print(f"TRUE-T llg={float(llg_at_translations(obs, cand, t0, fixed=fx)[0]):.1f} clash={pipe._clash_fraction_at(t0[0]):.2f}")
