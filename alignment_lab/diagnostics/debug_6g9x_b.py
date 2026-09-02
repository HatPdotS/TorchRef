"""Where does the pipeline put one chain, relative to itself and its siblings, and
how does the deposited pose score through the pipeline's own path?"""
import argparse
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import load_case, pose_error, random_rotation, seed_for  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain_pose_recovery import chain_selection, sibling_coordinates  # noqa: E402
from torchref.experimental.alignment import MolecularReplacementPipeline  # noqa: E402
from torchref.experimental.alignment.translation import (  # noqa: E402
    analytic_r_at, llg_at_translations, prepare_candidate, translation_score_at)

ap = argparse.ArgumentParser()
ap.add_argument("--pdb", required=True); ap.add_argument("--chain", required=True)
ap.add_argument("--trial", type=int, default=0)
args = ap.parse_args()
model, data = load_case(args.pdb)
ch = model.select(chain_selection(model, args.chain))
sibs = sibling_coordinates(model, ch, args.chain)
print("siblings:", [(c, n) for c, _, _, n in sibs], "n_atoms", ch.xyz().shape[0])
canonical = ch.xyz().clone()
for other, idx_own, xyz_other, n in sibs:
    print("deposited", args.chain, "vs deposited", other, "->",
          tuple(round(v, 2) for v in pose_error(canonical[idx_own], xyz_other, data.cell, data.spacegroup)))
R_true = random_rotation(seed_for(args.pdb, args.trial) + 7919 * (ord(args.chain[0]) % 26))
search = ch.copy(); search.spacegroup = "P 1"
search = search.copy().rotate(R_true.to(model.dtype_float), center=canonical.mean(0))
pipe = MolecularReplacementPipeline(data, search, d_min=4.0, d_max=15.0, n_shells=20,
                                    n_rotation_peaks=200, n_rotation_candidates=10, verbose=3)
sols = pipe.run()
for i, s in enumerate(sols[:4]):
    m = pipe.place(s)
    own = pose_error(m.xyz(), canonical, data.cell, data.spacegroup)
    line = f"SOL {i} k={s.candidate_index} llg={s.llg_score:.0f} R={s.r_factor:.4f} tf={s.translation_score:.1f} vs_{args.chain}={tuple(round(v,2) for v in own)}"
    for other, idx_own, xyz_other, n in sibs:
        line += f" vs_{other}={tuple(round(v,2) for v in pose_error(m.xyz()[idx_own], xyz_other, data.cell, data.spacegroup))}"
    print(line)
# The deposited chain, scored through the same path.
dep = ch.copy()
if pipe.tf_d_min > 0.0:
    dep.max_res = pipe.tf_d_min / 1.5
dep.spacegroup = "P 1"
cand = prepare_candidate(dep, pipe._obs, data.spacegroup, data.cell)
t0 = torch.zeros(3, dtype=torch.float64)
print(f"DEPOSITED tf={translation_score_at(pipe._obs, cand, t0):.1f} R={analytic_r_at(pipe._obs, cand, t0):.4f} "
      f"llg={float(llg_at_translations(pipe._obs, cand, t0.view(1, 3))[0]):.0f}")
