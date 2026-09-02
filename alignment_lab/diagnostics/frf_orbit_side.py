"""On which side does the crystal symmetry act on a rotation-function peak?

The FRF returns Euler matrices ``R`` mapping the search-model frame onto the
crystal frame. Two peaks are the same orientation when they differ by a
point-group rotation, but that rotation can compose on the left
(``R2 = R_g R1``) or on the right (``R2 = R1 R_g``), and the two are different
sets for a non-commuting group. Counting how many of the top peaks of a real
search collapse onto each other under each convention settles which one the
engine's peaks obey: mates of the true orientation appear many times in the
list, so the right convention finds many near-zero pairs and the wrong one few.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import BENCH_PDBS, cartesian_symops, load_case, random_rotation, seed_for  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="2DQ6", choices=list(BENCH_PDBS))
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--thr-deg", type=float, default=3.0)
    args = ap.parse_args()

    from torchref.experimental.alignment import rotation_search
    from torchref.experimental.alignment.sh import hkl_symops_to_cartesian

    model, data = load_case(args.pdb)
    search = model.copy()
    search = search.rotate(random_rotation(seed_for(args.pdb, 0)).to(model.dtype_float))
    sols = rotation_search(search, data, model_error_A=0.8, n_peaks=args.n)
    R = sols.rotations.to(torch.float64)                                   # (n, 3, 3)
    n = R.shape[0]

    sym_lab = cartesian_symops(data.spacegroup, data.cell)                 # B S B^-1
    sym_sh = hkl_symops_to_cartesian(
        data.spacegroup.matrices.to(torch.float64),
        data.cell.reciprocal_basis_matrix.to(torch.float64))
    agree = float((sym_lab - sym_sh).abs().max())
    print(f"# {args.pdb} {data.spacegroup} n_peaks={n} |B S B^-1 - hkl_symops_to_cartesian|max={agree:.2e}")

    def pair_count(orbit_of):
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                O = orbit_of(R[j])                                          # (g, 3, 3)
                tr = torch.einsum("gab,ab->g", O, R[i])
                ang = ((tr - 1) / 2).clamp(-1, 1).arccos().min() * 180 / torch.pi
                cnt += int(ang < args.thr_deg)
        return cnt

    left = pair_count(lambda Rj: sym_lab @ Rj.unsqueeze(0))
    right = pair_count(lambda Rj: Rj.unsqueeze(0) @ sym_lab)
    plain = pair_count(lambda Rj: Rj.unsqueeze(0))
    print(f"ROW pdb={args.pdb} pairs_within_{args.thr_deg:g}deg plain={plain} "
          f"left(R_g R)={left} right(R R_g)={right}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
