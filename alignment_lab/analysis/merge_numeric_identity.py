"""Did merging dev change any number the alignment path produces?

`a596ed9e` removes SfFFT's stored real-space coordinate grid. Reading it, the
change is plumbing: `build_electron_density` used the tensor only for `.device`
and `.shape[:-1]`, the four Triton kernels never dereferenced the `grid_ptr`
they were handed, and the new `voxel_size = frac_matrix @ (1/gridsize)` is
algebraically the old `grid[2,2,2] - grid[1,1,1]` -- `cart = B @ f`, so column j
of the fractionalisation matrix is cell edge vector j.

"Reading it, it looks equivalent" is not the same as equivalent. This dumps
hashes of the quantities the alignment stack actually consumes so the two trees
can be compared bit for bit.

Deliberately covers a monoclinic and two high-symmetry cells: for an orthogonal
cell the fractionalisation matrix is diagonal, so a transposed edge-vector
convention would be invisible. 1DAW is C2 and 2DQ6 is P3121, where it would not.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import BENCH_PDBS, FRFConfig, load_case, run_frf  # noqa: E402


def _h(t: torch.Tensor) -> str:
    """Bit-exact hash of a tensor's contents, dtype and shape."""
    t = t.detach().cpu().contiguous()
    m = hashlib.sha256()
    m.update(str(tuple(t.shape)).encode())
    m.update(str(t.dtype).encode())
    m.update(t.numpy().tobytes())
    return m.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True, choices=list(BENCH_PDBS))
    ap.add_argument("--tag", default="?")
    ap.add_argument("--lmax-cap", type=int, default=64)
    args = ap.parse_args()

    model, data = load_case(args.pdb)
    hkl = data.hkl

    # 1. Structure factors -- the thing every downstream number is built on.
    F = model.get_structure_factor(hkl, recalc=True)
    print(f"OUT {args.tag} {args.pdb} F_calc {_h(F)}")

    # 2. The density map itself, one step earlier than F_calc, so a difference
    #    can be localised to the splat rather than the FFT.
    model.setup_grid()
    dm = model.build_complete_map()
    print(f"OUT {args.tag} {args.pdb} density {_h(dm)}")
    print(f"OUT {args.tag} {args.pdb} gridshape {tuple(dm.shape)}")

    # 3. voxel_size: the one quantity whose FORMULA changed, rather than only
    #    its call site. Nothing reads it downstream, so a difference here is
    #    reportable but not itself a regression.
    vs = model.voxel_size
    print(f"OUT {args.tag} {args.pdb} voxel_size "
          f"{'None' if vs is None else _h(vs)} "
          f"{'' if vs is None else [f'{float(v):.17g}' for v in vs.flatten()]}")

    # 4. The FRF peak list -- what this branch is actually judged on.
    res = run_frf(model, data, FRFConfig(n_peaks=500, lmax_cap=args.lmax_cap),
                  capture_arf=False, verbose=0)
    ang = torch.tensor([[p.alpha, p.beta, p.gamma] for p in res.peaks],
                       dtype=torch.float64)
    sc = torch.tensor([p.score for p in res.peaks], dtype=torch.float64)
    print(f"OUT {args.tag} {args.pdb} peaks_angles {_h(ang)}")
    print(f"OUT {args.tag} {args.pdb} peaks_scores {_h(sc)}")
    print(f"OUT {args.tag} {args.pdb} n_peaks {len(res.peaks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
