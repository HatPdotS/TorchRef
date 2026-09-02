"""How coarse can the P1 copy's FFT grid be for the translation set?

The placement stage evaluates the P1 model's transform at the symmetry-rotated
indices of the translation set. Its grid was sized by the model's default
``max_res = 1.0 A`` whatever the set's resolution. This measures the complex
coherence of ``F_calc`` at the 15-4 A reflections between that grid and grids
sized to ``tf_d_min / oversampling``, and the time of each.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import BENCH_PDBS, load_case, random_rotation, seed_for  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="1DAW", choices=list(BENCH_PDBS))
    ap.add_argument("--tf-d-min", type=float, default=4.0)
    ap.add_argument("--tf-d-max", type=float, default=15.0)
    ap.add_argument("--oversampling", default="1.0,1.33,2.0")
    args = ap.parse_args()

    model, data = load_case(args.pdb)
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64)
    mask = data.get_valid_mask()
    hkl = data.hkl[mask]
    s = (hkl.to(torch.float64) @ rec).norm(dim=-1)
    keep = (s >= 1.0 / args.tf_d_max) & (s <= 1.0 / args.tf_d_min)
    hkl = hkl[keep]
    sym_R = data.spacegroup.matrices.to(torch.float64)
    hkl_SN = torch.einsum("ne,ied->ind", hkl.to(torch.float64), sym_R
                          ).reshape(-1, 3).round().to(torch.int64)
    print(f"# {args.pdb} sg={data.spacegroup.hm} S={sym_R.shape[0]} N={hkl.shape[0]} "
          f"window={args.tf_d_max}-{args.tf_d_min} A", flush=True)

    rot = model.copy()
    rot = rot.rotate(random_rotation(seed_for(args.pdb, 0)).to(model.dtype_float))

    ref = None
    for over in [None] + [float(x) for x in args.oversampling.split(",")]:
        m = rot.copy()
        m.max_res = 1.0 if over is None else args.tf_d_min / over
        m.spacegroup = "P 1"
        with torch.no_grad():
            m.reset_cache(); m(hkl_SN)                  # warm
            t0 = time.perf_counter()
            for _ in range(3):
                m.reset_cache(); sf = m(hkl_SN)
            t = (time.perf_counter() - t0) / 3
        x = sf.to(torch.complex128)
        if ref is None:
            ref, coh, tag = x, 1.0, "1.0A"
        else:
            coh = float((ref.conj() * x).sum().abs()
                        / (ref.abs().norm() * x.abs().norm()).clamp(min=1e-30))
            tag = f"d_min/{over:g}"
        print(f"ROW pdb={args.pdb} arm={tag} max_res={m.max_res:.3f} "
              f"grid={tuple(int(v) for v in m.fft.gridsize)} t={t*1e3:.1f}ms "
              f"coh={coh:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
