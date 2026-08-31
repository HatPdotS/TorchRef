"""Why does one orientation's structure-factor evaluation cost ~1 s?

``tf_cost.py`` put 93-97% of the translation stage in ``precompute_G``, which is
a single ``ModelFT.__call__`` at ``S*N`` Miller indices. That call goes through
``SfFFT`` (``model_ft.py:779``) -- splat the atoms onto a real-space grid, FFT
the box, sample the result. The grid is sized by the crystal cell and the
resolution, and it is built whether you wanted 12000 reflections or 12 million.

The translation search wants a sparse, fixed set: ``S*N`` is 13k-160k here,
against 5-20k atoms. That is the regime direct summation is for, and
:class:`SfDS` -- same ``compute_structure_factors`` signature, no grid -- is
already in the tree but is not what ``ModelFT`` dispatches to.

Times both on identical inputs and checks they agree, at the thread counts a
production run would see. Also separates first call from repeat: ``ModelFT`` is
rebuilt per orientation, so anything amortised across calls is paid in full by
the translation loop and has to be counted as setup, not as throughput.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import BENCH_PDBS, load_case, random_rotation, seed_for  # noqa: E402


def _time(fn, repeats=3):
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn()
    return (time.perf_counter() - t0) / repeats, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="1DAW", choices=list(BENCH_PDBS))
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=15.0)
    ap.add_argument("--threads", default="4,8")
    args = ap.parse_args()

    from torchref.model.sf_ds import SfDS

    model, data = load_case(args.pdb)
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64)
    s = (data.hkl.to(torch.float64) @ rec).norm(dim=-1)
    keep = data.get_valid_mask() & (s >= 1.0 / args.d_max) & (s <= 1.0 / args.d_min)
    hkl = data.hkl[keep]
    sym_R = data.spacegroup.matrices.to(torch.float64)
    h_R = torch.einsum("ne,ied->ind", hkl.to(torch.float64), sym_R)
    hkl_SN = h_R.reshape(-1, 3).round().to(torch.int64)
    S, N = int(sym_R.shape[0]), int(hkl.shape[0])

    rot = model.copy()
    rot.spacegroup = "P 1"
    rot = rot.rotate(random_rotation(seed_for(args.pdb, 0)).to(model.dtype_float),
                     center=torch.zeros(3, dtype=model.xyz().dtype))
    n_at = int(rot.xyz().shape[0])
    print(f"# {args.pdb} sg={data.spacegroup.hm} S={S} N={N} S*N={S*N} "
          f"atoms={n_at} model_max_res={rot.max_res}", flush=True)

    ds = SfDS(cell=rot.cell, spacegroup="P 1", dtype_float=rot.dtype_float,
              device=rot.xyz().device)
    iso, aniso = rot.get_iso(), rot.get_aniso()

    def fft_call(m):
        m.reset_cache()                # the loop gets a fresh model per candidate
        return m(hkl_SN)

    # The grid is sized by the model's ``max_res``, which ModelFT defaults to
    # 1.0 A. The translation search runs at d_min, so the default asks for
    # (d_min/1.0)^3 times the voxels it needs. Both the FRF's dense calc
    # (dense_calc.py:73) and the rigid-body stage (rigid_body.py:120) set this;
    # the translation stage does not.
    coarse = rot.copy()
    coarse.max_res = float(args.d_min)
    coarse.spacegroup = "P 1"

    for nt in [int(x) for x in args.threads.split(",")]:
        torch.set_num_threads(nt)
        t_fft, sf_fft = _time(lambda: fft_call(rot))
        t_coarse, sf_coarse = _time(lambda: fft_call(coarse))
        t_ds, (sf_ds, _) = _time(lambda: ds.compute_structure_factors(
            hkl_SN, *iso, *aniso, apply_symmetry=True))

        a = sf_fft.to(torch.complex128)
        agree = lambda x: float((a.conj() * x.to(torch.complex128)).sum().abs()
                                / (a.abs().norm()
                                   * x.abs().norm()).clamp(min=1e-30))
        print(f"ROW pdb={args.pdb} threads={nt} S={S} N={N} SN={S*N} "
              f"atoms={n_at} grid_1A={tuple(int(v) for v in rot.fft.gridsize)} "
              f"grid_dmin={tuple(int(v) for v in coarse.fft.gridsize)} "
              f"t_fft_1A={t_fft*1e3:.0f}ms t_fft_dmin={t_coarse*1e3:.0f}ms "
              f"t_ds={t_ds*1e3:.0f}ms "
              f"gain_grid={t_fft/max(t_coarse,1e-9):.1f}x "
              f"gain_ds={t_fft/max(t_ds,1e-9):.1f}x "
              f"coh_dmin={agree(sf_coarse):.6f} coh_ds={agree(sf_ds):.6f}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
