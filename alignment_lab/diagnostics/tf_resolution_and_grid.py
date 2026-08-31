"""What resolution does the translation stage actually run at, and what grid does it need?

Two things to pin down, and the first invalidates my earlier numbers.

``_prepare_translation_arrays`` (``pipeline.py:638``) is documented as
"Resolution/validity-masked" but applies **only** ``data.get_valid_mask()`` --
there is no resolution cut. So the translation search runs at the data's full
resolution while the FRF runs at ``[d_max, d_min] = [15, 4]``. Any timing taken
on a 4 A subset is measuring a smaller problem than the pipeline solves.

Second, the FFT grid is sized from ``ModelFT.max_res``, which defaults to 1.0 A
and which this stage never sets. Whether that is oversized depends entirely on
the answer to the first question: against 4 A data it is 64x too many voxels,
against 2 A data it is the ~2x oversampling one would ask for anyway.

Reports the real N, the real ``d_min``, and times the structure-factor call on
grids sized at a range of ``max_res``, each checked for coherence against the
current 1.0 A grid -- because undersampling an FFT does not fail, it just
quietly returns different structure factors.
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
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--oversampling", default="1.0,1.5,2.0,3.0")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    model, data = load_case(args.pdb)
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64)
    # Exactly what the pipeline masks with -- no resolution window.
    mask = data.get_valid_mask()
    hkl = data.hkl[mask]
    s = (hkl.to(torch.float64) @ rec).norm(dim=-1)
    d_min = float(1.0 / s.max())
    d_max = float(1.0 / s.min().clamp(min=1e-9))
    sym_R = data.spacegroup.matrices.to(torch.float64)
    hkl_SN = torch.einsum("ne,ied->ind", hkl.to(torch.float64), sym_R
                          ).reshape(-1, 3).round().to(torch.int64)
    S, N = int(sym_R.shape[0]), int(hkl.shape[0])

    # For contrast: what the FRF's own window would leave.
    in_frf = ((s >= 1.0 / 15.0) & (s <= 1.0 / 4.0)).sum().item()
    print(f"# {args.pdb} sg={data.spacegroup.hm} S={S} "
          f"N_pipeline={N} N_in_4to15A={in_frf} "
          f"d_min={d_min:.2f} d_max={d_max:.1f} atoms={model.xyz().shape[0]} "
          f"S*N={S*N} threads={args.threads}", flush=True)

    rot = model.copy()
    rot.spacegroup = "P 1"
    rot = rot.rotate(random_rotation(seed_for(args.pdb, 0)).to(model.dtype_float),
                     center=torch.zeros(3, dtype=model.xyz().dtype))

    ref_sf = None
    for over in [1.0] + [float(x) for x in args.oversampling.split(",")]:
        m = rot.copy()
        m.max_res = 1.0 if ref_sf is None else d_min / over
        m.spacegroup = "P 1"
        t, sf = _time(lambda: (m.reset_cache(), m(hkl_SN))[1])
        if ref_sf is None:
            ref_sf, tag = sf.to(torch.complex128), "current(1.0A)"
            coh = 1.0
        else:
            x = sf.to(torch.complex128)
            coh = float((ref_sf.conj() * x).sum().abs()
                        / (ref_sf.abs().norm() * x.abs().norm()).clamp(min=1e-30))
            tag = f"d_min/{over:g}"
        print(f"ROW pdb={args.pdb} arm={tag} max_res={m.max_res:.3f} "
              f"grid={tuple(int(v) for v in m.fft.gridsize)} "
              f"t={t*1e3:.0f}ms coh={coh:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
