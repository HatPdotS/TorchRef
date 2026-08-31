"""Can the translation stage share one molecular transform across orientations?

``tf_cost.py`` says the per-candidate placement is ~95% ``precompute_G``: a full
structure-factor evaluation of a *re-rotated* model at ``S*N`` Miller indices,
paid again for every orientation. The Crowther-Blow accumulation and the local
refine that follow it are milliseconds.

But the rotation does not have to live in the coordinates. ``F(h; R x) =
F(R^T h; x)``, which is exactly what :class:`LattmanLoveInterpolator` is for and
what the *rotation* search already uses -- and its ``evaluate`` is already
batched over ``R``. So ``G`` for M orientations could be one dense grid plus
``M*S*N`` trilinear lookups instead of M structure-factor calculations.

Two things have to hold and neither is obvious:

* **The phase must survive trilinear interpolation.** The rotation search only
  ever reads ``|F|``; the translation function reads ``arg F``, and the class's
  own docstring warns that complex interpolation is only safe on a
  well-oversampled grid. Measured here as the agreement of the *translation
  peaks*, not of ``F`` -- a phase error that does not move the peak does not
  matter.
* **It must actually be faster**, including the one-off grid build, at the
  orientation counts we would carry.

Reports both against the exact per-rotation path.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="1DAW", choices=list(BENCH_PDBS))
    ap.add_argument("--n-rot", type=int, default=8, help="orientations to time")
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=15.0)
    ap.add_argument("--max-res", type=float, default=3.0)
    ap.add_argument("--padding", type=float, default=2.0)
    ap.add_argument("--grid-steps", type=int, default=16)
    args = ap.parse_args()

    from torchref.experimental.alignment.align import _DirectModelEvaluator
    from torchref.experimental.alignment.lattman_love import LattmanLoveInterpolator
    from torchref.experimental.alignment.translation import (
        amplitude_translation_search, precompute_G_for_rotation,
    )

    model, data = load_case(args.pdb)
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64)
    mask = data.get_valid_mask()
    s_mag = (data.hkl.to(torch.float64) @ rec).norm(dim=-1)
    mask = mask & (s_mag >= 1.0 / args.d_max) & (s_mag <= 1.0 / args.d_min)
    hkl = data.hkl[mask]
    F_obs = data.F[mask].abs().to(torch.float64)
    S = int(data.spacegroup.matrices.shape[0])
    N = int(hkl.shape[0])
    print(f"# {args.pdb} sg={data.spacegroup.hm} S={S} N={N} "
          f"atoms={model.xyz().shape[0]} max_res={args.max_res} "
          f"padding={args.padding}", flush=True)

    base = model.copy()
    base.spacegroup = "P 1"
    origin = torch.zeros(3, dtype=base.xyz().dtype)
    eye3 = torch.eye(3, dtype=torch.float64)

    t0 = time.perf_counter()
    ll = LattmanLoveInterpolator(base, padding_factor=args.padding,
                                 max_res_A=args.max_res, verbose=0)
    t_grid = time.perf_counter() - t0
    print(f"# dense grid build {t_grid:.2f} s  shape="
          f"{tuple(ll.reciprocal_grid.shape)}", flush=True)

    # h_R is a function of hkl and the space group only -- not of the candidate
    # rotation -- so the index side of the whole stage is shared.
    sym_R = data.spacegroup.matrices.to(torch.float64)
    h_R = torch.einsum("ne,ied->ind", hkl.to(torch.float64), sym_R)
    h_R_flat = h_R.reshape(-1, 3)

    rots = [random_rotation(seed_for(args.pdb, 0) + 97 * k)
            for k in range(args.n_rot)]

    t_direct = t_interp = 0.0
    for k, R in enumerate(rots):
        rot = base.copy().rotate(R.to(base.dtype_float), center=origin)
        ev = _DirectModelEvaluator(rot)
        t0 = time.perf_counter()
        G_d, h_R_d = precompute_G_for_rotation(ev, eye3, hkl,
                                               data.spacegroup, data.cell)
        t_direct += time.perf_counter() - t0

        t0 = time.perf_counter()
        F_i = ll.evaluate(R.to(torch.float32), h_R_flat, data.cell,
                          return_amplitude=False).reshape(S, N)
        phase_sym = torch.exp(2j * torch.pi * torch.einsum(
            "ne,ie->in", hkl.to(torch.float64),
            data.spacegroup.translations.to(torch.float64),
        ).to(torch.complex128))
        G_i = F_i.to(torch.complex128) * phase_sym
        t_interp += time.perf_counter() - t0

        if k == 0:
            a, b = G_d.reshape(-1), G_i.reshape(-1)
            coh = float((a.conj() * b).sum().abs()
                        / (a.abs().norm() * b.abs().norm()).clamp(min=1e-30))
            amp = float(torch.corrcoef(torch.stack(
                [a.abs(), b.abs()]).to(torch.float64))[0, 1])
            print(f"# G agreement: complex coherence={coh:.4f} "
                  f"|F| corr={amp:.4f}", flush=True)

        tf = lambda G: amplitude_translation_search(
            F_obs=F_obs, interpolator=ev, R_rotation=eye3, hkl=hkl,
            spacegroup=data.spacegroup, real_cell=data.cell,
            grid_steps=args.grid_steps, n_peaks=5,
            precomputed_G=G, precomputed_h_R=h_R_d)[2]
        pd, pi = tf(G_d), tf(G_i)
        dt = min(float(torch.tensor(
            ((torch.as_tensor(pd[0].translation)
              - torch.as_tensor(pi[j].translation) + 0.5) % 1.0 - 0.5).norm()))
            for j in range(len(pi)))
        dt_top = float(torch.tensor(
            ((torch.as_tensor(pd[0].translation)
              - torch.as_tensor(pi[0].translation) + 0.5) % 1.0 - 0.5).norm()))
        print(f"ROW pdb={args.pdb} rot={k} dt_top={dt_top:.4f} "
              f"dt_best_of_5={dt:.4f} "
              f"score_direct={pd[0].score:.4f} score_interp={pi[0].score:.4f}",
              flush=True)

    M = args.n_rot
    print(f"SUM pdb={args.pdb} S={S} N={N} n_rot={M} "
          f"t_direct_per_rot={t_direct / M:.3f} "
          f"t_interp_per_rot={t_interp / M:.4f} "
          f"t_grid={t_grid:.2f} "
          f"breakeven_rots={t_grid / max(t_direct / M - t_interp / M, 1e-9):.1f} "
          f"speedup_at_100={100 * (t_direct / M) / (t_grid + 100 * t_interp / M):.1f}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
