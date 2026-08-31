"""Where the translation stage actually spends its time, per rotation candidate.

The pipeline carries ``n_rotation_candidates`` orientations through
:func:`_placement_for_candidate` one at a time in a Python loop, and every
orientation repays the whole stage. Before batching it over 100 orientations we
need to know which part of it is the cost: the structure-factor evaluation, the
Crowther-Blow accumulation, the LLG re-rank, or the local refine.

Reports the per-candidate breakdown alongside the problem geometry (``N``
reflections, ``S`` sym-ops, grid), because the four stages scale differently --
``O(n_atoms*S*N)``, ``O(S^2*N)``, ``O(K*S*N)`` and ``O(S^2*N)`` respectively --
and which one dominates is a property of the structure, not of the code.
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


def _time(fn, repeats=1):
    out = None
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn()
    return (time.perf_counter() - t0) / repeats, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="1DAW", choices=list(BENCH_PDBS))
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=15.0)
    ap.add_argument("--grid-steps", type=int, default=16)
    ap.add_argument("--n-peaks", type=int, default=20)
    args = ap.parse_args()

    from torchref.experimental.alignment.translation import (
        DirectModelEvaluator, TranslationObs, amplitude_translation_search,
        local_translation_refine, precompute_G_for_rotation,
    )

    seed = seed_for(args.pdb, args.trial)
    model, data = load_case(args.pdb)
    R_true = random_rotation(seed)
    rot = model.copy()
    rot.spacegroup = "P 1"
    rot = rot.rotate(R_true.to(model.dtype_float), center=model.xyz().mean(0))

    # Same masking the pipeline's _prepare_translation_arrays does.
    mask = data.get_valid_mask()
    d = 1.0 / (data.hkl.to(torch.float64)
               @ data.cell.reciprocal_basis_matrix.to(torch.float64)
               ).norm(dim=-1).clamp(min=1e-9)
    mask = mask & (d >= args.d_min) & (d <= args.d_max)
    hkl = data.hkl[mask]
    sig_F = getattr(data, "F_sigma", None)
    obs = TranslationObs.build(
        data.F[mask], hkl, data.spacegroup, data.cell,
        sig_F=None if sig_F is None else sig_F[mask],
    )
    S = int(data.spacegroup.matrices.shape[0])
    N = int(hkl.shape[0])
    n_at = int(rot.xyz().shape[0])
    print(f"# {args.pdb} sg={data.spacegroup.hm} S={S} N={N} atoms={n_at} "
          f"grid={args.grid_steps} seed={seed}", flush=True)

    ev = DirectModelEvaluator(rot)
    eye3 = torch.eye(3, dtype=torch.float64)

    t_G, (G, h_R) = _time(lambda: precompute_G_for_rotation(
        ev, eye3, hkl, data.spacegroup, data.cell))
    t_tf, (_, _, peaks) = _time(lambda: amplitude_translation_search(
        obs=obs, interpolator=ev, R_rotation=eye3,
        spacegroup=data.spacegroup, real_cell=data.cell,
        grid_steps=args.grid_steps, n_peaks=args.n_peaks,
        precomputed_G=G, precomputed_h_R=h_R))
    t_ref, _ = _time(lambda: local_translation_refine(
        obs=obs, interpolator=ev, R_rotation=eye3,
        spacegroup=data.spacegroup, real_cell=data.cell,
        t_init=torch.as_tensor(peaks[0].translation, dtype=torch.float64),
        radius=0.06, grid_steps=13, n_refinement_passes=1,
        precomputed_G=G, precomputed_h_R=h_R))

    # The (S,N) working set the Crowther-Blow loop materialises, and the
    # (K,S,N) one the LLG re-rank does, in complex128.
    mb = lambda *shape: 16.0 * float(torch.tensor(shape).prod()) / 2**20
    print(f"ROW pdb={args.pdb} sg={data.spacegroup.hm} S={S} N={N} atoms={n_at} "
          f"t_G={t_G:.3f} t_tf={t_tf:.3f} t_refine={t_ref:.3f} "
          f"t_place={t_G + t_tf + 3 * t_ref:.3f} "
          f"work_S2N={S * S * N / 1e6:.1f}M "
          f"mem_SN_MB={mb(S, N):.0f} mem_KSN_MB={mb(args.n_peaks, S, N):.0f}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
