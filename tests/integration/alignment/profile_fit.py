#!/usr/bin/env python
"""
Profile `ModelFT.fit_to_data` to find where the time goes.

Run from the repo root:
    .venv/bin/python tests/integration/alignment/profile_fit.py [--pdb 1DAW] \
                     [--n-rotation-candidates 3] [--n-translation-candidates 3] \
                     [--translation-grid-steps 16]

Output: top-50 cumulative-time entries from cProfile + a custom per-stage timer
breakdown (rotation search, ML rescore, TF, local refine, joint refine,
final Scaler refit).
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from torchref.experimental.alignment import align_model_to_data
from torchref.experimental.alignment.frf.rotation_utils import rotation_matrix_from_edmonds_euler
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT


TEST_FILES = Path("/das/work/p17/p17490/Peter/Library/work_trees_torchref/fix_alignment/tests/files")

PAIRS = {
    "1DAW": (TEST_FILES / "pdb" / "1DAW.pdb", TEST_FILES / "mtz" / "1DAW.mtz"),
    "1AK5": (TEST_FILES / "pdb" / "1AK5_with_H.pdb", TEST_FILES / "mtz" / "1AK5.mtz"),
    "3A5V": (TEST_FILES / "pdb" / "3A5V.pdb", TEST_FILES / "mtz" / "3A5V.mtz"),
}


_TIMINGS: dict[str, float] = {}


@contextmanager
def _stage(name: str):
    t0 = time.time()
    try:
        yield
    finally:
        _TIMINGS[name] = _TIMINGS.get(name, 0.0) + (time.time() - t0)
        print(f"  [{name}] {time.time()-t0:.2f}s", flush=True)


def _patch_for_timing():
    """Wrap key fit_to_data stages so we get an inline breakdown."""
    from torchref.experimental.alignment import rotation_search, translation
    from torchref import scaling

    originals = {}

    def wrap(module, attr, label):
        original = getattr(module, attr)
        originals[(module, attr)] = original

        def wrapper(*args, **kwargs):
            with _stage(label):
                return original(*args, **kwargs)

        setattr(module, attr, wrapper)

    wrap(rotation_search, "search_peaks", "rotation_search")
    wrap(translation, "amplitude_translation_search", "amplitude_translation_search")
    wrap(translation, "local_translation_refine", "local_translation_refine")
    wrap(translation, "precompute_G_for_rotation", "precompute_G_for_rotation")
    wrap(translation, "llg_translation_rescore", "llg_translation_rescore")
    return originals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="1DAW", choices=sorted(PAIRS.keys()))
    ap.add_argument("--n-rotation-candidates", type=int, default=3)
    ap.add_argument("--n-translation-candidates", type=int, default=3)
    ap.add_argument("--translation-grid-steps", type=int, default=16)
    ap.add_argument("--top", type=int, default=40, help="top N cProfile entries")
    args = ap.parse_args()

    pdb_path, mtz_path = PAIRS[args.pdb]
    print(f"=== Profiling fit_to_data on {args.pdb} ===", flush=True)

    data = ReflectionData().load_mtz(str(mtz_path))
    canonical = ModelFT().load_pdb(str(pdb_path))
    canonical.spacegroup = "P 1"
    R_true = rotation_matrix_from_edmonds_euler(0.6, 0.4, 1.2)
    rotated_p = canonical.rotate(
        R_true.to(canonical.dtype_float), center=canonical.xyz().mean(dim=0),
    )
    t_true = torch.tensor([0.18, -0.07, 0.23], dtype=canonical.dtype_float)
    perturbed = rotated_p.translate(t_true, fractional=True)
    print(f"  spacegroup={data.spacegroup}, n_atoms={canonical.xyz().shape[0]}, "
          f"n_hkl={data.hkl.shape[0]}", flush=True)

    _patch_for_timing()

    profiler = cProfile.Profile()
    t0 = time.time()
    profiler.enable()
    aligned = align_model_to_data(
        perturbed,
        data,
        n_rotation_candidates=args.n_rotation_candidates,
        n_translation_candidates=args.n_translation_candidates,
        translation_grid_steps=args.translation_grid_steps,
        verbose=0,
    )
    profiler.disable()
    total = time.time() - t0

    print(f"\n=== Stage breakdown (total {total:.2f}s) ===", flush=True)
    other = total - sum(_TIMINGS.values())
    for name, t in sorted(_TIMINGS.items(), key=lambda kv: -kv[1]):
        print(f"  {name:40s} {t:8.2f}s   ({100*t/total:5.1f}%)", flush=True)
    print(f"  {'(unattributed)':40s} {other:8.2f}s   ({100*other/total:5.1f}%)",
          flush=True)

    print(f"\n=== Top-{args.top} cProfile (cumulative time) ===", flush=True)
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(args.top)

    print(f"\n=== Top-{args.top} cProfile (own time) ===", flush=True)
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats(args.top)


if __name__ == "__main__":
    main()
