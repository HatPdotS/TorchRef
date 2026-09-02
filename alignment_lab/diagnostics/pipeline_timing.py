"""Warm, single-process wall clock of the placement pipeline, by stage.

One process, each structure aligned twice, the first pass discarded: the first
call in a process pays kernel builds and cold file-system reads that are not
compute (161 s against 1.5 s has been measured for one stage). Prints the
pipeline's own stage table for the second pass and the pose error, so a timing
is never quoted for a run that did not place the model.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import (BENCH_PDBS, load_case, pose_error, random_rotation,  # noqa: E402
                 seed_for)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdbs", default="1DAW,2DQ6,6G9X,3K7M,4BX9")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from torchref.experimental.alignment import MolecularReplacementPipeline

    for pdb in [p.strip() for p in args.pdbs.split(",") if p.strip()]:
        assert pdb in BENCH_PDBS
        model, data = load_case(pdb)
        canonical = model.xyz().clone()
        R_true = random_rotation(seed_for(pdb, 0))
        for run in range(2):
            search = model.copy()
            search.spacegroup = "P 1"
            search = search.copy().rotate(R_true.to(model.dtype_float),
                                          center=canonical.mean(0))
            pipe = MolecularReplacementPipeline(
                data, search, d_min=4.0, d_max=15.0, n_shells=20,
                n_rotation_peaks=200, n_rotation_candidates=25,
                verbose=2 if run == 1 else 0,
            )
            t0 = time.perf_counter()
            sols = pipe.run(do_translation=True)
            secs = time.perf_counter() - t0
            rot, trans = pose_error(sols[0].model.xyz(), canonical, data.cell,
                                    data.spacegroup)
            print(f"ROW pdb={pdb} run={run} seconds={secs:.2f} rot_deg={rot:.2f} "
                  f"trans_A={trans:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
