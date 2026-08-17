#!/usr/bin/env python
"""TorchRef 0-cycle scoring: re-score models' R-factors with TorchRef.

The TorchRef analog of ``refmac5 NCYCLES 0`` / ``phenix.model_vs_data``: load a
model + data, fit only the scaler (overall + anisotropic scale + bulk solvent;
``get_scales`` does NOT move coordinates or B-factors), then report R-work/R-free
on the data's own free set. The R-factor formula matches ``torchref.refine -n 0``
``final_statistics`` so a TorchRef-refined model re-scored here reproduces its
self-reported value.

The real per-model work is ~1 s; the cost is the one-time torch/torchref import
(~14 s) plus per-model restraint/target construction (~13 s), all paid per
PROCESS. So this runs in BATCH mode: one warm process scores a chunk of a
worklist, amortizing imports + numba JIT (and the cold network-FS penalty) over
many models instead of spawning a fresh process per model.

Usage
-----
    # single model (testing)
    torchref_score.py -m model.pdb -sf data.mtz -o out.json

    # batch: score worklist lines [start, start+count) (1-indexed, tab-separated
    # "engine<TAB>code<TAB>model<TAB>mtz<TAB>out_json")
    torchref_score.py --worklist worklist.txt --start 1 --count 64
"""

import argparse
import gc
import json
import sys
import time

import torch

from torchref.refinement.lbfgs_refinement import LBFGSRefinement


def _has_result(out_path):
    try:
        with open(out_path) as f:
            return "r_free" in json.load(f)
    except (OSError, ValueError):
        return False


def score_one(model, mtz, out, xray_mode, device):
    """Score one model; write {r_work,r_free,n_work,n_test} to `out`."""
    ref = LBFGSRefinement(data_file=mtz, pdb=model, device=torch.device(device),
                          target_mode=xray_mode, verbose=0)
    ref.get_scales()  # 0-cycle: fit scaler + bulk solvent only
    with torch.no_grad():
        rd = ref.reflection_data
        # hkl=None routes through reflection_data.structure_factors(), the supported
        # entry point: it evaluates on the signed indices AND conjugates back to the
        # canonical ASU. Passing _hkl_for_sf() explicitly (as this did until the
        # accessor went private) returns the signed convention unconjugated, so the
        # Friedel half of every anomalous dataset carried a negated phase.
        fcalc = ref.get_F_calc_scaled(recalc=True)
        # work/free accessor (scaled, validity-masked); .select() aligns |F_calc|.
        fobs_w, fobs_f = rd.work.F, rd.free.F
        r_work = (torch.sum(torch.abs(fobs_w - rd.work.select(fcalc)))
                  / torch.sum(fobs_w)).item()
        r_free = (torch.sum(torch.abs(fobs_f - rd.free.select(fcalc)))
                  / torch.sum(fobs_f)).item()
        n_work, n_test = rd.work.n, rd.free.n
    with open(out, "w") as f:
        json.dump({"r_work": r_work, "r_free": r_free,
                   "n_work": n_work, "n_test": n_test}, f)
    return r_work, r_free


def run_batch(worklist, start, count, xray_mode, device):
    lines = [ln for ln in open(worklist).read().splitlines() if ln.strip()]
    chunk = lines[start - 1:start - 1 + count]  # 1-indexed inclusive range
    print(f"batch: {len(chunk)} tasks (lines {start}..{start + len(chunk) - 1} "
          f"of {len(lines)})", flush=True)
    ok = skip = fail = 0
    for i, ln in enumerate(chunk):
        eng, code, model, mtz, out = ln.split("\t")
        if _has_result(out):
            skip += 1
            continue
        t0 = time.time()
        try:
            rw, rf = score_one(model, mtz, out, xray_mode, device)
            ok += 1
            print(f"  [{i+1}/{len(chunk)}] {eng}/{code}: "
                  f"r_work={rw:.4f} r_free={rf:.4f}  ({time.time()-t0:.1f}s)",
                  flush=True)
        except Exception as e:  # one bad model must not kill the chunk
            fail += 1
            print(f"  [{i+1}/{len(chunk)}] {eng}/{code}: FAILED {type(e).__name__}: {e}",
                  flush=True)
        finally:
            gc.collect()
    print(f"batch done: ok={ok} skipped={skip} failed={fail}", flush=True)
    return fail


def main():
    ap = argparse.ArgumentParser(description="TorchRef 0-cycle R-factor scoring.")
    ap.add_argument("-m", "--model")
    ap.add_argument("-sf", "--structure-factor")
    ap.add_argument("-o", "--out", help="output JSON path (single-model mode)")
    ap.add_argument("--worklist", help="batch mode: worklist file")
    ap.add_argument("--start", type=int, default=1, help="batch: first line (1-indexed)")
    ap.add_argument("--count", type=int, default=1, help="batch: number of lines")
    ap.add_argument("--xray-mode", default="ml")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if a.worklist:
        # one bad model is logged, not fatal: always exit 0 so the chunk isn't
        # re-run wholesale (failures are picked up by the resume-safe worklist).
        run_batch(a.worklist, a.start, a.count, a.xray_mode, a.device)
        sys.exit(0)
    if not (a.model and a.structure_factor and a.out):
        ap.error("single-model mode needs -m, -sf, and -o")
    rw, rf = score_one(a.model, a.structure_factor, a.out, a.xray_mode, a.device)
    print(f"r_work={rw:.4f} r_free={rf:.4f}")


if __name__ == "__main__":
    main()
