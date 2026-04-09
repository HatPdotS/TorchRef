#!/usr/bin/env python3
"""Benchmark GPU memory usage for Extended Figure 3.

Profiles GPU memory during refinement, tracking allocation at each stage.
Also profiles peak memory across ~25 structures for the scatter plot.

Output: data/extfig3_memory.csv          (peak memory per structure)
        data/extfig3_memory_timeline.json (per-stage memory for selected structures)

Usage:
    # Full run (timeline + scatter):
    python benchmark_extfig3_memory.py

    # Timeline only for specific structures:
    python benchmark_extfig3_memory.py --timeline-only --codes 138L 3K7M 1AQ3

    # Include the OOM structures from the GPU pipeline:
    python benchmark_extfig3_memory.py --include-oom
"""

import argparse
import csv
import gc
import json
import sys
import traceback
from pathlib import Path

import gemmi
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
PAPER_ROOT = BASE.parent
DATA_DIR = PAPER_ROOT / "data"
STRUCTURES_JSON = PAPER_ROOT / "figure2_validation" / "structures.json"
OUT_CSV = BASE / "data" / "extfig3_memory.csv"
OUT_TIMELINE = BASE / "data" / "extfig3_memory_timeline.json"

# Structures that OOM'd on 8 GB GPUs during the 1000-structure pipeline
OOM_CODES = [
    "1AQ3", "1N0L", "1Q4K", "1U1I", "2X1C", "2XDO", "3L1W",
    "4JMQ", "4KH9", "4QH8", "5BOV", "5UX5", "6GBS", "6H0B",
]


# ── Memory tracking ─────────────────────────────────────────────────────────

def _snap(label, trace):
    """Record GPU memory snapshot: current allocated + peak since last reset."""
    torch.cuda.synchronize()
    cur = torch.cuda.memory_allocated() / (1024**2)
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    trace.append({"label": label, "current_mb": cur, "peak_mb": peak})
    return cur, peak


def _reset():
    torch.cuda.reset_peak_memory_stats()


# ── Structure info ───────────────────────────────────────────────────────────

def get_structure_info(code):
    pdb_path = DATA_DIR / code / f"{code}_shaken.pdb"
    mtz_path = DATA_DIR / code / f"{code}.mtz"
    if not pdb_path.exists() or not mtz_path.exists():
        return None
    try:
        st = gemmi.read_structure(str(pdb_path))
        n_atoms = sum(1 for model in st for chain in model for res in chain for _ in res)
        mtz = gemmi.read_mtz_file(str(mtz_path))
        return {"code": code, "n_atoms": n_atoms, "d_min": mtz.resolution_high(),
                "n_refl": mtz.nreflections}
    except Exception:
        return None


def select_structures(n_target=25, include_oom=False):
    with open(STRUCTURES_JSON) as f:
        codes = json.load(f)
    print(f"Scanning {len(codes)} structures...")
    infos = [get_structure_info(c) for c in codes]
    infos = [i for i in infos if i is not None]
    infos.sort(key=lambda x: x["n_atoms"])
    print(f"  {len(infos)} valid structures")

    if len(infos) <= n_target:
        selected = infos
    else:
        indices = np.linspace(0, len(infos) - 1, n_target, dtype=int)
        selected = [infos[i] for i in indices]

    if include_oom:
        selected_codes = {s["code"] for s in selected}
        for code in OOM_CODES:
            if code not in selected_codes:
                info = get_structure_info(code)
                if info is not None:
                    selected.append(info)
        selected.sort(key=lambda x: x["n_atoms"])

    print(f"  Selected {len(selected)} (atoms: {selected[0]['n_atoms']}–{selected[-1]['n_atoms']})")
    return selected


# ── Timeline profiling ───────────────────────────────────────────────────────

def profile_timeline(code, device):
    """Profile GPU memory at each stage of a refinement run."""
    from torchref import LBFGSRefinement

    pdb_path = str(DATA_DIR / code / f"{code}_shaken.pdb")
    mtz_path = str(DATA_DIR / code / f"{code}.mtz")

    gc.collect()
    torch.cuda.empty_cache()
    _reset()
    trace = []

    _snap("start", trace)

    # ── Init ──
    _reset()
    ref = LBFGSRefinement(
        data_file=mtz_path, pdb=pdb_path, device=device, verbose=0,
    )
    _snap("after_init", trace)

    n_atoms = int(ref.model.xyz().shape[0])
    n_refl = int(ref.reflection_data()[0].shape[0])
    d_min = float(ref.reflection_data.d_min)

    # ── Scaling ──
    _reset()
    ref.get_scales()
    _snap("after_scaling", trace)

    # ── Loss state (XYZ) ──
    _reset()
    ref.model.unfreeze("xyz")
    ref.model.freeze("b")
    state = ref.complete_loss_state()
    _snap("after_loss_state_xyz", trace)

    # ── Forward (XYZ) ──
    _reset()
    loss = state.aggregate()
    _snap("after_forward_xyz", trace)

    # ── Backward (XYZ) ──
    _reset()
    loss.backward()
    _snap("after_backward_xyz", trace)

    # ── Full refine_xyz ──
    del loss, state
    gc.collect()
    torch.cuda.empty_cache()
    _reset()
    ref.refine_xyz()
    _snap("after_refine_xyz", trace)

    # ── Full refine_adp ──
    gc.collect()
    torch.cuda.empty_cache()
    _reset()
    ref.refine_adp()
    _snap("after_refine_adp", trace)

    # ── Cycle 2 ──
    gc.collect()
    torch.cuda.empty_cache()
    _reset()
    ref.get_scales()
    _snap("cycle2_scaling", trace)

    _reset()
    ref.refine_xyz()
    _snap("cycle2_xyz", trace)

    _reset()
    ref.refine_adp()
    _snap("cycle2_adp", trace)

    del ref
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "code": code, "n_atoms": n_atoms, "n_reflections": n_refl,
        "d_min": d_min, "trace": trace,
    }


# ── Peak-only profiling ─────────────────────────────────────────────────────

def profile_peak(code, device):
    """Profile peak GPU memory during init + one refinement cycle."""
    from torchref import LBFGSRefinement

    pdb_path = str(DATA_DIR / code / f"{code}_shaken.pdb")
    mtz_path = str(DATA_DIR / code / f"{code}.mtz")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        ref = LBFGSRefinement(
            data_file=mtz_path, pdb=pdb_path, device=device, verbose=0,
        )
        n_atoms = int(ref.model.xyz().shape[0])
        n_refl = int(ref.reflection_data()[0].shape[0])
        d_min = float(ref.reflection_data.d_min)

        # Peak during init
        torch.cuda.synchronize()
        peak_init_gb = torch.cuda.max_memory_allocated() / (1024**3)

        # One full cycle
        torch.cuda.reset_peak_memory_stats()
        ref.get_scales()
        ref.refine_xyz()
        ref.refine_adp()
        torch.cuda.synchronize()
        peak_cycle_gb = torch.cuda.max_memory_allocated() / (1024**3)

        # Overall peak = max of init and cycle
        peak_gb = max(peak_init_gb, peak_cycle_gb)

        del ref
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "code": code, "n_atoms": n_atoms, "n_reflections": n_refl,
            "d_min": d_min,
            "peak_memory_gb": peak_gb,
            "peak_init_gb": peak_init_gb,
            "peak_cycle_gb": peak_cycle_gb,
        }
    except Exception as e:
        print(f"  FAILED: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPU memory profiling for ExtFig 3")
    parser.add_argument("--n-structures", type=int, default=25)
    parser.add_argument("--timeline-only", action="store_true")
    parser.add_argument("--scatter-only", action="store_true")
    parser.add_argument("--codes", nargs="+")
    parser.add_argument("--include-oom", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA not available", file=sys.stderr)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    device = torch.device("cuda")

    # ── Timeline ──
    if not args.scatter_only:
        timeline_codes = args.codes if args.codes else ["138L", "3K7M", "1AQ3"]
        timelines = {}
        for code in timeline_codes:
            print(f"\n{'='*60}")
            print(f"Timeline: {code}")
            print(f"{'='*60}")
            try:
                tl = profile_timeline(code, device)
                timelines[code] = tl
                print(f"  {tl['n_atoms']} atoms, {tl['n_reflections']} refl, {tl['d_min']:.2f} Å")
                print(f"  {'Stage':<30} {'Current':>10} {'Peak':>10}")
                print(f"  {'-'*50}")
                for pt in tl["trace"]:
                    print(f"  {pt['label']:<30} {pt['current_mb']:>8.0f} MB {pt['peak_mb']:>8.0f} MB")
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()

        OUT_TIMELINE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_TIMELINE, "w") as f:
            json.dump(timelines, f, indent=2)
        print(f"\nSaved: {OUT_TIMELINE}")

    if args.timeline_only:
        return

    # ── Scatter ──
    structures = select_structures(args.n_structures, include_oom=args.include_oom)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "code", "n_atoms", "n_reflections", "d_min",
            "peak_memory_gb", "peak_init_gb", "peak_cycle_gb",
        ])
        writer.writeheader()
        for i, info in enumerate(structures):
            code = info["code"]
            print(f"\n[{i+1}/{len(structures)}] {code} "
                  f"({info['n_atoms']} atoms, {info['d_min']:.2f} Å)...")
            result = profile_peak(code, device)
            if result is not None:
                results.append(result)
                writer.writerow(result)
                f.flush()
                print(f"  Peak: {result['peak_memory_gb']:.2f} GB "
                      f"(init={result['peak_init_gb']:.2f}, cycle={result['peak_cycle_gb']:.2f})")
    print(f"\nSaved {len(results)} results to {OUT_CSV}")


if __name__ == "__main__":
    main()
