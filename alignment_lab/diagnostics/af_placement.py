#!/usr/bin/env python3
"""Place one structure's AlphaFold search model(s) with TorchRef's molecular
replacement, refine the result, and refine the Phaser placement alongside it.

Inputs are the Figure 2 AlphaFold-start arm: the pLDDT->B processed search
models (``search_models/<code>/components.json`` in the review tree), the
reflection file ``paper/data/<code>/<code>.mtz`` and Phaser's top solution
``placed/<code>_af.pdb``. Components are placed largest first, each copy
against everything placed so far; later copies of the same component reuse the
first copy's rotation shortlist (same search frame).

Both arms are refined in this process with the same recipe so the comparison is
paired. Writes ``<out>/torchref_placed.pdb``, ``<out>/summary.json`` and prints
one ``ROW`` line.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lab import pose_error  # noqa: E402

REVIEW = Path("/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/review/paper/figure2_alphafold_start")
DEV = Path("/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper")
REPO = HERE.parent.parent
REFINE = REPO / "torchref" / "cli" / "refine.py"
CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def components(code):
    comps = json.loads((REVIEW / "search_models" / code / "components.json").read_text())
    import gemmi
    out = []
    for c in comps:
        s = gemmi.read_structure(c["processed_pdb"])
        n_res = sum(len(ch) for ch in s[0])
        out.append((n_res, c["acc"], c["processed_pdb"], int(c["copies"])))
    out.sort(key=lambda x: -x[0])
    return out


def load_search(path, device):
    from torchref.model import ModelFT
    m = ModelFT(device=device).load_pdb(str(path))
    m.spacegroup = "P 1"
    # process_predicted_model writes two-character chain ids ("A1"), which the
    # PDB round trip does not survive; the assembled solution renames anyway.
    m.pdb["chainid"] = "A"
    return m


def assemble(placed, cell, spacegroup_hm, out_path, tmp_dir):
    """Write every placed copy and merge them into one PDB with chains A, B, ..."""
    import gemmi
    merged = gemmi.Structure()
    merged.cell = gemmi.UnitCell(*cell)
    merged.spacegroup_hm = spacegroup_hm
    model = gemmi.Model("1")
    k = 0
    for i, m in enumerate(placed):
        p = tmp_dir / f"placed_{i}.pdb"
        m.write_pdb(str(p))
        s = gemmi.read_structure(str(p))
        for ch in s[0]:
            ch.name = CHAIN_IDS[k]
            k += 1
            model.add_chain(ch)
    merged.add_model(model)
    merged.remove_hydrogens()   # the loader adds riding H; Phaser's file has none
    merged.setup_entities()
    merged.write_pdb(str(out_path))


def ca_xyz_by_chain(path):
    import gemmi
    import torch
    s = gemmi.read_structure(str(path))
    out = []
    for ch in s[0]:
        xyz = [[a.pos.x, a.pos.y, a.pos.z] for r in ch for a in r if a.name == "CA"]
        out.append((ch.name, torch.tensor(xyz)))
    return out


def pose_vs_phaser(ours_pdb, phaser_pdb, data):
    """For each of our chains, the closest Phaser chain of the same length
    (rotation deg, translation A) allowing symmetry and origin freedom."""
    from torchref.config import get_float_dtype
    dtype = get_float_dtype()
    ours, theirs = ca_xyz_by_chain(ours_pdb), ca_xyz_by_chain(phaser_pdb)
    rows = []
    for name, xyz in ours:
        best = (None, float("inf"), float("inf"))
        for tname, txyz in theirs:
            if txyz.shape != xyz.shape:
                continue
            r, t = pose_error(xyz.to(dtype), txyz.to(dtype), data.cell,
                              data.spacegroup, allow_origin_freedom=True)
            if t < best[2]:
                best = (tname, r, t)
        rows.append({"chain": name, "phaser_chain": best[0], "rot_deg": best[1], "trans_A": best[2]})
    return rows


def refine(pdb, mtz, outdir, n_cycles, python):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [python, "-u", str(REFINE), "-m", str(pdb), "-sf", str(mtz), "-o", str(outdir),
           "-n", str(n_cycles), "--mode", "separate", "--xray-mode", "ml",
           "--weights", '{"adp": 0.02}']
    env = dict(os.environ, PYTHONPATH=str(REPO))
    t0 = time.time()
    with open(outdir / "refine.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    hist = outdir / "refinement_history.json"
    stats = {}
    if rc == 0 and hist.exists():
        d = json.loads(hist.read_text())
        stats = d.get("final_statistics", {})
        hist_d = d.get("history") or {}
        cycles = next(iter(hist_d.values()), []) if isinstance(hist_d, dict) else hist_d
        first = (cycles[0].get("before_scaling", {}) if cycles else {})
        stats["initial"] = {"R_work": first.get("rwork"), "R_free": first.get("rfree")}
    return {"rc": rc, "seconds": time.time() - t0, **stats}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("code")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-rotation-candidates", type=int, default=10)
    ap.add_argument("--n-cycles", type=int, default=10)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()
    code = args.code
    out = Path(args.out or HERE.parent / "runs" / "af_mr" / code)
    out.mkdir(parents=True, exist_ok=True)

    from torchref.io.datasets.reflection_data import ReflectionData
    from torchref.experimental.alignment import MolecularReplacementPipeline

    mtz = DEV / "data" / code / f"{code}.mtz"
    phaser_pdb = DEV / "figure2_alphafold_start" / "placed" / f"{code}_af.pdb"
    data = ReflectionData(device=args.device).load_mtz(str(mtz))
    comps = components(code)

    placed, chains, t_total = [], [], time.time()
    for n_res, acc, path, copies in comps:
        search = load_search(path, args.device)
        candidates = None
        for k in range(copies):
            t0 = time.time()
            pipe = MolecularReplacementPipeline(
                data, search, d_min=4.0, d_max=15.0, n_shells=20, n_rotation_peaks=200,
                n_rotation_candidates=args.n_rotation_candidates, verbose=args.verbose,
                fixed=list(placed))
            sols = pipe.run(do_translation=True, candidates=candidates)
            candidates = pipe.rotation_candidates
            s = sols[0]
            placed.append(pipe.place(s))
            chains.append({"acc": acc, "copy": k, "n_res": n_res, "llg": float(s.llg_score),
                           "llg_next": float(sols[1].llg_score) if len(sols) > 1 else None,
                           "tf": float(s.translation_score), "r_factor": float(s.r_factor),
                           "clash": float(getattr(s, "clash_fraction", 0.0) or 0.0),
                           "seconds": time.time() - t0})
            print(f"PLACED {code} {acc} copy={k} n_res={n_res} llg={s.llg_score:.0f} "
                  f"r={s.r_factor:.3f} clash={chains[-1]['clash']:.2f} "
                  f"seconds={chains[-1]['seconds']:.1f}", flush=True)
    seconds_place = time.time() - t_total

    ours = out / "torchref_placed.pdb"
    assemble(placed, [float(x) for x in data.cell.data.tolist()], data.spacegroup.hm, ours, out)
    pose = pose_vs_phaser(ours, phaser_pdb, data)
    summary = {"code": code, "spacegroup": data.spacegroup.hm, "n_chains": len(placed),
               "seconds_place": seconds_place, "chains": chains, "pose_vs_phaser": pose}

    if not args.no_refine:
        summary["refine_torchref_mr"] = refine(ours, mtz, out / "refine_torchref_mr",
                                               args.n_cycles, sys.executable)
        summary["refine_phaser_mr"] = refine(phaser_pdb, mtz, out / "refine_phaser_mr",
                                             args.n_cycles, sys.executable)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    worst_t = max(p["trans_A"] for p in pose)
    worst_r = max(p["rot_deg"] for p in pose)
    rt = summary.get("refine_torchref_mr", {})
    rp = summary.get("refine_phaser_mr", {})
    print(f"ROW code={code} sg='{data.spacegroup.hm}' n_chains={len(placed)} "
          f"place_s={seconds_place:.1f} pose_rot={worst_r:.1f} pose_trans={worst_t:.2f} "
          f"llg={chains[0]['llg']:.0f} "
          f"rfree_ours={rt.get('R_free', float('nan')):.4f} rfree_phaser={rp.get('R_free', float('nan')):.4f} "
          f"rwork_ours={rt.get('R_work', float('nan')):.4f} rwork_phaser={rp.get('R_work', float('nan')):.4f} "
          f"rc={rt.get('rc', -1)},{rp.get('rc', -1)}", flush=True)


if __name__ == "__main__":
    main()
