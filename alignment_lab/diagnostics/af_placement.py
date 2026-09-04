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


def _cost(rt, rot_tol, trans_tol):
    """Pairing cost: each error as a fraction of its own tolerance, summed."""
    r, t = rt
    return r / rot_tol + t / trans_tol


def pose_vs_phaser(ours_pdb, phaser_pdb, data, rot_tol=5.0, trans_tol=2.0):
    """For each of our chains, the pose difference to the matching Phaser copy
    (rotation deg, translation A) allowing symmetry and origin freedom.

    Phaser writes a chain break as a new chain, so its chains are grouped in
    file order into copies whose C-alpha counts add up to ours; the copy with
    the smallest translation error is reported.
    """
    import torch
    from torchref.config import get_float_dtype
    dtype = get_float_dtype()
    ours, theirs = ca_xyz_by_chain(ours_pdb), ca_xyz_by_chain(phaser_pdb)
    counts = sorted({xyz.shape[0] for _, xyz in ours}, reverse=True)
    groups, i = [], 0
    while i < len(theirs):
        for n in counts:
            acc, j = 0, i
            while j < len(theirs) and acc < n:
                acc += theirs[j][1].shape[0]
                j += 1
            if acc == n:
                groups.append(("+".join(t[0] for t in theirs[i:j]),
                               torch.cat([t[1] for t in theirs[i:j]])))
                i = j
                break
        else:
            i += 1   # a Phaser chain no copy of ours accounts for
    # Each of our chains against each Phaser copy of the same size. A
    # homodimer's sites are indistinguishable, so our chain A matching their
    # copy B is a correct answer -- but a copy can only be matched ONCE, or two
    # of our chains both claim the copy nearest them and the other copy's site
    # is scored as a miss.
    #
    # The pairing is chosen on rotation AND translation, each scaled by its own
    # tolerance. Translation alone does not discriminate: with origin freedom
    # allowed it is 0.00 A for every pair in a P1 cell (2XN4, two copies of one
    # 286-residue chain: all four pairings 0.00 A, rotations 2.6, 59, 177, 179
    # degrees), so a translation-ranked assignment picks among them at random.
    err = {}
    for i, (_, xyz) in enumerate(ours):
        for j, (_, gxyz) in enumerate(groups):
            if gxyz.shape != xyz.shape:
                continue
            err[i, j] = pose_error(xyz.to(dtype), gxyz.to(dtype), data.cell,
                                   data.spacegroup, allow_origin_freedom=True)

    n, m = len(ours), len(groups)
    best_assign, best_cost = {}, float("inf")
    if n <= 7 and m <= 7:
        # Exhaustive over injective assignments: the counts here are single
        # digits, and greedy can pick a pair that strands a better one.
        import itertools
        for perm in itertools.permutations(range(m), min(n, m)):
            cand, cost = {}, 0.0
            for i, j in enumerate(perm):
                if (i, j) not in err:
                    cost = float("inf")
                    break
                cand[i] = j
                cost += _cost(err[i, j], rot_tol, trans_tol)
            if cost < best_cost:
                best_assign, best_cost = cand, cost
    if not best_assign:
        # Fall back to greedy on the global minimum, each copy used once.
        taken, remaining = set(), sorted(
            err, key=lambda k: _cost(err[k], rot_tol, trans_tol))
        for i, j in remaining:
            if i not in best_assign and j not in taken:
                best_assign[i] = j
                taken.add(j)

    rows = []
    for i, (name, _) in enumerate(ours):
        j = best_assign.get(i)
        if j is None:
            rows.append({"chain": name, "phaser_chain": None,
                         "rot_deg": float("inf"), "trans_A": float("inf")})
            continue
        r, t = err[i, j]
        rows.append({"chain": name, "phaser_chain": groups[j][0],
                     "rot_deg": r, "trans_A": t})
    return rows


def refine(pdb, mtz, outdir, n_cycles, python):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [python, "-u", str(REFINE), "-m", str(pdb), "-sf", str(mtz), "-o", str(outdir),
           "-n", str(n_cycles), "--mode", "separate", "--xray-mode", "ml",
           "--weights", '{"adp": 0.02}', "--with-rigid-body"]
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
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=15.0)
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
                data, search, d_min=args.d_min, d_max=args.d_max, n_shells=20, n_rotation_peaks=200,
                n_rotation_candidates=args.n_rotation_candidates, verbose=args.verbose,
                fixed=list(placed))
            # A component the pipeline cannot place -- no translation peak
            # survives the packing rejection, say -- is a result, not a crash:
            # record it and keep the components that did place, so one bad
            # component does not cost the whole structure.
            try:
                sols = pipe.run(do_translation=True, candidates=candidates)
            except Exception as exc:
                chains.append({"acc": acc, "copy": k, "n_res": n_res,
                               "failed": True, "error": f"{type(exc).__name__}: {exc}",
                               "seconds": time.time() - t0})
                print(f"UNPLACED {code} {acc} copy={k} n_res={n_res} "
                      f"error={type(exc).__name__}: {exc}", flush=True)
                break
            candidates = pipe.rotation_candidates
            s = sols[0]
            placed.append(pipe.place(s))
            chains.append({"acc": acc, "copy": k, "n_res": n_res, "failed": False,
                           "llg": float(s.llg_score),
                           "llg_next": float(sols[1].llg_score) if len(sols) > 1 else None,
                           "tf": float(s.translation_score), "r_factor": float(s.r_factor),
                           "clash": float(getattr(s, "clash_fraction", 0.0) or 0.0),
                           "seconds": time.time() - t0})
            print(f"PLACED {code} {acc} copy={k} n_res={n_res} llg={s.llg_score:.0f} "
                  f"r={s.r_factor:.3f} clash={chains[-1]['clash']:.2f} "
                  f"seconds={chains[-1]['seconds']:.1f}", flush=True)
    if not placed:
        raise SystemExit(f"{code}: no component could be placed")
    seconds_place = time.time() - t_total
    # Warm timing: the first placement again, with every cache and kernel built.
    n_res, acc, path, copies = comps[0]
    search = load_search(path, args.device)
    t0 = time.time()
    pipe = MolecularReplacementPipeline(
        data, search, d_min=args.d_min, d_max=args.d_max, n_shells=20, n_rotation_peaks=200,
        n_rotation_candidates=args.n_rotation_candidates, verbose=0)
    pipe.run(do_translation=True)
    seconds_warm = time.time() - t0

    ours = out / "torchref_placed.pdb"
    assemble(placed, [float(x) for x in data.cell.data.tolist()], data.spacegroup.hm, ours, out)
    pose = pose_vs_phaser(ours, phaser_pdb, data)
    # `pose` has one row per chain of the assembled file, in the order the
    # components were placed; the failed ones contributed no chain.
    for entry, row in zip([c for c in chains if not c.get("failed")], pose):
        entry["rot_deg"] = row["rot_deg"]
        entry["trans_A"] = row["trans_A"]
        entry["phaser_chain"] = row["phaser_chain"]
    summary = {"code": code, "spacegroup": data.spacegroup.hm, "n_chains": len(placed),
               "d_min": args.d_min, "d_max": args.d_max,
               "n_rotation_candidates": args.n_rotation_candidates,
               "n_cycles": args.n_cycles,
               "seconds_place": seconds_place, "seconds_warm": seconds_warm, "chains": chains, "pose_vs_phaser": pose}

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
    print(f"ROW code={code} dmin={args.d_min} sg='{data.spacegroup.hm}' n_chains={len(placed)} "
          f"place_s={seconds_place:.1f} warm_s={seconds_warm:.1f} pose_rot={worst_r:.1f} pose_trans={worst_t:.2f} "
          f"llg={chains[0]['llg']:.0f} "
          f"rfree_ours={rt.get('R_free', float('nan')):.4f} rfree_phaser={rp.get('R_free', float('nan')):.4f} "
          f"rwork_ours={rt.get('R_work', float('nan')):.4f} rwork_phaser={rp.get('R_work', float('nan')):.4f} "
          f"rc={rt.get('rc', -1)},{rp.get('rc', -1)}", flush=True)


if __name__ == "__main__":
    main()
