"""Recompute each finished run's pose against the Phaser placement.

The pose is a property of two coordinate files, not of the search, so a change
to the metric can be applied to runs already on disk instead of repeating the
placement. Reads ``torchref_placed.pdb`` and Phaser's placement, rewrites
``rot_deg`` / ``trans_A`` / ``phaser_chain`` on each chain entry of
``summary.json``, and reports every structure whose verdict moved.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEV = Path("/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper")

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "alignment_lab"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="dmin3.0")
    ap.add_argument("--rot-tol", type=float, default=5.0)
    ap.add_argument("--trans-tol", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    from diagnostics.af_placement import pose_vs_phaser
    from torchref.io.datasets.reflection_data import ReflectionData

    moved = []
    for sfile in sorted((HERE / "runs" / args.tag).glob("*/summary.json")):
        code = sfile.parent.name
        placed = sfile.parent / "torchref_placed.pdb"
        phaser = DEV / "figure2_alphafold_start" / "placed" / f"{code}_af.pdb"
        if not (placed.exists() and phaser.exists()):
            print(f"{code}: no placed file, skipped")
            continue
        d = json.loads(sfile.read_text())
        data = ReflectionData(verbose=0).load_mtz(str(DEV / "data" / code / f"{code}.mtz"))
        pose = pose_vs_phaser(placed, phaser, data)

        live = [c for c in d.get("chains", []) if not c.get("failed")]
        # The pose the run recorded. Summaries written before the harness
        # carried it onto each chain keep it in a parallel list, so read that
        # when the chain entries have none -- otherwise every structure looks
        # like it changed from nothing.
        old_pose = d.get("pose_vs_phaser") or []
        if live and "rot_deg" not in live[0] and len(old_pose) == len(live):
            prior = [(p["rot_deg"], p["trans_A"]) for p in old_pose]
        else:
            prior = [(c.get("rot_deg", 1e9), c.get("trans_A", 1e9)) for c in live]
        before = sum(c["n_res"] for c, (r, t) in zip(live, prior)
                     if r <= args.rot_tol and t <= args.trans_tol)
        for c, row in zip(live, pose):
            c["rot_deg"], c["trans_A"] = row["rot_deg"], row["trans_A"]
            c["phaser_chain"] = row["phaser_chain"]
        after = sum(c["n_res"] for c in live
                    if c["rot_deg"] <= args.rot_tol and c["trans_A"] <= args.trans_tol)
        d["pose_vs_phaser"] = pose
        total = sum(c["n_res"] for c in d.get("chains", [])) or 1
        if before != after:
            moved.append((code, before / total, after / total))
            print(f"{code}: placed {before/total*100:.0f}% -> {after/total*100:.0f}%")
        if not args.dry_run:
            sfile.write_text(json.dumps(d, indent=1))

    print(f"\n{len(moved)} structures changed verdict"
          + (" (dry run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
