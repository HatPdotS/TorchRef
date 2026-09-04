"""Read the AlphaFold molecular-replacement benchmark into a table.

Two questions, kept apart because they fail independently:

* **Placement** -- did each component land on Phaser's pose? A component counts
  as placed within ``--rot-tol`` degrees and ``--trans-tol`` Angstrom of the
  Phaser copy it matches; ``placed`` reports the fraction of residues placed,
  so a structure whose large chains are right and whose small fragment is
  missed is not scored the same as one that failed outright.
* **Refinement** -- after the same recipe, how far is our R-free from the
  refinement of Phaser's placement? That comparison is only meaningful where
  the placement is right, so the summary counts those separately.

Reads ``runs/<tag>/<code>/summary.json``, which the harness writes.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(tag):
    out = {}
    for f in sorted((HERE / "runs" / tag).glob("*/summary.json")):
        try:
            out[f.parent.name] = json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    return out


def placement(d, rot_tol, trans_tol):
    """Fraction of residues placed on Phaser's pose, and the worst miss."""
    poses, chains = d.get("pose_vs_phaser", []), d.get("chains", [])
    if not poses or len(poses) != len(chains):
        return 0.0, float("inf")
    ok = [p["rot_deg"] <= rot_tol and p["trans_A"] <= trans_tol for p in poses]
    sizes = [c["n_res"] for c in chains]
    total = sum(sizes) or 1
    worst = max((p["rot_deg"] for p, k in zip(poses, ok) if not k), default=0.0)
    return sum(s for s, k in zip(sizes, ok) if k) / total, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="dmin3.0")
    ap.add_argument("--rot-tol", type=float, default=5.0)
    ap.add_argument("--trans-tol", type=float, default=2.0)
    ap.add_argument("--close", type=float, default=0.010,
                    help="R-free within this of the Phaser arm counts as a match")
    args = ap.parse_args()

    runs = load(args.tag)
    if not runs:
        raise SystemExit(f"no runs under {HERE / 'runs' / args.tag}")

    hdr = (f"{'code':6s} {'spacegroup':13s} {'placed':>7s} {'worst':>7s} "
           f"{'ours':>7s} {'phaser':>7s} {'diff':>8s} {'place_s':>8s}")
    print(hdr)
    print("-" * len(hdr))
    n_full = n_partial = n_failed = 0
    n_close = n_behind = n_ahead = 0
    diffs = []
    for code in sorted(runs):
        d = runs[code]
        frac, worst = placement(d, args.rot_tol, args.trans_tol)
        ours = (d.get("refine_torchref_mr") or {}).get("R_free")
        phas = (d.get("refine_phaser_mr") or {}).get("R_free")
        if frac >= 0.999:
            n_full += 1
        elif frac >= 0.5:
            n_partial += 1
        else:
            n_failed += 1
        diff = (ours - phas) if (ours is not None and phas is not None) else None
        if diff is not None and frac >= 0.999:
            diffs.append(diff)
            if diff < -args.close:
                n_ahead += 1
            elif diff <= args.close:
                n_close += 1
            else:
                n_behind += 1
        print(f"{code:6s} {d.get('spacegroup',''):13s} {frac*100:6.0f}% "
              f"{worst:7.1f} "
              f"{ours if ours is not None else float('nan'):7.4f} "
              f"{phas if phas is not None else float('nan'):7.4f} "
              f"{diff if diff is not None else float('nan'):+8.4f} "
              f"{d.get('seconds_warm', float('nan')):8.1f}")

    n = len(runs)
    print(f"\n{n} structures at {args.tag} "
          f"({d.get('d_max', '?')}-{d.get('d_min', '?')} A)")
    print(f"  placement: {n_full} every component on Phaser's pose, "
          f"{n_partial} large chains only, {n_failed} failed")
    print(f"  refinement, over the {n_full} fully placed: {n_ahead} better than the "
          f"Phaser arm by more than {args.close:.3f}, {n_close} within it, "
          f"{n_behind} worse")
    if diffs:
        diffs.sort()
        med = diffs[len(diffs) // 2]
        print(f"  median R-free difference (ours minus Phaser): {med:+.4f}")


if __name__ == "__main__":
    main()
