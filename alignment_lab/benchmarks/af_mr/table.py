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
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        # Summaries written before the harness carried the pose onto each
        # chain keep it in a parallel list; line the two up here so the rest
        # of the reader sees one shape.
        chains, poses = d.get("chains", []), d.get("pose_vs_phaser", [])
        if chains and poses and "rot_deg" not in chains[0]:
            for c, p in zip([c for c in chains if not c.get("failed")], poses):
                c["rot_deg"], c["trans_A"] = p["rot_deg"], p["trans_A"]
        out[f.parent.name] = d
    return out


def placement(d, rot_tol, trans_tol):
    """Fraction of residues placed on Phaser's pose, and the worst miss.

    Scored over every component the run attempted, so a component the pipeline
    could not place at all counts against the structure exactly like one it
    placed in the wrong orientation.
    """
    chains = d.get("chains", [])
    if not chains:
        return 0.0, float("inf")
    total = sum(c["n_res"] for c in chains) or 1
    placed_res, worst = 0, 0.0
    for c in chains:
        if c.get("failed") or "rot_deg" not in c:
            worst = float("inf")
            continue
        if c["rot_deg"] <= rot_tol and c["trans_A"] <= trans_tol:
            placed_res += c["n_res"]
        else:
            worst = max(worst, c["rot_deg"])
    return placed_res / total, worst


def compare(args):
    """Two windows, paired on the structures both ran.

    Paired, because the structures differ enormously in difficulty: the median
    of the per-structure differences is the number, not the difference of the
    two medians.
    """
    a, b = load(args.tag), load(args.compare)
    both = sorted(set(a) & set(b))
    if not both:
        raise SystemExit(f"no structure ran under both {args.tag} and {args.compare}")
    hdr = (f"{'code':6s} {'placed A':>9s} {'placed B':>9s} "
           f"{'R-free A':>9s} {'R-free B':>9s} {'B - A':>8s}")
    print(f"A = {args.tag}, B = {args.compare}\n")
    print(hdr)
    print("-" * len(hdr))
    gained, lost, diffs = [], [], []
    for c in both:
        fa, _ = placement(a[c], args.rot_tol, args.trans_tol)
        fb, _ = placement(b[c], args.rot_tol, args.trans_tol)
        ra = (a[c].get("refine_torchref_mr") or {}).get("R_free")
        rb = (b[c].get("refine_torchref_mr") or {}).get("R_free")
        d = (rb - ra) if (ra is not None and rb is not None) else None
        # Only compare R-free where both windows placed the whole structure;
        # elsewhere the number measures the misplacement, not the window.
        if d is not None and fa >= 0.999 and fb >= 0.999:
            diffs.append(d)
        if fb >= 0.999 > fa:
            gained.append(c)
        if fa >= 0.999 > fb:
            lost.append(c)
        print(f"{c:6s} {fa*100:8.0f}% {fb*100:8.0f}% "
              f"{ra if ra is not None else float('nan'):9.4f} "
              f"{rb if rb is not None else float('nan'):9.4f} "
              f"{d if d is not None else float('nan'):+8.4f}")
    print(f"\n{len(both)} structures ran under both")
    print(f"  fully placed by {args.compare} but not {args.tag}: "
          f"{len(gained)} {gained if gained else ''}")
    print(f"  fully placed by {args.tag} but not {args.compare}: "
          f"{len(lost)} {lost if lost else ''}")
    if diffs:
        diffs.sort()
        med = diffs[len(diffs) // 2]
        print(f"  median paired R-free difference over the {len(diffs)} placed by both "
              f"({args.compare} minus {args.tag}): {med:+.4f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="dmin3.0")
    ap.add_argument("--rot-tol", type=float, default=5.0)
    ap.add_argument("--trans-tol", type=float, default=2.0)
    ap.add_argument("--close", type=float, default=0.010,
                    help="R-free within this of the Phaser arm counts as a match")
    ap.add_argument("--compare", metavar="TAG",
                    help="second tag to compare against, paired per structure")
    args = ap.parse_args()

    if args.compare:
        return compare(args)

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
