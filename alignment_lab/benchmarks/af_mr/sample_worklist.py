"""Draw the benchmark's structure sample.

A structure is usable when it has a Phaser placement to judge against, a
components manifest to search with, and its own data. The sample is a plain
uniform draw from those, seeded, so the worklist can be rebuilt exactly;
``worklist.txt`` is committed and is what the benchmark actually runs.
"""
import argparse
import glob
import os
import random

REVIEW = ("/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/"
          "review/paper/figure2_alphafold_start")
DEV = "/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/paper"


def usable():
    placed = {os.path.basename(p)[:-7]
              for p in glob.glob(f"{DEV}/figure2_alphafold_start/placed/*_af.pdb")}
    comps = {os.path.basename(os.path.dirname(p))
             for p in glob.glob(f"{REVIEW}/search_models/*/components.json")}
    data = {os.path.basename(os.path.dirname(p))
            for p in glob.glob(f"{DEV}/data/*/*.mtz")}
    return sorted(placed & comps & data)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()
    pool = usable()
    sample = sorted(random.Random(args.seed).sample(pool, args.n))
    print(f"# {args.n} of {len(pool)} usable structures, seed {args.seed}")
    for c in sample:
        print(c)


if __name__ == "__main__":
    main()
