"""Aggregate lab result CSVs.

One aggregator, because every row shares the core schema. It reports **paired,
per-trial** differences rather than a bare median of each arm: on this benchmark
the seed-to-seed truth-rank spread at ``lmax_cap=64`` is +-4-6 (1AK5 has been
seen at 9, 11 and 17 for the same configuration), so a difference of medians
over a handful of trials is noise. Three findings that looked strong at n<=7
vanished at full n.

Anything dropped is printed. A silent truncation reads as full coverage.

Usage::

    python alignment_lab/analysis/aggregate.py 'alignment_lab/runs/*.csv'
    python alignment_lab/analysis/aggregate.py 'runs/*.csv' --compare obs_mode
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
from collections import defaultdict
from typing import Dict, List


def load(patterns: List[str]) -> List[dict]:
    """Read every CSV matching the patterns into a list of row dicts."""
    rows: List[dict] = []
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise SystemExit(f"no CSVs matched {patterns}")
    for f in files:
        with open(f, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    print(f"# {len(rows)} rows from {len(files)} file(s)")
    return rows


def _rank(row: dict) -> float:
    """Truth rank as a number; a miss (-1) sorts as worst, not as best."""
    try:
        r = int(row["truth_rank"])
    except (KeyError, ValueError):
        return float("nan")
    return float("inf") if r < 0 else float(r)


def summarise(rows: List[dict]) -> None:
    """Per-structure rank summary, with misses counted separately."""
    by_pdb: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_pdb[r.get("pdb", "?")].append(_rank(r))
    print(f"\n{'pdb':8s} {'n':>3s} {'median':>7s} {'min':>5s} {'max':>5s} "
          f"{'misses':>7s}  per-trial ranks")
    for pdb in sorted(by_pdb):
        vals = by_pdb[pdb]
        finite = [v for v in vals if v != float("inf")]
        misses = sum(1 for v in vals if v == float("inf"))
        med = statistics.median(finite) if finite else float("nan")
        lo = min(finite) if finite else float("nan")
        hi = max(finite) if finite else float("nan")
        shown = ", ".join("miss" if v == float("inf") else f"{int(v)}" for v in vals)
        print(f"{pdb:8s} {len(vals):3d} {med:7.1f} {lo:5.0f} {hi:5.0f} "
              f"{misses:7d}  [{shown}]")
    if any(v == float("inf") for vs in by_pdb.values() for v in vs):
        print("# 'miss' = truth not found in the peak list; excluded from median/min/max")


def compare(rows: List[dict], key: str, base: str = None) -> None:
    """Paired per-(pdb, seed) comparison across the arms of ``key``."""
    arms = sorted({r.get(key, "") for r in rows})
    if len(arms) < 2:
        print(f"\n# only one arm for {key!r}; nothing to pair")
        return
    cells: Dict[tuple, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        cells[(r.get("pdb"), r.get("seed"))][r.get(key, "")] = _rank(r)

    if base is not None and base not in arms:
        raise SystemExit(f"--base {base!r} not among {key} values {arms}")
    base = base if base is not None else arms[0]
    arms = [a for a in arms if a != base]
    print(f"\n# paired against {key}={base!r}; + means rank got worse")
    for arm in arms:
        deltas, unpaired = [], 0
        for (pdb, seed), by_arm in sorted(cells.items()):
            a, b = by_arm.get(base), by_arm.get(arm)
            if a is None or b is None:
                unpaired += 1
                continue
            if a == float("inf") or b == float("inf"):
                unpaired += 1  # a miss has no meaningful numeric difference
                continue
            deltas.append(b - a)
        if not deltas:
            print(f"  {arm:>16s}: no comparable pairs ({unpaired} unpaired)")
            continue
        better = sum(1 for d in deltas if d < 0)
        worse = sum(1 for d in deltas if d > 0)
        same = sum(1 for d in deltas if d == 0)
        print(f"  {arm:>16s}: median delta {statistics.median(deltas):+.1f}  "
              f"(better {better} / worse {worse} / unchanged {same}, n={len(deltas)})"
              + (f"  [{unpaired} pair(s) dropped: missing arm or a miss]" if unpaired else ""))
        if len(deltas) < 10:
            print(f"  {'':16s}  n={len(deltas)} is below the ~10 trials this "
                  f"benchmark needs; treat as indicative only")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("patterns", nargs="+", help="CSV glob(s)")
    ap.add_argument("--base", default=None,
                    help="which value of --compare is the control arm "
                         "(default: first alphabetically)")
    ap.add_argument("--compare", default=None,
                    help="column whose values are the arms to pair on, "
                         "e.g. obs_mode or lmax_cap")
    args = ap.parse_args()
    rows = load(args.patterns)
    summarise(rows)
    if args.compare:
        compare(rows, args.compare, args.base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
