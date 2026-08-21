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


def _cmp_rank(row: dict, n_peaks_default: int = 500) -> float:
    """Rank for pairing: a miss counts as worse than the worst hit.

    ``compare`` drops pairs where either side missed, which silently removes
    exactly the cases an arm is being blamed for. The sweep writes
    ``rank_for_compare`` for this; fall back to the peak-list length.
    """
    try:
        return float(int(row["rank_for_compare"]))
    except (KeyError, ValueError):
        pass
    try:
        r = int(row["truth_rank"])
    except (KeyError, ValueError):
        return float("nan")
    if r >= 0:
        return float(r)
    try:
        return float(int(row["n_peaks"]))
    except (KeyError, ValueError):
        return float(n_peaks_default)


def gate(rows: List[dict], key: str = "arm", base: str = "production",
         top_n: int = 20, min_hits: int = 9) -> None:
    """Report each arm against the shipping criterion, per structure.

    The criterion is not "truth at rank 0": the pipeline carries the top ~20
    candidates forward, so rank 7 and rank 0 are the same outcome and rank 223
    is not. An arm passes when truth lands in the top ``top_n`` on at least
    ``min_hits`` of the trials for **every** structure -- so one bad structure
    cannot be averaged away by nine good ones.
    """
    arms = sorted({r.get(key, "") for r in rows})
    pdbs = sorted({r.get("pdb", "?") for r in rows})
    per: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        per[(r.get(key, ""), r.get("pdb", "?"))].append(r)

    def _in_top(r: dict) -> bool:
        try:
            v = int(r["truth_rank"])
        except (KeyError, ValueError):
            return False
        return 0 <= v < top_n

    # A structure appearing with more trials than the others means rows were
    # collected twice -- a re-run after a partial failure, say -- and the
    # per-structure hit counts are then not comparable. That has to be loud: it
    # silently flips which arms pass.
    counts = {}
    for arm in arms:
        for pdb in pdbs:
            n = len(per.get((arm, pdb), []))
            if n:
                counts.setdefault(n, []).append(f"{arm}/{pdb}")
    if len(counts) > 1:
        detail = ", ".join(
            f"{n} trials: {len(v)} cell(s) e.g. {v[0]}"
            for n, v in sorted(counts.items()))
        raise SystemExit(
            f"inconsistent trial counts across cells ({detail}). Deduplicate the "
            f"inputs -- comparing 10 trials of one structure against 20 of "
            f"another makes the gate meaningless."
        )

    print(f"\n# shipping gate: truth in the top {top_n} on >= {min_hits} trials, "
          f"for every structure")
    print(f"{key:<26} {'pass':>5} {'worst structure':>16} {'total':>7} "
          f"{'rank0':>6} {'median':>7}")
    verdicts = {}
    for arm in arms:
        worst_pdb, worst_hits, tot, hits, rank0, ranks = None, None, 0, 0, 0, []
        for pdb in pdbs:
            rs = per.get((arm, pdb), [])
            if not rs:
                continue
            h = sum(1 for r in rs if _in_top(r))
            if worst_hits is None or h < worst_hits:
                worst_hits, worst_pdb = h, f"{pdb} {h}/{len(rs)}"
            tot += len(rs); hits += h
            rank0 += sum(1 for r in rs if str(r.get("truth_rank")) == "0")
            ranks += [_cmp_rank(r) for r in rs]
        ok = worst_hits is not None and worst_hits >= min_hits
        verdicts[arm] = ok
        med = statistics.median(ranks) if ranks else float("nan")
        print(f"{arm:<26} {'PASS' if ok else 'fail':>5} {worst_pdb or '-':>16} "
              f"{hits}/{tot:<5} {rank0:>6} {med:>7.1f}")

    # Paired against the shipped configuration, misses included as worst.
    cells: Dict[tuple, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        cells[(r.get("pdb"), r.get("trial"))][r.get(key, "")] = _cmp_rank(r)
    if base not in arms:
        print(f"\n# no {key}={base!r} rows; skipping the paired report")
        return
    print(f"\n# paired against {key}={base!r} over every (structure, trial) cell; "
          f"+ means worse")
    for arm in arms:
        if arm == base:
            continue
        d = [by[arm] - by[base] for by in cells.values()
             if arm in by and base in by]
        if not d:
            print(f"  {arm:<26} no comparable cells")
            continue
        print(f"  {arm:<26} n={len(d):<4} better={sum(x < 0 for x in d):<4} "
              f"same={sum(x == 0 for x in d):<4} worse={sum(x > 0 for x in d):<4} "
              f"median={statistics.median(d):+8.1f}")
    dup = f"{base}_dup"
    if dup in arms:
        d = [by[dup] - by[base] for by in cells.values()
             if dup in by and base in by]
        moved = sum(1 for x in d if x != 0)
        print(f"\n# control: {dup} repeats {base} verbatim. {moved}/{len(d)} cells "
              f"differ -- that is the engine's own spread, and no effect smaller "
              f"than it is resolvable here.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("patterns", nargs="+", help="CSV glob(s)")
    ap.add_argument("--base", default=None,
                    help="which value of --compare is the control arm "
                         "(default: first alphabetically)")
    ap.add_argument("--compare", default=None,
                    help="column whose values are the arms to pair on, "
                         "e.g. obs_mode or lmax_cap")
    ap.add_argument("--gate", action="store_true",
                    help="report each arm against the shipping criterion "
                         "(truth in the top N on most trials, every structure)")
    ap.add_argument("--top-n", type=int, default=20,
                    help="how many candidates the downstream pipeline carries")
    ap.add_argument("--min-hits", type=int, default=9,
                    help="trials per structure that must land in the top N")
    args = ap.parse_args()
    rows = load(args.patterns)
    if args.gate:
        gate(rows, key=args.compare or "arm", base=args.base or "production",
             top_n=args.top_n, min_hits=args.min_hits)
        return 0
    summarise(rows)
    if args.compare:
        compare(rows, args.compare, args.base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
