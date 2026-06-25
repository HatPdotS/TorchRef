#!/usr/bin/env python3
"""Collect exF4 single-core wall-clock timings into data/exF4_singlecore.csv.

Walks ``runs/{program}/{code}/n1/timing.txt`` (written by submit_singlecore.py:
``wall_s <float>`` + ``rc <int>``) for the n=715 conserved codes. Cells with a missing
timing or nonzero rc are reported and skipped.

Only structures in the CONSERVED-AT-1-CORE set — those where all three programs
(torchref/phenix/refmac) succeeded at 1 core — are written, so the per-program box-plot
distributions are over the same structures (matching the Figure-2c convention). The kept
codes are written to data/singlecore_conserved.txt.

Output columns: program,code,wall_s.

Usage
-----
    ./.dev/bin/python paper/extended_figures/exF4/aggregate_singlecore.py
"""
from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
CODES_TXT = HERE / "codes_conserved.txt"
OUT_CSV = HERE / "data" / "exF4_singlecore.csv"
CONSERVED_TXT = HERE / "data" / "singlecore_conserved.txt"

PROGRAMS = ["torchref", "phenix", "refmac"]
LABEL = "n1"
RE_WALL = re.compile(r"wall_s\s+([\d.]+)")
RE_RC = re.compile(r"rc\s+(-?\d+)")


def parse_timing(path: Path):
    """wall_s (float) if the cell completed with rc 0, else None."""
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    mw, mr = RE_WALL.search(text), RE_RC.search(text)
    if not mw:
        return None
    if mr and int(mr.group(1)) != 0:
        return None
    return float(mw.group(1))


def main():
    codes = [c.strip() for c in CODES_TXT.read_text().split() if c.strip()]

    wall = {}          # (program, code) -> wall_s
    missing, failed = [], []
    for program in PROGRAMS:
        for code in codes:
            tpath = RUNS / program / code / LABEL / "timing.txt"
            w = parse_timing(tpath)
            if w is None:
                (missing if not tpath.exists() else failed).append(f"{program}/{code}")
                continue
            wall[(program, code)] = w

    # Conserved-at-1-core: every program produced a successful timing.
    conserved = [c for c in codes if all((p, c) in wall for p in PROGRAMS)]
    CONSERVED_TXT.parent.mkdir(parents=True, exist_ok=True)
    CONSERVED_TXT.write_text("\n".join(conserved) + "\n")

    rows = [{"program": p, "code": c, "wall_s": f"{wall[(p, c)]:.3f}"}
            for p in PROGRAMS for c in conserved]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["program", "code", "wall_s"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {OUT_CSV}: {len(rows)} rows over {len(conserved)} conserved structures "
          f"(of {len(codes)} requested)")
    if failed:
        print(f"  FAILED rc!=0 ({len(failed)}): {', '.join(failed[:8])}"
              + (" ..." if len(failed) > 8 else ""))
    if missing:
        print(f"  not-yet-run ({len(missing)}): {', '.join(missing[:8])}"
              + (" ..." if len(missing) > 8 else ""))

    # median + paired ratios over the conserved set (the Fig-2c annotation numbers)
    if conserved:
        med = {p: statistics.median([wall[(p, c)] for c in conserved]) for p in PROGRAMS}
        print("\nmedian wall-clock (s), conserved set:")
        for p in PROGRAMS:
            print(f"  {p:10s} {med[p]:8.1f}s  ({med[p]/60:.2f} min)")
        r_ref = statistics.median([wall[("torchref", c)] / wall[("refmac", c)]
                                   for c in conserved])
        r_phe = statistics.median([wall[("torchref", c)] / wall[("phenix", c)]
                                   for c in conserved])
        print(f"\n  TorchRef vs Refmac: {r_ref:.1f}x slower")
        print(f"  TorchRef vs PHENIX: {1/r_phe:.1f}x faster (ratio {r_phe:.2f})")


if __name__ == "__main__":
    main()
