#!/usr/bin/env python3
"""Summarize Phaser-MR outcomes into mr_status.csv.

For each af_complete code: whether `placed/{code}_af.pdb` exists (solved) and the
top-solution LLG / TFZ parsed from `search_models/{code}/{code}_phaser.sol`.
The solved subset defines the final AlphaFold-start benchmark arm.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
SEARCH = HERE / "search_models"
PLACED = HERE / "placed"
OUT = HERE / "mr_status.csv"

# Last LLG=… and TFZ==… on the SOLU SET line are the top solution's values.
_LLG = re.compile(r"LLG=([-\d.]+)")
_TFZ = re.compile(r"TFZ==([-\d.]+)")


def _llg_tfz(code):
    sol = SEARCH / code / f"{code}_phaser.sol"
    if not sol.exists():
        return None, None
    for line in sol.read_text().splitlines():
        if line.startswith("SOLU SET"):
            llg = _LLG.findall(line)
            tfz = _TFZ.findall(line)
            return (float(llg[-1]) if llg else None,
                    float(tfz[-1]) if tfz else None)
    return None, None


def main():
    records = json.loads(MANIFEST.read_text())
    af_codes = [r["code"] for r in records if r.get("af_complete")]

    n_solved = 0
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "solved", "llg", "tfz"])
        for code in sorted(af_codes):
            solved = (PLACED / f"{code}_af.pdb").exists()
            llg, tfz = _llg_tfz(code)
            n_solved += solved
            w.writerow([code, int(solved), llg if llg is not None else "",
                        tfz if tfz is not None else ""])

    print(f"af_complete: {len(af_codes)}")
    print(f"solved (placed): {n_solved}")
    print(f"unsolved: {len(af_codes) - n_solved}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
