#!/usr/bin/env python3
"""Write worklist.txt = the af_complete codes that still need a placed model.

A code is included unless `placed/{code}_af.pdb` already exists, so re-running the
Phaser array only reprocesses unfinished/failed codes (resumable).
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
PLACED = HERE / "placed"
WORKLIST = HERE / "worklist.txt"


def main():
    records = json.loads(MANIFEST.read_text())
    af_codes = [r["code"] for r in records if r.get("af_complete")]
    todo = [c for c in af_codes if not (PLACED / f"{c}_af.pdb").exists()]

    WORKLIST.write_text("\n".join(todo) + ("\n" if todo else ""))
    print(f"af_complete: {len(af_codes)}")
    print(f"already placed: {len(af_codes) - len(todo)}")
    print(f"worklist (to do): {len(todo)} -> {WORKLIST}")
    if todo:
        print(f"submit array with: --array=0-{len(todo) - 1}%50")


if __name__ == "__main__":
    main()
