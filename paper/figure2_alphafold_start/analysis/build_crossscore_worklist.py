#!/usr/bin/env python
"""Build the PHENIX-vs-REFMAC cross-scoring worklist.

For every final model (refmac / phenix / torchref) that has input data, emit one
tab-separated task line

    engine <TAB> code <TAB> model_pdb <TAB> mtz <TAB> out_log

to ``runs/crossscore/worklist.txt`` for consumption by ``crossscore_array.sh``
(which runs ``phenix.model_vs_data``). Tasks whose ``out_log`` already contains a
computed ``r_free:`` are skipped, so the script is resume-safe: re-run it after a
partial array to shrink the worklist to the remaining work.

Usage
-----
    ./.dev/bin/python paper/figure2_alphafold_start/analysis/build_crossscore_worklist.py
"""

import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent      # paper/figure2_alphafold_start
DATA = BASE.parent / "data"
RUNS = BASE / "runs"
XDIR = RUNS / "crossscore"
WORKLIST = XDIR / "worklist.txt"

# model_engine -> (arm subdir [for the phenix_validate.log], final-PDB path
# relative to RUNS). The prediction model lives in ../placed, not under its arm.
MODELS = {
    "refmac": ("refmac", "refmac/{code}/refined.pdb"),
    "phenix": ("phenix_norb", "phenix_norb/{code}/{code}_refined_001.pdb"),
    "torchref": ("torchref_devbuild", "torchref_devbuild/{code}/refined.pdb"),
    # New locked-in default weights (xray=1/geom=0.2/adp=0.005); scored
    # alongside the old torchref_devbuild (xray=10/geom=1/adp=0.1) arm.
    "torchref_locked": ("torchref_locked", "torchref_locked/{code}/refined.pdb"),
    # adp=0.01 variant of the locked weights (1/0.2/0.01), to settle adp 0.005 vs 0.01.
    "torchref_g0p2_a0p01": ("torchref_g0p2_a0p01",
                            "torchref_g0p2_a0p01/{code}/refined.pdb"),
    # New locked default (1/0.2/0.02) — full-pipeline arm.
    "torchref_g0p2_a0p02": ("torchref_g0p2_a0p02",
                            "torchref_g0p2_a0p02/{code}/refined.pdb"),
    "prediction": ("af_initial", "../placed/{code}_af.pdb"),
}
RE_DONE = re.compile(r"^\s*r_free:\s*[\d.]+", re.M)


def done(log: Path) -> bool:
    return log.exists() and bool(RE_DONE.search(log.read_text(errors="replace")))


def main():
    XDIR.mkdir(parents=True, exist_ok=True)
    (XDIR / "slurm").mkdir(exist_ok=True)  # SLURM won't create the log dir itself

    codes = sorted({d.name for sub, _ in MODELS.values()
                    for d in (RUNS / sub).iterdir() if (RUNS / sub).is_dir()
                    and d.is_dir()})

    lines, skipped, nomtz, nomodel = [], 0, 0, 0
    for code in codes:
        mtz = DATA / code / f"{code}.mtz"
        if not mtz.exists():
            nomtz += 1
            continue
        for eng, (sub, patt) in MODELS.items():
            pdb = (RUNS / patt.format(code=code)).resolve()
            if not pdb.exists():
                nomodel += 1
                continue
            log = RUNS / sub / code / "phenix_validate.log"
            if not log.parent.is_dir():
                nomodel += 1
                continue
            if done(log):
                skipped += 1
                continue
            lines.append(f"{eng}\t{code}\t{pdb}\t{mtz}\t{log}")

    WORKLIST.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"codes={len(codes)}  tasks={len(lines)}  already_done={skipped}  "
          f"missing_mtz={nomtz}  missing_model={nomodel}")
    print(f"worklist: {WORKLIST}")
    if lines:
        print(f"submit:  sbatch --array=1-{len(lines)}%60 "
              f"{BASE}/analysis/crossscore_array.sh")


if __name__ == "__main__":
    main()
