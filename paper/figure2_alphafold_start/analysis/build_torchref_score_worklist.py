#!/usr/bin/env python
"""Build the TorchRef 0-cycle scoring worklist (all arms, all codes).

For every final model (prediction / refmac / phenix / torchref) with input data,
emit a task line

    engine <TAB> code <TAB> model_pdb <TAB> mtz <TAB> out_json

to ``runs/crossscore/torchref_worklist.txt`` for ``torchref_score_array.sh``
(which runs ``analysis/torchref_score.py``). Resume-safe: tasks whose out_json
already holds an ``r_free`` are skipped.

Usage
-----
    ./.dev/bin/python paper/figure2_alphafold_start/analysis/build_torchref_score_worklist.py
"""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE.parent / "data"
RUNS = BASE / "runs"
XDIR = RUNS / "crossscore"
WORKLIST = XDIR / "torchref_worklist.txt"

# engine -> (arm subdir [for the out_json], final-PDB path relative to RUNS)
MODELS = {
    "refmac": ("refmac", "refmac/{code}/refined.pdb"),
    "phenix": ("phenix_norb", "phenix_norb/{code}/{code}_refined_001.pdb"),
    "torchref": ("torchref_devbuild", "torchref_devbuild/{code}/refined.pdb"),
    "prediction": ("af_initial", "../placed/{code}_af.pdb"),
}


def done(js: Path) -> bool:
    if not js.exists():
        return False
    try:
        return "r_free" in json.loads(js.read_text())
    except (OSError, ValueError):
        return False


def main():
    XDIR.mkdir(parents=True, exist_ok=True)
    (XDIR / "slurm_tr").mkdir(exist_ok=True)

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
            out = RUNS / sub / code / "torchref_validate.json"
            if not pdb.exists() or not out.parent.is_dir():
                nomodel += 1
                continue
            if done(out):
                skipped += 1
                continue
            lines.append(f"{eng}\t{code}\t{pdb}\t{mtz}\t{out}")

    WORKLIST.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"codes={len(codes)}  tasks={len(lines)}  already_done={skipped}  "
          f"missing_mtz={nomtz}  missing_model={nomodel}")
    print(f"worklist: {WORKLIST}")
    if lines:
        print(f"submit:  sbatch --array=1-{len(lines)}%60 "
              f"{BASE}/analysis/torchref_score_array.sh")


if __name__ == "__main__":
    main()
