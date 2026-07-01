#!/usr/bin/env python
"""Aggregate the PHENIX-vs-REFMAC cross-scoring matrix.

For every final model (refmac / phenix / torchref) read the two independent
re-scorings against identical data + free flags:

  REFMAC scorer : runs/<arm>/<code>/validate.log        ("Overall/Free R factor")
  PHENIX scorer : runs/<arm>/<code>/phenix_validate.log (phenix.model_vs_data)

Writes ``runs/metrics/fig_crossscore.csv`` (code, model_engine, scorer, r_work,
r_free, free_frac) and prints the model x scorer median-R-free matrix plus the
home-court deltas. The off-diagonal cells (a model judged by a program that is
*not* its own) are the bias-robust comparison; TorchRef has no "home" scorer here
and is therefore judged only by the two established programs.

Usage
-----
    ./.dev/bin/python paper/figure2_alphafold_start/analysis/aggregate_crossscore.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RUNS = BASE / "runs"
OUT = RUNS / "metrics"
ARM = {"refmac": "refmac", "phenix": "phenix_norb", "torchref": "torchref",
       "prediction": "af_initial"}

RE_RM_W = re.compile(r"Overall R factor\s+=\s+([\d.]+)")
RE_RM_F = re.compile(r"Free R factor\s+=\s+([\d.]+)")
RE_PX_W = re.compile(r"^\s*r_work:\s*([\d.]+)", re.M)   # computed (not header echo)
RE_PX_F = re.compile(r"^\s*r_free:\s*([\d.]+)", re.M)
RE_PX_HDR = re.compile(r"Resolution.*Compl.*Nwork.*Nfree")
RE_PX_ROW = re.compile(r"\s*[\d.]+-[\d.]+\s+[\d.]+\s+(\d+)\s+(\d+)")


def parse_refmac(log: Path):
    if not log.exists():
        return None
    t = log.read_text(errors="replace")
    w, f = RE_RM_W.search(t), RE_RM_F.search(t)
    return (float(w.group(1)), float(f.group(1))) if (w and f) else None


def parse_phenix(log: Path):
    """Return (r_work, r_free, free_frac) from phenix.model_vs_data, or None."""
    if not log.exists():
        return None
    t = log.read_text(errors="replace")
    w, f = RE_PX_W.search(t), RE_PX_F.search(t)
    if not (w and f):
        return None
    nw = nf = 0
    in_tbl = False
    for ln in t.splitlines():
        if RE_PX_HDR.search(ln):
            in_tbl = True
            continue
        if in_tbl:
            m = RE_PX_ROW.match(ln)
            if m:
                nw += int(m.group(1))
                nf += int(m.group(2))
            elif ln.strip().startswith("r_work:"):
                break
    frac = nf / (nw + nf) if (nw + nf) else np.nan
    return float(w.group(1)), float(f.group(1)), frac


def parse_torchref(js: Path):
    """Return (r_work, r_free) from TorchRef 0-cycle scoring JSON, or None."""
    if not js.exists():
        return None
    try:
        d = json.loads(js.read_text())
    except (OSError, ValueError):
        return None
    if "r_work" in d and "r_free" in d:
        return float(d["r_work"]), float(d["r_free"])
    return None


def main():
    # Three independent R-factor scorers, each applied uniformly (fit scaling +
    # bulk solvent only, no coordinate refinement) to EVERY arm's final model on
    # the same data + free set: refmac5 NCYCLES 0, phenix.model_vs_data, and
    # TorchRef -n 0 (torchref_score.py).
    rows = []
    for model_eng, sub in ARM.items():
        d = RUNS / sub
        if not d.is_dir():
            continue
        for code in sorted(p.name for p in d.iterdir() if p.is_dir()):
            rm = parse_refmac(d / code / "validate.log")
            if rm:
                rows.append(dict(code=code, model_engine=model_eng, scorer="refmac",
                                 r_work=rm[0], r_free=rm[1], free_frac=np.nan))
            px = parse_phenix(d / code / "phenix_validate.log")
            if px:
                rows.append(dict(code=code, model_engine=model_eng, scorer="phenix",
                                 r_work=px[0], r_free=px[1], free_frac=px[2]))
            tr = parse_torchref(d / code / "torchref_validate.json")
            if tr:
                rows.append(dict(code=code, model_engine=model_eng, scorer="torchref",
                                 r_work=tr[0], r_free=tr[1], free_frac=np.nan))

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "fig_crossscore.csv", index=False)
    print(f"Wrote {OUT / 'fig_crossscore.csv'}  ({len(df)} rows)\n")

    # sanity: phenix must have used the small (test) free set, not the work set
    bad = df[(df.scorer == "phenix") & (df.free_frac > 0.3)]
    if len(bad):
        print(f"WARNING: {len(bad)} phenix scores have free_frac>0.3 "
              f"(wrong free set?): {sorted(bad.code.unique())[:10]} ...\n")

    # model x scorer median-R-free matrix, paired per model on common codes
    scorers = ["refmac", "phenix", "torchref"]
    print(f"{'':12}" + "".join(f"{'by ' + s:>12}" for s in scorers)
          + f"{'n(paired)':>11}")
    print("-" * 51)
    medians = {}
    for model_eng in ARM:
        sub = df[df.model_engine == model_eng]
        wide = sub.pivot_table(index="code", columns="scorer", values="r_free")
        avail = [s for s in scorers if s in wide.columns]
        if not avail:
            print(f"{model_eng:12}{'(incomplete)':>25}")
            continue
        paired = wide.dropna(subset=avail)
        medians[model_eng] = {s: paired[s].median() for s in avail}
        cells = "".join(f"{medians[model_eng].get(s, float('nan')):>12.4f}"
                        for s in scorers)
        home = " *" if model_eng in scorers else ""  # judged by its own program
        print(f"{model_eng:12}{cells}{len(paired):>11}{home}")
    print("\n  (* row has a 'home-turf' cell: a model judged by its own program)")

    # ranking under each scorer
    print("\nRanking by median R-free (lower = better):")
    for s in scorers:
        order = sorted((m for m in medians if s in medians[m]),
                       key=lambda m: medians[m][s])
        print(f"  by {s:9}: "
              + " < ".join(f"{m}({medians[m][s]:.4f})" for m in order))


if __name__ == "__main__":
    main()
