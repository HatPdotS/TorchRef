#!/usr/bin/env python
"""Aggregate per-structure metrics for the AlphaFold-start benchmark figure.

Parses the per-structure log files written by ``run_af_pipeline.py`` into four
tidy CSVs under ``runs/metrics/``:

  fig_rfactors.csv  code,engine,r_work,r_free
      PHENIX *validated* R-factors (phenix.model_vs_data, the apples-to-apples
      metric), from each arm's ``phenix_validate.log``. (REFMAC NCYCLES-0 scores
      are still on disk in ``validate.log``; cross-scoring is in fig_crossscore.csv.)
  fig_geometry.csv  code,engine,rmsBOND,sigBOND,rmsANGL,sigANGL,
                    rmsCHIRAL,sigCHIRAL,rmsB_mc_bond,sigB_mc_bond
      Geometry RMS deltas and Av(Sigma) from the same ``validate.log`` restraint
      table (uniform REFMAC restraints across all engines -> fair RMSZ).
  fig_runtime.csv   code,engine,wall_s
      Wall-clock seconds per refinement (program-reported).
  fig_percycle.csv  code,engine,cycle,r_work,r_free
      *Program-reported* per-cycle R-factors (validation only runs on the final
      model, so these necessarily come from each program's own log).

Engines use the no-rigid-body protocol:
  prediction = af_initial, refmac, phenix = phenix_norb,
  torchref   = torchref_scalerfix_nocoref_n10 (the current default config).

Usage
-----
    ./.dev/bin/python paper/figure2_alphafold_start/analysis/aggregate_figure_metrics.py
"""

import json
import re
from pathlib import Path

import pandas as pd

# ── Paths / engine layout ─────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent              # paper/figure2_alphafold_start
RUNS = BASE / "runs"
OUT = RUNS / "metrics"

# engine -> arm subdirectory under runs/
ENGINE_DIR = {
    "prediction": "af_initial",
    "refmac": "refmac",
    "phenix": "phenix_norb",
    "torchref": "torchref_devbuild",
}


def struct_dir(engine: str, code: str) -> Path:
    return RUNS / ENGINE_DIR[engine] / code


# ── validate.log: validated R-factors + geometry (all engines, same format) ──
RE_RWORK = re.compile(r"Overall R factor\s+=\s+([\d.]+)")
RE_RFREE = re.compile(r"Free R factor\s+=\s+([\d.]+)")
# restraint-table row: "<label> ... <Nrestraints:int> <RmsDelta:float> <AvSigma:float>"
RE_TAIL3 = re.compile(r"(\d+)\s+([\d.]+)\s+([\d.]+)\s*$")

# restraint-table label prefix -> (rms_col, sig_col)
GEOM_ROWS = {
    "Bond distances": ("rmsBOND", "sigBOND"),
    "Bond angles": ("rmsANGL", "sigANGL"),
    "Chiral centres": ("rmsCHIRAL", "sigCHIRAL"),
    "M. chain bond B values": ("rmsB_mc_bond", "sigB_mc_bond"),
}


# PHENIX (phenix.model_vs_data) is the primary R-factor scorer: independent of
# TorchRef (the engine under test), no significant home-turf effect in the
# cross-scoring (Fig. supp), and it scales difficult datasets that refmac5
# NCYCLES 0 fails on (fewer outliers). Geometry RMSZ stays on the uniform REFMAC
# restraint table (a restraint-library choice, unaffected by the scorer switch).
RE_PX_RWORK = re.compile(r"^\s*r_work:\s*([\d.]+)", re.M)   # computed, not header echo
RE_PX_RFREE = re.compile(r"^\s*r_free:\s*([\d.]+)", re.M)


def parse_phenix_rfactors(path: Path):
    """R-work/R-free dict from phenix.model_vs_data log, or None."""
    if not path.exists():
        return None
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    mw, mf = RE_PX_RWORK.search(text), RE_PX_RFREE.search(text)
    if mw and mf:
        return {"r_work": float(mw.group(1)), "r_free": float(mf.group(1))}
    return None


def parse_geometry(path: Path):
    """Geometry RMS-delta / Av(Sigma) dict from the REFMAC validate.log, or None."""
    if not path.exists():
        return None
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    geom = {}
    for line in text.splitlines():
        for label, (rms_col, sig_col) in GEOM_ROWS.items():
            if line.startswith(label):
                m = RE_TAIL3.search(line)
                if m:
                    geom[rms_col] = float(m.group(2))
                    geom[sig_col] = float(m.group(3))
    return geom or None


# ── runtime (program-reported wall-clock, seconds) ───────────────────────────
RE_TORCHREF_WALL = re.compile(r"Timing:\s+([\d.]+)s wall")
RE_PHENIX_EPOCH = re.compile(r"#\s*Date .*?\(([\d.]+)\s*s\)")
RE_REFMAC_ELAPSED = re.compile(r"Elapsed:\s+(\d+):(\d+)")


def runtime_torchref(code: str):
    p = struct_dir("torchref", code) / "out.log"
    if not p.exists():
        return None
    m = None
    for m in RE_TORCHREF_WALL.finditer(p.read_text(errors="replace")):
        pass  # keep last
    return float(m.group(1)) if m else None


def runtime_phenix(code: str):
    p = struct_dir("phenix", code) / f"{code}_refined_001.log"
    if not p.exists():
        return None
    epochs = [float(x) for x in RE_PHENIX_EPOCH.findall(p.read_text(errors="replace"))]
    return (epochs[-1] - epochs[0]) if len(epochs) >= 2 else None


def runtime_refmac(code: str):
    p = struct_dir("refmac", code) / "refmac.log"
    if not p.exists():
        return None
    m = RE_REFMAC_ELAPSED.search(p.read_text(errors="replace"))
    return (int(m.group(1)) * 60 + int(m.group(2))) if m else None


RUNTIME = {
    "torchref": runtime_torchref,
    "phenix": runtime_phenix,
    "refmac": runtime_refmac,
}


# ── per-cycle program-reported R-factors ─────────────────────────────────────
def percycle_refmac(code: str):
    """Refmac Ncyc table: rows '<cyc> <Rfact> <Rfree> ...' until '$$'."""
    p = struct_dir("refmac", code) / "refmac.log"
    if not p.exists():
        return []
    lines = p.read_text(errors="replace").splitlines()
    out, in_tbl = [], False
    for ln in lines:
        if "Ncyc" in ln and "rmsBOND" in ln:
            in_tbl = True
            continue
        if in_tbl:
            toks = ln.split()
            if len(toks) >= 3 and toks[0].isdigit():
                out.append((int(toks[0]), float(toks[1]), float(toks[2])))
            elif "$$" in ln and out:
                break
    return out


def percycle_phenix(code: str):
    """Phenix per-macrocycle R-FACTORS (work/free in %) from the 'bonds angl' blocks.

    Each block's *first* row is the state at the start of that macrocycle and the
    *last* row is the post-(geometry-)refinement state. Per Phenix's own legend,
    the very first block's first row is the pre-refinement starting state, which
    we emit as cycle 0 (matching Refmac's Ncyc row 0) so the start R-factor is not
    silently dropped.
    """
    p = struct_dir("phenix", code) / f"{code}_refined_001.log"
    if not p.exists():
        return []
    lines = p.read_text(errors="replace").splitlines()
    blocks, collecting, first, last = [], False, None, None
    for ln in lines:
        low = ln.lower()
        if "work" in low and "free" in low and "bonds" in low and "angl" in low:
            collecting, first, last = True, None, None
            continue
        if collecting:
            toks = ln.split()
            row = None
            if len(toks) >= 5:
                try:
                    row = (float(toks[0]) / 100.0, float(toks[1]) / 100.0)
                except ValueError:
                    row = None
            if row is not None:
                if first is None:
                    first = row
                last = row
            else:
                if last is not None:
                    blocks.append((first, last))
                collecting, first, last = False, None, None
    if last is not None:
        blocks.append((first, last))
    if not blocks:
        return []
    # cycle 0 = pre-refinement start, then end-of-each-macrocycle
    series = [blocks[0][0]] + [b[1] for b in blocks]
    return [(i, w, f) for i, (w, f) in enumerate(series)]


def _deep_after(entry: dict):
    """End-of-cycle R from a torchref cycle entry (prefer adp -> xyz -> scaling)."""
    for key in ("adp", "xyz"):
        sub = entry.get(key, {}).get("after")
        if sub and "rwork" in sub:
            return sub["rwork"], sub["rfree"]
    sc = entry.get("after_scaling")
    if sc and "rwork" in sc:
        return sc["rwork"], sc["rfree"]
    return None


def _deep_before(entry: dict):
    """Pre-refinement (post-scaling) R from a torchref cycle entry.

    The state entering coordinate refinement (``xyz.before``) is the analog of
    Refmac's Ncyc row 0 / Phenix's geometry-table starting row, used as cycle 0.
    """
    b = entry.get("xyz", {}).get("before")
    if b and "rwork" in b:
        return b["rwork"], b["rfree"]
    sc = entry.get("after_scaling")
    if sc and "rwork" in sc:
        return sc["rwork"], sc["rfree"]
    return None


def percycle_torchref(code: str):
    p = struct_dir("torchref", code) / "refinement_history.json"
    if not p.exists():
        return []
    try:
        hist = json.loads(p.read_text())["history"]
    except (OSError, ValueError, KeyError):
        return []
    out = []
    first_entry = True
    for seg in hist.values():  # refinement_1, ...
        for entry in seg:
            if first_entry:
                start = _deep_before(entry)  # pre-refinement start -> cycle 0
                if start is not None:
                    out.append((0, start[0], start[1]))
                first_entry = False
            r = _deep_after(entry)
            if r is not None:
                out.append((int(entry["cycle"]), r[0], r[1]))
    return out


PERCYCLE = {
    "refmac": percycle_refmac,
    "phenix": percycle_phenix,
    "torchref": percycle_torchref,
}


# ── driver ───────────────────────────────────────────────────────────────────
def all_codes():
    codes = set()
    for d in ENGINE_DIR.values():
        p = RUNS / d
        if p.is_dir():
            codes |= {c.name for c in p.iterdir() if c.is_dir()}
    return sorted(codes)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    codes = all_codes()
    print(f"Found {len(codes)} candidate structure codes across arms\n")

    rf_rows, geom_rows, rt_rows, pc_rows = [], [], [], []
    counts = {e: {"rfactor": 0, "geom": 0, "runtime": 0, "percycle": 0}
              for e in ENGINE_DIR}

    for code in codes:
        for engine in ENGINE_DIR:
            # R-factors: PHENIX (model_vs_data); geometry: uniform REFMAC restraints
            rfac = parse_phenix_rfactors(struct_dir(engine, code) / "phenix_validate.log")
            geom = parse_geometry(struct_dir(engine, code) / "validate.log")
            if rfac:
                rf_rows.append({"code": code, "engine": engine, **rfac})
                counts[engine]["rfactor"] += 1
            if geom:
                geom_rows.append({"code": code, "engine": engine, **geom})
                counts[engine]["geom"] += 1

            if engine in RUNTIME:
                w = RUNTIME[engine](code)
                if w is not None and w > 0:
                    rt_rows.append({"code": code, "engine": engine, "wall_s": w})
                    counts[engine]["runtime"] += 1

            if engine in PERCYCLE:
                cyc = PERCYCLE[engine](code)
                if cyc:
                    counts[engine]["percycle"] += 1
                    for c, w, f in cyc:
                        pc_rows.append({"code": code, "engine": engine,
                                        "cycle": c, "r_work": w, "r_free": f})

    pd.DataFrame(rf_rows).to_csv(OUT / "fig_rfactors.csv", index=False)
    pd.DataFrame(geom_rows).to_csv(OUT / "fig_geometry.csv", index=False)
    pd.DataFrame(rt_rows).to_csv(OUT / "fig_runtime.csv", index=False)
    pd.DataFrame(pc_rows).to_csv(OUT / "fig_percycle.csv", index=False)

    # ── report + headline results.csv (PHENIX-validated medians per engine) ──
    print(f"{'engine':<12}{'rfactor':>9}{'geom':>7}{'runtime':>9}{'percycle':>10}"
          f"{'medRfree':>10}")
    rf = pd.DataFrame(rf_rows)
    res_rows = []
    for e in ENGINE_DIR:
        sub = rf[rf.engine == e] if not rf.empty else rf
        med_w = sub["r_work"].median() if len(sub) else float("nan")
        med_f = sub["r_free"].median() if len(sub) else float("nan")
        c = counts[e]
        print(f"{e:<12}{c['rfactor']:>9}{c['geom']:>7}{c['runtime']:>9}"
              f"{c['percycle']:>10}{med_f:>10.4f}")
        res_rows.append({"engine": e, "n": len(sub),
                         "median_rwork": round(med_w, 4),
                         "median_rfree": round(med_f, 4),
                         "overfit_gap": round(med_f - med_w, 4)})
    # order to match the legacy file: torchref, phenix, refmac, prediction
    order = ["torchref", "phenix", "refmac", "prediction"]
    res = pd.DataFrame(res_rows).set_index("engine").reindex(order).reset_index()
    res.to_csv(BASE / "results.csv", index=False)
    print(f"\nWrote 4 CSVs to {OUT}")
    print(f"Wrote {BASE / 'results.csv'}  (PHENIX-validated medians)")


if __name__ == "__main__":
    main()
