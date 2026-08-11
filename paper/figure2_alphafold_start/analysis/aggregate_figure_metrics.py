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
  torchref   = torchref (canonical arm: -n 10 separate ml, weights 1/0.2/0.02).

Usage
-----
    ./.dev/bin/python paper/figure2_alphafold_start/analysis/aggregate_figure_metrics.py
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# ── Paths / engine layout ─────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent              # paper/figure2_alphafold_start
RUNS = BASE / "runs"
OUT = RUNS / "metrics"

# engine -> arm subdirectory under runs/. Rebound by main() when --arm is given, so the
# same four parsers serve both Figure 2 (four engines) and ExtFig 5 (four target modes,
# all of them `torchref`-kind arms) instead of the mode figure carrying a second copy.
DEFAULT_ENGINE_DIR = {
    "prediction": "af_initial",
    "refmac": "refmac",
    "phenix": "phenix_norb",
    "torchref": "torchref",
}
ENGINE_DIR = dict(DEFAULT_ENGINE_DIR)
#: engine -> which PROGRAM wrote its logs, i.e. which runtime/per-cycle/failure parser
#: applies. Defaults to the engine name; only differs for renamed arms (`ml_full` is a
#: `torchref`-kind arm). Rebound by main() alongside ENGINE_DIR.
PARSER_KIND = {e: e for e in ENGINE_DIR}
#: output filename prefix, so a second arm set cannot overwrite Figure 2's CSVs.
PREFIX = "fig"


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


def runtime_torchref(engine: str, code: str):
    p = struct_dir(engine, code) / "out.log"
    if not p.exists():
        return None
    m = None
    for m in RE_TORCHREF_WALL.finditer(p.read_text(errors="replace")):
        pass  # keep last
    return float(m.group(1)) if m else None


def runtime_phenix(engine: str, code: str):
    p = struct_dir(engine, code) / f"{code}_refined_001.log"
    if not p.exists():
        return None
    epochs = [float(x) for x in RE_PHENIX_EPOCH.findall(p.read_text(errors="replace"))]
    return (epochs[-1] - epochs[0]) if len(epochs) >= 2 else None


def runtime_refmac(engine: str, code: str):
    p = struct_dir(engine, code) / "refmac.log"
    if not p.exists():
        return None
    m = RE_REFMAC_ELAPSED.search(p.read_text(errors="replace"))
    return (int(m.group(1)) * 60 + int(m.group(2))) if m else None


# Keyed by PARSER KIND (which program wrote the logs), not by engine name -- the
# target-mode arms are all `torchref` kind under names like `ml_noalpha`.
RUNTIME = {
    "torchref": runtime_torchref,
    "phenix": runtime_phenix,
    "refmac": runtime_refmac,
}


# ── per-cycle program-reported R-factors ─────────────────────────────────────
def percycle_refmac(engine: str, code: str):
    """Refmac Ncyc table: rows '<cyc> <Rfact> <Rfree> ...' until '$$'."""
    p = struct_dir(engine, code) / "refmac.log"
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


def percycle_phenix(engine: str, code: str):
    """Phenix per-macrocycle R-FACTORS (work/free in %) from the 'bonds angl' blocks.

    Each block's *first* row is the state at the start of that macrocycle and the
    *last* row is the post-(geometry-)refinement state. Per Phenix's own legend,
    the very first block's first row is the pre-refinement starting state, which
    we emit as cycle 0 (matching Refmac's Ncyc row 0) so the start R-factor is not
    silently dropped.
    """
    p = struct_dir(engine, code) / f"{code}_refined_001.log"
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


def percycle_torchref(engine: str, code: str):
    p = struct_dir(engine, code) / "refinement_history.json"
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
_CODE_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")  # PDB id, e.g. 1DAW (skip tmp_scripts)


def all_codes():
    codes = set()
    for d in ENGINE_DIR.values():
        p = RUNS / d
        if p.is_dir():
            codes |= {c.name for c in p.iterdir()
                      if c.is_dir() and _CODE_RE.match(c.name)}
    return sorted(codes)


def _read_log(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def classify_failure(engine: str, code: str):
    """Why a structure has no PHENIX-validated R-factor, as (category, reason).

    category 'refine' = the engine itself could not produce a refined model;
    'score' = refinement succeeded but the uniform PHENIX validator
    (model_vs_data, used to score every engine) could not score it. The
    recurring driver is a special-position pathology in some placed AF models:
    it crashes the PHENIX validator (so the structure is unscoreable for *all*
    engines) and makes phenix.refine itself abort, while REFMAC/TorchRef refine
    it without error.
    """
    d = struct_dir(engine, code)
    out_log = _read_log(d / "out.log")
    err_log = _read_log(d / "error.log")
    pval = _read_log(d / "phenix_validate.log")
    vlog = _read_log(d / "validate.log")
    pref = _read_log(d / "phenix_refine.log")
    for g in d.glob("*_refined_001.log"):
        pref += _read_log(g)

    # 1) engine refinement failures
    kind = PARSER_KIND.get(engine, engine)
    if kind == "torchref":
        if "oom_kill" in err_log or "out of memory" in (err_log + out_log).lower():
            return ("refine", "TorchRef OOM at 8 GB (large structure)")
        if out_log and "Refinement completed successfully" not in out_log:
            return ("refine", "TorchRef refinement error")
    if kind == "phenix":
        low = pref.lower()
        if "polymer crosses special position" in low:
            return ("refine", "phenix.refine aborts: polymer crosses special position")
        if "key not in c++ map" in low:
            return ("refine", "phenix.refine crash: special-position pathology")
        if "merge_equivalents" in low or "incompatible flags" in low:
            return ("refine", "phenix.refine: incompatible data flags (Friedel mismatch)")
        if "please try again" in low:
            return ("refine", "phenix.refine transient error")
        m = re.search(r"Sorry:\s*(.+)", pref)
        if m:
            return ("refine", "phenix.refine Sorry: " + m.group(1).strip()[:50])

    # 2) scoring/validation failures (the uniform PHENIX validator)
    low = (pval + " " + vlog).lower()
    if ("key not in c++ map" in low or "special position" in low
            or "polymer crosses" in low):
        return ("score", "PHENIX validator fails (special-position pathology)")
    if "merge_equivalents" in low or "incompatible flags" in low:
        return ("score", "PHENIX validator: incompatible data flags")
    if "please try again" in low:
        return ("score", "PHENIX validator transient error")
    return ("other", "unclassified")


def _parse_arm(spec: str):
    """``NAME=DIR`` or ``NAME=DIR:KIND`` -> ``(name, dir, kind)``.

    ``KIND`` names the program whose logs are being parsed and defaults to ``NAME``, so
    Figure 2's four engines need no annotation while a renamed arm does:
    ``ml_full=torchref_ml_full:torchref``.
    """
    if "=" not in spec:
        raise SystemExit(f"--arm expects NAME=DIR[:KIND], got {spec!r}")
    name, rest = spec.split("=", 1)
    directory, _, kind = rest.partition(":")
    return name, directory, (kind or name)


def main():
    global ENGINE_DIR, PARSER_KIND, PREFIX, CONSERVED_NAME
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=None, metavar="NAME=DIR[:KIND]",
                    help="Override the engine set. Repeatable. KIND selects the log "
                         "parsers (torchref/phenix/refmac/prediction) and defaults to "
                         "NAME. With no --arm the Figure 2 default set is used and the "
                         "output is byte-identical to the un-flagged behaviour.")
    ap.add_argument("--prefix", default=PREFIX,
                    help="Output filename prefix (default 'fig'); a second arm set MUST "
                         "use its own prefix or it overwrites Figure 2's CSVs.")
    args = ap.parse_args()

    if args.arm:
        parsed = [_parse_arm(a) for a in args.arm]
        ENGINE_DIR = {n: d for n, d, _ in parsed}
        PARSER_KIND = {n: k for n, _, k in parsed}
        unknown = set(PARSER_KIND.values()) - set(DEFAULT_ENGINE_DIR)
        if unknown:
            raise SystemExit(f"unknown parser kind(s) {sorted(unknown)}; "
                             f"choose from {sorted(DEFAULT_ENGINE_DIR)}")
        missing = [d for d in ENGINE_DIR.values() if not (RUNS / d).is_dir()]
        if missing:
            raise SystemExit(f"arm directories do not exist: {missing}")
    PREFIX = args.prefix
    # Figure 2 keeps the historical filename; any other arm set gets its own, so the two
    # conserved sets (four engines vs four target modes) cannot be confused.
    CONSERVED_NAME = ("conserved_codes.txt" if PREFIX == "fig"
                      else f"{PREFIX}_conserved_codes.txt")

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Arms: {ENGINE_DIR}")
    print(f"Parsers: {PARSER_KIND}\nPrefix: {PREFIX}\n")
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

            kind = PARSER_KIND[engine]
            if kind in RUNTIME:
                w = RUNTIME[kind](engine, code)
                if w is not None and w > 0:
                    rt_rows.append({"code": code, "engine": engine, "wall_s": w})
                    counts[engine]["runtime"] += 1

            if kind in PERCYCLE:
                cyc = PERCYCLE[kind](engine, code)
                if cyc:
                    counts[engine]["percycle"] += 1
                    for c, w, f in cyc:
                        pc_rows.append({"code": code, "engine": engine,
                                        "cycle": c, "r_work": w, "r_free": f})

    pd.DataFrame(rf_rows).to_csv(OUT / f"{PREFIX}_rfactors.csv", index=False)
    pd.DataFrame(geom_rows).to_csv(OUT / f"{PREFIX}_geometry.csv", index=False)
    pd.DataFrame(rt_rows).to_csv(OUT / f"{PREFIX}_runtime.csv", index=False)
    pd.DataFrame(pc_rows).to_csv(OUT / f"{PREFIX}_percycle.csv", index=False)

    # ── conserved set: codes with a PHENIX-validated R-free for ALL engines ──
    # Headline medians MUST be over the same structures or they are not comparable
    # (an engine that solved more/fewer structures is otherwise scored on a
    # different, not-equally-hard subset). Persist the set so plot_figure_af.py and
    # summarize_medians.py report on the identical structures.
    rf = pd.DataFrame(rf_rows)
    rf_sets = {e: set(rf[rf.engine == e].code) for e in ENGINE_DIR} if not rf.empty else {}
    conserved = (set.intersection(*rf_sets.values())
                 if rf_sets and all(rf_sets.values()) else set())
    (OUT / f"{CONSERVED_NAME}").write_text("\n".join(sorted(conserved)) + "\n")

    # ── run accounting: per-engine success + grouped failure reasons ──
    # Attempted = candidate codes whose arm dir exists; success = has a validated
    # R-factor; everything else is a failure, classified by classify_failure().
    fail_rows = []
    for e in ENGINE_DIR:
        succ = set(rf[rf.engine == e].code) if not rf.empty else set()
        for code in codes:
            if code in succ or not struct_dir(e, code).is_dir():
                continue
            cat, reason = classify_failure(e, code)
            fail_rows.append({"engine": e, "code": code,
                              "category": cat, "reason": reason})
    pd.DataFrame(fail_rows,
                 columns=["engine", "code", "category", "reason"]).to_csv(
        OUT / f"{PREFIX}_failures.csv", index=False)
    print(f"\nRun accounting (candidate N={len(codes)}):")
    for e in ENGINE_DIR:
        succ = len(set(rf[rf.engine == e].code)) if not rf.empty else 0
        ef = [r for r in fail_rows if r["engine"] == e]
        nref = sum(r["category"] == "refine" for r in ef)
        nsco = sum(r["category"] == "score" for r in ef)
        print(f"  {e:<11} success={succ:>4}  fail={len(ef):>3} "
              f"(refine {nref}, score {nsco})")

    # ── report + headline results.csv (PHENIX-validated medians, conserved set) ──
    print(f"Conserved set (R-free for all {len(ENGINE_DIR)} engines): "
          f"n={len(conserved)}\n")
    print(f"{'engine':<12}{'rfactor':>9}{'geom':>7}{'runtime':>9}{'percycle':>10}"
          f"{'medRfree*':>10}")
    res_rows = []
    for e in ENGINE_DIR:
        sub = rf[(rf.engine == e) & (rf.code.isin(conserved))] if not rf.empty else rf
        med_w = sub["r_work"].median() if len(sub) else float("nan")
        med_f = sub["r_free"].median() if len(sub) else float("nan")
        c = counts[e]
        print(f"{e:<12}{c['rfactor']:>9}{c['geom']:>7}{c['runtime']:>9}"
              f"{c['percycle']:>10}{med_f:>10.4f}")
        res_rows.append({"engine": e, "n": len(sub),
                         "median_rwork": round(med_w, 4),
                         "median_rfree": round(med_f, 4),
                         "overfit_gap": round(med_f - med_w, 4)})
    print("  * medRfree is over the conserved set (identical structures per engine)")
    # order to match the legacy file: torchref, phenix, refmac, prediction
    order = [e for e in ("torchref", "phenix", "refmac", "prediction") if e in ENGINE_DIR]
    order += [e for e in ENGINE_DIR if e not in order]
    res = pd.DataFrame(res_rows).set_index("engine").reindex(order).reset_index()
    res_path = BASE / ("results.csv" if PREFIX == "fig" else f"{PREFIX}_results.csv")
    res.to_csv(res_path, index=False)
    print(f"\nWrote 4 CSVs to {OUT} with prefix {PREFIX!r}")
    print(f"Wrote {res_path}  (PHENIX-validated medians)")


if __name__ == "__main__":
    main()
