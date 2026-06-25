#!/usr/bin/env python3
"""Compare the TorchRef AF-start arm WITH vs WITHOUT the Ramachandran restraint.

Both arms are identical (-n 10 --mode separate --xray-mode ml, group weights
xray 1 / geometry 0.2 / adp 0.02); the only difference is that the *_norama arm
adds ``--weights '{"geometry/ramachandran": 0}'`` so the Ramachandran component
is skipped (its effective weight 0.2 * 0 = 0).

Each structure's R-factors and geometry come from the co-located REFMAC 0-cycle
``validate.log`` (apples-to-apples; same parser as aggregate_figure_metrics.py),
so the only thing that moved is what TorchRef refined, not how it was scored.

Pairs by PDB code, reports medians over the paired (both-arms-present) set and a
paired Δ (norama − rama), and writes a CSV + markdown table.

Usage
-----
    ./.dev/bin/python paper/figure2_alphafold_start/analysis/compare_ramachandran.py
    ./.dev/bin/python paper/figure2_alphafold_start/analysis/compare_ramachandran.py \
        --with-arm torchref_g0p2_a0p02 --without-arm torchref_g0p2_a0p02_norama
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_af_pipeline as P  # noqa: E402

# Same regexes as aggregate_figure_metrics.py (one source of truth for parsing).
RE_RWORK = re.compile(r"Overall R factor\s+=\s+([\d.]+)")
RE_RFREE = re.compile(r"Free R factor\s+=\s+([\d.]+)")
RE_TAIL3 = re.compile(r"(\d+)\s+([\d.]+)\s+([\d.]+)\s*$")
GEOM_ROWS = {
    "Bond distances": ("rmsBOND", "sigBOND"),
    "Bond angles": ("rmsANGL", "sigANGL"),
    "Chiral centres": ("rmsCHIRAL", "sigCHIRAL"),
    "M. chain bond B values": ("rmsB_mc_bond", "sigB_mc_bond"),
}


def parse_validate(path: Path):
    """R-work/R-free + geometry RMS-delta/Av(sigma) from a REFMAC validate.log."""
    if not path.exists():
        return None
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    mw, mf = RE_RWORK.search(text), RE_RFREE.search(text)
    if not (mw and mf):
        return None
    rec = {"r_work": float(mw.group(1)), "r_free": float(mf.group(1))}
    for line in text.splitlines():
        for label, (rms_col, sig_col) in GEOM_ROWS.items():
            if line.startswith(label):
                m = RE_TAIL3.search(line)
                if m:
                    rec[rms_col] = float(m.group(2))
                    rec[sig_col] = float(m.group(3))
    return rec


def collect(arm, codes):
    out = {}
    for code in codes:
        rec = parse_validate(P.RUNS / arm / code / "validate.log")
        if rec is not None:
            out[code] = rec
    return out


def med(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return (float(np.median(vals)), len(vals)) if vals else (float("nan"), 0)


def rmsz(rec, rms_col, sig_col):
    rms, sig = rec.get(rms_col), rec.get(sig_col)
    if rms is None or sig is None or sig <= 0:
        return None
    return rms / sig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--with-arm", default="torchref_g0p2_a0p02",
                    help="Arm WITH Ramachandran (the locked default).")
    ap.add_argument("--without-arm", default="torchref_g0p2_a0p02_norama",
                    help="Arm WITHOUT Ramachandran.")
    ap.add_argument("--out-csv", default=None,
                    help="Per-structure CSV (default runs/metrics/ramachandran_compare.csv).")
    ap.add_argument("--out-md", default=None,
                    help="Markdown summary (default runs/metrics/ramachandran_compare.md).")
    args = ap.parse_args()

    codes = P.load_solved_codes()
    rama = collect(args.with_arm, codes)
    nora = collect(args.without_arm, codes)
    paired = sorted(set(rama) & set(nora))

    print(f"WITH-rama arm   '{args.with_arm}':    {len(rama)} validated")
    print(f"WITHOUT-rama    '{args.without_arm}': {len(nora)} validated")
    print(f"paired (both present):                 {len(paired)}\n")
    if not paired:
        print("No paired structures yet — let the *_norama jobs finish, then re-run.")
        return

    metrics_dir = P.RUNS / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv) if args.out_csv else metrics_dir / "ramachandran_compare.csv"
    out_md = Path(args.out_md) if args.out_md else metrics_dir / "ramachandran_compare.md"

    # ── per-structure CSV ────────────────────────────────────────────────────
    fields = ["code",
              "rwork_rama", "rfree_rama", "rwork_norama", "rfree_norama",
              "d_rwork", "d_rfree",
              "rmsz_bond_rama", "rmsz_bond_norama",
              "rmsz_angle_rama", "rmsz_angle_norama"]
    rows = []
    for code in paired:
        a, b = rama[code], nora[code]
        rows.append({
            "code": code,
            "rwork_rama": a["r_work"], "rfree_rama": a["r_free"],
            "rwork_norama": b["r_work"], "rfree_norama": b["r_free"],
            "d_rwork": b["r_work"] - a["r_work"],
            "d_rfree": b["r_free"] - a["r_free"],
            "rmsz_bond_rama": rmsz(a, "rmsBOND", "sigBOND"),
            "rmsz_bond_norama": rmsz(b, "rmsBOND", "sigBOND"),
            "rmsz_angle_rama": rmsz(a, "rmsANGL", "sigANGL"),
            "rmsz_angle_norama": rmsz(b, "rmsANGL", "sigANGL"),
        })
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ── medians over the paired set ──────────────────────────────────────────
    def pm(key):
        return med([r[key] for r in rows])[0]

    rw_a, rw_b = pm("rwork_rama"), pm("rwork_norama")
    rf_a, rf_b = pm("rfree_rama"), pm("rfree_norama")
    gap_a = pm("rfree_rama") - pm("rwork_rama")
    gap_b = pm("rfree_norama") - pm("rwork_norama")
    d_rw = pm("d_rwork")
    d_rf = pm("d_rfree")
    n_rf_worse = sum(1 for r in rows if r["d_rfree"] > 0)

    bond_a, bond_b = pm("rmsz_bond_rama"), pm("rmsz_bond_norama")
    ang_a, ang_b = pm("rmsz_angle_rama"), pm("rmsz_angle_norama")

    md = [
        "# TorchRef AF-start: Ramachandran restraint on vs off",
        "",
        f"Paired over **{len(paired)}** structures (both arms REFMAC-0-cycle "
        "validated). Both arms: `-n 10 --mode separate --xray-mode ml`, group "
        "weights xray 1 / geometry 0.2 / adp 0.02; the only difference is "
        "`geometry/ramachandran` weight (default vs 0).",
        "",
        "## Median R-factors (paired set)",
        "",
        "| Arm | median R-work | median R-free | R-free − R-work gap |",
        "|---|---|---|---|",
        f"| with Ramachandran (`{args.with_arm}`) | {rw_a:.4f} | {rf_a:.4f} | {gap_a:.4f} |",
        f"| without Ramachandran (`{args.without_arm}`) | {rw_b:.4f} | {rf_b:.4f} | {gap_b:.4f} |",
        "",
        "## Paired Δ (without − with)",
        "",
        f"- median ΔR-work: **{d_rw:+.4f}**",
        f"- median ΔR-free: **{d_rf:+.4f}**  (>0 ⇒ removing Ramachandran hurts R-free)",
        f"- structures where R-free got worse without Ramachandran: "
        f"**{n_rf_worse}/{len(rows)}** ({100*n_rf_worse/len(rows):.0f}%)",
        "",
        "## Geometry quality (median RMSZ; ideal ≈ 1.0)",
        "",
        "| Restraint | with Ramachandran | without Ramachandran |",
        "|---|---|---|",
        f"| Bond RMSZ | {bond_a:.2f} | {bond_b:.2f} |",
        f"| Angle RMSZ | {ang_a:.2f} | {ang_b:.2f} |",
        "",
        f"Per-structure data: `{out_csv}`",
        "",
    ]
    out_md.write_text("\n".join(md))

    print("\n".join(md))
    print(f"\nwrote {out_csv}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
