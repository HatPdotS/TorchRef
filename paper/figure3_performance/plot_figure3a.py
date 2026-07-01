"""Figure 3a — structure-factor calculation speed (message-first Cleveland dot plot).

One row per structure (sorted by atom count; 5BOV dropped as a triclinic outlier),
forward F_calc time on a log x-axis, with:
  * TorchRef CPU at 1 / 4 / 16 cores   (light -> dark blue, connected sweep)
  * TorchRef GPU (A100)                 (red star)
  * SFcalculator GPU (A100)             (orange diamond; OOM cases omitted)
  * cctbx single-thread reference       (black tick)

Reads TWO benchmarks:
  --fcalc-dir : fcalc_benchmark results dir with summary.csv
                (device=cpu rows -> TorchRef torchref_min @ 1/4/16 threads + cctbx_min;
                 device=gpu rows -> TorchRef GPU torchref_min)
  --sf-dir    : SF_calc_comparison results dir with cuda/*_sfcalc.json
                (SFcalculator GPU fwd.min_time where status.time_fwd == "ok")

Usage:
    plot_figure3a.py --fcalc-dir data/fcalc/results_XXXX \
                     --sf-dir SF_calc_comparison/results_XXXX \
                     [--output output/figure3a_fcalc.png] [--drop 5BOV]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent

CORE_DOTS = {1: ("#9ecae1", 55), 4: ("#4292c6", 95), 16: ("#08519c", 140)}
GPU_C = "#d62728"     # TorchRef GPU
SC_C = "#ff7f0e"      # SFcalculator GPU
CCTBX_C = "#333333"
ALPHA = 0.8


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_fcalc(fcalc_dir: Path, drop: set):
    """Return tr[struct][threads]=min_s, cctbx[struct]=min_s, trgpu[struct]=min_s,
    plus n_atoms / d_min per structure."""
    csv_path = fcalc_dir / "summary.csv"
    tr = defaultdict(dict)
    cc, trgpu, natoms, dmin = {}, {}, {}, {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            s = r["structure"]
            if s in drop:
                continue
            if r["device"] == "gpu":
                v = _f(r.get("torchref_min"))
                if v:
                    trgpu[s] = v
                continue
            natoms[s] = int(r["n_atoms"])
            dmin[s] = float(r["d_min"])
            v = _f(r.get("torchref_min"))
            if v is not None:
                tr[s][int(r["n_threads"])] = v
            cv = _f(r.get("cctbx_min"))
            if cv is not None:
                cc[s] = cv
    return tr, cc, trgpu, natoms, dmin


def load_sfcalc_gpu(sf_dir: Path, drop: set):
    """SFcalculator GPU forward min_time per structure (OOM -> absent)."""
    scgpu = {}
    for p in glob.glob(str(sf_dir / "cuda" / "*_sfcalc.json")):
        j = json.load(open(p))
        s = j["structure"]
        if s in drop:
            continue
        if j.get("status", {}).get("time_fwd") == "ok" and isinstance(j.get("fwd"), dict):
            scgpu[s] = j["fwd"]["min_time"]
    return scgpu


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fcalc-dir", required=True,
                    help="fcalc_benchmark results dir (contains summary.csv)")
    ap.add_argument("--sf-dir", required=True,
                    help="SF_calc_comparison results dir (contains cuda/*_sfcalc.json)")
    ap.add_argument("--output", default=str(SCRIPT_DIR / "output" / "figure3a_fcalc.png"))
    ap.add_argument("--drop", nargs="*", default=["5BOV"],
                    help="Structures to exclude (default: 5BOV, a triclinic outlier).")
    args = ap.parse_args()

    drop = set(args.drop)
    tr, cc, trgpu, natoms, dmin = load_fcalc(Path(args.fcalc_dir), drop)
    scgpu = load_sfcalc_gpu(Path(args.sf_dir), drop)
    if not tr:
        raise SystemExit(f"no CPU rows in {args.fcalc_dir}/summary.csv")

    order = sorted(tr, key=lambda s: natoms[s])
    ypos = {s: i for i, s in enumerate(order)}

    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    for s in order:
        y = ypos[s]
        ax.axhline(y, color="0.94", lw=7, zorder=0)
        valid = [(tr[s][n], n) for n in (1, 4, 16) if n in tr[s]]
        if len(valid) >= 2:
            ax.plot([v[0] * 1e3 for v in valid], [y] * len(valid), "-",
                    color="#9ecae1", lw=1.3, zorder=2)
        for n in (1, 4, 16):
            if n in tr[s]:
                c, sz = CORE_DOTS[n]
                ax.scatter(tr[s][n] * 1e3, y, s=sz, c=c, alpha=ALPHA,
                           edgecolors="white", lw=0.7, zorder=3)
        if s in cc:
            ax.scatter(cc[s] * 1e3, y, s=170, marker="|", c=CCTBX_C, alpha=ALPHA,
                       lw=2.5, zorder=4)
        if s in trgpu:
            ax.scatter(trgpu[s] * 1e3, y, s=200, marker="*", c=GPU_C, alpha=ALPHA,
                       edgecolors="white", lw=0.7, zorder=6)
        if s in scgpu:
            ax.scatter(scgpu[s] * 1e3, y, s=95, marker="D", c=SC_C, alpha=ALPHA,
                       edgecolors="white", lw=0.7, zorder=5)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{s}\n{natoms[s]:,} at · {dmin[s]:.1f} Å" for s in order],
                       fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(r"forward $F_{calc}$ time (ms)")
    ax.grid(axis="x", which="both", color="0.9", lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.margins(y=0.03)

    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=GPU_C, alpha=ALPHA,
               markersize=15, label="TorchRef GPU (A100)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=SC_C, alpha=ALPHA,
               markersize=9, label="SFcalculator GPU (A100)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CORE_DOTS[1][0],
               alpha=ALPHA, markersize=8, label="TorchRef 1 core"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CORE_DOTS[4][0],
               alpha=ALPHA, markersize=10, label="TorchRef 4 cores"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CORE_DOTS[16][0],
               alpha=ALPHA, markersize=12, label="TorchRef 16 cores"),
        Line2D([0], [0], marker="|", color=CCTBX_C, markersize=13, lw=0,
               markeredgewidth=2.5, label="cctbx (ref, 1 thread)"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    missing = [s for s in order if s not in scgpu]
    if missing:
        print(f"SFcalculator GPU absent (OOM or no data): {missing}")


if __name__ == "__main__":
    main()
