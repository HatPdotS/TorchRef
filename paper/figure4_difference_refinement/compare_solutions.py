#!/usr/bin/env python3
"""Compare two difference refinement solutions via DED validation.

Produces a 3-panel figure:
  1. Amplitude CC by resolution (both solutions overlaid)
  2. WDFo vs WDFc scatter for solution A (within selection mask)
  3. WDFo vs WDFc scatter for solution B (within selection mask)

Usage:
    python compare_solutions.py
"""

from pathlib import Path

import numpy as np

from torchref.cli._common import load_model
from torchref.cli.validate_ded import setup_ded_context, compute_ded_maps

# ── Configuration ──────────────────────────────────────────────────────────
DATA = Path(__file__).resolve().parent / "data"
DARK_SF = DATA / "8QL2-sf.cif"
LIGHT_SF = DATA / "7YYZ-light.mtz"
DARK_MODEL = DATA / "8QL2_no_altloc.pdb"
DMIN = 2.2
SELECTION = "resname IBL"
MASK_RADIUS = 2.5

SOLUTION_A = {
    "label": "TorchRef",
    "light_model": DATA / "torchref_0p22.pdb",
    "fraction": 0.22,
}

SOLUTION_B = {
    "label": "Extrapolation",
    "light_model": DATA / "7YYZ.pdb",
    "fraction": 0.22,
}

OUTDIR = Path(__file__).resolve().parent / "comparison"
# ───────────────────────────────────────────────────────────────────────────


def _get_scatter_data(res):
    """Extract scatter data from a result dict."""
    mask = res["mask_dict"].get("selection", res["mask_dict"]["full_cell"])
    v_dfo = res["map_dfo"][mask].detach().cpu().numpy()
    v_dfc = res["map_dfc"][mask].detach().cpu().numpy()
    v1 = v_dfo - v_dfo.mean()
    v2 = v_dfc - v_dfc.mean()
    cc = np.sum(v1 * v2) / (np.sqrt(np.sum(v1**2) * np.sum(v2**2)) + 1e-12)
    return v_dfo, v_dfc, cc


def _plot_cc_by_resolution(ax, res_a, res_b, label_a, label_b):
    """Plot amplitude CC by resolution for two solutions."""
    for res, label, marker, color in [
        (res_a, label_a, "o-", "tab:blue"),
        (res_b, label_b, "s-", "tab:red"),
    ]:
        d = [b["d_min"] for b in res["resolution_bins"]]
        cc = [b["cc"] for b in res["resolution_bins"]]
        ax.plot(d, cc, marker, color=color, markersize=4,
                label=f"{label} (CC={res['reciprocal_cc_overall']:.3f})")
    ax.set_xlabel("Resolution (Å)")
    ax.set_ylabel("CC(WDFo, WDFcalc)")
    ax.set_title("Amplitude CC by Resolution", fontsize=18)
    ax.invert_xaxis()
    ax.axhline(y=0, color="gray", ls="--", alpha=0.3)
    ax.set_ylim(-0.1, 0.7)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)


def _plot_scatter(ax, v_dfo, v_dfc, cc, label, color):
    """Plot WDFo vs WDFc scatter."""
    ax.scatter(v_dfo, v_dfc, s=1, alpha=0.15, color=color)
    ax.set_xlabel("WDFo (σ)")
    ax.set_ylabel("WDFcalc (σ)")
    ax.set_title(f"{label}\nCC = {cc:.3f}", fontsize=18)
    lim = max(abs(v_dfo).max(), abs(v_dfc).max())
    ax.plot([-lim, lim], [-lim, lim], "k-", linewidth=1, alpha=0.4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def plot_comparison(res_a, res_b, label_a, label_b, outdir):
    """Generate combined 3-panel figure and individual panel PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    scatter_a = _get_scatter_data(res_a)
    scatter_b = _get_scatter_data(res_b)

    W, H = 6, 4.5  # 4:3 aspect ratio per panel

    # ── Combined figure ──
    fig, axes = plt.subplots(1, 3, figsize=(3 * W, H))
    fig.suptitle("DED Comparison: WDFo vs WDFcalc", fontsize=21, y=1.02)
    _plot_cc_by_resolution(axes[0], res_a, res_b, label_a, label_b)
    _plot_scatter(axes[1], *scatter_a, label_a, "tab:blue")
    _plot_scatter(axes[2], *scatter_b, label_b, "tab:red")
    plt.tight_layout()
    png = outdir / "compare_solutions.png"
    plt.savefig(str(png), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")

    # ── Individual panels ──
    # CC by resolution
    fig, ax = plt.subplots(figsize=(W, H))
    _plot_cc_by_resolution(ax, res_a, res_b, label_a, label_b)
    plt.tight_layout()
    p = outdir / "cc_by_resolution.png"
    plt.savefig(str(p), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # Scatter A
    fig, ax = plt.subplots(figsize=(W, H))
    _plot_scatter(ax, *scatter_a, label_a, "tab:blue")
    p = outdir / "scatter_a.png"
    plt.savefig(str(p), dpi=200)
    plt.close(fig)
    print(f"Saved: {p}")

    # Scatter B
    fig, ax = plt.subplots(figsize=(W, H))
    _plot_scatter(ax, *scatter_b, label_b, "tab:red")
    p = outdir / "scatter_b.png"
    plt.savefig(str(p), dpi=200)
    plt.close(fig)
    print(f"Saved: {p}")


def main():
    print("Setting up shared context...")
    ctx = setup_ded_context(DARK_SF, LIGHT_SF, dmin=DMIN, verbose=1, n_bins=10)

    model_dark = load_model(str(DARK_MODEL), max_res=DMIN, device=ctx["device"], verbose=0)

    results = {}
    for key, sol in [("a", SOLUTION_A), ("b", SOLUTION_B)]:
        print(f"\nComputing: {sol['label']}...")
        model_light = load_model(
            str(sol["light_model"]), max_res=DMIN, device=ctx["device"], verbose=0
        )
        results[key] = compute_ded_maps(
            ctx, model_dark, model_light,
            fraction=sol["fraction"],
            selection=SELECTION,
            mask_radius=MASK_RADIUS,
            verbose=1,
        )
        print(f"  CC_overall = {results[key]['reciprocal_cc_overall']}")

    print("\nPlotting...")
    plot_comparison(
        results["a"], results["b"],
        SOLUTION_A["label"], SOLUTION_B["label"],
        OUTDIR,
    )


if __name__ == "__main__":
    main()
