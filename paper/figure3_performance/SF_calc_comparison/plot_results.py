#!/usr/bin/env python
"""
Plot the SFcalculator vs TorchRef (SF_DS / SF_FFT) benchmark.

Reads results_<ts>/<device>/*.json and produces, per device, a 3-panel grouped
bar chart: forward time, forward+backward time, and peak memory; x-axis is the
structure (ordered by atom count), one bar group per method.

Usage:
    python plot_results.py --results-dir results_20260608_120000
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["STIXGeneral"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 13
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["axes.titlesize"] = 16
import numpy as np

# Method display order, labels, colors.
METHOD_ORDER = ["sf_fft", "sf_ds", "sfcalc", "cctbx"]
METHOD_LABEL = {
    "sf_fft": "TorchRef SF_FFT",
    "sf_ds": "TorchRef SF_DS",
    "sfcalc": "SFcalculator",
    "cctbx": "cctbx (ref)",
}
METHOD_COLOR = {
    "sf_fft": "#2563eb",   # blue
    "sf_ds": "#7c3aed",    # purple
    "sfcalc": "#e67e22",   # orange
    "cctbx": "#6b7280",    # gray
}


def load_device(device_dir: Path) -> list[dict]:
    runs = []
    for jf in sorted(device_dir.glob("*.json")):
        with open(jf) as f:
            runs.append(json.load(f))
    return runs


def _structures_sorted(runs):
    # Order structures by atom count (ascending).
    by_struct = {}
    for r in runs:
        by_struct.setdefault(r["structure"], r.get("n_atoms", 0))
    return sorted(by_struct, key=lambda s: by_struct[s])


def _value(runs, structure, method, kind):
    """kind in {'fwd', 'fwd_bwd', 'mem_fwd', 'mem_fwd_bwd'}; returns float or nan."""
    for r in runs:
        if r["structure"] == structure and r["method"] == method:
            if kind == "fwd":
                return r["fwd"]["mean_time"] if r.get("fwd") else np.nan
            if kind == "fwd_bwd":
                return r["fwd_bwd"]["mean_time"] if r.get("fwd_bwd") else np.nan
            if kind == "mem_fwd":
                v = r.get("mem_fwd_bytes")
                return v / 1e6 if v is not None else np.nan
            if kind == "mem_fwd_bwd":
                v = r.get("mem_fwd_bwd_bytes")
                return v / 1e6 if v is not None else np.nan
    return np.nan


def _bar_panel(ax, runs, structures, methods, kind, ylabel, title):
    n_m = len(methods)
    width = 0.8 / n_m
    x = np.arange(len(structures))
    # Log scale first so axis limits derive only from finite bar heights.
    ax.set_yscale("log")
    nan_marks = []  # (x_position,) for n/a annotations, drawn after autoscale
    for i, m in enumerate(methods):
        vals = [_value(runs, s, m, kind) for s in structures]
        offset = (i - (n_m - 1) / 2) * width
        ax.bar(
            x + offset, vals, width, label=METHOD_LABEL[m],
            color=METHOD_COLOR[m], edgecolor="black", linewidth=0.4,
        )
        nan_marks += [xi for xi, v in zip(x + offset, vals) if np.isnan(v)]
    ax.set_xticks(x)
    ax.set_xticklabels(structures)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", which="both", ls=":", alpha=0.4)
    # Annotate missing (OOM / skipped) cells. Use a blended transform so the
    # text sits just above the x-axis in AXES-fraction y — never at data y=0,
    # which on a log axis is -inf and would explode the figure size.
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    for xi in nan_marks:
        ax.text(xi, 0.02, "n/a", transform=trans, ha="center", va="bottom",
                fontsize=8, rotation=90, color="gray")


def _norm_bar_panel(ax, runs, structures, methods, kind, title, ref="sf_fft"):
    """Bar panel of each method's value divided by the reference method's.

    The reference (SF_FFT) is the baseline = 1.0 (drawn as a dashed line). Cells
    where either the method or the reference is missing are marked 'n/a'.
    """
    n_m = len(methods)
    width = 0.8 / n_m
    x = np.arange(len(structures))
    ax.set_yscale("log")
    base = {s: _value(runs, s, ref, kind) for s in structures}
    nan_marks = []
    for i, m in enumerate(methods):
        ratios = []
        for s in structures:
            v, b = _value(runs, s, m, kind), base[s]
            ratios.append(v / b if (np.isfinite(v) and np.isfinite(b) and b > 0)
                          else np.nan)
        offset = (i - (n_m - 1) / 2) * width
        ax.bar(x + offset, ratios, width, label=METHOD_LABEL[m],
               color=METHOD_COLOR[m], edgecolor="black", linewidth=0.4)
        nan_marks += [xi for xi, r in zip(x + offset, ratios) if np.isnan(r)]
    ax.axhline(1.0, color="black", ls="--", lw=1.0, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(structures)
    ax.set_ylabel(f"× relative to {METHOD_LABEL[ref]}")
    ax.set_title(title)
    ax.grid(axis="y", which="both", ls=":", alpha=0.4)
    trans = ax.get_xaxis_transform()
    for xi in nan_marks:
        ax.text(xi, 0.02, "n/a", transform=trans, ha="center", va="bottom",
                fontsize=8, rotation=90, color="gray")


def plot_device_normalized(runs, device, out_path, ref="sf_fft"):
    """Performance normalized to TorchRef SF_FFT (separate figure)."""
    if not any(r["method"] == ref for r in runs):
        print(f"Reference method {ref} absent for {device}; skipping normalized plot.")
        return
    structures = _structures_sorted(runs)
    present = {r["method"] for r in runs}
    # Plot every method except the reference itself.
    methods = [m for m in METHOD_ORDER if m in present and m != ref]
    diff_methods = [m for m in methods if m != "cctbx"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    _norm_bar_panel(axes[0], runs, structures, methods, "fwd", "Forward", ref)
    _norm_bar_panel(axes[1], runs, structures, diff_methods, "fwd_bwd",
                    "Forward + backward", ref)
    _norm_bar_panel(axes[2], runs, structures, diff_methods, "mem_fwd_bwd",
                    "Peak memory (fwd + bwd)", ref)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(methods)),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    dev_label = "GPU" if device == "gpu" else "CPU (16 threads)"
    gpu_name = next((r.get("gpu_name") for r in runs if r.get("gpu_name")), None)
    if gpu_name:
        dev_label += f" — {gpu_name}"
    fig.suptitle(f"Performance relative to TorchRef SF_FFT (higher = worse) "
                 f"[{dev_label}]", fontsize=17)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_device(runs, device, out_path):
    structures = _structures_sorted(runs)
    present = {r["method"] for r in runs}
    methods = [m for m in METHOD_ORDER if m in present]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    _bar_panel(axes[0], runs, structures, methods, "fwd",
               "time per call (s)", "Forward")
    # Backward / memory: only the differentiable methods (drop cctbx).
    diff_methods = [m for m in methods if m != "cctbx"]
    _bar_panel(axes[1], runs, structures, diff_methods, "fwd_bwd",
               "time per call (s)", "Forward + backward")
    _bar_panel(axes[2], runs, structures, diff_methods, "mem_fwd_bwd",
               "peak memory (MB)", "Peak memory (fwd + bwd)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    dev_label = "GPU" if device == "gpu" else "CPU (16 threads)"
    gpu_name = next((r.get("gpu_name") for r in runs if r.get("gpu_name")), None)
    if gpu_name:
        dev_label += f" — {gpu_name}"
    fig.suptitle(f"Structure-factor calculation: SFcalculator vs TorchRef "
                 f"[{dev_label}]", fontsize=17)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    root = Path(args.results_dir)

    # Device subdir name (as written by run_benchmarks: --device value) -> label.
    devices = {"cpu": "cpu", "cuda": "gpu"}
    any_plotted = False
    for subdir, label in devices.items():
        device_dir = root / subdir
        if not device_dir.is_dir():
            continue
        runs = load_device(device_dir)
        if not runs:
            print(f"No JSON results in {device_dir}, skipping.")
            continue
        plot_device(runs, label, root / f"figure3_sf_comparison_{label}.png")
        plot_device_normalized(
            runs, label, root / f"figure3_sf_comparison_{label}_normalized.png"
        )
        any_plotted = True

    if not any_plotted:
        print(f"No results found under {root}/(cpu|cuda).")


if __name__ == "__main__":
    main()
