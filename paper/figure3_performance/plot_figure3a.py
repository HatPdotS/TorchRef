#!/usr/bin/env python
"""
Plot benchmark results for TorchRef vs cctbx.

Generates PNG figures:
  - execution_time.png: All forward variants + cctbx time vs thread count (log scale Y)
  - speedup.png: All forward variants + cctbx speedup vs thread count
  - combined.png: Side-by-side execution time + speedup with shared legend

Usage:
    python plot_thread_scaling.py results_20260225_120000/
    python plot_thread_scaling.py results_20260225_120000/ --output_dir plots/
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from matplotlib.lines import Line2D

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["STIXGeneral"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 17
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["legend.fontsize"] = 12
import numpy as np

COLOR_TR = "#2563eb"       # blue       — TorchRef fwd no_grad (CPU)
COLOR_TR_FG = "#0ea5e9"   # sky blue   — TorchRef fwd with graph (CPU)
COLOR_TR_BO = "#dc2626"   # red        — TorchRef bwd only (CPU)
COLOR_TR_FB = "#7c3aed"   # purple     — TorchRef fwd+bwd (CPU)
COLOR_CC = "#e67e22"       # orange     — cctbx
COLOR_GPU = "#10b981"      # green      — TorchRef fwd (GPU)
COLOR_GPU_FG = "#34d399"   # light green — TorchRef fwd graph (GPU)
COLOR_GPU_BO = "#f97316"   # dark orange — TorchRef bwd only (GPU)
COLOR_GPU_FB = "#059669"   # dark green — TorchRef fwd+bwd (GPU)


def load_summary(results_dir: Path) -> tuple[dict, dict | None]:
    """Load CPU results from summary.csv and GPU metadata from gpu.json."""
    csv_path = results_dir / "summary.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    cpu_data = {
        "n_threads": [],
        "torchref_mean": [], "torchref_min": [], "torchref_max": [],
        "torchref_speedup": [],
        "torchref_fwd_graph_mean": [], "torchref_fwd_graph_min": [],
        "torchref_fwd_graph_max": [], "torchref_fwd_graph_speedup": [],
        "torchref_bwd_only_mean": [], "torchref_bwd_only_min": [],
        "torchref_bwd_only_max": [], "torchref_bwd_only_speedup": [],
        "torchref_fwd_bwd_mean": [], "torchref_fwd_bwd_min": [],
        "torchref_fwd_bwd_max": [], "torchref_fwd_bwd_speedup": [],
        "cctbx_mean": [], "cctbx_min": [], "cctbx_max": [],
        "cctbx_speedup": [],
    }

    gpu_data = None

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            device = row.get("device", "cpu")
            if device == "gpu":
                gpu_data = {
                    "torchref_mean": float(row["torchref_mean"]),
                    "torchref_min": float(row["torchref_min"]),
                    "torchref_max": float(row["torchref_max"]),
                    "torchref_speedup": float(row["torchref_speedup"]),
                    "torchref_fwd_graph_mean": float(row["torchref_fwd_graph_mean"]),
                    "torchref_fwd_graph_min": float(row["torchref_fwd_graph_min"]),
                    "torchref_fwd_graph_max": float(row["torchref_fwd_graph_max"]),
                    "torchref_fwd_graph_speedup": float(row["torchref_fwd_graph_speedup"]),
                    "torchref_bwd_only_mean": float(row["torchref_bwd_only_mean"]),
                    "torchref_bwd_only_min": float(row["torchref_bwd_only_min"]),
                    "torchref_bwd_only_max": float(row["torchref_bwd_only_max"]),
                    "torchref_bwd_only_speedup": float(row["torchref_bwd_only_speedup"]),
                    "torchref_fwd_bwd_mean": float(row["torchref_fwd_bwd_mean"]),
                    "torchref_fwd_bwd_min": float(row["torchref_fwd_bwd_min"]),
                    "torchref_fwd_bwd_max": float(row["torchref_fwd_bwd_max"]),
                    "torchref_fwd_bwd_speedup": float(row["torchref_fwd_bwd_speedup"]),
                }
                continue

            cpu_data["n_threads"].append(int(row["n_threads"]))
            cpu_data["torchref_mean"].append(float(row["torchref_mean"]))
            cpu_data["torchref_min"].append(float(row["torchref_min"]))
            cpu_data["torchref_max"].append(float(row["torchref_max"]))
            cpu_data["torchref_speedup"].append(float(row["torchref_speedup"]))
            cpu_data["torchref_fwd_graph_mean"].append(float(row["torchref_fwd_graph_mean"]))
            cpu_data["torchref_fwd_graph_min"].append(float(row["torchref_fwd_graph_min"]))
            cpu_data["torchref_fwd_graph_max"].append(float(row["torchref_fwd_graph_max"]))
            cpu_data["torchref_fwd_graph_speedup"].append(float(row["torchref_fwd_graph_speedup"]))
            cpu_data["torchref_bwd_only_mean"].append(float(row["torchref_bwd_only_mean"]))
            cpu_data["torchref_bwd_only_min"].append(float(row["torchref_bwd_only_min"]))
            cpu_data["torchref_bwd_only_max"].append(float(row["torchref_bwd_only_max"]))
            cpu_data["torchref_bwd_only_speedup"].append(float(row["torchref_bwd_only_speedup"]))
            cpu_data["torchref_fwd_bwd_mean"].append(float(row["torchref_fwd_bwd_mean"]))
            cpu_data["torchref_fwd_bwd_min"].append(float(row["torchref_fwd_bwd_min"]))
            cpu_data["torchref_fwd_bwd_max"].append(float(row["torchref_fwd_bwd_max"]))
            cpu_data["torchref_fwd_bwd_speedup"].append(float(row["torchref_fwd_bwd_speedup"]))
            cpu_data["cctbx_mean"].append(float(row["cctbx_mean"]))
            cpu_data["cctbx_min"].append(float(row["cctbx_min"]))
            cpu_data["cctbx_max"].append(float(row["cctbx_max"]))
            cpu_data["cctbx_speedup"].append(float(row["cctbx_speedup"]))

    cpu_np = {k: np.array(v) for k, v in cpu_data.items()}

    # Try to load GPU name from gpu.json
    gpu_json_path = results_dir / "gpu.json"
    if gpu_data and gpu_json_path.exists():
        with open(gpu_json_path) as f:
            gpu_json = json.load(f)
        gpu_data["gpu_name"] = gpu_json.get("gpu_name", "GPU")

    return cpu_np, gpu_data


def _plot_time(ax, threads, cpu, gpu, prefix, color_cpu, color_gpu, label_cpu, label_gpu_suffix):
    """Helper to plot execution time curves for a given metric prefix (ms)."""
    mean = cpu[f"{prefix}_mean"] * 1000
    lo = mean - cpu[f"{prefix}_min"] * 1000
    hi = cpu[f"{prefix}_max"] * 1000 - mean
    ax.errorbar(
        threads, mean, yerr=[lo, hi],
        fmt="o-", color=color_cpu,
        capsize=3, capthick=1.2,
        markersize=5, linewidth=1.5,
        label=label_cpu,
    )
    if gpu:
        gpu_name = gpu.get("gpu_name", "GPU")
        ax.axhline(
            gpu[f"{prefix}_mean"] * 1000, color=color_gpu,
            linestyle="--", linewidth=1.5,
            label=f"TorchRef {label_gpu_suffix} ({gpu_name})",
        )
        ax.axhspan(
            gpu[f"{prefix}_min"] * 1000, gpu[f"{prefix}_max"] * 1000,
            color=color_gpu, alpha=0.1,
        )


def plot_execution_time(cpu: dict, gpu: dict | None, output_path: Path):
    """Plot forward execution times vs thread count (log scale Y).

    Shows all three TorchRef forward variants: no_grad, with graph, fwd+bwd.
    """
    threads = cpu["n_threads"]
    fig, ax = plt.subplots(figsize=(8, 5))

    _plot_time(ax, threads, cpu, gpu,
               "torchref", COLOR_TR, COLOR_GPU,
               "TorchRef fwd no_grad (CPU)", "fwd no_grad")

    _plot_time(ax, threads, cpu, gpu,
               "torchref_fwd_graph", COLOR_TR_FG, COLOR_GPU_FG,
               "TorchRef fwd with graph (CPU)", "fwd graph")

    _plot_time(ax, threads, cpu, gpu,
               "torchref_bwd_only", COLOR_TR_BO, COLOR_GPU_BO,
               "TorchRef bwd only (CPU)", "bwd only")

    _plot_time(ax, threads, cpu, gpu,
               "torchref_fwd_bwd", COLOR_TR_FB, COLOR_GPU_FB,
               "TorchRef fwd+bwd (CPU)", "fwd+bwd")

    # cctbx
    mean = cpu["cctbx_mean"] * 1000
    lo = mean - cpu["cctbx_min"] * 1000
    hi = cpu["cctbx_max"] * 1000 - mean
    ax.errorbar(
        threads, mean, yerr=[lo, hi],
        fmt="s-", color=COLOR_CC,
        capsize=3, capthick=1.2,
        markersize=5, linewidth=1.5,
        label="cctbx (CPU)",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Number of CPU threads")
    ax.set_ylabel("Time per Fcalc (ms)")
    ax.set_title("Structure Factor Calculation — Execution Time")
    ax.legend(fontsize=12, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0, threads.max() + 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()




def plot_speedup(cpu: dict, gpu: dict | None, output_path: Path):
    """Plot speedup vs thread count for all TorchRef variants + cctbx."""
    threads = cpu["n_threads"]
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        threads, cpu["torchref_speedup"], "o-",
        color=COLOR_TR, markersize=5, linewidth=1.5,
        label="TorchRef fwd no_grad (CPU)",
    )
    ax.plot(
        threads, cpu["torchref_fwd_graph_speedup"], "^-",
        color=COLOR_TR_FG, markersize=5, linewidth=1.5,
        label="TorchRef fwd with graph (CPU)",
    )
    ax.plot(
        threads, cpu["torchref_bwd_only_speedup"], "v-",
        color=COLOR_TR_BO, markersize=5, linewidth=1.5,
        label="TorchRef bwd only (CPU)",
    )
    ax.plot(
        threads, cpu["torchref_fwd_bwd_speedup"], "D-",
        color=COLOR_TR_FB, markersize=5, linewidth=1.5,
        label="TorchRef fwd+bwd (CPU)",
    )
    ax.plot(
        threads, cpu["cctbx_speedup"], "s-",
        color=COLOR_CC, markersize=5, linewidth=1.5,
        label="cctbx (CPU)",
    )
    ax.plot(
        threads, threads, "--",
        color="grey", alpha=0.5, linewidth=1,
        label="Ideal",
    )
    if gpu:
        gpu_name = gpu.get("gpu_name", "GPU")
        ax.axhline(
            gpu["torchref_speedup"], color=COLOR_GPU,
            linestyle="--", linewidth=1.5,
            label=f"TorchRef fwd no_grad ({gpu_name})",
        )
        ax.axhline(
            gpu["torchref_fwd_graph_speedup"], color=COLOR_GPU_FG,
            linestyle=":", linewidth=1.5,
            label=f"TorchRef fwd graph ({gpu_name})",
        )
        ax.axhline(
            gpu["torchref_bwd_only_speedup"], color=COLOR_GPU_BO,
            linestyle=":", linewidth=1.5,
            label=f"TorchRef bwd only ({gpu_name})",
        )
        ax.axhline(
            gpu["torchref_fwd_bwd_speedup"], color=COLOR_GPU_FB,
            linestyle="-.", linewidth=1.5,
            label=f"TorchRef fwd+bwd ({gpu_name})",
        )

    ax.set_xlabel("Number of CPU threads")
    ax.set_ylabel("Speedup vs 1 CPU thread")
    ax.set_title("Structure Factor Calculation — Speedup")
    ax.legend(fontsize=12, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, threads.max() + 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined(cpu: dict, gpu: dict | None, output_path: Path):
    """Combined plot: execution time (left) + speedup (right), legend outside."""
    threads = cpu["n_threads"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Execution time (log scale) ---
    _plot_time(ax1, threads, cpu, gpu,
               "torchref", COLOR_TR, COLOR_GPU,
               "TorchRef fwd no_grad (CPU)", "fwd no_grad")
    _plot_time(ax1, threads, cpu, gpu,
               "torchref_fwd_graph", COLOR_TR_FG, COLOR_GPU_FG,
               "TorchRef fwd with graph (CPU)", "fwd graph")
    _plot_time(ax1, threads, cpu, gpu,
               "torchref_bwd_only", COLOR_TR_BO, COLOR_GPU_BO,
               "TorchRef bwd only (CPU)", "bwd only")
    _plot_time(ax1, threads, cpu, gpu,
               "torchref_fwd_bwd", COLOR_TR_FB, COLOR_GPU_FB,
               "TorchRef fwd+bwd (CPU)", "fwd+bwd")

    mean = cpu["cctbx_mean"] * 1000
    lo = mean - cpu["cctbx_min"] * 1000
    hi = cpu["cctbx_max"] * 1000 - mean
    ax1.errorbar(
        threads, mean, yerr=[lo, hi],
        fmt="s-", color=COLOR_CC,
        capsize=3, capthick=1.2,
        markersize=5, linewidth=1.5,
        label="cctbx (CPU)",
    )

    ax1.set_yscale("log")
    ax1.set_xlabel("Number of CPU threads")
    ax1.set_ylabel("Time per Fcalc (ms)")
    ax1.set_title("Execution Time")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_xlim(0, threads.max() + 1)

    # --- Right: Speedup ---
    ax2.plot(threads, cpu["torchref_speedup"], "o-",
             color=COLOR_TR, markersize=5, linewidth=1.5,
             label="TorchRef fwd no_grad (CPU)")
    ax2.plot(threads, cpu["torchref_fwd_graph_speedup"], "^-",
             color=COLOR_TR_FG, markersize=5, linewidth=1.5,
             label="TorchRef fwd with graph (CPU)")
    ax2.plot(threads, cpu["torchref_bwd_only_speedup"], "v-",
             color=COLOR_TR_BO, markersize=5, linewidth=1.5,
             label="TorchRef bwd only (CPU)")
    ax2.plot(threads, cpu["torchref_fwd_bwd_speedup"], "D-",
             color=COLOR_TR_FB, markersize=5, linewidth=1.5,
             label="TorchRef fwd+bwd (CPU)")
    ax2.plot(threads, cpu["cctbx_speedup"], "s-",
             color=COLOR_CC, markersize=5, linewidth=1.5,
             label="cctbx (CPU)")
    ax2.plot(threads, threads, "--",
             color="grey", alpha=0.5, linewidth=1,
             label="Ideal")
    if gpu:
        gpu_name = gpu.get("gpu_name", "GPU")
        ax2.axhline(gpu["torchref_speedup"], color=COLOR_GPU,
                     linestyle="--", linewidth=1.5,
                     label=f"TorchRef fwd no_grad ({gpu_name})")
        ax2.axhline(gpu["torchref_fwd_graph_speedup"], color=COLOR_GPU_FG,
                     linestyle=":", linewidth=1.5,
                     label=f"TorchRef fwd graph ({gpu_name})")
        ax2.axhline(gpu["torchref_bwd_only_speedup"], color=COLOR_GPU_BO,
                     linestyle=":", linewidth=1.5,
                     label=f"TorchRef bwd only ({gpu_name})")
        ax2.axhline(gpu["torchref_fwd_bwd_speedup"], color=COLOR_GPU_FB,
                     linestyle="-.", linewidth=1.5,
                     label=f"TorchRef fwd+bwd ({gpu_name})")

    ax2.set_xlabel("Number of CPU threads")
    ax2.set_ylabel("Speedup vs 1 CPU thread")
    ax2.set_title("Speedup")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, threads.max() + 1)

    # Shared legend outside on the right
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=12,
               bbox_to_anchor=(1.22, 0.5))

    fig.suptitle("Structure Factor Calculation Benchmark", fontsize=17, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

def plot_calc(cpu: dict, gpu: dict | None, output_path: Path):
    """Two-panel figure: (a) fwd+grad thread scaling, (b) fwd/grad/combined bars."""
    from matplotlib.patches import Patch

    color_fwd = "#2563eb"  # blue — forward
    color_bwd = "#dc2626"  # red  — gradient
    color_cc  = "#e67e22"  # orange — cctbx
    color_cmb = "#7c3aed"  # purple — combined

    # Limit to max 8 CPU threads
    all_threads = cpu["n_threads"]
    mask = all_threads <= 8
    cpu_plot = {k: v[mask] if isinstance(v, np.ndarray) and v.shape == all_threads.shape else v
                for k, v in cpu.items()}
    threads = cpu_plot["n_threads"]
    gpu_name = gpu.get("gpu_name", "GPU") if gpu else "GPU"

    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))

    # ---- ax1: Thread scaling for forward, gradient, cctbx ----
    # TorchRef forward (CPU)
    mean = cpu_plot["torchref_fwd_graph_mean"] * 1000
    lo = mean - cpu_plot["torchref_fwd_graph_min"] * 1000
    hi = cpu_plot["torchref_fwd_graph_max"] * 1000 - mean
    ax1.errorbar(
        threads, mean, yerr=[lo, hi],
        fmt="o-", color=color_fwd,
        capsize=3, capthick=1.2, markersize=5, linewidth=1.5
    )

    # TorchRef gradient (CPU)
    mean = cpu_plot["torchref_bwd_only_mean"] * 1000
    lo = mean - cpu_plot["torchref_bwd_only_min"] * 1000
    hi = cpu_plot["torchref_bwd_only_max"] * 1000 - mean
    ax1.errorbar(
        threads, mean, yerr=[lo, hi],
        fmt="^-", color=color_bwd,
        capsize=3, capthick=1.2, markersize=5, linewidth=1.5,
    )

    # cctbx CPU
    mean = cpu_plot["cctbx_mean"] * 1000
    lo = mean - cpu_plot["cctbx_min"] * 1000
    hi = cpu_plot["cctbx_max"] * 1000 - mean
    ax1.errorbar(
        threads, mean, yerr=[lo, hi],
        fmt="s-", color=color_cc,
        capsize=3, capthick=1.2, markersize=5, linewidth=1.5,
    )

    # GPU as dashed horizontal lines with labels
    if gpu:
        gpu_fwd_mean = gpu["torchref_fwd_graph_mean"] * 1000
        gpu_bwd_mean = gpu["torchref_bwd_only_mean"] * 1000
        ax1.axhline(gpu_fwd_mean, color=color_fwd, linestyle="--", linewidth=1.5, alpha=0.7)
        ax1.axhline(gpu_bwd_mean, color=color_bwd, linestyle="--", linewidth=1.5, alpha=0.7)
        # Labels: gradient above its line, forward below its line
        # Use multiplicative offset for log-scale y-axis
        x_label = threads.max() + 0.3
        ax1.text(x_label, gpu_bwd_mean * 1.25, "GPU", color=color_bwd,
                 fontsize=11, va="bottom", ha="left", fontstyle="italic")
        ax1.text(x_label, gpu_fwd_mean / 1.25, "GPU", color=color_fwd,
                 fontsize=11, va="top", ha="left", fontstyle="italic")

    ax1.set_yscale("log")
    ax1.set_xlabel("Number of CPU threads")
    ax1.set_ylabel("Time (ms)")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_xlim(0, threads.max() + 1)

    # Shared y-limits
    ax1.set_ylim(0.1, 100)

    # Legend: color patches for categories + hatching for device
    legend_handles = [
        Line2D([0], [0], color=color_fwd, linewidth=2, marker='o', markersize=6, label=r"$F_{\mathrm{calc}}$"),
        Line2D([0], [0], color=color_bwd, linewidth=2, marker='^', markersize=6, label="Gradient"),
        Line2D([0], [0], color="grey", linewidth=2, label="CPU"),
        Line2D([0], [0], color="grey", linewidth=2, linestyle="--", label=f"GPU ({gpu_name})"),
        Line2D([0], [0], color=color_cc, linewidth=2, marker='s', markersize=6, label="cctbx (CPU)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center", ncol=len(legend_handles),

    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

def plot_fwd_bwd_comparison(cpu: dict, gpu: dict | None, output_path: Path):
    """Two-panel figure: (a) fwd+grad thread scaling, (b) fwd/grad/combined bars."""
    from matplotlib.patches import Patch

    color_fwd = "#2563eb"  # blue — forward
    color_bwd = "#dc2626"  # red  — gradient
    color_cc  = "#e67e22"  # orange — cctbx
    color_cmb = "#7c3aed"  # purple — combined

    # Limit to max 8 CPU threads
    all_threads = cpu["n_threads"]
    mask = all_threads <= 8
    cpu_plot = {k: v[mask] if isinstance(v, np.ndarray) and v.shape == all_threads.shape else v
                for k, v in cpu.items()}
    threads = cpu_plot["n_threads"]
    gpu_name = gpu.get("gpu_name", "GPU") if gpu else "GPU"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # ---- ax1: Thread scaling for forward, gradient, cctbx ----
    # TorchRef forward (CPU)
    mean = cpu_plot["torchref_fwd_graph_mean"] * 1000
    lo = mean - cpu_plot["torchref_fwd_graph_min"] * 1000
    hi = cpu_plot["torchref_fwd_graph_max"] * 1000 - mean
    ax1.errorbar(
        threads, mean, yerr=[lo, hi],
        fmt="o-", color=color_fwd,
        capsize=3, capthick=1.2, markersize=5, linewidth=1.5,
    )

    # TorchRef gradient (CPU)
    mean = cpu_plot["torchref_bwd_only_mean"] * 1000
    lo = mean - cpu_plot["torchref_bwd_only_min"] * 1000
    hi = cpu_plot["torchref_bwd_only_max"] * 1000 - mean
    ax1.errorbar(
        threads, mean, yerr=[lo, hi],
        fmt="^-", color=color_bwd,
        capsize=3, capthick=1.2, markersize=5, linewidth=1.5,
    )

    # cctbx CPU
    mean = cpu_plot["cctbx_mean"] * 1000
    lo = mean - cpu_plot["cctbx_min"] * 1000
    hi = cpu_plot["cctbx_max"] * 1000 - mean
    ax1.errorbar(
        threads, mean, yerr=[lo, hi],
        fmt="s-", color=color_cc,
        capsize=3, capthick=1.2, markersize=5, linewidth=1.5,
    )

    # GPU as dashed horizontal lines (no shading)
    if gpu:
        for prefix, color in [
            ("torchref_fwd_graph", color_fwd),
            ("torchref_bwd_only", color_bwd),
        ]:
            gpu_mean = gpu[f"{prefix}_mean"] * 1000
            ax1.axhline(gpu_mean, color=color, linestyle="--", linewidth=1.5, alpha=0.7)

    ax1.set_yscale("log")
    ax1.set_xlabel("Number of CPU threads")
    ax1.set_ylabel("Time (ms)")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_xlim(0, threads.max() + 1)

    # Panel labels
    for ax, label in [(ax1, "A"), (ax2, "B")]:
        ax.text(
            0.03, 0.95, label, transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top", ha="left",
        )

    # ---- ax2: Forward / Backward / Combined bar chart ----
    if gpu:
        best_idx = int(np.argmin(cpu_plot["torchref_fwd_bwd_mean"]))
        best_threads = int(cpu_plot["n_threads"][best_idx])

        labels = [r"$F_{\mathrm{calc}}$", "Gradient", r"$F_{\mathrm{calc}}$" + " +\nGradient"]
        bar_colors = [color_fwd, color_bwd, color_cmb]

        cpu_means = np.array([
            cpu_plot["torchref_fwd_graph_mean"][best_idx],
            cpu_plot["torchref_bwd_only_mean"][best_idx],
            cpu_plot["torchref_fwd_bwd_mean"][best_idx],
        ])
        cpu_mins = np.array([
            cpu_plot["torchref_fwd_graph_min"][best_idx],
            cpu_plot["torchref_bwd_only_min"][best_idx],
            cpu_plot["torchref_fwd_bwd_min"][best_idx],
        ])
        cpu_maxs = np.array([
            cpu_plot["torchref_fwd_graph_max"][best_idx],
            cpu_plot["torchref_bwd_only_max"][best_idx],
            cpu_plot["torchref_fwd_bwd_max"][best_idx],
        ])
        gpu_means = np.array([
            gpu["torchref_fwd_graph_mean"],
            gpu["torchref_bwd_only_mean"],
            gpu["torchref_fwd_bwd_mean"],
        ])
        gpu_mins = np.array([
            gpu["torchref_fwd_graph_min"],
            gpu["torchref_bwd_only_min"],
            gpu["torchref_fwd_bwd_min"],
        ])
        gpu_maxs = np.array([
            gpu["torchref_fwd_graph_max"],
            gpu["torchref_bwd_only_max"],
            gpu["torchref_fwd_bwd_max"],
        ])

        x = np.arange(len(labels))
        bar_w = 0.32

        cpu_err_lo = cpu_means - cpu_mins
        cpu_err_hi = cpu_maxs - cpu_means
        for i in range(len(labels)):
            ax2.bar(
                x[i] - bar_w / 2, cpu_means[i] * 1000, bar_w,
                yerr=[[cpu_err_lo[i] * 1000], [cpu_err_hi[i] * 1000]],
                color=bar_colors[i], edgecolor="white", capsize=4,
            )

        gpu_err_lo = gpu_means - gpu_mins
        gpu_err_hi = gpu_maxs - gpu_means
        for i in range(len(labels)):
            ax2.bar(
                x[i] + bar_w / 2, gpu_means[i] * 1000, bar_w,
                yerr=[[gpu_err_lo[i] * 1000], [gpu_err_hi[i] * 1000]],
                color=bar_colors[i], alpha=0.3,
                edgecolor=bar_colors[i], linewidth=2.5,
                capsize=4, linestyle="--",
            )

        # Annotate bars
        for i in range(len(labels)):
            ax2.text(
                x[i] - bar_w / 2,
                cpu_means[i] * 1000 + cpu_err_hi[i] * 1000 + 0.3,
                f"{cpu_means[i] * 1000:.1f}",
                ha="center", va="bottom", fontsize=11,
            )
            ax2.text(
                x[i] + bar_w / 2,
                gpu_means[i] * 1000 + gpu_err_hi[i] * 1000 + 0.3,
                f"{gpu_means[i] * 1000:.1f}",
                ha="center", va="bottom", fontsize=11,
            )

        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=14)

    ax2.grid(True, alpha=0.3, which="both", axis="y")

    # Shared y-limits (sharey syncs both axes)
    ax1.set_ylim(0.1, 100)

    # Legend: color patches for categories + hatching for device
    legend_handles = [
        Patch(facecolor=color_fwd, label=r"$F_{\mathrm{calc}}$"),
        Patch(facecolor=color_bwd, label="Gradient"),
        Patch(facecolor=color_cmb, label=r"$F_{\mathrm{calc}}$" + " + Gradient"),
        Patch(facecolor="grey", label="CPU"),
        Patch(facecolor="grey", alpha=0.3, edgecolor="grey",
              linewidth=2.5, linestyle="--", label=f"GPU ({gpu_name})"),
        Patch(facecolor=color_cc, label="cctbx (CPU)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center", ncol=len(legend_handles),
        fontsize=14, framealpha=0.9,
        bbox_to_anchor=(0.5, 1.05),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "fcalc"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"


def main():
    parser = argparse.ArgumentParser(
        description="Plot Figure 3a: Fcalc benchmark results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to results directory (default: data/fcalc/ next to this script)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: output/ next to this script)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_DATA_DIR
    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu, gpu = load_summary(results_dir)

    plot_calc(cpu, gpu, out_dir / "figure3a_fcalc.png")


if __name__ == "__main__":
    main()
