#!/usr/bin/env python
"""
Plot benchmark results for refinement cycle profiling.

Generates PNG figures:
  - aggregate_times.png: Forward/backward/combined vs thread count (log scale Y)
  - aggregate_speedup.png: Speedup vs thread count
  - target_breakdown.png: Per-target stacked bar (CPU best vs GPU)
  - combined.png: Two-panel: thread scaling + target breakdown

Usage:
    python plot_results.py results_20260302_120000/
    python plot_results.py results_20260302_120000/ --output_dir plots/
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["STIXGeneral"]
plt.rcParams["mathtext.fontset"] = "stix"
import numpy as np

# --- Colors ---
# Aggregate modes
COLOR_FWD = "#2563eb"       # blue
COLOR_FWD_GRAPH = "#0ea5e9" # sky blue
COLOR_BWD = "#dc2626"       # red
COLOR_FWD_BWD = "#7c3aed"   # purple

# GPU variants (same hue, lighter / dashed)
COLOR_GPU_FWD = "#10b981"
COLOR_GPU_FG = "#34d399"
COLOR_GPU_BWD = "#f97316"
COLOR_GPU_FB = "#059669"

# Per-target breakdown
TARGET_COLORS = {
    "xray":                "#2563eb",  # blue
    "geometry/bond":       "#059669",  # green
    "geometry/angle":      "#10b981",  # light green
    "geometry/torsion":    "#34d399",  # lighter green
    "geometry/planarity":  "#6ee7b7",  # very light green
    "geometry/chiral":     "#a7f3d0",  # pale green
    "geometry/nonbonded":  "#d1fae5",  # lightest green
    "geometry/ramachandran": "#047857", # dark green
    "adp/simu":            "#dc2626",  # red
    "adp/locality":        "#f97316",  # orange
    "adp/KL":              "#fbbf24",  # yellow
}

TARGET_LABELS = {
    "xray":                  "X-ray",
    "geometry/bond":         "Bond",
    "geometry/angle":        "Angle",
    "geometry/torsion":      "Torsion",
    "geometry/planarity":    "Planarity",
    "geometry/chiral":       "Chiral",
    "geometry/nonbonded":    "Non-bonded",
    "geometry/ramachandran": "Ramachandran",
    "adp/simu":              "ADP similarity",
    "adp/locality":          "ADP locality",
    "adp/KL":                "ADP KL",
}

# Default CPU thread count for per-target breakdown plots
DEFAULT_BREAKDOWN_THREADS = 8


def load_summary(results_dir: Path) -> tuple[dict, dict | None]:
    """Load CPU results from summary.csv and GPU data."""
    csv_path = results_dir / "summary.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    # CSV field names for our benchmark
    agg_keys = [
        "agg_fwd_no_grad", "agg_fwd_graph", "agg_bwd_only", "agg_fwd_bwd",
    ]
    cpu_data = {"n_threads": []}
    for k in agg_keys:
        for suffix in ["_mean", "_min", "_max", "_speedup"]:
            cpu_data[k + suffix] = []
    for k in ["target_xray_mean", "target_geometry_total_mean", "target_adp_total_mean"]:
        cpu_data[k] = []

    gpu_data = None

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("device") == "gpu":
                gpu_data = {}
                for k in agg_keys:
                    for suffix in ["_mean", "_min", "_max", "_speedup"]:
                        gpu_data[k + suffix] = float(row[k + suffix])
                for k in ["target_xray_mean", "target_geometry_total_mean",
                           "target_adp_total_mean"]:
                    gpu_data[k] = float(row[k]) if row[k] else 0.0
                continue

            cpu_data["n_threads"].append(int(row["n_threads"]))
            for k in agg_keys:
                for suffix in ["_mean", "_min", "_max", "_speedup"]:
                    cpu_data[k + suffix].append(float(row[k + suffix]))
            for k in ["target_xray_mean", "target_geometry_total_mean",
                       "target_adp_total_mean"]:
                cpu_data[k].append(float(row[k]) if row[k] else 0.0)

    cpu_np = {k: np.array(v) for k, v in cpu_data.items()}

    # Load GPU name from gpu.json
    gpu_json_path = results_dir / "gpu.json"
    if gpu_data and gpu_json_path.exists():
        with open(gpu_json_path) as f:
            gpu_json = json.load(f)
        gpu_data["gpu_name"] = gpu_json.get("gpu_name", "GPU")

    return cpu_np, gpu_data


def load_per_target_from_json(
    results_dir: Path,
    mode: str = "fwd_bwd",
    n_threads: int = DEFAULT_BREAKDOWN_THREADS,
) -> tuple[dict | None, dict | None]:
    """Load per-target breakdown from JSON files.

    Parameters
    ----------
    results_dir : Path
        Directory containing benchmark JSON files.
    mode : str
        Which timing mode: 'forward', 'backward', or 'fwd_bwd'.
    n_threads : int
        Preferred CPU thread count. Falls back to closest available.

    Returns
    -------
    (cpu_per_target, gpu_per_target) where each is a dict
    mapping target name to mean_time (seconds), or None if unavailable.
    """
    # Load all CPU results, pick requested thread count (or closest)
    cpu_runs = {}
    for json_path in sorted(results_dir.glob("threads_*.json")):
        with open(json_path) as f:
            data = json.load(f)
        cpu_runs[data["n_threads"]] = data

    cpu_selected = None
    if cpu_runs:
        if n_threads in cpu_runs:
            cpu_selected = cpu_runs[n_threads]
        else:
            closest = min(cpu_runs.keys(), key=lambda t: abs(t - n_threads))
            cpu_selected = cpu_runs[closest]

    cpu_per_target = None
    if cpu_selected and "per_target" in cpu_selected:
        cpu_per_target = {
            name: stats[mode]["mean_time"]
            for name, stats in cpu_selected["per_target"].items()
        }
        cpu_per_target["_n_threads"] = cpu_selected["n_threads"]

    gpu_per_target = None
    gpu_json_path = results_dir / "gpu.json"
    if gpu_json_path.exists():
        with open(gpu_json_path) as f:
            data = json.load(f)
        if "per_target" in data:
            gpu_per_target = {
                name: stats[mode]["mean_time"]
                for name, stats in data["per_target"].items()
            }
            gpu_per_target["_gpu_name"] = data.get("gpu_name", "GPU")

    return cpu_per_target, gpu_per_target


def _plot_time(ax, threads, cpu, gpu, prefix, color_cpu, color_gpu,
               label_cpu, label_gpu_suffix):
    """Plot execution time curve for a given aggregate mode."""
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
            label=f"{label_gpu_suffix} ({gpu_name})",
        )
        ax.axhspan(
            gpu[f"{prefix}_min"] * 1000, gpu[f"{prefix}_max"] * 1000,
            color=color_gpu, alpha=0.1,
        )


def plot_aggregate_times(cpu: dict, gpu: dict | None, output_path: Path):
    """Aggregate forward/backward/combined vs thread count (log Y)."""
    threads = cpu["n_threads"]
    fig, ax = plt.subplots(figsize=(8, 5))

    _plot_time(ax, threads, cpu, gpu,
               "agg_fwd_no_grad", COLOR_FWD, COLOR_GPU_FWD,
               "Forward no_grad (CPU)", "Fwd no_grad")
    _plot_time(ax, threads, cpu, gpu,
               "agg_fwd_graph", COLOR_FWD_GRAPH, COLOR_GPU_FG,
               "Forward with graph (CPU)", "Fwd graph")
    _plot_time(ax, threads, cpu, gpu,
               "agg_bwd_only", COLOR_BWD, COLOR_GPU_BWD,
               "Backward only (CPU)", "Bwd only")
    _plot_time(ax, threads, cpu, gpu,
               "agg_fwd_bwd", COLOR_FWD_BWD, COLOR_GPU_FB,
               "Forward+backward (CPU)", "Fwd+bwd")

    ax.set_yscale("log")
    ax.set_xlabel("Number of CPU threads", fontsize=12)
    ax.set_ylabel("Time per cycle (ms)", fontsize=12)
    ax.set_title("Refinement Cycle — Execution Time", fontsize=13)
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0, threads.max() + 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_aggregate_speedup(cpu: dict, gpu: dict | None, output_path: Path):
    """Speedup vs thread count for all aggregate modes."""
    threads = cpu["n_threads"]
    fig, ax = plt.subplots(figsize=(8, 5))

    modes = [
        ("agg_fwd_no_grad_speedup", "o-", COLOR_FWD, "Fwd no_grad (CPU)"),
        ("agg_fwd_graph_speedup", "^-", COLOR_FWD_GRAPH, "Fwd with graph (CPU)"),
        ("agg_bwd_only_speedup", "v-", COLOR_BWD, "Bwd only (CPU)"),
        ("agg_fwd_bwd_speedup", "D-", COLOR_FWD_BWD, "Fwd+bwd (CPU)"),
    ]
    for key, fmt, color, label in modes:
        ax.plot(threads, cpu[key], fmt, color=color,
                markersize=5, linewidth=1.5, label=label)

    ax.plot(threads, threads, "--", color="grey", alpha=0.5, linewidth=1,
            label="Ideal")

    if gpu:
        gpu_name = gpu.get("gpu_name", "GPU")
        gpu_modes = [
            ("agg_fwd_no_grad_speedup", "--", COLOR_GPU_FWD, f"Fwd no_grad ({gpu_name})"),
            ("agg_fwd_graph_speedup", ":", COLOR_GPU_FG, f"Fwd graph ({gpu_name})"),
            ("agg_bwd_only_speedup", ":", COLOR_GPU_BWD, f"Bwd only ({gpu_name})"),
            ("agg_fwd_bwd_speedup", "-.", COLOR_GPU_FB, f"Fwd+bwd ({gpu_name})"),
        ]
        for key, ls, color, label in gpu_modes:
            ax.axhline(gpu[key], color=color, linestyle=ls,
                       linewidth=1.5, label=label)

    ax.set_xlabel("Number of CPU threads", fontsize=12)
    ax.set_ylabel("Speedup vs 1 CPU thread", fontsize=12)
    ax.set_title("Refinement Cycle — Speedup", fontsize=13)
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, threads.max() + 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def _stacked_barh(ax, bar_positions, all_targets, data_sets, bar_height=0.6):
    """Draw stacked horizontal bars for per-target breakdown.

    Parameters
    ----------
    ax : matplotlib Axes
    bar_positions : np.array of y positions
    all_targets : list of target name strings
    data_sets : list of dicts, one per bar, mapping target name → time (seconds)
    bar_height : float
    """
    lefts = np.zeros(len(bar_positions))
    for target_name in all_targets:
        widths = np.array([
            ds.get(target_name, 0.0) * 1000 for ds in data_sets
        ])
        color = TARGET_COLORS.get(target_name, "#888888")
        label = TARGET_LABELS.get(target_name, target_name)
        ax.barh(bar_positions, widths, left=lefts, height=bar_height,
                color=color, edgecolor="white", linewidth=0.5,
                label=label)
        lefts += widths


def plot_target_breakdown(results_dir: Path, output_path: Path):
    """Stacked horizontal bar chart of per-target times (fwd, bwd, fwd+bwd)."""
    modes = [
        ("forward", "Forward"),
        ("backward", "Backward"),
        ("fwd_bwd", "Fwd+Bwd"),
    ]

    # Load all three modes
    mode_data = {}
    for mode_key, _ in modes:
        cpu_pt, gpu_pt = load_per_target_from_json(results_dir, mode=mode_key)
        mode_data[mode_key] = (cpu_pt, gpu_pt)

    # Check we have any data
    has_any = any(
        cpu is not None or gpu is not None
        for cpu, gpu in mode_data.values()
    )
    if not has_any:
        print("No per-target data available, skipping target_breakdown.png")
        return

    # Collect target names from any available data
    all_targets = []
    for cpu_pt, gpu_pt in mode_data.values():
        for pt in [cpu_pt, gpu_pt]:
            if pt is None:
                continue
            for name in pt:
                if not name.startswith("_") and name not in all_targets:
                    all_targets.append(name)

    # Determine device labels
    sample_cpu = next((c for c, _ in mode_data.values() if c), None)
    sample_gpu = next((g for _, g in mode_data.values() if g), None)
    cpu_label = f"CPU ({sample_cpu.get('_n_threads', '?')}T)" if sample_cpu else None
    gpu_name = sample_gpu.get("_gpu_name", "GPU") if sample_gpu else None

    # Build rows: for each mode × each device
    bar_labels = []
    data_sets = []
    for mode_key, mode_label in modes:
        cpu_pt, gpu_pt = mode_data[mode_key]
        if cpu_pt is not None:
            bar_labels.append(f"{cpu_label} — {mode_label}")
            data_sets.append(cpu_pt)
        if gpu_pt is not None:
            bar_labels.append(f"GPU — {mode_label}")
            data_sets.append(gpu_pt)

    bar_positions = np.arange(len(bar_labels))

    fig, ax = plt.subplots(figsize=(11, max(4, len(bar_labels) * 0.7 + 1)))

    _stacked_barh(ax, bar_positions, all_targets, data_sets)

    ax.set_yticks(bar_positions)
    ax.set_yticklabels(bar_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_title("Per-Target Time Breakdown", fontsize=13)

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(unique_handles, unique_labels, fontsize=9,
              bbox_to_anchor=(1.02, 1), loc="upper left", title="Target")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

def plot_target_forward_all_only(results_dir: Path, output_path: Path):
    """Stacked horizontal bar chart of per-target times (fwd, bwd, fwd+bwd)."""
    modes = [
        ("forward", "Forward"),
        ("fwd_bwd", "Fwd+Bwd"),
    ]

    # Load all three modes
    mode_data = {}
    for mode_key, _ in modes:
        cpu_pt, gpu_pt = load_per_target_from_json(results_dir, mode=mode_key)
        mode_data[mode_key] = (cpu_pt, gpu_pt)

    # Check we have any data
    has_any = any(
        cpu is not None or gpu is not None
        for cpu, gpu in mode_data.values()
    )
    if not has_any:
        print("No per-target data available, skipping target_breakdown.png")
        return

    # Collect target names from any available data
    all_targets = []
    for cpu_pt, gpu_pt in mode_data.values():
        for pt in [cpu_pt, gpu_pt]:
            if pt is None:
                continue
            for name in pt:
                if not name.startswith("_") and name not in all_targets:
                    all_targets.append(name)

    # Determine device labels
    sample_cpu = next((c for c, _ in mode_data.values() if c), None)
    sample_gpu = next((g for _, g in mode_data.values() if g), None)
    cpu_label = f"CPU ({sample_cpu.get('_n_threads', '?')}T)" if sample_cpu else None
    gpu_name = sample_gpu.get("_gpu_name", "GPU") if sample_gpu else None

    # Build rows: for each mode × each device
    bar_labels = []
    data_sets = []
    for mode_key, mode_label in modes:
        cpu_pt, gpu_pt = mode_data[mode_key]
        if cpu_pt is not None:
            bar_labels.append(f"{cpu_label} — {mode_label}")
            data_sets.append(cpu_pt)
        if gpu_pt is not None:
            bar_labels.append(f"GPU — {mode_label}")
            data_sets.append(gpu_pt)

    bar_positions = np.arange(len(bar_labels))

    fig, ax = plt.subplots(figsize=(10,5))

    _stacked_barh(ax, bar_positions, all_targets, data_sets)

    ax.set_yticks(bar_positions)
    ax.set_yticklabels(bar_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)", fontsize=12)

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(unique_handles, unique_labels, fontsize=9,
              bbox_to_anchor=(1.02, 1), loc="upper left", title="Target")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

def plot_combined(cpu: dict, gpu: dict | None, results_dir: Path, output_path: Path):
    """Two-panel: (A) thread scaling for fwd+bwd, (B) per-target fwd+bwd breakdown."""
    threads = cpu["n_threads"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Panel A: Thread scaling for fwd+bwd ----
    _plot_time(ax1, threads, cpu, gpu,
               "agg_fwd_bwd", COLOR_FWD_BWD, COLOR_GPU_FB,
               "Fwd+bwd (CPU)", "Fwd+bwd")
    _plot_time(ax1, threads, cpu, gpu,
               "agg_fwd_no_grad", COLOR_FWD, COLOR_GPU_FWD,
               "Fwd no_grad (CPU)", "Fwd no_grad")
    _plot_time(ax1, threads, cpu, gpu,
               "agg_bwd_only", COLOR_BWD, COLOR_GPU_BWD,
               "Bwd only (CPU)", "Bwd only")

    ax1.set_yscale("log")
    ax1.set_xlabel("Number of CPU threads", fontsize=12)
    ax1.set_ylabel("Time per cycle (ms)", fontsize=12)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_xlim(0, threads.max() + 1)
    ax1.legend(fontsize=9, loc="upper right")

    # ---- Panel B: Per-target fwd+bwd stacked bar ----
    cpu_pt, gpu_pt = load_per_target_from_json(results_dir, mode="fwd_bwd")
    if cpu_pt is not None or gpu_pt is not None:
        all_targets = []
        for pt in [cpu_pt, gpu_pt]:
            if pt is None:
                continue
            for name in pt:
                if not name.startswith("_") and name not in all_targets:
                    all_targets.append(name)

        bar_labels = []
        data_sets = []
        if cpu_pt:
            n_threads = cpu_pt.get("_n_threads", "?")
            bar_labels.append(f"CPU ({n_threads}T)")
            data_sets.append(cpu_pt)
        if gpu_pt:
            bar_labels.append("GPU")
            data_sets.append(gpu_pt)

        bar_positions = np.arange(len(bar_labels))
        _stacked_barh(ax2, bar_positions, all_targets, data_sets)

        ax2.set_yticks(bar_positions)
        ax2.set_yticklabels(bar_labels, fontsize=11)
        ax2.invert_yaxis()
        ax2.set_xlabel("Time (ms)", fontsize=12)

        # De-duplicate legend
        handles, labels = ax2.get_legend_handles_labels()
        seen = {}
        uh, ul = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = True
                uh.append(h)
                ul.append(l)
        ax2.legend(uh, ul, fontsize=8, bbox_to_anchor=(1.02, 1),
                   loc="upper left", title="Target")
        ax2.grid(True, alpha=0.3, axis="x")

    # Panel labels
    for ax, label in [(ax1, "A"), (ax2, "B")]:
        ax.text(
            0.03, 0.95, label, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="left",
        )

    fig.suptitle("Refinement Cycle Benchmark", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_total_eval_times(results_dir: Path, output_path: Path):
    """Grouped bar chart of total forward / backward / fwd+bwd at 8 threads and GPU."""
    modes = [
        ("forward", "Forward"),
        ("backward", "Backward"),
        ("fwd_bwd", "Forward + Backward"),
    ]

    # Load CPU (8T) and GPU aggregate per-target data, sum to get totals
    totals = {}  # {(device_label, mode_label): total_ms}
    device_labels = []

    # CPU
    cpu_pt_check, _ = load_per_target_from_json(results_dir, mode="forward")
    if cpu_pt_check is not None:
        n_threads = cpu_pt_check.get("_n_threads", DEFAULT_BREAKDOWN_THREADS)
        cpu_label = f"CPU ({n_threads}T)"
        device_labels.append(cpu_label)
        for mode_key, mode_label in modes:
            cpu_pt, _ = load_per_target_from_json(results_dir, mode=mode_key)
            total = sum(v for k, v in cpu_pt.items() if not k.startswith("_"))
            totals[(cpu_label, mode_label)] = total * 1000

    # GPU
    _, gpu_pt_check = load_per_target_from_json(results_dir, mode="forward")
    if gpu_pt_check is not None:
        device_labels.append("GPU")
        for mode_key, mode_label in modes:
            _, gpu_pt = load_per_target_from_json(results_dir, mode=mode_key)
            total = sum(v for k, v in gpu_pt.items() if not k.startswith("_"))
            totals[("GPU", mode_label)] = total * 1000

    if not totals:
        print("No per-target data available, skipping total_eval_times.png")
        return

    mode_labels = [m[1] for m in modes]
    x = np.arange(len(mode_labels))
    n_devices = len(device_labels)
    bar_w = 0.7 / n_devices
    colors = [COLOR_FWD, COLOR_BWD, COLOR_FWD_BWD]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, dev_label in enumerate(device_labels):
        vals = [totals.get((dev_label, ml), 0) for ml in mode_labels]
        offset = (i - (n_devices - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, vals, bar_w,
            color=colors, edgecolor="white" if i == 0 else colors,
            alpha=1.0 if i == 0 else 0.35,
            linewidth=2 if i > 0 else 0.5,
            linestyle="--" if i > 0 else "-",
            label=dev_label,
        )
        # Annotate values
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, fontsize=12)
    ax.set_ylabel("Total evaluation time (ms)", fontsize=12)
    ax.set_title("Total Cycle Evaluation Time", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "refinement_cycle"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"


def main():
    parser = argparse.ArgumentParser(
        description="Plot Figure 3b: refinement cycle profiling results"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Path to results directory (default: data/refinement_cycle/ next to this script)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots (default: output/ next to this script)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_DATA_DIR
    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_target_forward_all_only(results_dir, out_dir / "figure3b_profiling.png")


if __name__ == "__main__":
    main()
