#!/usr/bin/env python
"""Figure 3b — per-target refinement-cycle time breakdown (one structure).

Stacked horizontal bars of the per-target time for one structure (default 1DAW),
for CPU (8 threads) and GPU, in forward and forward+backward passes. Each bar
stacks the individually-timed targets (X-ray dominates); the hollow red box is
the independently-measured full cycle.

The gap between the stack and the red box is physically meaningful and opposite
on the two devices: on GPU the full cycle runs *faster* than the sum of isolated
targets (successive target kernels overlap/pipeline on the device — they cannot
when each is timed alone), so the red box sits inside the stack; on CPU the two
agree to within ~1%. (Explain this in the figure caption.)

Usage:
    plot_figure3b.py --results-dir data/refinement_cycle/results_XXXX \
                     [--output-dir output] [--breakdown-structure 1DAW]
"""

import argparse
from pathlib import Path

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
import json
import numpy as np

# --- Per-target colours / labels ---
TARGET_COLORS = {
    "xray":                  "#2563eb",  # blue
    "geometry/bond":         "#059669",  # green
    "geometry/angle":        "#10b981",
    "geometry/torsion":      "#34d399",
    "geometry/planarity":    "#6ee7b7",
    "geometry/chiral":       "#a7f3d0",
    "geometry/nonbonded":    "#d1fae5",
    "geometry/ramachandran": "#047857",
    "adp/simu":              "#dc2626",  # red
    "adp/locality":          "#f97316",  # orange
    "adp/KL":                "#fbbf24",  # yellow
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

DEFAULT_BREAKDOWN_THREADS = 8

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "refinement_cycle"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"


def load_per_target_from_json(
    results_dir: Path,
    mode: str = "fwd_bwd",
    n_threads: int = DEFAULT_BREAKDOWN_THREADS,
    structure: str | None = None,
) -> tuple[dict | None, dict | None]:
    """Load per-target breakdown from JSON files.

    Returns (cpu_per_target, gpu_per_target); each maps target name -> mean_time
    (seconds) plus meta keys ``_n_threads`` / ``_gpu_name`` / ``_aggregate_time``.
    Files use the flat multi-structure layout ``{structure}_threads_NN.json`` /
    ``{structure}_gpu.json``; if ``structure`` is None the first found is used.
    """
    cpu_glob = (f"{structure}_threads_*.json" if structure else "*threads_*.json")
    candidates = sorted(results_dir.glob(cpu_glob))
    if structure is None and candidates:
        stem = candidates[0].name
        prefix = stem.split("threads_")[0]
        structure = prefix.rstrip("_") or None
        if structure:
            candidates = sorted(results_dir.glob(f"{structure}_threads_*.json"))

    cpu_runs = {}
    for json_path in candidates:
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

    aggregate_key = {
        "forward": "aggregate_fwd_no_grad",
        "backward": "aggregate_bwd_only",
        "fwd_bwd": "aggregate_fwd_bwd",
    }.get(mode)

    cpu_per_target = None
    if cpu_selected and "per_target" in cpu_selected:
        cpu_per_target = {
            name: stats[mode]["mean_time"]
            for name, stats in cpu_selected["per_target"].items()
        }
        cpu_per_target["_n_threads"] = cpu_selected["n_threads"]
        if aggregate_key and aggregate_key in cpu_selected:
            cpu_per_target["_aggregate_time"] = cpu_selected[aggregate_key]["mean_time"]

    gpu_per_target = None
    gpu_json_path = (results_dir / f"{structure}_gpu.json" if structure
                     else results_dir / "gpu.json")
    if gpu_json_path.exists():
        with open(gpu_json_path) as f:
            data = json.load(f)
        if "per_target" in data:
            gpu_per_target = {
                name: stats[mode]["mean_time"]
                for name, stats in data["per_target"].items()
            }
            gpu_per_target["_gpu_name"] = data.get("gpu_name", "GPU")
            if aggregate_key and aggregate_key in data:
                gpu_per_target["_aggregate_time"] = data[aggregate_key]["mean_time"]

    return cpu_per_target, gpu_per_target


def _stacked_barh(ax, bar_positions, all_targets, data_sets, bar_height=0.6):
    """Draw stacked horizontal bars, plus a hollow red full-cycle box per bar.

    Per-target measurements pay setup costs (MixedTensor / SF cache rebuilds,
    per-target get_data caches) that a real cycle amortizes, so the stacked sum
    over-estimates the cycle. The red rectangle from x=0 to ``_aggregate_time``
    shows the true cycle time on top of the stack (see module docstring for the
    CPU/GPU direction of the gap).
    """
    lefts = np.zeros(len(bar_positions))
    for target_name in all_targets:
        widths = np.array([ds.get(target_name, 0.0) * 1000 for ds in data_sets])
        color = TARGET_COLORS.get(target_name, "#888888")
        label = TARGET_LABELS.get(target_name, target_name)
        ax.barh(bar_positions, widths, left=lefts, height=bar_height,
                color=color, edgecolor="white", linewidth=0.5, label=label)
        lefts += widths

    overlay_label_done = False
    for y_pos, ds in zip(bar_positions, data_sets):
        agg_s = ds.get("_aggregate_time")
        if agg_s is None:
            continue
        ax.barh([y_pos], [agg_s * 1000], left=0, height=bar_height,
                facecolor="none", edgecolor="red", linewidth=2.0, zorder=5,
                label=("Full cycle (aggregate)" if not overlay_label_done else None))
        overlay_label_done = True


def plot_target_breakdown(results_dir: Path, output_path: Path,
                          structure: str | None = None):
    """Stacked per-target time breakdown (forward and fwd+bwd, CPU + GPU)."""
    modes = [("forward", "Forward"), ("fwd_bwd", "Fwd+Bwd")]

    mode_data = {}
    for mode_key, _ in modes:
        cpu_pt, gpu_pt = load_per_target_from_json(
            results_dir, mode=mode_key, structure=structure)
        mode_data[mode_key] = (cpu_pt, gpu_pt)

    if not any(c is not None or g is not None for c, g in mode_data.values()):
        print("No per-target data available, skipping target_breakdown.png")
        return

    all_targets = []
    for cpu_pt, gpu_pt in mode_data.values():
        for pt in (cpu_pt, gpu_pt):
            if pt is None:
                continue
            for name in pt:
                if not name.startswith("_") and name not in all_targets:
                    all_targets.append(name)

    sample_cpu = next((c for c, _ in mode_data.values() if c), None)
    sample_gpu = next((g for _, g in mode_data.values() if g), None)
    cpu_label = f"CPU ({sample_cpu.get('_n_threads', '?')}T)" if sample_cpu else None

    bar_labels, data_sets = [], []
    for mode_key, mode_label in modes:
        cpu_pt, gpu_pt = mode_data[mode_key]
        if cpu_pt is not None:
            bar_labels.append(f"{cpu_label} — {mode_label}")
            data_sets.append(cpu_pt)
        if gpu_pt is not None:
            bar_labels.append(f"GPU — {mode_label}")
            data_sets.append(gpu_pt)

    bar_positions = np.arange(len(bar_labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    _stacked_barh(ax, bar_positions, all_targets, data_sets)
    ax.set_yticks(bar_positions)
    ax.set_yticklabels(bar_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)")

    handles, labels = ax.get_legend_handles_labels()
    seen, uh, ul = {}, [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            uh.append(h)
            ul.append(l)
    ax.legend(uh, ul, fontsize=12, bbox_to_anchor=(1.02, 1), loc="upper left",
              title="Target")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def _resolve_structure(results_dir: Path, requested: str | None) -> str | None:
    """Use `requested` if its files exist, else the first structure present."""
    if requested and sorted(results_dir.glob(f"{requested}_threads_*.json")):
        return requested
    hits = sorted(results_dir.glob("*_threads_*.json"))
    if not hits:
        return requested
    return hits[0].name.split("_threads_")[0]


def main():
    parser = argparse.ArgumentParser(
        description="Figure 3b: per-target refinement-cycle breakdown")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="refinement_cycle results dir (default: data/refinement_cycle/)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: output/ next to this script)")
    parser.add_argument("--breakdown-structure", type=str, default="1DAW",
                        help="Structure for the breakdown (default: 1DAW).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_DATA_DIR
    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    structure = _resolve_structure(results_dir, args.breakdown_structure)
    plot_target_breakdown(
        results_dir, out_dir / "figure3b_target_breakdown.png", structure=structure)


if __name__ == "__main__":
    main()
