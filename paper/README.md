# Paper Figures — TorchRef IUCrJ Publication

This folder contains scripts, data, and figures for the TorchRef paper.
All paths are relative — no hardcoded absolute paths.

## Prerequisites

```bash
pip install torchref[dev]          # TorchRef with dev dependencies
module load ccp4                    # REFMAC5 for validation
module load phenix/phenix-1.20-4459 # PHENIX for comparison refinement
```

## Data Layout

The `paper/` directory expects two data sources linked as:

```
paper/
├── data -> ../scientific_testing/data            # ~1000 benchmark structures
│   └── {CODE}/{CODE}.mtz, {CODE}.pdb, ...        # reflections + deposited model
├── phenix_refinements -> .../refinements         # PHENIX comparison results
│   └── {CODE}/{CODE}_refined_001.pdb, .log, ...
```

Figure 2 refines **Phaser-placed AlphaFold models**, not the deposited or perturbed
coordinates: its starting models are `figure2_alphafold_start/placed/{CODE}_af.pdb`, and the
structure set is the MR-solved subset recorded in `figure2_alphafold_start/mr_status.csv`
(`solved == 1`, n = 767). Those placed models plus `manifest.json` / `mr_status.csv` are
*inputs* — produced once by the molecular-replacement pipeline in
`figure2_alphafold_start/` (`fetch_alphafold.py` → `prepare_search_model.py` →
`run_mr_one.sh` → `collect_mr_status.py`) and reused by every arm thereafter.

## Figures

Every `plot_*.py` also writes per-panel **source data** to `source_data/` as it renders — one
CSV holding the values that panel draws, after all filtering and derivation. See
[`source_data/README.md`](source_data/README.md) for the file list, units and caveats. There is
no separate export step: re-render a figure and its CSVs update with it.

### Figure 2: AlphaFold-start Benchmark
**Location:** `figure2_alphafold_start/`

TorchRef vs REFMAC5 vs phenix.refine, all four models (including the AlphaFold starting
point) scored by the same independent validator, `phenix.model_vs_data`. Panels: A
R-work/R-free, B geometry RMSZ, C wall-clock, D convergence.

Submit the three arms with `analysis/submit_fig2_arms.py`, which emits each structure's
arms adjacently on one pinned CPU model — Panel C compares wall-clock across engines, so
submitting one engine at a time confounds engine identity with cluster-load window. Then
`run_af_pipeline.py validate` (REFMAC 0-cycle), the two scoring sweeps, and
`analysis/aggregate_figure_metrics.py`; render with `plot_figure_af.py`.

### Figure 3: Performance Benchmarks
**Location:** `figure3_performance/`

- **Panel A**: Structure factor (Fcalc) thread scaling and GPU performance vs cctbx.
- **Panel B**: Full refinement cycle profiling with per-target breakdown.

Benchmark scripts in `fcalc_benchmark/` and `refinement_cycle_benchmark/`.
Test data: `data/1DAW.pdb` and `data/1DAW.mtz` (included in-tree).

### Figure 4: Difference Refinement
**Location:** `figure4_difference_refinement/`

Time-resolved difference refinement of IBL isomerisation in tubulin.
All input data and refinement outputs included in the directory.

### Extended Figures
**Location:** `extended_figures/`

| Figure | Description | Data status |
|--------|-------------|-------------|
| ExtFig 1 | AF-start loss-weight landscape (10×10 geometry×adp grid) | Plotting only — uses `figure2_alphafold_start/runs/metrics/weight_grid.csv` |
| ExtFig 2 | R-factor gap vs resolution (TorchRef − PHENIX & − REFMAC, AF-start, PHENIX-scored) | Plotting only — uses `figure2_alphafold_start/runs/metrics/fig_crossscore.csv` |
| ExtFig 3 | R-factor scorer consistency (REFMAC / PHENIX / TorchRef cross-scoring) | Plotting only — uses `figure2_alphafold_start/runs/metrics/fig_crossscore.csv` |
| ExtFig 4 | Single-core wall-clock runtime, 3 engines (reproduces Fig 2c at 1 CPU core, AF-start, n=715) | Requires re-running all three engines — `exF4/submit_singlecore.py` (2145 cells) then `aggregate_singlecore.py` |
| ExtFig 5 | X-ray target-mode comparison (`ml` / `ml_noalpha` / `ml_full` / `nll_beta`, AF-start, PHENIX-scored) | Plotting only — uses `figure2_alphafold_start/runs/metrics/mode_*.csv` |

See `extended_figures/instructions.md` for the design document.

### ExtFig 5: reproducing the target-mode comparison

Answers the reviewer question "does the choice of maximum-likelihood variant matter?" Reuses
Figure 2's exact 2×2 layout via `plot_figure_af.py --prefix/--engines`, so the two figures are
directly comparable — same scorer (`phenix.model_vs_data`), same conserved-set discipline.

The `ml` cell **is** Figure 2's `torchref` arm; only the other three are extra runs.

```bash
cd paper/figure2_alphafold_start
for m in ml_noalpha nll_beta; do
    ../../.dev/bin/python analysis/submit_local_arm.py --arm mode_$m --xray-mode $m
done
# ml_full costs ~4x per gradient (32-node quadrature)
../../.dev/bin/python analysis/submit_local_arm.py --arm mode_ml_full --xray-mode ml_full \
    --partition day --time 04:00:00 --mem 16G

# PHENIX-score the new arms (resume-safe; only the missing tasks are emitted)
../../.dev/bin/python analysis/build_crossscore_worklist.py \
    --arm ml_noalpha=mode_ml_noalpha --arm ml_full=mode_ml_full --arm nll_beta=mode_nll_beta
sbatch --array=1-N%60 analysis/crossscore_array.sh

../../.dev/bin/python analysis/aggregate_figure_metrics.py --prefix mode \
    --arm ml=torchref:torchref --arm ml_noalpha=mode_ml_noalpha:torchref \
    --arm ml_full=mode_ml_full:torchref --arm nll_beta=mode_nll_beta:torchref

../../.dev/bin/python plot_figure_af.py --prefix mode \
    -o ../extended_figures/exF5/output/extended_figure5.png \
    --engines ml_noalpha=ml_noalpha:#2166ac ml=ml:#b2182b \
              ml_full=ml_full:#762a83 nll_beta=nll_beta:#e08214
```

Before submitting anything, `../../.dev/bin/python check_submitter_flags.py` asserts every
submitter still emits flags `refine.py` accepts — a removed or renamed flag otherwise kills
several thousand jobs at argparse, one at a time.
