# Source data

One CSV per figure panel, holding the values that panel **draws** — after every filter and
derivation the plot script applies. Each panel can be redrawn from its CSV alone, with no run
directories and nothing left to re-derive.

## Regenerating

These files are written *by the plot scripts*, as a side effect of rendering. There is no
separate export step, deliberately: a standalone exporter would be a second implementation of
each panel's filtering and would drift out of step (the first attempt at one silently wrote 0
rows for ExtFig 3 because it guessed the wrong pivot columns).

So: re-render a figure and its CSVs update.

```bash
cd paper/figure2_alphafold_start
../../.dev/bin/python plot_figure_af.py                                  # Figure 2
../../.dev/bin/python plot_figure_af.py --prefix mode \
    -o ../extended_figures/exF5/output/extended_figure5.png \
    --engines ml_noalpha=ml_noalpha:#2166ac ml=ml:#b2182b \
              ml_full=ml_full:#762a83 nll_beta=nll_beta:#e08214         # ExtFig 5

cd ../figure3_performance
../../.dev/bin/python plot_figure3a.py --fcalc-dir data/fcalc/results_<TS> \
                                      --sf-dir SF_calc_comparison/results_<TS>
../../.dev/bin/python plot_figure3b.py --results-dir data/refinement_cycle/results_<TS>

cd ../extended_figures
(cd exF1 && ../../../.dev/bin/python plot_exF1.py)
(cd exF2 && ../../../.dev/bin/python plot_exF2.py)
(cd exF3 && ../../../.dev/bin/python plot_exF3.py)
(cd exF4 && ../../../.dev/bin/python plot_singlecore.py)
```

The shared writer is `paper/figure_source_data.py`. It refuses to write a 0-row file (a
0-row source-data CSV looks like a successful export of an empty panel).

## Files

### Figure 2 — AlphaFold-start benchmark (n=723 conserved structures, PHENIX-scored)

| file | panel | rows |
|---|---|---|
| `figure2_panelA_rfactors.csv` | A, R-work vs R-free scatter | one per plotted point |
| `figure2_panelA_paired_delta.csv` | A, paired ΔR-free / ΔR-work vs each reference | one per reference x metric |
| `figure2_panelB_geometry_rmsz.csv` | B, geometry strips | one per engine x metric |
| `figure2_panelC_runtime.csv` | C, runtime box plot | one per plotted point |
| `figure2_panelD_convergence.csv` | D, convergence band | one per engine x macrocycle |

- Panel A/C are restricted to the **conserved set** (`conserved_codes.txt`): the structures
  every engine has a score for, so no engine is judged on an easier subset.
- `figure2_panelA_paired_delta.csv` is the **paired** comparison — median of the
  per-structure `TorchRef − reference` differences (negative = TorchRef better), with a
  bootstrap CI on that median, a Wilcoxon signed-rank p, and win/loss counts.
  `delta_of_medians` is the unpaired difference of the two medians, kept alongside because
  the two can disagree: against PHENIX the unpaired ΔR-free reads +0.0022 while the paired
  median is +0.0007 (p=0.16, not significant). Quote the paired number.
  Note the CI and the p answer different questions — the CI is on the median, Wilcoxon tests
  signed-rank symmetry and weights by magnitude, so a near-zero median can still carry a
  small p (see `phenix` / `r_work`). Neither alone is the whole story.
  Written by `figure2_alphafold_start/analysis/summarize_medians.py`, which also puts the
  same numbers in `paper/FIGURE_MEDIANS.md`.
- Panel B holds the `q25`/`median`/`q75` the strips draw, in **RMSZ units** (rms/sigma), not
  the 0–1 axis fraction the panel maps them onto.
- Panel C is in **minutes**; `box_position` gives the measured left-to-right speed order.
- Panel D is normalized per structure (start 0 → final 1) so it shows convergence *speed*
  rather than incomparable program-reported absolute R-factors. `guard_min_delta_rfree`
  records the 0.02 total-improvement cut, and `n_structures` how many survived it.

### Figure 3 — performance

| file | panel |
|---|---|
| `figure3a_fcalc_speed.csv` | 3a, F_calc time per structure |
| `figure3b_target_breakdown.csv` | 3b, per-target cycle time (1DAW) |

- Both are in **milliseconds**. 3a's `row` is the plot's y-order (by atom count).
- 3a emits a row per structure x series **even when absent**, with an empty `time_ms`, so a
  gap is visibly a gap: SFcalculator OOMs on the three largest structures.
- 3a series: `torchref_cpu_{1,4,16}core`, `torchref_gpu`, `cctbx_single_thread`,
  `sfcalculator_gpu`. **`cctbx_single_thread` is the external control** — it is unchanged
  code, so if it moves between runs the measurement drifted rather than TorchRef changing. It
  is the fastest cctbx observed across the thread sweep, not the 1-core row specifically.
- 3a includes structures the panel omits, flagged by `in_plot`: the figure drops 5BOV as a
  triclinic outlier, while `FIGURE_MEDIANS.md` quotes all 10 (median 121x vs 125x on the 9
  plotted). Both are derivable from this file; neither is hidden.
- All times are best-of-10 (`*_min`), the same statistic the panel plots and
  `FIGURE_MEDIANS.md` quotes.
- 3b's `_aggregate_full_cycle_overlay` is the independently measured full-cycle time drawn
  over the stack, not a sum of the segments.
- CPU timings come from one pinned CPU model, recorded in each results dir's
  `cpu_model.txt`. Figure 3 uses **`cpu_xeon6230R`** (Cascade Lake, Q1 2020) because it is
  contemporaneous with the A100 (mid-2020); pairing the 2020 GPU against a 2024 EPYC 9335
  instead halves the apparent GPU speedup (119x -> 59x) purely because that CPU runs the
  unchanged cctbx reference 1.95x faster. Both runs are kept under
  `figure3_performance/data/fcalc/`.
- The TorchRef-vs-cctbx ratio is largely CPU-generation independent (median at 16 cores 5.6x
  on the Xeon, 6.2x on the EPYC) because both are timed in the same process on the same node.
  The A100-vs-CPU ratio is not, which is why the reference CPU is stated wherever a GPU
  speedup is quoted.

### Extended figures

| file | figure |
|---|---|
| `exF1_weight_grid.csv` | ExtFig 1, 10x10 loss-weight landscape, 4 heatmap panels |
| `exF2_panel{A,B}_{rfree,rwork}_points.csv` | ExtFig 2, scatter |
| `exF2_panel{A,B}_{rfree,rwork}_runningmedian.csv` | ExtFig 2, the bold median lines |
| `exF3_{r_work,r_free}_<scorer>_vs_<scorer>.csv` | ExtFig 3, 6 scorer-consistency panels |
| `exF4_singlecore_runtime.csv` | ExtFig 4, single-core runtime box plot |
| `exF5_panel{A,B,C,D}_*.csv` | ExtFig 5, target-mode comparison (same layout as Figure 2) |

- `exF1` `value` is the per-cell median of 47 structures; `is_locked_default` flags the
  production weights (geometry 0.2 / adp 0.02), which are also the grid optimum.
- `exF2` running-median lines are on their own x-grid (a 200-point sliding window), hence the
  separate file from the scatter. Units are **percentage points**.
- `exF3` rows come from the same pivot the panel uses, so they are exactly the
  `(code, model_engine)` pairs both scorers scored — the counts differ slightly per pair.
- `exF4` is in **minutes**, `box_position` is the measured speed order. Timings are from the
  structure-major (interleaved) submission; a program-major sweep confounds engine identity
  with cluster-load window and is not comparable.

## Caveats worth carrying into a caption

- Wall-clock figures (2c, exF4) are **relative** measurements. Absolute times depend on the
  node generation, so they are not comparable across runs; the per-structure ratio between
  engines is.
- Figure 2 R-factors are PHENIX-scored (`phenix.model_vs_data`) for all engines, so no engine
  is scored by its own code.
