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
│   └── {CODE}/{CODE}.mtz, {CODE}_shaken.pdb, ...
├── phenix_refinements -> .../refinements         # PHENIX comparison results
│   └── {CODE}/{CODE}_refined_001.pdb, .log, ...
```

**For reviewers:** if cloning fresh, create the data directory and populate it:

```bash
cd paper/

# Step 1: Download 1000 structures (PDB + structure factors)
python figure2_validation/scripts/get_pdb_and_data_files.py
python figure2_validation/scripts/download_pdb_files.py

# Step 2: Prepare input data
python figure2_validation/scripts/make_standardized_mtzs.py   # CIF → MTZ
python figure2_validation/scripts/convert_cifx.py              # CIF → PDB
python figure2_validation/scripts/shake_cifs.py                # Shake coordinates + B-factors

# Step 3: Create the data symlink (or copy)
ln -s ../scientific_testing/data data
```

## Figures

### Figure 2: Validation Benchmark
**Location:** `figure2_validation/`

Comparison of TorchRef vs PHENIX refinement across ~1000 PDB structures.
See `figure2_validation/README.md` for the full reproduction pipeline.

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
| ExtFig 4 | F_calc splatting optimization breakdown (1DAW) | Requires CPU + GPU benchmarking |

See `extended_figures/instructions.md` for the design document.
