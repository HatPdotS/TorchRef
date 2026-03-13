# Figure 2: Validation Benchmark (TorchRef vs PHENIX)

Publication-quality comparison of TorchRef refinement against PHENIX across ~940 structures from the PDB.

## Pipeline Overview

The full data generation pipeline consists of the following steps:

### 1. Download structures
- `scripts/get_pdb_and_data_files.py` - Query RCSB PDB for structures matching criteria (resolution, organism, etc.)
- `scripts/download_pdb_files.py` - Download PDB/CIF structure files and MTZ reflection data from RCSB
- `scripts/filter_by_structure_factors.py` - Filter structures by availability of structure factor data

### 2. Prepare input data
- `scripts/convert_cifx.py` - Convert CIF structure files to PDB format (with hydrogens)
- `scripts/shake_cifs.py` - Perturb coordinates (0.2 A) and B-factors (5.0) to create starting models
- `scripts/make_standardized_mtzs.py` - Standardize MTZ files (consistent R-free flags and format)

### 3. Run refinements
- `scripts/run_refine_all.py` - Submit batch TorchRef refinement jobs (via SLURM)
- `phenix_refinement/phenix_refine.sh` - PHENIX refinement script (10 macro cycles, with Ramachandran restraints)
- `phenix_refinement/submit_all_refinements.sh` - Batch submission of PHENIX jobs

### 4. Orchestrate and validate
- `run_pipeline.py` - Master automation script with subcommands:
  - `auto` - Full pipeline: submit jobs, wait, validate, collect metrics, plot
  - `validate` - Submit REFMAC 0-cycle validation jobs
  - `analyze` - Collect metrics and generate plots

### 5. Compute deviations
- `calc_deviations.py` - Per-atom coordinate RMSD and B-factor comparisons between initial, TorchRef, and PHENIX structures

### 6. Generate figure
- `plot_figure2.py` - Create the publication 2x2 panel figure from pre-computed metrics

## Regenerating Figure 2

The pre-computed metrics are in `data/`. To regenerate the figure:

```bash
python plot_figure2.py
```

Output: `output/figure2.png`

## Figure Layout

| Panel | Content |
|-------|---------|
| A | R-work vs R-free scatter (Initial, PHENIX, TorchRef) |
| B | Quality radar: R-factors, geometry RMSD, B-factor RMSD |
| C | Histogram of per-atom RMSD from initial structure |
| D | Runtime comparison: TorchRef vs PHENIX (log-log scatter) |
