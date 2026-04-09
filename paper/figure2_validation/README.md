# Figure 2: Validation Benchmark (TorchRef vs PHENIX)

Comparison of TorchRef refinement against PHENIX across ~1000 PDB structures,
validated with REFMAC5 zero-cycle refinement for fair R-factor comparison.

## Quick Start — Regenerate Figure Only

Pre-computed metrics are in `data/`. To regenerate the figure from existing data:

```bash
python plot_figure2.py
# Output: output/figure2.png
```

## Full Reproduction Pipeline

### Prerequisites

- **TorchRef** installed (`pip install torchref`)
- **PHENIX** available (`module load phenix/phenix-1.20-4459`)
- **CCP4/REFMAC5** available (`module load ccp4`)
- **SLURM** cluster for batch job submission
- **Data directory**: `paper/data/` must contain the 1000 benchmark structures
  (see `paper/README.md` for data setup instructions)

### Step 1: Run TorchRef Refinement (SLURM)

```bash
# Create experiment and submit 1000 refinement jobs
python run_pipeline.py auto --name paper_v1

# Or step by step:
python run_pipeline.py refine --name paper_v1        # submit jobs
python run_pipeline.py status --name paper_v1         # check progress
python run_pipeline.py validate --name paper_v1       # REFMAC validation
python run_pipeline.py analyze --name paper_v1        # collect metrics + plot
```

The `auto` subcommand runs all steps sequentially with SLURM polling.
Results go to `experiments/paper_v1/`.

### Step 2: Run PHENIX Refinement (SLURM)

```bash
cd phenix_refinement/
./submit_all_refinements.sh
# Or for specific structures:
sbatch phenix_refine.sh 138L
```

Results go to `paper/phenix_refinements/{CODE}/`.

### Step 3: Validate and Collect Metrics

If you ran `auto`, this is already done. Otherwise:

```bash
python run_pipeline.py validate --name paper_v1    # REFMAC 0-cycle validation
python run_pipeline.py analyze --name paper_v1     # collect CSVs + plots
```

Metrics are written to `experiments/paper_v1/metrics/`:
- `refmac_metrics.csv` — REFMAC-validated R-factors and geometry for all variants
- `internal_metrics.csv` — Program-reported R-factors and B-factor statistics
- `runtimes.csv` — Wall-clock times for TorchRef and PHENIX
- `summary.csv` — Median metrics across all structures

### Step 4: Compute Deviations

```bash
python calc_deviations.py --name paper_v1
```

### Step 5: Generate Figure

```bash
python plot_figure2.py
```

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `run_pipeline.py` | Master orchestration (refine → validate → analyze) |
| `plot_figure2.py` | Generate publication figure from pre-computed metrics |
| `calc_deviations.py` | Per-atom RMSD and B-factor deviations |
| `phenix_refinement/phenix_refine.sh` | PHENIX refinement (single structure, SLURM) |
| `phenix_refinement/submit_all_refinements.sh` | Batch PHENIX submission |

### Data Preparation Scripts (one-time)

| Script | Purpose |
|--------|---------|
| `scripts/get_pdb_and_data_files.py` | Query RCSB PDB for structures |
| `scripts/download_pdb_files.py` | Download PDB/CIF + MTZ from RCSB |
| `scripts/filter_by_structure_factors.py` | Filter by SF availability |
| `scripts/make_standardized_mtzs.py` | Convert SF-CIF → MTZ with R-free flags |
| `scripts/convert_cifx.py` | Convert CIF → PDB format |
| `scripts/shake_cifs.py` | Perturb coordinates (0.2 Å) and B-factors (5.0) |

## Figure Layout

| Panel | Content |
|-------|---------|
| A | R-work vs R-free scatter (Initial, PHENIX, TorchRef) |
| B | Quality radar: R-factors, geometry RMSD, B-factor RMSD |
| C | Histogram of per-atom RMSD from initial structure |
| D | Runtime comparison: TorchRef vs PHENIX (log-log scatter) |
