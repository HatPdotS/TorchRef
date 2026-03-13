# Paper Figures

This folder contains the data, scripts, and figures for the TorchRef publication.

## Figures

### Figure 2: Validation Benchmark
**Location:** `figure2_validation/`

Comparison of TorchRef vs PHENIX refinement across ~940 PDB structures. Shows R-factors, geometry quality, structural deviations, and runtime comparison.

See `figure2_validation/README.md` for the full data generation pipeline.

### Figure 3: Performance Benchmarks
**Location:** `figure3_performance/`

- **Panel A** (`figure3a_fcalc.png`): Structure factor calculation (Fcalc) thread scaling and GPU performance, compared to cctbx.
- **Panel B** (`figure3b_profiling.png`): Full refinement cycle profiling with per-target breakdown (X-ray, geometry, ADP targets).

Benchmark scripts in `fcalc_benchmark/` and `refinement_cycle_benchmark/`. Test data: `data/1DAW.pdb` and `data/1DAW.mtz`.

### Figure 4: Difference Refinement
**Location:** `figure4_difference_refinement/`

Time-resolved difference refinement of IBL isomerization in tubulin. Shows recovery of the light-state (cis) conformation from dark-state observations using TorchRef's MixedModel framework.

- `data/` - Input crystallographic data (dark/light PDB structures and reflections)
- `refinement_output/` - TorchRef difference refinement results (82% dark, 18% light fractions)
- `validation/` - DED validation results (TorchRef vs extrapolation method, IBL mask vs full chain)
- `panels/` - PyMOL-rendered figure panels (manually positioned, not auto-regenerable)
- `scripts/` - Refinement and validation scripts

## Regenerating Figures

Figures 2, 3a, and 3b can be regenerated from the bundled pre-computed data:

```bash
python generate_all_figures.py
```

Figure 4 panels are PyMOL renders that require manual positioning. The CCP4 maps needed for rendering can be regenerated via:

```bash
cd figure4_difference_refinement/validation
bash run_validation.sh
```

## Dependencies

- Python 3.10+
- matplotlib, numpy, pandas (included with TorchRef)
- TorchRef (for regenerating benchmark data or validation maps)
- PyMOL (for Figure 4 panel rendering only)
