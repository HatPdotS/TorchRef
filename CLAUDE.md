# TorchRef - Developer Guide

TorchRef is a PyTorch-based crystallographic refinement library that uses automatic differentiation and GPU acceleration for structure refinement. It follows PyTorch's `nn.Module` architecture throughout.

**Repository:** https://github.com/HatPdotS/TorchRef
**Documentation:** https://torchref.readthedocs.io/
**Version:** 0.5.1
**License:** MIT

## Quick Reference

```python
from torchref import LBFGSRefinement, ModelFT, ReflectionData, Scaler, ROOT_TORCHREF

# Load and refine
refinement = LBFGSRefinement(data_file="data.mtz", pdb="structure.pdb")
refinement.refine_xyz()         # coordinate refinement
refinement.refine_adp()         # B-factor refinement
refinement.refine(macro_cycles=5)  # full refinement
rwork, rfree = refinement.get_rfactor()
```

## Environment Setup

```bash
# Threading: auto-detected from available cores; override with
export TORCHREF_NUM_THREADS=4
```

Requires Python >= 3.10. Install into a fresh virtual environment or conda
environment of your choice (see `pyproject.toml` for full dependency pins).

## Building & Testing

```bash
# Install (development mode)
pip install -e ".[dev]"

# Run tests
pytest tests/                        # all tests
pytest tests/unit/                   # fast unit tests
pytest tests/integration/            # integration tests
pytest tests/functional/             # full workflow tests
pytest -m "not gpu and not slow"     # skip GPU/slow tests

# Test with coverage
pytest tests/ --cov=torchref

# Linting and formatting
black torchref/ --check              # check formatting (88 char line length)
isort torchref/ --check              # check import order
ruff check torchref/                 # lint (ignores F841, E741, E402, E722)
```

## Building Documentation

```bash
cd docs/

# Regenerate API docs from source
make apidoc

# Build HTML docs
make html
# Output: docs/_build/html/

# Run doctests in documentation
make doctest
```

Sphinx config is in `docs/conf.py`. Uses RTD theme, Napoleon (NumPy docstrings), autodoc, intersphinx (Python, NumPy, PyTorch, Pandas).

## Project Structure

```
torchref/
├── __init__.py              # Public API exports, threading config
├── config.py                # DtypeConfig singleton (dtypes.float, dtypes.int, dtypes.complex)
├── _bootstrap.py            # Threading configuration
│
├── model/                   # Atomic structure models
│   ├── model.py             # Model (base) - coordinates, B-factors, occupancies
│   ├── model_ft.py          # ModelFT - FFT structure factors, electron density
│   ├── parameter_wrappers.py  # MixedTensor, PositiveMixedTensor, OccupancyTensor
│   ├── sf_fft.py            # Structure factor calculation via FFT
│   ├── sf_ds.py             # Structure factor via direct summation
│   ├── internal_coordinates.py              # Internal coordinate systems
│   ├── segmented_internal_coordinates.py    # Segmented IC (chains)
│   ├── closed_segmented_internal_coordinates.py  # IC with chain closure
│   └── mixed_model.py       # Multi-model handling
│
├── io/                      # File I/O
│   ├── pdb.py, mtz.py, cif.py  # Format readers/writers
│   ├── data_router.py       # Auto file format detection
│   └── datasets/            # ReflectionData, FcalcData, DatasetCollection
│
├── refinement/              # Refinement framework
│   ├── base_refinement.py   # Refinement base class
│   ├── lbfgs_refinement.py  # LBFGSRefinement (primary refinement class)
│   ├── loss_state.py        # LossState - target aggregation, weights, torch.compile
│   ├── logger.py            # Progress logging
│   ├── targets/             # Target functions (xray/, geometry/, adp/, realspace, etc.)
│   ├── weighting/           # Loss weighting schemes (random, policy, component)
│   └── optimizers/          # Custom optimizers (SA, Langevin, exploratory LBFGS)
│
├── restraints/              # Geometry restraints (bonds, angles, torsions, planes, Ramachandran)
│   ├── restraints.py        # Main Restraints class
│   ├── builders.py          # Restraint construction from monomer library
│   └── library.py           # Monomer library (lazy download)
│
├── scaling/                 # Structure factor scaling
│   ├── scaler.py            # Scaler (overall scale, anisotropy, bulk solvent)
│   └── solvent.py           # Bulk solvent model
│
├── symmetry/                # Crystallographic symmetry (Cell, SpaceGroup, MapSymmetry)
├── maps/                    # Electron density maps (Map, DifferenceMap)
├── alignment/               # Patterson-based structure alignment
├── base/                    # Core math: coordinates, reciprocal, scattering, ED, FFT, kernels
├── utils/                   # Caching, device mixin, debugging, timing, serialization
├── kinetic/                 # Time-resolved / kinetic refinement support
├── scripts/                 # Helper scripts
├── cli/                     # 8 CLI entry points
└── data/                    # Package data (scattering tables, monomer library CIFs)
```

## Architecture

### Class Hierarchy

```
nn.Module
├── Model (base atomic model)
│   └── ModelFT (+ FFT structure factors, CachedForwardMixin)
├── Refinement (base refinement coordinator)
│   └── LBFGSRefinement (LBFGS optimization)
├── Target (base target/loss function)
│   ├── ModelTarget (needs Model reference)
│   │   ├── GeometryTarget → BondTarget, AngleTarget, TorsionTarget, ...
│   │   └── ADPTarget → ADPSimilarityTarget, RigidBondTarget, ...
│   └── DataTarget (needs ReflectionData + optional Model/Scaler)
│       └── XrayTarget → LeastSquaresXrayTarget, MaximumLikelihoodXrayTarget, ...
├── ScalerBase
│   └── Scaler
└── MixedTensor, PositiveMixedTensor, OccupancyTensor (parameter wrappers)
```

### Key Design Patterns

1. **Parameter Wrappers**: `MixedTensor` holds `refinable_params` (nn.Parameter) and `fixed_values` (buffer). Freeze/unfreeze moves atoms between the two.

2. **Smart Caching** (`CachedForwardMixin` in `utils/caching.py`):
   - Tracks `(data_ptr, _version)` of all parameters and buffers
   - Also fingerprints input tensors/kwargs
   - Backward hook increments generation counter for automatic invalidation
   - `reset_forward_cache()` for manual invalidation
   - Used by: MixedTensor, PositiveMixedTensor, OccupancyTensor, SegmentedInternalCoordinateTensor, ModelFT

3. **Structure Factor Cache** (in `model_ft.py`):
   - Separate from `CachedForwardMixin`, HKL-specific
   - `reset_cache()` resets SF cache + all wrapper forward caches

4. **Hierarchical Loss Management** (`LossState`):
   ```python
   state.register_target('xray/work', target)
   state.register_target('geometry/bond', target)
   state.set_weight('xray', 1.0)        # group weight
   state.set_weight('geometry', 0.5)     # group weight
   total = state.aggregate()             # evaluates all, applies weights
   ```

5. **Device Movement**: All major classes inherit `DeviceMovementMixin` for automatic `.to(device)` propagation.

6. **Empty Initialization**: Classes can be instantiated empty for `state_dict` loading:
   ```python
   model = Model()
   model.load_state_dict(torch.load('model.pt'))
   ```

## Key Public API

### Model

```python
model = ModelFT(max_res=1.0, radius_angstrom=4.0)
model.load_pdb("structure.pdb")
model.load_cif("structure.cif")

# Structure factors
fcalc = model(hkl)                          # or model.forward(hkl)
fcalc = model.get_structure_factor(hkl)

# Parameter control
model.freeze('b')                           # freeze B-factors
model.unfreeze('xyz')                       # unfreeze coordinates
model.freeze_selection("chain A and resseq 10:20")  # phenix-style syntax
model.unfreeze_selection("all")
model.shake_coords(0.1)                     # perturb for testing

# Access parameters
xyz = model.xyz.refinable_params            # nn.Parameter (N_refinable, 3)
b = model.adp.refinable_params              # nn.Parameter (N_refinable,)
```

### Reflection Data

```python
data = ReflectionData()
data.load_mtz("reflections.mtz")
hkl, F, sigF, rfree_flags = data()          # unpack
```

### Refinement

```python
ref = LBFGSRefinement(
    data_file="data.mtz",
    pdb="structure.pdb",
    device=torch.device("cuda"),             # GPU acceleration
)
ref.refine_xyz()                             # coordinate refinement
ref.refine_adp()                             # B-factor refinement
ref.refine(macro_cycles=5)                   # full refinement
rwork, rfree = ref.get_rfactor()

# Output
ref.write_out_pdb("refined.pdb")
ref.write_out_mtz("refined.mtz")
```

### Custom Targets

```python
from torchref.refinement.targets import Target

class MyTarget(Target):
    name = 'my_custom_target'

    def __init__(self, refinement):
        super().__init__()
        # store what you need

    def forward(self):
        # define loss - gradients computed automatically by PyTorch
        return loss

# Register with LossState
loss_state = refinement.create_loss_state()
loss_state.register_target('custom', MyTarget(refinement))
loss_state.set_weight('custom', 1.0)
total_loss = loss_state.aggregate()
total_loss.backward()  # autograd handles gradients
```

### Scaling

```python
scaler = Scaler(model, data, nbins=20)
scaler.initialize()
scaler.refine_lbfgs()
scaled_fcalc = scaler(fcalc)
```

### Maps

```python
from torchref import Map, DifferenceMap

m = Map(model, data, scaler)
diff = DifferenceMap(model, data, scaler)
```

## CLI Tools

| Command | Description |
|---------|-------------|
| `torchref.refine` | Basic LBFGS refinement |
| `torchref.refine-hyper` | User-provided hyperparameters |
| `torchref.difference-refine` | Difference refinement (time-resolved) |
| `torchref.mtz2map` | MTZ to CCP4 map conversion |
| `torchref.validate-ded` | Validate difference electron density |
| `torchref.phased-difference-map` | Phased difference maps |
| `torchref.add-metadata` | Add metadata to MTZ/PDB outputs |
| `torchref.strip-altlocs` | Strip alternate location indicators from PDB |

Common CLI usage:
```bash
torchref.refine -f reflections.mtz -s structure.pdb --output refined.pdb
```

## Code Conventions

- **Formatter:** Black (88 char line length, target py310)
- **Imports:** isort with Black profile
- **Docstrings:** NumPy style (mandatory for public API)
- **Type hints:** Required for public functions
- **Linting:** Flake8 (ignores F841, E741, E402, E722)
- **Naming conventions:**
  - `f_calc` = complex structure factors (with phase)
  - `F_calc` = amplitudes (absolute values)
  - `adp` = atomic displacement parameters
  - `u` = anisotropic U tensor (6 components)
  - `b` = B-factor
  - `xyz` = Cartesian coordinates (Angstroms)
  - `xyz_fractional` = fractional coordinates (0-1)
  - `d_min` / `d_max` = resolution bounds
  - `spacegroup` = one word (not "space_group")

## Test Data

Test structures in `tests/files/`: 1DAW, 2DQ6, 3A5V, 3E98, 3GR5, 3K7M, 3VRJ, 4BX9, 5BOV, 6G9X.

Example data in `example_notebooks/`: 1DAW.pdb, 1DAW.mtz.

## Dependencies

**Core:** numpy>=2.0, torch>=2.4, pandas>=2.0, scipy>=1.10, gemmi>=0.5, reciprocalspaceship>=0.9.18, numba>=0.59, matplotlib>=3.7, tqdm, pyarrow, ninja

**Optional:**
- `pip install torchref[dev]` - pytest, pytest-cov, black, isort, flake8
- `pip install torchref[alignment]` - JAX, s2fft, s2ball (Patterson alignment)
- `pip install torchref[forcefield]` - torchmd-net
- `pip install torchref[amber]` - OpenMM, pdbfixer
- `pip install torchref[ihm]` - ihm (integrative/hybrid model I/O)
- `pip install torchref[docs]` - Sphinx, RTD theme, numpydoc

**CCTBX (conda-only, optional):** Some utilities in
`torchref/math_functions/CCTBX_related.py` depend on `iotbx`. Install with
`conda install -c conda-forge cctbx-base`.
