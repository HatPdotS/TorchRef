# AGENTS.md

Working notes for AI agents (and humans) editing TorchRef. This file is the authority on
conventions; where `.github/instructions/*.md` disagrees with it, this file wins.

TorchRef is GPU-accelerated **crystallographic refinement** built on PyTorch: models are
`nn.Module`s, targets are losses, and gradients come from autograd. The domain is X-ray
crystallography — structure factors, Miller indices, space groups, ADPs, R-factors — not
machine learning for its own sake and not structural biology. Screen every change against
crystallographic convention (units, sign of phases, Å and Å², d-spacing vs resolution).

---

## 1. Environment

Use the project's own virtual environment / conda environment — never a bare `python` or
`python3`. Machines that host this repo usually carry several other interpreters with
TorchRef's dependencies at the wrong versions, and those fail subtly (wrong numbers) rather
than loudly. If you don't already know which interpreter is the right one, find it before
running anything: look for a `.venv`/`.env` beside the repo, check `which -a python`, or
confirm with `python -c "import torchref, torch; print(torchref.__file__, torch.__version__)"`.

**Probe for a batch scheduler before deciding how to run tests.** Don't assume either way:

```bash
command -v sinfo && sinfo -s     # SLURM present? which partitions, what's idle
```

- **No scheduler** (laptop, workstation, CI container): just run the command normally.
- **SLURM present**: read `sinfo` output and pick a partition from what actually exists and
  has idle nodes — partition names are site-specific, so never hardcode one. Then use
  `srun -c 8 --pty …` for short interactive work and `sbatch -c 8 …` for anything long.
  Keep off the login node for real work, and don't request a GPU node unless the change
  touches a CUDA/Triton path (`sinfo` also tells you whether GPU nodes exist and are free).
  Check `squeue -u "$USER"` before queuing more.

---

## 2. Hard rules

### 2.1 The whole package runs in float32

**If float64 is needed for a result to be correct, the formula is wrong — fix the formula.**

Single precision is the design constraint, not a performance compromise. MPS has no float64
at all, and CUDA float64 throughput is a small fraction of float32 on the hardware this runs
on, so a float64 dependency makes a code path unusable on half the supported devices.

Practically:

- **Never hardcode a dtype.** Take it from the config: `torchref.config.get_float_dtype()`,
  `get_int_dtype()`, `get_complex_dtype()`, or from an input tensor. Roughly 200 call sites
  already do this; follow them.
- `torch.float64` *is* a supported configuration (`TORCHREF_DTYPE_FLOAT=float64`) used as an
  eager numerical reference and in gradient checks. Code must **work** in float64, must not
  **require** it, and must not silently downcast (see `tests/integration/test_dtype_config_float64.py`).
- When you hit precision trouble, reformulate: log-space accumulation, `logsumexp`,
  `log1p`/`expm1`, shift-and-center before squaring, Kahan/pairwise sums, subtract the mean
  before a covariance, factor out the dominant scale. Cast to `.double()` only as a
  last resort, and then only for a small, contained, non-differentiable block (a 3×3 eigen
  solve, a matrix exponential) — with a comment saying which conditioning problem forces it.
- NumPy interop at I/O boundaries (gemmi, reciprocalspaceship, OpenMM) is naturally float64;
  that is fine. Convert at the boundary, not deep inside a kernel.
- Integers default to **int32**, complex to **complex64**. Note the MPS trap: `scatter_reduce`
  `amax`/`amin` on int64 fails there — use `index_add_` or a stable `argsort` for grouped
  reductions.

### 2.2 Code does not track its own history

Git and `docs/changelog.rst` record what changed. Source files describe what **is**.

Do not write, in code or docstrings:

- `# NEW:`, `# CHANGED:`, `# FIXED:`, `# was: ...`, `# previously we did X`
- "This replaces the old `foo()`", "as of 0.6.2", "after the refactor", "legacy path"
- Commented-out former implementations kept "for reference"
- Benchmark numbers from a one-off run ("3.2× faster than before", "took 4.1 s on 1DAW")
- Dated notes, initials, TODO owners, or audit-cluster references

Two narrow exceptions:

1. A **public deprecation** the user must act on — use a `Warnings`/`.. deprecated::` block
   or `DeprecationWarning`, stating the replacement, not the history.
2. A constant whose value was chosen empirically may state **the criterion and the
   conclusion** in one or two lines (e.g. "3.0 is the floor: 2.5 degrades the F-residual by
   20× on the worst test case") — never the full sweep table.

When you change something, put the note in `docs/changelog.rst` under the current version
heading, one line, user-facing.

### 2.3 Docstrings: NumPy style, on every public thing

Rendered by Sphinx with `napoleon` (`napoleon_numpy_docstring = True`,
`autodoc_typehints = 'description'`). Google style is off and will render wrong.

A docstring answers *"how do I call this and what will it do to me?"* for someone who will
**not** read the body. Document the contract, plus any trap that produces a silently wrong
result.

```python
def compute_structure_factors(
    hkl: torch.Tensor, xyz: torch.Tensor, b_factors: torch.Tensor
) -> torch.Tensor:
    """Compute structure factors for the given reflections.

    Parameters
    ----------
    hkl : torch.Tensor
        Miller indices of shape (n_reflections, 3), integer dtype.
    xyz : torch.Tensor
        Atomic coordinates of shape (n_atoms, 3) in Ångströms.
    b_factors : torch.Tensor
        Isotropic B-factors of shape (n_atoms,) in Å².

    Returns
    -------
    torch.Tensor
        Complex structure factors of shape (n_reflections,).

    Raises
    ------
    ValueError
        If tensor shapes are incompatible.
    """
```

Rules:

- **Required on** every public module, class, function, method and property. Private helpers
  get one when the reason for their existence is not obvious from the name.
- **Tensors**: always give shape and units. Coordinates in Å, ADPs in Å², angles in degrees
  unless stated, resolution/d-spacing in Å. Say "fractional" or "Cartesian" explicitly.
- **Types live in the annotations.** Type-hint every public signature; don't restate the
  signature in prose.
- **Traps belong in the docstring**: device or dtype restrictions, in-place mutation, a cache
  the caller must invalidate, a process-wide import side effect, non-determinism (CUDA
  atomics), a GPU→CPU sync, a return value that aliases an input.
- **Rationale is a clause, not a section.** Say why the design is the way it is in one
  sentence if a caller could otherwise misuse it; leave the rest to the code.
- **No `Examples` block that is entirely `# doctest: +SKIP`.** It costs lines and tests
  nothing. Runnable examples go in `docs/quickstart.rst`, which runs under
  `sphinx.ext.doctest`.
- **Module docstrings** state what the module is for and how its pieces relate — see
  `torchref/utils/backends.py` or `torchref/model/__init__.py` for the target quality. A
  package `__init__.py` docstring should say what is re-exported and what deliberately is
  not.
- Imperative mood ("Compute", not "Computes").

### 2.4 Comments explain *why*, never *what*

The code already says what it does. A comment earns its place only when a reader who
understands Python and crystallography would still be surprised.

**Write a comment for:**

- A non-obvious constraint: *"Must be set before torch is imported below, or it has no
  effect."*
- An ordering dependency, a workaround for an upstream bug, a device-specific quirk.
- A numerically motivated formulation (why the log-space form, why the clamp, why the epsilon).
- A deliberate omission — something a reader would otherwise "fix".
- The crystallographic convention being followed when more than one exists.

**Do not write:**

- Restatements: `# loop over atoms`, `# increment counter`, `# return the result`
- Section banners inside short functions
- History (§2.2)
- Docstring content displaced into a comment above the `def`

Density target: roughly what `torchref/config.py` and `torchref/utils/backends.py` do —
long comments where the reasoning is load-bearing, none at all where the code is plain.

### 2.5 Style mechanics

Black, 88 columns, `isort` with the black profile. Ruff lint with
`ignore = ["F841", "E741", "E402", "E722"]`. Python ≥ 3.10.

---

## 3. Package structure

`torchref/` — 260 modules, ~92k lines. Top-level exports live in `torchref/__init__.py`
(`Model`, `ModelFT`, `LBFGSRefinement`, `ReflectionData`, `Cell`, `SpaceGroup`, `Map`, …).

| Package | Contents |
|---|---|
| `base/` | Low-level math and crystallography. `coordinates/` (Cartesian↔fractional), `reciprocal/` (basis, HKL, d-spacing, interpolation, symmetry), `direct_summation/` (F_calc by summation; eager + Triton), `electron_density/` (real-space splatting with CPU/CUDA/MPS kernels, solvent mask, radius policy), `fourier/` (FFT and grids), `scattering/` (form-factor and anomalous tables), `metrics/` (R-factors, binwise scale, loss), `targets/` (the *kernels* behind refinement targets, eager + `triton/`), `french_wilson.py`, `math_torch.py`, `alignment/` |
| `io/` | `ReflectionData`, `DatasetCollection`, `FcalcDataset`; MTZ / PDB / CIF / IHM readers and writers; `read_mtz` / `read_pdb` / `read_cif` |
| `model/` | `Model` (refinable atomic parameters), `ModelFT` (adds F_calc via `SfFFT` or `SfDS`), `MixedModel`, `ModelCollection`, and the parametrizations in `parameter_wrappers.py` / `rigid_xyz.py` that decide what is refinable |
| `refinement/` | Drivers (`Refinement`, `LBFGSRefinement`, `RigidBodyRefinementStep`), `targets/` (`xray/`, `geometry/`, `adp/`, `collection/`, `combined.py`), `weighting/`, `optimizers/` (annealing, Langevin, preconditioned/seeded L-BFGS), `model_error_estimation/` (σ_A, σ_M), `loss_state.py`, `logger.py` |
| `restraints/` | Bonds, angles, torsions, planes, chirals, VDW. Built from the CCP4 Monomer Library, resolved lazily via `get_library_manager()` — importing this package must not trigger a library download |
| `scaling/` | `ScalerBase` (model-independent), `Scaler`, `CollectionScaler`, `SolventModel` (k_sol, B_sol) |
| `symmetry/` | `Symmetry` (operations plus everything derived from them), `SpaceGroup` (adds the crystallographic identity and the CCP4 ASU verbs), `Cell`. All dataclasses over `DeviceMixin`, not `nn.Module` — they hold no refinable parameters. Map and reciprocal-grid operators are private, reached through `Symmetry` |
| `maps/` | `Map` (2Fo−Fc, Fcalc), `DifferenceMap` |
| `cli/` | Entry points: `torchref.refine`, `torchref.difference-refine`, `torchref.mtz2map`, `torchref.validate-ded`, `torchref.phased-difference-map`, `torchref.add-metadata`, `torchref.strip-altlocs` |
| `experimental/` | APIs that may change without notice: `alignment/` (Patterson MR), `kinetic/` (time-resolved), `ensemble/`, `monolithic_refinement/`, `targets/` (AMBER/GAFF2, real-space, sampled-ML phase) |
| `utils/` | See §5 |
| `config.py` | See §4 |

Layering: `base` and `utils` depend on nothing above them; `model`/`io`/`symmetry` build on
`base`; `refinement`/`scaling`/`maps` build on those; `cli` sits on top. Don't import
downward from `base` or `utils` into the higher layers.

Two conventions worth knowing before adding re-exports: `__all__` in a package `__init__.py`
is the public surface, and several packages deliberately keep helpers out of it (e.g.
`torchref.utils.timing.register_timing`, the `restraints` builder classes). Where a name
exists in two modules, the docstring says which is the source of truth — don't re-export the
copy.

---

## 4. Configuration — `torchref/config.py`

Everything is read from the environment at import and is mutable at runtime by attribute
assignment. There is no config file and no per-call dtype/device argument threading.

```python
import torch, torchref
torchref.dtypes.float = torch.float64          # eager reference only
torchref.device.current = torch.device('cpu')
torchref.config.sigma_cutoff_ed.value = 3.5
torchref.config.caching.value = False
```

| Env var | Object / accessor | Default | Meaning |
|---|---|---|---|
| `TORCHREF_DTYPE_FLOAT` | `dtypes.float` / `get_float_dtype()` | `float32` | Global float dtype. `float64` is a reference/debug setting only (§2.1) and is unavailable on MPS |
| `TORCHREF_DTYPE_INT` | `dtypes.int` / `get_int_dtype()` | `int32` | Global integer dtype |
| `TORCHREF_DTYPE_COMPLEX` | `dtypes.complex` / `get_complex_dtype()` | `complex64` | Global complex dtype; `complex128` unavailable on MPS |
| `TORCHREF_DEVICE` | `device.current` / `get_default_device()` | `auto` | `auto` picks cuda → mps → cpu. Auto-selecting CUDA also requires compute capability ≥ the minimum `sm_*` in the torch build **and** ≥ 10 GB VRAM (`_MIN_CUDA_VRAM_GB`), else it warns and falls back. An explicit value bypasses those gates but raises if the backend is missing |
| `TORCHREF_SIGMA_CUTOFF_ED` | `sigma_cutoff_ed.value` | `3.0` | Number of σ at which each atom's Gaussian density is truncated. Per-atom radius `clamp(ceil₀.₂₅(Nσ·σ_eff), [2, 7] Å)` with `σ_eff = sqrt((b_form + B) / 8π²)`. 3.0 is the floor |
| `TORCHREF_COMPILE_TARGETS` | `compile_targets.value` / `get_compile_targets()` | `False` | `torch.compile` the quadrature X-ray target kernels (`--xray-mode ml_full`). Off by default: ~2 min backward-compile latency. Keep off for float64 and gradient verification |
| `TORCHREF_CACHING` | `caching.value` / `get_caching_enabled()` | `True` | Gates `CachedForwardMixin` **only**. Off ⇒ every `forward()` recomputes; numbers unchanged. Primary use is diagnosing stale-looking results in one step |
| `TORCHREF_NUM_THREADS` | `torchref.N_CPUS` | auto-detected | Thread count; must be applied before torch imports, which `torchref/_bootstrap.py` handles |
| `TORCHREF_MONOMER_LIB` | `restraints.library` | unset | Path to a local CCP4 monomer library install, checked first |

Also in `config.py`, not env-configurable:

- `NYQUIST_OVERSAMPLING = 3.0` — grid spacing is `d_min / 3`. This is the single constant
  every grid-sizing helper reads, so real-space maps and FFT structure-factor grids stay
  consistent. Bare Nyquist (2.0) has been tried: Gaussian atomic density is not band-limited,
  and 2.0 measurably degrades convergence.
- `canonical_device(dev)` — fills in the default index so `torch.device('cuda')` and
  `torch.device('cuda:0')` compare equal; `cpu` stays bare. Allocation-free, deliberately
  not memoised.
- `normalize_device(dev=None)` — pure coercion of one device source. The counterpart that
  reconciles *several* device-bearing inputs is `utils.resolve_device`, which also *moves*
  them.

Note that `torchref` also sets `PYTORCH_ENABLE_MPS_FALLBACK=1` at import.

---

## 5. `torchref/utils`

`__all__` in `torchref/utils/__init__.py` is the public surface. Anything not listed there
must be imported from its own module.

| Module | Public names | Notes |
|---|---|---|
| `utils.py` | `ModuleReference`, `TensorDict`, `TensorMasks`, `sanitize_pdb_dataframe`, `parse_phenix_selection`, `create_selection_mask` | `ModuleReference` holds an `nn.Module` without registering it (keeps its parameters out of the parent tree). `TensorDict` is buffer-backed. `TensorMasks` caches the combined logical-AND mask. Phenix-style atom-selection strings → boolean masks |
| `device_mixin.py` | `DeviceMixin` (`DeviceMovementMixin` alias) | Hijacks `.to()`/`.cuda()`/`.cpu()` for both `nn.Module` subclasses and plain classes; recursively moves params, buffers, raw tensor attributes, tensors nested in list/tuple/dict, and unregistered modules. Cycle-safe via a thread-local `id()` set. **Every moved node gets `reset_forward_cache()`/`reset_cache()` called — a `.to()` is never free, even onto the current device** |
| `device_resolution.py` | `resolve_device`, `require_cell_dtype` | Collapse several device-bearing constructor inputs onto one device with fixed precedence; dtype-axis counterpart |
| `caching.py` | `CachedForwardMixin`, `ParameterFingerprint`, `no_caching` | Caches `forward()` with invalidation on parameter mutation or backward. Gated globally by `TORCHREF_CACHING`; `no_caching` scopes it to a block. `ParameterFingerprint` is standalone and ungated |
| `backends.py` | `Backend`, `BackendTable`, `select`, `will_use`, `run_or_degrade`, `triton_available`, `force_portable`, `set_force_portable`, `use_portable`, `TorchRefDegradationWarning` | Declarative kernel dispatch — see §6 |
| `loss_validation.py` | `validate_loss`, `NonFiniteLossError`, `reset_diagnostic_budget` | Checks loss (optionally grads and params) and dumps a per-target breakdown on failure. With `raise_on_fail=False` **the caller must reject the step itself** — in an L-BFGS closure: zero grads and return `+inf` so strong-Wolfe backtracks. Costs one GPU→CPU sync (two with `check_grads=True`) |
| `autograd_introspection.py` | `collect_loss_leaves` | Walks a loss's autograd graph to find the leaf `nn.Parameter`s it touches; used by `LossState` to disable `requires_grad` on leaves the optimizer wasn't built with |
| `autograd_ops.py` | `gather_with_index_add` | 1-D gather whose backward uses `index_add_` instead of the radix-sorting `_index_put_impl_`. **Not bit-reproducible on CUDA** (atomics) |
| `stats.py` | `stat`, `StatEntry`, `StatEntryEncoder`, `filter_stats`, `flatten_stats`, `format_stats_table` | Verbosity-tagged reporting: `VERBOSITY_ESSENTIAL` (0) … `_DEBUG` (3). **Import side effect: replaces stdlib `json.dumps`/`json.dump` process-wide** to default `cls=StatEntryEncoder` |
| `debug_utils.py` | `DebugMixin`, `print_module_summary` | Module introspection |
| `gradnorm.py` | `gradnorm` | Gradient-norm monitoring |
| `serialization.py` | `convert_to_serializable` | Tensors/arrays → JSON-safe |
| `pse.py` | `PERIODIC_TABLE` | `{symbol: {"number", "name", "mass"}}`, mass in u |
| `timing.py` | `register_timing` | CLI-only; **not** re-exported — import from the module |

---

## 6. Devices and backend dispatch

Three device families are supported: **CPU**, **CUDA** (with Triton kernels) and **MPS**
(with Metal shaders). Any new kernel needs a working path on all three; the CPU path is the
portable reference.

Which kernel runs is decided **declaratively**, not by an `if/elif` ladder. `utils/backends.py`
holds the machinery; the tables live next to the kernels
(`base/electron_density/_backends.py`, `base/direct_summation/_backends.py`). Each row states
device, dtype, an availability probe and a failure policy; kernels are stored as
`(module, attr)` and resolved per call, which keeps them monkeypatchable where they are
defined. Accelerator gates are pairwise device-disjoint, so at most one non-base backend
matches. `set_force_portable(True)` pins dispatch to the reference kernel — the one override
that exists, for the failure automatic fallback cannot detect: an accelerator that runs and
returns wrong numbers.

Silent degradation is a test failure: `TorchRefDegradationWarning` is promoted to an error in
`pyproject.toml`'s `filterwarnings`.

When touching device-bearing classes, register them in `tests/helpers/device_cases.py` —
`tests/unit/test_device_conformance.py` asserts that **every** device-bearing class in the
tree is either covered there or listed in `UNCOVERED` with a reason.

---

## 7. Testing

```bash
pytest tests/                          # everything the host supports
pytest tests/unit/                     # fast
pytest tests/ --run-slow               # include slow
pytest tests/ --cov=torchref
```

- Accelerator tests **run automatically** when the hardware is present and skip when it is
  not. `--run-gpu` is a deprecated no-op. `--run-cuda` / `--run-mps` turn a missing
  accelerator into a failure instead of a skip — for CI runners that lost their GPU.
- Do **not** add a default `-m` expression to `pyproject.toml`: it deselects at collection
  time and no flag can undo it. Keep root `pyproject.toml` and `tests/pytest.ini` in step so
  behaviour doesn't depend on the working directory.
- Markers: `unit`, `integration`, `gpu` (any accelerator), `cuda`, `mps`, `slow`, `openmm`,
  `amber`.
- Layout: `tests/unit/` mirrors the package; `tests/integration/` covers pipelines and the
  CLI; `tests/helpers/` holds `device_cases`, `device_asserts`, `grad_asserts`,
  `device_inventory`; `tests/files/` holds ten deposited structures (1DAW, 2DQ6, 3A5V, 3E98,
  3GR5, 3K7M, 3VRJ, 4BX9, 5BOV, 6G9X) in cif/pdb/mtz/cif_sf. **1DAW is the small one** — use
  it for anything that must stay fast.
- Numerical claims get verified on a **real deposited structure**, not a synthetic scene.
  Synthetic scenes overstate derivative error by roughly 1/√N in atom count.
- Test docstrings should say what behaviour is pinned, not which past bug prompted them
  (§2.2). Existing references to `TORCHREF_AUDIT.md` are legacy; don't add more.

---

## 8. Docs

Sphinx in `docs/`: `api.rst` (autodoc), `quickstart.rst` (doctested — put runnable examples
here), `user_guide/` (cli, refinement, restraints, scaling, targets, testing,
naming_conventions), `contributing.rst`, `changelog.rst`.

Every user-visible change gets one line in `docs/changelog.rst` under the current version.
That is the *only* place changes are recorded outside git.
