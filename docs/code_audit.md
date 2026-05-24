# TorchRef Code Audit — Redundancy & Reuse

**Scope:** `model/`, `scaling/`, `maps/`, `base/`, `symmetry/`, `refinement/` (incl. targets, weighting, optimizers), `restraints/`, `io/`, `utils/`, `cli/`, `scripts/`, top-level helpers.
**Excluded:** `alignment/`, `kinetic/`.
**Date:** 2026-05-21

This is a static audit — no code was changed. Findings are grouped by severity and cite concrete file:line ranges. Line-count estimates are rough.

---

## Estimated impact

| Tier | Direct LOC removable | Notes |
|------|----------------------|-------|
| High | ~285 | 6 findings, hot-path code |
| Medium | ~315 | 7 findings, mostly mechanical |
| Low | ~130 | 8 cleanups |
| **Direct net** | **~700–800** | Counting only obvious dup blocks |
| With follow-on (imports, duplicated tests, dead branches that fall out) | **~1000** | Plausible upper bound |

---

## HIGH value

### H1. `SfFFT` and `SfDS` share ~60 LOC of cell/spacegroup plumbing
- [torchref/model/sf_fft.py:174-234](../torchref/model/sf_fft.py#L174-L234) vs [torchref/model/sf_ds.py:142-195](../torchref/model/sf_ds.py#L142-L195)
- Identical `cell` / `spacegroup` properties + setters, `set_cell_and_spacegroup()`, `fractional_matrix` / `inv_fractional_matrix` properties, identical error strings.
- **Fix:** extract `StructureFactorBase` mixin; both inherit.

### H2. Three coordinate classes duplicate `freeze`/`unfreeze` boilerplate
- [torchref/model/internal_coordinates.py](../torchref/model/internal_coordinates.py), [segmented_internal_coordinates.py](../torchref/model/segmented_internal_coordinates.py), [closed_segmented_internal_coordinates.py](../torchref/model/closed_segmented_internal_coordinates.py)
- Identical `freeze` / `unfreeze` / `freeze_all` / `unfreeze_all` methods with the same `fix`/`refine` aliases and defaults (`freeze_at_current=True`, `rebuild=True`). ~40 LOC per file.
- **Fix:** `FreezableTensor` mixin in `utils/`.

### H3. Geometry/ADP targets re-implement the same restraint-stats pattern
- [targets/geometry/bonds.py:46-63](../torchref/refinement/targets/geometry/bonds.py#L46-L63), [angles.py:48-68](../torchref/refinement/targets/geometry/angles.py#L48-L68), [adp/rigid_bond.py:270-285](../torchref/refinement/targets/adp/rigid_bond.py#L270-L285), [adp/similarity.py:98-112](../torchref/refinement/targets/adp/similarity.py#L98-L112), [geometry/planarity.py:136-146](../torchref/refinement/targets/geometry/planarity.py#L136-L146), [geometry/chiral.py:166-176](../torchref/refinement/targets/geometry/chiral.py#L166-L176)
- All compute `z = deviations/sigmas` then return `{loss, n, rms_delta, rms_z, mean_sigma}`.
- **Fix:** `compute_restraint_stats(deviations, sigmas, loss, verbosity)` helper in `targets/base.py`.

### H4. Cell/spacegroup extraction duplicated in 3 file-format readers
- [io/pdb.py:79-121](../torchref/io/pdb.py#L79-L121), [io/cif_readers.py:1165-1189](../torchref/io/cif_readers.py#L1165-L1189) (both `ReflectionCIFReader` and `ModelCIFReader`), [io/ihm.py:336-356](../torchref/io/ihm.py#L336-L356)
- Each independently parses `[a, b, c, α, β, γ]` and spacegroup.
- **Fix:** `io/common.py::extract_crystallographic_metadata(block) -> (cell, spacegroup, z_value)`.

### H5. Two `von_mises_nll` implementations
- Local `_von_mises_nll` in [targets/geometry/torsions.py:20-44](../torchref/refinement/targets/geometry/torsions.py#L20-L44) vs module-level `von_mises_nll` in [targets/geometry/base.py:445-487](../torchref/refinement/targets/geometry/base.py#L445-L487).
- The torsions copy uses `torch.special.i0e(κ) + κ` (more stable for large κ); the base version uses `log(i0(...))`. Base version is imported but never called.
- **Fix:** keep the `i0e` form, move to `base.py`, delete the local copy.

### H6. Caching design drift in `ModelFT`
- [model/model_ft.py:126-127](../torchref/model/model_ft.py#L126-L127) (`_anomalous_cache`) and [model/model_ft.py:635-648](../torchref/model/model_ft.py#L635-L648) (`reset_cache`/`invalidate_cache`) live alongside `CachedForwardMixin` from [utils/caching.py:37-137](../torchref/utils/caching.py#L37-L137).
- Two parallel caching systems on the same class; `CachedForwardMixin` is used nowhere else.
- **Fix:** profile first; then either fold the anomalous cache into `CachedForwardMixin`'s fingerprint, or drop the mixin from `ModelFT` if it isn't carrying weight.

---

## MEDIUM value

### M1. Reader classes share boilerplate
- [io/mtz.py:47-203](../torchref/io/mtz.py#L47-L203), [io/cif_readers.py:569-750](../torchref/io/cif_readers.py#L569-L750), [io/pdb.py:284-368](../torchref/io/pdb.py#L284-L368)
- Same `__init__` → `read(filepath)` → `__call__()` → `(data, cell, spacegroup)` shape; identical logging and attribute init.
- **Fix:** `BaseReader` with `_read_impl()` + `_extract_cell_spacegroup()` hooks. Subclasses become ~30 LOC each.

### M2. Column-priority lists for F/I/SIGF/SIGI/RFREE duplicated
- [io/mtz.py:77-127](../torchref/io/mtz.py#L77-L127) vs hard-coded lists scattered in [io/cif_readers.py](../torchref/io/cif_readers.py) under `_extract_amplitudes_and_intensities`.
- **Fix:** single `COLUMN_REGISTRY[data_type][format]` dict in `io/common.py`.

### M3. `XrayTarget` subclasses repeat `forward()` shell
- [targets/xray/least_squares.py:40-58](../torchref/refinement/targets/xray/least_squares.py#L40-L58), [xray/gaussian.py:23-39](../torchref/refinement/targets/xray/gaussian.py#L23-L39), [xray/maximum_likelihood.py:21-37](../torchref/refinement/targets/xray/maximum_likelihood.py#L21-L37)
- Identical `forward(fcalc=None)` → `get_data()` → delegate-to-math pattern.
- **Fix:** template method in `XrayTarget` base; subclasses provide only the math function.

### M4. Tensor-row unpacking duplicated across writers
- [io/pdb.py:550-610](../torchref/io/pdb.py#L550-L610), [io/cif.py:228-337](../torchref/io/cif.py#L228-L337), [io/mtz.py:391-448](../torchref/io/mtz.py#L391-L448)
- Same pattern: row extract → dtype coerce → NaN→empty → format.
- **Fix:** `unpack_atom_row(row) -> AtomTuple` helper.

### M5. `cat_dict` safety check duplicated in geometry targets
- [bonds.py:39](../torchref/refinement/targets/geometry/bonds.py#L39), [angles.py:36](../torchref/refinement/targets/geometry/angles.py#L36), [torsions.py:165](../torchref/refinement/targets/geometry/torsions.py#L165)
- Identical `if "all" not in restraints["TYPE"]: cat_dict(...)`.
- **Fix:** move to `GeometryTarget._ensure_concatenated(type)`.

### M6. CPU-backed `.to()` override duplicated
- [model/internal_coordinates.py:147-195](../torchref/model/internal_coordinates.py#L147-L195) and [segmented_internal_coordinates.py](../torchref/model/segmented_internal_coordinates.py)
- Both override `to()` to keep params on CPU while updating `_output_device`.
- **Fix:** `CPUBackedDeviceMixin` in `utils/`.

### M7. Dataset device movement bypasses `DeviceMixin`
- [io/datasets/base.py:137-211](../torchref/io/datasets/base.py#L137-L211) defines `_get_state`/`_from_state` and walks `_tensor_fields()`; [reflection_data.py:337-344](../torchref/io/datasets/reflection_data.py#L337-L344) does manual `.to(device)` instead of relying on [utils/device_mixin.py:180-194](../torchref/utils/device_mixin.py#L180-L194).
- **Fix:** consolidate on `DeviceMixin._apply()`; reuse `_tensor_fields()` for both serialization and device walks.

---

## LOW value (small cleanups)

### L1. `dtype` / `device` resolution boilerplate
Repeated `dtype = dtype or initial_values.dtype` / `device = device or initial_values.device` in every `model/` constructor (≥6 sites). Extract `resolve_dtype_device(value, dtype, device)` helper.

### L2. Copy/clone boilerplate (~20 LOC)
[sf_fft.py:656-682](../torchref/model/sf_fft.py#L656-L682), [sf_ds.py:493-517](../torchref/model/sf_ds.py#L493-L517), [parameter_wrappers.py:586-606 and 1205-1225](../torchref/model/parameter_wrappers.py#L586-L606). Generic `deepcopy_with_components()` utility.

### L3. Unused `gaussian_nll` imports
[bonds.py:15](../torchref/refinement/targets/geometry/bonds.py#L15) and [angles.py:15](../torchref/refinement/targets/geometry/angles.py#L15) — imported, never used.

### L4. Unused `symmetry` property alias
[sf_fft.py:199-202](../torchref/model/sf_fft.py#L199-L202) — verify and drop if unreferenced.

### L5. Deprecated `FFT(*args, **kwargs)` wrapper
[sf_fft.py:685-695](../torchref/model/sf_fft.py#L685-L695) — confirm no callers, then remove.

### L6. Bare `except: pass` blocks
[io/pdb.py:274-275](../torchref/io/pdb.py#L274-L275) and [io/cif.py:446](../torchref/io/cif.py#L446) — at minimum log under verbose.

### L7. Weighting subclasses repeat trivial `__init__`
[base_weighting.py](../torchref/refinement/weighting/base_weighting.py), [policy_weighting.py](../torchref/refinement/weighting/policy_weighting.py), [component_weighting.py](../torchref/refinement/weighting/component_weighting.py), [random_weighting.py](../torchref/refinement/weighting/random_weighting.py). ~15 LOC each.

### L8. Legacy aliases
`PDB`, `MTZ`, `read = read_reflections` in [io/pdb.py:737-740](../torchref/io/pdb.py#L737-L740), [io/mtz.py:491-492](../torchref/io/mtz.py#L491-L492), [io/cif.py:456-457](../torchref/io/cif.py#L456-L457). Move behind `_deprecated` module or expose via explicit `__all__`.

---

## Already clean (no action)

- `RestraintBuilder._filter_usable_restraints`, `_build_name_to_index_map`, `_map_atoms_to_indices` in [restraints/builders.py:206-283](../torchref/restraints/builders.py#L206-L283) are correctly factored.
- `cli/_common.py` already centralizes dual-model argument parsing — no duplication between `refine.py` and `collection_difference_refine.py`.
- `ModelTarget` vs `DataTarget` property surface in [targets/base.py](../torchref/refinement/targets/base.py) is shallow-similar but intentional given different state.
- `Logger` pattern compilation in [refinement/logger.py:76-77](../torchref/refinement/logger.py#L76-L77) is already optimized.

---

## Recommended sequencing

If acting on this later, the lowest-risk highest-ROI ordering is:

1. **L3, L4, L5, L8** — quick wins, ~20 LOC, no behavior change.
2. **H3, H5, M5** — pure helper extraction in `refinement/targets/`, well-tested area.
3. **H4, M1, M2, M4** — `io/` consolidation; needs round-trip tests on real MTZ/CIF/PDB fixtures.
4. **H2, M6, L1** — mixins in `model/`; touches hot path, run full refinement benchmark before/after.
5. **H1** — `StructureFactorBase` mixin; same caution as #4, requires SF correctness check (compare `f_calc` against pre-refactor reference on at least one test structure).
6. **H6** — caching consolidation; **profile first** before deciding which direction.
