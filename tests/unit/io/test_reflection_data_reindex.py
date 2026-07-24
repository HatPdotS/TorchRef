"""Regression tests for per-reflection field reindexing.

``validate_hkl`` / ``remap`` / ``reduce_to_spacegroup`` must carry EVERY
per-reflection field onto the new HKL grid, not a hand-maintained subset. The
historical bug left ``hkl_anomalous`` (read by ``hkl_for_sf``) at the
pre-alignment length, which crashed difference refinement whenever the dark and
light datasets had different reflection sets. See docs/changelog.rst 0.6.2.
"""

import pytest
import torch

from torchref.io.datasets.reflection_data import ReflectionData


def _base_grid(h=10, k=10, lmax=10):
    """A block of Miller indices with strictly-positive l (already in the ASU)."""
    hs = torch.arange(-h, h + 1)
    ks = torch.arange(-k, k + 1)
    ls = torch.arange(1, lmax + 1)
    return (
        torch.stack(torch.meshgrid(hs, ks, ls, indexing="ij"), dim=-1)
        .reshape(-1, 3)
        .to(torch.int32)
    )


def _synthetic(hkl, seed=0, device="cpu"):
    n = hkl.shape[0]
    g = torch.Generator().manual_seed(seed)
    F = torch.rand(n, generator=g) * 100.0 + 1.0
    return ReflectionData.from_tensors(
        hkl,
        F,
        F * 0.1,
        (60.0, 60.0, 60.0, 90.0, 90.0, 90.0),
        "P 21 21 21",
        device=device,
        verbose=0,
    )


class TestValidateHklReindex:
    """The reported crash lives in validate_hkl (collection HKL alignment)."""

    def test_carries_all_per_reflection_fields(self):
        grid = _base_grid()
        n = grid.shape[0]
        # ``light`` lacks the last 10%; the reference grid lacks the first 10%,
        # so the two sets genuinely differ (each has reflections the other lacks).
        light = _synthetic(grid[: int(n * 0.9)], seed=1)
        ref_hkl = grid[int(n * 0.1):].clone()
        assert len(light.hkl_anomalous) == len(light.hkl)  # sane before

        light.validate_hkl(ref_hkl)

        m = len(light.hkl)
        assert m == len(ref_hkl)
        # The field the old code left stale (the direct cause of the crash):
        assert light.hkl_anomalous.shape[0] == m
        assert light.hkl_for_sf().shape[0] == m
        # Derived-from-HKL fields recompute for the new grid:
        assert light.centric.shape[0] == m
        assert light.friedel_flags.shape[0] == m
        # Global invariant: no per-reflection tensor left at the old length.
        light._assert_per_reflection_consistent()

    def test_identical_hkl_preserves_count(self):
        grid = _base_grid(6, 6, 6)
        d = _synthetic(grid, seed=2)
        n0 = len(d.hkl)
        d.validate_hkl(d.hkl.clone())
        assert len(d.hkl) == n0
        d._assert_per_reflection_consistent()


class TestP1RoundTripReindex:
    """The same class of bug lived latently in remap/expand_to_p1 and
    reduce_to_spacegroup (silent data loss rather than a crash)."""

    def test_expand_to_p1_carries_validation_flags(self):
        grid = _base_grid(6, 6, 6)
        d = _synthetic(grid, seed=3)
        d.generate_validation_set(val_fraction_of_free=0.5, seed=0)
        assert d.validation_flags is not None

        p1 = d.expand_to_p1()
        # validation_flags used to be dropped to None by remap; now carried.
        assert p1.validation_flags is not None
        assert p1.validation_flags.shape[0] == len(p1.hkl)
        p1._assert_per_reflection_consistent()

    def test_reduce_to_spacegroup_consistent(self):
        grid = _base_grid(6, 6, 6)
        d = _synthetic(grid, seed=4)
        p1 = d.expand_to_p1()
        back = p1.reduce_to_spacegroup("P 21 21 21")
        back._assert_per_reflection_consistent()


@pytest.mark.integration
class TestCollectionDifferenceMismatchedHKL:
    """In-process reproduction of the reported failure: a DatasetCollection
    built from two different reflection sets, run through the collection
    difference target's forward (the exact call site that crashed)."""

    def test_forward_finite_with_different_reflection_sets(self, pdb_dir, mtz_dir):
        pdb = pdb_dir / "3GR5.pdb"
        mtz = mtz_dir / "3GR5.mtz"
        if not (pdb.exists() and mtz.exists()):
            pytest.skip("3GR5 fixture not present")

        from torchref.cli._common import load_model
        from torchref.config import get_default_device
        from torchref.io.datasets.collection import DatasetCollection
        from torchref.io.datasets.reflection_data import ReflectionData
        from torchref.model.model_collection import ModelCollection
        from torchref.refinement.targets.collection import CollectionDifferenceTarget
        from torchref.scaling.collection_scaler import CollectionScaler

        dev = get_default_device()
        full = ReflectionData(device=dev, verbose=0).load_mtz(str(mtz))
        n = len(full.hkl)
        idx = torch.arange(n, device=full.hkl.device)
        # Two DIFFERENT reflection sets with DIFFERENT counts (drops the last
        # 10% vs the first 15%) -> reproduces the shape-mismatch crash pre-fix.
        dark = full.__select__(idx < int(n * 0.90))
        light = full.__select__(idx >= int(n * 0.15))
        assert len(dark.hkl) != len(light.hkl)

        dc = DatasetCollection(device=dev, verbose=0)
        dc.add_dataset("dark", dark, set_as_reference=True)
        dc.add_dataset("light", light)  # -> validate_hkl aligns onto dark grid

        # High-resolution limit for the model FFT grid (covers the data).
        d_min = float(full.resolution.min())
        model_dark = load_model(str(pdb), max_res=d_min, device=dev, verbose=0)
        model_light = load_model(str(pdb), max_res=d_min, device=dev, verbose=0)
        mc = ModelCollection([model_dark, model_light], dark_key="dark", verbose=0)
        mc.add_dark()
        mc.add_timepoint("light", [0.7, 0.3])

        scaler = CollectionScaler(dc, mc, verbose=0)
        scaler.initialize()

        target = CollectionDifferenceTarget(dc, mc, scaler=scaler, verbose=0)
        loss = target.forward()
        assert torch.isfinite(loss)
