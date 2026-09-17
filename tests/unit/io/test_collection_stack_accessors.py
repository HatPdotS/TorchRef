"""The batched observation accessors must return the same data the targets fit.

Two things they could quietly get wrong, both invisible in the shape:

* returning **raw** ``F``/``I`` instead of the scaled ones, which drops the per-dataset
  joint scale parameters that ``DatasetCollection.scale()`` fits;
* masking with the 2-way ``rfree_flags`` instead of the 3-way work/free/validation
  subsets, which lets validation reflections into the work set.

Each is pinned against the per-dataset accessor it is a batched form of.
"""

import pytest
import torch


@pytest.fixture(scope="module")
def collection(mtz_dir):
    """Two 1DAW datasets with *different* scales, so raw and scaled disagree."""
    mtz = mtz_dir / "1DAW.mtz"
    if not mtz.exists():
        pytest.skip("1DAW fixture not present")

    from torchref import ReflectionData
    from torchref.io.datasets.collection import DatasetCollection

    a = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))
    b = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))
    if a.I is None:
        pytest.skip("1DAW loaded without intensities")

    dc = DatasetCollection(verbose=0, device="cpu")
    dc.add_dataset("dark", a, set_as_reference=True)
    dc.add_dataset("light", b)
    dc.scale(nsteps=1)
    with torch.no_grad():
        dc.scaler.raw_parameters[1, 0] += 0.8
    return dc


@pytest.mark.unit
class TestScaledNotRaw:
    def test_amplitude_rows_match_the_per_dataset_corrected_accessor(self, collection):
        stacked = collection.stack_F_obs()
        sigma = collection.stack_F_sigma()
        for row, key in enumerate(collection.keys()):
            F, sig = collection[key].get_corrected_data()
            assert torch.equal(stacked[row], F)
            assert torch.equal(sigma[row], sig)

    def test_intensity_rows_match_the_per_dataset_corrected_accessor(self, collection):
        stacked = collection.stack_I_obs()
        sigma = collection.stack_I_sigma()
        for row, key in enumerate(collection.keys()):
            I, sig = collection[key].get_corrected_intensities()
            assert torch.equal(stacked[row], I)
            assert torch.equal(sigma[row], sig)

    def test_the_two_datasets_differ_after_scaling(self, collection):
        """Anti-vacuity: with identical scales, raw and corrected agree and neither
        assertion above could detect the wrong accessor."""
        stacked = collection.stack_F_obs()
        assert not torch.allclose(stacked[0], stacked[1])
        raw = torch.stack([collection[k].F_raw for k in collection.keys()], dim=0)
        assert torch.allclose(raw[0], raw[1]), "raw amplitudes should be identical here"
        assert not torch.allclose(stacked, raw)

    def test_scaled_intensities_are_the_square_of_the_scaled_amplitude_factor(
        self, collection
    ):
        """Ties the two stacks together, so they cannot drift apart in scale."""
        F = collection.stack_F_obs()
        I = collection.stack_I_obs()
        raw_F = torch.stack([collection[k].F_raw for k in collection.keys()], dim=0)
        raw_I = torch.stack([collection[k].I_raw for k in collection.keys()], dim=0)

        keep = (raw_F.abs() > 1e-6) & (raw_I.abs() > 1e-6)
        amp = (F[keep] / raw_F[keep]) ** 2
        inten = I[keep] / raw_I[keep]
        assert torch.allclose(inten, amp, rtol=1e-5)


@pytest.mark.unit
class TestThreeWayMasks:
    @pytest.mark.parametrize("use_set", ["work", "free", "val"])
    def test_rows_match_the_per_dataset_subset(self, collection, use_set):
        attr = {"work": "work", "free": "free", "val": "validation"}[use_set]
        stacked = collection.stack_masks(use_set=use_set)
        for row, key in enumerate(collection.keys()):
            assert torch.equal(stacked[row], getattr(collection[key], attr).mask)

    def test_the_three_subsets_partition_the_valid_reflections(self, collection):
        work = collection.stack_masks(use_set="work")
        free = collection.stack_masks(use_set="free")
        val = collection.stack_masks(use_set="val")

        assert not (work & free).any()
        assert not (work & val).any()
        assert not (free & val).any()

        for row, key in enumerate(collection.keys()):
            valid = collection[key].masks().to(torch.bool)
            assert torch.equal(work[row] | free[row] | val[row], valid)

    def test_a_validation_set_is_carved_out_of_free_not_work(self, collection):
        """The 3-way behaviour a 2-way flag array cannot reproduce."""
        data = collection["dark"]
        saved = None if data.validation_flags is None else data.validation_flags.clone()
        try:
            free_before = int(collection.stack_masks(use_set="free")[0].sum())
            data.generate_validation_set(val_fraction_of_free=0.5, seed=0)

            free_after = int(collection.stack_masks(use_set="free")[0].sum())
            val_after = int(collection.stack_masks(use_set="val")[0].sum())

            assert val_after > 0
            assert free_after < free_before
            assert free_after + val_after == pytest.approx(free_before, abs=1)
        finally:
            data.validation_flags = saved
            data._subset_fp = None

    def test_an_unknown_subset_name_is_rejected(self, collection):
        with pytest.raises(ValueError, match="use_set must be"):
            collection.stack_masks(use_set="test")


@pytest.mark.unit
class TestSelectionAndErrors:
    def test_keys_argument_selects_and_orders_the_rows(self, collection):
        both = collection.stack_F_obs()
        one = collection.stack_F_obs(keys=["light"])
        assert one.shape[0] == 1
        assert torch.equal(one[0], both[1])

        reversed_ = collection.stack_F_obs(keys=["light", "dark"])
        assert torch.equal(reversed_[0], both[1])
        assert torch.equal(reversed_[1], both[0])

    def test_unknown_key_is_rejected(self, collection):
        with pytest.raises(KeyError, match="Unknown dataset keys"):
            collection.stack_F_obs(keys=["nope"])

    def test_centric_flags_are_shared_and_hkl_shaped(self, collection):
        centric = collection.get_centric_flags()
        assert centric is not None
        assert centric.shape == (len(collection.hkl),)
        assert centric.dtype == torch.bool

    def test_missing_intensities_name_the_offending_dataset(self, mtz_dir):
        mtz = mtz_dir / "3GR5.mtz"
        if not mtz.exists():
            pytest.skip("3GR5 fixture not present")

        from torchref import ReflectionData
        from torchref.io.datasets.collection import DatasetCollection

        amp_only = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))
        if amp_only.I is not None:
            pytest.skip("3GR5 unexpectedly carries intensities")

        dc = DatasetCollection(verbose=0, device="cpu")
        dc.add_dataset("amps", amp_only, set_as_reference=True)
        with pytest.raises(ValueError, match="'amps'"):
            dc.stack_I_obs()
