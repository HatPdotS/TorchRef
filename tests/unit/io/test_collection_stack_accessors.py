"""Collection row selection, partitions and error handling."""

import pytest
import torch


@pytest.fixture(scope="module")
def collection(mtz_dir):
    """Two 1DAW datasets for selection and partition checks."""
    mtz = mtz_dir / "1DAW.mtz"
    if not mtz.exists():
        pytest.skip("1DAW fixture not present")

    from torchref import ReflectionData
    from torchref.io.datasets.collection import DatasetCollection

    a = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))
    if a.I is None:
        pytest.skip("1DAW loaded without intensities")

    dc = DatasetCollection(verbose=0, device="cpu")
    dc.add_dataset("dark", a, set_as_reference=True)
    dc.add_dataset("light", a)
    return dc


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
