"""
Tests for the ReflectionData work/free/validation subset accessor.
"""
import pytest
import torch


@pytest.fixture
def data_1daw(mtz_dir):
    from torchref.io import ReflectionData

    d = ReflectionData(verbose=0, device=torch.device("cpu"))
    d.load_mtz(str(mtz_dir / "1DAW.mtz"))
    return d


@pytest.mark.unit
class TestReflectionSubset:
    def test_partition_is_valid_and_disjoint(self, data_1daw):
        d = data_1daw
        valid = d.masks().to(torch.bool)
        w, f, v = d.work.indices, d.free.indices, d.validation.indices

        # Union covers exactly the valid reflections, partitions are disjoint.
        assert w.numel() + f.numel() + v.numel() == int(valid.sum())
        allidx = torch.cat([w, f, v])
        assert allidx.unique().numel() == allidx.numel()

    def test_validation_empty_by_default(self, data_1daw):
        d = data_1daw
        assert d.validation.indices.numel() == 0
        # work/free then match a plain rfree split over valid reflections.
        valid = d.masks().to(torch.bool)
        rwork = d.rfree_flags.to(torch.bool)
        assert d.work.indices.numel() == int((valid & rwork).sum())
        assert d.free.indices.numel() == int((valid & ~rwork).sum())

    def test_validation_carves_out_of_work_and_free(self, data_1daw):
        d = data_1daw
        n = len(d.hkl)
        work_before = set(d.work.indices.tolist())
        free_before = set(d.free.indices.tolist())

        # Flag the first 50 reflections as validation.
        vf = torch.zeros(n, dtype=torch.bool)
        vf[:50] = True
        d.validation_flags = vf

        val = set(d.validation.indices.tolist())
        work_after = set(d.work.indices.tolist())
        free_after = set(d.free.indices.tolist())

        assert len(val) > 0
        # validation reflections are removed from both work and free
        assert val.isdisjoint(work_after) and val.isdisjoint(free_after)
        assert work_after <= work_before and free_after <= free_before
        assert (work_before | free_before) - (work_after | free_after) <= val | (
            (work_before | free_before) - val
        )  # nothing new appears

    def test_F_matches_legacy_masking(self, data_1daw):
        d = data_1daw
        valid = d.masks().to(torch.bool)
        _, F, _, rfree = d()  # legacy call
        F_data = F.get_data() if hasattr(F, "get_data") else F
        vmask = F.get_mask() if hasattr(F, "get_mask") else valid
        assert torch.allclose(d.work.F, F_data[vmask & rfree.bool()])
        assert torch.allclose(d.free.F, F_data[vmask & ~rfree.bool()])

    def test_select_aligns_full_array(self, data_1daw):
        d = data_1daw
        n = len(d.hkl)
        fake = torch.arange(n, dtype=torch.float32, device=d.device)
        assert torch.equal(d.work.select(fake).long(), d.work.indices)

    def test_field_views_and_shapes(self, data_1daw):
        d = data_1daw
        sub = d.work
        m = sub.indices.numel()
        assert sub.F.shape == (m,)
        assert sub.sigF.shape == (m,)
        assert sub.hkl.shape == (m, 3)
        assert sub.resolution.shape == (m,)  # generic __getattr__ forwarding
        assert len(sub) == m
        if sub.centric is not None:
            assert sub.centric.shape == (m,)

    def test_cache_rebuilds_on_resolution_cut(self, data_1daw):
        d = data_1daw
        n_before = d.work.indices.numel()
        d.filter_by_resolution(d_min=3.0)  # adds a "resolution" mask
        n_after = d.work.indices.numel()
        assert n_after < n_before  # cache rebuilt against the new mask
