"""
Unit tests for the work / free / validation sub-set accessor API on
``ReflectionData`` (the ``_ReflectionSubset`` views and the separate
boolean ``validation_flags``).
"""

import os

import pytest
import torch

from torchref.io.datasets import ReflectionData


TEST_MTZ = os.path.join(
    os.path.dirname(__file__), "..", "..", "files", "mtz", "1DAW.mtz"
)


@pytest.fixture
def data() -> ReflectionData:
    d = ReflectionData(verbose=0)
    d.load_mtz(TEST_MTZ)
    return d


def _n_valid(data) -> int:
    """Number of reflections passing the validity masks (the union of the
    work/free/validation subsets, since each excludes invalid reflections)."""
    return int(data.masks().bool().sum())


def test_two_class_flags_have_zero_val_set(data):
    """A freshly loaded 2-class MTZ has work + free covering the valid set,
    with an empty validation set."""
    n_work = data.work.n
    n_free = data.free.n
    n_val = data.validation.n
    assert n_work + n_free + n_val == _n_valid(data)
    assert n_val == 0


def test_generate_validation_set_splits_free(data):
    """generate_validation_set carves validation out of the free set."""
    free_before = data.free.n
    data.generate_validation_set(val_fraction_of_free=0.5, seed=42)
    n_work = data.work.n
    n_free = data.free.n
    n_val = data.validation.n
    assert n_work + n_free + n_val == _n_valid(data)
    # Free shrank by roughly half; new free + val equals old free.
    assert n_free + n_val == free_before
    assert n_val > 0


def test_validation_flags_separate_from_rfree(data):
    """generate_validation_set leaves rfree_flags binary and records the
    held-out set in the separate boolean validation_flags."""
    data.generate_validation_set(val_fraction_of_free=0.5, seed=42)
    assert data.validation_flags is not None
    assert data.validation_flags.dtype == torch.bool
    # rfree_flags stays two-valued (work/free), validation is orthogonal.
    assert set(int(v) for v in torch.unique(data.rfree_flags)) <= {0, 1}


def test_view_subset_matches_index_select(data):
    """data.work.F equals data.F[data.work.indices]."""
    f_view = data.work.F
    f_expected = data.F.index_select(0, data.work.indices)
    assert torch.equal(f_view, f_expected)


def test_subsets_are_disjoint(data):
    """work / free / validation index sets never overlap."""
    data.generate_validation_set(val_fraction_of_free=0.5, seed=0)
    iw = set(data.work.indices.tolist())
    ifr = set(data.free.indices.tolist())
    iv = set(data.validation.indices.tolist())
    assert iw.isdisjoint(ifr)
    assert iw.isdisjoint(iv)
    assert ifr.isdisjoint(iv)


def test_flag_change_propagates(data):
    """Mutating the flags invalidates the cached subset indices (fingerprint)."""
    work_before = data.work.n
    # Demote 100 work reflections to the validation set.
    work_idx = data.work.indices
    vf = torch.zeros(len(data), dtype=torch.bool, device=work_idx.device)
    vf[work_idx[:100]] = True
    data.validation_flags = vf
    assert data.work.n == work_before - 100
    assert data.validation.n == 100


def test_mtz_round_trip_preserves_three_classes(tmp_path, data):
    """Writing then reloading preserves work / free / validation counts."""
    data.generate_validation_set(val_fraction_of_free=0.5, seed=42)
    n_w, n_f, n_v = data.work.n, data.free.n, data.validation.n
    out = str(tmp_path / "rt.mtz")
    data.write_mtz(out)
    d2 = ReflectionData(verbose=0)
    d2.load_mtz(out)
    assert d2.work.n == n_w
    assert d2.free.n == n_f
    assert d2.validation.n == n_v


def test_mask_and_indices_consistent(data):
    """Each subset's boolean mask and integer indices select the same rows."""
    data.generate_validation_set(val_fraction_of_free=0.5, seed=0)
    for name in ("work", "free", "validation"):
        sub = getattr(data, name)
        assert int(sub.mask.sum()) == sub.n
        assert bool(sub.mask[sub.indices].all())
