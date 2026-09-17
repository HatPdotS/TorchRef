"""CrystFEL parsing, intensity conversion and alignment on real split-half excerpts."""

import pytest
import torch

from torchref import DatasetCollection, ReflectionData

CELL = [14.97, 18.85, 18.89, 89.4, 84.9, 67.8]


@pytest.fixture(scope="module")
def halves(test_files_dir):
    return [
        ReflectionData(device="cpu", verbose=0).load_crystfel_hkl(
            str(test_files_dir / "hkl" / f"dark_half{i}.hkl"),
            cell=CELL,
            spacegroup="P 1",
        )
        for i in (1, 2)
    ]


@pytest.mark.parametrize("index", [0, 1])
def test_observations_and_metadata(halves, index, test_files_dir):
    """Preserve negative intensities, uncertainties, caller metadata and valid rows."""
    data = halves[index]
    rows = (test_files_dir / "hkl" / f"dark_half{index+1}.hkl").read_text().splitlines()
    assert rows[-1] == "End of reflections"
    assert len(data.hkl) == len(rows) - 4
    assert data.I.shape == data.I_sigma.shape == data.F.shape == (len(data),)
    assert torch.isfinite(data.I_sigma).all() and (data.I_sigma >= 0).all()
    assert (data.I < 0).any()
    assert (data.F >= 0).all() and data._FrenchWilson is not None
    torch.testing.assert_close(data.cell.data, data.cell.data.new_tensor(CELL))
    assert data.spacegroup.number == 1


def test_partial_overlap_alignment_preserves_sources(halves):
    """Align overlapping split halves on their union without mutating either input."""
    a, b = halves
    original = [data.hkl.clone() for data in halves]
    sets = [{tuple(row) for row in data.hkl.tolist()} for data in halves]
    assert sets[0] != sets[1] and sets[0] & sets[1]
    dc = DatasetCollection(verbose=0, device="cpu")
    dc.add_dataset("a", a).add_dataset("b", b)
    assert len(dc) == len(sets[0] | sets[1])
    assert dc.stack_I_obs().shape == (2, len(dc))
    for data, hkl in zip(halves, original):
        assert torch.equal(data.hkl, hkl)
