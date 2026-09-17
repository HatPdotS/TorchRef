"""Reading CrystFEL ``partialator`` reflection lists.

The format that merged serial data actually arrives in. Two properties matter beyond
"it parses":

* **negative intensities survive.** A merged weak reflection legitimately comes out below
  zero, and that is information -- dropping or clamping it biases the mean upward exactly
  where the noise dominates.
* **cell and space group come from the caller**, because the format carries neither.

The fixtures are 400-reflection excerpts of a real ``partialator`` custom-split pair, so the
two halves cover overlapping-but-different reflections measured independently -- the property
that makes such a pair usable as a null, where any difference between them is noise plus
systematics with no real signal in it.
"""

import pytest
import torch

# Cell of the small-molecule dataset these excerpts come from. The format does not carry
# it, so the caller must supply it; a wrong cell would silently give wrong d-spacings.
CELL = [14.97, 18.85, 18.89, 89.4, 84.9, 67.8]
SPACEGROUP = "P 1"


@pytest.fixture(scope="module")
def hkl_dir(test_files_dir):
    d = test_files_dir / "hkl"
    if not d.is_dir():
        pytest.skip("CrystFEL hkl fixtures not present")
    return d


def _load(path):
    from torchref import ReflectionData

    return ReflectionData(device="cpu", verbose=0).load_crystfel_hkl(
        str(path), cell=CELL, spacegroup=SPACEGROUP
    )


@pytest.mark.unit
class TestReaderBasics:
    def test_reads_intensities_and_sigmas(self, hkl_dir):
        data = _load(hkl_dir / "dark_half1.hkl")
        assert data.I is not None and data.I_sigma is not None
        assert len(data.I) == len(data.hkl)
        assert torch.isfinite(data.I_sigma).all()
        # Real partialator output does contain sigma(I) == 0 -- which is why an
        # intensity likelihood has to floor it rather than trust it.
        assert (data.I_sigma >= 0).all()

    def test_amplitudes_are_derived_by_french_wilson(self, hkl_dir):
        """The format is intensity-native, so F comes from the same path an MTZ with
        I/SIGI takes."""
        data = _load(hkl_dir / "dark_half1.hkl")
        assert data.F is not None
        assert (data.F >= 0).all()
        assert data._FrenchWilson is not None

    def test_cell_and_spacegroup_come_from_the_caller(self, hkl_dir):
        data = _load(hkl_dir / "dark_half1.hkl")
        assert torch.allclose(
            data.cell.data[:3].to(torch.float64),
            torch.tensor(CELL[:3], dtype=torch.float64),
            atol=1e-3,
        )
        assert data.spacegroup is not None

    def test_the_trailing_marker_is_not_read_as_a_reflection(self, hkl_dir):
        """``partialator`` ends the list with 'End of reflections'."""
        raw = (hkl_dir / "dark_half1.hkl").read_text().splitlines()
        assert raw[-1].startswith("End of reflections")
        data = _load(hkl_dir / "dark_half1.hkl")
        # 3 header lines + N reflections + 1 trailer
        assert len(data.hkl) <= len(raw) - 4


@pytest.mark.unit
class TestNegativeIntensitiesSurvive:
    def test_the_fixture_contains_negatives(self, hkl_dir):
        """Precondition, asserted: without negatives the next test proves nothing."""
        n = 0
        for line in (hkl_dir / "dark_half1.hkl").read_text().splitlines()[3:]:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    n += float(parts[3]) < 0
                except ValueError:
                    pass
        assert n > 5, f"only {n} negative intensities in the fixture"

    def test_negatives_reach_the_dataset(self, hkl_dir):
        data = _load(hkl_dir / "dark_half1.hkl")
        assert bool((data.I < 0).any()), (
            "negative intensities were dropped or clamped by the reader"
        )


@pytest.mark.unit
class TestSplitHalves:
    def test_the_two_halves_are_independent_measurements(self, hkl_dir):
        """Different reflection sets and different values -- which is what makes them
        usable as a null: any difference between them is noise plus systematics, with no
        real signal in it.

        (The excerpts are equal-length slices of the full halves, which are 33613 and
        33523 reflections; the sets still differ, which is the property that matters.)
        """
        a = _load(hkl_dir / "dark_half1.hkl")
        b = _load(hkl_dir / "dark_half2.hkl")

        set_a = {tuple(row) for row in a.hkl.tolist()}
        set_b = {tuple(row) for row in b.hkl.tolist()}
        assert set_a != set_b, "the halves cover identical reflections"
        assert set_a & set_b, "the halves share no reflections at all"

    def test_the_halves_align_into_a_collection(self, hkl_dir):
        """``add_dataset`` must reconcile two different reflection lists onto one grid."""
        from torchref.io.datasets.collection import DatasetCollection

        a = _load(hkl_dir / "dark_half1.hkl")
        b = _load(hkl_dir / "dark_half2.hkl")

        original_a, original_b = a.hkl.clone(), b.hkl.clone()
        dc = DatasetCollection(verbose=0, device="cpu")
        dc.add_dataset("half1", a, set_as_reference=True)
        dc.add_dataset("half2", b)

        assert dc.n_datasets == 2
        assert len(dc["half1"].hkl) == len(dc["half2"].hkl) == len(dc.hkl)
        assert torch.equal(a.hkl, original_a)
        assert torch.equal(b.hkl, original_b)
        # Both carry intensities, so an intensity target can run on the pair.
        assert dc["half1"].I is not None and dc["half2"].I is not None
        assert dc.stack_I_obs().shape == (2, len(dc.hkl))
