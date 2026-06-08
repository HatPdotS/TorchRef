"""
Regression tests for PDB CRYST1 parsing and occupancy handling.

Covers three deposited-PDB cases that previously crashed the reader:
  1. A header/REVDAT line that merely contains the word "CRYST1" being
     mis-parsed as the cell record.
  2. The space-group field over-reading into the Z column for 3-digit Z.
  3. Genuine occupancies > 1 (e.g. waters/ligands) rejected outright.
"""
import warnings

import pytest
import torch

# Real CRYST1 records (exact column layout) from deposited entries.
CRYST1_LARGE_Z = (
    "CRYST1  283.500  401.800  284.000  90.00  89.40  90.00 P 1 21 1    360"
)  # 1F2N: sGroup "P 1 21 1", Z=360
CRYST1_SMALL_Z = (
    "CRYST1   90.650   95.390   51.210  90.00  90.00  90.00 P 21 21 21    8"
)  # 1A0F: sGroup "P 21 21 21", Z=8
REVDAT_DECOY = (
    "REVDAT   2   31-MAY-00 1TML    1       TITLE  KEYWDS EXPDTA CRYST1"
)  # contains "CRYST1" but is not a CRYST1 record


def _write(tmp_path, lines):
    p = tmp_path / "x.pdb"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


@pytest.mark.unit
class TestReadCrystallographicInfo:
    def test_three_digit_z_does_not_corrupt_spacegroup(self, tmp_path):
        from torchref.io.pdb import read_crystallographic_info

        cell, sg, z = read_crystallographic_info(_write(tmp_path, [CRYST1_LARGE_Z]))
        assert sg == "P 1 21 1"
        assert z == "360"
        assert cell[0] == pytest.approx(283.5)

    def test_small_z_unchanged(self, tmp_path):
        from torchref.io.pdb import read_crystallographic_info

        cell, sg, z = read_crystallographic_info(_write(tmp_path, [CRYST1_SMALL_Z]))
        assert sg == "P 21 21 21"
        assert z == "8"

    def test_decoy_line_is_ignored(self, tmp_path):
        from torchref.io.pdb import read_crystallographic_info

        # The REVDAT line precedes the real CRYST1; only the latter must be read.
        cell, sg, z = read_crystallographic_info(
            _write(tmp_path, [REVDAT_DECOY, CRYST1_SMALL_Z])
        )
        assert sg == "P 21 21 21"
        assert cell[0] == pytest.approx(90.65)

    def test_no_cryst1_returns_none(self, tmp_path):
        from torchref.io.pdb import read_crystallographic_info

        assert read_crystallographic_info(_write(tmp_path, [REVDAT_DECOY])) == (
            None,
            None,
            None,
        )

    def test_malformed_cryst1_returns_none_not_raises(self, tmp_path):
        from torchref.io.pdb import read_crystallographic_info

        bad = "CRYST1   not   a   number   here  90.00  90.00 P 1"
        assert read_crystallographic_info(_write(tmp_path, [bad])) == (None, None, None)


@pytest.mark.unit
class TestOccupancyOutOfRange:
    def test_occupancy_above_one_is_clamped_with_warning(self):
        from torchref.model.parameter_wrappers import OccupancyTensor

        values = torch.tensor([0.5, 1.75, 1.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            occ = OccupancyTensor(initial_values=values)
        assert any("occupancy" in str(w.message).lower() for w in caught)
        # Effective occupancies are valid probabilities, never above 1.
        assert torch.all(occ() <= 1.0 + 1e-4)

    def test_in_range_occupancy_does_not_warn(self):
        from torchref.model.parameter_wrappers import OccupancyTensor

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            OccupancyTensor(initial_values=torch.tensor([0.3, 1.0, 0.0]))
        assert not any("occupancy" in str(w.message).lower() for w in caught)
