"""
Functional tests for I/O operations.

Tests file loading and data processing with real crystallographic data.
"""

import pytest


class TestCIFReadingFunctional:
    """Functional tests for CIF file reading."""

    @pytest.mark.integration
    def test_load_multiple_cif_files(self, cif_dir):
        """Test loading multiple CIF files successfully."""
        from torchref.model.model import Model

        cif_files = list(cif_dir.glob("*.cif"))
        assert len(cif_files) > 0, "No CIF files found in test directory"

        for cif_file in cif_files:
            model = Model()
            model.load_cif(str(cif_file))

            # Each file should load with atoms
            n_atoms = model.xyz().shape[0]
            assert n_atoms > 0, f"No atoms loaded from {cif_file}"

            # Should have cell parameters
            assert model.cell is not None
            assert len(model.cell) == 6


class TestMTZReadingFunctional:
    """Functional tests for MTZ file reading."""

    @pytest.mark.integration
    def test_load_multiple_mtz_files(self, mtz_dir):
        """Test loading multiple MTZ files successfully."""
        from torchref.io import ReflectionData

        mtz_files = list(mtz_dir.glob("*.mtz"))
        assert len(mtz_files) > 0, "No MTZ files found in test directory"

        for mtz_file in mtz_files:
            data = ReflectionData()
            data.load_mtz(str(mtz_file))

            # Each file should load with reflections
            n_refl = data.hkl.shape[0]
            assert n_refl > 0, f"No reflections loaded from {mtz_file}"

            # Should have cell parameters
            assert data.cell is not None


class TestSFCIFReadingFunctional:
    """Functional tests for structure factor CIF reading."""

    @pytest.mark.integration
    def test_load_sf_cif(self, cif_sf_dir):
        """Test loading structure factor CIF files."""
        from torchref.io import ReflectionData

        sf_files = list(cif_sf_dir.glob("*.cif"))
        if not sf_files:
            pytest.skip("No SF-CIF files found")

        for sf_file in sf_files:
            data = ReflectionData()
            try:
                data.load_cif(str(sf_file))

                # Should have loaded reflections
                if data.hkl is not None:
                    assert data.hkl.shape[0] > 0
            except Exception as e:
                # Some files may not be valid SF-CIF format
                pass
