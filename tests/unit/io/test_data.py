"""
Unit tests for torchref.io.Data

Tests ReflectionData class for handling crystallographic reflection data.
Note: Unit tests use mock data, not real file I/O.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn


class TestReflectionDataInitialization:
    """Tests for ReflectionData initialization."""

    @pytest.mark.unit
    def test_reflection_data_empty_init(self):
        """Test ReflectionData can be initialized empty."""
        from torchref.io import ReflectionData

        data = ReflectionData()

        assert data.hkl is None
        assert data.F is None
        assert data.F_sigma is None

    @pytest.mark.unit
    def test_reflection_data_is_dataclass(self):
        """ReflectionData should be a dataclass."""
        from dataclasses import is_dataclass

        from torchref.io import ReflectionData

        data = ReflectionData()

        assert is_dataclass(data)

    @pytest.mark.unit
    def test_reflection_data_default_device(self):
        """Test default device matches the package-wide configured default."""
        from torchref.config import get_default_device
        from torchref.io import ReflectionData

        data = ReflectionData()

        assert data.device == get_default_device()

    @pytest.mark.unit
    def test_reflection_data_custom_device(self):
        """Test custom device specification."""
        from torchref.io import ReflectionData

        data = ReflectionData(device="cpu")

        assert data.device.type == "cpu"

    @pytest.mark.unit
    def test_reflection_data_verbose(self):
        """Test verbosity setting."""
        from torchref.io import ReflectionData

        data = ReflectionData(verbose=2)

        assert data.verbose == 2


class TestReflectionDataDeviceMovement:
    """Tests for device movement in ReflectionData."""

    @pytest.mark.unit
    def test_reflection_data_cpu(self):
        """Test explicit CPU movement."""
        from torchref.io import ReflectionData

        data = ReflectionData()
        data = data.cpu()

        assert data.device.type == "cpu"

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_reflection_data_accelerator(self, gpu_device):
        """Movement onto whichever accelerator this host has."""
        from torchref.io import ReflectionData

        data = ReflectionData()
        data = data.to(gpu_device)

        assert data.device.type == gpu_device.type


class TestReflectionDataAttributes:
    """Tests for attribute access in ReflectionData."""

    @pytest.mark.unit
    def test_has_device_attribute(self):
        """Test device attribute is accessible."""
        from torchref.io import ReflectionData

        data = ReflectionData()

        assert hasattr(data, "device")

    @pytest.mark.unit
    def test_has_spacegroup_attribute(self):
        """Test spacegroup attribute is accessible."""
        from torchref.io import ReflectionData

        data = ReflectionData()

        assert hasattr(data, "spacegroup")

    @pytest.mark.unit
    def test_has_verbose_attribute(self):
        """Test verbose attribute is accessible."""
        from torchref.io import ReflectionData

        data = ReflectionData()

        assert hasattr(data, "verbose")


class TestReflectionDataProperties:
    """Tests for ReflectionData computed properties."""

    @pytest.mark.unit
    def test_wilson_b_default_none(self):
        """Wilson B should be None initially."""
        from torchref.io import ReflectionData

        data = ReflectionData()

        assert data.wilson_b is None

    @pytest.mark.unit
    def test_spacegroup_default_none(self):
        """Space group should be None initially."""
        from torchref.io import ReflectionData

        data = ReflectionData()

        assert data.spacegroup is None

    @pytest.mark.unit
    def test_amplitude_source_default_none(self):
        """Amplitude source should be None initially."""
        from torchref.io import ReflectionData

        data = ReflectionData()

        assert data.amplitude_source is None


class TestMockReflectionData:
    """Tests using mock reflection data."""

    @pytest.mark.unit
    def test_set_mock_hkl(self, mock_hkl_indices):
        """Test setting mock HKL indices."""
        from torchref.io import ReflectionData

        data = ReflectionData()
        hkl = mock_hkl_indices(n_reflections=100)

        # Set hkl directly (dataclass attribute)
        data.hkl = hkl.to(torch.int32)

        assert data.hkl is not None
        assert data.hkl.shape[0] == hkl.shape[0]
        assert data.hkl.shape[1] == 3

    @pytest.mark.unit
    def test_set_mock_amplitudes(self, mock_F_obs, mock_F_sigma):
        """Test setting mock structure factor amplitudes."""
        from torchref.io import ReflectionData

        data = ReflectionData()
        F = mock_F_obs(n_reflections=100)
        sigma = mock_F_sigma(n_reflections=100)

        # Set directly (dataclass attributes)
        data.F = F
        data.F_sigma = sigma

        assert data.F is not None
        assert data.F_sigma is not None
        assert torch.all(data.F > 0)
        assert torch.all(data.F_sigma > 0)


class TestFrenchWilsonToggle:
    """Tests for the french_wilson flag on ReflectionData.load()."""

    @staticmethod
    def _reader_with_I_and_F():
        """Mock reader returning data that carry BOTH intensities and
        (deliberately inconsistent) amplitudes, plus a P1 cell."""
        n = 50
        rng = np.random.default_rng(0)
        h = rng.integers(-5, 6, size=(n, 3))
        data_dict = {
            "HKL": h,
            "I": (rng.random(n) * 100 + 1.0),
            "SIGI": (rng.random(n) * 5 + 1.0),
            # Sentinel amplitudes: clearly not equal to FrenchWilson(I).
            "F": np.full(n, 7.0, dtype=np.float64),
            "SIGF": np.full(n, 0.5, dtype=np.float64),
            "I_col": "I",
            "F_col": "F",
        }
        cell = [20.0, 20.0, 20.0, 90.0, 90.0, 90.0]
        spacegroup = 1  # P1
        return lambda: (data_dict, cell, spacegroup)

    @pytest.mark.unit
    def test_french_wilson_off_uses_amplitudes_directly(self):
        """french_wilson=False uses the existing F/SIGF columns verbatim."""
        from torchref.io import ReflectionData

        data = ReflectionData(verbose=0)
        data.load(self._reader_with_I_and_F(), french_wilson=False)

        # F should be exactly the sentinel amplitude column, untouched.
        assert torch.allclose(data.F, torch.full_like(data.F, 7.0))
        # French-Wilson must not have run.
        assert data._FrenchWilson is None

    @pytest.mark.unit
    def test_french_wilson_on_derives_from_intensities(self):
        """french_wilson=True (default) re-derives F from intensities."""
        from torchref.io import ReflectionData

        data = ReflectionData(verbose=0)
        data.load(self._reader_with_I_and_F(), french_wilson=True)

        # F is computed from I, so it differs from the sentinel 7.0 column.
        assert not torch.allclose(data.F, torch.full_like(data.F, 7.0))
        assert data._FrenchWilson is not None

    @pytest.mark.unit
    def test_french_wilson_off_falls_back_when_no_amplitudes(self):
        """With only intensities present, F is derived even if FW is off."""
        from torchref.io import ReflectionData

        reader = self._reader_with_I_and_F()
        data_dict, cell, spacegroup = reader()
        del data_dict["F"], data_dict["SIGF"]

        data = ReflectionData(verbose=0)
        data.load(lambda: (data_dict, cell, spacegroup), french_wilson=False)

        # No amplitude columns => French-Wilson runs regardless of the flag.
        assert data.F is not None
        assert data._FrenchWilson is not None
