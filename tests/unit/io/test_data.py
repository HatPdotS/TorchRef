"""
Unit tests for torchref.io.Data

Tests ReflectionData class for handling crystallographic reflection data.
Note: Unit tests use mock data, not real file I/O.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


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
        from torchref.io import ReflectionData
        from dataclasses import is_dataclass

        data = ReflectionData()

        assert is_dataclass(data)

    @pytest.mark.unit
    def test_reflection_data_default_device(self):
        """Test default device is CPU."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        
        assert data.device == torch.device('cpu')

    @pytest.mark.unit
    def test_reflection_data_custom_device(self):
        """Test custom device specification."""
        from torchref.io import ReflectionData
        
        data = ReflectionData(device='cpu')
        
        assert data.device.type == 'cpu'

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
        
        assert data.device.type == 'cpu'

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_reflection_data_cuda(self, gpu_device):
        """Test CUDA movement."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        data = data.cuda()
        
        assert data.device.type == 'cuda'


class TestReflectionDataAttributes:
    """Tests for attribute access in ReflectionData."""

    @pytest.mark.unit
    def test_has_device_attribute(self):
        """Test device attribute is accessible."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        
        assert hasattr(data, 'device')

    @pytest.mark.unit
    def test_has_spacegroup_attribute(self):
        """Test spacegroup attribute is accessible."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        
        assert hasattr(data, 'spacegroup')

    @pytest.mark.unit
    def test_has_verbose_attribute(self):
        """Test verbose attribute is accessible."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        
        assert hasattr(data, 'verbose')


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
    def test_set_mock_amplitudes(self, mock_fobs, mock_sigfobs):
        """Test setting mock structure factor amplitudes."""
        from torchref.io import ReflectionData

        data = ReflectionData()
        F = mock_fobs(n_reflections=100)
        sigma = mock_sigfobs(n_reflections=100)

        # Set directly (dataclass attributes)
        data.F = F
        data.F_sigma = sigma

        assert data.F is not None
        assert data.F_sigma is not None
        assert torch.all(data.F > 0)
        assert torch.all(data.F_sigma > 0)
