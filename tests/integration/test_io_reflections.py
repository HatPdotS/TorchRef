"""
Integration tests for reflection data loading.

Tests loading MTZ and SF-CIF files.
"""

import pytest
import torch

from torchref.config import (
    canonical_device,
    get_default_device,
    get_float_dtype,
    get_int_dtype,
)


@pytest.mark.integration
def test_mtz_loading_contract(loaded_reflection_data, sample_mtz_file) -> None:
    """MTZ loading supplies aligned observations, masks and crystal metadata."""
    import gemmi

    data = loaded_reflection_data
    reference = gemmi.read_mtz_file(str(sample_mtz_file))
    n = len(data.hkl)
    assert n > 0
    assert data.hkl.shape == (n, 3)
    assert data.hkl.dtype == get_int_dtype()
    assert canonical_device(data.hkl.device) == canonical_device(get_default_device())
    for tensor in (data.F, data.F_sigma, data.resolution):
        assert tensor.shape == (n,)
        assert tensor.dtype == get_float_dtype()
        assert canonical_device(tensor.device) == canonical_device(get_default_device())
    mask = data.masks()
    assert mask.shape == (n,)
    assert mask.dtype == torch.bool
    assert mask.any()
    assert torch.isfinite(data.F[mask]).all()
    assert torch.all(data.F[mask] >= 0)
    assert torch.isfinite(data.F_sigma[mask]).all()
    assert torch.all(data.F_sigma[mask] > 0)
    assert torch.isfinite(data.resolution).all()
    assert torch.all(data.resolution > 0)
    assert data.rfree_flags.shape == (n,)
    assert data.rfree_flags.dtype == torch.bool
    assert data.rfree_flags.any() and (~data.rfree_flags).any()
    assert 0.7 < data.rfree_flags.to(get_float_dtype()).mean().item() < 1.0
    torch.testing.assert_close(
        data.cell.data, data.F.new_tensor(reference.cell.parameters)
    )
    assert data.spacegroup.number == reference.spacegroup.number


@pytest.mark.integration
def test_resolution_bins(loaded_reflection_data) -> None:
    """Every bin mean equals the mean d-spacing of its unmasked reflections."""
    data = loaded_reflection_data
    bins, n_bins = data.get_bins(n_bins=10)
    assert bins.shape == (len(data.hkl),)
    assert n_bins > 0
    assert bins.min() >= 0 and bins.max() < n_bins
    groups = [(bins == i) & data.masks() for i in range(n_bins)]
    assert all(group.any() for group in groups)
    expected = torch.stack([data.resolution[group].mean() for group in groups])
    torch.testing.assert_close(data.mean_res_per_bin(), expected)


@pytest.mark.integration
def test_structure_pair_consistency(model_and_data) -> None:
    """Matching model and reflection files describe the same crystal."""
    model, data = model_and_data["model"], model_and_data["data"]
    assert len(model.xyz()) > 0 and len(data.hkl) > 0
    torch.testing.assert_close(model.cell.data, data.cell.data, rtol=0.01, atol=0.1)
    assert model.spacegroup.number == data.spacegroup.number


class TestSFCIFLoading:
    """Tests for loading structure factor CIF files."""

    @pytest.mark.integration
    def test_load_sf_cif(self, sample_structure_factor_cif):
        """Test loading a structure factor CIF file."""
        from torchref.io import ReflectionData

        data = ReflectionData()
        data.load_cif(str(sample_structure_factor_cif))

        assert data.hkl.shape[0] > 0
        assert data.hkl.shape[1] == 3


class TestReflectionDataProperties:
    """Tests for computed properties of reflection data."""

    @pytest.mark.integration
    def test_data_device_movement(self, sample_mtz_file, cpu_device):
        """Test moving reflection data to different devices."""
        from torchref.io import ReflectionData

        data = ReflectionData()
        data.load_mtz(str(sample_mtz_file))

        # Move to device
        data = data.to(cpu_device)

        assert data.hkl.device == cpu_device
        assert data.F.device == cpu_device
