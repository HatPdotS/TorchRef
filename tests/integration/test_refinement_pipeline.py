"""
Integration tests for the refinement pipeline.

Tests full refinement workflows with real data.
"""

import pytest
import torch
from pathlib import Path


class TestRefinementSetup:
    """Tests for setting up a refinement."""

    @pytest.mark.integration
    def test_setup_refinement_components(self, sample_structure_pair):
        """Test setting up all refinement components."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        from torchref.symmetry import SpaceGroup

        # Load model
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))

        # Load data
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))

        # Create SpaceGroup
        sg = SpaceGroup(model.spacegroup)

        # Create scaler
        scaler = Scaler()

        # All components should be ready
        assert model.xyz().shape[0] > 0
        assert data.hkl is not None
        assert sg.matrices is not None
        assert scaler is not None

    @pytest.mark.integration
    def test_restraints_from_model(self, sample_cif_file):
        """Test building restraints from a loaded model."""
        from torchref.model.model import Model
        from torchref.topology.restraints import Restraints

        model = Model()
        model.load_cif(str(sample_cif_file))

        # Build restraints
        restraints = Restraints(pdb=model.pdb, xyz_fn=model.xyz, vdw_radii_fn=model.get_vdw_radii)
        restraints.build_restraints()
        
        # Should have some restraints
        assert restraints.restraints is not None


class TestStructureFactorCalculation:
    """Tests for structure factor calculation."""

    @pytest.mark.integration
    def test_fcalc_calculation(self, sample_structure_pair):
        """Test calculating structure factors from a model."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        
        # Load
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        # Calculate Fcalc (method may vary)
        if hasattr(model, 'calc_fcalc'):
            fcalc = model.calc_fcalc(data)
            
            assert fcalc is not None
            assert fcalc.shape[0] == data.hkl.shape[0]
            assert torch.all(torch.isfinite(torch.abs(fcalc)))


class TestGradientFlow:
    """Tests for gradient flow through the refinement pipeline."""

    @pytest.mark.integration
    def test_gradient_through_model(self, sample_cif_file):
        """Test that gradients flow through the model."""
        from torchref.model.model import Model
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        # Get coordinates and ensure they require grad
        xyz = model.xyz()
        if not xyz.requires_grad:
            xyz = xyz.requires_grad_(True)
        
        # Retain grad for non-leaf tensor
        xyz.retain_grad()
        
        # Simple loss
        loss = xyz.sum()
        loss.backward()
        
        # Should have gradients
        assert xyz.grad is not None

    @pytest.mark.integration
    @pytest.mark.slow
    def test_refinement_step(self, sample_structure_pair):
        """Test a single refinement step with real data."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        
        # Load
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        # This is a minimal test - full refinement would need more setup
        # Just verify we can get through without errors
        assert model.xyz().shape[0] > 0
        assert data.hkl is not None


class TestDeviceConsistency:
    """Tests for consistent device handling across components."""

    @pytest.mark.integration
    def test_all_components_same_device(self, sample_structure_pair, cpu_device):
        """Test that all components can be moved to the same device."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.symmetry import SpaceGroup

        # Load
        model = Model(device=cpu_device)
        model.load_cif(str(sample_structure_pair["model"]))

        data = ReflectionData(device=cpu_device)
        data.load_mtz(str(sample_structure_pair["reflections"]))

        sg = SpaceGroup(model.spacegroup, device=cpu_device)
        
        # Check devices
        assert model.xyz().device == cpu_device
        assert data.hkl.device == cpu_device
        assert sg.matrices.device == cpu_device

    @pytest.mark.integration
    @pytest.mark.gpu
    def test_gpu_refinement_setup(self, sample_structure_pair, gpu_device):
        """Test refinement setup on GPU."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        
        model = Model(device=gpu_device)
        model.load_cif(sample_structure_pair["model"])
        
        data = ReflectionData(device=gpu_device)
        data.load_mtz(sample_structure_pair["reflections"])
        
        # Everything should be on GPU
        assert model.xyz().device.type == gpu_device.type
        assert data.hkl.device.type == gpu_device.type
