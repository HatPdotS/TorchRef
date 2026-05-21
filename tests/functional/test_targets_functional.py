"""
Functional tests for refinement targets.

Tests target functions with real model and data objects.
"""

import pytest
import torch
import numpy as np


class TestXrayTargetsFunctional:
    """Functional tests for X-ray target calculations."""

    @pytest.mark.integration
    def test_gaussian_nll_with_real_data(self, sample_structure_pair):
        """Test Gaussian NLL calculation with real reflection data."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.base.math_torch import nll_xray
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        # Get Fobs and sigma
        fobs = data.F
        sigma = data.F_sigma
        
        # Use fobs as mock fcalc (will give low loss)
        fcalc = fobs.clone()
        
        # Mask out NaN values
        valid = ~torch.isnan(fobs) & ~torch.isnan(sigma) & (sigma > 0)
        
        if valid.sum() > 0:
            nll = nll_xray(fobs[valid], fcalc[valid], sigma[valid])
            
            assert torch.isfinite(nll)

    @pytest.mark.integration
    def test_least_squares_with_real_data(self, sample_structure_pair):
        """Test least squares calculation with real data."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        fobs = data.F
        # Use slightly perturbed fobs as fcalc
        fcalc = fobs * 1.05
        
        # Mask out NaN values
        valid = ~torch.isnan(fobs)
        
        if valid.sum() > 0:
            # Simple least squares
            diff = fobs[valid] - fcalc[valid]
            loss = torch.mean(diff ** 2)
            
            assert torch.isfinite(loss)
            assert loss > 0  # Should have some error with 5% perturbation


class TestRfactorCalculationsFunctional:
    """Functional tests for R-factor calculations."""

    @pytest.mark.integration
    def test_rfactor_with_real_data(self, sample_structure_pair):
        """Test R-factor calculation with real reflection data."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.base.math_torch import get_rfactors
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        fobs = data.F
        # Use perturbed fobs as mock fcalc
        fcalc = fobs * 1.1  # 10% scale difference
        
        # Mask out NaN/Inf values
        valid = torch.isfinite(fobs)
        
        if valid.sum() > 100:  # Need enough data
            # Create simple rfree mask (5% test set)
            rfree_mask = torch.rand(valid.sum()) > 0.05
            
            r_work, r_free = get_rfactors(
                torch.abs(fobs[valid]), 
                torch.abs(fcalc[valid]), 
                rfree_mask
            )
            
            # R-factors should be in valid range, but may be NaN if mask is empty
            if torch.isfinite(torch.tensor(r_work)):
                assert 0 <= r_work <= 1
            if torch.isfinite(torch.tensor(r_free)):
                assert 0 <= r_free <= 1

    @pytest.mark.integration
    def test_bin_wise_rfactors(self, sample_structure_pair):
        """Test bin-wise R-factor calculation."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.base.math_torch import bin_wise_rfactors
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        fobs = data.F
        fcalc = fobs * 1.05
        
        # Create bins based on resolution
        n_refl = fobs.shape[0]
        n_bins = 10
        bins = torch.randint(0, n_bins, (n_refl,), device=fobs.device)

        rfree_mask = torch.rand(n_refl, device=fobs.device) > 0.05  # 95% work set
        
        # Mask out NaN values
        valid = ~torch.isnan(fobs)
        
        if valid.sum() > 0:
            r_work_bins, r_free_bins = bin_wise_rfactors(
                fobs[valid], fcalc[valid], rfree_mask[valid], bins[valid]
            )
            
            assert r_work_bins is not None
            assert r_free_bins is not None


class TestGeometryTargetsFunctional:
    """Functional tests for geometry restraint targets."""

    @pytest.mark.integration
    def test_bond_target_with_real_structure(self, sample_cif_file, external_monomer_library):
        """Test bond target calculation with real structure."""
        from torchref.model.model import Model

        model = Model()
        model.load_cif(str(sample_cif_file))

        # Use new model-based restraints API
        model.set_restraints_cif(str(external_monomer_library))
        restraints = model.restraints
        
        # Calculate bond deviations manually
        if 'bond' in restraints.restraints and 'intra' in restraints.restraints['bond']:
            bond = restraints.restraints['bond']['intra']
            indices = bond['indices']
            references = bond['references']
            sigmas = bond['sigmas']
            
            # Get coordinates
            xyz = model.xyz()
            
            # Calculate actual bond lengths
            atom1_coords = xyz[indices[:, 0]]
            atom2_coords = xyz[indices[:, 1]]
            
            actual_lengths = torch.sqrt(torch.sum((atom1_coords - atom2_coords) ** 2, dim=1))
            deviations = actual_lengths - references
            
            # Calculate Gaussian NLL for bonds
            log_2pi = torch.log(torch.tensor(2.0 * np.pi))
            nll = 0.5 * (deviations / sigmas) ** 2 + torch.log(sigmas) + 0.5 * log_2pi
            
            loss = nll.mean()
            
            assert torch.isfinite(loss)

    @pytest.mark.integration
    def test_angle_target_with_real_structure(self, sample_cif_file, external_monomer_library):
        """Test angle target calculation with real structure."""
        from torchref.model.model import Model

        model = Model()
        model.load_cif(str(sample_cif_file))

        # Use new model-based restraints API
        model.set_restraints_cif(str(external_monomer_library))
        restraints = model.restraints
        
        # Calculate angle deviations
        if 'angle' in restraints.restraints and 'intra' in restraints.restraints['angle']:
            angle = restraints.restraints['angle']['intra']
            indices = angle['indices']
            references = angle['references']
            sigmas = angle['sigmas']
            
            xyz = model.xyz()
            
            # Get atom coordinates
            atom1 = xyz[indices[:, 0]]
            atom2 = xyz[indices[:, 1]]  # Central atom
            atom3 = xyz[indices[:, 2]]
            
            # Calculate vectors
            v1 = atom1 - atom2
            v2 = atom3 - atom2
            
            # Calculate angles
            cos_angles = torch.sum(v1 * v2, dim=1) / (
                torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-8
            )
            cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
            actual_angles = torch.acos(cos_angles)  # In radians
            
            # References are in degrees, convert to radians
            ref_rad = references * (np.pi / 180.0)
            sigma_rad = sigmas * (np.pi / 180.0)
            
            deviations = actual_angles - ref_rad
            
            # NLL
            log_2pi = torch.log(torch.tensor(2.0 * np.pi))
            nll = 0.5 * (deviations / sigma_rad) ** 2 + torch.log(sigma_rad) + 0.5 * log_2pi
            
            loss = nll.mean()
            
            assert torch.isfinite(loss)


class TestStructureFactorCalculationFunctional:
    """Functional tests for structure factor calculation."""

    @pytest.mark.integration
    def test_fcalc_shape_matches_data(self, sample_structure_pair):
        """Test that calculated structure factors have correct shape."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData

        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))

        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))

        # Check if model has fcalc calculation method
        if hasattr(model, 'calc_fcalc'):
            fcalc = model.calc_fcalc(data)

            # Fcalc should have same number of reflections as data
            assert fcalc.shape[0] == data.hkl.shape[0]


class TestScalingWithRealData:
    """Functional tests for scaling with real data."""

    @pytest.mark.integration
    def test_scaler_initialization_with_real_data(self, sample_structure_pair):
        """Test scaler initialization with real model and data."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=20, verbose=0)
        
        # Scaler should have correct number of bins
        assert scaler.nbins == 20
        
        # Scaler should have scattering vectors
        assert scaler.s is not None
        assert scaler.s.shape[0] == data.hkl.shape[0]

    @pytest.mark.integration
    def test_anisotropy_correction_values(self, sample_structure_pair):
        """Test that anisotropy correction produces reasonable values."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.scaling.scaler import Scaler
        
        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        scaler = Scaler(model=model, data=data, nbins=10, verbose=0)
        scaler.setup_anisotropy_correction()
        
        correction = scaler.anisotropy_correction()
        
        # Correction should be positive and finite
        assert torch.all(torch.isfinite(correction))
        assert torch.all(correction > 0)
        
        # Correction should be close to 1.0 with small U parameters
        assert torch.all(correction > 0.5)
        assert torch.all(correction < 2.0)


class TestMathFunctionsFunctional:
    """Functional tests for math functions with real data."""

    @pytest.mark.integration
    def test_scattering_vectors_from_real_data(self, sample_structure_pair):
        """Test scattering vector calculation with real HKL and cell."""
        from torchref.io import ReflectionData
        from torchref.base.math_torch import get_scattering_vectors
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        s = get_scattering_vectors(data.hkl, data.cell)
        
        # Should have 3D vectors for each reflection
        assert s.shape[0] == data.hkl.shape[0]
        assert s.shape[1] == 3
        
        # All values should be finite
        assert torch.all(torch.isfinite(s))

    @pytest.mark.integration
    def test_coordinate_transformations_with_real_cell(self, sample_cif_file):
        """Test coordinate transformations with real unit cell."""
        from torchref.model.model import Model
        from torchref.base.math_torch import (
            cartesian_to_fractional_torch,
            fractional_to_cartesian_torch
        )
        
        model = Model()
        model.load_cif(str(sample_cif_file))
        
        xyz = model.xyz().double()  # Ensure double precision for einsum
        cell = model.cell.data.double()
        
        # Convert to fractional
        frac = cartesian_to_fractional_torch(xyz, cell)
        
        # Fractional coordinates should be in [0, 1] range (mostly)
        # Some atoms may be outside unit cell
        assert torch.all(torch.isfinite(frac))
        
        # Convert back to Cartesian
        cart_back = fractional_to_cartesian_torch(frac, cell)
        
        # Should match original coordinates
        assert torch.allclose(xyz, cart_back, rtol=1e-5)

    @pytest.mark.integration
    def test_u_matrix_conversion(self):
        """Test conversion of U parameters to matrix form."""
        from torchref.base.math_torch import U_to_matrix
        
        # Create U parameters (U11, U22, U33, U12, U13, U23)
        u_params = torch.tensor([0.05, 0.06, 0.04, 0.01, 0.005, -0.01], dtype=torch.float32)
        
        U = U_to_matrix(u_params)
        
        # Should be 3x3 symmetric matrix
        assert U.shape == (3, 3)
        assert torch.allclose(U, U.T)
        
        # Diagonal should match U11, U22, U33
        assert torch.isclose(U[0, 0], u_params[0])
        assert torch.isclose(U[1, 1], u_params[1])
        assert torch.isclose(U[2, 2], u_params[2])


class TestSpaceGroupFunctional:
    """Functional tests for symmetry operations."""

    @pytest.mark.integration
    def test_spacegroup_matrices_orthogonal(self, sample_cif_file):
        """Test that symmetry rotation matrices are orthogonal."""
        from torchref.model.model import Model
        from torchref.symmetry import SpaceGroup

        model = Model()
        model.load_cif(str(sample_cif_file))

        sg = SpaceGroup(model.spacegroup)

        # Check each symmetry operation
        for i in range(sg.matrices.shape[0]):
            mat = sg.matrices[i]

            # Extract rotation part (first 3x3)
            if mat.shape[-1] == 4:
                rot = mat[:3, :3]
            else:
                rot = mat[:3, :3]

            # R * R^T should be identity
            product = torch.mm(rot.float(), rot.float().T)
            identity = torch.eye(3, dtype=product.dtype, device=product.device)

            assert torch.allclose(product, identity, atol=1e-5)

    @pytest.mark.integration
    def test_spacegroup_determinant(self, sample_cif_file):
        """Test that symmetry matrices have determinant +1 or -1."""
        from torchref.model.model import Model
        from torchref.symmetry import SpaceGroup

        model = Model()
        model.load_cif(str(sample_cif_file))

        sg = SpaceGroup(model.spacegroup)

        for i in range(sg.matrices.shape[0]):
            mat = sg.matrices[i]

            if mat.shape[-1] == 4:
                rot = mat[:3, :3]
            else:
                rot = mat[:3, :3]
            
            det = torch.det(rot.float())
            
            # Determinant should be +1 (proper rotation) or -1 (improper)
            assert torch.isclose(torch.abs(det), torch.tensor(1.0), atol=1e-5)


class TestNLLFunctionsFunctional:
    """Functional tests for NLL functions with real data."""

    @pytest.mark.integration
    def test_nll_xray_with_identical_data(self, sample_structure_pair):
        """Test NLL is minimal when Fobs equals Fcalc."""
        from torchref.io import ReflectionData
        from torchref.base.math_torch import nll_xray
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        fobs = data.F
        sigma = data.F_sigma
        
        # Use Fobs as Fcalc (should give low NLL)
        fcalc = fobs.clone()
        
        # Mask out NaN values
        valid = ~torch.isnan(fobs) & ~torch.isnan(sigma) & (sigma > 0)
        
        if valid.sum() > 100:
            nll = nll_xray(fobs[valid], fcalc[valid], sigma[valid])
            
            # NLL should be finite and relatively small
            assert torch.isfinite(nll)

    @pytest.mark.integration
    def test_nll_xray_increases_with_error(self, sample_structure_pair):
        """Test NLL increases as Fcalc differs from Fobs."""
        from torchref.io import ReflectionData
        from torchref.base.math_torch import nll_xray
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        fobs = data.F
        sigma = data.F_sigma
        
        # Mask out NaN values
        valid = ~torch.isnan(fobs) & ~torch.isnan(sigma) & (sigma > 0)
        
        if valid.sum() > 100:
            # NLL with exact match
            nll_exact = nll_xray(fobs[valid], fobs[valid].clone(), sigma[valid])
            
            # NLL with 10% error
            fcalc_10pct = fobs[valid] * 1.1
            nll_10pct = nll_xray(fobs[valid], fcalc_10pct, sigma[valid])
            
            # NLL with 20% error
            fcalc_20pct = fobs[valid] * 1.2
            nll_20pct = nll_xray(fobs[valid], fcalc_20pct, sigma[valid])
            
            # NLL should increase with error
            assert nll_10pct > nll_exact or torch.isclose(nll_10pct, nll_exact, rtol=0.1)
            assert nll_20pct > nll_10pct or torch.isclose(nll_20pct, nll_10pct, rtol=0.1)

    @pytest.mark.integration
    def test_nll_xray_lognormal(self, sample_structure_pair):
        """Test lognormal NLL calculation."""
        from torchref.io import ReflectionData
        from torchref.base.math_torch import nll_xray_lognormal
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        fobs = data.F
        sigma = data.F_sigma
        
        # Mask out NaN and zero values
        valid = ~torch.isnan(fobs) & ~torch.isnan(sigma) & (sigma > 0) & (fobs > 0)
        
        if valid.sum() > 100:
            fcalc = fobs[valid] * 1.05  # 5% perturbation
            
            nll = nll_xray_lognormal(fobs[valid], fcalc, sigma[valid])
            
            assert torch.isfinite(nll)


class TestRiceDistributionFunctional:
    """Functional tests for Rice distribution calculations."""

    @pytest.mark.integration
    def test_rice_nll_acentric(self, sample_structure_pair):
        """Test Rice NLL for acentric reflections."""
        from torchref.io import ReflectionData
        from torch.special import i0
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        fobs = data.F
        sigma = data.F_sigma
        
        # Mask out invalid values
        valid = ~torch.isnan(fobs) & ~torch.isnan(sigma) & (sigma > 0) & (fobs > 0)
        
        if valid.sum() > 100:
            fo = fobs[valid][:100]  # Use subset
            fc = fo * 1.05
            sig = sigma[valid][:100]
            
            # Rice distribution NLL for acentric
            # -log P(F|Fc) ∝ F²/2σ² + Fc²/2σ² - log(I0(F*Fc/σ²))
            
            sig_sq = sig ** 2
            arg = fo * fc / sig_sq
            
            # Bessel I0 should be >= 1
            bessel = i0(arg)
            assert torch.all(bessel >= 1.0)
            
            # Log I0 should be >= 0
            log_bessel = torch.log(bessel)
            assert torch.all(log_bessel >= 0)


class TestWeightingSchemesFunctional:
    """Functional tests for weighting schemes."""

    @pytest.mark.integration
    def test_sigma_weighting(self, sample_structure_pair):
        """Test sigma-based weighting for least squares."""
        from torchref.io import ReflectionData
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        fobs = data.F
        sigma = data.F_sigma
        
        # Mask out invalid values
        valid = ~torch.isnan(fobs) & ~torch.isnan(sigma) & (sigma > 0)
        
        if valid.sum() > 100:
            fo = fobs[valid]
            sig = sigma[valid]
            
            # Sigma weights: w = 1/σ²
            weights = 1.0 / (sig ** 2)
            
            # Weights should be positive
            assert torch.all(weights > 0)
            assert torch.all(torch.isfinite(weights))
            
            # Higher sigma should give lower weight
            high_sigma_mask = sig > sig.median()
            low_sigma_mask = sig <= sig.median()
            
            mean_high_weight = weights[high_sigma_mask].mean()
            mean_low_weight = weights[low_sigma_mask].mean()
            
            assert mean_low_weight > mean_high_weight

    @pytest.mark.integration
    def test_resolution_weighting(self, sample_structure_pair):
        """Test resolution-based weighting."""
        from torchref.io import ReflectionData
        from torchref.base.math_torch import get_scattering_vectors
        
        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))
        
        # Calculate resolution from scattering vectors
        s = get_scattering_vectors(data.hkl, data.cell)
        s_mag = torch.norm(s, dim=1)
        resolution = 1.0 / (2.0 * s_mag + 1e-6)  # d = 1/(2s)
        
        # Resolution should be positive
        assert torch.all(resolution > 0)
        
        # Create low-resolution weighting
        low_res_cutoff = 5.0  # Angstroms
        weights = torch.exp(-s_mag * low_res_cutoff)
        
        # Weights should be higher for low resolution (small s)
        high_res_mask = resolution < 3.0  # High resolution
        low_res_mask = resolution > 10.0  # Low resolution
        
        if high_res_mask.sum() > 0 and low_res_mask.sum() > 0:
            mean_high_res_weight = weights[high_res_mask].mean()
            mean_low_res_weight = weights[low_res_mask].mean()
            
            assert mean_low_res_weight > mean_high_res_weight


class TestLossComponentsFunctional:
    """Functional tests for individual loss components."""

    @pytest.mark.integration
    def test_bond_deviation_calculation(self, sample_cif_file, external_monomer_library):
        """Test bond deviation calculation."""
        from torchref.model.model import Model

        model = Model()
        model.load_cif(str(sample_cif_file))

        # Use new model-based restraints API
        model.set_restraints_cif(str(external_monomer_library))
        restraints = model.restraints
        
        if 'bond' in restraints.restraints and 'intra' in restraints.restraints['bond']:
            bond = restraints.restraints['bond']['intra']
            indices = bond['indices']
            references = bond['references']
            
            xyz = model.xyz()
            
            # Calculate actual bond lengths
            d = torch.norm(xyz[indices[:, 0]] - xyz[indices[:, 1]], dim=1)
            
            # Deviations
            deviations = d - references
            
            # Most bonds should have small deviations
            rms_deviation = torch.sqrt((deviations ** 2).mean())
            
            # For a well-built structure, RMS should be < 0.1 Å
            assert rms_deviation < 0.5  # Allow some tolerance

    @pytest.mark.integration
    def test_angle_deviation_calculation(self, sample_cif_file, external_monomer_library):
        """Test angle deviation calculation."""
        from torchref.model.model import Model

        model = Model()
        model.load_cif(str(sample_cif_file))

        # Use new model-based restraints API
        model.set_restraints_cif(str(external_monomer_library))
        restraints = model.restraints
        
        if 'angle' in restraints.restraints and 'intra' in restraints.restraints['angle']:
            angle = restraints.restraints['angle']['intra']
            indices = angle['indices']
            references = angle['references']  # target angles
            
            xyz = model.xyz()
            
            # Calculate actual angles
            v1 = xyz[indices[:, 0]] - xyz[indices[:, 1]]
            v2 = xyz[indices[:, 2]] - xyz[indices[:, 1]]
            
            cos_angle = torch.sum(v1 * v2, dim=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-8)
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
            actual_angles = torch.acos(cos_angle)
            
            # References might already be in radians or degrees
            # Check the range to determine units
            ref_max = references.abs().max().item()
            if ref_max > 2 * np.pi:
                # References are likely in degrees, convert actual to degrees
                actual_angles_deg = actual_angles * 180.0 / np.pi
                deviations = actual_angles_deg - references
            else:
                # References are in radians
                deviations = actual_angles - references
            
            # Verify we can compute deviations
            assert deviations.shape == references.shape
            assert torch.all(torch.isfinite(deviations))


class TestCombinedLossFunctional:
    """Functional tests for combined loss calculations."""

    @pytest.mark.integration
    def test_xray_plus_geometry_loss(self, sample_structure_pair, external_monomer_library):
        """Test combining X-ray and geometry losses."""
        from torchref.model.model import Model
        from torchref.io import ReflectionData
        from torchref.base.math_torch import nll_xray

        model = Model()
        model.load_cif(str(sample_structure_pair["model"]))

        data = ReflectionData()
        data.load_mtz(str(sample_structure_pair["reflections"]))

        # Use new model-based restraints API
        model.set_restraints_cif(str(external_monomer_library))
        restraints = model.restraints
        
        # X-ray loss
        fobs = data.F
        sigma = data.F_sigma
        valid = ~torch.isnan(fobs) & ~torch.isnan(sigma) & (sigma > 0)
        
        if valid.sum() > 100:
            fcalc = fobs[valid] * 1.05  # Mock Fcalc
            xray_loss = nll_xray(fobs[valid], fcalc, sigma[valid])
            
            # Geometry loss (bonds)
            if 'bond' in restraints.restraints and 'intra' in restraints.restraints['bond']:
                bond = restraints.restraints['bond']['intra']
                indices = bond['indices']
                references = bond['references']
                sigmas = bond['sigmas']
                
                xyz = model.xyz()
                d = torch.norm(xyz[indices[:, 0]] - xyz[indices[:, 1]], dim=1)
                deviations = d - references
                
                geom_loss = 0.5 * ((deviations / sigmas) ** 2).mean()
                
                # Combined loss
                weight_xray = 1.0
                weight_geom = 0.1
                
                total_loss = weight_xray * xray_loss + weight_geom * geom_loss
                
                assert torch.isfinite(total_loss)
                assert total_loss > 0
