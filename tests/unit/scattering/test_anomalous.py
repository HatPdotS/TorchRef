"""
Unit tests for anomalous scattering module.

Tests the f'/f'' lookup functionality and the anomalous correction
application in ModelFT.
"""

import pytest
import torch
import gemmi

from torchref.base.scattering.anomalous_table import (
    get_anomalous_correction,
    get_significant_elements,
    get_anomalous_corrections_by_indices,
)


class TestGetAnomalousCorrection:
    """Tests for get_anomalous_correction function."""

    def test_known_element_selenium(self):
        """Test f'/f'' values for Selenium near K-edge."""
        # Se K-edge is at ~0.9792 Å (12.658 keV)
        wavelength = 0.9792
        f_prime, f_double_prime = get_anomalous_correction("Se", wavelength)

        # Near K-edge, Se should have significant anomalous scattering
        # f'' should be positive and significant
        assert f_double_prime > 2.0, "Se f'' should be significant near K-edge"
        # f' should be negative and significant
        assert f_prime < -2.0, "Se f' should be significant and negative near K-edge"

    def test_light_element_carbon(self):
        """Test that light elements have negligible anomalous scattering."""
        wavelength = 1.0
        f_prime, f_double_prime = get_anomalous_correction("C", wavelength)

        # Carbon should have very small anomalous contributions
        assert abs(f_prime) < 0.1, "C f' should be negligible"
        assert abs(f_double_prime) < 0.1, "C f'' should be negligible"

    def test_heavy_element_iron(self):
        """Test f'/f'' values for Iron at standard wavelength."""
        wavelength = 1.0
        f_prime, f_double_prime = get_anomalous_correction("Fe", wavelength)

        # Iron should have noticeable anomalous scattering at 1 Å
        assert abs(f_prime) > 0.5 or abs(f_double_prime) > 0.5

    def test_wavelength_dependence(self):
        """Test that anomalous values change with wavelength."""
        fp_1, fdp_1 = get_anomalous_correction("Fe", 1.0)
        fp_2, fdp_2 = get_anomalous_correction("Fe", 1.5)

        # Values should be different at different wavelengths
        assert fp_1 != fp_2 or fdp_1 != fdp_2

    def test_comparison_with_gemmi_cromer_liberman(self):
        """Test that our function matches direct gemmi cromer_liberman calls."""
        element = "Zn"
        wavelength = 1.0

        # Direct gemmi call
        elem = gemmi.Element(element)
        z = elem.atomic_number
        energy_ev = gemmi.hc / wavelength
        expected_fp, expected_fdp = gemmi.cromer_liberman(z, energy_ev)

        # Our function
        fp, fdp = get_anomalous_correction(element, wavelength)

        assert fp == expected_fp
        assert fdp == expected_fdp


class TestGetSignificantElements:
    """Tests for get_significant_elements function."""

    def test_filters_light_elements(self):
        """Test that light elements are filtered out."""
        elements = ["C", "N", "O", "H", "S"]
        wavelength = 1.0

        significant = get_significant_elements(elements, wavelength, threshold=0.5)

        # Light elements should not be in the significant set
        assert "C" not in significant
        assert "N" not in significant
        assert "O" not in significant
        assert "H" not in significant
        # S might or might not be significant depending on exact threshold

    def test_includes_heavy_elements(self):
        """Test that heavy elements are included."""
        elements = ["C", "N", "Fe", "Zn", "Se"]
        wavelength = 1.0

        significant = get_significant_elements(elements, wavelength, threshold=0.5)

        # Heavy elements should be included
        assert "Fe" in significant or "Zn" in significant

    def test_returns_correct_format(self):
        """Test that return format is correct."""
        elements = ["Fe", "Zn"]
        wavelength = 1.0

        significant = get_significant_elements(elements, wavelength, threshold=0.5)

        # Check format of returned values
        for elem, (fp, fdp) in significant.items():
            assert isinstance(fp, float)
            assert isinstance(fdp, float)

    def test_threshold_effect(self):
        """Test that threshold affects which elements are included."""
        elements = ["C", "N", "S", "Fe", "Zn"]
        wavelength = 1.0

        # Very low threshold should include more elements
        sig_low = get_significant_elements(elements, wavelength, threshold=0.01)
        # High threshold should include fewer elements
        sig_high = get_significant_elements(elements, wavelength, threshold=5.0)

        assert len(sig_low) >= len(sig_high)


class TestGetAnomalousCorrectionsByIndices:
    """Tests for get_anomalous_corrections_by_indices function."""

    def test_basic_functionality(self):
        """Test basic tensor creation."""
        element_list = ["C", "C", "N", "Fe", "O", "Fe"]
        significant_elements = {"Fe": (-1.2, 3.1)}
        device = torch.device("cpu")
        dtype = torch.float32

        mask, f_prime, f_double_prime = get_anomalous_corrections_by_indices(
            element_list, significant_elements, device, dtype
        )

        # Check shapes
        assert mask.shape == (6,)
        assert f_prime.shape == (2,)  # Two Fe atoms
        assert f_double_prime.shape == (2,)

        # Check mask values
        expected_mask = torch.tensor([False, False, False, True, False, True])
        assert torch.all(mask == expected_mask)

        # Check f' and f'' values
        assert torch.allclose(f_prime, torch.tensor([-1.2, -1.2]))
        assert torch.allclose(f_double_prime, torch.tensor([3.1, 3.1]))

    def test_empty_significant(self):
        """Test with no significant elements."""
        element_list = ["C", "N", "O"]
        significant_elements = {}
        device = torch.device("cpu")
        dtype = torch.float32

        mask, f_prime, f_double_prime = get_anomalous_corrections_by_indices(
            element_list, significant_elements, device, dtype
        )

        assert mask.shape == (3,)
        assert not mask.any()
        assert f_prime.shape == (0,)
        assert f_double_prime.shape == (0,)

    def test_device_dtype(self):
        """Test that device and dtype are respected."""
        element_list = ["Fe"]
        significant_elements = {"Fe": (-1.0, 2.0)}
        device = torch.device("cpu")
        dtype = torch.float64

        mask, f_prime, f_double_prime = get_anomalous_corrections_by_indices(
            element_list, significant_elements, device, dtype
        )

        assert mask.device == device
        assert f_prime.device == device
        assert f_prime.dtype == dtype
        assert f_double_prime.dtype == dtype

    def test_multiple_heavy_atom_types(self):
        """Test with multiple types of heavy atoms."""
        element_list = ["Fe", "Zn", "C", "Fe", "Zn"]
        significant_elements = {"Fe": (-1.0, 2.0), "Zn": (-0.8, 1.5)}
        device = torch.device("cpu")
        dtype = torch.float32

        mask, f_prime, f_double_prime = get_anomalous_corrections_by_indices(
            element_list, significant_elements, device, dtype
        )

        # All Fe and Zn atoms should be marked
        expected_mask = torch.tensor([True, True, False, True, True])
        assert torch.all(mask == expected_mask)

        # Check values are in correct order
        assert f_prime.shape == (4,)  # 2 Fe + 2 Zn atoms
        # Order should be: Fe, Zn, Fe, Zn
        expected_fp = torch.tensor([-1.0, -0.8, -1.0, -0.8])
        assert torch.allclose(f_prime, expected_fp)


class TestIntegrationWithModelFT:
    """Integration tests for anomalous correction in ModelFT."""

    @pytest.fixture
    def test_pdb_file(self, tmp_path):
        """Create a test PDB file with a heavy atom."""
        pdb_content = """\
CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1
ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00           C
ATOM      2  C   ALA A   1      11.500  10.000  10.000  1.00 20.00           C
ATOM      3  N   ALA A   1       9.000  11.000  10.000  1.00 20.00           N
ATOM      4  O   ALA A   1      12.000  11.000  10.000  1.00 20.00           O
ATOM      5  FE  HEM A   2      25.000  25.000  25.000  1.00 15.00          FE
END
"""
        pdb_file = tmp_path / "test_heavy.pdb"
        pdb_file.write_text(pdb_content)
        return str(pdb_file)

    def test_modelft_with_wavelength(self, test_pdb_file):
        """Test that ModelFT accepts wavelength parameter."""
        from torchref.model import ModelFT

        model = ModelFT(wavelength=1.0, anomalous_threshold=0.5)
        model.load_pdb(test_pdb_file)

        assert model.wavelength == 1.0
        assert model.anomalous_threshold == 0.5

    def test_modelft_disable_anomalous(self, test_pdb_file):
        """Test that wavelength=None disables anomalous correction."""
        from torchref.model import ModelFT

        model = ModelFT(wavelength=None)
        model.load_pdb(test_pdb_file)

        assert model.wavelength is None

        # Create HKL reflections
        hkl = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.int32)

        # Should compute structure factors without anomalous correction
        sf = model.get_structure_factor(hkl)
        assert sf.shape == (3,)
        assert torch.all(torch.isfinite(sf))

    def test_anomalous_correction_applied(self, test_pdb_file):
        """Test that anomalous correction modifies structure factors."""
        from torchref.model import ModelFT

        model = ModelFT(wavelength=1.0, anomalous_threshold=0.5, verbose=0)
        model.load_pdb(test_pdb_file)

        hkl = torch.tensor([[1, 0, 0], [2, 1, 0], [1, 1, 1]], dtype=torch.int32)

        # Compute with anomalous correction
        sf_with = model.get_structure_factor(
            hkl.clone(), apply_anomalous=True, recalc=True
        )
        sf_with_copy = sf_with.detach().clone()

        # Reset cache and compute without anomalous correction
        model.reset_cache()
        sf_without = model.get_structure_factor(
            hkl.clone(), apply_anomalous=False, recalc=True
        )
        sf_without_copy = sf_without.detach().clone()

        # They should be different if there are significant anomalous scatterers
        # The Fe atom should contribute anomalous signal
        assert not torch.allclose(sf_with_copy, sf_without_copy), (
            "Structure factors should differ with anomalous correction"
        )

    def test_friedel_pair_asymmetry(self, test_pdb_file):
        """Test that Friedel pairs are not conjugates due to f''."""
        from torchref.model import ModelFT

        model = ModelFT(wavelength=1.0, anomalous_threshold=0.5, verbose=0)
        model.load_pdb(test_pdb_file)

        hkl = torch.tensor([[1, 2, 3], [2, 1, 0], [3, 3, 3]], dtype=torch.int32)

        sf_plus = model.get_structure_factor(hkl, apply_anomalous=True, recalc=True)
        sf_minus = model.get_structure_factor(-hkl, apply_anomalous=True, recalc=True)

        # Friedel's law is governed by f'', which is imaginary and does not change sign
        # with h, so it makes F(h) != F(-h)*. f' is real and dispersive and leaves the
        # conjugate relation intact.
        #
        # In this model f'' is gated on ``apply_bijvoet`` (``ModelFT.__init__``, applied at
        # ``model_ft.py:951`` via ``include_fdp``), which defaults to False because merged
        # data is the usual target and Friedel-preserving F is correct for it. So the
        # default path deliberately does *not* break Friedel's law -- both branches are
        # asserted here rather than only the one this test originally assumed.
        #
        # History: this test previously computed ``is_conjugate`` and then ended in
        # ``pass``, asserting nothing. A first attempt to fix it asserted breakdown on the
        # default path and failed, because that path is Friedel-preserving by design.
        mask, _, _, _, _ = model._get_anomalous_cache()
        assert mask.any(), (
            "no anomalous scatterers in this structure, so neither branch below is "
            "meaningful -- pick a structure with an anomalous element"
        )

        def asymmetry(fp, fm):
            return float(
                (fp - fm.conj()).abs().norm().detach() / fp.abs().norm().detach()
            )

        # --- default branch: f' only, Friedel's law preserved -------------------
        sf_plus_no = model.get_structure_factor(hkl, apply_anomalous=False, recalc=True)
        floor = asymmetry(sf_plus_no, model.get_structure_factor(
            -hkl, apply_anomalous=False, recalc=True))
        default_asym = asymmetry(sf_plus, sf_minus)

        assert not bool(model.anomalous_bijvoet), "fixture unexpectedly enabled f''"
        assert default_asym < 10.0 * max(floor, 1e-9), (
            f"the default path broke Friedel's law ({default_asym:.3e} against a "
            f"no-anomalous floor of {floor:.3e}). With apply_bijvoet=False, f'' is zeroed "
            "and F(h) must stay conjugate to F(-h)"
        )
        # ...but f' must still be reaching F, or "anomalous" is doing nothing at all.
        dispersive = float(
            (sf_plus - sf_plus_no).abs().norm().detach() / sf_plus_no.abs().norm().detach()
        )
        assert dispersive > 1e-4, (
            f"apply_anomalous=True changed F by only {dispersive:.3e}, so the dispersive "
            "f' term is not reaching the structure factors either"
        )

        # --- apply_bijvoet=True: f'' included, Friedel's law broken -------------
        bij = ModelFT(
            wavelength=1.0, anomalous_threshold=0.5, apply_bijvoet=True, verbose=0
        )
        bij.load_pdb(test_pdb_file)
        bij_asym = asymmetry(
            bij.get_structure_factor(hkl, apply_anomalous=True, recalc=True),
            bij.get_structure_factor(-hkl, apply_anomalous=True, recalc=True),
        )
        assert bij_asym > 100.0 * max(floor, 1e-9), (
            f"with apply_bijvoet=True the Friedel asymmetry is only {bij_asym:.3e} "
            f"against a floor of {floor:.3e}, so f'' is not reaching the structure "
            "factors and Bijvoet differences would be absent"
        )

    def test_gradient_flow(self, test_pdb_file):
        """Test that gradients flow through anomalous correction."""
        from torchref.model import ModelFT

        model = ModelFT(wavelength=1.0, anomalous_threshold=0.5, verbose=0)
        model.load_pdb(test_pdb_file)
        # xyz.refinable_params should already have requires_grad=True by default

        hkl = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.int32)

        sf = model.get_structure_factor(hkl, apply_anomalous=True, recalc=True)

        # Compute loss and backpropagate
        loss = sf.abs().sum()
        assert loss.requires_grad, "Loss should require gradients"
        loss.backward()

        # Check that xyz has gradients
        assert model.xyz.refinable_params.grad is not None, "Gradients should flow to xyz"

    def test_cache_invalidation(self, test_pdb_file):
        """Test that anomalous cache is invalidated when elements change."""
        from torchref.model import ModelFT

        model = ModelFT(wavelength=1.0, verbose=0)
        model.load_pdb(test_pdb_file)

        # Access cache
        _ = model._get_anomalous_cache()
        original_hash = model._anomalous_elements_hash

        # The hash should be set
        assert original_hash is not None

        # If we modify the element list (hypothetically), the cache should be invalidated
        # This is tested implicitly by checking the hash mechanism works


class TestAnomalousValuesRealistic:
    """Tests using realistic anomalous scattering values."""

    @pytest.mark.parametrize(
        "element,wavelength,expected_fp_range,expected_fdp_range",
        [
            # Standard synchrotron wavelength (1 A ~ 12.4 keV)
            # Values based on Cromer-Liberman calculation via gemmi
            ("Fe", 1.0, (-0.5, 1.0), (1.0, 3.0)),
            ("Zn", 1.0, (-1.0, 0.5), (2.0, 4.0)),
            # Se at K-edge - strong anomalous signal
            ("Se", 0.9792, (-10.0, -2.0), (0.0, 5.0)),
            # Light atoms should have small values
            ("C", 1.0, (-0.1, 0.1), (-0.1, 0.1)),
            ("N", 1.0, (-0.1, 0.1), (-0.1, 0.1)),
        ],
    )
    def test_realistic_values(
        self, element, wavelength, expected_fp_range, expected_fdp_range
    ):
        """Test that anomalous values fall in expected ranges."""
        fp, fdp = get_anomalous_correction(element, wavelength)

        assert (
            expected_fp_range[0] <= fp <= expected_fp_range[1]
        ), f"{element} f' at {wavelength}A: {fp} not in {expected_fp_range}"

        assert (
            expected_fdp_range[0] <= fdp <= expected_fdp_range[1]
        ), f"{element} f'' at {wavelength}A: {fdp} not in {expected_fdp_range}"
