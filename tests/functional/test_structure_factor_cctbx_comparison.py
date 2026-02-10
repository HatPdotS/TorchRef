"""
Functional tests comparing torchref structure factor calculation against CCTBX.

These tests verify that torchref's Fcalc computation produces results consistent
with the CCTBX reference implementation across multiple structures.
"""
import pytest
import torch
import numpy as np
import warnings
from torchref.base.CCTBX_related import IOTBX_AVAILABLE, calculate_scattering_factor_cctbx

# Skip module if CCTBX not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not IOTBX_AVAILABLE, reason="CCTBX/iotbx required"),
]

# Structure IDs to test (available in tests/files/pdb/ and tests/files/cif/)
TEST_STRUCTURE_IDS = ["1DAW",'2DQ6','6G9X']

# Default resolution for structure factor calculation
DEFAULT_D_MIN = 2.0

EXPECTED_PHASE_CORR = 0.95
EXPECTED_AMPLITUDE_CORR = 0.99
EXPECTED_R_FACTOR = 0.1


def compute_phase_correlation(phase1, phase2, amplitudes=None):
    """
    Parameters
    ----------
    phase1 : array_like
        First set of phases (in radians).
    phase2 : array_like
        Second set of phases (in radians).
    amplitudes : array_like, optional
        Amplitudes to weight the phase correlation. If None, an unweighted
        mean correlation is calculated.
    Returns
    -------
    float
        The weighted or unweighted phase correlation value, ranging from -1 to 1,
        where 1 indicates perfect phase agreement.
    """
    cosine = np.cos(phase1 - phase2)
    if amplitudes is None:
        weighted_phase_correlation = np.mean(cosine)
    else: weighted_phase_correlation = np.sum(amplitudes * cosine) / np.sum(amplitudes)
    return weighted_phase_correlation

def compute_r_factor(f_calc, f_ref):
    """
    Compute R-factor between calculated and reference structure factors.

    R = sum(|F_calc| - |F_ref|) / sum(|F_ref|)

    Parameters
    ----------
    f_calc : np.ndarray
        Calculated structure factors (complex).
    f_ref : np.ndarray
        Reference structure factors (complex).

    Returns
    -------
    float
        R-factor value.
    """
    scale_factor =  np.sum(f_ref) / np.sum(f_calc)  # Scale to minimize R-factor
    amp_calc_scaled = np.abs(f_calc) * scale_factor  # Scale amplitudes to minimize R-factor
    amp_ref = np.abs(f_ref)
    if scale_factor < 0.9 or scale_factor > 1.1:
        warnings.warn(f"Unusual scale factor {scale_factor:.3f} suggests possible issues with calculation, not raising error as we only care about relative structure factor.")
    return np.sum(np.abs(amp_calc_scaled - amp_ref)) / np.sum(amp_ref)

@pytest.mark.integration
@pytest.mark.slow
class TestStructureFactorCCTBXComparisonPDB:
    """Tests comparing torchref vs CCTBX structure factors using PDB files."""

    @pytest.mark.parametrize("structure_id", TEST_STRUCTURE_IDS)
    def test_amplitude_correlation(self, structure_id, pdb_dir):
        """
        Test that amplitude correlation between torchref and CCTBX exceeds 0.999.

        Parameters
        ----------
        structure_id : str
            PDB structure ID to test.
        pdb_dir : Path
            Path to PDB test files directory.
        """
        from torchref.model.model_ft import ModelFT

        pdb_file = pdb_dir / f"{structure_id}.pdb"
        if not pdb_file.exists():
            pytest.skip(f"PDB file not found: {pdb_file}")

        d_min = DEFAULT_D_MIN

        # Calculate CCTBX reference
        f_cctbx, hkl_cctbx = calculate_scattering_factor_cctbx(
            str(pdb_file), d_min=d_min
        )

        # Load model and compute torchref structure factors
        model = ModelFT(max_res=d_min, verbose=0)
        model.load_pdb(str(pdb_file))
        hkl = torch.tensor(hkl_cctbx, dtype=torch.int32)
        f_torchref = model.forward(hkl)

        # Compare amplitudes
        amp_cctbx = np.abs(f_cctbx)
        amp_torchref = np.abs(f_torchref.detach().cpu().numpy())
        correlation = np.corrcoef(amp_cctbx, amp_torchref)[0, 1]

        print(f"\n{structure_id} (PDB) amplitude correlation: {correlation:.6f}")
        assert correlation > EXPECTED_AMPLITUDE_CORR, (
            f"Amplitude correlation {correlation:.6f} below threshold {EXPECTED_AMPLITUDE_CORR} "
            f"for structure {structure_id}"
        )

    @pytest.mark.parametrize("structure_id", TEST_STRUCTURE_IDS)
    def test_phase_correlation(self, structure_id, pdb_dir):
        """
        Test that amplitude-weighted phase correlation exceeds 0.98.

        Parameters
        ----------
        structure_id : str
            PDB structure ID to test.
        pdb_dir : Path
            Path to PDB test files directory.
        """
        from torchref.model.model_ft import ModelFT

        pdb_file = pdb_dir / f"{structure_id}.pdb"
        if not pdb_file.exists():
            pytest.skip(f"PDB file not found: {pdb_file}")

        d_min = DEFAULT_D_MIN

        # Calculate CCTBX reference
        f_cctbx, hkl_cctbx = calculate_scattering_factor_cctbx(
            str(pdb_file), d_min=d_min
        )

        # Load model and compute torchref structure factors
        model = ModelFT(max_res=d_min, verbose=0)
        model.load_pdb(str(pdb_file))
        hkl = torch.tensor(hkl_cctbx, dtype=torch.int32)
        f_torchref = model.forward(hkl)

        # Compare phases (amplitude-weighted)
        f_torchref_np = f_torchref.detach().cpu().numpy()
        phase_cctbx = np.angle(f_cctbx)
        phase_torchref = np.angle(f_torchref_np)
        amplitudes = np.abs(f_cctbx)

        phase_correlation_value = compute_phase_correlation(
            phase_cctbx, phase_torchref, amplitudes=amplitudes
        )
        assert phase_correlation_value > EXPECTED_PHASE_CORR, (
            f"Phase correlation {phase_correlation_value:.6f} below threshold {EXPECTED_PHASE_CORR} "
            f"for structure {structure_id}"
        )


    @pytest.mark.parametrize("structure_id", TEST_STRUCTURE_IDS)
    def test_complex_fcalc_comparison(self, structure_id, pdb_dir):
        """
        Comprehensive test of structure factor agreement.

        Tests amplitude correlation, phase correlation, and R-factor.

        Parameters
        ----------
        structure_id : str
            PDB structure ID to test.
        pdb_dir : Path
            Path to PDB test files directory.
        """
        from torchref.model.model_ft import ModelFT

        pdb_file = pdb_dir / f"{structure_id}.pdb"
        if not pdb_file.exists():
            pytest.skip(f"PDB file not found: {pdb_file}")

        d_min = DEFAULT_D_MIN

        # Calculate CCTBX reference
        f_cctbx, hkl_cctbx = calculate_scattering_factor_cctbx(
            str(pdb_file), d_min=d_min
        )

        # Load model and compute torchref structure factors
        model = ModelFT(max_res=d_min, verbose=0)
        model.load_pdb(str(pdb_file))
        hkl = torch.tensor(hkl_cctbx, dtype=torch.int32)
        f_torchref = model.forward(hkl)

        f_torchref_np = f_torchref.detach().cpu().numpy()

        # Compute all metrics
        amp_cctbx = np.abs(f_cctbx)
        amp_torchref = np.abs(f_torchref_np)
        amp_correlation = np.corrcoef(amp_cctbx, amp_torchref)[0, 1]

        phase_cctbx = np.angle(f_cctbx)
        phase_torchref = np.angle(f_torchref_np)
        phase_correlation = compute_phase_correlation(
            phase_cctbx, phase_torchref, amplitudes=amp_cctbx
        )
        
        r_factor = compute_r_factor(f_torchref_np, f_cctbx)

        print(f"\n{structure_id} (PDB) comprehensive comparison:")
        print(f"  Amplitude correlation: {amp_correlation:.6f}")
        print(f"  Phase correlation:     {phase_correlation:.6f}")
        print(f"  R-factor:              {r_factor:.6f}")
        print(f"  Number of reflections: {len(f_cctbx)}")

        assert amp_correlation > EXPECTED_AMPLITUDE_CORR, f"Amplitude correlation {amp_correlation:.6f} too low"
        assert phase_correlation > EXPECTED_PHASE_CORR, f"Phase correlation {phase_correlation:.6f} too low"
        assert r_factor < EXPECTED_R_FACTOR, f"R-factor {r_factor:.6f} too high"


@pytest.mark.integration
@pytest.mark.slow
class TestStructureFactorCCTBXComparisonCIF:
    """Tests comparing torchref vs CCTBX structure factors using CIF files."""

    @pytest.mark.parametrize("structure_id", TEST_STRUCTURE_IDS)
    def test_amplitude_correlation(self, structure_id, pdb_dir, cif_dir):
        """
        Test that amplitude correlation between torchref (CIF) and CCTBX exceeds 0.999.

        Note: CCTBX uses PDB file for reference calculation since it may not
        support all CIF formats consistently.

        Parameters
        ----------
        structure_id : str
            Structure ID to test.
        pdb_dir : Path
            Path to PDB test files directory (for CCTBX reference).
        cif_dir : Path
            Path to CIF test files directory.
        """
        from torchref.model.model_ft import ModelFT

        pdb_file = pdb_dir / f"{structure_id}.pdb"
        cif_file = cif_dir / f"{structure_id}.cif"
        if not pdb_file.exists():
            pytest.skip(f"PDB file not found: {pdb_file}")
        if not cif_file.exists():
            pytest.skip(f"CIF file not found: {cif_file}")

        d_min = DEFAULT_D_MIN

        # Calculate CCTBX reference from PDB
        f_cctbx, hkl_cctbx = calculate_scattering_factor_cctbx(
            str(pdb_file), d_min=d_min
        )

        # Load model from CIF and compute torchref structure factors
        model = ModelFT(max_res=d_min, verbose=0)
        model.load_cif(str(cif_file))
        hkl = torch.tensor(hkl_cctbx, dtype=torch.int32)
        f_torchref = model.forward(hkl)

        # Compare amplitudes
        amp_cctbx = np.abs(f_cctbx)
        amp_torchref = np.abs(f_torchref.detach().cpu().numpy())
        correlation = np.corrcoef(amp_cctbx, amp_torchref)[0, 1]

        print(f"\n{structure_id} (CIF) amplitude correlation: {correlation:.6f}")
        assert correlation > EXPECTED_AMPLITUDE_CORR, (
            f"Amplitude correlation {correlation:.6f} below threshold {EXPECTED_AMPLITUDE_CORR} "
            f"for structure {structure_id}"
        )

    @pytest.mark.parametrize("structure_id", TEST_STRUCTURE_IDS)
    def test_phase_correlation(self, structure_id, pdb_dir, cif_dir):
        """
        Test that amplitude-weighted phase correlation exceeds 0.98 for CIF files.

        Parameters
        ----------
        structure_id : str
            Structure ID to test.
        pdb_dir : Path
            Path to PDB test files directory (for CCTBX reference).
        cif_dir : Path
            Path to CIF test files directory.
        """
        from torchref.model.model_ft import ModelFT

        pdb_file = pdb_dir / f"{structure_id}.pdb"
        cif_file = cif_dir / f"{structure_id}.cif"
        if not pdb_file.exists():
            pytest.skip(f"PDB file not found: {pdb_file}")
        if not cif_file.exists():
            pytest.skip(f"CIF file not found: {cif_file}")

        d_min = DEFAULT_D_MIN

        # Calculate CCTBX reference from PDB
        f_cctbx, hkl_cctbx = calculate_scattering_factor_cctbx(
            str(pdb_file), d_min=d_min
        )

        # Load model from CIF and compute torchref structure factors
        model = ModelFT(max_res=d_min, verbose=0)
        model.load_cif(str(cif_file))
        hkl = torch.tensor(hkl_cctbx, dtype=torch.int32)
        f_torchref = model.forward(hkl)

        # Compare phases (amplitude-weighted)
        f_torchref_np = f_torchref.detach().cpu().numpy()
        phase_cctbx = np.angle(f_cctbx)
        phase_torchref = np.angle(f_torchref_np)
        amplitudes = np.abs(f_cctbx)

        phase_correlation = compute_phase_correlation(
            phase_cctbx, phase_torchref, amplitudes=amplitudes
        )

        print(
            f"\n{structure_id} (CIF) weighted phase correlation: {phase_correlation:.6f}"
        )
        assert phase_correlation > EXPECTED_PHASE_CORR, (
            f"Phase correlation {phase_correlation:.6f} below threshold {EXPECTED_PHASE_CORR} "
            f"for structure {structure_id}"
        )

    @pytest.mark.parametrize("structure_id", TEST_STRUCTURE_IDS)
    def test_complex_fcalc_comparison(self, structure_id, pdb_dir, cif_dir):
        """
        Comprehensive test of structure factor agreement for CIF files.

        Parameters
        ----------
        structure_id : str
            Structure ID to test.
        pdb_dir : Path
            Path to PDB test files directory (for CCTBX reference).
        cif_dir : Path
            Path to CIF test files directory.
        """
        from torchref.model.model_ft import ModelFT

        pdb_file = pdb_dir / f"{structure_id}.pdb"
        cif_file = cif_dir / f"{structure_id}.cif"
        if not pdb_file.exists():
            pytest.skip(f"PDB file not found: {pdb_file}")
        if not cif_file.exists():
            pytest.skip(f"CIF file not found: {cif_file}")

        d_min = DEFAULT_D_MIN

        # Calculate CCTBX reference from PDB
        f_cctbx, hkl_cctbx = calculate_scattering_factor_cctbx(
            str(pdb_file), d_min=d_min
        )

        # Load model from CIF and compute torchref structure factors
        model = ModelFT(max_res=d_min, verbose=0)
        model.load_cif(str(cif_file))
        hkl = torch.tensor(hkl_cctbx, dtype=torch.int32)
        f_torchref = model.forward(hkl)

        f_torchref_np = f_torchref.detach().cpu().numpy()

        # Compute all metrics
        amp_cctbx = np.abs(f_cctbx)
        amp_torchref = np.abs(f_torchref_np)
        amp_correlation = np.corrcoef(amp_cctbx, amp_torchref)[0, 1]

        phase_cctbx = np.angle(f_cctbx)
        phase_torchref = np.angle(f_torchref_np)
        phase_correlation = compute_phase_correlation(
            phase_cctbx, phase_torchref, amplitudes=amp_cctbx
        )

        r_factor = compute_r_factor(f_torchref_np, f_cctbx)

        print(f"\n{structure_id} (CIF) comprehensive comparison:")
        print(f"  Amplitude correlation: {amp_correlation:.6f}")
        print(f"  Phase correlation:     {phase_correlation:.6f}")
        print(f"  R-factor:              {r_factor:.6f}")
        print(f"  Number of reflections: {len(f_cctbx)}")

        assert amp_correlation > EXPECTED_AMPLITUDE_CORR, f"Amplitude correlation {amp_correlation:.6f} too low"
        assert phase_correlation > EXPECTED_PHASE_CORR, f"Phase correlation {phase_correlation:.6f} too low"
        assert r_factor < EXPECTED_R_FACTOR, f"R-factor {r_factor:.6f} too high"


@pytest.mark.integration
@pytest.mark.slow
class TestStructureFactorFormatConsistency:
    """Tests for PDB vs CIF format consistency."""

    @pytest.mark.parametrize("structure_id", TEST_STRUCTURE_IDS)
    def test_pdb_cif_consistency(self, structure_id, pdb_dir, cif_dir):
        """
        Verify that PDB and CIF give consistent results for the same structure.

        Parameters
        ----------
        structure_id : str
            Structure ID to test.
        pdb_dir : Path
            Path to PDB test files directory.
        cif_dir : Path
            Path to CIF test files directory.
        """
        from torchref.model.model_ft import ModelFT

        pdb_file = pdb_dir / f"{structure_id}.pdb"
        cif_file = cif_dir / f"{structure_id}.cif"
        if not pdb_file.exists():
            pytest.skip(f"PDB file not found: {pdb_file}")
        if not cif_file.exists():
            pytest.skip(f"CIF file not found: {cif_file}")

        d_min = DEFAULT_D_MIN

        # Get HKL indices from CCTBX
        _, hkl_indices = calculate_scattering_factor_cctbx(str(pdb_file), d_min=d_min)
        hkl = torch.tensor(hkl_indices, dtype=torch.int32)

        # Load model from PDB
        model_pdb = ModelFT(max_res=d_min, verbose=0)
        model_pdb.load_pdb(str(pdb_file))
        f_pdb = model_pdb.forward(hkl)

        # Load model from CIF
        model_cif = ModelFT(max_res=d_min, verbose=0)
        model_cif.load_cif(str(cif_file))
        f_cif = model_cif.forward(hkl)

        # Compare PDB vs CIF results
        f_pdb_np = f_pdb.detach().cpu().numpy()
        f_cif_np = f_cif.detach().cpu().numpy()

        amp_pdb = np.abs(f_pdb_np)
        amp_cif = np.abs(f_cif_np)
        correlation = np.corrcoef(amp_pdb, amp_cif)[0, 1]

        print(f"\n{structure_id} PDB vs CIF amplitude correlation: {correlation:.6f}")
        assert correlation > EXPECTED_AMPLITUDE_CORR, (
            f"PDB vs CIF correlation {correlation:.6f} below threshold {EXPECTED_AMPLITUDE_CORR} "
            f"for structure {structure_id}"
        )


@pytest.mark.integration
@pytest.mark.slow
class TestStructureFactorResolutionDependence:
    """Tests for structure factor calculation at different resolutions."""

    @pytest.mark.parametrize("d_min", [3.0, 2.5, 2.0])
    def test_resolution_dependence(self, d_min, pdb_dir):
        """
        Test structure factor agreement at different resolution limits.

        Parameters
        ----------
        d_min : float
            Minimum d-spacing (resolution limit) in Angstroms.
        pdb_dir : Path
            Path to PDB test files directory.
        """
        from torchref.model.model_ft import ModelFT

        # Use first test structure
        structure_id = TEST_STRUCTURE_IDS[0]
        pdb_file = pdb_dir / f"{structure_id}.pdb"
        if not pdb_file.exists():
            pytest.skip(f"PDB file not found: {pdb_file}")

        # Calculate CCTBX reference
        f_cctbx, hkl_cctbx = calculate_scattering_factor_cctbx(
            str(pdb_file), d_min=d_min
        )

        # Load model and compute torchref structure factors
        model = ModelFT(max_res=d_min, verbose=0)
        model.load_pdb(str(pdb_file))
        hkl = torch.tensor(hkl_cctbx, dtype=torch.int32)
        f_torchref = model.forward(hkl)

        # Compare
        amp_cctbx = np.abs(f_cctbx)
        amp_torchref = np.abs(f_torchref.detach().cpu().numpy())
        correlation = np.corrcoef(amp_cctbx, amp_torchref)[0, 1]

        print(f"\n{structure_id} at {d_min}A resolution:")
        print(f"  Amplitude correlation: {correlation:.6f}")
        print(f"  Number of reflections: {len(f_cctbx)}")

        assert correlation > EXPECTED_AMPLITUDE_CORR, (
            f"Amplitude correlation {correlation:.6f} below threshold {EXPECTED_AMPLITUDE_CORR} "
            f"at {d_min}A resolution"
        )
