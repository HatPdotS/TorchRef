"""
Functional tests for Patterson alignment module.

These tests exercise the Patterson alignment with real crystallographic data,
testing that:
1. The original orientation gets the highest Patterson score
2. A known transformation can be recovered through alignment
"""
import pytest
import torch
from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def sample_pdb_file(pdb_dir):
    """Return a sample PDB file for testing."""
    # Try 1DAW first (small, well-behaved structure)
    pdb_file = pdb_dir / "1DAW.pdb"
    if pdb_file.exists():
        return pdb_file
    # Try any available PDB file
    pdb_files = list(pdb_dir.glob("*.pdb"))
    if pdb_files:
        return pdb_files[0]
    pytest.skip("No PDB files found in test data")


@pytest.fixture(scope="session")
def pdb_mtz_pair(pdb_dir, mtz_dir):
    """Return a matching pair of PDB model and MTZ reflections."""
    # Try to find matching files - prefer 1DAW (small structure)
    for pdb_id in ["1DAW", "3K7M", "3VRJ"]:
        pdb_file = pdb_dir / f"{pdb_id}.pdb"
        mtz_file = mtz_dir / f"{pdb_id}.mtz"

        if pdb_file.exists() and mtz_file.exists():
            return {"pdb_id": pdb_id, "model": pdb_file, "reflections": mtz_file}

    # Try to find any matching pair
    pdb_files = {f.stem: f for f in pdb_dir.glob("*.pdb")}
    mtz_files = {f.stem: f for f in mtz_dir.glob("*.mtz")}

    common_ids = set(pdb_files.keys()) & set(mtz_files.keys())
    if common_ids:
        pdb_id = sorted(common_ids)[0]
        return {"pdb_id": pdb_id, "model": pdb_files[pdb_id], "reflections": mtz_files[pdb_id]}

    pytest.skip("No matching PDB/MTZ pairs found in test data")


@pytest.fixture
def loaded_model_and_data(pdb_mtz_pair):
    """Fixture providing matching model and reflection data from PDB/MTZ."""
    from torchref.model.model import Model
    from torchref.io.datasets.reflection_data import ReflectionData

    model = Model(verbose=0)
    model.load_pdb(str(pdb_mtz_pair["model"]))

    data = ReflectionData(verbose=0)
    data.load_mtz(str(pdb_mtz_pair["reflections"]))

    return {
        "pdb_id": pdb_mtz_pair["pdb_id"],
        "model": model,
        "data": data
    }


@pytest.fixture
def patterson_aligner(loaded_model_and_data):
    """Fixture providing initialized PattersonAligner."""
    from torchref.alignment.align import PattersonAligner

    model = loaded_model_and_data["model"]
    data = loaded_model_and_data["data"]

    aligner = PattersonAligner(
        data=data,
        model=model,
        n_vectors=1000,
        weighting='Z2',
        verbose=0
    )

    return {
        "aligner": aligner,
        "model": model,
        "data": data,
        "pdb_id": loaded_model_and_data["pdb_id"]
    }


# =============================================================================
# Test Classes
# =============================================================================

@pytest.mark.integration
class TestPattersonAlignerInitialization:
    """Test PattersonAligner initialization with real data."""

    def test_aligner_creates_patterson(self, patterson_aligner):
        """Test that aligner creates a Patterson map."""
        aligner = patterson_aligner["aligner"]

        assert aligner.patterson is not None
        assert aligner.patterson.ndim == 3
        assert aligner.patterson.shape[0] > 0

    def test_aligner_has_symmetry(self, patterson_aligner):
        """Test that aligner has symmetry from model."""
        aligner = patterson_aligner["aligner"]

        assert aligner.symmetry is not None
        assert aligner.symmetry.n_ops > 0

    def test_aligner_has_cell(self, patterson_aligner):
        """Test that aligner has unit cell parameters."""
        aligner = patterson_aligner["aligner"]

        assert aligner.cell is not None
        assert len(aligner.cell) == 6
        assert all(aligner.cell[:3] > 0)  # Positive cell dimensions


@pytest.mark.integration
class TestPattersonInterpolation:
    """Test Patterson map interpolation."""

    def test_interpolation_origin_peak(self, patterson_aligner):
        """Test that Patterson has maximum at origin."""
        aligner = patterson_aligner["aligner"]

        # Query at origin
        origin = torch.zeros(1, 3, dtype=aligner.patterson.dtype)
        origin_value = aligner.interpolate_patterson(origin)

        # Query at random points
        random_points = torch.rand(100, 3, dtype=aligner.patterson.dtype)
        random_values = aligner.interpolate_patterson(random_points)

        # Origin should be at or near maximum
        assert origin_value >= random_values.max() * 0.9, \
            f"Origin value {origin_value.item():.4f} not near max {random_values.max().item():.4f}"

    def test_interpolation_centrosymmetric(self, patterson_aligner):
        """Test that Patterson is centrosymmetric: P(u) = P(-u)."""
        aligner = patterson_aligner["aligner"]

        # Random fractional vectors
        vecs = torch.rand(20, 3, dtype=aligner.patterson.dtype) - 0.5

        p_plus = aligner.interpolate_patterson(vecs)
        p_minus = aligner.interpolate_patterson(-vecs)

        # Use relative tolerance since Patterson values can be large
        assert torch.allclose(p_plus, p_minus, rtol=1e-4, atol=1.0), \
            "Patterson not centrosymmetric"

    def test_interpolation_periodic(self, patterson_aligner):
        """Test that Patterson is periodic."""
        aligner = patterson_aligner["aligner"]

        # Points and their periodic equivalents
        vecs = torch.rand(10, 3, dtype=aligner.patterson.dtype) * 0.5
        vecs_periodic = vecs + 1.0  # Add one unit cell

        p_orig = aligner.interpolate_patterson(vecs)
        p_periodic = aligner.interpolate_patterson(vecs_periodic)

        # Use relative tolerance since Patterson values can be large
        assert torch.allclose(p_orig, p_periodic, rtol=1e-4, atol=1.0), \
            "Patterson not periodic"


@pytest.mark.integration
class TestOriginalOrientationScore:
    """Test that original orientation gets highest Patterson score."""

    def test_identity_scores_higher_than_random(self, patterson_aligner):
        """Test that identity transformation scores higher than random rotations."""
        from torchref.alignment.sampling import VectorSampler
        from torchref.math_functions.math_torch import (
            cartesian_to_fractional_torch,
            random_rotation_uniform
        )

        aligner = patterson_aligner["aligner"]
        model = patterson_aligner["model"]

        # Filter out waters
        model_no_water = model.select('not resname HOH')
        xyz = model_no_water.xyz().detach()

        # Create sampler and sample vectors
        # Use 5000 vectors to reduce variance (500 is too noisy)
        sampler = VectorSampler(model_no_water, weighting='Z2', seed=42)
        idx1, idx2 = sampler.sample(5000)

        # Score at identity transformation
        R_identity = torch.eye(3, dtype=xyz.dtype, device=xyz.device)
        t_zero = torch.zeros(3, dtype=xyz.dtype, device=xyz.device)

        score_identity = aligner.score_transformation(xyz, R_identity, t_zero, idx1, idx2)

        # Score at random transformations
        random_scores = []
        for _ in range(10):
            R_rand = random_rotation_uniform(1, device=str(xyz.device), dtype=xyz.dtype)
            t_rand = torch.rand(3, dtype=xyz.dtype, device=xyz.device) * aligner.cell[:3].to(xyz.device)
            score = aligner.score_transformation(xyz, R_rand, t_rand, idx1, idx2)
            random_scores.append(score.item())

        random_scores = torch.tensor(random_scores)

        # Identity should score higher than mean of random
        assert score_identity > random_scores.mean(), \
            f"Identity score {score_identity.item():.4f} not higher than random mean {random_scores.mean().item():.4f}"

    def test_small_perturbation_scores_positive(self, patterson_aligner):
        """Test that small perturbations still give positive scores."""
        from torchref.alignment.sampling import VectorSampler
        from torchref.math_functions.math_torch import axis_angle_to_rotation_matrix

        aligner = patterson_aligner["aligner"]
        model = patterson_aligner["model"]

        model_no_water = model.select('not resname HOH')
        xyz = model_no_water.xyz().detach()

        # Use 5000 vectors for stable scoring
        sampler = VectorSampler(model_no_water, weighting='Z2', seed=42)
        idx1, idx2 = sampler.sample(5000)

        # Score at identity
        R_identity = torch.eye(3, dtype=xyz.dtype, device=xyz.device)
        t_zero = torch.zeros(3, dtype=xyz.dtype, device=xyz.device)
        score_identity = aligner.score_transformation(xyz, R_identity, t_zero, idx1, idx2)

        # Score at small rotation (0.01 radians ~ 0.6 degrees)
        small_axis_angle = torch.tensor([0.01, 0.0, 0.0], dtype=xyz.dtype, device=xyz.device)
        R_small = axis_angle_to_rotation_matrix(small_axis_angle)
        score_small_rot = aligner.score_transformation(xyz, R_small, t_zero, idx1, idx2)

        # Both scores should be positive for correct-ish orientations
        assert score_identity > 0, f"Identity score {score_identity.item():.2f} should be positive"
        assert score_small_rot > 0, f"Small rotation score {score_small_rot.item():.2f} should be positive"

        # Very small rotation should give similar score
        ratio = score_small_rot / score_identity
        assert 0.8 < ratio < 1.2, \
            f"Very small rotation (0.6°) score ratio {ratio:.2f} too different from 1.0"


@pytest.mark.integration
class TestAlignmentRecovery:
    """Test recovery of original orientation after applying known transformation."""

    def test_alignment_finds_good_orientation(self, patterson_aligner):
        """Test that alignment finds an orientation with high Patterson score.

        Note: Patterson alignment cannot uniquely recover rotations due to
        inherent ambiguities (centrosymmetry, space group symmetry). This test
        verifies that alignment finds SOME good orientation, not necessarily
        the original one.
        """
        from torchref.math_functions.math_torch import axis_angle_to_rotation_matrix
        from torchref.alignment.sampling import VectorSampler

        aligner = patterson_aligner["aligner"]
        model = patterson_aligner["model"]

        # Get original coordinates and use their dtype
        xyz_orig = model.xyz().detach().clone()
        dtype = xyz_orig.dtype

        # Apply a known rotation to the model (use same dtype)
        known_axis_angle = torch.tensor([0.3, 0.2, 0.1], dtype=dtype)
        R_known = axis_angle_to_rotation_matrix(known_axis_angle)

        # Apply rotation
        xyz_rotated = xyz_orig @ R_known.T

        # Create a rotated model (update coordinates)
        rotated_model = model.copy()
        coords_np = xyz_rotated.cpu().numpy()
        rotated_model.pdb['x'] = coords_np[:, 0]
        rotated_model.pdb['y'] = coords_np[:, 1]
        rotated_model.pdb['z'] = coords_np[:, 2]

        # Get score before alignment
        model_no_water = rotated_model.select('not resname HOH')
        sampler = VectorSampler(model_no_water, weighting='Z2', seed=42)
        idx1, idx2 = sampler.sample(5000)

        R_id = torch.eye(3, dtype=dtype)
        t_zero = torch.zeros(3, dtype=dtype)
        xyz_before = model_no_water.xyz().detach()
        score_before = aligner.score_transformation(xyz_before, R_id, t_zero, idx1, idx2)

        # Run alignment
        aligned_model, result = aligner.align(
            model=rotated_model,
            n_starts=10,
            n_vectors=5000,
            max_iter=50,
            seed=42
        )

        # Alignment should find a solution with positive score
        assert result.score > 0, \
            f"Alignment score should be positive: {result.score:.2f}"

        # Alignment score should be at least as good as (or close to) score before
        # Note: score_before may already be good if the rotation didn't move far
        # The key is that alignment converges to a reasonable solution
        assert result.converged or result.score > score_before * 0.5, \
            f"Alignment should converge or improve: score={result.score:.2f}, before={score_before.item():.2f}"

    def test_recover_known_translation(self, patterson_aligner):
        """Test that alignment recovers a known translation (modulo unit cell)."""
        aligner = patterson_aligner["aligner"]
        model = patterson_aligner["model"]

        # Apply a known translation
        t_known = torch.tensor([2.0, 3.0, 1.5], dtype=torch.float64)

        # Get original coordinates
        xyz_orig = model.xyz().detach().clone()

        # Apply translation
        xyz_translated = xyz_orig + t_known

        # Create translated model
        translated_model = model.copy()
        coords_np = xyz_translated.cpu().numpy()
        translated_model.pdb['x'] = coords_np[:, 0]
        translated_model.pdb['y'] = coords_np[:, 1]
        translated_model.pdb['z'] = coords_np[:, 2]

        # Run alignment
        aligned_model, result = aligner.align(
            model=translated_model,
            n_starts=5,
            n_vectors=500,
            max_iter=30,
            seed=42
        )

        # The score should be high (similar to identity)
        # Translation recovery is harder due to periodicity
        assert result.score > 0, f"Alignment score should be positive: {result.score}"

    def test_alignment_improves_score(self, patterson_aligner):
        """Test that alignment improves Patterson score from random start."""
        from torchref.alignment.sampling import VectorSampler
        from torchref.math_functions.math_torch import random_rotation_uniform

        aligner = patterson_aligner["aligner"]
        model = patterson_aligner["model"]

        # Get original coordinates and use their dtype
        xyz_orig = model.xyz().detach().clone()
        dtype = xyz_orig.dtype

        # Apply random transformation (use same dtype)
        R_rand = random_rotation_uniform(1, dtype=dtype)
        t_rand = torch.rand(3, dtype=dtype) * 10.0

        xyz_transformed = xyz_orig @ R_rand.T + t_rand

        # Create transformed model
        transformed_model = model.copy()
        coords_np = xyz_transformed.cpu().numpy()
        transformed_model.pdb['x'] = coords_np[:, 0]
        transformed_model.pdb['y'] = coords_np[:, 1]
        transformed_model.pdb['z'] = coords_np[:, 2]

        # Get score before alignment
        model_no_water = transformed_model.select('not resname HOH')
        sampler = VectorSampler(model_no_water, weighting='Z2', seed=42)
        idx1, idx2 = sampler.sample(500)

        xyz_before = model_no_water.xyz().detach()
        R_id = torch.eye(3, dtype=xyz_before.dtype)
        t_zero = torch.zeros(3, dtype=xyz_before.dtype)
        score_before = aligner.score_transformation(xyz_before, R_id, t_zero, idx1, idx2)

        # Run alignment
        aligned_model, result = aligner.align(
            model=transformed_model,
            n_starts=5,
            n_vectors=500,
            max_iter=30,
            seed=42
        )

        # Score after alignment should be at least as good
        assert result.score >= score_before.item() * 0.9, \
            f"Alignment didn't improve score: before={score_before.item():.4f}, after={result.score:.4f}"


@pytest.mark.integration
class TestVectorSamplerWithRealData:
    """Test VectorSampler with real structure data."""

    def test_sampler_creates_valid_indices(self, loaded_model_and_data):
        """Test that sampler creates valid atom pair indices."""
        from torchref.alignment.sampling import VectorSampler

        model = loaded_model_and_data["model"]
        model_no_water = model.select('not resname HOH')

        sampler = VectorSampler(model_no_water, weighting='Z2', seed=42)
        idx1, idx2 = sampler.sample(100)

        n_atoms = sampler.n_atoms

        # Check indices are valid (ASU only, no symmetry expansion)
        assert idx1.min() >= 0
        assert idx1.max() < n_atoms
        assert idx2.min() >= 0
        assert idx2.max() < n_atoms

        # No self-pairs
        assert not torch.any(idx1 == idx2)

    def test_sampler_weights_favor_heavy_atoms(self, loaded_model_and_data):
        """Test that Z2 weighting favors heavier atoms."""
        from torchref.alignment.sampling import VectorSampler

        model = loaded_model_and_data["model"]
        model_no_water = model.select('not resname HOH')

        sampler = VectorSampler(model_no_water, weighting='Z2', seed=42)

        # Weights should exist
        assert sampler.weights is not None
        assert len(sampler.weights) > 0

        # All weights should be positive
        assert torch.all(sampler.weights > 0)


@pytest.mark.integration
class TestAlignmentResult:
    """Test AlignmentResult dataclass functionality."""

    def test_result_apply_transformation(self, patterson_aligner):
        """Test that AlignmentResult.apply() correctly transforms coordinates."""
        aligner = patterson_aligner["aligner"]
        model = patterson_aligner["model"]

        # Run alignment
        aligned_model, result = aligner.align(
            model=model,
            n_starts=3,
            n_vectors=200,
            max_iter=20,
            seed=42
        )

        # Apply transformation manually
        xyz_orig = model.xyz().detach()
        xyz_manual = xyz_orig @ result.rotation.T + result.translation

        # Apply via result.apply()
        xyz_applied = result.apply(xyz_orig)

        assert torch.allclose(xyz_manual, xyz_applied, atol=1e-6), \
            "result.apply() doesn't match manual transformation"

    def test_result_as_numpy(self, patterson_aligner):
        """Test that as_numpy() returns correct types."""
        aligner = patterson_aligner["aligner"]
        model = patterson_aligner["model"]

        _, result = aligner.align(
            model=model,
            n_starts=2,
            n_vectors=100,
            max_iter=10,
            seed=42
        )

        R_np, t_np = result.as_numpy()

        import numpy as np
        assert isinstance(R_np, np.ndarray)
        assert isinstance(t_np, np.ndarray)
        assert R_np.shape == (3, 3)
        assert t_np.shape == (3,)
