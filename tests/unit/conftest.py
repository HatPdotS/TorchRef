"""
Unit test specific fixtures.
Unit tests should NOT use real file I/O - use mocks or minimal in-memory data.
"""
import pytest
import torch
import numpy as np


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


@pytest.fixture
def random_coordinates():
    """Generate random atomic coordinates."""
    def _generate(n_atoms: int = 10, seed: int = 42):
        np.random.seed(seed)
        return torch.tensor(np.random.rand(n_atoms, 3) * 10, dtype=torch.float64)
    return _generate


@pytest.fixture
def random_fractional_coordinates():
    """Generate random fractional coordinates (0-1 range)."""
    def _generate(n_atoms: int = 10, seed: int = 42):
        np.random.seed(seed)
        return torch.tensor(np.random.rand(n_atoms, 3), dtype=torch.float64)
    return _generate


@pytest.fixture
def random_adp():
    """Generate random ADPs (atomic displacement parameters, reasonable range 10-60 Å²)."""
    def _generate(n_atoms: int = 10, seed: int = 42):
        np.random.seed(seed)
        return torch.tensor(np.random.rand(n_atoms) * 50 + 10, dtype=torch.float64)
    return _generate


@pytest.fixture
def random_occupancies():
    """Generate random occupancies (0-1 range)."""
    def _generate(n_atoms: int = 10, seed: int = 42):
        np.random.seed(seed)
        return torch.tensor(np.random.rand(n_atoms) * 0.5 + 0.5, dtype=torch.float64)
    return _generate


@pytest.fixture
def mock_cell():
    """Mock cell parameters [a, b, c, alpha, beta, gamma]."""
    return torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0], dtype=torch.float64)


@pytest.fixture
def mock_cell_triclinic():
    """Mock triclinic cell parameters."""
    return torch.tensor([40.0, 50.0, 60.0, 70.0, 80.0, 85.0], dtype=torch.float64)


@pytest.fixture
def mock_hkl_indices():
    """Generate mock HKL indices."""
    def _generate(n_reflections: int = 100, max_index: int = 10, seed: int = 42):
        np.random.seed(seed)
        h = np.random.randint(-max_index, max_index + 1, n_reflections)
        k = np.random.randint(-max_index, max_index + 1, n_reflections)
        l = np.random.randint(-max_index, max_index + 1, n_reflections)
        # Exclude (0,0,0)
        mask = ~((h == 0) & (k == 0) & (l == 0))
        h, k, l = h[mask], k[mask], l[mask]
        return torch.tensor(np.stack([h, k, l], axis=1), dtype=torch.float64)
    return _generate


@pytest.fixture
def mock_structure_factors():
    """Generate mock structure factors (complex)."""
    def _generate(n_reflections: int = 100, seed: int = 42):
        np.random.seed(seed)
        real = np.random.randn(n_reflections) * 100
        imag = np.random.randn(n_reflections) * 100
        return torch.tensor(real + 1j * imag, dtype=torch.complex64)
    return _generate


@pytest.fixture
def mock_F_obs():
    """Generate mock observed structure factor amplitudes."""
    def _generate(n_reflections: int = 100, seed: int = 42):
        np.random.seed(seed)
        # Positive values with realistic distribution
        return torch.tensor(np.abs(np.random.randn(n_reflections) * 100) + 10, dtype=torch.float64)
    return _generate


@pytest.fixture
def mock_F_sigma():
    """Generate mock sigma values for F_obs."""
    def _generate(n_reflections: int = 100, seed: int = 42):
        np.random.seed(seed)
        return torch.tensor(np.abs(np.random.randn(n_reflections) * 5) + 1, dtype=torch.float64)
    return _generate


@pytest.fixture
def mock_aniso_u():
    """Generate mock anisotropic U tensor components [U11, U22, U33, U12, U13, U23]."""
    def _generate(n_atoms: int = 10, seed: int = 42):
        np.random.seed(seed)
        # Diagonal elements (positive)
        u11 = np.random.rand(n_atoms) * 0.05 + 0.02
        u22 = np.random.rand(n_atoms) * 0.05 + 0.02
        u33 = np.random.rand(n_atoms) * 0.05 + 0.02
        # Off-diagonal elements (can be negative, smaller magnitude)
        u12 = (np.random.rand(n_atoms) - 0.5) * 0.02
        u13 = (np.random.rand(n_atoms) - 0.5) * 0.02
        u23 = (np.random.rand(n_atoms) - 0.5) * 0.02
        return torch.tensor(np.stack([u11, u22, u33, u12, u13, u23], axis=1), dtype=torch.float32)
    return _generate


@pytest.fixture
def mock_scattering_factors():
    """Generate mock scattering factors."""
    def _generate(n_reflections: int = 100, n_atoms: int = 10, seed: int = 42):
        np.random.seed(seed)
        # Decreasing with resolution (approximate)
        return torch.tensor(np.random.rand(n_reflections, n_atoms) * 5 + 1, dtype=torch.float64)
    return _generate


@pytest.fixture
def mock_weights():
    """Generate mock weights for atoms."""
    def _generate(n_atoms: int = 10, seed: int = 42):
        np.random.seed(seed)
        weights = np.random.rand(n_atoms)
        return torch.tensor(weights / weights.sum(), dtype=torch.float32).reshape(-1, 1)
    return _generate
