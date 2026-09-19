"""Generate small synthetic numerical inputs for unit tests.

Imported by the unit conftest only. Factories return fresh CPU tensors on every
call, using TorchRef's numeric dtypes. They reset the global NumPy random seed;
``random_seed`` also resets PyTorch's seed. Accelerator coverage requires an
explicit move by the caller under this allocation policy.
"""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from torchref.config import dtypes


@pytest.fixture
def random_seed() -> int:
    """Set random seed for reproducibility."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


@pytest.fixture
def random_coordinates() -> Callable[..., torch.Tensor]:
    """Return a factory for Cartesian coordinates of shape (n_atoms, 3) in Å."""

    def _generate(n_atoms: int = 10, seed: int = 42) -> torch.Tensor:
        np.random.seed(seed)
        return torch.tensor(np.random.rand(n_atoms, 3) * 10, dtype=dtypes.float)

    return _generate


@pytest.fixture
def random_fractional_coordinates() -> Callable[..., torch.Tensor]:
    """Return a factory for fractional coordinates (n_atoms, 3) in [0, 1)."""

    def _generate(n_atoms: int = 10, seed: int = 42) -> torch.Tensor:
        np.random.seed(seed)
        return torch.tensor(np.random.rand(n_atoms, 3), dtype=dtypes.float)

    return _generate


@pytest.fixture
def random_adp() -> Callable[..., torch.Tensor]:
    """Return a factory for isotropic B-factors (n_atoms,) in [10, 60) Å²."""

    def _generate(n_atoms: int = 10, seed: int = 42) -> torch.Tensor:
        np.random.seed(seed)
        return torch.tensor(np.random.rand(n_atoms) * 50 + 10, dtype=dtypes.float)

    return _generate


@pytest.fixture
def random_occupancies() -> Callable[..., torch.Tensor]:
    """Return a factory for dimensionless occupancies (n_atoms,) in [0.5, 1)."""

    def _generate(n_atoms: int = 10, seed: int = 42) -> torch.Tensor:
        np.random.seed(seed)
        return torch.tensor(np.random.rand(n_atoms) * 0.5 + 0.5, dtype=dtypes.float)

    return _generate


@pytest.fixture
def mock_cell() -> torch.Tensor:
    """Return an orthorhombic cell (6,), lengths in Å and angles in degrees."""
    return torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0], dtype=dtypes.float)


@pytest.fixture
def mock_cell_triclinic() -> torch.Tensor:
    """Return a triclinic cell (6,), lengths in Å and angles in degrees."""
    return torch.tensor([40.0, 50.0, 60.0, 70.0, 80.0, 85.0], dtype=dtypes.float)


@pytest.fixture
def mock_hkl_indices() -> Callable[..., torch.Tensor]:
    """Return a factory for floating HKL triples (n_kept, 3), excluding the origin.

    The output uses ``dtypes.float``; ``n_kept`` can be less than the requested
    reflection count when the origin is sampled.
    """

    def _generate(
        n_reflections: int = 100, max_index: int = 10, seed: int = 42
    ) -> torch.Tensor:
        np.random.seed(seed)
        h = np.random.randint(-max_index, max_index + 1, n_reflections)
        k = np.random.randint(-max_index, max_index + 1, n_reflections)
        l = np.random.randint(-max_index, max_index + 1, n_reflections)
        # Exclude (0,0,0)
        mask = ~((h == 0) & (k == 0) & (l == 0))
        h, k, l = h[mask], k[mask], l[mask]
        return torch.tensor(np.stack([h, k, l], axis=1), dtype=dtypes.float)

    return _generate


@pytest.fixture
def mock_structure_factors() -> Callable[..., torch.Tensor]:
    """Return a factory for complex structure factors (n_reflections,) in electrons."""

    def _generate(n_reflections: int = 100, seed: int = 42) -> torch.Tensor:
        np.random.seed(seed)
        real = np.random.randn(n_reflections) * 100
        imag = np.random.randn(n_reflections) * 100
        return torch.tensor(real + 1j * imag, dtype=dtypes.complex)

    return _generate


@pytest.fixture
def mock_F_obs() -> Callable[..., torch.Tensor]:
    """Return a factory for observed amplitudes (n_reflections,) in electrons."""

    def _generate(n_reflections: int = 100, seed: int = 42) -> torch.Tensor:
        np.random.seed(seed)
        # Positive values with realistic distribution
        return torch.tensor(
            np.abs(np.random.randn(n_reflections) * 100) + 10, dtype=dtypes.float
        )

    return _generate


@pytest.fixture
def mock_F_sigma() -> Callable[..., torch.Tensor]:
    """Return a factory for amplitude uncertainties (n_reflections,) in electrons."""

    def _generate(n_reflections: int = 100, seed: int = 42) -> torch.Tensor:
        np.random.seed(seed)
        return torch.tensor(
            np.abs(np.random.randn(n_reflections) * 5) + 1, dtype=dtypes.float
        )

    return _generate


@pytest.fixture
def mock_aniso_u() -> Callable[..., torch.Tensor]:
    """Return a factory for Cartesian U tensors (n_atoms, 6) in Å².

    Components are ordered U11, U22, U33, U12, U13, U23.
    """

    def _generate(n_atoms: int = 10, seed: int = 42) -> torch.Tensor:
        np.random.seed(seed)
        # Diagonal elements (positive)
        u11 = np.random.rand(n_atoms) * 0.05 + 0.02
        u22 = np.random.rand(n_atoms) * 0.05 + 0.02
        u33 = np.random.rand(n_atoms) * 0.05 + 0.02
        # Off-diagonal elements (can be negative, smaller magnitude)
        u12 = (np.random.rand(n_atoms) - 0.5) * 0.02
        u13 = (np.random.rand(n_atoms) - 0.5) * 0.02
        u23 = (np.random.rand(n_atoms) - 0.5) * 0.02
        return torch.tensor(
            np.stack([u11, u22, u33, u12, u13, u23], axis=1), dtype=dtypes.float
        )

    return _generate


@pytest.fixture
def mock_scattering_factors() -> Callable[..., torch.Tensor]:
    """Return a factory for scattering factors (n_reflections, n_atoms) in electrons."""

    def _generate(
        n_reflections: int = 100, n_atoms: int = 10, seed: int = 42
    ) -> torch.Tensor:
        np.random.seed(seed)
        # Decreasing with resolution (approximate)
        return torch.tensor(
            np.random.rand(n_reflections, n_atoms) * 5 + 1, dtype=dtypes.float
        )

    return _generate


@pytest.fixture
def mock_weights() -> Callable[..., torch.Tensor]:
    """Return a factory for dimensionless weights (n_atoms, 1) summing to one."""

    def _generate(n_atoms: int = 10, seed: int = 42) -> torch.Tensor:
        np.random.seed(seed)
        weights = np.random.rand(n_atoms)
        return torch.tensor(weights / weights.sum(), dtype=dtypes.float).reshape(-1, 1)

    return _generate
