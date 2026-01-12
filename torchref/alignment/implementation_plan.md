# Implementation Plan: Patterson-Based Alignment Module

## Objective

Create a TorchRef-integrated module for aligning predicted structures to observed diffraction data using Patterson map vector matching. The module takes `Model` and `ReflectionData` objects as primary inputs and leverages the existing TorchRef infrastructure.

---

## Summary: Math Function Reuse and Dependencies

### Existing Functions from `torchref/math_functions/math_torch.py`

The following functions are **directly reused** from the existing codebase:

| Function | Purpose in Alignment | Location |
|----------|---------------------|----------|
| `place_on_grid(hkls, structure_factor, grid_size)` | Place \|F\|² on reciprocal grid for Patterson calculation | math_torch.py |
| `find_grid_size(cell, max_res)` | Determine Patterson grid dimensions | math_torch.py |
| `cartesian_to_fractional_torch(coords, cell)` | Convert Cartesian vectors to fractional for Patterson lookup | math_torch.py |
| `fractional_to_cartesian_torch(coords, cell)` | Convert fractional to Cartesian after symmetry | math_torch.py |
| `axis_angle_to_rotation_matrix(axis_angle)` | Convert axis-angle to rotation matrix (Rodrigues' formula) | math_torch.py:2206 |
| `rotation_matrix_to_axis_angle(R)` | Convert rotation matrix to axis-angle | math_torch.py:2258 |
| `quaternion_to_rotation_matrix(q)` | Convert quaternion to rotation matrix | math_torch.py:2306 |
| `random_rotation_uniform(n, device, dtype)` | Generate uniform random SO(3) rotations | math_torch.py:2349 |
| `trilinear_interpolate(grid, points, mode)` | Trilinear interpolation with periodic boundaries | math_torch.py:2392 |

### Patterson Calculation (Moved to ReflectionData)

Patterson map calculation is now integrated into the `ReflectionData` class:

```python
# Calculate Patterson map from reflection data
patterson = data.calc_patterson(grid_size=None)  # Returns torch.Tensor (nx, ny, nz)
```

Implementation details (in `ReflectionData.calc_patterson()`):
1. Expands data to P1 symmetry
2. Uses `place_on_grid()` to place F² on reciprocal grid
3. Computes inverse FFT via `torch.fft.ifftn()`
4. Returns real-valued Patterson map

### Scipy Replacement Strategy

| Scipy Function | Replacement | Benefit |
|---------------|-------------|---------|
| `scipy.ndimage.map_coordinates` | `trilinear_interpolate()` | GPU acceleration, gradients |
| `scipy.optimize.minimize(method='L-BFGS-B')` | `torch.optim.LBFGS` | Differentiable, GPU support |
| `scipy.spatial.transform.Rotation` | `axis_angle_to_rotation_matrix()` | Batched, differentiable |

### Key Design Decisions

1. **Pure Torch Implementation**: All core computations use PyTorch for:
   - GPU acceleration
   - Automatic differentiation (enables end-to-end training)
   - Consistent API with rest of TorchRef

2. **Numpy Wrappers**: Thin numpy wrappers provided for external API compatibility in `rotation.py`

3. **Reuse Existing Infrastructure**:
   - Patterson grid: Uses `ReflectionData.calc_patterson()` (internally uses `place_on_grid` + FFT)
   - Coordinate transforms: Uses existing `cartesian_to_fractional_torch`, `fractional_to_cartesian_torch`
   - Symmetry: Uses existing `Symmetry` class from `torchref/symmetrie/symmetrie.py` via `model.symmetry`
   - Interpolation: Uses `torch.nn.functional.grid_sample()` for trilinear Patterson lookup

4. **No External Dependencies Beyond torch/numpy/gemmi**

---

## Design Philosophy

1. **TorchRef-native**: Primary interface accepts `Model` and `ReflectionData` objects
2. **Consistent API**: Follow TorchRef patterns (device handling, dtype, verbose flags)
3. **Reuse infrastructure**: Use existing `Symmetry`, coordinate transforms, and math utilities
4. **In-place or copy**: Support both modification patterns like `Model.copy()`

---

## File Structure

```
torchref/alignment/
├── __init__.py
├── sampling.py
├── rotation.py
└── align.py
```

**Note**: `patterson.py` and `symmetry.py` are not needed as separate files:
- Patterson calculation is handled by `ReflectionData.calc_patterson()`
- Symmetry operations use the existing `torchref.symmetrie.symmetrie.Symmetry` class

Plus tests:
```
tests/alignment/
├── test_sampling.py
├── test_rotation.py
└── test_align.py
```

---

## File 1: `rotation.py`

Rotation parameterization utilities. Provides both numpy (for compatibility) and torch (for optimization) versions.

**Note**: The primary torch implementations are in `math_torch.py`. This module provides numpy wrappers and torch convenience functions.

### Functions

```python
"""
Rotation parameterization utilities for Patterson alignment.

Provides numpy wrappers around torch implementations for external API
compatibility. Primary torch implementations are in math_torch.py.
"""

import numpy as np
import torch
from torchref.math_functions.math_torch import (
    axis_angle_to_rotation_matrix as axis_angle_to_matrix_torch,
    rotation_matrix_to_axis_angle as matrix_to_axis_angle_torch,
    random_rotation_uniform,
    quaternion_to_rotation_matrix
)


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert axis-angle representation to 3x3 rotation matrix (numpy API)."""

def matrix_to_axis_angle(R: np.ndarray) -> tuple:
    """Convert 3x3 rotation matrix to axis-angle representation (numpy API)."""

def params_to_matrix(params: np.ndarray) -> np.ndarray:
    """Convert 3 optimization parameters (scaled axis-angle) to rotation matrix."""

def matrix_to_params(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to 3 optimization parameters."""

def random_rotation_params(rng: np.random.Generator = None) -> np.ndarray:
    """Generate random rotation parameters (uniform over SO(3))."""

def random_rotation_matrix(rng: np.random.Generator = None) -> np.ndarray:
    """Generate a random rotation matrix (uniform over SO(3))."""


# Torch versions for direct use in optimization
def params_to_matrix_torch(params: torch.Tensor) -> torch.Tensor:
    """
    Convert 3 optimization parameters to rotation matrix (torch version).

    Parameters
    ----------
    params : torch.Tensor
        Tensor (3,) or (N, 3) representing scaled axis-angle.

    Returns
    -------
    torch.Tensor
        Rotation matrix (3, 3) or (N, 3, 3).
    """
    return axis_angle_to_matrix_torch(params)


def matrix_to_params_torch(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to 3 parameters (torch version).

    Parameters
    ----------
    R : torch.Tensor
        Rotation matrix (3, 3) or (N, 3, 3).

    Returns
    -------
    torch.Tensor
        Parameter tensor (3,) or (N, 3).
    """
    return matrix_to_axis_angle_torch(R)
```

### Implementation Notes

- **Primary implementations in math_torch.py**: `axis_angle_to_rotation_matrix`, `rotation_matrix_to_axis_angle`, `random_rotation_uniform`, `quaternion_to_rotation_matrix`
- Numpy versions wrap torch for consistency
- Torch versions (`params_to_matrix_torch`, `matrix_to_params_torch`) exported for direct optimization use
- Use scaled axis-angle: `params = axis * angle` (3 params, no singularities for small angles)
- For `random_rotation_params`: uses Shoemake's quaternion uniform sampling
- All torch versions support automatic differentiation for gradient-based optimization

---

## Patterson Calculation (Note: Moved to ReflectionData)

**patterson.py is not implemented as a separate file.** Patterson calculation is now a method of `ReflectionData`:

```python
# Usage
patterson = data.calc_patterson(grid_size=None)  # Returns torch.Tensor (nx, ny, nz)
```

The implementation in `ReflectionData.calc_patterson()`:
1. Expands reflection data to P1 symmetry
2. Uses `place_on_grid()` to place F² on reciprocal grid
3. Computes inverse FFT via `torch.fft.ifftn()`
4. Returns real-valued Patterson map

For Patterson interpolation, use `torch.nn.functional.grid_sample()` directly in `align.py`.

---

## File 2: `sampling.py`

Atom pair sampling for efficient Patterson vector generation. Uses torch tensors throughout for GPU compatibility.

### Class

```python
"""
Atom pair sampling for efficient Patterson vector generation.

Supports weighted sampling to prioritize informative pairs
(heavy atoms, close distances).
"""

import torch
from typing import Optional

class VectorSampler:
    """
    Samples atom pairs for Patterson vector matching.

    Supports weighted sampling to prioritize informative pairs
    (heavy atoms, close distances).

    Parameters
    ----------
    model : Model
        TorchRef Model object. Water molecules are automatically excluded.
    weighting : str, optional
        Weighting scheme: 'uniform', 'Z2', or 'Z2_distance'.
        Default is 'Z2'.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Attributes
    ----------
    model : Model
        Model with waters excluded.
    nsym_ops : int
        Number of symmetry operations (from model.symmetry.n_ops).
    weighting : str
        Weighting scheme used.
    weights : torch.Tensor
        Computed sampling weights per atom.
    """

    def __init__(
        self,
        model,
        weighting: str = 'Z2',
        seed: int = None
    ):
        self.model = model.select('not resname HOH')  # Exclude water molecules
        self.nsym_ops = model.symmetry.n_ops
        self.weighting = weighting
        self.rng = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
        self.weights = self._compute_weights()

    def _compute_weights(self) -> torch.Tensor:
        """
        Compute sampling probability for each atom.
        Weights are based on element weights (atomic numbers).

        Returns
        -------
        torch.Tensor
            Weight for each atom with shape (n_atoms * n_sym_ops,).
        """
        from torchref.utils.pse import PERIODIC_TABLE

        elements = self.model.pdb.element.values
        Zs = torch.tensor(
            [PERIODIC_TABLE[el]['number'] for el in elements],
            dtype=torch.float32, device=self.model.device
        )
        # Expand for all symmetry copies
        Zs = torch.stack([Zs for _ in range(self.nsym_ops)], dim=0).flatten()
        weights = Zs / Zs.mean()
        return weights

    def sample(
        self,
        n_vectors: int,
        weights: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample atom pairs according to weighting scheme.

        Parameters
        ----------
        n_vectors : int
            Number of atom pairs to sample.
        weights : torch.Tensor, optional
            Override weights for sampling. If None, uses self.weights.

        Returns
        -------
        idx1 : torch.Tensor
            Indices of first atoms in pairs (n_vectors,).
        idx2 : torch.Tensor
            Indices of second atoms in pairs (n_vectors,).

        Note: Ensures idx1 != idx2 for each pair.
        """
        weights = weights if weights is not None else self.weights

        # Sample first and second indices according to weights
        idx1 = torch.multinomial(self.weights, n_vectors, replacement=True, generator=self.rng)
        idx2 = torch.multinomial(self.weights, n_vectors, replacement=True, generator=self.rng)

        # Redraw idx2 where it equals idx1
        same_mask = idx1 == idx2
        while same_mask.any():
            n_resample = same_mask.sum().item()
            idx2[same_mask] = torch.multinomial(weights, n_resample, replacement=True, generator=self.rng)
            same_mask = idx1 == idx2

        return idx1, idx2
```

### Implementation Notes

- Takes `Model` object directly (not coordinates/atomic numbers separately)
- Automatically excludes water molecules via `model.select('not resname HOH')`
- Uses `model.symmetry.n_ops` to expand indices across all symmetry copies
- Uses `torch.multinomial` for GPU-accelerated weighted sampling
- Uses `torchref.utils.pse.PERIODIC_TABLE` for atomic number lookup
- Weights are computed once at initialization and reused

---

## Symmetry Handling (Note: No separate symmetry.py file)

**symmetry.py is not implemented as a separate file.** Symmetry operations use the existing `torchref.symmetrie.symmetrie.Symmetry` class accessed via `model.symmetry`.

### TorchRef Symmetry API

```python
from torchref.symmetrie.symmetrie import Symmetry

# Create from space group name
symmetry = Symmetry("P 21 21 21", dtype=torch.float64, device=torch.device('cpu'))

# Or get from Model (preferred)
symmetry = model.symmetry  # Already initialized

# Access operations (registered buffers)
symmetry.matrices       # torch.Tensor: (n_ops, 3, 3) rotation matrices (fractional space)
symmetry.translations   # torch.Tensor: (n_ops, 3) translation vectors (fractional space)
symmetry.n_ops          # int: number of symmetry operations

# Apply all operations at once
# Input: fractional coords of shape (N, 3)
# Output: (3, N, n_ops)
transformed = symmetry.apply(fractional_coords)
```

### Key Points

- **Access via `model.symmetry`**, not `model.spacegroup_function` (old API)
- Use `symmetry.n_ops` to get number of symmetry operations
- `Symmetry.apply()` works on fractional coordinates
- Operations stored in fractional space (rotation matrices and translation vectors)

---

## File 3: `align.py`

Main alignment class - **TorchRef-native interface**.

**Note**: This implementation is a work in progress. The current code shows the simplified structure.

### Classes

```python
"""
Main Patterson alignment class for TorchRef.

Aligns predicted structures to observed diffraction data via Patterson matching.
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import torch
import numpy as np

from torchref.model.model import Model
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.symmetrie.symmetrie import Symmetry
from torchref.math_functions.math_torch import (
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_axis_angle,
    random_rotation_uniform,
    cartesian_to_fractional_torch,
    fractional_to_cartesian_torch,
)

from .sampling import VectorSampler


@dataclass
class AlignmentResult:
    """
    Result of Patterson alignment.

    Attributes
    ----------
    rotation : torch.Tensor
        Rotation matrix (3, 3) in Cartesian space.
    translation : torch.Tensor
        Translation vector (3,) in Cartesian coordinates (Angstroms).
    score : float
        Patterson correlation score (higher = better match).
    n_starts : int
        Number of random starts used.
    converged : bool
        Whether optimization converged.
    """
    rotation: torch.Tensor
    translation: torch.Tensor
    score: float
    n_starts: int
    converged: bool

    def apply(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply transformation to coordinates."""
        return coords @ self.rotation.T + self.translation

    def as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (R, t) as numpy arrays."""
        return self.rotation.cpu().numpy(), self.translation.cpu().numpy()


class PattersonAligner:
    """
    Align predicted structures to diffraction data via Patterson matching.

    Parameters
    ----------
    data : ReflectionData
        Reflection data containing observed amplitudes (F), Miller indices (hkl),
        unit cell, and space group.
    model : Model
        Model object for extracting symmetry and coordinates.
    n_vectors : int, optional
        Number of atom pairs to sample for scoring. Default is 10000.
    weighting : str, optional
        Sampling weight scheme: 'uniform', 'Z2', or 'Z2_distance'. Default is 'Z2'.
    verbose : int, optional
        Verbosity level (0=silent, 1=normal, 2=debug). Default is 1.

    Attributes
    ----------
    patterson : torch.Tensor
        Precomputed Patterson map (nx, ny, nz) from data.calc_patterson().
    cell : torch.Tensor
        Unit cell parameters from data.
    symmetry : Symmetry
        Symmetry operations object from model.symmetry.
    model : Model
        Model with waters excluded.

    Examples
    --------
    >>> from torchref.model import Model
    >>> from torchref.io.datasets.reflection_data import ReflectionData
    >>> from torchref.alignment import PattersonAligner
    >>>
    >>> data = ReflectionData(verbose=1).load_mtz('observed.mtz')
    >>> model = Model(verbose=1).load_pdb('predicted.pdb')
    >>>
    >>> aligner = PattersonAligner(data, model)
    >>> aligned_model, result = aligner.align(model)
    """

    def __init__(
        self,
        data: ReflectionData,
        model: Model,
        n_vectors: int = 10000,
        weighting: str = 'Z2',
        verbose: int = 1
    ):
        # Store references
        self.data = data
        self.verbose = verbose
        self.device = torch.device(data.device) if isinstance(data.device, str) else data.device
        self.n_vectors = n_vectors
        self.weighting = weighting

        # Get cell parameters from data
        self.cell = data.cell.clone()

        # Get symmetry from model
        self.symmetry = model.symmetry

        # Exclude water molecules
        self.model = model.select('not resname HOH')

        # Precompute Patterson map using ReflectionData method
        self.patterson = data.calc_patterson()

    def evaluate_vectors_on_coords(
        self,
        idx1: torch.Tensor,
        idx2: torch.Tensor,
        xyz_fractional: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate Patterson score for vectors on fractional coordinates.

        Parameters
        ----------
        idx1 : torch.Tensor
            Indices of first atoms in pairs (n_vectors,).
        idx2 : torch.Tensor
            Indices of second atoms in pairs (n_vectors,).
        xyz_fractional : torch.Tensor
            Fractional coordinates of atoms (N, 3).

        Returns
        -------
        torch.Tensor
            Mean Patterson score for the vectors.
        """
        # Apply symmetry operations to get all copies
        coords_symmetrized = self.symmetry.apply(xyz_fractional).reshape(3, -1).T  # (N*n_ops, 3)

        # Compute difference vectors
        vecs = coords_symmetrized[idx1] - coords_symmetrized[idx2]

        # Use torch.nn.functional.grid_sample for trilinear interpolation
        # Convert fractional coordinates to grid coordinates [-1, 1]
        grid_coords = vecs * 2.0 - 1.0  # Map [0, 1] to [-1, 1]

        # Reshape for grid_sample: (1, 1, 1, n_vectors, 3)
        grid = grid_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Patterson map needs shape (1, 1, D, H, W)
        patterson_5d = self.patterson.unsqueeze(0).unsqueeze(0)

        # grid_sample uses (x, y, z) order -> flip coordinate order
        grid = grid.flip(-1)

        sampled = torch.nn.functional.grid_sample(
            patterson_5d, grid,
            mode='trilinear',
            padding_mode='border',
            align_corners=True
        )

        scores = sampled.squeeze()
        return scores.mean()
```

### Implementation Notes

- **Simplified API**: Model is required (not optional) - provides symmetry
- **Patterson from ReflectionData**: Uses `data.calc_patterson()` directly
- **Symmetry via model.symmetry**: Access symmetry operations from model
- **Water exclusion**: `model.select('not resname HOH')` at initialization
- **grid_sample for interpolation**: Uses `torch.nn.functional.grid_sample()` for Patterson lookup
- **Work in progress**: Full alignment optimization loop not yet complete

### Key API Changes from Design

| Design Document | Actual Implementation |
|-----------------|----------------------|
| `model` is optional | `model` is required |
| `grid_spacing` parameter | Removed (uses data defaults) |
| `calculate_patterson()` function | `data.calc_patterson()` method |
| `model.spacegroup_function` | `model.symmetry` |
| Custom trilinear interpolation | `torch.nn.functional.grid_sample()` |

---

## File 4: `__init__.py`

```python
"""
Patterson-based alignment module for TorchRef.

Aligns predicted structures to observed diffraction data using
Patterson map vector matching.

Example
-------
>>> from torchref.model import Model
>>> from torchref.io.datasets.reflection_data import ReflectionData
>>> from torchref.alignment import PattersonAligner
>>>
>>> data = ReflectionData().load_mtz('data.mtz')
>>> model = Model().load_pdb('predicted.pdb')
>>>
>>> aligner = PattersonAligner(data, model)
>>> aligned_model, result = aligner.align(model)
>>> aligned_model.write_pdb('aligned.pdb')
"""

from .align import PattersonAligner, AlignmentResult
from .sampling import VectorSampler
from .rotation import (
    params_to_matrix,
    matrix_to_params,
    random_rotation_params,
    random_rotation_matrix
)

__all__ = [
    # Main API
    'PattersonAligner',
    'AlignmentResult',
    # Lower-level utilities (for advanced users)
    'VectorSampler',
    'params_to_matrix',
    'matrix_to_params',
    'random_rotation_params',
    'random_rotation_matrix',
]
```

**Note**: Patterson calculation functions are no longer exported from alignment module.
Use `ReflectionData.calc_patterson()` directly instead.

---

## Testing Strategy

Tests are organized following the existing TorchRef test framework structure:
- **Unit tests** (`tests/unit/alignment/`): Fast, no I/O, use mock data
- **Functional tests** (`tests/functional/`): Real I/O, test with actual structures

### Test File Structure

```
tests/
├── unit/
│   └── alignment/
│       ├── test_rotation.py
│       ├── test_patterson.py
│       └── test_sampling.py
└── functional/
    └── test_alignment_functional.py   # Main recovery tests
```

---

### Unit Tests: `tests/unit/alignment/test_rotation.py`

```python
"""Unit tests for rotation parameterization."""
import pytest
import numpy as np
from torchref.alignment.rotation import (
    params_to_matrix, matrix_to_params, random_rotation_params,
    axis_angle_to_matrix
)


@pytest.mark.unit
class TestRotationParams:
    """Test rotation parameterization utilities."""

    def test_params_to_matrix_identity(self):
        """Zero params gives identity matrix."""
        params = np.zeros(3)
        R = params_to_matrix(params)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_params_to_matrix_roundtrip(self):
        """params -> matrix -> params recovers original."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            params_orig = rng.uniform(-np.pi, np.pi, size=3)
            R = params_to_matrix(params_orig)
            params_recovered = matrix_to_params(R)
            R_recovered = params_to_matrix(params_recovered)
            np.testing.assert_allclose(R, R_recovered, atol=1e-10)

    def test_rotation_matrix_orthogonal(self):
        """Output is valid rotation matrix (det=1, R.T @ R = I)."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            params = rng.uniform(-np.pi, np.pi, size=3)
            R = params_to_matrix(params)
            # Check orthogonality
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
            # Check determinant = 1 (proper rotation, not reflection)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_random_rotation_deterministic(self):
        """Same seed gives same rotation."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        params1 = random_rotation_params(rng1)
        params2 = random_rotation_params(rng2)
        np.testing.assert_array_equal(params1, params2)

    def test_random_rotation_uniform_coverage(self):
        """Random rotations cover SO(3) reasonably uniformly."""
        rng = np.random.default_rng(42)
        angles = []
        for _ in range(1000):
            params = random_rotation_params(rng)
            angle = np.linalg.norm(params)
            angles.append(angle)
        angles = np.array(angles)
        # Should have rotations across full range [0, π]
        assert angles.min() < 0.5
        assert angles.max() > 2.5
```

---

### Unit Tests: `tests/unit/alignment/test_patterson.py`

```python
"""Unit tests for Patterson map calculation."""
import pytest
import numpy as np
from torchref.alignment.patterson import (
    calculate_patterson, interpolate_patterson
)


@pytest.mark.unit
class TestPattersonMap:
    """Test Patterson map calculation and interpolation."""

    @pytest.fixture
    def simple_reflections(self):
        """Simple reflection data for testing."""
        # Create small HKL set
        hkl = np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1],
            [-1, 0, 0], [0, -1, 0], [0, 0, -1]
        ], dtype=np.int32)
        F = np.ones(len(hkl), dtype=np.float32) * 100.0
        cell = np.array([50.0, 50.0, 50.0, 90.0, 90.0, 90.0])
        return hkl, F, cell

    def test_patterson_shape(self, simple_reflections):
        """Output grid has expected dimensions based on cell and spacing."""
        hkl, F, cell = simple_reflections
        patterson, grid_info = calculate_patterson(F, hkl, cell, grid_spacing=1.0)

        # Grid should be roughly cell_length / spacing
        assert patterson.ndim == 3
        assert patterson.shape[0] >= 40  # ~50/1.0
        assert patterson.shape[1] >= 40
        assert patterson.shape[2] >= 40

    def test_patterson_origin_peak(self, simple_reflections):
        """Patterson has maximum at origin."""
        hkl, F, cell = simple_reflections
        patterson, grid_info = calculate_patterson(F, hkl, cell, grid_spacing=0.5)

        # Origin should be at (0, 0, 0) or near it
        origin_val = patterson[0, 0, 0]
        assert origin_val == patterson.max()

    def test_patterson_centrosymmetric(self, simple_reflections):
        """Patterson is centrosymmetric: P(u) == P(-u)."""
        hkl, F, cell = simple_reflections
        patterson, grid_info = calculate_patterson(F, hkl, cell, grid_spacing=0.5)

        nx, ny, nz = patterson.shape
        # Check several random points
        for i in range(1, min(10, nx // 2)):
            for j in range(1, min(10, ny // 2)):
                for k in range(1, min(10, nz // 2)):
                    assert np.isclose(
                        patterson[i, j, k],
                        patterson[-i % nx, -j % ny, -k % nz],
                        rtol=1e-5
                    )

    def test_interpolation_at_grid_points(self, simple_reflections):
        """Interpolation exact at grid points."""
        hkl, F, cell = simple_reflections
        patterson, grid_info = calculate_patterson(F, hkl, cell, grid_spacing=1.0)

        # Sample at exact grid point (in Cartesian)
        # Grid point (5, 5, 5) corresponds to fractional (5/nx, 5/ny, 5/nz)
        nx, ny, nz = patterson.shape
        frac_coords = np.array([[5/nx, 5/ny, 5/nz]])
        # Convert to Cartesian (for orthogonal cell, just multiply by cell lengths)
        cart_coords = frac_coords * cell[:3]

        interp_val = interpolate_patterson(patterson, cart_coords, cell, grid_info)
        expected_val = patterson[5, 5, 5]
        np.testing.assert_allclose(interp_val[0], expected_val, rtol=1e-5)

    def test_interpolation_periodic(self, simple_reflections):
        """Values wrap correctly at cell boundaries."""
        hkl, F, cell = simple_reflections
        patterson, grid_info = calculate_patterson(F, hkl, cell, grid_spacing=0.5)

        # Point at u and u + cell should give same value
        u = np.array([[10.0, 15.0, 20.0]])
        u_wrapped = u + cell[:3]

        val1 = interpolate_patterson(patterson, u, cell, grid_info)
        val2 = interpolate_patterson(patterson, u_wrapped, cell, grid_info)
        np.testing.assert_allclose(val1, val2, rtol=1e-5)
```

---

### Unit Tests: `tests/unit/alignment/test_sampling.py`

```python
"""Unit tests for vector sampling."""
import pytest
import numpy as np
from torchref.alignment.sampling import VectorSampler


@pytest.mark.unit
class TestVectorSampler:
    """Test atom pair sampling for Patterson matching."""

    @pytest.fixture
    def sample_structure(self):
        """Small structure for testing."""
        coords = np.array([
            [0, 0, 0],
            [1.5, 0, 0],
            [0, 1.5, 0],
            [0, 0, 1.5],
            [3, 3, 3],
        ], dtype=np.float32)
        Z = np.array([6, 6, 7, 8, 26], dtype=np.int32)  # C, C, N, O, Fe
        return coords, Z

    def test_sampler_correct_count(self, sample_structure):
        """Returns requested number of vectors."""
        coords, Z = sample_structure
        sampler = VectorSampler(n_vectors=100)
        i_idx, j_idx, weights = sampler.sample(coords, Z)

        assert len(i_idx) == 100
        assert len(j_idx) == 100
        assert len(weights) == 100

    def test_sampler_no_self_pairs(self, sample_structure):
        """Never samples i == j (self-vectors are zero)."""
        coords, Z = sample_structure
        sampler = VectorSampler(n_vectors=1000)
        i_idx, j_idx, weights = sampler.sample(coords, Z)

        assert not np.any(i_idx == j_idx)

    def test_sampler_deterministic(self, sample_structure):
        """Same seed gives same samples."""
        coords, Z = sample_structure

        sampler1 = VectorSampler(n_vectors=100, seed=42)
        i1, j1, w1 = sampler1.sample(coords, Z)

        sampler2 = VectorSampler(n_vectors=100, seed=42)
        i2, j2, w2 = sampler2.sample(coords, Z)

        np.testing.assert_array_equal(i1, i2)
        np.testing.assert_array_equal(j1, j2)
        np.testing.assert_array_equal(w1, w2)

    def test_z2_weighting_favors_heavy(self, sample_structure):
        """Heavy atoms (Fe) sampled more frequently with Z2 weighting."""
        coords, Z = sample_structure  # Fe is at index 4

        sampler = VectorSampler(n_vectors=10000, weighting='Z2', seed=42)
        i_idx, j_idx, weights = sampler.sample(coords, Z)

        # Count how often Fe (index 4) appears
        fe_count = np.sum(i_idx == 4) + np.sum(j_idx == 4)
        light_count = np.sum(i_idx == 0) + np.sum(j_idx == 0)  # First carbon

        # Fe (Z=26) should appear much more than C (Z=6)
        # Ratio should be roughly (26/6)^2 ≈ 18.8
        assert fe_count > light_count * 5  # Conservative check

    def test_compute_vectors(self, sample_structure):
        """Vectors computed correctly from indices."""
        coords, Z = sample_structure
        sampler = VectorSampler(n_vectors=10)
        i_idx, j_idx, weights = sampler.sample(coords, Z)

        vectors = sampler.compute_vectors(coords, i_idx, j_idx)

        # Manual check
        for k in range(len(i_idx)):
            expected = coords[j_idx[k]] - coords[i_idx[k]]
            np.testing.assert_allclose(vectors[k], expected)
```

---

### Functional Tests: `tests/functional/test_alignment_functional.py`

```python
"""
Functional tests for Patterson-based alignment.

These tests load real structures, apply known transformations,
and verify that the aligner can recover the original orientation.
"""
import pytest
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from torchref.model.model import Model
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.alignment import PattersonAligner, AlignmentResult
from torchref.alignment.rotation import params_to_matrix, random_rotation_params


@pytest.mark.integration
class TestAlignmentRecovery:
    """
    Test that PattersonAligner can recover known transformations.

    Strategy:
    1. Load a real structure and reflection data
    2. Apply a known rotation R_true and translation t_true
    3. Run alignment to find R_found, t_found
    4. Verify R_found ≈ R_true^(-1) and t_found recovers original position
    """

    @pytest.fixture
    def structure_and_data(self, sample_structure_pair):
        """Load a real model and reflection data pair."""
        model = Model(verbose=0)
        model.load_cif(str(sample_structure_pair["model"]))

        data = ReflectionData(verbose=0)
        data.load_mtz(str(sample_structure_pair["reflections"]))

        return model, data

    @pytest.fixture
    def aligner(self, structure_and_data):
        """Create PattersonAligner from real data."""
        model, data = structure_and_data
        return PattersonAligner(
            data=data,
            model=model,
            grid_spacing=0.5,
            n_vectors=5000,
            weighting='Z2',
            verbose=0
        )

    def apply_transformation(self, model: Model, R: np.ndarray, t: np.ndarray) -> Model:
        """Apply rotation R and translation t to model coordinates."""
        transformed = model.copy()
        coords = model.xyz().detach().cpu().numpy()
        new_coords = coords @ R.T + t
        new_tensor = torch.from_numpy(new_coords).to(
            dtype=model.dtype_float,
            device=model.device
        )
        transformed.xyz.fixed_values = new_tensor
        transformed.xyz.refinable_params.data = new_tensor[
            transformed.xyz.refinable_mask
        ]
        return transformed

    def get_random_transformation(self, cell: np.ndarray, seed: int = None):
        """Generate random rotation and translation within unit cell."""
        rng = np.random.default_rng(seed)

        # Random rotation (uniform over SO(3))
        rot_params = random_rotation_params(rng)
        R = params_to_matrix(rot_params)

        # Random translation within unit cell
        t = rng.uniform(0, 1, size=3) * cell[:3]

        return R, t

    # ================================================================
    # Test: Identity (no transformation)
    # ================================================================
    def test_align_identity(self, aligner, structure_and_data):
        """
        Already-aligned structure returns R ≈ I, t ≈ 0.

        When the model is already in the correct position, the aligner
        should find a transformation close to identity.
        """
        model, data = structure_and_data

        aligned_model, result = aligner.align(model, n_starts=5)

        # Check rotation is close to identity
        R = result.rotation.cpu().numpy()
        np.testing.assert_allclose(R, np.eye(3), atol=0.1)

        # Translation should be small or a lattice translation
        t = result.translation.cpu().numpy()
        cell = data.cell.cpu().numpy()

        # t should be near 0 or near cell edges (equivalent positions)
        t_frac = t / cell[:3]
        t_frac_wrapped = t_frac - np.round(t_frac)  # Wrap to [-0.5, 0.5]
        np.testing.assert_allclose(t_frac_wrapped, 0, atol=0.1)

    # ================================================================
    # Test: Pure rotation recovery
    # ================================================================
    @pytest.mark.parametrize("angle_deg", [30, 60, 90, 120, 180])
    def test_align_known_rotation(self, aligner, structure_and_data, angle_deg):
        """
        Recovers known rotation applied to structure.

        Apply rotation R_true, then align. The recovered transformation
        should rotate the structure back to the original.
        """
        model, data = structure_and_data

        # Create rotation around z-axis
        angle_rad = np.deg2rad(angle_deg)
        R_true = Rotation.from_rotvec([0, 0, angle_rad]).as_matrix()
        t_true = np.zeros(3)

        # Apply transformation
        transformed = self.apply_transformation(model, R_true, t_true)

        # Align back
        aligned_model, result = aligner.align(transformed, n_starts=10)

        # The alignment should find R_found such that R_found @ R_true ≈ I
        # (i.e., R_found ≈ R_true^T)
        R_found = result.rotation.cpu().numpy()
        R_combined = R_found @ R_true

        np.testing.assert_allclose(R_combined, np.eye(3), atol=0.15,
            err_msg=f"Failed for rotation angle {angle_deg}°")

    # ================================================================
    # Test: Pure translation recovery
    # ================================================================
    def test_align_known_translation(self, aligner, structure_and_data):
        """
        Recovers known translation within unit cell.

        Apply translation t_true, then align. The recovered position
        should match the original (modulo lattice translations).
        """
        model, data = structure_and_data
        cell = data.cell.cpu().numpy()

        # Apply translation (half cell in each direction)
        R_true = np.eye(3)
        t_true = cell[:3] * 0.3  # 30% along each axis

        transformed = self.apply_transformation(model, R_true, t_true)

        # Align back
        aligned_model, result = aligner.align(transformed, n_starts=10)

        # Compare centroids of aligned and original
        original_centroid = model.xyz().mean(dim=0).cpu().numpy()
        aligned_centroid = aligned_model.xyz().mean(dim=0).cpu().numpy()

        # Should match within tolerance (accounting for periodicity)
        diff = aligned_centroid - original_centroid
        diff_frac = diff / cell[:3]
        diff_frac_wrapped = diff_frac - np.round(diff_frac)

        np.testing.assert_allclose(diff_frac_wrapped, 0, atol=0.1)

    # ================================================================
    # Test: Combined rotation + translation
    # ================================================================
    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_align_rotation_and_translation(self, aligner, structure_and_data, seed):
        """
        Recovers combined random rotation and translation.

        This is the main functional test: apply arbitrary transformation,
        verify alignment recovers the original structure position.
        """
        model, data = structure_and_data
        cell = data.cell.cpu().numpy()

        # Random transformation
        R_true, t_true = self.get_random_transformation(cell, seed=seed)

        # Apply transformation
        transformed = self.apply_transformation(model, R_true, t_true)

        # Align back
        aligned_model, result = aligner.align(transformed, n_starts=15)

        # Compare coordinates after alignment
        original_coords = model.xyz().cpu().numpy()
        aligned_coords = aligned_model.xyz().cpu().numpy()

        # Center both for comparison (removes translation ambiguity)
        original_centered = original_coords - original_coords.mean(axis=0)
        aligned_centered = aligned_coords - aligned_coords.mean(axis=0)

        # Compute RMSD between aligned and original (should be small)
        rmsd = np.sqrt(np.mean(np.sum((aligned_centered - original_centered)**2, axis=1)))

        # RMSD should be small (< 1 Å for successful alignment)
        assert rmsd < 2.0, f"RMSD {rmsd:.2f} Å too high for seed {seed}"

    # ================================================================
    # Test: Alignment with atom selection
    # ================================================================
    def test_align_with_selection(self, aligner, structure_and_data):
        """
        Alignment using only backbone atoms for scoring.

        Verifies that selection parameter works correctly.
        """
        model, data = structure_and_data
        cell = data.cell.cpu().numpy()

        # Apply transformation
        R_true, t_true = self.get_random_transformation(cell, seed=42)
        transformed = self.apply_transformation(model, R_true, t_true)

        # Align using only CA atoms
        aligned_model, result = aligner.align(
            transformed,
            n_starts=10,
            selection="name CA"
        )

        # Should still recover the structure
        original_coords = model.xyz().cpu().numpy()
        aligned_coords = aligned_model.xyz().cpu().numpy()

        original_centered = original_coords - original_coords.mean(axis=0)
        aligned_centered = aligned_coords - aligned_coords.mean(axis=0)

        rmsd = np.sqrt(np.mean(np.sum((aligned_centered - original_centered)**2, axis=1)))
        assert rmsd < 2.0

    # ================================================================
    # Test: Robustness to coordinate noise
    # ================================================================
    @pytest.mark.parametrize("noise_std", [0.1, 0.2, 0.5])
    def test_align_with_noise(self, aligner, structure_and_data, noise_std):
        """
        Robust to small coordinate perturbations.

        Tests that alignment still works when coordinates have noise,
        simulating imperfect predicted structures.
        """
        model, data = structure_and_data
        cell = data.cell.cpu().numpy()

        # Apply transformation
        R_true, t_true = self.get_random_transformation(cell, seed=42)
        transformed = self.apply_transformation(model, R_true, t_true)

        # Add Gaussian noise to coordinates
        rng = np.random.default_rng(42)
        noisy_coords = transformed.xyz().detach().cpu().numpy()
        noisy_coords += rng.normal(0, noise_std, size=noisy_coords.shape)

        noisy_tensor = torch.from_numpy(noisy_coords).to(
            dtype=model.dtype_float, device=model.device
        )
        transformed.xyz.fixed_values = noisy_tensor
        transformed.xyz.refinable_params.data = noisy_tensor[
            transformed.xyz.refinable_mask
        ]

        # Align
        aligned_model, result = aligner.align(transformed, n_starts=15)

        # RMSD threshold scales with noise level
        original_coords = model.xyz().cpu().numpy()
        aligned_coords = aligned_model.xyz().cpu().numpy()

        original_centered = original_coords - original_coords.mean(axis=0)
        aligned_centered = aligned_coords - aligned_coords.mean(axis=0)

        rmsd = np.sqrt(np.mean(np.sum((aligned_centered - original_centered)**2, axis=1)))

        # Allow larger RMSD for noisier structures
        max_rmsd = 2.0 + noise_std * 2
        assert rmsd < max_rmsd, f"RMSD {rmsd:.2f} Å too high for noise {noise_std} Å"


@pytest.mark.integration
class TestAlignmentAllStructures:
    """
    Run alignment recovery on all available test structures.

    This ensures the aligner works across different space groups
    and structure types.
    """

    def test_all_structures_recovery(self, all_structure_pairs):
        """
        Test alignment recovery on all available test structures.

        For each structure:
        1. Load model and data
        2. Apply random transformation
        3. Verify alignment recovers original
        """
        results = []

        for pair in all_structure_pairs:
            # Load
            model = Model(verbose=0)
            model.load_cif(str(pair["model"]))

            data = ReflectionData(verbose=0)
            data.load_mtz(str(pair["reflections"]))

            # Skip if spacegroups don't match
            if model.spacegroup != data.spacegroup:
                continue

            # Create aligner
            aligner = PattersonAligner(
                data=data,
                model=model,
                grid_spacing=0.5,
                n_vectors=3000,
                weighting='Z2',
                verbose=0
            )

            # Random transformation
            rng = np.random.default_rng(42)
            rot_params = random_rotation_params(rng)
            R_true = params_to_matrix(rot_params)
            cell = data.cell.cpu().numpy()
            t_true = rng.uniform(0, 1, size=3) * cell[:3]

            # Apply and align
            coords = model.xyz().detach().cpu().numpy()
            new_coords = coords @ R_true.T + t_true
            transformed = model.copy()
            new_tensor = torch.from_numpy(new_coords).to(
                dtype=model.dtype_float, device=model.device
            )
            transformed.xyz.fixed_values = new_tensor
            transformed.xyz.refinable_params.data = new_tensor[
                transformed.xyz.refinable_mask
            ]

            aligned_model, result = aligner.align(transformed, n_starts=10)

            # Compute RMSD
            original_coords = model.xyz().cpu().numpy()
            aligned_coords = aligned_model.xyz().cpu().numpy()
            original_centered = original_coords - original_coords.mean(axis=0)
            aligned_centered = aligned_coords - aligned_coords.mean(axis=0)
            rmsd = np.sqrt(np.mean(np.sum(
                (aligned_centered - original_centered)**2, axis=1
            )))

            results.append({
                "pdb_id": pair["model"].stem,
                "spacegroup": model.spacegroup,
                "n_atoms": len(coords),
                "rmsd": rmsd,
                "score": result.score,
                "converged": result.converged
            })

        # Print summary
        print("\n" + "="*60)
        print("Alignment Recovery Test Results")
        print("="*60)
        for r in results:
            status = "✓" if r["rmsd"] < 2.0 else "✗"
            print(f"{status} {r['pdb_id']:6s} | SG: {r['spacegroup']:12s} | "
                  f"atoms: {r['n_atoms']:5d} | RMSD: {r['rmsd']:.2f} Å | "
                  f"score: {r['score']:.2f}")
        print("="*60)

        # All should pass
        for r in results:
            assert r["rmsd"] < 2.0, f"Failed for {r['pdb_id']}: RMSD={r['rmsd']:.2f}"
```

---

### Fixture Addition: `tests/functional/conftest.py`

Add this fixture to the functional conftest:

```python
# Add to tests/functional/conftest.py

@pytest.fixture(scope="session")
def all_structure_pairs(cif_dir, mtz_dir):
    """
    Return list of all matching CIF/MTZ pairs.

    Each pair is a dict: {"model": Path, "reflections": Path}
    """
    pairs = []
    for cif_file in sorted(cif_dir.glob("*.cif")):
        pdb_id = cif_file.stem
        mtz_file = mtz_dir / f"{pdb_id}.mtz"
        if mtz_file.exists():
            pairs.append({
                "model": cif_file,
                "reflections": mtz_file
            })
    return pairs
```

---

## Dependencies

### External Dependencies
- `torch` - primary computation framework (replaces numpy and scipy where possible)
- `numpy` - minimal usage for random number generation and compatibility
- `gemmi` - for atomic number lookup from element symbols (via `torchref.utils.pse`)

**No scipy dependency** - all operations use pure torch implementations.

### TorchRef Internal Dependencies

#### Model & Data Classes
- `torchref.model.model.Model` - atomic structure container
  - `.xyz()` - get coordinates tensor
  - `.pdb` - DataFrame with element column
  - `.cell` - unit cell tensor
  - `.symmetry` - Symmetry object (note: not `.spacegroup_function`)
  - `.select(selection)` - select atoms by selection string
  - `.copy()` - deep copy model
- `torchref.io.datasets.reflection_data.ReflectionData` - reflection data
  - `.F` - amplitudes
  - `.hkl` - Miller indices
  - `.cell` - unit cell tensor
  - `.masks()` - validity mask
  - `.calc_patterson()` - calculate Patterson map (returns torch.Tensor)
- `torchref.symmetrie.symmetrie.Symmetry` - symmetry operations
  - `.matrices` - rotation matrices buffer (n_ops, 3, 3)
  - `.translations` - translation vectors buffer (n_ops, 3)
  - `.n_ops` - number of symmetry operations
  - `.apply()` - apply all symops to fractional coordinates

#### Math Utilities (from `torchref.math_functions.math_torch`)

**Rotation utilities**:
- `axis_angle_to_rotation_matrix(axis_angle)` - Convert axis-angle to rotation matrix
- `rotation_matrix_to_axis_angle(R)` - Convert rotation matrix to axis-angle
- `quaternion_to_rotation_matrix(q)` - Convert quaternion to rotation matrix
- `random_rotation_uniform(n, device, dtype)` - Generate uniform random rotations
- `trilinear_interpolate(grid, points, mode)` - Trilinear interpolation

**Coordinate transforms**:
- `cartesian_to_fractional_torch(cart_coords, unit_cell)` - Cartesian → fractional
- `fractional_to_cartesian_torch(frac_coords, unit_cell)` - Fractional → Cartesian

#### Other Utilities
- `torchref.utils.pse.PERIODIC_TABLE` - periodic table with atomic numbers

---

## Implementation Order

1. **`rotation.py`** - numpy wrappers around math_torch functions ✓
2. **`sampling.py`** - torch-based atom pair sampling ✓
3. **`align.py`** - Patterson aligner (work in progress)
4. **Tests** - alongside each module

---

## Performance Targets

- Patterson calculation: < 1 second (one-time)
- Single score evaluation: < 10 ms
- Full alignment (10 starts × ~50 iterations): < 10 seconds

---

## Future Extensions

- Torch-differentiable version for end-to-end training
- Coarse-to-fine: start with fewer vectors, increase for refinement
- Parallel multi-start with joblib
- Cache sampled indices across training epochs

---

## Complete Integration Example

```python
"""
End-to-end example: Align a predicted structure to experimental data.

This is the recommended usage pattern for the PattersonAligner.
"""
from torchref.model import Model
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.alignment import PattersonAligner

# ============================================================
# 1. Load experimental data and predicted structure
# ============================================================
data = ReflectionData(verbose=1, device='cpu').load_mtz('experimental.mtz')
model = Model(verbose=1).load_pdb('predicted.pdb')

print(f"Data: {len(data)} reflections, {data.spacegroup}")
print(f"Model: {len(model.pdb)} atoms, {model.spacegroup}")

# ============================================================
# 2. Create aligner (precomputes Patterson map)
# ============================================================
aligner = PattersonAligner(
    data=data,
    model=model,           # Required: provides symmetry info
    n_vectors=10000,       # Atom pairs for scoring
    weighting='Z2',        # Weight by Z²
    verbose=1
)

# Patterson map is computed via data.calc_patterson()
print(f"Patterson map shape: {aligner.patterson.shape}")

# ============================================================
# 3. Align model to data (when align() is fully implemented)
# ============================================================
# Note: Full alignment loop is work in progress

# Basic alignment (returns copy) - future API:
# aligned_model, result = aligner.align(model, n_starts=20)

# ============================================================
# 4. Current functionality: evaluate Patterson score
# ============================================================
# Sample atom pairs
from torchref.alignment import VectorSampler

sampler = VectorSampler(model, weighting='Z2', seed=42)
idx1, idx2 = sampler.sample(n_vectors=1000)

# Evaluate score on fractional coordinates
from torchref.math_functions.math_torch import cartesian_to_fractional_torch

xyz_cart = model.xyz()
xyz_frac = cartesian_to_fractional_torch(xyz_cart, data.cell)
score = aligner.evaluate_vectors_on_coords(idx1, idx2, xyz_frac)
print(f"Patterson score: {score.item():.4f}")

# ============================================================
# 5. Validate alignment with structure factors (optional)
# ============================================================
from torchref.model.model_ft import ModelFT

model_ft = ModelFT(model, data)
F_calc = model_ft(data.hkl)

mask = data.masks()
F_obs = data.F[mask]
F_calc_amp = torch.abs(F_calc[mask])

correlation = torch.corrcoef(torch.stack([F_obs, F_calc_amp]))[0, 1].item()
print(f"F_obs/F_calc correlation = {correlation:.4f}")
```

---

## API Summary

```python
# ============================================================
# PattersonAligner - Main Class
# ============================================================
from torchref.alignment import PattersonAligner, AlignmentResult

# Create aligner from TorchRef objects
aligner = PattersonAligner(
    data: ReflectionData,          # Required: reflection data
    model: Model,                  # Required: provides symmetry
    n_vectors: int = 10000,        # Number of atom pairs to sample
    weighting: str = 'Z2',         # 'uniform', 'Z2', or 'Z2_distance'
    verbose: int = 1               # Verbosity level
)

# Attributes
aligner.patterson    # torch.Tensor (nx, ny, nz) - Patterson map from data.calc_patterson()
aligner.cell         # torch.Tensor (6,) - unit cell parameters
aligner.symmetry     # Symmetry - from model.symmetry
aligner.model        # Model - with waters excluded

# Evaluate Patterson score for atom pair vectors
score = aligner.evaluate_vectors_on_coords(
    idx1: torch.Tensor,            # First atom indices
    idx2: torch.Tensor,            # Second atom indices
    xyz_fractional: torch.Tensor   # Fractional coordinates
)

# AlignmentResult (when align() is fully implemented)
result.rotation      # torch.Tensor (3, 3) - rotation matrix
result.translation   # torch.Tensor (3,) - translation vector
result.score         # float - Patterson score
result.n_starts      # int - number of starts used
result.converged     # bool - optimization converged

result.apply(coords) # Apply transformation to coordinates
result.as_numpy()    # Return (R, t) as numpy arrays

# ============================================================
# VectorSampler - Atom Pair Sampling
# ============================================================
from torchref.alignment import VectorSampler

sampler = VectorSampler(
    model: Model,                  # Required: TorchRef Model
    weighting: str = 'Z2',         # Weighting scheme
    seed: int = None               # Random seed
)

idx1, idx2 = sampler.sample(n_vectors: int)  # Sample atom pairs
```