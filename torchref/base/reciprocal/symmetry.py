"""Reciprocal-space ("late") symmetry for structure factor calculation.

The alternative to symmetrizing the density map before the FFT ("early" symmetry,
:func:`~torchref.symmetry.MapSymmetry`): here symmetry is applied to the P1
transform afterwards, avoiding the map symmetrization entirely. Per operation
{R|t},

    F_sym(h) = Σ_ops exp(2πi h·t) · F_P1(Rᵀ·h)

and because crystallographic R is integer-valued, Rᵀ·h lands exactly on grid
points and needs no interpolation. **Every grid or map argument here is the P1
one** -- feeding in an already-symmetrized grid double-counts.
"""

from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

from torchref.config import canonical_device, get_float_dtype
from torchref.utils.autograd_ops import gather_with_index_add

from .grid_operations import extract_structure_factor_from_grid

if TYPE_CHECKING:
    from torchref.symmetry.spacegroup import SpaceGroup


def compute_symmetry_equivalent_hkls(
    hkl: torch.Tensor,
    rotation_matrices: torch.Tensor,
) -> torch.Tensor:
    """
    Compute symmetry-equivalent HKLs for each operation.

    Row-vector convention: h' = h @ R, equivalently Rᵀ·h for column h. The
    matrices are used as given -- do **not** pre-transpose them.

    Parameters
    ----------
    hkl : torch.Tensor, shape (N, 3)
        Miller indices.
    rotation_matrices : torch.Tensor, shape (n_ops, 3, 3)
        Real-space rotation matrices, applied directly as ``h @ R`` (no
        transpose).

    Returns
    -------
    torch.Tensor, shape (n_ops, N, 3)
        Equivalent HKLs for each symmetry operation.
    """
    device = hkl.device
    dtype = get_float_dtype()

    hkl_float = hkl.to(dtype=dtype, device=device)  # (N, 3)
    rot_matrices = rotation_matrices.to(dtype=dtype, device=device)  # (n_ops, 3, 3)

    n_ops = rot_matrices.shape[0]

    hkl_expanded = hkl_float.unsqueeze(0).expand(n_ops, -1, -1)  # (n_ops, N, 3)

    equiv_hkl = torch.bmm(hkl_expanded, rot_matrices)

    # Exact for valid crystallographic ops; round only mops up float error.
    equiv_hkl = torch.round(equiv_hkl).to(torch.int64)

    return equiv_hkl


def compute_translation_phases(
    hkl: torch.Tensor,
    translations: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the translation phase shifts exp(2πi h·t) for each operation.

    Parameters
    ----------
    hkl : torch.Tensor, shape (N, 3)
        Miller indices.
    translations : torch.Tensor, shape (n_ops, 3)
        Translation vectors in fractional coordinates.

    Returns
    -------
    torch.Tensor, shape (n_ops, N)
        Complex phase factors exp(2*pi*i * h.t).
    """
    device = hkl.device
    dtype = get_float_dtype()

    hkl_float = hkl.to(dtype=dtype, device=device)  # (N, 3)
    translations = translations.to(dtype=dtype, device=device)  # (n_ops, 3)

    h_dot_t = torch.matmul(hkl_float, translations.T).T  # (n_ops, N)

    # Leave ``phase`` at the configured float dtype; casting it to float32 would
    # force complex64 output even under a float64 configuration.
    phase = 2.0 * np.pi * h_dot_t
    phase_factor = torch.exp(1j * phase)

    return phase_factor  # (n_ops, N) complex


def extract_structure_factors_with_symmetry(
    reciprocal_grid: torch.Tensor,
    hkl: torch.Tensor,
    rotation_matrices: torch.Tensor,
    translations: torch.Tensor,
) -> torch.Tensor:
    """
    Extract structure factors with symmetry applied in reciprocal space.

    Sums F over the symmetry-equivalent positions with their translation phases,
    replacing the symmetrize-then-extract MapSymmetry route. For repeated calls
    with the same hkl and symmetry, use :class:`ReciprocalSymmetryExtractor`.

    Parameters
    ----------
    reciprocal_grid : torch.Tensor, shape (Nx, Ny, Nz)
        Complex reciprocal space grid from FFT of the **P1** density map.
    hkl : torch.Tensor, shape (N, 3)
        Target Miller indices.
    rotation_matrices : torch.Tensor, shape (n_ops, 3, 3)
        Real-space rotation matrices from symmetry operations.
    translations : torch.Tensor, shape (n_ops, 3)
        Translation vectors from symmetry operations.

    Returns
    -------
    torch.Tensor, shape (N,)
        Complex structure factors with symmetry applied.
    """
    device = reciprocal_grid.device
    Nx, Ny, Nz = reciprocal_grid.shape

    # Move everything to the same device
    hkl = hkl.to(device=device)
    rotation_matrices = rotation_matrices.to(device=device)
    translations = translations.to(device=device)

    n_ops = rotation_matrices.shape[0]
    N = hkl.shape[0]

    equiv_hkls = compute_symmetry_equivalent_hkls(hkl, rotation_matrices)

    # One flat gather for all symops. gather_with_index_add keeps the backward a
    # single index_add_ instead of a radix-sort + dedup scatter.
    flat_indices = _equiv_hkls_to_flat_indices(equiv_hkls, Nx, Ny, Nz)
    f_all = gather_with_index_add(
        reciprocal_grid.reshape(-1), flat_indices,
    )  # (n_ops * N,)
    f_p1 = f_all.view(n_ops, N)

    phases = compute_translation_phases(hkl, translations)

    f_sym = (f_p1 * phases).sum(dim=0)

    return f_sym


def _equiv_hkls_to_flat_indices(
    equiv_hkls: torch.Tensor, Nx: int, Ny: int, Nz: int,
) -> torch.Tensor:
    """Convert (n_ops, N, 3) equiv HKLs to flat linear grid indices."""
    all_hkl = equiv_hkls.reshape(-1, 3)  # (n_ops*N, 3)
    hi = torch.remainder(all_hkl[:, 0], Nx)
    ki = torch.remainder(all_hkl[:, 1], Ny)
    li = torch.remainder(all_hkl[:, 2], Nz)
    return (hi * (Ny * Nz) + ki * Nz + li).to(torch.int64)


from torchref.utils.device_mixin import DeviceMixin


class ReciprocalSymmetryExtractor(DeviceMixin):
    """
    Class-based interface for reciprocal space symmetry extraction.

    For repeated structure-factor evaluation at fixed hkl and symmetry, as in
    refinement: the equivalent HKLs, phase factors and flat grid indices are
    precomputed here, so each call is one gather, multiply and sum. The
    precomputation binds ``grid_shape`` -- a differently shaped grid needs a new
    extractor.

    Parameters
    ----------
    hkl : torch.Tensor, shape (N, 3)
        Target Miller indices.
    symmetry : SpaceGroup
        SpaceGroup object containing rotation matrices and translations.
    grid_shape : tuple of int
        Reciprocal grid dimensions (Nx, Ny, Nz).
    device : torch.device, optional
        Device for computation.

    Examples
    --------
    >>> extractor = ReciprocalSymmetryExtractor(hkl, symmetry, grid_shape=(209, 86, 67))
    >>> f_calc = extractor.extract_from_grid(reciprocal_grid)
    """

    def __init__(
        self,
        hkl: torch.Tensor,
        symmetry: "SpaceGroup",
        grid_shape: tuple,
        device: Optional[torch.device] = None,
    ):
        # ``is not None``, not ``or``: ``device=0`` means cuda:0/mps:0 and is
        # falsy, so ``or`` silently discarded it. ``hkl`` is a bare tensor, so
        # this reads its device rather than going through ``resolve_device``.
        self.device = canonical_device(
            device if device is not None else hkl.device
        )
        self.hkl = hkl.to(device=self.device)
        self.symmetry = symmetry
        self.n_ops = symmetry.n_ops
        self.N = len(hkl)
        self.grid_shape = grid_shape

        # Precompute equivalent HKLs
        self.equiv_hkls = compute_symmetry_equivalent_hkls(
            self.hkl,
            symmetry.matrices.to(device=self.device),
        )  # (n_ops, N, 3)

        # Precompute phase factors
        self.phases = compute_translation_phases(
            self.hkl,
            symmetry.translations.to(device=self.device),
        )  # (n_ops, N) complex

        # Precompute flat linear indices for single-gather extraction
        Nx, Ny, Nz = grid_shape
        self._flat_indices = _equiv_hkls_to_flat_indices(
            self.equiv_hkls, Nx, Ny, Nz,
        )  # (n_ops * N,) int64

    def __call__(self, density_map: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`extract`."""
        return self.extract(density_map)

    def extract(self, density_map: torch.Tensor) -> torch.Tensor:
        """
        Transform a P1 density map and extract symmetrized structure factors.

        Parameters
        ----------
        density_map : torch.Tensor, shape (Nx, Ny, Nz)
            **P1** electron density map -- passing a symmetrized map double-counts.

        Returns
        -------
        torch.Tensor, shape (N,)
            Complex structure factors with symmetry applied.
        """
        from torchref.base.fourier.fft import ifft
        reciprocal_grid = ifft(density_map)

        return self.extract_from_grid(reciprocal_grid)

    def extract_from_grid(self, reciprocal_grid: torch.Tensor) -> torch.Tensor:
        """
        Extract structure factors from an already-transformed P1 grid.

        Parameters
        ----------
        reciprocal_grid : torch.Tensor, shape (Nx, Ny, Nz)
            Complex reciprocal space grid from FFT of the **P1** map; its shape
            must match the ``grid_shape`` this extractor was built for.

        Returns
        -------
        torch.Tensor, shape (N,)
            Complex structure factors with symmetry applied.
        """
        # gather_with_index_add keeps the backward a single ``index_add_``
        # (atomic scatter, no radix sort + dedup).
        f_all = gather_with_index_add(
            reciprocal_grid.reshape(-1), self._flat_indices,
        )  # (n_ops * N,)
        f_sym = (f_all.view(self.n_ops, self.N) * self.phases).sum(dim=0)
        return f_sym

