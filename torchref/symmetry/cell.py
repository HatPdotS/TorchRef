"""
Cell - A dataclass for crystallographic unit cells with cached derived quantities.

Provides a simple container for unit cell parameters with automatic caching
of derived quantities (fractional matrix, volume, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Cell:
    """
    Dataclass for crystallographic unit cells with cached derived quantities.

    Stores 6 parameters: [a, b, c, alpha, beta, gamma]
    - a, b, c: cell lengths in Angstroms
    - alpha, beta, gamma: cell angles in degrees

    Derived quantities (fractional_matrix, volume, etc.) are computed on first
    access and cached. The cache is cleared when the cell is moved to a
    different device or dtype.

    Examples
    --------
    >>> cell = Cell([50, 60, 70, 90, 90, 90])
    >>> cell.volume  # Computed and cached
    tensor(210000.)
    >>> cell_gpu = cell.to('cuda')  # Move to GPU (returns new Cell)
    >>> cell_gpu.device.type
    'cuda'
    """

    _data: torch.Tensor
    _cache: dict = field(default_factory=dict, repr=False)

    def __init__(
        self,
        data: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        requires_grad: bool = False,
    ) -> None:
        """
        Create a new Cell.

        Parameters
        ----------
        data : array-like
            Unit cell parameters [a, b, c, alpha, beta, gamma].
            Can be a list, numpy array, or torch tensor.
        dtype : torch.dtype, optional
            Desired data type. Defaults to torch.float32.
        device : torch.device or str, optional
            Desired device. Defaults to CPU.
        requires_grad : bool, optional
            Whether to track gradients. Defaults to False.

        Raises
        ------
        ValueError
            If data does not have exactly 6 elements.
        """
        if dtype is None:
            dtype = torch.float32

        # Convert to tensor first to get shape
        if isinstance(data, torch.Tensor):
            tensor = data.to(dtype=dtype, device=device)
        else:
            tensor = torch.tensor(data, dtype=dtype, device=device)

        # Validate shape
        if tensor.numel() != 6:
            raise ValueError(
                f"Cell requires exactly 6 elements [a, b, c, alpha, beta, gamma], "
                f"got {tensor.numel()}"
            )

        # Ensure 1D shape
        tensor = tensor.reshape(6)

        if requires_grad:
            tensor = tensor.requires_grad_(True)

        object.__setattr__(self, "_data", tensor)
        object.__setattr__(self, "_cache", {})

    # =========================================================================
    # Device/dtype movement methods
    # =========================================================================

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Cell":
        """
        Move Cell to specified device and/or dtype.

        Returns a new Cell with the data on the target device/dtype.
        The cache is cleared in the new Cell.

        Parameters
        ----------
        device : torch.device or str, optional
            Target device.
        dtype : torch.dtype, optional
            Target data type.

        Returns
        -------
        Cell
            New Cell on the target device/dtype.
        """
        new_data = self._data.to(device=device, dtype=dtype)
        new_cell = Cell.__new__(Cell)
        object.__setattr__(new_cell, "_data", new_data)
        object.__setattr__(new_cell, "_cache", {})
        return new_cell

    def cpu(self) -> "Cell":
        """
        Move Cell to CPU.

        Returns
        -------
        Cell
            New Cell on CPU.
        """
        return self.to(device="cpu")

    def cuda(self, device: int | None = None) -> "Cell":
        """
        Move Cell to CUDA device.

        Parameters
        ----------
        device : int, optional
            CUDA device index. If None, uses current CUDA device.

        Returns
        -------
        Cell
            New Cell on CUDA.
        """
        if device is None:
            return self.to(device="cuda")
        return self.to(device=f"cuda:{device}")

    def detach(self) -> "Cell":
        """
        Return a new Cell with detached tensor (no gradient tracking).

        Returns
        -------
        Cell
            New Cell with detached data.
        """
        new_data = self._data.detach()
        new_cell = Cell.__new__(Cell)
        object.__setattr__(new_cell, "_data", new_data)
        object.__setattr__(new_cell, "_cache", {})
        return new_cell

    def clone(self) -> "Cell":
        """
        Return a new Cell with cloned tensor data.

        Returns
        -------
        Cell
            New Cell with cloned data.
        """
        new_data = self._data.clone()
        new_cell = Cell.__new__(Cell)
        object.__setattr__(new_cell, "_data", new_data)
        object.__setattr__(new_cell, "_cache", {})
        return new_cell

    # =========================================================================
    # Basic properties
    # =========================================================================

    @property
    def device(self) -> torch.device:
        """Return the device of the underlying tensor."""
        return self._data.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the underlying tensor."""
        return self._data.dtype

    @property
    def data(self) -> torch.Tensor:
        """Return the underlying tensor (for buffer registration)."""
        return self._data

    @property
    def requires_grad(self) -> bool:
        """Return whether gradients are tracked."""
        return self._data.requires_grad

    # =========================================================================
    # Convenience properties for cell parameters
    # =========================================================================

    @property
    def a(self) -> torch.Tensor:
        """Cell length a in Angstroms."""
        return self._data[0]

    @property
    def b(self) -> torch.Tensor:
        """Cell length b in Angstroms."""
        return self._data[1]

    @property
    def c(self) -> torch.Tensor:
        """Cell length c in Angstroms."""
        return self._data[2]

    @property
    def alpha(self) -> torch.Tensor:
        """Cell angle alpha in degrees."""
        return self._data[3]

    @property
    def beta(self) -> torch.Tensor:
        """Cell angle beta in degrees."""
        return self._data[4]

    @property
    def gamma(self) -> torch.Tensor:
        """Cell angle gamma in degrees."""
        return self._data[5]

    # =========================================================================
    # Cached derived properties
    # =========================================================================

    @property
    def fractional_matrix(self) -> torch.Tensor:
        """
        Orthogonalization matrix B (fractional -> Cartesian).

        Returns the 3x3 matrix B such that: cart = frac @ B.T

        Returns
        -------
        torch.Tensor
            Shape (3, 3) orthogonalization matrix.
        """
        if "fractional_matrix" not in self._cache:
            self._cache["fractional_matrix"] = self._compute_fractional_matrix()
        return self._cache["fractional_matrix"]

    @property
    def inv_fractional_matrix(self) -> torch.Tensor:
        """
        Fractionalization matrix B^-1 (Cartesian -> fractional).

        Returns the 3x3 matrix B^-1 such that: frac = cart @ B^-1.T

        Returns
        -------
        torch.Tensor
            Shape (3, 3) fractionalization matrix.
        """
        if "inv_fractional_matrix" not in self._cache:
            self._cache["inv_fractional_matrix"] = torch.linalg.inv(
                self.fractional_matrix
            )
        return self._cache["inv_fractional_matrix"]

    @property
    def volume(self) -> torch.Tensor:
        """
        Unit cell volume in cubic Angstroms.

        Returns
        -------
        torch.Tensor
            Scalar tensor with the cell volume.
        """
        if "volume" not in self._cache:
            self._cache["volume"] = self._compute_volume()
        return self._cache["volume"]

    @property
    def reciprocal_basis_matrix(self) -> torch.Tensor:
        """
        Reciprocal basis matrix with [a*, b*, c*] as rows.

        Returns
        -------
        torch.Tensor
            Shape (3, 3) matrix where rows are the reciprocal basis vectors.
        """
        if "reciprocal_basis_matrix" not in self._cache:
            self._cache["reciprocal_basis_matrix"] = (
                self._compute_reciprocal_basis_matrix()
            )
        return self._cache["reciprocal_basis_matrix"]

    # =========================================================================
    # Internal computation methods
    # =========================================================================

    def _compute_fractional_matrix(self) -> torch.Tensor:
        """
        Compute the fractional-to-Cartesian transformation matrix.

        Delegates to math_numpy.get_fractional_matrix for the computation.
        """
        from torchref.base import math_torch

        return math_torch.get_fractional_matrix(self._data)

    def _compute_volume(self) -> torch.Tensor:
        """
        Compute the unit cell volume.

        Uses the formula: V = abc * sqrt(1 - cos^2(alpha) - cos^2(beta) - cos^2(gamma)
                                         + 2*cos(alpha)*cos(beta)*cos(gamma))
        """
        a, b, c = self._data[0], self._data[1], self._data[2]
        angles_rad = torch.deg2rad(self._data[3:])
        cos_alpha, cos_beta, cos_gamma = torch.cos(angles_rad)

        volume_factor = torch.sqrt(
            1
            - cos_alpha**2
            - cos_beta**2
            - cos_gamma**2
            + 2 * cos_alpha * cos_beta * cos_gamma
        )

        return a * b * c * volume_factor

    def _compute_reciprocal_basis_matrix(self) -> torch.Tensor:
        """
        Compute the reciprocal space basis matrix.

        Delegates to math_torch.reciprocal_basis_matrix for the computation.
        """
        from torchref.base import math_torch

        return math_torch.reciprocal_basis_matrix(self._data)

    # =========================================================================
    # Grid computation methods
    # =========================================================================

    def compute_grid_size(self, max_res: float, oversampling: float = 3.0) -> tuple:
        """
        Compute minimum grid dimensions for a given resolution.

        Uses Shannon-Nyquist sampling criterion to determine the minimum
        number of grid points needed along each axis.

        Parameters
        ----------
        max_res : float
            Maximum resolution in Angstroms.
        oversampling : float, optional
            Oversampling factor relative to max_res. Default is 3.0
            (standard for crystallographic calculations).

        Returns
        -------
        tuple of int
            Minimum grid dimensions (nx, ny, nz).

        Examples
        --------
        >>> cell = Cell([50, 60, 70, 90, 90, 90])
        >>> cell.compute_grid_size(2.0)
        (75, 90, 105)
        """
        import math

        a, b, c = self.a.item(), self.b.item(), self.c.item()

        # Shannon-Nyquist: sample at oversampling × the maximum frequency
        nx = int(math.floor(a / max_res * oversampling))
        ny = int(math.floor(b / max_res * oversampling))
        nz = int(math.floor(c / max_res * oversampling))

        return (nx, ny, nz)

    def tolist(self) -> list:
        """
        Convert Cell parameters to a standard Python list.

        Returns
        -------
        list
            List of cell parameters [a, b, c, alpha, beta, gamma].
        """
        return self._data.tolist()

    # =========================================================================
    # Dunder methods
    # =========================================================================

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Cell([{self.a.item():.4f}, {self.b.item():.4f}, {self.c.item():.4f}, "
            f"{self.alpha.item():.4f}, {self.beta.item():.4f}, {self.gamma.item():.4f}], "
            f"device={self.device}, dtype={self.dtype})"
        )

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Allow indexing like cell[0] for cell length a."""
        return self._data[idx]

    def __len__(self) -> int:
        """Return 6 (number of cell parameters)."""
        return 6


# Keep CellTensor as an alias for backward compatibility
CellTensor = Cell
