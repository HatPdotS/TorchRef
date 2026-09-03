"""
Cell - A dataclass for crystallographic unit cells with cached derived quantities.

Provides a simple container for unit cell parameters with automatic caching
of derived quantities (fractional matrix, volume, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from torchref.config import (
    NYQUIST_OVERSAMPLING,
    get_float_dtype,
    normalize_device,
)
from torchref.utils.device_mixin import _NonModuleDeviceMixin


@dataclass(eq=False)
class Cell(_NonModuleDeviceMixin):
    """
    Dataclass for crystallographic unit cells with cached derived quantities.

    Stores 6 parameters: [a, b, c, alpha, beta, gamma]
    - a, b, c: cell lengths in Angstroms
    - alpha, beta, gamma: cell angles in degrees

    Derived quantities (fractional_matrix, volume, etc.) are computed on first
    access and cached. The cache is cleared when the cell is moved to a
    different device or dtype.

    Two cells compare and hash equal when their six parameters are equal
    (:attr:`key`), independent of device and dtype, so a cell can key a dict or
    a cache of quantities derived from it. A cell is therefore a value: editing
    its parameter tensor in place is refused at the next derived read. Build a
    new ``Cell`` and assign it instead.

    Examples
    --------
    >>> cell = Cell([50, 60, 70, 90, 90, 90])
    >>> float(cell.volume)  # computed and cached
    210000.0
    >>> cell_gpu = cell.to('cuda')  # move in place; returns self  # doctest: +SKIP
    >>> cell_gpu.device.type  # doctest: +SKIP
    'cuda'
    """

    _data: torch.Tensor
    _cache: dict = field(default_factory=dict, repr=False)
    _stamp: tuple = field(default=None, repr=False)

    def __init__(
        self,
        data: Any,
        *,
        dtype: torch.dtype = None,
        device: torch.device | str = None,
    ) -> None:
        """
        Create a new Cell.

        Parameters
        ----------
        data : array-like
            Unit cell parameters [a, b, c, alpha, beta, gamma].
            Can be a list, numpy array, or torch tensor.
        dtype : torch.dtype, optional
            Desired data type. Defaults to the configured ``dtypes.float``.
        device : torch.device or str, optional
            Desired device. Defaults to the configured ``device.current``.

        Raises
        ------
        ValueError
            If data does not have exactly 6 elements.
        """
        if dtype is None:
            dtype = get_float_dtype()
        device = normalize_device(device)
        # Convert to tensor first to get shape
        if isinstance(data, torch.Tensor):
            tensor = data.to(dtype=dtype, device=device)
            if tensor is data:
                # ``to`` returned the caller's tensor unchanged; own a copy so
                # their later edits cannot reach into this cell.
                tensor = tensor.clone()
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

        object.__setattr__(self, "_data", tensor)
        object.__setattr__(self, "_cache", {})
        self._stamp_data()

    # =========================================================================
    # Device/dtype movement methods
    # =========================================================================

    # ``to``, ``cuda``, ``cpu`` are inherited from ``_NonModuleDeviceMixin``
    # and operate in place — they walk ``self.__dict__`` (moving ``_data``
    # and any cached tensor values) and then call ``reset_cache`` below.

    def reset_cache(self) -> None:
        """Clear cached derived quantities (fractional matrix, volume, etc.).

        Also re-stamps the parameter tensor, so a device or dtype move (which
        rebinds it and then calls this) is not mistaken for an in-place edit.
        """
        object.__setattr__(self, "_cache", {})
        self._stamp_data()

    def _stamp_data(self) -> None:
        object.__setattr__(self, "_stamp", (id(self._data), self._data._version))

    def _assert_unmodified(self) -> None:
        """Refuse to serve derived quantities from a tensor edited in place."""
        stamp = getattr(self, "_stamp", None)
        if stamp is None:
            self._stamp_data()
            return
        if (id(self._data), self._data._version) == stamp:
            return
        cached = self._cache.get("key")
        held = f" It held {cached}." if cached is not None else ""
        raise RuntimeError(
            "This Cell was edited in place after it was built." + held + " Cells are "
            "values shared by reference (model context, structure-factor engine, "
            "scaler), so an in-place edit changes the crystal under every holder. "
            "Please don't edit Cell objects, create a new one -- "
            "Cell([a, b, c, alpha, beta, gamma], dtype=cell.dtype, device=cell.device) "
            "-- and assign it, e.g. model.cell = new_cell."
        )

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
        new_cell._stamp_data()
        return new_cell

    # =========================================================================
    # Value identity
    # =========================================================================

    @property
    def key(self) -> tuple:
        """The six parameters as a tuple of Python floats.

        Read off the tensor once and cached alongside the derived quantities, so
        the device synchronisation happens once per construction or
        :meth:`reset_cache` rather than on every comparison.

        Returns
        -------
        tuple of float
            ``(a, b, c, alpha, beta, gamma)``.
        """
        self._assert_unmodified()
        key = self._cache.get("key")
        if key is None:
            key = tuple(float(v) for v in self._data.tolist())
            self._cache["key"] = key
        return key

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Cell):
            return NotImplemented
        return self.key == other.key

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

        Note: the property name ``fractional_matrix`` is historical; the matrix
        it returns is the orthogonalization (de-fractionalizing) matrix.

        Returns
        -------
        torch.Tensor
            Shape (3, 3) orthogonalization matrix.
        """
        self._assert_unmodified()
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
        self._assert_unmodified()
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
        self._assert_unmodified()
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
        self._assert_unmodified()
        if "reciprocal_basis_matrix" not in self._cache:
            self._cache["reciprocal_basis_matrix"] = (
                self._compute_reciprocal_basis_matrix()
            )
        return self._cache["reciprocal_basis_matrix"]

    # =========================================================================
    # Internal computation methods
    # =========================================================================

    def _compute_fractional_matrix(self) -> torch.Tensor:
        """Fractional-to-Cartesian matrix, via ``math_torch.get_fractional_matrix``."""
        from torchref.base import math_torch

        return math_torch.get_fractional_matrix(self._data)

    def _compute_volume(self) -> torch.Tensor:
        """V = abc·sqrt(1 - Σcos²angle + 2·cosα·cosβ·cosγ)."""
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
        """Reciprocal basis, via ``math_torch.reciprocal_basis_matrix``."""
        from torchref.base import math_torch

        return math_torch.reciprocal_basis_matrix(self._data)

    # =========================================================================
    # Grid computation methods
    # =========================================================================

    def compute_grid_size(
        self, max_res: float, oversampling: float = NYQUIST_OVERSAMPLING
    ) -> tuple:
        """
        Minimum Shannon-Nyquist grid dimensions for a given resolution.

        Parameters
        ----------
        max_res : float
            Maximum resolution in Angstroms.
        oversampling : float, optional
            Factor relative to max_res. Defaults to
            :data:`torchref.config.NYQUIST_OVERSAMPLING`, shared by every
            grid-sizing helper.

        Returns
        -------
        tuple of int
            Minimum grid dimensions (nx, ny, nz).

        Examples
        --------
        >>> cell = Cell([50, 60, 70, 90, 90, 90])
        >>> cell.compute_grid_size(2.0)  # at the default oversampling of 3.0
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
    # Fractional/Cartesian conversion methods
    # =========================================================================

    def fractional_to_cartesian(self, frac_coords: torch.Tensor) -> torch.Tensor:
        """
        Convert fractional coordinates to Cartesian coordinates.

        Parameters
        ----------
        frac_coords : torch.Tensor
            Tensor of fractional coordinates, shape (..., 3).

        Returns
        -------
        torch.Tensor
            Tensor of Cartesian coordinates, shape (..., 3).
        """
        return torch.matmul(frac_coords, self.fractional_matrix.T)

    def cartesian_to_fractional(self, cart_coords: torch.Tensor) -> torch.Tensor:
        """
        Convert Cartesian coordinates to fractional coordinates.

        Parameters
        ----------
        cart_coords : torch.Tensor
            Tensor of Cartesian coordinates, shape (..., 3).

        Returns
        -------
        torch.Tensor
            Tensor of fractional coordinates, shape (..., 3).
        """
        return torch.matmul(cart_coords, self.inv_fractional_matrix.T)

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
