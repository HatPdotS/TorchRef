"""
Centralized configuration for TorchRef.

Default dtypes can be set via environment variables at import time:
- TORCHREF_DTYPE_FLOAT: float32 (default) or float64
- TORCHREF_DTYPE_INT: int32 (default) or int64
- TORCHREF_DTYPE_COMPLEX: complex64 (default) or complex128

Users can also change dtypes at runtime via attribute assignment:
    import torchref
    torchref.dtypes.float = torch.float64
    torchref.dtypes.int = torch.int64
    torchref.dtypes.complex = torch.complex128

Or read current values:
    torchref.dtypes.float   # torch.float32
    torchref.dtypes.int     # torch.int32
    torchref.dtypes.complex # torch.complex64
"""

import os

import torch

# Map strings to torch dtypes
_FLOAT_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}

_INT_DTYPE_MAP = {
    "int32": torch.int32,
    "int64": torch.int64,
}

_COMPLEX_DTYPE_MAP = {
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}


class DtypeConfig:
    """
    Dtype configuration with property-based access.

    Access dtypes as attributes:
        dtypes.float    # get current float dtype
        dtypes.int      # get current int dtype
        dtypes.complex  # get current complex dtype

    Set dtypes via assignment:
        dtypes.float = torch.float64
        dtypes.int = torch.int64
        dtypes.complex = torch.complex128
    """

    def __init__(self):
        # Parse environment variables with defaults
        float_str = os.environ.get("TORCHREF_DTYPE_FLOAT", "float32").lower()
        int_str = os.environ.get("TORCHREF_DTYPE_INT", "int32").lower()
        complex_str = os.environ.get("TORCHREF_DTYPE_COMPLEX", "complex64").lower()

        # Validate and set
        if float_str not in _FLOAT_DTYPE_MAP:
            raise ValueError(
                f"Invalid TORCHREF_DTYPE_FLOAT: {float_str}. "
                f"Valid values: {list(_FLOAT_DTYPE_MAP.keys())}"
            )
        if int_str not in _INT_DTYPE_MAP:
            raise ValueError(
                f"Invalid TORCHREF_DTYPE_INT: {int_str}. "
                f"Valid values: {list(_INT_DTYPE_MAP.keys())}"
            )
        if complex_str not in _COMPLEX_DTYPE_MAP:
            raise ValueError(
                f"Invalid TORCHREF_DTYPE_COMPLEX: {complex_str}. "
                f"Valid values: {list(_COMPLEX_DTYPE_MAP.keys())}"
            )

        self._float = _FLOAT_DTYPE_MAP[float_str]
        self._int = _INT_DTYPE_MAP[int_str]
        self._complex = _COMPLEX_DTYPE_MAP[complex_str]

    @property
    def float(self) -> torch.dtype:
        """Get the current default float dtype."""
        return self._float

    @float.setter
    def float(self, dtype: torch.dtype) -> None:
        """Set the default float dtype for all future operations."""
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(
                f"Invalid float dtype: {dtype}. Use torch.float32 or torch.float64."
            )
        self._float = dtype

    @property
    def int(self) -> torch.dtype:
        """Get the current default int dtype."""
        return self._int

    @int.setter
    def int(self, dtype: torch.dtype) -> None:
        """Set the default int dtype for all future operations."""
        if dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"Invalid int dtype: {dtype}. Use torch.int32 or torch.int64."
            )
        self._int = dtype

    @property
    def complex(self) -> torch.dtype:
        """Get the current default complex dtype."""
        return self._complex

    @complex.setter
    def complex(self, dtype: torch.dtype) -> None:
        """Set the default complex dtype for all future operations."""
        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError(
                f"Invalid complex dtype: {dtype}. Use torch.complex64 or torch.complex128."
            )
        self._complex = dtype

    def __repr__(self) -> str:
        return f"DtypeConfig(float={self._float}, int={self._int}, complex={self._complex})"


# Global singleton instance
dtypes = DtypeConfig()


# Convenience functions for internal use (avoid repeated attribute lookups)
def get_float_dtype() -> torch.dtype:
    """Get the current default float dtype."""
    return dtypes.float


def get_int_dtype() -> torch.dtype:
    """Get the current default int dtype."""
    return dtypes.int


def get_complex_dtype() -> torch.dtype:
    """Get the current default complex dtype."""
    return dtypes.complex
