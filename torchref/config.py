"""
Centralized configuration for TorchRef.

Default dtypes can be set via environment variables at import time:
- TORCHREF_DTYPE_FLOAT: float32 (default) or float64
- TORCHREF_DTYPE_INT: int32 (default) or int64
- TORCHREF_DTYPE_COMPLEX: complex64 (default) or complex128

Default device is auto-detected at import time using cuda -> mps -> cpu.
A CUDA device is only picked automatically if it satisfies *both*:
  * compute capability >= the minimum sm_* compiled into the current
    PyTorch build (``torch.cuda.get_arch_list()``), and
  * total VRAM >= ``_MIN_CUDA_VRAM_GB`` (10 GB).
Otherwise auto-detection falls back to MPS or CPU with a warning that
names the failing requirement. Override the resolved device with the
TORCHREF_DEVICE environment variable ('auto' (default), 'cuda', 'mps',
'cpu'); an explicit value bypasses the capability/VRAM gates but still
fails fast if the requested backend is unavailable on this host.

Users can also change dtypes/device at runtime via attribute assignment:
    import torchref
    torchref.dtypes.float = torch.float64
    torchref.device.current = torch.device('cpu')

Or read current values:
    torchref.dtypes.float   # torch.float32
    torchref.device.current # torch.device('cuda')

MPS caveat: Apple's MPS backend does not support float64 / complex128. If
the resolved device is MPS and the configured float dtype is float64, a
warning is emitted at import time. Either set TORCHREF_DTYPE_FLOAT=float32
or TORCHREF_DEVICE=cpu to silence it.
"""

import os
import warnings

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


# ---------------------------------------------------------------------------
# Device configuration
# ---------------------------------------------------------------------------

_VALID_DEVICE_TYPES = ("cuda", "mps", "cpu")

# Minimum GPU VRAM (in GB) required for CUDA to be picked by auto-detection.
# Smaller GPUs typically can't fit useful refinement workloads and surprise
# users with OOMs, so we fall back to CPU instead.
_MIN_CUDA_VRAM_GB = 10


def _cuda_is_usable() -> bool:
    """Return True iff at least one visible CUDA device is suitable for
    auto-selection as the default TorchRef device.

    A device qualifies when all of the following hold:

    * ``torch.cuda.is_available()`` is True.
    * Its compute capability is >= the minimum sm_* compiled into the
      current PyTorch wheel (introspected via ``torch.cuda.get_arch_list()``).
      Older GPUs would trigger runtime warnings and fail at the first
      kernel launch.
    * Its total VRAM is >= ``_MIN_CUDA_VRAM_GB``. Smaller GPUs typically
      cannot fit useful refinement workloads and tend to surprise users
      with OOMs, so we prefer CPU over a too-small GPU.

    If introspection fails on an older torch build (no ``get_arch_list``)
    or the arch list is empty, we trust ``is_available()`` and return True
    without the capability check. On failure a single ``warnings.warn``
    explains which requirement was missed before auto-detection falls
    through to MPS or CPU.
    """
    if not torch.cuda.is_available():
        return False
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        # If we cannot introspect supported archs, fall back to trusting
        # is_available() (older torch versions).
        return True
    if not arch_list:
        return True
    # Parse e.g. "sm_70" -> (7, 0). Ignore non-sm entries like "compute_xx".
    supported = []
    for entry in arch_list:
        if not entry.startswith("sm_"):
            continue
        try:
            num = entry[3:]
            major = int(num[:-1])
            minor = int(num[-1])
            supported.append((major, minor))
        except (ValueError, IndexError):
            continue
    if not supported:
        return True
    min_supported = min(supported)
    min_vram_bytes = _MIN_CUDA_VRAM_GB * (1024**3)
    for idx in range(torch.cuda.device_count()):
        try:
            cap = torch.cuda.get_device_capability(idx)
        except Exception:
            continue
        if cap < min_supported:
            continue
        try:
            total_mem = torch.cuda.get_device_properties(idx).total_memory
        except Exception:
            total_mem = 0
        if total_mem >= min_vram_bytes:
            return True
    warnings.warn(
        "TorchRef: no detected CUDA GPU meets the auto-selection requirements "
        f"(compute capability >= {min_supported[0]}.{min_supported[1]} and "
        f">= {_MIN_CUDA_VRAM_GB} GB VRAM; PyTorch build supports sm_*: "
        f"{arch_list}). Falling back to CPU. Set TORCHREF_DEVICE=cuda "
        "explicitly to override.",
        stacklevel=3,
    )
    return False


def _auto_detect_device() -> torch.device:
    """Pick the best available device: cuda -> mps -> cpu."""
    if _cuda_is_usable():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DeviceConfig:
    """
    Device configuration with property-based access.

    Resolved once at import time using cuda -> mps -> cpu via
    :func:`_auto_detect_device`, which gates CUDA on both compute
    capability and a minimum of ``_MIN_CUDA_VRAM_GB`` of VRAM. Override
    the resolved default via the ``TORCHREF_DEVICE`` environment variable
    (``'auto'`` (default), ``'cuda'``, ``'mps'``, or ``'cpu'``); explicit
    values bypass the auto-selection gates and instead raise if the
    requested backend is unavailable on this host.

        device.current              # get the active device
        device.current = "cpu"      # set at runtime (string or torch.device)

    Setter behaviour mirrors the env-var override: a bad value raises
    ``ValueError`` / ``RuntimeError`` rather than silently falling back,
    so callers can decide how to recover.
    """

    def __init__(self):
        override = os.environ.get("TORCHREF_DEVICE", "auto").lower()
        if override == "auto":
            self._device = _auto_detect_device()
        else:
            self._device = self._coerce(override)
        self._warn_if_mps_dtype_mismatch()

    @staticmethod
    def _coerce(value) -> torch.device:
        """Validate and convert a user-supplied device value."""
        if isinstance(value, torch.device):
            dev = value
        elif isinstance(value, str):
            dev = torch.device(value)
        else:
            raise TypeError(
                f"device must be a torch.device or string, got {type(value).__name__}"
            )
        if dev.type not in _VALID_DEVICE_TYPES:
            raise ValueError(
                f"Invalid device type: {dev.type!r}. "
                f"Valid types: {_VALID_DEVICE_TYPES}"
            )
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this system.")
        if dev.type == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS requested but not available on this system.")
        return dev

    def _warn_if_mps_dtype_mismatch(self) -> None:
        if self._device.type == "mps" and dtypes.float == torch.float64:
            warnings.warn(
                "TorchRef default device is MPS but dtypes.float is float64; "
                "MPS does not support float64. Set TORCHREF_DTYPE_FLOAT=float32 "
                "or TORCHREF_DEVICE=cpu to silence this warning.",
                stacklevel=2,
            )

    @property
    def current(self) -> torch.device:
        """Get the current default device."""
        return self._device

    @current.setter
    def current(self, value) -> None:
        """Set the default device for all future operations."""
        self._device = self._coerce(value)
        self._warn_if_mps_dtype_mismatch()

    def __repr__(self) -> str:
        return f"DeviceConfig(current={self._device})"


device = DeviceConfig()


def get_default_device() -> torch.device:
    """Get the current default device."""
    return device.current
