"""Centralized configuration for TorchRef.

Set at import time from the environment -- ``TORCHREF_DTYPE_FLOAT`` (float32 default /
float64), ``TORCHREF_DTYPE_INT`` (int32 / int64), ``TORCHREF_DTYPE_COMPLEX``
(complex64 / complex128), ``TORCHREF_DEVICE`` ('auto' default / 'cuda' / 'mps' / 'cpu'),
``TORCHREF_SIGMA_CUTOFF_ED`` (3.0), ``TORCHREF_COMPILE_TARGETS`` and ``TORCHREF_CACHING``
(on by default) -- or at runtime by attribute assignment::

    torchref.dtypes.float = torch.float64
    torchref.device.current = torch.device('cpu')
    torchref.sigma_cutoff_ed.value = 4.0   # density splat truncation, in sigmas
    torchref.config.caching.value = False  # recompute every cached forward()

The default device is auto-detected cuda -> mps -> cpu, and a CUDA device is picked only
if its compute capability is >= the minimum sm_* in this PyTorch build *and* its VRAM is
>= ``_MIN_CUDA_VRAM_GB``; otherwise auto-detection falls back with a warning naming the
failing requirement. An explicit ``TORCHREF_DEVICE`` bypasses those gates but still fails
fast if the backend is unavailable.

**MPS supports neither float64 nor complex128.** Resolving to MPS with float64
configured warns at import; set ``TORCHREF_DTYPE_FLOAT=float32`` or
``TORCHREF_DEVICE=cpu``.
"""

import os
import warnings

import torch

# ---------------------------------------------------------------------------
# Grid sampling
# ---------------------------------------------------------------------------
# Shannon-Nyquist oversampling factor used to size real-space / FFT grids from
# a unit cell and resolution: the number of grid points along an axis of length
# ``a`` at resolution ``d_min`` is ``floor(a / d_min * NYQUIST_OVERSAMPLING)``,
# i.e. a grid spacing of ``d_min / NYQUIST_OVERSAMPLING``.
#
# A value of 2.0 is the bare Nyquist limit, but electron density built from
# Gaussian atoms is not strictly band-limited, so sampling at exactly 2.0
# introduces aliasing/interpolation error in the density->F_calc transform that
# measurably degrades refinement convergence, with high-res / large-cell
# structures hit hardest. 3.0 is the standard
# crystallographic oversampling (matches gemmi's default sample_rate) and
# restores accuracy. All grid-sizing helpers reference this single constant so
# the real-space map grids and the FFT structure-factor grids stay consistent.
NYQUIST_OVERSAMPLING = 3.0

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
    """Dtype configuration by attribute: ``dtypes.float``, ``.int``, ``.complex``,
    readable and assignable (``dtypes.float = torch.float64``).
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
# Electron-density splat cutoff
# ---------------------------------------------------------------------------
# Number of sigmas at which each atom's Gaussian density is truncated. The
# per-atom real-space splat radius is r_i = clamp(ceil_0.25(N_sigma * sigma_eff_i),
# [2, 7] A), with sigma_eff_i = sqrt((b_form_i + B_i) / 8pi^2). Because the
# truncation is expressed in sigmas, every atom carries the same fractional tail
# mass regardless of its B-factor (3 sigma -> ~0.4%, 3.5 sigma -> ~0.09%,
# 4 sigma -> ~0.013% per-axis tail), so this single knob governs the structure-wide
# F-truncation residual. It replaces the old per-structure scalar ``radius_angstrom``.
#
# Default 3.0: an N_sigma sweep vs the direct-summation oracle (1DAW/3GR5/4BX9/7L84/
# 5BOV, 1.6-2.6 A) showed the F-residual at 3.0 is identical to 3.5 for 4/5 cases and
# only 1.0e-4 vs 3.3e-5 on the most demanding (4BX9) -- negligible against the ~1e-3
# floor from grid sampling -- while using ~33% fewer splat voxels. 2.5 is too tight
# (4BX9 degrades to 6.8e-4, 20x worse), so 3.0 is the floor.
_DEFAULT_SIGMA_CUTOFF_ED = 3.0


class SigmaCutoffConfig:
    """Number of sigmas at which the per-atom electron-density Gaussian is truncated.

    ``sigma_cutoff_ed.value`` reads or sets it; initialised from
    ``TORCHREF_SIGMA_CUTOFF_ED`` (default 3.0). Must be positive.
    """

    def __init__(self):
        raw = os.environ.get("TORCHREF_SIGMA_CUTOFF_ED")
        if raw is None:
            self._value = _DEFAULT_SIGMA_CUTOFF_ED
        else:
            try:
                self._value = float(raw)
            except ValueError:
                raise ValueError(
                    f"Invalid TORCHREF_SIGMA_CUTOFF_ED: {raw!r}. "
                    "Must be a positive number."
                )
            if self._value <= 0:
                raise ValueError(
                    f"Invalid TORCHREF_SIGMA_CUTOFF_ED: {raw!r}. "
                    "Must be a positive number."
                )

    @property
    def value(self) -> float:
        """Get the current sigma cutoff (number of sigmas)."""
        return self._value

    @value.setter
    def value(self, sigma: float) -> None:
        """Set the sigma cutoff for all future density builds."""
        if not isinstance(sigma, (int, float)) or isinstance(sigma, bool):
            raise TypeError(
                f"sigma_cutoff_ed must be a number, got {type(sigma).__name__}"
            )
        if sigma <= 0:
            raise ValueError(f"sigma_cutoff_ed must be positive, got {sigma}")
        self._value = float(sigma)

    def __repr__(self) -> str:
        return f"SigmaCutoffConfig(value={self._value})"


# Global singleton instance
sigma_cutoff_ed = SigmaCutoffConfig()


def get_sigma_cutoff_ed() -> float:
    """Get the current density-splat sigma cutoff (number of sigmas)."""
    return sigma_cutoff_ed.value


_DEFAULT_COMPILE_TARGETS = False


class CompileTargetsConfig:
    """Whether to ``torch.compile`` the quadrature X-ray target kernels.

    ``compile_targets.value`` reads or sets it; initialised from
    ``TORCHREF_COMPILE_TARGETS`` ("1"/"true"/"yes"/"on"). Applies to the full-form MLF
    target (``--xray-mode ml_full``), whose per-reflection fixed-node quadrature is bound by
    dispatch and memory traffic in eager mode, so fusing it is worth roughly an order of
    magnitude.

    **Off by default because of compile latency, which autograd dominates**: compiling the
    backward costs ~2 minutes on the first call, so a short refinement of a small structure
    gets slower, not faster. It pays off for big datasets (the target scales with reflection
    count) and for long or repeated runs in one process (ensembles, collection/PanDDA
    refinements, interactive sessions) where the compile amortises.

    Only the reflection-count dimension varies, so the kernels compile with ``dynamic=True``
    and **one** compilation serves every dataset size, work/free subset and gathered tensor
    -- no chunking or padding layer is needed. To cut latency point
    ``TORCHINDUCTOR_CACHE_DIR`` at node-local disk (never gpfs) so codegen is reused across
    processes; artifacts are ~22 MB.

    **Keep it off for float64 and gradient-verification work regardless**: that path is the
    eager reference and is deliberately unfused.
    """

    def __init__(self):
        raw = os.environ.get("TORCHREF_COMPILE_TARGETS")
        if raw is None:
            self._value = _DEFAULT_COMPILE_TARGETS
        else:
            self._value = raw.strip().lower() in ("1", "true", "yes", "on")

    @property
    def value(self) -> bool:
        """Whether quadrature target kernels are ``torch.compile``d."""
        return self._value

    @value.setter
    def value(self, flag: bool) -> None:
        if not isinstance(flag, bool):
            raise TypeError(
                f"compile_targets must be a bool, got {type(flag).__name__}"
            )
        self._value = flag

    def __repr__(self) -> str:
        return f"CompileTargetsConfig(value={self._value})"


compile_targets = CompileTargetsConfig()


def get_compile_targets() -> bool:
    """Whether quadrature X-ray target kernels should be ``torch.compile``d."""
    return compile_targets.value


# ---------------------------------------------------------------------------
# Forward-result caching
# ---------------------------------------------------------------------------

_TRUE_STRINGS = ("1", "true", "yes", "on")
_FALSE_STRINGS = ("0", "false", "no", "off")


def _parse_bool_env(name: str, raw: str) -> bool:
    """Parse a boolean environment variable, rejecting anything unrecognised.

    Stricter than the ``in ("1", "true", ...)`` idiom used for
    ``TORCHREF_COMPILE_TARGETS``, which is safe only for a default-off flag: for a
    default-on one it would turn a typo into a silent opt-out.
    """
    value = raw.strip().lower()
    if value in _TRUE_STRINGS:
        return True
    if value in _FALSE_STRINGS:
        return False
    raise ValueError(
        f"Invalid {name}: {raw!r}. "
        f"Valid values: {list(_TRUE_STRINGS + _FALSE_STRINGS)}"
    )


_DEFAULT_CACHING = True


class CachingConfig:
    """Whether :class:`torchref.utils.CachedForwardMixin` serves cached results.

    ``caching.value`` reads or sets it; initialised from ``TORCHREF_CACHING``
    ("1"/"true"/"yes"/"on" vs "0"/"false"/"no"/"off"), on by default. Turning it off makes
    the mixin inert: every module call runs ``forward()`` again, so ``ModelFT``,
    ``MixedTensor``, ``RigidXYZTensor`` and their subclasses recompute structure factors and
    parameter transforms from scratch on each access. The numbers are unchanged -- only the
    work done to get them.

    Gates that mixin **only**. The other hand-rolled caches (``Cell._cache``,
    ``SigmaAEstimator._cache``, the bulk-solvent and parity caches, the reflection-data
    subset views, the symmetry-extractor rebuild in ``sf_fft``) are unaffected.

    Intended for diagnosis rather than production: if refinement produces stale-looking
    numbers, rerunning with ``TORCHREF_CACHING=0`` says in one step whether the forward cache
    is responsible. Also useful as an eager reference when changing the mixin's fingerprinting.
    Use :func:`torchref.utils.no_caching` to scope the change to a block.
    """

    def __init__(self):
        raw = os.environ.get("TORCHREF_CACHING")
        if raw is None:
            self._value = _DEFAULT_CACHING
        else:
            self._value = _parse_bool_env("TORCHREF_CACHING", raw)

    @property
    def value(self) -> bool:
        """Whether cached ``forward()`` results are served."""
        return self._value

    @value.setter
    def value(self, flag: bool) -> None:
        if not isinstance(flag, bool):
            raise TypeError(f"caching must be a bool, got {type(flag).__name__}")
        self._value = flag

    def __repr__(self) -> str:
        return f"CachingConfig(value={self._value})"


caching = CachingConfig()


def get_caching_enabled() -> bool:
    """Whether ``CachedForwardMixin`` should serve cached ``forward()`` results."""
    return caching.value


# ---------------------------------------------------------------------------
# Device configuration
# ---------------------------------------------------------------------------

_VALID_DEVICE_TYPES = ("cuda", "mps", "cpu")

# Minimum GPU VRAM (in GB) required for CUDA to be picked by auto-detection.
# Smaller GPUs typically can't fit useful refinement workloads and surprise
# users with OOMs, so we fall back to CPU instead.
_MIN_CUDA_VRAM_GB = 10


def _cuda_is_usable() -> bool:
    """True iff a visible CUDA device is fit to auto-select as the default.

    Requires ``torch.cuda.is_available()``, a compute capability >= the minimum sm_* in this
    PyTorch wheel (older GPUs fail at the first kernel launch), and VRAM >=
    ``_MIN_CUDA_VRAM_GB`` (a too-small GPU OOMs on real refinements, so CPU is preferred).
    If ``get_arch_list`` is missing or empty the capability check is skipped and
    ``is_available()`` trusted. On failure one warning names the requirement missed.
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


def canonical_device(dev):
    """Return ``dev`` with its default index filled in (``None`` passes through).

    ``torch.device('cuda') != torch.device('cuda:0')`` although both name one physical
    device, and a device read off a real tensor always carries an index -- so without a
    shared normal form ``obj.device == tensor.device`` is False on a fresh object and True
    after its first ``.to()``. ``cpu`` deliberately stays bare, since a CPU tensor's device
    has no index and ``cpu:0`` would recreate the mismatch.

    This is the allocation-free form of ``torch.empty(0, device=d).device``, which matters
    on constructor and comparison paths. Deliberately **not** memoised:
    ``torch.cuda.current_device()`` is mutable via ``set_device``, so a cache would freeze
    a stale answer.
    """
    if dev is None:
        return None
    d = dev if isinstance(dev, torch.device) else torch.device(dev)
    if d.index is not None:
        return d
    if d.type == "cpu":
        return d
    if d.type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    if d.type == "mps":
        return torch.device("mps", 0)
    # Unknown/future backend: fall back to asking PyTorch directly.
    return torch.empty(0, device=d).device


def normalize_device(dev=None) -> torch.device:
    """Coerce a user-supplied ``device`` (or ``None``) to a canonical device.

    ``None`` resolves to :func:`get_default_device`. The pure, side-effect-free counterpart
    of :func:`torchref.utils.resolve_device`, which *moves* the objects it is given: use
    this for one device source, that one to reconcile several.
    """
    if dev is None:
        return get_default_device()
    return canonical_device(dev)


class DeviceConfig:
    """The active device: ``device.current`` reads it, assignment sets it.

    Resolved once at import cuda -> mps -> cpu by :func:`_auto_detect_device`, which gates
    CUDA on compute capability and ``_MIN_CUDA_VRAM_GB`` of VRAM; ``TORCHREF_DEVICE``
    overrides that, bypassing the gates but raising if the backend is unavailable. The
    setter mirrors it -- a bad value raises ``ValueError``/``RuntimeError`` rather than
    silently falling back, so callers can decide how to recover.
    """

    def __init__(self):
        override = os.environ.get("TORCHREF_DEVICE", "auto").lower()
        if override == "auto":
            # Canonicalise here too, not just in ``_coerce``: the ``auto``
            # branch bypasses ``_coerce`` entirely, and it is the branch almost
            # every user takes.
            self._device = canonical_device(_auto_detect_device())
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
        return canonical_device(dev)

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


# ---------------------------------------------------------------------------
# Double-precision availability
# ---------------------------------------------------------------------------
#: Device types with no float64 at all. MPS is the live case and it *raises*
#: rather than quietly downcasting, so a float64 tensor there is an error and not
#: merely slow.
_NO_DOUBLE_DEVICE_TYPES = ("mps",)


def supports_double(dev=None) -> bool:
    """Whether ``dev`` can hold float64 / complex128 at all."""
    return normalize_device(dev).type not in _NO_DOUBLE_DEVICE_TYPES


def widest_float_dtype(dev=None) -> torch.dtype:
    """``float64`` where the device has it, else the configured working float.

    For computations whose *precision* is load-bearing rather than their storage:
    accumulating single-precision data in double is the ordinary remedy, and the
    dynamic range of an unnormalised recurrence is a hard requirement rather than
    a preference. The right width for those is a property of the device, so it
    belongs here and not in a constant at the call site.

    Where the device lacks float64 the caller gets the working dtype and whatever
    accuracy that implies. That is the only option there, not a choice -- callers
    that care should say what it costs in their own docstring.
    """
    return torch.float64 if supports_double(dev) else get_float_dtype()


def widest_complex_dtype(dev=None) -> torch.dtype:
    """``complex128`` where the device has it, else the configured working complex."""
    return torch.complex128 if supports_double(dev) else get_complex_dtype()
