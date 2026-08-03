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

Users can also change dtypes/device/density-cutoff at runtime via attribute
assignment:
    import torchref
    torchref.dtypes.float = torch.float64
    torchref.device.current = torch.device('cpu')
    torchref.sigma_cutoff_ed.value = 4.0   # density splat truncation (sigmas)

Or read current values:
    torchref.dtypes.float        # torch.float32
    torchref.device.current      # torch.device('cuda')
    torchref.sigma_cutoff_ed.value  # 3.0

The density-splat sigma cutoff can also be set at import time via the
TORCHREF_SIGMA_CUTOFF_ED environment variable (default 3.0).

MPS caveat: Apple's MPS backend does not support float64 / complex128. If
the resolved device is MPS and the configured float dtype is float64, a
warning is emitted at import time. Either set TORCHREF_DTYPE_FLOAT=float32
or TORCHREF_DEVICE=cpu to silence it.
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
# measurably degrades refinement convergence (median R-free regressed ~+0.004,
# with high-res / large-cell structures hit much harder). 3.0 is the standard
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
    """
    Density-splat sigma cutoff with property-based access.

    Read/set the number of sigmas at which the per-atom electron-density
    Gaussian is truncated::

        sigma_cutoff_ed.value          # get current cutoff (default 3.0)
        sigma_cutoff_ed.value = 4.0     # set at runtime

    Initialised from the ``TORCHREF_SIGMA_CUTOFF_ED`` environment variable at
    import time (default ``3.0``). Must be a positive number.
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

        compile_targets.value          # get (default False)
        compile_targets.value = True   # enable at runtime

    Initialised from ``TORCHREF_COMPILE_TARGETS`` at import time (set to
    "1"/"true"/"yes"/"on" to enable).

    Applies to the full-form MLF target (``--xray-mode ml_full``), which evaluates a
    fixed-node quadrature per reflection. In eager PyTorch that is a separate pass
    over the reflection arrays for every operation in every node, so it is bound by
    dispatch and memory traffic rather than by the maths, and fusing it is worth a
    lot: measured at 25 408 reflections, forward+backward goes from **101 ms eager to
    9.9 ms compiled (13x)**.

    **Off by default because of the compile latency, which autograd dominates.**
    A forward-only compile is ~22 s cold / ~9 s warm, but once the backward has to be
    compiled too the first call costs **~131 s**. Saving 91 ms per call, that only
    breaks even after ~1440 gradient evaluations -- more than a typical 10-cycle
    refinement of a small structure performs, so enabling it there makes the job
    slower, not faster.

    It pays off decisively when the per-call saving is larger:

    * **big datasets** -- the target scales with reflection count while ``F_calc``
      scales with atoms *and* reflections, so at ~740 000 reflections the eager cost
      is ~29x this measurement and the crossover falls to a few tens of calls;
    * **long or repeated runs in one process** -- ensembles, collection/PanDDA-style
      refinements, interactive sessions -- where the compile is amortised to nothing.

    Only the leading (reflection-count) dimension varies, so the kernels compile with
    ``dynamic=True`` and **one** compilation serves every dataset size: verified by
    counting ``unique_graphs`` across 13 sizes from 2 to 739 272 and across separate
    datasets, work/free subsets and gathered tensors. There is no per-structure
    recompilation, so no chunking or padding layer is needed.

    To cut the latency, point ``TORCHINDUCTOR_CACHE_DIR`` at node-local disk (never
    gpfs) so codegen is reused across processes; the artifacts are ~22 MB. The proper
    fix for a CLI is AOTInductor -- compile once at install to a ``.so`` and load it
    in milliseconds -- which would let this default back to on.

    Keep it off for float64 / gradient-verification work regardless: that path is
    the eager reference and deliberately unfused.
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


def canonical_device(dev):
    """Return ``dev`` with its default index filled in.

    ``torch.device('cuda') != torch.device('cuda:0')`` even though both name the
    same physical device. Devices read back off a real tensor always carry an
    index, so any comparison between a *requested* device and an *observed* one
    needs a shared normal form -- otherwise ``obj.device == tensor.device`` is
    False on a freshly constructed object and True after its first ``.to()``.

    ``cpu`` deliberately stays bare: ``torch.empty(0, device='cpu').device`` has
    no index, so canonicalising it to ``cpu:0`` would recreate the very mismatch
    this function exists to remove.

    This is the allocation-free form of ``torch.empty(0, device=d).device``,
    which matters because it is called on constructor and comparison paths.
    Deliberately *not* memoised: ``torch.cuda.current_device()`` is mutable via
    ``torch.cuda.set_device``, so a cache would freeze a stale answer.

    Parameters
    ----------
    dev : torch.device or str or int or None
        Device to normalise. ``None`` passes through so callers can chain.

    Returns
    -------
    torch.device or None
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
    """Coerce a user-supplied ``device`` argument to a canonical device.

    ``None`` resolves to :func:`get_default_device`. This is the pure,
    side-effect-free counterpart of :func:`torchref.utils.resolve_device`:

    * one device source (or none)  -> ``normalize_device``
    * several device-bearing inputs to reconcile -> ``resolve_device``

    ``resolve_device`` *moves* the objects it is given, so using it for a
    single-input constructor with nothing to reconcile is overreach.
    """
    if dev is None:
        return get_default_device()
    return canonical_device(dev)


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
