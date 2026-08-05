"""Declarative backend selection: the dispatch criteria, in one place, as data.

Every dispatch site answers the same question -- given these tensors, which kernel runs?
This module holds the machinery for answering it from a table, so each criterion (device,
dtype, availability, failure policy) is written down once. The tables themselves live with
their kernels: ``torchref/base/electron_density/_backends.py`` and
``torchref/base/direct_summation/_backends.py``.

The accelerator gates are pairwise device-disjoint, so at most one non-base backend can
match and :func:`select` returns a single :class:`Backend`, not a candidate chain. Kernels
are stored as ``(module, attr)`` and resolved per call: that keeps ``torchref.utils`` from
reaching into kernel packages at module scope, keeps availability readable late, and keeps
each kernel monkeypatchable where it is defined -- a table of function objects captured at
import time would make the dispatch-provenance tests pass while measuring nothing.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field
from importlib import import_module
from typing import Callable, Iterator, Optional, Sequence, Tuple

import torch


__all__ = [
    "Backend",
    "BackendTable",
    "TorchRefDegradationWarning",
    "force_portable",
    "run_or_degrade",
    "select",
    "set_force_portable",
    "use_portable",
    "will_use",
]


# ---------------------------------------------------------------------------
# Forcing the portable reference kernel
# ---------------------------------------------------------------------------
# Named after the table row it selects (``portable``), so the flag and the backend cannot
# drift apart in the reader's head. This is the whole override surface: it exists for the
# one failure automatic fallback cannot detect -- an accelerator that runs and returns
# wrong numbers rather than raising.
_FORCE_PORTABLE: bool = False


def force_portable() -> bool:
    """Whether dispatch is currently pinned to the portable reference kernel."""
    return _FORCE_PORTABLE


def set_force_portable(value: bool) -> None:
    """Pin (or unpin) dispatch to the portable reference kernel, process-wide."""
    global _FORCE_PORTABLE
    _FORCE_PORTABLE = bool(value)


@contextlib.contextmanager
def use_portable() -> Iterator[None]:
    """Pin the portable reference kernel for the duration of the block.

    Restores the previous setting on exit, including on exception.
    """
    previous = force_portable()
    set_force_portable(True)
    try:
        yield
    finally:
        set_force_portable(previous)


class TorchRefDegradationWarning(UserWarning):
    """An accelerator kernel failed at runtime and dispatch fell back to the reference.

    Its own category so tests can promote it to an error while production keeps running: a
    fallback is a performance problem for a user and a correctness signal for CI.
    """


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
_has_triton: Optional[bool] = None


def triton_available() -> bool:
    """Whether ``triton`` is importable (cheap, cached for the process).

    Lives here beside the per-backend probes that consume it. ``except Exception``, not
    ``except ImportError``: a Triton that is installed but skewed against the driver or LLVM
    raises something else, and that must read as "unavailable" rather than propagating.
    """
    global _has_triton
    if _has_triton is None:
        try:
            import triton  # noqa: F401

            _has_triton = True
        except Exception:
            _has_triton = False
    return _has_triton


def _mps_present() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


#: Host conditions a backend can declare via ``Backend.expect_available``.
_EXPECTATIONS = {
    "never": lambda: False,
    "always": lambda: True,
    "cuda": torch.cuda.is_available,
    "mps": _mps_present,
}


@dataclass(frozen=True)
class Backend:
    """One row of a dispatch table: a kernel plus every criterion for choosing it.

    Parameter-heavy by intent -- each field is one dispatch criterion, and four of them
    carry hazards a caller cannot see from the table.

    Parameters
    ----------
    name : str
        Stable identifier, used in error messages and by tests to name a path.
    kernel : (str, str, str), optional
        ``(module_path, isotropic_attr, anisotropic_attr)``, resolved per call; pass the
        same attribute twice for a single-variant backend. ``None`` marks a **gate-only**
        backend, whose call site names its own kernel: :func:`run_or_degrade` refuses one,
        only :func:`select` and :func:`will_use` accept it.
    device : str, optional
        Required device type (``"cuda"``/``"mps"``/``"cpu"``). ``None`` means any, which
        is what marks the base case.
    dtypes : tuple[torch.dtype, ...], optional
        Permitted **floating-point** dtypes; ``None`` means any. Integer tensors are exempt
        (Miller indices arrive as ``int32`` and every kernel casts them itself), but
        **complex tensors are refused, not exempt** -- the kernels behind these tables read
        one real scalar per element, so an admitted complex64 buffer would be reinterpreted
        as interleaved re/im pairs. Checked separately, since ``is_floating_point()`` is
        ``False`` for complex as well as for integers.
    require_uniform_dtype : bool
        Whether every probed float tensor must share **one** dtype, rather than each
        independently being in ``dtypes``. Memory safety, not taste: the fused CPU kernel
        picks one ``scalar_t`` from the output map and reads every other tensor through a
        raw pointer of that type, so a float64 map beside float32 atoms is a 2x
        out-of-bounds read -- which ``dtypes=(f32, f64)`` alone *admits*.
    probes : tuple[int, ...], optional
        Which argument positions carry the device/dtype contract; ``None`` probes all. Set
        per table, because for some kernels the dtype is a capability and for others only
        policy over the leaves whose precision the caller chose.
    probe : (str, str), optional
        ``(module_path, attr)`` of a zero-argument callable returning ``None`` when the
        backend is usable, else a human-readable reason; ``None`` means always available.
        Resolved late, so a cache the callable consults can still be invalidated.
    expect_available : {"never", "always", "cuda", "mps"}
        The host condition under which this backend *must* work, so a broken build fails
        instead of skipping. Distinct from ``probe``, which reports what is true; this
        states what ought to be, and is **not** consulted by :func:`select`.
    on_failure : {"raise", "degrade"}
        What a *runtime* exception from the kernel means: ``"degrade"`` falls back to the
        base case (with a warning), ``"raise"`` propagates. ``"degrade"`` carries a
        precondition -- the kernel must not have mutated its inputs before failing, or the
        fallback double-counts; both accelerator splats clone the map before accumulating.
    second_order : bool
        Whether the kernel composes under ``create_graph=True``. Not consulted by
        :func:`select`; it is here so the test matrix can be derived from this table.
    """

    name: str
    kernel: Optional[Tuple[str, str, str]]
    device: Optional[str] = None
    dtypes: Optional[Tuple[torch.dtype, ...]] = None
    require_uniform_dtype: bool = False
    probes: Optional[Tuple[int, ...]] = None
    probe: Optional[Tuple[str, str]] = None
    expect_available: str = "never"
    on_failure: str = "raise"
    second_order: bool = True

    def __post_init__(self):
        if self.on_failure not in ("raise", "degrade"):
            raise ValueError(
                f"{self.name}: on_failure must be 'raise' or 'degrade', "
                f"got {self.on_failure!r}"
            )
        if self.expect_available not in _EXPECTATIONS:
            raise ValueError(
                f"{self.name}: expect_available must be one of "
                f"{sorted(_EXPECTATIONS)}, got {self.expect_available!r}"
            )

    def expected_here(self) -> bool:
        """Whether this host is one where this backend is required to work."""
        return _EXPECTATIONS[self.expect_available]()

    # -- criteria ---------------------------------------------------------
    def requirement(self) -> str:
        """The device/dtype contract as an error fragment, e.g. ``requires MPS float32``."""
        parts = []
        if self.device is not None:
            parts.append(self.device.upper())
        if self.dtypes is not None:
            parts.append("/".join(str(d).replace("torch.", "") for d in self.dtypes))
        if not parts:
            return "accepts any inputs"
        return "requires " + " ".join(parts) + " inputs"

    def mismatch(self, tensors: Sequence[Optional[torch.Tensor]]) -> Optional[str]:
        """Why these tensors fail this backend's device/dtype contract, or ``None``."""
        probed = _probed(self, tensors)
        if self.device is not None and any(
            t.device.type != self.device for t in probed
        ):
            return self.requirement()
        if self.dtypes is not None:
            # Checked separately from the membership test below, which cannot see complex
            # at all: ``is_floating_point()`` is False for it. Refused outright rather than
            # mapped to its component dtype -- these kernels read one real scalar per
            # element, so a complex64 buffer would be read as interleaved re/im pairs.
            complexes = [t for t in probed if t.is_complex()]
            if complexes:
                got = sorted({str(t.dtype).replace("torch.", "") for t in complexes})
                return (
                    f"{self.requirement()}, and a complex tensor ({', '.join(got)}) does "
                    "not satisfy a real-dtype contract"
                )
            floats = [t for t in probed if t.is_floating_point()]
            if any(t.dtype not in self.dtypes for t in floats):
                return self.requirement()
            if self.require_uniform_dtype and len({t.dtype for t in floats}) > 1:
                return (
                    f"{self.requirement()} sharing a single dtype (got "
                    + ", ".join(
                        sorted(str(d).replace("torch.", "") for d in {t.dtype for t in floats})
                    )
                    + ")"
                )
        return None

    def unavailable(self) -> Optional[str]:
        """Why this backend cannot run on this host, or ``None``."""
        if self.probe is None:
            return None
        module, attr = self.probe
        try:
            fn = getattr(import_module(module), attr)
        except Exception as exc:  # noqa: BLE001 - a stripped install must not crash a gate
            return f"its availability probe could not be imported ({type(exc).__name__}: {exc})"
        return fn()

    def resolve(self, aniso: bool) -> Callable:
        """The kernel function, looked up now rather than captured at import."""
        if self.kernel is None:
            raise TypeError(
                f"{self.name} is gate-only (kernel=None): its call site names its own "
                "kernel, so there is nothing here to run."
            )
        module, iso_attr, aniso_attr = self.kernel
        return getattr(import_module(module), aniso_attr if aniso else iso_attr)


@dataclass(frozen=True)
class BackendTable:
    """An ordered set of backends plus the invariant that makes it a total policy.

    Checked at import: exactly one base case, i.e. one backend with no device and no dtype
    restriction. That is what makes :func:`select` unable to fail -- with no unrestricted row
    there would be inputs no backend matched, and selection would need an error path.
    """

    name: str
    backends: Tuple[Backend, ...]
    base: Backend = field(init=False)

    def __post_init__(self):
        bases = [b for b in self.backends if b.device is None and b.dtypes is None]
        if len(bases) != 1:
            raise ValueError(
                f"{self.name}: expected exactly one unrestricted base backend, "
                f"found {[b.name for b in bases]}"
            )
        object.__setattr__(self, "base", bases[0])

    def by_name(self, name: str) -> Backend:
        """The backend named ``name``; raises ``KeyError`` if this table has none."""
        for b in self.backends:
            if b.name == name:
                return b
        raise KeyError(f"{self.name}: no backend named {name!r}")


def _probed(backend: Backend, tensors: Sequence[Optional[torch.Tensor]]):
    """The tensors this backend's contract applies to: selected, then ``None``-filtered."""
    if backend.probes is None:
        chosen = tensors
    else:
        chosen = [tensors[i] for i in backend.probes if i < len(tensors)]
    return [t for t in chosen if t is not None]


def select(
    table: BackendTable,
    tensors: Sequence[Optional[torch.Tensor]],
    force_portable: Optional[bool] = None,
) -> Backend:
    """The one backend that runs. Cannot fail.

    Totality is structural: the base case restricts neither device nor dtype and carries no
    probe, so the loop always terminates there.

    Two-phase by contract: device and dtype are checked for every candidate *before* any
    availability probe is called. That ordering is load-bearing -- it keeps an MPS host from
    compiling the CPU C++ extension it will never use, and a CUDA host from importing Triton
    to answer a question about a CPU tensor.

    Parameters
    ----------
    table : BackendTable
        The policy to apply.
    tensors : sequence of torch.Tensor or None
        Positional arguments to probe. ``None`` entries are ignored, so a site may pass
        optional inputs straight through.
    force_portable : bool, optional
        Pin the portable reference kernel. ``None`` defers to the process-wide setting.
    """
    pin = force_portable if force_portable is not None else _FORCE_PORTABLE
    if pin:
        return table.base

    for backend in table.backends:
        if backend.mismatch(tensors) is None and backend.unavailable() is None:
            return backend
    return table.base


def will_use(
    table: BackendTable,
    name: str,
    tensors: Sequence[Optional[torch.Tensor]],
    force_portable: Optional[bool] = None,
) -> bool:
    """Whether ``name`` is the backend that would actually run for these inputs."""
    return select(table, tensors, force_portable=force_portable).name == name


def run_or_degrade(
    table: BackendTable,
    backend: Backend,
    aniso: bool,
    *args,
    **kwargs,
):
    """Run ``backend``'s kernel, falling back to the table's base case only if allowed.

    ``on_failure`` alone decides: ``"degrade"`` falls back and always emits a
    :class:`TorchRefDegradationWarning`, ``"raise"`` propagates. Requires the ``"degrade"``
    precondition documented on :class:`Backend` -- an input mutated before the failure
    would be counted twice.
    """
    fn = backend.resolve(aniso)
    if backend.on_failure == "raise" or backend is table.base:
        return fn(*args, **kwargs)
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the fallback is the point
        warnings.warn(
            f"{table.name}: the {backend.name} kernel failed and dispatch fell back to "
            f"{table.base.name} ({type(exc).__name__}: {exc}). Results are correct but "
            "slower; this is worth investigating.",
            TorchRefDegradationWarning,
            stacklevel=2,
        )
        return table.base.resolve(aniso)(*args, **kwargs)
