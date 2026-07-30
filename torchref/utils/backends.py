"""Declarative backend selection: the dispatch criteria, in one place, as data.

Every dispatch site in TorchRef answers the same question -- given these tensors and this
:class:`~torchref.utils.triton_dispatch.Engine`, which kernel runs? This module holds the
machinery for answering it from a table, so each criterion (device, dtype, engine,
availability, failure policy) is written down exactly once, next to the kernels it selects.
The tables themselves live with their kernels: see
``torchref/base/electron_density/_backends.py`` and
``torchref/base/direct_summation/_backends.py``.

There is no chain
----------------
The accelerator gates are pairwise device-disjoint -- CUDA, CPU and MPS -- so for any
given input **at most one** non-base backend can match. :func:`select` therefore returns a
single :class:`Backend`, not an ordered candidate list, and the only fallback is the
table's base case. An earlier design walked a chain; that models a control structure the
problem does not have.

Strictness is one boolean
-------------------------
A forcing engine (``TRITON``, ``METAL``) is admitted by exactly one row, so if that row
does not match, nothing does and :func:`select` raises with the reason. The base case
deliberately does **not** list the forcing engines in its ``engines`` set -- that omission
is the whole mechanism by which forcing is strict. Failure handling follows the same
principle: under ``AUTO`` a backend marked ``on_failure="degrade"`` falls back, under any
forcing engine everything propagates (see :func:`run_or_degrade`).

Why kernels are stored as ``(module, attr)`` rather than as functions
--------------------------------------------------------------------
Resolving ``getattr(import_module(path), attr)`` on every call costs a ``sys.modules`` hit
and a ``getattr``, and buys four things a captured function object would break:

* ``torchref.utils`` is imported very early and must not reach into
  ``torchref.base.electron_density`` at module scope;
* the Metal MSL source and ``import triton`` stay off hosts that never dispatch to them;
* availability stays readable *late*, so a test that flips
  ``mps.compile._lib_failed`` is still seen;
* the kernel stays monkeypatchable at its defining module, which is what keeps the
  provenance tests in ``tests/unit/structure_factor/test_dispatch.py`` meaningful. A table
  holding function objects captured at import time would make those tests pass while
  measuring nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Callable, Optional, Sequence, Tuple

import torch

from torchref.utils.triton_dispatch import Engine, get_engine

__all__ = ["Backend", "BackendTable", "admits", "select", "run_or_degrade"]


def _mps_present() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


#: Host conditions a backend can declare via ``Backend.expect_available``. Parametrized
#: rather than hand-written per kernel: "this must work here" is the same question for the
#: fused C++ splat, the Metal shader and the Triton kernels, and only the condition differs.
_EXPECTATIONS = {
    "never": lambda: False,
    "always": lambda: True,
    "cuda": torch.cuda.is_available,
    "mps": _mps_present,
}


@dataclass(frozen=True)
class Backend:
    """One row of a dispatch table: a kernel plus every criterion for choosing it.

    Parameters
    ----------
    name : str
        Stable identifier, used in error messages and by tests to name a path.
    kernel : (str, str, str), optional
        ``(module_path, isotropic_attr, anisotropic_attr)``, resolved per call. For
        single-variant backends pass the same attribute twice.

        ``None`` marks a **gate-only** backend: one whose selection criteria are worth
        declaring in a table even though the call site names its own kernel. The geometry
        targets are the case -- twelve modules each with a single ``if use_triton(x): from
        .triton.X import f`` call site, where the three-line local import is more legible
        than a registry hop, but "which engines and devices admit Triton here" still needs
        to be stated once. :func:`run_or_degrade` refuses such a backend; only
        :func:`select` and :func:`admits` accept it.
    engines : frozenset[Engine]
        Which engines admit this backend. A forcing engine listed here makes this the
        *only* candidate under it; omitting a forcing engine from the base case is what
        makes that engine strict.
    device : str, optional
        Required device type (``"cuda"``/``"mps"``/``"cpu"``). ``None`` means any, which
        is what marks the base case.
    dtypes : tuple[torch.dtype, ...], optional
        Permitted **floating-point** dtypes. ``None`` means any.

        Integer tensors are exempt, deliberately. Miller indices arrive as ``int32`` from
        the MTZ reader and every kernel casts them itself; that cast is exact for
        \\|h\\| < 2**24, so an identity test against ``float32`` would reject the
        production dtype and disable the kernel it was meant to protect. What the rule
        catches is a *float* in the wrong precision -- a float64 ``hkl`` whose silent
        downcast would truncate the phase.
    require_uniform_dtype : bool
        Whether every probed floating-point tensor must share **one** dtype, as opposed to
        each independently being in ``dtypes``.

        Not the same condition, and the difference is memory safety rather than taste. The
        fused CPU kernel selects one ``scalar_t`` from the output map via
        ``AT_DISPATCH_FLOATING_TYPES`` and then reads every other tensor through a raw
        pointer of that type, so a float64 map beside float32 atoms reinterprets the
        coordinate buffer as doubles -- a 2x out-of-bounds read. ``dtypes=(f32, f64)``
        alone *admits* that call.
    probes : tuple[int, ...], optional
        Which argument positions carry the device/dtype contract. ``None`` probes them all.

        Per-table because the requirement means different things in different tables. For
        the density splats float32 is a *capability* -- the C++ and the MSL shader cannot
        consume anything else. For direct summation it is *policy*: the Triton kernel casts
        everything itself, so the gate polices only the leaves whose precision the caller
        chose, and must not probe the float32 stored scattering-factor table (which would
        decline the kernel unconditionally).
    probe : (str, str), optional
        ``(module_path, attr)`` of a zero-argument callable returning ``None`` when the
        backend is usable, or a human-readable reason when it is not. ``None`` means
        always available. Resolved late, so a cache the callable consults can still be
        invalidated.

        One function per backend, deliberately: a separate boolean ``*_available()`` would
        be the same test written twice. Where such a bool already exists as public API it
        derives from this probe rather than repeating it.
    expect_available : {"never", "always", "cuda", "mps"}
        The host condition under which this backend *must* work. Distinct from ``probe``,
        which reports what is true; this states what ought to be.

        The two together are what let a broken build fail rather than skip. Dispatch under
        ``AUTO`` degrades quietly, which is right for users and dangerous for CI: if every
        test skips when a kernel is missing, a build that stopped working produces an
        all-green run while production has silently fallen back. Declaring the expectation
        here means one parametrized test covers every backend, instead of each kernel
        needing its own bespoke compile check.

        Not consulted by :func:`select` -- availability is a fact at dispatch time, never an
        expectation.
    on_failure : {"raise", "degrade"}
        What a *runtime* exception from the kernel means under ``Engine.AUTO``. Under any
        forcing engine this is ignored and the exception propagates.

        ``"degrade"`` carries a precondition: the kernel must not have mutated its inputs
        before failing, or the fallback would double-count. Both accelerator splats satisfy
        it by cloning the density map before accumulating.
    second_order : bool
        Whether the kernel composes under ``create_graph=True``. Not consulted by
        :func:`select`; it is here so the test matrix can be derived from this table rather
        than maintaining its own copy.
    """

    name: str
    kernel: Optional[Tuple[str, str, str]]
    engines: frozenset
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
    """An ordered set of backends plus the invariants that make it a complete policy.

    Two things are checked at import, both of which are currently unwritable against
    hand-rolled if/elif ladders:

    * **Every** :class:`Engine` member is handled by at least one backend. Without this,
      adding an engine -- or forgetting to let the base case absorb one -- silently turns a
      working call into a ``RuntimeError``. ``Engine.METAL`` is the live example: it selects
      the Metal density splat, and at *every other* dispatch site it must mean "run eager",
      a fact that otherwise exists only as an early ``return False`` inside a predicate.
    * Exactly one base case, i.e. one backend with no device and no dtype restriction.
      Selection under ``AUTO`` must always terminate somewhere.
    """

    name: str
    backends: Tuple[Backend, ...]
    base: Backend = field(init=False)

    def __post_init__(self):
        covered = frozenset().union(*(b.engines for b in self.backends))
        missing = frozenset(Engine) - covered
        if missing:
            raise ValueError(
                f"{self.name}: no backend handles "
                f"{sorted(e.name for e in missing)}. Every engine must select "
                "something -- if it should run the fallback, add it to that "
                "backend's `engines`."
            )
        bases = [b for b in self.backends if b.device is None and b.dtypes is None]
        if len(bases) != 1:
            raise ValueError(
                f"{self.name}: expected exactly one unrestricted base backend, "
                f"found {[b.name for b in bases]}"
            )
        object.__setattr__(self, "base", bases[0])

    def by_name(self, name: str) -> Backend:
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
    engine: Optional[Engine] = None,
) -> Backend:
    """The one backend that runs, or a ``RuntimeError`` explaining why none can.

    Two-phase by contract: device and dtype are checked for every candidate *before* any
    availability probe is called. That ordering is load-bearing twice over. It keeps an MPS
    host from compiling the CPU C++ extension it will never use, and it makes a forced
    engine report the device mismatch (``requires MPS float32``) rather than a compile
    failure it never got far enough to observe.

    Parameters
    ----------
    table : BackendTable
        The policy to apply.
    tensors : sequence of torch.Tensor or None
        Positional arguments to probe. ``None`` entries are ignored, so a site may pass
        optional inputs straight through.
    engine : Engine, optional
        Per-call override; defaults to the process-wide engine.
    """
    eng = engine if engine is not None else get_engine()
    if not isinstance(eng, Engine):
        # Loud on purpose. This used to be an implicit ``else`` that selected Triton, so
        # any unrecognised value silently chose an accelerator.
        raise ValueError(f"{table.name}: unhandled engine {eng!r}")

    reasons = []
    for backend in table.backends:
        if eng not in backend.engines:
            continue
        why = backend.mismatch(tensors)
        if why is None:
            why = backend.unavailable()
        if why is None:
            return backend
        reasons.append((backend, why))

    detail = "; ".join(f"{b.name} {why}" for b, why in reasons) or (
        "no backend admits this engine"
    )
    raise RuntimeError(f"engine=Engine.{eng.name} {detail}")


def admits(
    table: BackendTable,
    name: str,
    tensors: Sequence[Optional[torch.Tensor]],
    engine: Optional[Engine] = None,
) -> bool:
    """Whether one named backend would run -- the shape the public predicates need.

    Differs from :func:`select` in what a non-match means. ``select`` asks "what runs?" and
    raises when the answer is nothing. This asks "would *this* one run?", which has a third
    answer: an engine that does not admit this backend at all is not an error, it simply is
    not this backend's turn. ``should_use_metal`` under ``Engine.TRITON`` is False, not a
    failure.

    Strictness still applies where it should: if the engine *does* admit this backend and
    forces it, a failed criterion raises rather than returning False, because a forced
    engine that quietly declines has silently degraded.
    """
    eng = engine if engine is not None else get_engine()
    if not isinstance(eng, Engine):
        raise ValueError(f"{table.name}: unhandled engine {eng!r}")
    backend = table.by_name(name)
    if eng not in backend.engines:
        return False
    why = backend.mismatch(tensors)
    if why is None:
        why = backend.unavailable()
    if why is None:
        return True
    if eng is not Engine.AUTO:
        raise RuntimeError(f"engine=Engine.{eng.name} {why}")
    return False


def run_or_degrade(
    table: BackendTable,
    backend: Backend,
    aniso: bool,
    *args,
    engine: Optional[Engine] = None,
    **kwargs,
):
    """Run ``backend``'s kernel, falling back to the table's base case only if allowed.

    A forcing engine never degrades: the caller asked for a specific kernel, and quietly
    substituting another would make an A/B comparison or a benchmark measure the wrong
    thing. Under ``Engine.AUTO`` a backend marked ``on_failure="degrade"`` falls back,
    which is what keeps an unavailable accelerator a performance problem rather than an
    outage.
    """
    eng = engine if engine is not None else get_engine()
    fn = backend.resolve(aniso)
    strict = eng is not Engine.AUTO
    if strict or backend.on_failure == "raise" or backend is table.base:
        return fn(*args, **kwargs)
    try:
        return fn(*args, **kwargs)
    except Exception:
        return table.base.resolve(aniso)(*args, **kwargs)
