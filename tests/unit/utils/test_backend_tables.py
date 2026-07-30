"""The three production dispatch tables, checked as policy rather than as numerics.

Separate from ``test_backends.py``, which exercises the resolver against synthetic tables.
This file asserts things about the tables that actually ship: that every engine is handled
somewhere, that each backend is available on the hosts where it is declared to be, and that
selection returns what the docs claim for each reachable ``(device, dtype, engine)``.

Everything here runs on any host. That is the point -- the accelerator provenance tests can
only cover the backend the machine happens to have, whereas ``select`` is a pure function of
the table and can be interrogated for combinations this host cannot execute.
"""

import pytest
import torch

from torchref.base.direct_summation._backends import DS_BACKENDS
from torchref.base.electron_density._backends import DENSITY_BACKENDS
from torchref.base.targets._dispatch import TARGET_BACKENDS
from torchref.utils.backends import select
from torchref.utils.triton_dispatch import Engine

pytestmark = pytest.mark.unit

ALL_TABLES = (DENSITY_BACKENDS, DS_BACKENDS, TARGET_BACKENDS)

ALL_BACKENDS = [
    pytest.param(t, b, id=f"{t.name.split()[0]}-{b.name}")
    for t in ALL_TABLES
    for b in t.backends
]


# ---------------------------------------------------------------------------
# Table-level invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table", ALL_TABLES, ids=lambda t: t.name.split()[0])
def test_every_table_handles_every_engine(table):
    """No engine may fall through a table with nothing claiming it.

    Constructing a ``BackendTable`` already asserts this, so importing the tables is most of
    the test. Stated explicitly anyway because it is the invariant the whole design exists
    for, and because the failure it prevents is invisible: ``Engine.METAL`` means "run the
    Metal density splat" at one site and "run eager" at every other, and if a table stopped
    absorbing it, ``with use_engine(Engine.METAL): loss = bond_target(x)`` would go from
    working to raising with no test noticing.
    """
    covered = frozenset().union(*(b.engines for b in table.backends))
    assert covered == frozenset(Engine), (
        f"{table.name}: {sorted(e.name for e in frozenset(Engine) - covered)} "
        "handled by no backend"
    )


@pytest.mark.parametrize("table", ALL_TABLES, ids=lambda t: t.name.split()[0])
def test_forcing_engines_are_not_absorbed_by_the_base_case(table):
    """``TRITON``/``METAL`` must be absent from the base case, or forcing stops being strict.

    This is the mechanism, not a detail: a forced engine raises precisely because no
    fallback lists it. Add either to the base case and every "strict engine refuses" test
    starts passing for the wrong reason.
    """
    for engine in (Engine.TRITON, Engine.METAL):
        if any(engine in b.engines and b is not table.base for b in table.backends):
            assert engine not in table.base.engines, (
                f"{table.name}: base case {table.base.name} absorbs {engine}, "
                "so forcing it can silently degrade"
            )


# ---------------------------------------------------------------------------
# Availability: declared expectations must hold on the hosts that declare them
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table,backend", ALL_BACKENDS)
def test_backend_is_available_where_it_is_expected(table, backend):
    """A backend declared required on this host must actually be usable here.

    One parametrized test in place of a bespoke compile check per kernel, which is what
    ``expect_available`` on the table row is for.

    It **fails rather than skips**, and that asymmetry is deliberate. Dispatch under
    ``Engine.AUTO`` degrades quietly -- correct for users, dangerous for CI -- so if every
    test skipped when a kernel went missing, a broken build would produce an all-green run
    while production had silently fallen back to a ~100x slower path. A host that is not
    expected to provide the backend simply generates no assertion.
    """
    if not backend.expected_here():
        pytest.skip(f"{backend.name} is not expected on this host")
    why = backend.unavailable()
    assert why is None, (
        f"{table.name}/{backend.name} is expected to work on this host but is not "
        f"available: {why}"
    )


# ---------------------------------------------------------------------------
# Selection, enumerated
# ---------------------------------------------------------------------------
def _six(device, dtype):
    return [torch.zeros(4, device=device, dtype=dtype)] * 6


_CPU_CASES = [
    (torch.float32, Engine.AUTO, "cpu_sphere"),
    (torch.float64, Engine.AUTO, "cpu_sphere"),
    (torch.float32, Engine.EAGER, "portable"),
    (torch.float64, Engine.EAGER, "portable"),
]


@pytest.mark.parametrize("dtype,engine,expected", _CPU_CASES,
                         ids=[f"{d}-{e.name}" for d, e, _ in _CPU_CASES])
def test_density_selection_on_cpu(dtype, engine, expected):
    """The density policy, asserted directly instead of inferred from a call recorder.

    ``select`` is a pure function of the table, so this replaces most of what previously
    needed a monkeypatched spy -- and unlike a spy it can assert the float64 leg, which the
    accelerator provenance tests cannot reach at all.
    """
    got = select(DENSITY_BACKENDS, _six("cpu", dtype), engine).name
    assert got == expected


@pytest.mark.mps
@pytest.mark.parametrize(
    "engine,expected", [(Engine.AUTO, "mps_metal"), (Engine.EAGER, "portable")]
)
def test_density_selection_on_mps(engine, expected):
    got = select(DENSITY_BACKENDS, _six("mps", torch.float32), engine).name
    assert got == expected


@pytest.mark.parametrize("engine", [Engine.TRITON, Engine.METAL])
def test_forced_accelerator_engines_refuse_cpu_inputs(engine):
    """Host-independent: a CPU tensor is wrong for both accelerators everywhere."""
    with pytest.raises(RuntimeError, match="requires (CUDA|MPS) float32"):
        select(DENSITY_BACKENDS, _six("cpu", torch.float32), engine)


def test_metal_engine_selects_eager_at_the_other_two_sites():
    """``Engine.METAL`` is density-only, and must mean "run eager" elsewhere.

    The characterization test in ``test_ds_dispatch.py`` covers this end to end; here it is
    asserted against the tables, where it is now a declared fact rather than an early
    ``return False`` inside a predicate.
    """
    cpu = _six("cpu", torch.float32)
    assert select(DS_BACKENDS, cpu, Engine.METAL).name == "checkpointed"
    assert select(TARGET_BACKENDS, cpu, Engine.METAL).name == "eager"


def test_ds_selection_ignores_integer_hkl():
    """An ``int32`` ``hkl`` must not change what direct summation selects.

    Miller indices are ``int32`` in production and every DS kernel casts them itself,
    exactly. ``hkl`` is outside the probe set for that reason, and the dtype rule exempts
    integers regardless -- two independent guards against the production dtype reading as a
    capability failure.
    """
    n = 4
    f32 = lambda: torch.zeros(n, dtype=torch.float32)  # noqa: E731
    hkl_int = torch.zeros(n, 3, dtype=torch.int32)
    hkl_f32 = torch.zeros(n, 3, dtype=torch.float32)
    args_int = [hkl_int, f32(), torch.zeros(n, 3), f32(), f32(), f32(), f32()]
    args_f32 = [hkl_f32, f32(), torch.zeros(n, 3), f32(), f32(), f32(), f32()]
    assert (
        select(DS_BACKENDS, args_int, Engine.AUTO).name
        == select(DS_BACKENDS, args_f32, Engine.AUTO).name
    )
