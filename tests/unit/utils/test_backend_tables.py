"""The three production dispatch tables, checked as policy rather than as numerics.

Separate from ``test_backends.py``, which exercises the resolver against synthetic tables.
This file asserts things about the tables that actually ship: that each backend is available
on the hosts where it is declared to be, that each row resolves to the kernel its name claims,
and that selection returns what the docs say for every reachable ``(device, dtype)``.

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

pytestmark = pytest.mark.unit

ALL_TABLES = (DENSITY_BACKENDS, DS_BACKENDS, TARGET_BACKENDS)

ALL_BACKENDS = [
    pytest.param(t, b, id=f"{t.name.split()[0]}-{b.name}")
    for t in ALL_TABLES
    for b in t.backends
]




# ---------------------------------------------------------------------------
# Availability: declared expectations must match reality
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table,backend", ALL_BACKENDS)
def test_backend_is_available_where_it_is_expected(table, backend):
    """A backend declared required on this host must actually be usable here.

    One parametrized test in place of a bespoke compile check per kernel -- that is what
    ``expect_available`` on the table row is for.

    It **fails rather than skips**, and that asymmetry is deliberate. Dispatch degrades
    quietly, which is correct for users and dangerous for CI: if every test skipped when a
    kernel went missing, a broken build would produce an all-green run while production had
    silently fallen back to a ~100x slower path.

    This is also what keeps the accelerator-vs-reference comparisons honest now that forcing
    an engine is no longer possible. A skewed Triton install used to be caught by
    ``Engine.TRITON`` raising; it is now caught here, on any run on that host, which is
    strictly earlier and louder.
    """
    if not backend.expected_here():
        pytest.skip(f"{backend.name} is not expected on this host")
    why = backend.unavailable()
    assert why is None, (
        f"{table.name}/{backend.name} is expected to work on this host but is not "
        f"available: {why}"
    )


# ---------------------------------------------------------------------------
# Provenance without a call recorder
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table", ALL_TABLES, ids=lambda t: t.name.split()[0])
def test_every_row_resolves_to_a_distinct_kernel(table):
    """No two rows may resolve to the same function, and iso must differ from aniso.

    Catches the one failure mode nothing else can: a row naming the *wrong but valid* kernel.
    If ``cpu_sphere`` pointed at ``add_isotropic_plain_var``, the accuracy tests would pass
    (portable *is* the reference implementation), the selection tests would pass (the name it
    returns is still right), and the availability test would pass (the extension built fine) --
    while the CPU fast path was silently gone. Here it collides with ``portable`` and fails.

    Asserting distinctness rather than comparing against the declared attribute is what makes
    this non-tautological: re-reading ``backend.kernel`` would only prove ``getattr`` works.

    Runs on every host, unlike the monkeypatched provenance tests, because resolving a kernel
    imports its module without launching it -- the CUDA module is guarded on ``_HAVE_TRITON``
    and the Metal one defers shader compilation to first use.
    """
    seen = {}
    for backend in table.backends:
        if backend.kernel is None:  # gate-only row; the call site owns the kernel
            continue
        iso, aniso = backend.resolve(False), backend.resolve(True)
        assert iso is not aniso, (
            f"{table.name}/{backend.name} resolves iso and aniso to the same function "
            f"({iso.__module__}.{iso.__name__})"
        )
        for variant, fn in (("iso", iso), ("aniso", aniso)):
            key = (fn.__module__, fn.__name__)
            if key in seen:
                other, other_variant = seen[key]
                pytest.fail(
                    f"{table.name}: {backend.name}/{variant} and {other}/{other_variant} "
                    f"both resolve to {key[0]}.{key[1]} -- one of these rows names the "
                    "wrong kernel, and every other test would still pass"
                )
            seen[key] = (backend.name, variant)


# ---------------------------------------------------------------------------
# Selection, enumerated
# ---------------------------------------------------------------------------
def _six(device, dtype):
    return [torch.zeros(4, device=device, dtype=dtype)] * 6


_CPU_CASES = [
    (torch.float32, False, "cpu_sphere"),
    (torch.float64, False, "cpu_sphere"),
    (torch.float32, True, "portable"),
    (torch.float64, True, "portable"),
]


@pytest.mark.parametrize(
    "dtype,pin,expected",
    _CPU_CASES,
    ids=[f"{d}-{'portable' if p else 'default'}" for d, p, _ in _CPU_CASES],
)
def test_density_selection_on_cpu(dtype, pin, expected):
    """The density policy, asserted directly instead of inferred from a call recorder.

    ``select`` is a pure function of the table, so this replaces most of what previously
    needed a monkeypatched spy -- and unlike a spy it can assert the float64 legs, which the
    accelerator provenance tests cannot reach at all.
    """
    got = select(DENSITY_BACKENDS, _six("cpu", dtype), force_portable=pin).name
    assert got == expected


@pytest.mark.mps
@pytest.mark.parametrize("pin,expected", [(False, "mps_metal"), (True, "portable")],
                         ids=["default", "portable"])
def test_density_selection_on_mps(pin, expected):
    got = select(DENSITY_BACKENDS, _six("mps", torch.float32), force_portable=pin).name
    assert got == expected


@pytest.mark.parametrize("table", ALL_TABLES, ids=lambda t: t.name.split()[0])
def test_force_portable_reaches_the_base_case_in_every_table(table):
    """The override is uniform: one flag, same meaning at every dispatch site.

    Not true of what it replaced -- ``Engine.METAL`` selected the Metal splat at the density
    site and silently meant "run eager" at the other two, a per-site asymmetry that existed
    only as an early ``return False`` inside a predicate.
    """
    ts = _six("cpu", torch.float32)
    assert select(table, ts, force_portable=True) is table.base


def test_ds_selection_ignores_integer_hkl():
    """An ``int32`` ``hkl`` must not change what direct summation selects.

    Miller indices are ``int32`` in production and every DS kernel casts them itself,
    exactly. ``hkl`` is outside the probe set for that reason, and the dtype rule exempts
    integers regardless -- two independent guards against the production dtype reading as a
    capability failure.
    """
    n = 4
    f32 = lambda: torch.zeros(n, dtype=torch.float32)  # noqa: E731
    args_int = [torch.zeros(n, 3, dtype=torch.int32), f32(), torch.zeros(n, 3)] + [f32()] * 4
    args_f32 = [torch.zeros(n, 3, dtype=torch.float32), f32(), torch.zeros(n, 3)] + [f32()] * 4
    assert (
        select(DS_BACKENDS, args_int).name == select(DS_BACKENDS, args_f32).name
    )


def test_second_order_capability_is_declared_not_guessed():
    """``second_order`` is the single source for which kernels compose under ``create_graph``.

    ``test_second_order.py`` derives its skip list from this, so a kernel gaining or losing a
    double-backward path cannot leave the test matrix stale.
    """
    density = {b.name: b.second_order for b in DENSITY_BACKENDS.backends}
    assert density == {
        "cuda_triton": False,
        "mps_metal": False,
        "cpu_sphere": True,
        "portable": True,
    }
