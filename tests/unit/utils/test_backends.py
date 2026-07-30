"""The dispatch resolver, tested against synthetic tables rather than real kernels.

Synthetic on purpose: this file is about the *policy* machinery -- engine admission,
device/dtype matching, probe ordering, failure handling -- and a synthetic table can
express cases no real host can reach (a CUDA row on a CPU-only machine, a probe that
raises if called at all). The real tables are checked where they live, against the kernels
they name.
"""

import pytest
import torch

from torchref.utils.backends import Backend, BackendTable, run_or_degrade, select
from torchref.utils.triton_dispatch import Engine

pytestmark = pytest.mark.unit

_THIS = "tests.unit.utils.test_backends"


# --- kernels the synthetic tables point at -------------------------------
def accel_iso(*a, **k):
    return "accel_iso"


def accel_aniso(*a, **k):
    return "accel_aniso"


def base_iso(*a, **k):
    return "base_iso"


def base_aniso(*a, **k):
    return "base_aniso"


def boom(*a, **k):
    raise RuntimeError("kernel exploded")


def probe_ok():
    return None


def probe_missing():
    return "the widget is not installed"


def probe_must_not_run():
    raise AssertionError("availability probed before device/dtype was checked")


def _table(
    *,
    accel_probe=(_THIS, "probe_ok"),
    accel_on_failure="degrade",
    accel_device="cuda",
    base_engines=(Engine.AUTO, Engine.EAGER, Engine.METAL),
):
    return BackendTable(
        name="synthetic",
        backends=(
            Backend(
                name="accel",
                kernel=(_THIS, "accel_iso", "accel_aniso"),
                engines=frozenset({Engine.AUTO, Engine.TRITON}),
                device=accel_device,
                dtypes=(torch.float32,),
                probe=accel_probe,
                on_failure=accel_on_failure,
                second_order=False,
            ),
            Backend(
                name="base",
                kernel=(_THIS, "base_iso", "base_aniso"),
                engines=frozenset(base_engines),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Table invariants, asserted at construction
# ---------------------------------------------------------------------------
def test_table_rejects_an_unhandled_engine():
    """A table that no backend claims for some engine is a construction error.

    This is the invariant the whole design is for. ``Engine.METAL`` selects the Metal
    density splat and must mean "run eager" everywhere else; if a table forgets to let its
    base case absorb it, calls that work today start raising. Catching it at import turns a
    silent behavioural regression into a failure at the definition site.
    """
    with pytest.raises(ValueError, match="no backend handles"):
        _table(base_engines=(Engine.AUTO, Engine.EAGER))  # METAL unclaimed


def test_table_requires_exactly_one_base_case():
    with pytest.raises(ValueError, match="exactly one unrestricted base"):
        BackendTable(
            name="two-bases",
            backends=(
                Backend("a", (_THIS, "base_iso", "base_aniso"), frozenset(Engine)),
                Backend("b", (_THIS, "base_iso", "base_aniso"), frozenset(Engine)),
            ),
        )


def test_backend_rejects_an_unknown_failure_policy():
    with pytest.raises(ValueError, match="on_failure"):
        Backend(
            "x",
            (_THIS, "base_iso", "base_aniso"),
            frozenset(Engine),
            on_failure="explode",
        )


def test_backend_rejects_an_unknown_expectation():
    with pytest.raises(ValueError, match="expect_available"):
        Backend(
            "x",
            (_THIS, "base_iso", "base_aniso"),
            frozenset(Engine),
            expect_available="on tuesdays",
        )


def test_expectation_conditions_are_evaluated_against_the_host():
    """``expect_available`` is a host predicate, not a static flag."""
    always = Backend(
        "a", (_THIS, "base_iso", "base_aniso"), frozenset(Engine),
        expect_available="always",
    )
    never = Backend(
        "n", (_THIS, "base_iso", "base_aniso"), frozenset(Engine),
        expect_available="never",
    )
    cuda = Backend(
        "c", (_THIS, "base_iso", "base_aniso"), frozenset(Engine),
        expect_available="cuda",
    )
    assert always.expected_here() is True
    assert never.expected_here() is False
    assert cuda.expected_here() is torch.cuda.is_available()


def test_expectation_does_not_influence_selection():
    """A backend that "should" work still loses to an honest probe saying it does not.

    Keeping these separate is the point: ``expect_available`` is what CI asserts,
    ``probe`` is what dispatch obeys. Conflating them would make a mis-declared
    expectation route real work to a kernel that cannot run.
    """
    t = BackendTable(
        name="expectation-vs-fact",
        backends=(
            Backend(
                "accel",
                (_THIS, "accel_iso", "accel_aniso"),
                frozenset({Engine.AUTO, Engine.TRITON}),
                device="cpu",
                probe=(_THIS, "probe_missing"),
                expect_available="always",
            ),
            Backend("base", (_THIS, "base_iso", "base_aniso"), frozenset(Engine)),
        ),
    )
    assert t.by_name("accel").expected_here() is True
    assert select(t, [torch.zeros(2)], Engine.AUTO).name == "base"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def test_auto_falls_through_to_the_base_case_on_a_wrong_device():
    t = _table()
    cpu = torch.zeros(2)
    assert select(t, [cpu], Engine.AUTO).name == "base"


def test_auto_selects_the_accelerator_when_everything_matches():
    """Device match is expressed through the table, not the host, so this runs anywhere."""
    t = _table(accel_device="cpu")
    assert select(t, [torch.zeros(2)], Engine.AUTO).name == "accel"


def test_eager_reaches_the_base_case_even_when_the_accelerator_would_match():
    t = _table(accel_device="cpu")
    assert select(t, [torch.zeros(2)], Engine.EAGER).name == "base"


def test_a_forcing_engine_raises_rather_than_degrading():
    """The base case does not list TRITON, so nothing absorbs a failed match."""
    t = _table()
    with pytest.raises(RuntimeError, match="requires CUDA float32"):
        select(t, [torch.zeros(2)], Engine.TRITON)


def test_forcing_engine_message_names_the_engine_and_the_requirement():
    t = _table()
    with pytest.raises(RuntimeError) as exc:
        select(t, [torch.zeros(2)], Engine.TRITON)
    assert "engine=Engine.TRITON" in str(exc.value)


def test_unavailability_degrades_under_auto_and_raises_when_forced():
    t = _table(accel_probe=(_THIS, "probe_missing"), accel_device="cpu")
    cpu = torch.zeros(2)
    assert select(t, [cpu], Engine.AUTO).name == "base"
    with pytest.raises(RuntimeError, match="the widget is not installed"):
        select(t, [cpu], Engine.TRITON)


def test_non_engine_values_are_rejected_loudly():
    """An unrecognised engine must not fall through to a default.

    The predicates this replaces had an explicit guard for the same reason: the AUTO case
    was once an implicit ``else``, so any new or bogus value silently selected Triton.
    """

    class NotAnEngine:
        pass

    with pytest.raises(ValueError, match="unhandled engine"):
        select(_table(), [torch.zeros(2)], NotAnEngine())


def test_none_entries_are_ignored():
    t = _table(accel_device="cpu")
    assert select(t, [None, torch.zeros(2), None], Engine.AUTO).name == "accel"


def test_probes_restrict_which_arguments_carry_the_contract():
    """A tensor outside ``probes`` must not affect selection.

    This is what lets direct summation police the refinable leaves while ignoring the
    float32 stored scattering-factor table, which would otherwise decline the kernel
    unconditionally.
    """
    table = BackendTable(
        name="probed",
        backends=(
            Backend(
                "accel",
                (_THIS, "accel_iso", "accel_aniso"),
                frozenset({Engine.AUTO, Engine.TRITON}),
                device="cpu",
                dtypes=(torch.float32,),
                probes=(0,),
            ),
            Backend("base", (_THIS, "base_iso", "base_aniso"), frozenset(Engine)),
        ),
    )
    f32, f64 = torch.zeros(2), torch.zeros(2, dtype=torch.float64)
    # position 1 is float64 but unprobed
    assert select(table, [f32, f64], Engine.AUTO).name == "accel"
    # probing position 0 still bites
    assert select(table, [f64, f32], Engine.AUTO).name == "base"


# ---------------------------------------------------------------------------
# Two-phase ordering
# ---------------------------------------------------------------------------
def test_availability_is_not_probed_when_device_already_mismatches():
    """Device/dtype must be rejected before the probe is consulted.

    Enforced with a probe that raises if reached, because the consequences of getting the
    order wrong are invisible to a return-value assertion: an MPS host would compile the
    CPU C++ extension it never uses, and a forced engine would report a compile failure it
    never got far enough to observe instead of the device mismatch.
    """
    t = _table(accel_probe=(_THIS, "probe_must_not_run"))
    cpu = torch.zeros(2)
    assert select(t, [cpu], Engine.AUTO).name == "base"
    with pytest.raises(RuntimeError, match="requires CUDA float32"):
        select(t, [cpu], Engine.TRITON)


def test_a_broken_probe_import_is_a_reason_not_a_crash():
    t = _table(accel_probe=("torchref.no.such.module", "nope"), accel_device="cpu")
    assert select(t, [torch.zeros(2)], Engine.AUTO).name == "base"
    with pytest.raises(RuntimeError, match="could not be imported"):
        select(t, [torch.zeros(2)], Engine.TRITON)


# ---------------------------------------------------------------------------
# dtype uniformity
# ---------------------------------------------------------------------------
def test_require_uniform_dtype_rejects_a_mixed_set_that_membership_would_admit():
    """The distinction is memory safety, not taste.

    ``dtypes=(f32, f64)`` read as membership admits a float64 map beside float32 atoms,
    which the fused CPU kernel would reinterpret through a raw pointer of the map's type.
    """
    mixed = BackendTable(
        name="uniform",
        backends=(
            Backend(
                "fused",
                (_THIS, "accel_iso", "accel_aniso"),
                frozenset({Engine.AUTO}),
                device="cpu",
                dtypes=(torch.float32, torch.float64),
                require_uniform_dtype=True,
            ),
            Backend("base", (_THIS, "base_iso", "base_aniso"), frozenset(Engine)),
        ),
    )
    f32, f64 = torch.zeros(2), torch.zeros(2, dtype=torch.float64)
    assert select(mixed, [f32, f32], Engine.AUTO).name == "fused"
    assert select(mixed, [f64, f64], Engine.AUTO).name == "fused"
    assert select(mixed, [f64, f32], Engine.AUTO).name == "base"


def test_integer_tensors_are_exempt_from_the_dtype_contract():
    """Integer inputs must not disqualify a float32 backend.

    Miller indices are ``int32`` in production and every kernel casts them itself, exactly
    and losslessly. An identity test against float32 would reject the production dtype and
    disable the very kernel the rule protects, so the contract applies to floats only.
    """
    t = _table(accel_device="cpu")
    hkl_int = torch.zeros(4, 3, dtype=torch.int32)
    f32 = torch.zeros(2)
    assert select(t, [hkl_int, f32], Engine.AUTO).name == "accel"
    # a float in the wrong precision still disqualifies
    f64 = torch.zeros(2, dtype=torch.float64)
    assert select(t, [hkl_int, f64], Engine.AUTO).name == "base"


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def test_run_resolves_the_kernel_late_so_it_stays_patchable(monkeypatch):
    """The kernel is looked up at call time, at its defining module.

    Load-bearing: the provenance tests prove a named kernel really ran by patching it. A
    table that captured function objects at import would leave those tests green while
    measuring nothing.
    """
    import tests.unit.utils.test_backends as self_mod

    t = _table(accel_device="cpu")
    monkeypatch.setattr(self_mod, "accel_iso", lambda *a, **k: "patched")
    assert run_or_degrade(t, t.by_name("accel"), False, engine=Engine.AUTO) == "patched"


def test_run_picks_the_anisotropic_variant():
    t = _table(accel_device="cpu")
    b = t.by_name("accel")
    assert run_or_degrade(t, b, False, engine=Engine.AUTO) == "accel_iso"
    assert run_or_degrade(t, b, True, engine=Engine.AUTO) == "accel_aniso"


def test_degrade_backend_falls_back_under_auto(monkeypatch):
    import tests.unit.utils.test_backends as self_mod

    t = _table(accel_device="cpu", accel_on_failure="degrade")
    monkeypatch.setattr(self_mod, "accel_iso", boom)
    assert run_or_degrade(t, t.by_name("accel"), False, engine=Engine.AUTO) == "base_iso"


def test_degrade_backend_still_raises_under_a_forcing_engine(monkeypatch):
    """Forcing a kernel and silently getting another one would make a benchmark lie."""
    import tests.unit.utils.test_backends as self_mod

    t = _table(accel_device="cpu", accel_on_failure="degrade")
    monkeypatch.setattr(self_mod, "accel_iso", boom)
    with pytest.raises(RuntimeError, match="kernel exploded"):
        run_or_degrade(t, t.by_name("accel"), False, engine=Engine.TRITON)


def test_raise_backend_does_not_fall_back_even_under_auto(monkeypatch):
    """A production kernel that built fine and then threw is a bug, not a capability miss.

    Degrading would convert a wrong-results bug into a large silent slowdown whose output
    still looks plausible, because the fallback implements the same contract.
    """
    import tests.unit.utils.test_backends as self_mod

    t = _table(accel_device="cpu", accel_on_failure="raise")
    monkeypatch.setattr(self_mod, "accel_iso", boom)
    with pytest.raises(RuntimeError, match="kernel exploded"):
        run_or_degrade(t, t.by_name("accel"), False, engine=Engine.AUTO)


def test_base_case_failure_always_propagates(monkeypatch):
    import tests.unit.utils.test_backends as self_mod

    t = _table()
    monkeypatch.setattr(self_mod, "base_iso", boom)
    with pytest.raises(RuntimeError, match="kernel exploded"):
        run_or_degrade(t, t.base, False, engine=Engine.AUTO)
