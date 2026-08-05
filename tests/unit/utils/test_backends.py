"""The dispatch resolver, tested against synthetic tables rather than real kernels.

Synthetic on purpose: this file is about the *policy* machinery -- device/dtype matching,
probe ordering, failure handling, the portable override -- and a synthetic table can express
cases no real host can reach (a CUDA row on a CPU-only machine, a probe that raises if called
at all). The shipping tables are checked where they live, against the kernels they name, in
``test_backend_tables.py``.

The three ``use_portable`` tests at the top are what survived the deletion of
``test_triton_dispatch.py``. Of its nineteen tests, only its state-management ones had a
subject that outlived the ``Engine`` enum; everything else asserted behaviour of forcing an
*accelerator*, which is no longer a concept, or device/dtype rules now asserted directly
against the shipping tables.
"""

import pytest
import torch

from torchref.utils.backends import (
    Backend,
    BackendTable,
    TorchRefDegradationWarning,
    force_portable,
    run_or_degrade,
    select,
    set_force_portable,
    use_portable,
    will_use,
)

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
):
    return BackendTable(
        name="synthetic",
        backends=(
            Backend(
                name="accel",
                kernel=(_THIS, "accel_iso", "accel_aniso"),
                device=accel_device,
                dtypes=(torch.float32,),
                probe=accel_probe,
                on_failure=accel_on_failure,
                second_order=False,
            ),
            Backend(name="base", kernel=(_THIS, "base_iso", "base_aniso")),
        ),
    )


# ---------------------------------------------------------------------------
# The portable override
# ---------------------------------------------------------------------------
def test_force_portable_roundtrip():
    previous = force_portable()
    try:
        set_force_portable(True)
        assert force_portable() is True
        set_force_portable(False)
        assert force_portable() is False
    finally:
        set_force_portable(previous)


def test_use_portable_nests_and_restores():
    previous = force_portable()
    try:
        set_force_portable(False)
        with use_portable():
            assert force_portable() is True
            with use_portable():
                assert force_portable() is True
            assert force_portable() is True
        assert force_portable() is False
    finally:
        set_force_portable(previous)


def test_use_portable_restores_on_exception():
    """A raise inside the block must not leave dispatch pinned for the rest of the process.

    Worth its own test because the failure is invisible and contagious: every later call in
    the process would silently run the reference kernel, and a benchmark would report a
    regression with no obvious cause.
    """
    previous = force_portable()
    try:
        set_force_portable(False)
        with pytest.raises(ValueError):
            with use_portable():
                raise ValueError("boom")
        assert force_portable() is False
    finally:
        set_force_portable(previous)


def test_force_portable_selects_the_base_case_over_a_matching_accelerator():
    """The override's whole job: pin the reference even when something faster would run.

    This is the one thing automatic selection cannot do for you. A fallback covers an
    accelerator that is missing or throws; it cannot cover one that runs and returns *wrong
    numbers*, which is what you are checking for when you reach for this.
    """
    t = _table(accel_device="cpu")
    ts = [torch.zeros(2)]
    assert select(t, ts).name == "accel"
    assert select(t, ts, force_portable=True).name == "base"
    with use_portable():
        assert select(t, ts).name == "base"


def test_per_call_argument_overrides_the_process_wide_setting():
    previous = force_portable()
    try:
        t = _table(accel_device="cpu")
        ts = [torch.zeros(2)]
        set_force_portable(True)
        assert select(t, ts).name == "base"
        assert select(t, ts, force_portable=False).name == "accel"
    finally:
        set_force_portable(previous)


# ---------------------------------------------------------------------------
# Table invariants, asserted at construction
# ---------------------------------------------------------------------------
def test_table_requires_exactly_one_base_case():
    """Totality depends on it: with no unrestricted row, some inputs would match nothing.

    ``select`` has no error path, and that is structural rather than optimistic -- so the
    condition that makes it true is checked where the table is defined.
    """
    with pytest.raises(ValueError, match="exactly one unrestricted base"):
        BackendTable(
            name="two-bases",
            backends=(
                Backend("a", (_THIS, "base_iso", "base_aniso")),
                Backend("b", (_THIS, "base_iso", "base_aniso")),
            ),
        )


def test_backend_rejects_an_unknown_failure_policy():
    with pytest.raises(ValueError, match="on_failure"):
        Backend("x", (_THIS, "base_iso", "base_aniso"), on_failure="explode")


def test_backend_rejects_an_unknown_expectation():
    with pytest.raises(ValueError, match="expect_available"):
        Backend("x", (_THIS, "base_iso", "base_aniso"), expect_available="on tuesdays")


def test_expectation_conditions_are_evaluated_against_the_host():
    """``expect_available`` is a host predicate, not a static flag."""
    mk = lambda exp: Backend(  # noqa: E731
        "b", (_THIS, "base_iso", "base_aniso"), expect_available=exp
    )
    assert mk("always").expected_here() is True
    assert mk("never").expected_here() is False
    assert mk("cuda").expected_here() is torch.cuda.is_available()


def test_expectation_does_not_influence_selection():
    """A backend that "should" work still loses to an honest probe saying it does not.

    Keeping these separate is the point: ``expect_available`` is what CI asserts, ``probe``
    is what dispatch obeys. Conflating them would route real work to a kernel that cannot
    run, on the strength of a mis-declared expectation.
    """
    t = BackendTable(
        name="expectation-vs-fact",
        backends=(
            Backend(
                "accel",
                (_THIS, "accel_iso", "accel_aniso"),
                device="cpu",
                probe=(_THIS, "probe_missing"),
                expect_available="always",
            ),
            Backend("base", (_THIS, "base_iso", "base_aniso")),
        ),
    )
    assert t.by_name("accel").expected_here() is True
    assert select(t, [torch.zeros(2)]).name == "base"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def test_falls_through_to_the_base_case_on_a_wrong_device():
    assert select(_table(), [torch.zeros(2)]).name == "base"


def test_selects_the_accelerator_when_everything_matches():
    """Device match is expressed through the table, not the host, so this runs anywhere."""
    assert select(_table(accel_device="cpu"), [torch.zeros(2)]).name == "accel"


def test_selection_never_fails():
    """There is no input for which nothing is selected.

    The base case declares no device and no dtype and carries no probe, so it matches
    unconditionally. Asserted against deliberately hostile inputs -- wrong device, wrong
    dtype, an unavailable accelerator, and no tensors at all.
    """
    t = _table(accel_probe=(_THIS, "probe_missing"))
    for ts in (
        [torch.zeros(2)],
        [torch.zeros(2, dtype=torch.float64)],
        [None, None],
        [],
    ):
        assert select(t, ts).name == "base"


def test_unavailability_falls_through():
    t = _table(accel_probe=(_THIS, "probe_missing"), accel_device="cpu")
    assert select(t, [torch.zeros(2)]).name == "base"


def test_none_entries_are_ignored():
    t = _table(accel_device="cpu")
    assert select(t, [None, torch.zeros(2), None]).name == "accel"


def test_probes_restrict_which_arguments_carry_the_contract():
    """A tensor outside ``probes`` must not affect selection.

    This is what lets direct summation police the refinable leaves while ignoring ``hkl``,
    whose dtype provably costs nothing.
    """
    table = BackendTable(
        name="probed",
        backends=(
            Backend(
                "accel",
                (_THIS, "accel_iso", "accel_aniso"),
                device="cpu",
                dtypes=(torch.float32,),
                probes=(0,),
            ),
            Backend("base", (_THIS, "base_iso", "base_aniso")),
        ),
    )
    f32, f64 = torch.zeros(2), torch.zeros(2, dtype=torch.float64)
    assert select(table, [f32, f64]).name == "accel"  # position 1 unprobed
    assert select(table, [f64, f32]).name == "base"  # position 0 still bites


def test_will_use_names_the_selected_backend():
    t = _table(accel_device="cpu")
    ts = [torch.zeros(2)]
    assert will_use(t, "accel", ts) is True
    assert will_use(t, "base", ts) is False
    assert will_use(t, "base", ts, force_portable=True) is True


# ---------------------------------------------------------------------------
# Two-phase ordering
# ---------------------------------------------------------------------------
def test_availability_is_not_probed_when_device_already_mismatches():
    """Device/dtype must be rejected before the probe is consulted.

    Enforced with a probe that raises if reached, because getting the order wrong is
    invisible to a return-value assertion: an MPS host would compile the CPU C++ extension it
    never uses, and a CPU-only host would import Triton to answer a question about a CPU
    tensor.
    """
    t = _table(accel_probe=(_THIS, "probe_must_not_run"))
    assert select(t, [torch.zeros(2)]).name == "base"


def test_a_broken_probe_import_is_a_reason_not_a_crash():
    t = _table(accel_probe=("torchref.no.such.module", "nope"), accel_device="cpu")
    assert select(t, [torch.zeros(2)]).name == "base"
    assert "could not be imported" in t.by_name("accel").unavailable()


# ---------------------------------------------------------------------------
# dtype rules
# ---------------------------------------------------------------------------
def test_require_uniform_dtype_rejects_a_mixed_set_that_membership_would_admit():
    """The distinction is memory safety, not taste.

    ``dtypes=(f32, f64)`` read as membership admits a float64 map beside float32 atoms, which
    the fused CPU kernel would reinterpret through a raw pointer of the map's type.
    """
    mixed = BackendTable(
        name="uniform",
        backends=(
            Backend(
                "fused",
                (_THIS, "accel_iso", "accel_aniso"),
                device="cpu",
                dtypes=(torch.float32, torch.float64),
                require_uniform_dtype=True,
            ),
            Backend("base", (_THIS, "base_iso", "base_aniso")),
        ),
    )
    f32, f64 = torch.zeros(2), torch.zeros(2, dtype=torch.float64)
    assert select(mixed, [f32, f32]).name == "fused"
    assert select(mixed, [f64, f64]).name == "fused"
    assert select(mixed, [f64, f32]).name == "base"


def test_integer_tensors_are_exempt_from_the_dtype_contract():
    """Integer inputs must not disqualify a float32 backend.

    Miller indices are ``int32`` in production and every kernel casts them itself, exactly
    and losslessly. An identity test against float32 would reject the production dtype and
    disable the very kernel the rule protects, so the contract applies to floats only.
    """
    t = _table(accel_device="cpu")
    hkl_int = torch.zeros(4, 3, dtype=torch.int32)
    assert select(t, [hkl_int, torch.zeros(2)]).name == "accel"
    # a float in the wrong precision still disqualifies
    assert select(t, [hkl_int, torch.zeros(2, dtype=torch.float64)]).name == "base"


def test_complex_tensors_are_refused_not_exempted():
    """Complex must disqualify a real-dtype backend, matching component precision or not.

    The mirror image of the integer rule above, and the reason both are worth pinning: they
    are the same line of code. The exemption is written with ``is_floating_point()``, which
    is ``False`` for complex as well as for integer -- so read literally it admitted a
    complex64 tensor through a ``dtypes=(float32,)`` gate. That is how a complex ``F_calc``
    reached the Triton Gaussian X-ray kernel, whose only defence was its own ``assert``.

    ``complex64`` is the interesting case precisely because its components *are* float32.
    Matching precision is not the question: every kernel behind these tables reads one real
    scalar per element, so the complex buffer would be walked as interleaved re/im pairs.
    Mapping complex to its component dtype and admitting it would be the wrong repair.
    """
    t = _table(accel_device="cpu")
    real = torch.zeros(2)
    assert select(t, [real, real]).name == "accel"
    for cdtype in (torch.complex64, torch.complex128):
        got = select(t, [real, torch.zeros(2, dtype=cdtype)])
        assert got.name == "base", f"{cdtype} satisfied a float32-only contract"
    # The message has to name the culprit, or a fallback is indistinguishable from a
    # device mismatch when someone is debugging why the accelerator went quiet.
    reason = t.by_name("accel").mismatch([real, torch.zeros(2, dtype=torch.complex64)])
    assert reason is not None and "complex" in reason


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
    assert run_or_degrade(t, t.by_name("accel"), False) == "patched"


def test_run_picks_the_anisotropic_variant():
    t = _table(accel_device="cpu")
    b = t.by_name("accel")
    assert run_or_degrade(t, b, False) == "accel_iso"
    assert run_or_degrade(t, b, True) == "accel_aniso"


def test_gate_only_backend_cannot_be_run():
    """A row with ``kernel=None`` declares criteria for a call site that names its own kernel.

    The geometry targets are the case. Selecting such a row is fine; running it is a
    programming error and says so, rather than failing with an opaque unpack.
    """
    t = BackendTable(
        name="gate-only",
        backends=(
            Backend("gate", kernel=None, device="cpu"),
            Backend("base", kernel=(_THIS, "base_iso", "base_aniso")),
        ),
    )
    with pytest.raises(TypeError, match="gate-only"):
        run_or_degrade(t, t.by_name("gate"), False)


def test_degrade_backend_falls_back_and_warns(monkeypatch):
    """The fallback runs -- and says so.

    Asserted with ``pytest.warns`` because a silent fallback is the failure mode the warning
    exists to prevent: both pytest configs promote ``TorchRefDegradationWarning`` to an error,
    so a degradation anywhere else in the suite fails the run. This is the one place it is
    expected, so it opts in.
    """
    import tests.unit.utils.test_backends as self_mod

    t = _table(accel_device="cpu", accel_on_failure="degrade")
    monkeypatch.setattr(self_mod, "accel_iso", boom)
    with pytest.warns(TorchRefDegradationWarning, match="kernel exploded"):
        got = run_or_degrade(t, t.by_name("accel"), False)
    assert got == "base_iso"


def test_raise_backend_does_not_fall_back(monkeypatch):
    """A production kernel that built fine and then threw is a bug, not a capability miss.

    Degrading would convert a wrong-results bug into a large silent slowdown whose output
    still looks plausible, because the fallback implements the same contract.
    """
    import tests.unit.utils.test_backends as self_mod

    t = _table(accel_device="cpu", accel_on_failure="raise")
    monkeypatch.setattr(self_mod, "accel_iso", boom)
    with pytest.raises(RuntimeError, match="kernel exploded"):
        run_or_degrade(t, t.by_name("accel"), False)


def test_base_case_failure_always_propagates(monkeypatch):
    import tests.unit.utils.test_backends as self_mod

    t = _table()
    monkeypatch.setattr(self_mod, "base_iso", boom)
    with pytest.raises(RuntimeError, match="kernel exploded"):
        run_or_degrade(t, t.base, False)
