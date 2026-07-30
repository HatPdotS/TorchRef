"""Unit tests for the shared Triton/eager dispatch engine (CPU-only)."""

import pytest
import torch

from torchref.utils import (
    Engine,
    get_engine,
    set_engine,
    should_use_metal,
    should_use_triton,
    triton_available,
    use_engine,
)

pytestmark = pytest.mark.unit


def test_engine_roundtrip():
    prev = get_engine()
    try:
        set_engine(Engine.TRITON)
        assert get_engine() is Engine.TRITON
        set_engine(Engine.EAGER)
        assert get_engine() is Engine.EAGER
    finally:
        set_engine(prev)


def test_use_engine_nests_and_restores():
    prev = get_engine()
    try:
        set_engine(Engine.AUTO)
        with use_engine(Engine.EAGER):
            assert get_engine() is Engine.EAGER
            with use_engine(Engine.TRITON):
                assert get_engine() is Engine.TRITON
            assert get_engine() is Engine.EAGER
        assert get_engine() is Engine.AUTO
    finally:
        set_engine(prev)


def test_use_engine_restores_on_exception():
    prev = get_engine()
    try:
        set_engine(Engine.AUTO)
        with pytest.raises(ValueError):
            with use_engine(Engine.EAGER):
                raise ValueError("boom")
        assert get_engine() is Engine.AUTO
    finally:
        set_engine(prev)


def test_eager_engine_never_uses_triton():
    cpu = torch.zeros(3)
    assert should_use_triton(cpu, engine=Engine.EAGER) is False
    # even with a (hypothetical) cuda fp32 tensor, EAGER forces eager
    assert should_use_triton(None, engine=Engine.EAGER) is False


def test_auto_false_on_cpu_and_fp64():
    assert should_use_triton(torch.zeros(3, dtype=torch.float32), engine=Engine.AUTO) is False
    assert should_use_triton(torch.zeros(3, dtype=torch.float64), engine=Engine.AUTO) is False


def test_triton_engine_raises_on_cpu_or_fp64():
    with pytest.raises(RuntimeError):
        should_use_triton(torch.zeros(3, dtype=torch.float32), engine=Engine.TRITON)
    with pytest.raises(RuntimeError):
        should_use_triton(torch.zeros(3, dtype=torch.float64), engine=Engine.TRITON)


def test_none_tensors_are_ignored():
    # all-None probes: capability is vacuously satisfied; AUTO depends only on
    # triton availability (False on a CPU-only box, True if triton importable).
    assert should_use_triton(None, None, engine=Engine.AUTO) is triton_available()


def test_should_use_triton_reads_global_engine_by_default():
    prev = get_engine()
    try:
        set_engine(Engine.EAGER)
        assert should_use_triton(torch.zeros(3)) is False
    finally:
        set_engine(prev)


# ---------------------------------------------------------------------------
# Engine.METAL / should_use_metal
#
# All CPU-safe, so they run on every host: the point is the *strict* contract,
# which is observable without an accelerator because "wrong device" is one of
# the conditions that must raise.
# ---------------------------------------------------------------------------


def test_metal_engine_roundtrip():
    prev = get_engine()
    try:
        set_engine(Engine.METAL)
        assert get_engine() is Engine.METAL
    finally:
        set_engine(prev)


def test_should_use_triton_false_under_metal():
    """METAL must not leak into the Triton gate.

    Before the explicit-AUTO tail below, ``should_use_triton`` ended in an
    implicit ``else``, so a new member fell through and selected Triton -- on a
    CUDA host, ``Engine.METAL`` would have run the Triton kernel.
    """
    assert should_use_triton(torch.zeros(3), engine=Engine.METAL) is False


def test_should_use_triton_rejects_unknown_engine():
    """An unhandled member must fail loudly rather than silently pick Triton."""

    class _NotAnEngine:
        pass

    with pytest.raises(ValueError, match="unhandled engine"):
        should_use_triton(torch.zeros(3), engine=_NotAnEngine())


def test_should_use_metal_rejects_unknown_engine():
    class _NotAnEngine:
        pass

    with pytest.raises(ValueError, match="unhandled engine"):
        should_use_metal(torch.zeros(3), engine=_NotAnEngine())


@pytest.mark.parametrize("engine", [Engine.EAGER, Engine.TRITON])
def test_should_use_metal_false_for_non_metal_engines(engine):
    assert should_use_metal(torch.zeros(3), engine=engine) is False


def test_should_use_metal_false_on_cpu_under_auto():
    """AUTO never raises -- it just declines the Metal path off MPS."""
    assert should_use_metal(torch.zeros(3), engine=Engine.AUTO) is False


def test_should_use_metal_raises_on_cpu_under_metal():
    """Forcing METAL on a non-MPS tensor is an error, mirroring TRITON."""
    with pytest.raises(RuntimeError, match="requires MPS float32"):
        should_use_metal(torch.zeros(3), engine=Engine.METAL)


def test_should_use_metal_raises_on_wrong_dtype_under_metal():
    """A forced engine must not silently downcast: float64 has no Metal kernel."""
    with pytest.raises(RuntimeError, match="requires MPS float32"):
        should_use_metal(torch.zeros(3, dtype=torch.float64), engine=Engine.METAL)


def test_should_use_metal_skips_none_but_still_checks_real_tensors():
    """``None`` entries are optional inputs and are skipped, but a real tensor
    beside them is still probed -- host-independent, since a CPU tensor is
    wrong for Metal everywhere."""
    with pytest.raises(RuntimeError, match="requires MPS float32"):
        should_use_metal(None, torch.zeros(3), None, engine=Engine.METAL)


# ---------------------------------------------------------------------------
# Characterization: facts that hold today and are not asserted anywhere else
# ---------------------------------------------------------------------------
# ``Engine.METAL`` selects the native Metal *density splat* and nothing else. Every
# other dispatch site is expected to run eager under it -- see the ``Engine`` docstring,
# which recommends scoping METAL with ``use_engine`` precisely because a process-wide
# METAL sends target math down the eager path.
#
# Today that fact exists only as ``if eng is Engine.EAGER or eng is Engine.METAL:
# return False`` inside ``should_use_triton``. Nothing asserts it, so a dispatch rework
# that treats "no backend claims this engine" as an error would turn
# ``with use_engine(Engine.METAL): loss = bond_target(xyz)`` from working into a
# RuntimeError, silently, in a documented usage pattern. These two tests are the guard.


def test_metal_engine_means_eager_at_non_density_sites():
    """``Engine.METAL`` must be *permissive* elsewhere, not an error.

    There are no Metal direct-summation or target kernels, so METAL at those sites means
    "run eager", not "fail". Asserted through the shared gate and through the targets
    wrapper, which is the entry point all twelve target math functions use.
    """
    from torchref.base.targets._dispatch import use_triton

    cuda_like = torch.zeros(3)  # CPU here; the engine short-circuits before device
    assert should_use_triton(cuda_like, engine=Engine.METAL) is False

    with use_engine(Engine.METAL):
        assert use_triton(cuda_like) is False


def test_metal_gate_does_not_probe_availability_on_a_wrong_device(monkeypatch):
    """Device/dtype must be rejected *before* shader availability is consulted.

    Two things depend on the ordering. Under a forced engine the error has to name the
    device mismatch (``requires MPS float32``) rather than a compile failure, and on a
    non-Apple host nothing should pay the cost of the availability probe. The same
    ordering is what keeps an MPS host from compiling the CPU C++ extension it will never
    use.

    Asserted by making the probe *fail loudly if reached*, rather than by inspecting
    ``sys.modules``: a module cannot be unimported, so an import check silently weakens to
    a no-op as soon as any earlier test has loaded the package -- which in a full run is
    always.
    """
    from torchref.base.electron_density.kernels.mps import compile as mps_compile

    def must_not_be_called():
        raise AssertionError(
            "availability was probed for a CPU tensor; the device check must "
            "short-circuit first"
        )

    monkeypatch.setattr(mps_compile, "mps_kernels_available", must_not_be_called)

    assert should_use_metal(torch.zeros(3), engine=Engine.AUTO) is False
    with pytest.raises(RuntimeError, match="requires MPS float32"):
        should_use_metal(torch.zeros(3), engine=Engine.METAL)
