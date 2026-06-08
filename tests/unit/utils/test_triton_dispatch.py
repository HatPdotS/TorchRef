"""Unit tests for the shared Triton/eager dispatch engine (CPU-only)."""

import pytest
import torch

from torchref.utils import (
    Engine,
    get_engine,
    set_engine,
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
