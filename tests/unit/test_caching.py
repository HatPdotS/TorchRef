"""Tests for the forward-result cache and its global on/off switch.

Covers ``CachedForwardMixin``'s cache-hit contract (previously only asserted through
``torch.allclose``, which passes with the cache both on and off), the
``torchref.config.caching`` flag that disables it, and the ``no_caching()`` context manager.
"""

import pytest
import torch
import torch.nn as nn

import torchref.config as cfg
from torchref.utils import CachedForwardMixin, no_caching


class _CountingModule(CachedForwardMixin, nn.Module):
    """Cached module that records how many times ``forward`` actually ran."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.arange(3, dtype=torch.float32))
        self.n_forward = 0

    def forward(self, scale=2.0):
        self.n_forward += 1
        return self.p * scale


@pytest.fixture(autouse=True)
def restore_caching_flag():
    """Keep a test's flag mutations from leaking into the rest of the session."""
    previous = cfg.caching.value
    yield
    cfg.caching.value = previous


@pytest.fixture
def module():
    return _CountingModule()


# ---------------------------------------------------------------------------
# Caching enabled (the default)
# ---------------------------------------------------------------------------


def test_cache_hit_returns_the_same_object(module):
    cfg.caching.value = True

    first = module()
    second = module()

    assert second is first
    assert module.n_forward == 1


def test_default_is_enabled():
    assert cfg.caching.value is True
    assert cfg.get_caching_enabled() is True


def test_changed_argument_still_misses_while_enabled(module):
    cfg.caching.value = True

    first = module(scale=2.0)
    second = module(scale=3.0)

    assert second is not first
    assert module.n_forward == 2
    assert torch.allclose(second, module.p * 3.0)


# ---------------------------------------------------------------------------
# Caching disabled
# ---------------------------------------------------------------------------


def test_disabled_recomputes_every_call(module):
    cfg.caching.value = False

    first = module()
    second = module()

    assert second is not first
    assert module.n_forward == 2
    assert torch.allclose(second, first)


def test_disabled_leaves_no_cached_output(module):
    cfg.caching.value = False

    module()

    assert getattr(module, "_fwd_cached_output", None) is None


def test_disabled_keeps_the_autograd_graph(module):
    cfg.caching.value = False

    module().sum().backward()

    assert module.p.grad is not None
    assert torch.allclose(module.p.grad, torch.full((3,), 2.0))


def test_recalc_is_consumed_while_disabled(module):
    cfg.caching.value = False

    # ``forward`` takes no ``recalc``; a leak through would raise TypeError.
    result = module(recalc=True)

    assert torch.allclose(result, module.p * 2.0)
    assert module.n_forward == 1


# ---------------------------------------------------------------------------
# Flipping the flag mid-life
# ---------------------------------------------------------------------------


def test_disabling_drops_an_existing_cached_result(module):
    cfg.caching.value = True
    cached = module()
    assert module._fwd_cached_output is cached

    cfg.caching.value = False
    recomputed = module()

    assert recomputed is not cached
    assert getattr(module, "_fwd_cached_output", None) is None


def test_reenabling_does_not_serve_a_pre_flip_result(module):
    cfg.caching.value = True
    stale = module()

    cfg.caching.value = False
    module()  # drops the cache

    # Mutate the parameter the way an optimizer step would.
    with torch.no_grad():
        module.p.add_(10.0)

    cfg.caching.value = True
    fresh = module()

    assert fresh is not stale
    assert torch.allclose(fresh, module.p * 2.0)


# ---------------------------------------------------------------------------
# no_caching()
# ---------------------------------------------------------------------------


def test_no_caching_disables_inside_and_restores_after(module):
    cfg.caching.value = True

    with no_caching():
        assert cfg.caching.value is False
        first = module()
        second = module()

    assert cfg.caching.value is True
    assert second is not first
    assert module.n_forward == 2


def test_no_caching_restores_on_exception():
    cfg.caching.value = True

    with pytest.raises(RuntimeError):
        with no_caching():
            raise RuntimeError("boom")

    assert cfg.caching.value is True


def test_no_caching_restores_a_previously_disabled_flag():
    cfg.caching.value = False

    with no_caching():
        assert cfg.caching.value is False

    assert cfg.caching.value is False


def test_no_caching_nests():
    cfg.caching.value = True

    with no_caching():
        with no_caching():
            assert cfg.caching.value is False
        assert cfg.caching.value is False

    assert cfg.caching.value is True


# ---------------------------------------------------------------------------
# Configuration surface
# ---------------------------------------------------------------------------


def test_env_unset_defaults_to_enabled(monkeypatch):
    monkeypatch.delenv("TORCHREF_CACHING", raising=False)

    assert cfg.CachingConfig().value is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", " off "])
def test_env_falsy_values_disable(monkeypatch, raw):
    monkeypatch.setenv("TORCHREF_CACHING", raw)

    assert cfg.CachingConfig().value is False


@pytest.mark.parametrize("raw", ["1", "true", "YES", "on"])
def test_env_truthy_values_enable(monkeypatch, raw):
    monkeypatch.setenv("TORCHREF_CACHING", raw)

    assert cfg.CachingConfig().value is True


def test_env_rejects_unrecognised_value(monkeypatch):
    """A typo must not silently switch caching off."""
    monkeypatch.setenv("TORCHREF_CACHING", "maybe")

    with pytest.raises(ValueError, match="TORCHREF_CACHING"):
        cfg.CachingConfig()


def test_setter_rejects_non_bool():
    with pytest.raises(TypeError):
        cfg.caching.value = 1


def test_repr_reports_the_value():
    cfg.caching.value = False

    assert repr(cfg.caching) == "CachingConfig(value=False)"
