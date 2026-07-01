"""Tests for AdamWithAdaptiveNoise.update_noise_scale overfitting signal.

Guards the sign bug where the noise scale was driven by
``log(train_nll) - log(test_nll)`` (negative under overfitting → clamped to
zero → no noise), the opposite of the intended behaviour.
"""
import pytest
import torch

from torchref.refinement.optimizers.adam_noise import AdamWithAdaptiveNoise


def _make_opt():
    p = torch.nn.Parameter(torch.zeros(3))
    return AdamWithAdaptiveNoise([p], lr=1e-3, update_weight=1.0)


@pytest.mark.unit
def test_noise_increases_under_overfitting():
    """test_nll > train_nll (overfitting) must drive a positive noise scale."""
    opt = _make_opt()
    opt.update_noise_scale(
        train_nll=torch.tensor(1.0), test_nll=torch.tensor(2.0)
    )
    assert opt.noise_scale > 0.0


@pytest.mark.unit
def test_no_noise_when_not_overfitting():
    """train_nll >= test_nll (no overfitting) must leave the noise scale at zero."""
    opt = _make_opt()
    opt.update_noise_scale(
        train_nll=torch.tensor(2.0), test_nll=torch.tensor(1.0)
    )
    assert opt.noise_scale == 0.0


@pytest.mark.unit
def test_noise_scale_is_clamped():
    """The per-update contribution is clamped to <= 0.1 even for huge gaps."""
    opt = _make_opt()
    opt.update_noise_scale(
        train_nll=torch.tensor(1e-3), test_nll=torch.tensor(1e3)
    )
    assert 0.0 < opt.noise_scale <= 0.1 + 1e-6
