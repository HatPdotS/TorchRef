"""Pin the amplitude-metric Gaussian likelihood's value and reduction contract."""

import math

import pytest
import torch

from torchref.base.metrics.loss import estimate_sigma_F, nll_xray, nll_xray_mean
from torchref.config import get_default_device, get_float_dtype

pytestmark = pytest.mark.unit


def test_gaussian_nll_value_and_reduction() -> None:
    """The NLL includes its normalization and sums over reflections."""
    obs = torch.tensor(
        [10.0, 20.0, 30.0], dtype=get_float_dtype(), device=get_default_device()
    )
    sigma = obs.new_tensor([1.0, 2.0, 4.0])
    calc = obs + sigma
    expected = obs.new_tensor(1.5 + math.log(8.0) + 1.5 * math.log(2.0 * math.pi))

    torch.testing.assert_close(nll_xray(obs, calc, sigma), expected)
    torch.testing.assert_close(nll_xray_mean(obs, calc, sigma), expected / obs.numel())


def test_gaussian_nll_penalizes_amplitude_error() -> None:
    """A one-sigma residual adds one half per reflection to the perfect-fit NLL."""
    obs = torch.tensor(
        [10.0, 20.0, 30.0], dtype=get_float_dtype(), device=get_default_device()
    )
    sigma = torch.ones_like(obs)
    good = nll_xray(obs, obs, sigma)
    bad = nll_xray(obs, obs + sigma, sigma)
    torch.testing.assert_close(bad - good, obs.new_tensor(1.5))


def test_estimated_sigma_ignores_missing_amplitudes() -> None:
    """A missing amplitude leaves only its own estimated sigma NaN."""
    F = torch.tensor(
        [10.0, float("nan"), 20.0], dtype=get_float_dtype(), device=get_default_device()
    )
    sigma = estimate_sigma_F(F)
    torch.testing.assert_close(sigma[[0, 2]], F.new_tensor([0.65, 1.15]))
    assert torch.isnan(sigma[1])
