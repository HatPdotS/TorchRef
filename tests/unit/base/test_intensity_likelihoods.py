"""Intensity targets read measured intensities and propagate their uncertainties."""

import math

import pytest
import torch

from torchref.base.targets.xray_likelihoods import (
    VAR_FLOOR,
    amplitude_var_from_sigma_obs,
    floor_sigma_obs,
    gaussian_per_refl,
    intensity_var_from_sigma_obs,
    nll_per_refl,
)
from torchref.config import get_default_device, get_float_dtype


@pytest.mark.unit
def test_the_shared_gaussian_reproduces_the_amplitude_one_bitwise():
    """``nll_per_refl`` is ``gaussian_per_refl`` on ``|F_calc|``, exactly."""
    options = dict(dtype=get_float_dtype(), device=get_default_device())
    obs = torch.linspace(0.1, 100.0, 5000, **options)
    calc = torch.linspace(-100.0, 100.0, 5000, **options)
    var = amplitude_var_from_sigma_obs(torch.linspace(0.1, 10.0, 5000, **options))
    assert torch.equal(
        nll_per_refl(obs, calc, var), gaussian_per_refl(obs, calc.abs(), var)
    )


@pytest.mark.unit
def test_the_absolute_variance_floor_is_opt_out_and_matters(rtol):
    """``VAR_FLOOR`` is a distortion, not a safeguard, once the builder has floored
    sigma."""
    sigma = torch.full(
        (256,), 1e-3, dtype=get_float_dtype(), device=get_default_device()
    )
    var = intensity_var_from_sigma_obs(sigma)  # 1e-6, comfortably above VAR_FLOOR
    obs = torch.zeros(256, dtype=get_float_dtype(), device=get_default_device())
    model = torch.full(
        (256,), 1e-4, dtype=get_float_dtype(), device=get_default_device()
    )
    assert torch.equal(
        gaussian_per_refl(obs, model, var, var_floor=0.0),
        gaussian_per_refl(obs, model, var, var_floor=VAR_FLOOR),
    ), "the floor must be inert when the variance is above it"

    # Below it, the two differ -- and by a lot, not by an ulp.
    tiny = torch.full(
        (256,), 1e-6, dtype=get_float_dtype(), device=get_default_device()
    )  # var = 1e-12 << VAR_FLOOR
    var_tiny = intensity_var_from_sigma_obs(tiny)
    free = gaussian_per_refl(obs, model, var_tiny, var_floor=0.0)
    clamped = gaussian_per_refl(obs, model, var_tiny, var_floor=VAR_FLOOR)
    assert not torch.allclose(free, clamped)
    # `free` is the honest one: it uses the variance the builder actually produced.
    expected = (
        0.5 * (1e-4) ** 2 / 1e-12 + 0.5 * math.log(1e-12) + 0.5 * math.log(2 * math.pi)
    )
    assert free[0].item() == pytest.approx(expected, rel=rtol)


@pytest.mark.unit
def test_the_intensity_sigma_floor_respects_the_fitted_subset():
    """``mask`` restricts the median, because unfitted rows carry filler."""
    # The first 20 fitted rows are BELOW the fitted median's floor, so the floor is what
    # they come back as -- which is the only way to observe which median was used.
    sigma = torch.cat(
        [
            torch.full(
                (20,), 0.01, device=get_default_device(), dtype=get_float_dtype()
            ),  # fitted, and below floor either way
            torch.full(
                (80,), 10.0, device=get_default_device(), dtype=get_float_dtype()
            ),  # fitted, sets the fitted median
            torch.full(
                (900,), 1e6, device=get_default_device(), dtype=get_float_dtype()
            ),  # NOT fitted: filler
        ]
    )
    mask = torch.cat(
        [
            torch.ones(100, dtype=torch.bool, device=get_default_device()),
            torch.zeros(900, dtype=torch.bool, device=get_default_device()),
        ]
    )
    masked = floor_sigma_obs(sigma, mask, abs_floor=1e-12)
    unmasked = floor_sigma_obs(sigma, None, abs_floor=1e-12)
    assert masked[:20].min().item() == pytest.approx(1.0)  # floor = 10 * 0.1
    assert unmasked[:20].min().item() == pytest.approx(
        1e5
    )  # floor = 1e6 * 0.1, swamped
    # An explicit floor overrides the median entirely -- the set-independent path.
    assert floor_sigma_obs(sigma, mask, floor=0.5)[:20].min().item() == pytest.approx(
        0.5
    )
