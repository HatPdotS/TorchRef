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


@pytest.mark.unit
def test_the_shared_gaussian_reproduces_the_amplitude_one_bitwise():
    """``nll_per_refl`` is ``gaussian_per_refl`` on ``|F_calc|``, exactly."""
    for dtype in (torch.float32, torch.float64):
        torch.manual_seed(3)
        F_obs = torch.rand(5000, dtype=dtype) * 100
        F_calc = torch.randn(5000, dtype=dtype) * 100
        var = amplitude_var_from_sigma_obs(torch.rand(5000, dtype=dtype) * 10)
        assert torch.equal(
            nll_per_refl(F_obs, F_calc, var),
            gaussian_per_refl(F_obs, torch.abs(F_calc), var),
        )


@pytest.mark.unit
def test_the_absolute_variance_floor_is_opt_out_and_matters():
    """``VAR_FLOOR`` is a distortion, not a safeguard, once the builder has floored
    sigma."""
    sigma = torch.full((256,), 1e-3, dtype=torch.float64)
    var = intensity_var_from_sigma_obs(sigma)  # 1e-6, comfortably above VAR_FLOOR
    obs = torch.zeros(256, dtype=torch.float64)
    model = torch.full((256,), 1e-4, dtype=torch.float64)
    assert torch.equal(
        gaussian_per_refl(obs, model, var, var_floor=0.0),
        gaussian_per_refl(obs, model, var, var_floor=VAR_FLOOR),
    ), "the floor must be inert when the variance is above it"

    # Below it, the two differ -- and by a lot, not by an ulp.
    tiny = torch.full((256,), 1e-6, dtype=torch.float64)  # var = 1e-12 << VAR_FLOOR
    var_tiny = intensity_var_from_sigma_obs(tiny)
    free = gaussian_per_refl(obs, model, var_tiny, var_floor=0.0)
    clamped = gaussian_per_refl(obs, model, var_tiny, var_floor=VAR_FLOOR)
    assert not torch.allclose(free, clamped)
    # `free` is the honest one: it uses the variance the builder actually produced.
    expected = (
        0.5 * (1e-4) ** 2 / 1e-12 + 0.5 * math.log(1e-12) + 0.5 * math.log(2 * math.pi)
    )
    assert free[0].item() == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_the_intensity_sigma_floor_respects_the_fitted_subset():
    """``mask`` restricts the median, because unfitted rows carry filler."""
    # The first 20 fitted rows are BELOW the fitted median's floor, so the floor is what
    # they come back as -- which is the only way to observe which median was used.
    sigma = torch.cat(
        [
            torch.full((20,), 0.01),  # fitted, and below floor either way
            torch.full((80,), 10.0),  # fitted, sets the fitted median
            torch.full((900,), 1e6),  # NOT fitted: filler
        ]
    )
    mask = torch.cat(
        [torch.ones(100, dtype=torch.bool), torch.zeros(900, dtype=torch.bool)]
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


@pytest.fixture(scope="module")
def refinement(pdb_dir, mtz_dir):
    """A scaled 1DAW refinement -- the only fixture carrying BOTH I/SIGI and FP/SIGFP,
    so the only one on which the two observables can be compared at all."""
    pdb = pdb_dir / "1DAW.pdb"
    mtz = mtz_dir / "1DAW.mtz"
    if not (pdb.exists() and mtz.exists()):
        pytest.skip("1DAW fixture not present")
    from torchref import LBFGSRefinement

    ref = LBFGSRefinement(data_file=str(mtz), pdb=str(pdb), target_mode="ml", verbose=0)
    ref.get_scales()
    return ref


def _t(refinement, mode, use_set="work"):
    from torchref.refinement.targets.xray.factory import create_xray_target

    return create_xray_target(
        data=refinement.reflection_data,
        model=refinement.model,
        scaler=refinement.scaler,
        mode=mode,
        use_set=use_set,
    )


@pytest.mark.integration
def test_the_row_is_selectable_and_reads_intensities(refinement):
    """``nll_i`` comes out of the factory and its ``get_data`` returns the I columns."""
    t = _t(refinement, "nll_i")
    obs, calc, sigma, centric, sub = t.get_data()
    data = refinement.reflection_data

    torch.testing.assert_close(obs, data.work.I)
    torch.testing.assert_close(sigma, data.work.sigI)
    # The model is the SQUARED scaled amplitude, not the amplitude.
    torch.testing.assert_close(calc, sub.select(t.get_F_calc_scaled(recalc=False) ** 2))
    assert obs.shape == calc.shape == sigma.shape == (sub.n,)
    assert centric.shape == (sub.n,)


@pytest.mark.integration
def test_the_intensity_model_is_the_squared_scaled_amplitude(refinement):
    """``get_I_calc_scaled`` squares the SCALED amplitude, not the raw one."""
    t = _t(refinement, "nll_i")
    with torch.no_grad():
        amp = t.get_F_calc_scaled(recalc=False)
        inten = t.get_I_calc_scaled(recalc=False)
    torch.testing.assert_close(inten, amp**2, rtol=1e-6, atol=1e-6)


@pytest.mark.integration
def test_rfactors_stay_on_amplitudes(refinement):
    """An intensity row reports the SAME R-factors as an amplitude row."""
    r_i = _t(refinement, "nll_i").get_rfactor()
    r_a = _t(refinement, "nll").get_rfactor()
    assert r_i == pytest.approx(r_a, abs=1e-9)


@pytest.mark.integration
def test_the_loss_is_finite_differentiable_and_summed(refinement):
    t = _t(refinement, "nll_i")
    loss = t.forward()
    assert torch.isfinite(loss) and loss.ndim == 0
    loss.backward()
    grads = [
        p.grad
        for p in refinement.model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert grads, "no gradient reached the model"
    assert all(torch.isfinite(g).all() for g in grads)
    refinement.model.zero_grad(set_to_none=True)


@pytest.mark.integration
@pytest.mark.parametrize("use_set", ["work", "free"])
def test_a_reflections_residual_does_not_depend_on_the_arrays_length(
    refinement, use_set
):
    """``residuals()`` restricted to a subset must equal ``forward()`` on that subset."""
    t = _t(refinement, "nll_i", use_set=use_set)
    sub = t._subset()
    with torch.no_grad():
        fwd = t.forward()
        summed = t.residuals().index_select(0, sub.indices).sum()
    torch.testing.assert_close(summed, fwd, rtol=1e-6, atol=1e-6)


@pytest.mark.integration
def test_missing_intensities_raise_at_construction_not_at_forward(refinement):
    """LossState probes ``forward()`` at registration, so a missing column has to be
    caught in ``__init__`` or it surfaces from deep inside setup with no mention of why.
    """
    import copy

    data = copy.copy(refinement.reflection_data)
    data.I = None
    from torchref.refinement.targets.xray import NLLIntensityXrayTarget

    with pytest.raises(ValueError, match="dataset carries none"):
        NLLIntensityXrayTarget(
            data=data, model=refinement.model, scaler=refinement.scaler
        )
