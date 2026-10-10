"""Intensity targets consume measured observations and their uncertainties."""

import pytest
import torch

pytestmark = pytest.mark.integration


@pytest.fixture
def refinement(sample_structure_pair):
    """Fit a fresh 1DAW refinement for each mutable target check."""
    from torchref import LBFGSRefinement

    ref = LBFGSRefinement(
        data_file=str(sample_structure_pair["reflections"]),
        pdb=str(sample_structure_pair["model"]),
        target_mode="ml",
        verbose=0,
    )
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


def test_the_intensity_model_is_the_squared_scaled_amplitude(refinement):
    """``get_I_calc_scaled`` squares the SCALED amplitude, not the raw one."""
    t = _t(refinement, "nll_i")
    with torch.no_grad():
        amp = t.get_F_calc_scaled(recalc=False)
        inten = t.get_I_calc_scaled(recalc=False)
    torch.testing.assert_close(inten, amp**2, rtol=1e-6, atol=1e-6)


def test_rfactors_stay_on_amplitudes(refinement):
    """An intensity row reports the SAME R-factors as an amplitude row."""
    r_i = _t(refinement, "nll_i").get_rfactor()
    r_a = _t(refinement, "nll").get_rfactor()
    assert r_i == pytest.approx(r_a, abs=1e-9)


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
