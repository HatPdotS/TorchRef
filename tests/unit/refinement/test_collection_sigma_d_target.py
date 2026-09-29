"""The ``difference_sd`` collection row on a real dark/light pair.

Pinned on 1DAW with a 0.2 A shifted light model: the loss is finite, gradients reach
the light model through ``dF_calc`` only, the difference-power fit is owned by the
target, fitted on free reflections of the timepoint row with the reported sigmas taken
as calibrated, gives a positive unexplained power everywhere, is cached across forwards
and cleared by ``maintenance()``, and the fit summary reaches ``stats()``.
"""

import pytest
import torch

from torchref.refinement.model_error_estimation.difference_power import (
    DifferencePowerFit,
)
from torchref.refinement.targets.collection import xray as xray_module
from torchref.refinement.targets.collection import (
    CollectionDifferenceSigmaDTarget,
    CollectionSigmaDLossInputs,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def target(loaded_reflection_data, sample_structure_pair):
    """A dark/light collection whose light amplitudes carry a resolution-dependent
    difference proportional to ``F``, so the sigma_D coupling is not zero, and a light
    model shifted by 0.2 A."""
    from torchref import ReflectionData
    from torchref.cli._common import load_model
    from torchref.io import DatasetCollection
    from torchref.model import ModelCollection
    from torchref.scaling import CollectionScaler

    data = loaded_reflection_data
    g = torch.Generator().manual_seed(11)
    f = data.F
    dss = 1.0 / data.resolution**2
    change = (
        0.08 * f * torch.exp(-2.0 * dss) * torch.randn(len(data), generator=g).to(f)
    )
    light = ReflectionData.from_tensors(
        hkl=data.hkl,
        F=(f + change).clamp(min=0.0),
        F_sigma=data.F_sigma,
        cell=data.cell,
        spacegroup=data.spacegroup,
        rfree_flags=data.rfree_flags,
        device=str(data.device),
        verbose=0,
    )
    models = [
        load_model(
            str(sample_structure_pair["model"]),
            max_res=2.05,
            device=data.device,
            verbose=0,
        )
        for _ in range(2)
    ]
    with torch.no_grad():
        models[1].xyz.refinable_params += 0.2
    dc = DatasetCollection(device=data.device, verbose=0)
    dc.add_dataset("dark", data, set_as_reference=True).add_dataset("light", light)
    mc = ModelCollection(models, dark_key="dark", verbose=0)
    mc.add_dark().add_timepoint("light", [0.78, 0.22])
    scaler = CollectionScaler(dc, mc, verbose=0).initialize()
    return dc, mc, CollectionDifferenceSigmaDTarget(dc, mc, scaler=scaler)


def test_forward_is_finite_and_owns_its_fit(target):
    _dc, _mc, t = target
    loss = t.forward()
    assert torch.isfinite(loss)
    assert isinstance(t._fit, DifferencePowerFit)
    assert t._fit.sigma_scale == 1.0 and len(t._fit.alpha_coeffs) > 0
    ctx = t._loss_inputs()
    assert isinstance(ctx, CollectionSigmaDLossInputs)
    assert ctx.alpha.shape == ctx.beta_model.shape == (ctx.obs.shape[1],)
    assert not ctx.alpha.requires_grad and not ctx.beta_model.requires_grad
    assert torch.isfinite(ctx.alpha).all() and torch.isfinite(ctx.beta_model).all()
    assert (ctx.beta_model > 0).all()


def test_gradient_reaches_the_light_model(target):
    _dc, mc, t = target
    light = mc.base_models[1]
    light.zero_grad(set_to_none=True)
    t.forward().backward()
    grads = [p.grad for p in light.parameters() if p.grad is not None]
    assert grads and any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_fit_is_cached_until_maintenance(target):
    _dc, _mc, t = target
    t.forward()
    first = t._fit
    assert first is not None
    t.forward()
    assert t._fit is first
    t.maintenance()
    assert t._fit is None


def test_fit_uses_free_reflections_of_the_timepoint_row(target, monkeypatch):
    dc, _mc, t = target
    seen = {}
    real = xray_module.fit_difference_power

    def spy(delta_obs, sigma_diff, d_star_sq, **kw):
        seen["fit_mask"] = kw["fit_mask"].clone()
        seen["n"] = delta_obs.numel()
        return real(delta_obs, sigma_diff, d_star_sq, **kw)

    monkeypatch.setattr(xray_module, "fit_difference_power", spy)
    t.forward()
    n_hkl = dc.hkl.shape[0]
    assert seen["n"] == n_hkl
    free = dc["light"].free.mask.to(seen["fit_mask"].device)
    assert bool((seen["fit_mask"] & ~free).sum() == 0)
    assert int(seen["fit_mask"].sum()) > 0


def test_stats_carry_the_fit_summary(target):
    _dc, _mc, t = target
    t.forward()
    stats = t.stats()
    for key in (
        "difference_gamma",
        "difference_alpha_low_res",
        "difference_alpha_high_res",
    ):
        assert key in stats
