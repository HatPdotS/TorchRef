"""
Unit tests for :class:`~torchref.refinement.targets.wilson_prior.WilsonPriorTarget`.
"""

import os

import pytest
import torch

from torchref.io.datasets import ReflectionData
from torchref.model import EnsembleModel
from torchref.refinement.targets import WilsonPriorTarget
from torchref.scaling import Scaler


TEST_MTZ = os.path.join(
    os.path.dirname(__file__), "..", "..", "files", "mtz", "1DAW.mtz"
)
TEST_PDB = os.path.join(
    os.path.dirname(__file__), "..", "..", "files", "pdb", "1DAW.pdb"
)


@pytest.fixture
def setup_target():
    data = ReflectionData(verbose=0)
    data.load_mtz(TEST_MTZ)
    data._calculate_wilson_b()
    ens = EnsembleModel.from_single(
        TEST_PDB, n_members=4, perturb_sigma=0.0, b_const=5.0,
        seed=42, verbose=0, max_res=data.get_max_res(),
    )
    ens.cell = data.cell
    ens.spacegroup = data.spacegroup
    ens.setup_grid(max_res=data.get_max_res())

    scaler = Scaler(model=ens, data=data, nbins=10, verbose=0)
    fcalc0 = ens(data.hkl)
    scaler.initialize(fcalc0)

    target = WilsonPriorTarget(data=data, model=ens, scaler=scaler, nbins=15)
    return target, ens, data, scaler


def test_finite_loss(setup_target):
    target, _, _, _ = setup_target
    loss = target.forward()
    assert torch.isfinite(loss)


def test_loss_grows_when_model_is_shaken(setup_target):
    target, ens, _, _ = setup_target
    loss_low = target.forward().item()
    with torch.no_grad():
        ens.xyz.refinable_params += torch.randn_like(
            ens.xyz.refinable_params
        ) * 5.0
    if hasattr(ens, "reset_cache"):
        ens.reset_cache()
    loss_high = target.forward().item()
    assert loss_high >= loss_low, \
        f"Wilson loss did not grow after shake ({loss_low} -> {loss_high})"


def test_gradient_finite(setup_target):
    target, ens, _, _ = setup_target
    ens.xyz.refinable_params.grad = None
    loss = target.forward()
    loss.backward()
    g = ens.xyz.refinable_params.grad
    assert g is not None
    assert torch.isfinite(g).all()


# --------------------------------------------------------------------------
# per_reflection mode
# --------------------------------------------------------------------------

def _build(data, ens, scaler, mode):
    return WilsonPriorTarget(
        data=data, model=ens, scaler=scaler, nbins=15, mode=mode,
    )


def test_per_reflection_finite_and_grad(setup_target):
    _t, ens, data, scaler = setup_target
    target = _build(data, ens, scaler, "per_reflection")
    ens.xyz.refinable_params.grad = None
    loss = target.forward()
    assert torch.isfinite(loss)
    loss.backward()
    g = ens.xyz.refinable_params.grad
    assert g is not None and torch.isfinite(g).all()


def test_per_reflection_sees_within_bin_deviations(setup_target):
    """
    The whole point: per_reflection captures within-bin scatter that
    bin_mean averages away. By Jensen (mean-of-squares >= square-of-mean
    applied within each bin), the per-reflection penalty must be >= the
    bin-mean penalty for the same model — and strictly greater whenever
    individual reflections stray from the curve while their bin mean does
    not. Use a shaken model so reflections genuinely scatter.
    """
    _t, ens, data, scaler = setup_target
    with torch.no_grad():
        ens.xyz.refinable_params += torch.randn_like(
            ens.xyz.refinable_params
        ) * 2.0
    if hasattr(ens, "reset_cache"):
        ens.reset_cache()

    bin_loss = _build(data, ens, scaler, "bin_mean").forward().item()
    perrefl_loss = _build(data, ens, scaler, "per_reflection").forward().item()

    assert perrefl_loss >= bin_loss, (
        f"per_reflection ({perrefl_loss}) should capture at least as much "
        f"as bin_mean ({bin_loss})"
    )
    # On a shaken model there is real within-bin scatter, so it should be
    # strictly (substantially) larger, not just equal.
    assert perrefl_loss > 1.5 * bin_loss, (
        f"per_reflection ({perrefl_loss}) should be substantially larger "
        f"than bin_mean ({bin_loss}) — within-bin deviations are the signal"
    )


def test_invalid_mode_raises(setup_target):
    _t, ens, data, scaler = setup_target
    with pytest.raises(ValueError):
        WilsonPriorTarget(data=data, model=ens, scaler=scaler, mode="nonsense")


# --------------------------------------------------------------------------
# rice mode (zero-structure Wilson NLL)
# --------------------------------------------------------------------------

def test_rice_finite_and_grad(setup_target):
    _t, ens, data, scaler = setup_target
    target = _build(data, ens, scaler, "rice")
    ens.xyz.refinable_params.grad = None
    loss = target.forward()
    assert torch.isfinite(loss)
    # rice returns a SUM over the work reflections (matching the ML X-ray
    # target's reduction). Per-reflection it is an O(1) NLL in nats, so the
    # mean must be small/finite — not the thousands a mis-scaled penalty gives.
    n = int(target._refl_subset_idx.numel())
    assert -50.0 < float(loss.detach()) / n < 200.0
    loss.backward()
    g = ens.xyz.refinable_params.grad
    assert g is not None and torch.isfinite(g).all()


def test_rice_matches_ml_target_at_zero_centroid(setup_target):
    """
    The defining identity: the rice-mode Wilson prior is the ML X-ray Rice
    likelihood evaluated with the calc/centroid amplitude pinned to zero and
    the width pinned to sqrt(Sigma(s_h)). Feed the model amplitude through the
    eager ML math's F_obs slot with F_calc=0 and sigma=sqrt(Sigma); since both
    reduce by summing over the (work) reflections, the sums must be equal.
    """
    from torchref.base.targets.xray_ml import _ml_xray_loss_math_eager

    _t, ens, data, scaler = setup_target
    target = _build(data, ens, scaler, "rice")
    # First forward fits K/B_W and the bin/subset bookkeeping.
    got = target.forward()

    with torch.no_grad():
        idx = target._refl_subset_idx
        F_model = torch.abs(scaler(ens(data.hkl))).index_select(0, idx)
        res = data.resolution.index_select(0, idx)
        Sigma = target._wilson_curve(res).clamp(min=1e-6)
        centric = (
            data.centric.index_select(0, idx)
            if data.centric is not None
            else torch.zeros_like(F_model, dtype=torch.bool)
        )
        sigma = Sigma.sqrt()
        mask = torch.ones_like(F_model, dtype=torch.bool)
        ref_sum = _ml_xray_loss_math_eager(
            F_model, torch.zeros_like(F_model), sigma, centric, mask
        )

    assert torch.allclose(got.detach(), ref_sum, rtol=1e-4, atol=1e-4), (
        f"rice mode ({float(got)}) must equal the ML Rice target at zero "
        f"centroid ({float(ref_sum)})"
    )
