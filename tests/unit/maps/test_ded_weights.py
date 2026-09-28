"""The registered difference-coefficient weight schemes.

Pinned: the three schemes exist with their MTZ column names; ``none`` is flat;
``inverse_variance`` has mean one and floors a zero sigma; ``q`` gives strong
reflections more weight than weak ones where inverse variance cannot, keeps every
reflection at or above its floor when noise dominates, and falls back to inverse
variance with a warning that names why when too few reflections exist to fit.
"""

import pytest
import torch

from tests.unit.refinement.test_sigma_d import synth_diff
from torchref.maps.ded_weights import (
    DEFAULT_SCHEME,
    SCHEMES,
    WEIGHT_COLUMNS,
    DedWeightFallbackWarning,
    all_ded_weights,
    compute_ded_weights,
    normalise_mean_one,
)
from torchref.symmetry import SpaceGroup


def _inputs(n=20000, sig_frac=1.0, device="cpu"):
    d = synth_diff(n=n, sig_frac=sig_frac, device=device)
    g = torch.Generator().manual_seed(5)
    hkl = torch.randint(-20, 21, (n, 3), generator=g).to(device)
    cell = torch.tensor([40.0, 50.0, 60.0, 90.0, 90.0, 90.0], device=device)
    return d, hkl, cell, SpaceGroup("P 1", device=device)


@pytest.mark.unit
def test_registry_is_consistent():
    assert DEFAULT_SCHEME in SCHEMES
    assert set(WEIGHT_COLUMNS) == set(SCHEMES) - {"none"}
    with pytest.raises(ValueError):
        compute_ded_weights(
            "bogus",
            delta_obs=torch.zeros(3),
            sigma_diff=torch.ones(3),
            hkl=torch.zeros(3, 3),
            cell=torch.ones(6),
            spacegroup=None,
        )


@pytest.mark.unit
def test_normalise_mean_one_handles_nonfinite_and_zero():
    # Non-finite entries drop to zero and count in the mean, so the column mean is one
    # however many reflections carry weight.
    w = normalise_mean_one(torch.tensor([1.0, 3.0, float("nan"), float("inf")]))
    assert torch.allclose(w, torch.tensor([1.0, 3.0, 0.0, 0.0]))
    assert w.mean() == pytest.approx(1.0)
    half = normalise_mean_one(torch.tensor([0.0, 0.0, 2.0, 6.0]))
    assert torch.allclose(half, torch.tensor([0.0, 0.0, 1.0, 3.0]))
    z = normalise_mean_one(torch.zeros(4))
    assert torch.equal(z, torch.zeros(4))


@pytest.mark.unit
def test_none_and_inverse_variance(any_device):
    d, hkl, cell, sg = _inputs(n=2000, device=any_device)
    sig = d["sigma_diff"].clone()
    sig[0] = 0.0
    kw = {
        "delta_obs": d["delta_obs"],
        "sigma_diff": sig,
        "hkl": hkl,
        "cell": cell,
        "spacegroup": sg,
    }
    flat = compute_ded_weights("none", **kw)
    assert torch.equal(flat.weights, torch.ones_like(sig))
    ivw = compute_ded_weights("inverse_variance", **kw)
    assert ivw.applied == "inverse_variance"
    assert abs(float(ivw.weights.mean()) - 1.0) < 1e-5
    # The zero sigma is floored, so it carries the largest finite weight.
    assert torch.isfinite(ivw.weights).all()
    assert (
        float(ivw.weights[0]) == float(ivw.weights.max()) > float(ivw.weights[1:].max())
    )
    assert ivw.weights.device == d["delta_obs"].device


@pytest.mark.unit
def test_q_favours_strong_reflections_where_inverse_variance_cannot(any_device):
    d, hkl, cell, sg = _inputs(device=any_device)
    kw = {
        "delta_obs": d["delta_obs"],
        "sigma_diff": d["sigma_diff"],
        "hkl": hkl,
        "cell": cell,
        "spacegroup": sg,
        "f_dark": d["f_dark"],
    }
    every = all_ded_weights(**kw)
    assert set(every) == set(SCHEMES)
    q = every["q"]
    assert q.applied == "q" and abs(float(q.weights.mean()) - 1.0) < 1e-4
    assert 0.8 < q.diagnostics["gamma"] < 1.2
    assert q.diagnostics["converged"]
    # The strong half carries more weight; inverse variance cannot tell the halves apart
    # because the sigmas are constant. The floor compresses the weights into at most a
    # factor three, so the margin is smaller than an unbounded Wiener weight would give.
    f, w = d["f_dark"], q.weights
    strong, weak = f > f.median(), f <= f.median()
    assert float(w[strong].mean()) > 1.2 * float(w[weak].mean())
    ivw = every["inverse_variance"].weights
    assert abs(float(ivw[strong].mean()) - float(ivw[weak].mean())) < 1e-4


@pytest.mark.unit
def test_q_never_removes_a_reflection_when_noise_dominates():
    d, hkl, cell, sg = _inputs(n=5000, sig_frac=50.0)
    q = compute_ded_weights(
        "q",
        delta_obs=d["delta_obs"],
        sigma_diff=d["sigma_diff"] * 1.2,
        hkl=hkl,
        cell=cell,
        spacegroup=sg,
        f_dark=d["f_dark"],
    )
    assert q.applied == "q"
    floor = q.diagnostics["snr_floor"] / (1.0 + q.diagnostics["snr_floor"])
    assert q.diagnostics["weight_min"] >= floor - 1e-6
    assert bool((q.weights > 0).all())


@pytest.mark.unit
def test_too_few_reflections_fall_back_to_inverse_variance_with_a_warning():
    d, hkl, cell, sg = _inputs(n=5)
    kw = {
        "delta_obs": d["delta_obs"],
        "sigma_diff": d["sigma_diff"],
        "hkl": hkl,
        "cell": cell,
        "spacegroup": sg,
        "f_dark": d["f_dark"],
    }
    with pytest.warns(DedWeightFallbackWarning, match="inverse-variance"):
        q = compute_ded_weights("q", **kw)
    assert q.scheme == "q" and q.applied == "inverse_variance"
    assert "fallback_reason" in q.diagnostics
    ivw = compute_ded_weights("inverse_variance", **kw)
    assert torch.allclose(q.weights, ivw.weights)
