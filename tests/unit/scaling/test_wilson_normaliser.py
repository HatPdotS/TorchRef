"""Invariants of the absolute Wilson normaliser.

The load-bearing one is the first: ``<E^2> = 1`` is the *stationarity condition*
of the Gamma GLM's constant term, not a normalisation applied afterwards. Every
consumer that centres an intensity as ``E^2 - 1`` depends on it being exact
rather than approximate -- the previous convention centred against a measured
``<E^2> = 0.954`` and nothing noticed.

The rest pin the properties that make two fits comparable, which is what the
weighting design will be built on: a shared abscissa, epsilon reaching both
sides of the ratio, and invariance to the units the data arrive in.
"""

import pytest
import torch

from torchref.scaling.wilson import WilsonNormaliser
from torchref.symmetry import Cell, SpaceGroup

pytestmark = pytest.mark.unit


def _wilson_data(n=20000, seed=0, centric_frac=0.1, eps_value=None):
    """Intensities actually drawn from the distribution the fit assumes."""
    g = torch.Generator().manual_seed(seed)
    s = torch.rand(n, generator=g, dtype=torch.float64) * 0.45 + 0.05
    # A real curve to recover: Wilson falloff plus a low-resolution deficit of
    # the shape a missing bulk solvent produces.
    sigma = (torch.exp(-120.0 * (s / 2) ** 2) * 3000.0
             * (1 - 0.9 * torch.exp(-300.0 * (s / 2) ** 2)))
    centric = torch.zeros(n, dtype=torch.bool)
    centric[: int(n * centric_frac)] = True
    k = torch.where(centric, 0.5, 1.0).to(torch.float64)
    eps = (torch.ones(n, dtype=torch.float64) if eps_value is None
           else torch.full((n,), float(eps_value), dtype=torch.float64))
    # `_standard_gamma` takes no generator, so seed the global RNG too --
    # otherwise the draw depends on whatever ran before it and the test
    # passes alone and fails in a suite.
    torch.manual_seed(seed)
    I = torch._standard_gamma(k.clone()) / k * (eps * sigma)
    return I, s, eps, centric, sigma


def _k_weighted_mean(v, centric):
    k = torch.where(centric, 0.5, 1.0).to(torch.float64)
    return float((k * v.to(torch.float64)).sum() / k.sum())


def test_unit_mean_is_an_identity_of_the_fit():
    """The constant column's score equation IS ``<E^2> = 1``.

    ``sum_h k_h (I_h/mu_h - 1) = 0`` at the optimum, and the fit puts the
    intercept on it in closed form, so this does NOT degrade as the convergence
    tolerance is loosened. Measured 1e-9 to 2e-7 over five draws in float32; the
    bar is the package's usual 1e-4 relative, so a failure here means the
    intercept solve is broken rather than that the fit stopped early.
    """
    I, s, eps, centric, _ = _wilson_data()
    w = WilsonNormaliser(I, s, eps=eps, centric=centric, n_coeff=6)
    assert _k_weighted_mean(w.E_squared, centric) == pytest.approx(1.0, rel=1e-4)


@pytest.mark.parametrize("n_coeff", [1, 2, 6, 12])
def test_unit_mean_holds_at_every_order(n_coeff):
    """It is the intercept that pins the mean, so the order must not matter."""
    I, s, eps, centric, _ = _wilson_data()
    w = WilsonNormaliser(I, s, eps=eps, centric=centric, n_coeff=n_coeff)
    assert _k_weighted_mean(w.E_squared, centric) == pytest.approx(1.0, abs=1e-6)


def test_a_uniform_epsilon_cancels_out_of_the_ratio():
    """``E^2 = (I/eps) / <I/eps>``, so a constant eps must change nothing.

    This is the defect the previous convention carried: it divided the shell
    mean by eps without dividing the intensity by it, leaving ``<E^2> = <eps>``
    -- 1 on a primitive lattice and 2 on a centred one, so a normaliser whose
    absolute scale depended on the space group.
    """
    I, s, _, centric, _ = _wilson_data()
    plain = WilsonNormaliser(I, s, centric=centric, n_coeff=6)
    doubled = WilsonNormaliser(
        I, s, eps=torch.full_like(s, 2.0), centric=centric, n_coeff=6,
    )
    # eps=2 halves the intensity going in AND halves Sigma, so E is unchanged.
    assert torch.allclose(doubled.E, plain.E, rtol=1e-4, atol=1e-6)


def test_invariant_to_the_units_the_data_arrive_in():
    I, s, eps, centric, _ = _wilson_data()
    base = WilsonNormaliser(I, s, eps=eps, centric=centric, n_coeff=6).E
    for c in (1e-6, 1e6):
        scaled = WilsonNormaliser(
            I * c, s, eps=eps, centric=centric, n_coeff=6,
        ).E
        assert torch.allclose(scaled, base, rtol=1e-4, atol=1e-6), (
            f"scaling I by {c:g} moved E by "
            f"{float((scaled - base).abs().max()):.3e}"
        )


def test_the_resolution_trend_is_removed():
    I, s, eps, centric, _ = _wilson_data()
    w = WilsonNormaliser(I, s, eps=eps, centric=centric, n_coeff=6)
    order = torch.argsort(s)
    means = [float(w.E_squared[order[i::10]].to(torch.float64).mean())
             for i in range(10)]
    assert max(means) / min(means) < 1.15, f"residual trend: {means}"


def test_the_fitted_curve_recovers_the_true_one():
    I, s, eps, centric, sigma_true = _wilson_data(n=60000)
    w = WilsonNormaliser(I, s, eps=eps, centric=centric, n_coeff=6)
    rel = (w.sigma_wilson.to(torch.float64) / sigma_true - 1).abs()
    assert float(rel.median()) < 0.05
    assert float(rel.max()) < 0.30


def test_one_coefficient_is_a_single_global_scale():
    I, s, eps, centric, _ = _wilson_data()
    w = WilsonNormaliser(I, s, eps=eps, centric=centric, n_coeff=1)
    sig = w.sigma_wilson.to(torch.float64)
    assert float((sig / sig[0] - 1).abs().max()) < 1e-9


def test_the_range_only_matters_outside_the_fitted_data():
    """An affine remap does not change what a polynomial basis spans.

    So two fits over different ranges recover the same *function* where their
    data overlap -- the coefficients differ, the curve does not. What the range
    controls is the other side of it: ``u`` saturates at the ends, so beyond the
    fitted data the curve is frozen flat rather than extrapolated. That is the
    whole reason to pass an explicit range, and it is why a fit made on one
    reflection set can be used on another only if the range covers both.
    """
    I, s, eps, centric, _ = _wilson_data(n=30000)
    lo, hi = float(s.min()), float(s.max())
    sub = s < 0.3
    kw = dict(eps=eps[sub], centric=centric[sub], n_coeff=6)
    shared = WilsonNormaliser(I[sub], s[sub], s_lo=lo, s_hi=hi, **kw)
    own = WilsonNormaliser(I[sub], s[sub], **kw)

    # Looser than the package's usual 1e-4, and the reason is the point of the
    # test rather than an excuse. These are two INDEPENDENT fits, each stopped
    # when its own objective stops improving by 1e-4 of what it has gained. The
    # valley is flat along the high-order coefficients, so equal objectives
    # there do not mean equal coefficients, and the curves separate by more than
    # the objective did. Measured 0.2-1.6% over five draws; 3% catches a real
    # dependence on the parameterisation without chasing the stopping rule.
    inside = torch.linspace(0.06, 0.29, 40, dtype=torch.float64)
    assert torch.allclose(shared.evaluate(inside), own.evaluate(inside),
                          rtol=3e-2), "the fitted function must not depend on " \
                                      "how the basis was parameterised"

    # Outside its own data, the narrow fit is pinned at its endpoint; the one
    # given the full range keeps varying because it is still inside its basis.
    outside = torch.linspace(0.32, 0.49, 20, dtype=torch.float64)
    own_out = own.evaluate(outside).to(torch.float64)
    assert float((own_out / own_out[0] - 1).abs().max()) < 1e-9, \
        "beyond the fitted range the curve should be flat, not extrapolated"
    shared_out = shared.evaluate(outside).to(torch.float64)
    assert float((shared_out / shared_out[0] - 1).abs().max()) > 1e-3


def test_evaluate_reproduces_the_fitted_curve():
    I, s, eps, centric, _ = _wilson_data()
    w = WilsonNormaliser(I, s, eps=eps, centric=centric, n_coeff=6)
    assert torch.allclose(w.evaluate(s), w.sigma_wilson, rtol=1e-10, atol=1e-12)


def test_negative_intensities_are_kept_but_do_not_inform_the_fit():
    """Negative measurements are meaningful and unbiased; they are not errors.

    The Gamma likelihood has no support there, so they are held out of the
    estimate -- but they still get a Sigma and a signed ``E_squared``, because
    excluding them from the fit is not the same as refusing to normalise them.
    """
    I, s, eps, centric, _ = _wilson_data()
    I = I.clone()
    I[:500] = -torch.rand(500, dtype=torch.float64) * 10.0
    w = WilsonNormaliser(I, s, eps=eps, centric=centric, n_coeff=6)
    assert w.n_fitted == I.numel() - 500
    assert bool((w.E_squared[:500] < 0).all()), "sign must survive"
    assert bool(torch.isfinite(w.sigma_wilson).all())
    assert bool((w.E[:500] == 0).all()), "E clamps, E_squared does not"


def test_it_raises_rather_than_quietly_degrading():
    """No fallback. A normaliser that becomes a different normaliser on the
    hard cases is two normalisers wearing one name."""
    I, s, eps, centric, _ = _wilson_data(n=20)
    with pytest.raises(ValueError, match="usable reflections"):
        WilsonNormaliser(I[:3], s[:3], eps=eps[:3], centric=centric[:3],
                         n_coeff=6)


def test_from_hkl_excludes_systematic_absences():
    """Absences are zero by symmetry, not by measurement, so they say nothing
    about Sigma -- and a Gamma fit told otherwise is dragged toward zero."""
    sg = SpaceGroup("P 43 21 2")
    cell = Cell([70.0, 70.0, 90.0, 90.0, 90.0, 90.0])
    g = torch.Generator().manual_seed(5)
    hkl = torch.randint(-14, 15, (12000, 3), generator=g)
    hkl = hkl[hkl.abs().sum(dim=-1) > 0]
    absent = sg.is_absent(hkl).to(torch.bool)
    assert int(absent.sum()) > 0, "test needs a group with real absences"

    I = torch.rand(hkl.shape[0], generator=g, dtype=torch.float64) * 100 + 1
    I = torch.where(absent, torch.zeros_like(I), I)      # absences really are 0

    w = WilsonNormaliser.from_hkl(I, hkl, sg, cell, n_coeff=6)
    assert w.n_fitted == int((~absent).sum())
    assert bool(torch.isfinite(w.sigma_wilson).all())
    # The zeros must not have dragged the curve down.
    assert float(w.sigma_wilson.min()) > 0.0
