"""The translation likelihood's variance convention, pinned as a pdf.

A likelihood is only right up to what its variance argument means, and that is
exactly the sort of thing that survives code review and unit tests written
against the implementation rather than against the distribution. The alignment
package carried its own Rice and Woolfson for a while, parameterised by the
*amplitude* variance, and handed both branches the same number -- which put
acentrics at twice their intended variance on 90-95% of reflections, for as long
as nobody integrated the thing.

These tests integrate it. They assert the property the pipeline actually depends
on: at ``D = 0`` the likelihood must believe ``<E^2> = 1``, because
:class:`~torchref.scaling.WilsonNormaliser` makes ``<E^2> = 1`` an identity of
the fit that produced ``E_obs``. Anything else means the model and the data
disagree about the scale of the very quantity being compared.
"""
import pytest
import torch

from torchref.base.targets.xray_likelihoods import rice_per_refl

pytestmark = pytest.mark.unit


def _pdf_moments(logp, F, dF):
    """Norm and second moment of ``exp(logp)`` treated as a density in ``F``."""
    p = torch.exp(logp)
    norm = float((p * dF).sum())
    return norm, float((p * F**2 * dF).sum()) / norm


@pytest.fixture(scope="module")
def grid():
    F = torch.linspace(1e-6, 12.0, 200001, dtype=torch.float64)
    return F, float(F[1] - F[0])


@pytest.mark.parametrize("centric", [False, True])
def test_unit_sigma_means_unit_second_moment(grid, centric):
    """At Sigma = 1 and no model, the likelihood expects <E^2> = 1.

    This is the property the whole sigma_A path rests on. Both branches must
    satisfy it from the SAME Sigma -- that is what makes a single complex
    variance the right parameterisation and an amplitude variance the wrong one.
    """
    F, dF = grid
    ll = -rice_per_refl(F, torch.zeros_like(F), torch.ones_like(F),
                        torch.full_like(F, centric, dtype=torch.bool))
    norm, m2 = _pdf_moments(ll, F, dF)
    assert norm == pytest.approx(1.0, rel=1e-4), "not a normalised density"
    assert m2 == pytest.approx(1.0, rel=1e-4), (
        f"{'centric' if centric else 'acentric'} branch expects <E^2> = {m2:.4f} "
        f"at Sigma = 1; the observations have <E^2> = 1 by construction"
    )


@pytest.mark.parametrize("sigma", [0.25, 0.5, 2.0])
@pytest.mark.parametrize("centric", [False, True])
def test_second_moment_tracks_sigma(grid, sigma, centric):
    """<E^2> = Sigma with no model, for both branches. Fixes the scale, not just the shape."""
    F, dF = grid
    ll = -rice_per_refl(F, torch.zeros_like(F), torch.full_like(F, sigma),
                        torch.full_like(F, centric, dtype=torch.bool))
    norm, m2 = _pdf_moments(ll, F, dF)
    assert norm == pytest.approx(1.0, rel=1e-4)
    assert m2 == pytest.approx(sigma, rel=1e-4)


@pytest.mark.parametrize("centric", [False, True])
def test_second_moment_with_a_model_present(grid, centric):
    """With a model, <E^2> = Sigma + Fc^2 -- the signal adds to the noise.

    The sigma_A likelihood is evaluated at ``Fc = D E_calc`` and
    ``Sigma = 1 - D^2``, so on data normalised to ``<E_calc^2> = 1`` this gives
    ``<E^2> = 1`` for every D. That invariance is why D is identifiable at all,
    and it fails if the two branches disagree about what the variance means.
    """
    F, dF = grid
    D, E_calc = 0.6, 1.0
    Fc, Sigma = D * E_calc, 1.0 - D * D
    ll = -rice_per_refl(F, torch.full_like(F, Fc), torch.full_like(F, Sigma),
                        torch.full_like(F, centric, dtype=torch.bool))
    norm, m2 = _pdf_moments(ll, F, dF)
    assert norm == pytest.approx(1.0, rel=1e-4)
    assert m2 == pytest.approx(Sigma + Fc**2, rel=1e-4)
    assert m2 == pytest.approx(1.0, rel=1e-4), "D must not change the expected <E^2>"
