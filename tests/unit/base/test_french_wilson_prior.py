"""The Wilson prior French-Wilson converts with, and the guard on it.

French-Wilson needs a positive mean intensity ``Sigma``. With ``Sigma <= 0`` the
``-sigma/Sigma`` term of ``h`` changes sign, and an observation no Wilson
reflection could explain is converted as a strong one with a tiny ``sigma_F``.
These tests pin that such a row is never converted, that the fitted prior stays
positive where a shell mean does not, and that it still follows the mean
intensity: on deposited data, when sigma grows with the intensity, and next to
a single reflection whose sigma dwarfs everything around it.
"""

from functools import partial

import pytest
import torch

from torchref.base.french_wilson import (
    _bspline,
    fit_mean_intensity,
    french_wilson,
    french_wilson_h,
    french_wilson_valid_mask,
)
from torchref.io.datasets.french_wilson import _anisotropy_design, french_wilson_auto
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.symmetry import Cell, SpaceGroup


def _wilson_data(n, seed, sigma_of_J, d_max=20.0, d_min=2.0, signal_beyond=None):
    """Acentric Wilson intensities measured with Gaussian error.

    Reflections are sampled uniformly in reciprocal volume, as a real dataset's
    are, and the true ``Sigma`` falls off as ``exp(-20/d^2)``. Past
    ``signal_beyond`` (a d-spacing in Å) the true intensity is zero, so those
    rows are pure noise.

    Returns ``d`` (Å, descending), the true ``Sigma``, ``I`` and ``sigma``.
    """
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n, generator=g)
    s = (d_max**-3 + u * (d_min**-3 - d_max**-3)) ** (1.0 / 3.0)
    d = torch.sort(1.0 / s, descending=True).values
    Sigma = 1000.0 * torch.exp(-20.0 / d**2)
    J = Sigma * -torch.log(torch.rand(n, generator=g))
    if signal_beyond is not None:
        J = torch.where(d < signal_beyond, torch.zeros_like(J), J)
    sigma = sigma_of_J(J)
    return d, Sigma, J + sigma * torch.randn(n, generator=g), sigma


def _hkl(n, centric_every=None):
    """Distinct Miller indices; every ``centric_every``-th row has k = 0.

    In P 1 21 1 the h0l zone is centric, so this controls the centric fraction.
    """
    idx = torch.arange(n)
    k = idx % 41 + 1
    if centric_every is not None:
        k = torch.where(idx % centric_every == 0, torch.zeros_like(k), k)
    return torch.stack([idx % 37 + 1, k, idx // 37 + 1], dim=1)


def _inconsistent(F, keep, I, sigma_I):
    """Kept amplitudes whose square sits more than 6 sigma above the measurement.

    A posterior mean cannot do that: shrinkage towards a positive prior only
    ever pulls ``F^2`` below ``I`` once ``I`` is clear of the noise.
    """
    return keep & (F * F > I + 6.0 * sigma_I)


def _shell_means(I, d, n_shells):
    """Unweighted mean intensity of equal-count resolution shells, per row."""
    order = torch.argsort(d, descending=True)
    shell = torch.empty_like(order)
    shell[order] = torch.arange(len(d)) * n_shells // len(d)
    sums = torch.zeros(n_shells, dtype=I.dtype).index_add_(0, shell, I)
    counts = torch.zeros(n_shells, dtype=I.dtype).index_add_(
        0, shell, torch.ones_like(I)
    )
    return (sums / counts)[shell]


# =============================================================================
# The guard
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize("centric", [False, True])
def test_h_is_undefined_without_a_positive_prior(centric):
    I = torch.tensor([-6.1, 23.3, 0.5, 40.0])
    sigma_I = torch.tensor([14.8, 9.3, 1.0, 5.0])
    Sigma = torch.tensor([-0.035, -0.039, 0.0, 80.0])

    h = french_wilson_h(I, sigma_I, Sigma, is_centric=centric)

    assert torch.isnan(h[:3]).all()
    assert torch.isfinite(h[3])
    assert not french_wilson_valid_mask(I, sigma_I, Sigma, centric)[:3].any()


@pytest.mark.unit
@pytest.mark.parametrize("centric", [False, True])
def test_non_positive_prior_gives_no_amplitude(centric):
    """A row without a positive prior is neither kept nor given a number."""
    I = torch.tensor([-6.1, 23.3, 0.5, 40.0])
    sigma_I = torch.tensor([14.8, 9.3, 1.0, 5.0])
    Sigma = torch.tensor([-0.035, -0.039, 0.0, 80.0])
    is_centric = torch.full((4,), centric)

    F, sigma_F, keep = french_wilson(I, sigma_I, Sigma, is_centric=is_centric)

    assert not keep[:3].any()
    assert torch.isnan(F[:3]).all() and torch.isnan(sigma_F[:3]).all()
    # The row with a prior converts exactly as it does on its own.
    F_alone, sigma_F_alone, keep_alone = french_wilson(
        I[3:], sigma_I[3:], Sigma[3:], is_centric=is_centric[3:]
    )
    torch.testing.assert_close(F[3:], F_alone)
    torch.testing.assert_close(sigma_F[3:], sigma_F_alone)
    assert bool(keep[3]) and bool(keep_alone[0])


@pytest.mark.unit
@pytest.mark.parametrize("centric", [False, True])
def test_rejection_threshold_does_not_move_the_posterior(centric):
    """The cut decides which rows are kept, not what their amplitudes are."""
    I = torch.linspace(-35.0, 300.0, 400)
    sigma_I = torch.full_like(I, 10.0)
    Sigma = torch.full_like(I, 80.0)
    is_centric = torch.full_like(I, centric, dtype=torch.bool)

    F, sigma_F, keep = french_wilson(I, sigma_I, Sigma, is_centric)
    F_strict, sigma_F_strict, keep_strict = french_wilson(
        I, sigma_I, Sigma, is_centric, min_i_over_sigma=-2.0
    )

    torch.testing.assert_close(F_strict, F)
    torch.testing.assert_close(sigma_F_strict, sigma_F)
    assert bool(keep.all())
    assert int(keep_strict.sum()) < int(keep.sum())


@pytest.mark.unit
@pytest.mark.parametrize(
    "centric, h, mean, sd",
    [
        # Posterior moments of F / sqrt(sigma_I) by quadrature of the truncated
        # posterior, at 40 digits.
        (False, -6.0, 0.3538422, 0.18242341),
        (False, -20.0, 0.19773715, 0.1032138),
        (True, -6.0, 0.22663359, 0.16979324),
        (True, -20.0, 0.12596102, 0.095084145),
    ],
)
def test_posterior_below_the_tables_matches_the_integral(centric, h, mean, sd):
    sigma_I = torch.tensor([4.0])
    # Sigma chosen so that I = 0 lands on this h.
    Sigma = sigma_I / (-h * (2.0 if centric else 1.0))
    F, sigma_F, keep = french_wilson(
        torch.zeros(1), sigma_I, Sigma, torch.tensor([centric])
    )
    assert bool(keep[0])
    torch.testing.assert_close(F, torch.tensor([2.0 * mean]), rtol=2e-4, atol=0)
    torch.testing.assert_close(sigma_F, torch.tensor([2.0 * sd]), rtol=2e-3, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("centric", [False, True])
def test_posterior_is_continuous_where_the_tables_end(centric):
    sigma_I = torch.ones(2, dtype=torch.float64)
    I = torch.tensor([-4.0 + 1e-6, -4.0 - 1e-6], dtype=torch.float64)
    Sigma = torch.full_like(I, 1e300)
    F, sigma_F, _ = french_wilson(I, sigma_I, Sigma, torch.tensor([centric] * 2))
    assert abs(float(F[1] - F[0])) < 0.03 * float(sigma_F[0])
    assert abs(float(sigma_F[1] - sigma_F[0])) < 0.03 * float(sigma_F[0])


@pytest.mark.unit
def test_noise_far_below_its_prior_is_shrunk_to_the_prior():
    """A reflection the noise swamps keeps its row and takes the prior's value."""
    g = torch.Generator().manual_seed(13)
    sigma_I = torch.full((4000,), 10.0)
    I = sigma_I * torch.randn(4000, generator=g)
    Sigma = torch.full_like(I, 0.1)

    F, sigma_F, keep = french_wilson(I, sigma_I, Sigma)

    # Only what the I/sigma cut removes -- a few in 1e4 for correct sigmas.
    assert int((~keep).sum()) == int((I / sigma_I < -3.7).sum())
    # Mean and standard deviation of sqrt(J) for J ~ Exp(Sigma).
    prior_F, prior_sd = 0.886227 * Sigma.sqrt(), 0.463251 * Sigma.sqrt()
    assert float((F[keep] / prior_F[keep] - 1.0).abs().max()) < 0.02
    assert float((sigma_F[keep] / prior_sd[keep] - 1.0).abs().max()) < 0.02


# =============================================================================
# The prior
# =============================================================================


@pytest.mark.unit
def test_shell_mean_goes_negative_where_the_fitted_prior_does_not():
    """A noise-only shell averages to about zero and half the time below it."""
    d, _, I, sigma = _wilson_data(
        6000, 0, lambda J: torch.full_like(J, 10.0), signal_beyond=3.0
    )

    assert bool((_shell_means(I, d, 60) <= 0).any())

    Sigma = fit_mean_intensity(I, sigma, d)
    assert bool(torch.isfinite(Sigma).all())
    assert bool((Sigma > 0).all())


@pytest.mark.unit
@pytest.mark.parametrize("space_group", ["P 1", "P 1 21 1"])
def test_noise_is_never_converted_into_a_strong_amplitude(space_group):
    """No kept amplitude sits far above its measurement, centric or not."""
    n = 6000
    d, _, I, sigma = _wilson_data(
        n, 1, lambda J: torch.full_like(J, 10.0), signal_beyond=3.0
    )
    hkl = _hkl(n, centric_every=3)

    F, _, keep = french_wilson_auto(I, sigma, hkl, d, space_group)

    assert not bool(_inconsistent(F, keep, I, sigma).any())
    # Signal is kept: the noise-free region converts.
    assert bool(keep[d > 4.0].float().mean() > 0.99)
    if space_group == "P 1 21 1":
        centric = hkl[:, 1] == 0
        assert bool((centric & (d < 3.0)).any()), "no centric noise rows tested"


@pytest.mark.unit
def test_prior_follows_the_mean_when_sigma_grows_with_intensity():
    """Counting statistics make strong reflections noisier.

    A fit that weighted each reflection by its own sigma would follow the weak
    ones and come out low; this one weights a resolution's reflections alike.
    """
    d, Sigma_true, I, sigma = _wilson_data(
        20000, 2, lambda J: torch.sqrt(25.0 + 2.0 * J)
    )

    ratio = fit_mean_intensity(I, sigma, d) / Sigma_true

    assert 0.97 < float(ratio.median()) < 1.03
    assert float(ratio.quantile(0.02)) > 0.88
    assert float(ratio.quantile(0.98)) < 1.12


@pytest.mark.unit
def test_one_wild_reflection_does_not_drag_its_neighbours():
    """A reflection with an enormous sigma carries almost no weight."""
    d, Sigma_true, I, sigma = _wilson_data(5000, 3, lambda J: torch.full_like(J, 5.0))
    wild = int(torch.argmin((d - 3.0).abs()))
    I[wild], sigma[wild] = -3.7e5, 6.8e5
    near = (d - 3.0).abs() < 0.15

    # A shell mean is dragged far below zero by it ...
    assert float(_shell_means(I, d, 60)[near].min()) < 0
    # ... the fitted prior is not.
    ratio = (fit_mean_intensity(I, sigma, d) / Sigma_true)[near]
    assert float(ratio.min()) > 0.9
    assert float(ratio.max()) < 1.12


@pytest.mark.unit
def test_prior_matches_the_shell_mean_on_deposited_data(mtz_dir):
    """On well-measured data the fitted prior changes no amplitude materially."""
    data = ReflectionData(verbose=0).load_mtz(str(mtz_dir / "1DAW.mtz"))
    assert data.I is not None, "1DAW should load via the intensity path"
    I, sigma_I, d = data.I, data.I_sigma, data.resolution

    fitted = fit_mean_intensity(I, sigma_I, d)
    binned = _shell_means(I, d, 40)
    assert float(torch.log(fitted / binned).abs().median()) < 0.05

    F_fit, _, keep_fit = french_wilson(I, sigma_I, fitted)
    F_bin, sigma_F_bin, keep_bin = french_wilson(I, sigma_I, binned)
    assert torch.equal(keep_fit, keep_bin)
    shift = ((F_fit - F_bin).abs() / sigma_F_bin)[keep_bin]
    assert float(shift.quantile(0.99)) < 0.1


@pytest.mark.unit
def test_amplitudes_are_the_same_in_float32_and_float64():
    """The prior needs no double precision to give the same amplitudes.

    Compared through the amplitudes rather than ``Sigma`` itself: where the
    data hold no signal ``Sigma`` is barely determined, and it does not matter
    there, because every reflection it touches is rejected.
    """
    d, _, I, sigma = _wilson_data(
        8000, 4, lambda J: torch.full_like(J, 10.0), signal_beyond=2.5
    )
    single = fit_mean_intensity(I, sigma, d)
    double = fit_mean_intensity(I.double(), sigma.double(), d.double())
    assert single.dtype == torch.float32 and double.dtype == torch.float64

    I64, sigma64 = I.double(), sigma.double()
    F_single, _, keep_single = french_wilson(I64, sigma64, single.double())
    F_double, sigma_F, keep_double = french_wilson(I64, sigma64, double)
    both = keep_single & keep_double
    assert int((keep_single ^ keep_double).sum()) <= 0.005 * len(I)
    assert float(((F_single - F_double).abs() / sigma_F)[both].max()) < 0.1


@pytest.mark.unit
def test_prior_survives_degenerate_inputs():
    d = torch.full((50,), 3.0)
    I = 100.0 + torch.arange(50.0)
    sigma = torch.full((50,), 5.0)

    # A single resolution has no shape to fit; the curve is one positive level.
    flat = fit_mean_intensity(I, sigma, d)
    assert bool((flat > 0).all())
    torch.testing.assert_close(flat, torch.full_like(flat, float(flat[0])))

    # Nothing can inform the fit, so nothing gets a prior or an amplitude.
    hkl = _hkl(50)
    unusable = fit_mean_intensity(I, torch.zeros_like(sigma), d)
    assert bool(torch.isnan(unusable).all())
    _, _, keep = french_wilson_auto(I, torch.zeros_like(sigma), hkl, d, "P 1")
    assert not bool(keep.any())


# =============================================================================
# Held-out reflections
# =============================================================================


@pytest.mark.unit
def test_held_out_reflections_do_not_inform_the_prior():
    """Whatever the test set measures, the working set converts the same."""
    n = 6000
    d, _, I, sigma = _wilson_data(n, 6, lambda J: torch.full_like(J, 10.0))
    hkl = _hkl(n)
    free = torch.arange(n) % 20 == 0

    F, sigma_F, keep = french_wilson_auto(
        I, sigma, hkl, d, "P 1", exclude_from_fit=free
    )
    I_moved = torch.where(free, 100.0 * I.abs() + 1000.0, I)
    F_moved, sigma_F_moved, keep_moved = french_wilson_auto(
        I_moved, sigma, hkl, d, "P 1", exclude_from_fit=free
    )

    work = ~free
    torch.testing.assert_close(F_moved[work], F[work])
    torch.testing.assert_close(sigma_F_moved[work], sigma_F[work])
    assert torch.equal(keep_moved[work], keep[work])
    # The held-out rows are still converted, from their own intensities.
    assert bool(keep_moved[free].all())
    assert bool((F_moved[free] > F[free]).all())


# =============================================================================
# Anisotropy
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "space_group, expected",
    [
        ("P 1", 5),
        ("P 1 21 1", 3),
        ("C 1 2 1", 3),
        ("P 21 21 21", 2),
        ("P 43 21 2", 1),
        ("P 63", 1),
        ("P 31 2 1", 1),
        ("P 21 3", 0),
    ],
)
def test_anisotropy_has_the_parameters_the_laue_class_allows(space_group, expected):
    """Invariant quadratic forms less the isotropic one: 5 triclinic, 0 cubic."""
    hkl = torch.randint(-9, 10, (400, 3), generator=torch.Generator().manual_seed(7))
    hkl = hkl[(hkl != 0).any(dim=1)]
    n = len(hkl)
    radial = _bspline(torch.linspace(-1.0, 1.0, n, dtype=torch.float64), 8)
    s2 = torch.rand(n, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    rows = torch.ones(n, dtype=torch.bool)

    design = _anisotropy_design(hkl, space_group, radial, s2, rows)

    assert design.shape == (n, expected)


def _anisotropic_data(seed):
    """Wilson intensities in P 1 21 1 whose fall-off is 3x faster along c*."""
    cell = Cell([60.0, 70.0, 80.0, 90.0, 100.0, 90.0])
    hkl = torch.stack(
        torch.meshgrid(
            torch.arange(-25, 26),
            torch.arange(0, 30),
            torch.arange(-30, 31),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    s = hkl.double() @ cell.inv_fractional_matrix.double()
    d = 1.0 / s.norm(dim=1)
    keep = (d > 2.5) & (d < 20.0)
    hkl, s, d = hkl[keep], s[keep], d[keep]
    # B along c* 60 A^2, 20 A^2 across it.
    B = 20.0 + 40.0 * (s[:, 2] / s.norm(dim=1)) ** 2
    Sigma = 1000.0 * torch.exp(-B * (s * s).sum(dim=1) / 4.0)
    g = torch.Generator().manual_seed(seed)
    J = Sigma * -torch.log(torch.rand(len(d), generator=g, dtype=torch.float64))
    sigma = torch.full_like(J, 5.0)
    I = J + sigma * torch.randn(len(d), generator=g, dtype=torch.float64)
    return hkl, d.float(), Sigma.float(), I.float(), sigma.float()


@pytest.mark.unit
def test_anisotropic_prior_recovers_an_ellipsoidal_fall_off():
    hkl, d, Sigma_true, I, sigma = _anisotropic_data(9)
    centric = SpaceGroup("P 1 21 1").is_centric(hkl)

    aniso = fit_mean_intensity(
        I,
        sigma,
        d,
        anisotropy=partial(_anisotropy_design, hkl, "P 1 21 1"),
        is_centric=centric,
    )
    iso = fit_mean_intensity(I, sigma, d, fit_mask=~centric)

    error_aniso = torch.log(aniso / Sigma_true).abs()
    error_iso = torch.log(iso / Sigma_true).abs()
    assert float(error_aniso.quantile(0.95)) < 0.15
    # The isotropic curve cannot follow the direction dependence at all.
    assert float(error_iso.quantile(0.95)) > 3.0 * float(error_aniso.quantile(0.95))


# =============================================================================
# Multiplicity and absences
# =============================================================================


@pytest.mark.unit
def test_reflections_on_symmetry_axes_get_epsilon_times_the_prior(mtz_dir):
    """Axial reflections of a deposited dataset are epsilon times stronger.

    With epsilon in the fit their measured intensities match their expected
    ones; without it they are a multiple of it.
    """
    data = ReflectionData(verbose=0).load_mtz(str(mtz_dir / "4BX9.mtz"))
    I, sigma_I, d, hkl = data.I, data.I_sigma, data.resolution, data.hkl
    group = data.spacegroup
    multiplicity = group.epsilon(hkl, friedel=False)
    axial = multiplicity > multiplicity.min()
    assert int(axial.sum()) > 50

    def expected(epsilon):
        return fit_mean_intensity(
            I,
            sigma_I,
            d,
            anisotropy=partial(_anisotropy_design, hkl, group),
            is_centric=group.is_centric(hkl),
            epsilon=epsilon,
        )

    with_epsilon = float(I[axial].sum() / expected(multiplicity)[axial].sum())
    without = float(I[axial].sum() / expected(None)[axial].sum())
    assert 0.8 < with_epsilon < 1.3
    assert without > 2.0


@pytest.mark.unit
def test_systematic_absences_do_not_inform_the_prior():
    """Whatever an absent reflection measures, the others convert the same."""
    hkl, d, _, I, sigma = _anisotropic_data(12)
    absent = SpaceGroup("P 1 21 1").is_absent(hkl)
    assert int(absent.sum()) > 5

    F, sigma_F, keep = french_wilson_auto(I, sigma, hkl, d, "P 1 21 1")
    I_moved = torch.where(absent, I + 1.0e6, I)
    F_moved, sigma_F_moved, keep_moved = french_wilson_auto(
        I_moved, sigma, hkl, d, "P 1 21 1"
    )

    present = ~absent
    torch.testing.assert_close(F_moved[present], F[present])
    torch.testing.assert_close(sigma_F_moved[present], sigma_F[present])
    assert torch.equal(keep_moved[present], keep[present])
