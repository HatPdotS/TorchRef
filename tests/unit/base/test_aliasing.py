"""The alias-error relationship behind the FFT route's blur, and the helpers built on it.

Pinned here: a grid of reciprocal period ``sigma`` returns, for a Gaussian of width
``g`` at ``|s|``, a relative alias error ``exp(-g * sigma (sigma - 2 s) / 4)``; blurring
by ``B_add`` and unblurring multiplies it by ``exp(-B_add * kappa)``; and the blur chosen
from the sharpest atom keeps every atom inside the tolerance.
"""

import math

import pytest
import torch

import torchref
from torchref.base.direct_summation import compute_scattering_factors_batch
from torchref.base.electron_density.radius_policy import EIGHT_PI2, _min_eig_sym3
from torchref.base.fourier import aliasing
from torchref.base.scattering.scattering_table import get_scattering_params_by_z
from torchref.model.context import ModelContext
from torchref.model.sf_fft import SfFFT
from torchref.symmetry import Cell, SpaceGroup

pytestmark = pytest.mark.unit

_CPU = torch.device("cpu")


@pytest.fixture
def settings():
    """Restore the aliasing tolerance and sigma cutoff after the test."""
    tol, n_sigma = torchref.aliasing_tolerance.value, torchref.sigma_cutoff_ed.value
    yield
    torchref.aliasing_tolerance.value = tol
    torchref.sigma_cutoff_ed.value = n_sigma


def _engine(cell_params, d_min, dtype=torch.float64):
    cell = Cell(torch.tensor(cell_params, dtype=dtype), dtype=dtype, device=_CPU)
    ctx = ModelContext(
        cell=cell, spacegroup=SpaceGroup("P 1", dtype=dtype, device=_CPU)
    )
    return SfFFT(ctx, max_res=d_min, dtype_float=dtype, device=_CPU)


def _single_atom(sf, z, b, axis=0, dtype=torch.float64):
    """FFT and analytic F of one atom near the cell centre, along one reciprocal axis.

    Returns ``(s, F_fft, F_true, A, B)`` for every reflection ``n e_axis`` with
    ``|s| < 1 / max_res``.
    """
    cell = sf.ctx.cell
    inv_frac = cell.inv_fractional_matrix.to(dtype)
    step = float(inv_frac[axis].norm())
    n_max = int(1.0 / (sf.max_res * step))
    hkl = torch.zeros(n_max, 3, dtype=dtype)
    hkl[:, axis] = torch.arange(1, n_max + 1, dtype=dtype)
    s = (hkl @ inv_frac).norm(dim=1)
    xyz = torch.tensor([[0.5, 0.5, 0.5]], dtype=dtype) @ cell.fractional_matrix.to(
        dtype
    ).T + torch.tensor([[0.13, -0.21, 0.07]], dtype=dtype)
    A, B = get_scattering_params_by_z(torch.tensor([z]), dtype=dtype)
    adp = torch.tensor([b], dtype=dtype)
    F, _ = sf.compute_structure_factors(hkl, xyz, adp, torch.ones(1, dtype=dtype), A, B)
    phase = torch.exp(2j * math.pi * (hkl @ (xyz @ inv_frac.T).T)).squeeze(1)
    f = compute_scattering_factors_batch(s, A, B).squeeze(1)
    return s, F, f * torch.exp(-b * s**2 / 4) * phase, A, B


def test_alias_period_orthogonal_is_coarsest_axis():
    inv_frac = torch.diag(torch.tensor([1 / 40.0, 1 / 50.0, 1 / 30.0]))
    assert aliasing.alias_period(inv_frac, (48, 75, 40)) == pytest.approx(
        min(48 / 40, 75 / 50, 40 / 30)
    )


def test_alias_period_oblique_is_shortest_lattice_vector():
    dt = torch.float64
    cell = Cell(
        torch.tensor([30.0, 31.0, 29.0, 70.0, 75.0, 65.0], dtype=dt),
        dtype=dt,
        device=_CPU,
    )
    shape = (40, 42, 36)
    rows = (
        cell.inv_fractional_matrix * torch.tensor(shape, dtype=torch.float64)[:, None]
    )
    brute = min(
        float((torch.tensor([i, j, k], dtype=torch.float64) @ rows).norm())
        for i in range(-4, 5)
        for j in range(-4, 5)
        for k in range(-4, 5)
        if (i, j, k) != (0, 0, 0)
    )
    assert aliasing.alias_period(cell.inv_fractional_matrix, shape) == pytest.approx(
        brute
    )


def test_alias_kappa_vanishes_at_nyquist():
    assert aliasing.alias_kappa(1.0, 0.25) == pytest.approx(1.0 * 0.5 / 4)
    assert aliasing.alias_kappa(1.0, 0.5) is None
    assert aliasing.alias_kappa(1.0, 0.6) is None


def test_blur_b_closed_form():
    kappa, rtol = 0.1, 1e-2
    needed = math.log(1 / rtol) / kappa
    for b_min in (0.0, 10.0, needed - 0.3, needed + 5.0):
        got = float(aliasing.blur_b(torch.tensor(b_min), kappa, rtol, b_max=1e4))
        assert got == pytest.approx(max(0.0, math.ceil(needed - b_min)))
    assert float(aliasing.blur_b(torch.tensor(0.0), kappa, rtol, b_max=12.0)) == 12.0


def test_sharpest_b_uses_smallest_principal_axis():
    u = torch.tensor([[0.30, 0.10, 0.20, 0.05, -0.02, 0.01]])
    from torchref.base.targets.adp import U_to_matrix

    lam_min = torch.linalg.eigvalsh(U_to_matrix(u)).min()
    got = aliasing.sharpest_b(torch.tensor([50.0, 30.0]), u)
    assert float(got) == pytest.approx(min(30.0, float(EIGHT_PI2 * lam_min)), rel=1e-5)
    assert float(_min_eig_sym3(U_to_matrix(u))[0]) == pytest.approx(
        float(lam_min), abs=1e-6
    )
    assert aliasing.sharpest_b(torch.zeros(0), None) is None
    assert float(aliasing.sharpest_b(torch.tensor([-3.0, 4.0]), None)) == 0.0


@pytest.mark.parametrize("d_min", [1.0, 2.0, 3.0, 5.0, 8.0, 15.0])
def test_grid_oversampling_keeps_blur_inside_splat(d_min):
    rtol, n_sigma, base = 1e-2, 3.0, 3.0
    r = aliasing.grid_oversampling(d_min, rtol, n_sigma, base)
    assert r >= base
    kappa = aliasing.alias_kappa(r / d_min, 1 / d_min)
    b_cap = aliasing.widest_splat_b(n_sigma) - aliasing._B_FORM_BROADEST
    assert math.log(1 / rtol) / kappa <= b_cap * (1 + 1e-9)
    if d_min <= 5.0:
        assert r == base


@pytest.mark.parametrize("beta", [90.0, 120.0, 150.0])
def test_auto_grid_reaches_oversampling_for_oblique_cells(settings, beta):
    """The sized grid's shortest alias vector is at least ``oversampling / max_res``.

    Lattice-plane spacings alone do not bound it: at beta = 150 they would give
    ``(30, 60, 30)`` at 2 A, folding ``(15, 0, -15)`` onto ``(-15, 0, 15)`` although
    both lie inside ``max_res``.
    """
    sf = _engine([40.0, 40.0, 40.0, 90.0, beta, 90.0], 2.0)
    period = aliasing.alias_period(sf.ctx.cell.inv_fractional_matrix, sf.grid_shape)
    assert period >= torchref.config.NYQUIST_OVERSAMPLING / 2.0 * (1 - 1e-6)
    assert sf.alias_kappa is not None


@pytest.mark.parametrize("b", [2.0, 5.0, 20.0])
def test_unblurred_fft_is_the_alias_sum(settings, b):
    """Without blur, the FFT returns ``sum_m F(h + m N)`` -- the mechanism itself.

    The sigma cutoff is raised so truncation is negligible; what remains of the FFT
    error is then every image of the reflection, summed with its phase.
    """
    torchref.sigma_cutoff_ed.value = 5.0
    torchref.aliasing_tolerance.value = 1.0 - 1e-9
    sf = _engine([40.0, 44.0, 36.0, 90.0, 90.0, 90.0], 3.0)
    dt = torch.float64
    cell = sf.ctx.cell
    inv_frac = cell.inv_fractional_matrix.to(dt)
    hkl = torch.tensor([[11, 0, 0], [12, 3, 0], [4, 13, -2], [-6, 5, 10]], dtype=dt)
    xyz = torch.tensor([[20.13, 21.79, 18.07]], dtype=dt)
    A, B = get_scattering_params_by_z(torch.tensor([6]), dtype=dt)
    F, _ = sf.compute_structure_factors(
        hkl, xyz, torch.tensor([b], dtype=dt), torch.ones(1, dtype=dt), A, B
    )
    N = torch.tensor(sf.grid_shape, dtype=dt)
    images = torch.tensor(
        [[i, j, k] for i in range(-4, 5) for j in range(-4, 5) for k in range(-4, 5)],
        dtype=dt,
    )
    h_all = (hkl[:, None, :] + images[None] * N).reshape(-1, 3)
    s = (h_all @ inv_frac).norm(dim=1)
    f = compute_scattering_factors_batch(s, A, B).squeeze(1) * torch.exp(-b * s**2 / 4)
    phase = torch.exp(2j * math.pi * (h_all @ (xyz @ inv_frac.T).T)).squeeze(1)
    expected = (f * phase).reshape(hkl.shape[0], -1).sum(dim=1)
    torch.testing.assert_close(F, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("b", [20.0, 40.0])
def test_unblurred_alias_error_follows_closed_form(settings, b):
    """Where the error is small, it is the nearest image: ``exp(-g kappa(s))``.

    The prediction weights each ITC92 component by its amplitude at ``s`` and decays it
    by ``exp(-(b_k + B) kappa(s))``. Images along the other axes fall off as
    ``exp(-g ((sigma^2 + s^2) - (sigma - s)^2) / 4)`` relative to it, which is what
    restricts the closed form to ``g kappa >~ 2`` -- the regime a tolerance lives in.
    """
    torchref.sigma_cutoff_ed.value = 5.0
    torchref.aliasing_tolerance.value = 1.0 - 1e-9
    sf = _engine([40.0, 44.0, 36.0, 90.0, 90.0, 90.0], 3.0)
    s, F, F_true, A, B = _single_atom(sf, 6, b)
    sigma = sf.grid_shape[0] / 40.0
    g = B[0] + b
    weights = A[0] * torch.exp(-g * s[:, None] ** 2 / 4)
    kappa = sigma * (sigma - 2 * s[:, None]) / 4
    predicted = (weights * torch.exp(-g * kappa)).sum(dim=1) / weights.sum(dim=1)
    measured = (F - F_true).abs() / F_true.abs()
    outer = s > 0.6 / sf.max_res
    ratio = measured[outer] / predicted[outer]
    assert torch.all((ratio > 0.7) & (ratio < 1.3)), ratio


@pytest.mark.parametrize(
    "cell_params",
    [
        [40.0, 44.0, 36.0, 90.0, 90.0, 90.0],
        [40.0, 44.0, 36.0, 90.0, 112.0, 90.0],
    ],
)
@pytest.mark.parametrize("d_min", [1.5, 2.0, 3.0, 4.0])
@pytest.mark.parametrize("b", [0.5, 5.0, 15.0])
def test_blur_bounds_sharp_atoms_at_tolerance(settings, cell_params, d_min, b):
    """With the blur, a sharp atom stays inside the tolerance out to ``d_min``.

    Truncation is made negligible as above, so the bound tested is the alias one.
    """
    torchref.sigma_cutoff_ed.value = 5.0
    rtol = torchref.aliasing_tolerance.value
    sf = _engine(cell_params, d_min)
    for axis in range(3):
        _, F, F_true, _, _ = _single_atom(sf, 8, b, axis=axis)
        assert ((F - F_true).abs() / F_true.abs()).max() <= rtol


@pytest.mark.parametrize("d_min", [1.5, 3.0, 6.0, 10.0])
def test_blur_with_default_cutoff_stays_at_tolerance(settings, d_min):
    """At the default 3-sigma cutoff the remaining error is truncation plus alias.

    Measured after fitting an overall ``k exp(-dB s^2 / 4)``, which the scaler always
    refines. A strong blur widens every ITC92 component to about the broadest one, so
    each loses the 3-sigma tail (~3% of its mass in 3D); a truncated Gaussian also reads
    as slightly sharper. Both are common to every atom once the blur dominates their
    widths, so they are an overall scale and B rather than an error in the shape of F.
    Measured: ``k - 1 < 3e-3`` and ``|dB| < 0.2`` A^2 down to 3 A, rising to
    ``k - 1 ~ 1.5e-2`` and ``dB`` of 3-7 A^2 at 6-10 A; the fitted residual stays below
    5e-3 throughout.
    """
    rtol = torchref.aliasing_tolerance.value
    sf = _engine([40.0, 44.0, 36.0, 90.0, 90.0, 90.0], d_min)
    for z in (6, 8, 16):
        s, F, F_true, _, _ = _single_atom(sf, z, 2.0)
        design = torch.stack([torch.ones_like(s), -(s**2) / 4], dim=1)
        log_k, d_b = torch.linalg.lstsq(
            design, torch.log(F_true.abs() / F.abs())[:, None]
        ).solution[:, 0]
        assert abs(float(log_k)) < 0.03 and abs(float(d_b)) < 10.0
        fitted = F * torch.exp(log_k - d_b * s**2 / 4)
        assert ((fitted - F_true).abs() / F_true.abs()).max() <= rtol


def test_blunt_atoms_are_not_blurred(settings):
    sf = _engine([40.0, 44.0, 36.0, 90.0, 90.0, 90.0], 2.0)
    assert float(sf.alias_blur(torch.tensor([80.0, 120.0], dtype=torch.float64))) == 0.0
    blurred = _single_atom(sf, 6, 80.0)[1]
    torchref.aliasing_tolerance.value = 1.0 - 1e-9
    plain = _single_atom(sf, 6, 80.0)[1]
    assert torch.equal(blurred, plain)


def test_iso_and_isotropic_aniso_atoms_blur_alike(settings):
    sf = _engine([40.0, 44.0, 36.0, 90.0, 112.0, 90.0], 3.0)
    dt = torch.float64
    g = torch.Generator().manual_seed(0)
    n = 12
    xyz = 10.0 + 20.0 * torch.rand(n, 3, generator=g, dtype=dt)
    b = 3.0 + 10.0 * torch.rand(n, generator=g, dtype=dt)
    occ = torch.ones(n, dtype=dt)
    A, B = get_scattering_params_by_z(torch.full((n,), 7), dtype=dt)
    u = torch.zeros(n, 6, dtype=dt)
    u[:, :3] = (b / EIGHT_PI2)[:, None]
    hkl = torch.tensor([[1, 2, 3], [5, -4, 2], [12, 0, -3], [0, 13, 1]], dtype=dt)
    F_iso, _ = sf.compute_structure_factors(hkl, xyz, b, occ, A, B)
    empty = torch.zeros(0, dtype=dt)
    F_aniso, _ = sf.compute_structure_factors(
        hkl,
        empty.reshape(0, 3),
        empty,
        empty,
        empty.reshape(0, 5),
        empty.reshape(0, 5),
        xyz,
        u,
        occ,
        A,
        B,
    )
    assert float(sf.alias_blur(b)) > 0
    torch.testing.assert_close(F_aniso, F_iso, rtol=1e-6, atol=1e-6)


def test_tolerance_and_oversampling_are_read_at_call_time(settings):
    sf = _engine([40.0, 44.0, 36.0, 90.0, 90.0, 90.0], 2.0)
    key = sf.grid_key
    torchref.aliasing_tolerance.value = 1e-3
    assert sf.grid_key != key
    b = torch.tensor([5.0], dtype=torch.float64)
    tight = float(sf.alias_blur(b))
    assert tight > 0
    torchref.config.NYQUIST_OVERSAMPLING = 4.0
    try:
        assert sf.grid_shape[0] > 60
        assert float(sf.alias_blur(b)) < tight
    finally:
        torchref.config.NYQUIST_OVERSAMPLING = 3.0


def test_tolerance_rejects_values_outside_unit_interval(settings):
    for bad in (0.0, 1.0, -1e-3, 2.0):
        with pytest.raises(ValueError):
            torchref.aliasing_tolerance.value = bad
    with pytest.raises(TypeError):
        torchref.aliasing_tolerance.value = True
