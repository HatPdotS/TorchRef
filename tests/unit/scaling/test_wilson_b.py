"""fit_wilson_b: the Wilson B of the form-factor and protein-corrected Wilson plot.

Pinned behaviour: B is recovered from intensities drawn from the model itself,
at full and at truncated resolution; on deposited data it is stable when the
high-resolution shells are cut away; it gives up with a warning rather than
returning a stand-in value; and the vendored protein correction evaluates
cctbx's Chebyshev series exactly as scitbx does.
"""

import io
import contextlib
import math

import pytest
import torch

from torchref.io import ReflectionData
from torchref.scaling._protein_gamma import (
    _COEFFS,
    D_STAR_SQ_HIGH,
    D_STAR_SQ_LOW,
    protein_gamma,
)
from torchref.scaling.wilson import fit_wilson_b, sum_f_squared


def _clenshaw(x, coeffs, lo=D_STAR_SQ_LOW, hi=D_STAR_SQ_HIGH):
    """scitbx ``chebyshev_base::cheb_base_f``, transcribed line by line."""
    t = (x - (lo + hi) * 0.5) / (0.5 * (hi - lo))
    x2, d, dd = 2.0 * t, 0.0, 0.0
    for c in reversed(coeffs[1:]):
        d, dd = x2 * d - dd + c, d
    return t * d - dd + 0.5 * coeffs[0]


@pytest.mark.unit
def test_protein_gamma_matches_the_scitbx_series():
    x = torch.linspace(D_STAR_SQ_LOW, D_STAR_SQ_HIGH, 50, dtype=torch.float64)
    expected = torch.tensor([_clenshaw(float(v), _COEFFS) for v in x], dtype=x.dtype)
    torch.testing.assert_close(protein_gamma(x), expected, rtol=0, atol=1e-10)
    # Held at the end values outside the fitted range.
    ends = protein_gamma(torch.tensor([0.001, D_STAR_SQ_LOW, 0.9, D_STAR_SQ_HIGH]))
    assert ends[0] == ends[1] and ends[2] == ends[3]


def _synthetic_intensities(B, d_min, seed=0, cell=40.0):
    """Acentric Wilson intensities in a cubic P1 cell, drawn from the fitted model."""
    n = int(cell / d_min) + 1
    r = torch.arange(-n, n + 1)
    hkl = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    hkl = hkl[(hkl[:, 2] > 0) | ((hkl[:, 2] == 0) & (hkl[:, 1] > 0))]
    d = cell / hkl.to(torch.float64).norm(dim=-1)
    d = d[(d >= d_min) & (d < 20.0)]
    d_star_sq = d.pow(-2)
    mean = sum_f_squared(d_star_sq) * (1 + protein_gamma(d_star_sq))
    mean = mean * torch.exp(-0.5 * B * d_star_sq)
    g = torch.Generator().manual_seed(seed)
    return mean * torch.empty_like(mean).exponential_(generator=g), d


@pytest.mark.unit
@pytest.mark.parametrize("B", [20.0, 50.0, 80.0])
@pytest.mark.parametrize("d_min, tol", [(1.8, 1.5), (3.5, 5.0)])
def test_recovers_the_b_of_its_own_model(B, d_min, tol):
    I, d = _synthetic_intensities(B, d_min)
    fit = fit_wilson_b(I, d)
    assert fit is not None
    assert fit.B == pytest.approx(B, abs=tol)
    assert fit.sigma_B < tol


@pytest.mark.unit
def test_expected_intensity_follows_the_data():
    I, d = _synthetic_intensities(40.0, 2.0)
    fit = fit_wilson_b(I, d)
    order = torch.argsort(d)
    for shell in torch.tensor_split(order, 10):
        predicted = fit.expected_intensity(d[shell]).mean()
        assert float(I[shell].mean() / predicted) == pytest.approx(1.0, abs=0.1)


@pytest.mark.unit
def test_amplitudes_use_f_squared_plus_variance():
    I, d = _synthetic_intensities(30.0, 2.0)
    F = I.sqrt()
    sigma = torch.full_like(F, 1.0)
    plain = fit_wilson_b(F, d, amplitudes=True)
    with_var = fit_wilson_b(F, d, sigma=sigma, amplitudes=True)
    assert plain.B == pytest.approx(30.0, abs=1.5)
    # Adding a constant variance flattens the falloff, so B comes out lower.
    assert with_var.B < plain.B


@pytest.mark.unit
def test_gives_up_with_a_warning_instead_of_guessing():
    I, d = _synthetic_intensities(30.0, 2.0)
    with pytest.warns(UserWarning, match="No Wilson B"):
        assert fit_wilson_b(I[:30], d[:30]) is None
    thin = (d > 3.0) & (d < 3.05)
    with pytest.warns(UserWarning, match="too short"):
        assert fit_wilson_b(I[thin], d[thin]) is None


#: B measured with this model on the deposited test data. The deposited
#: ``B_iso_Wilson_estimate`` values follow the plain ctruncate-style fit instead
#: and sit 2-16 Å² higher; these pin the γ-corrected estimate.
MEASURED = {
    "1DAW": 28.6,
    "3E98": 49.6,
    "3K7M": 29.3,
    "4BX9": 60.5,
    "5BOV": 14.0,
    "6G9X": 66.6,
}


def _fit_deposited(name, mtz_dir, d_cut=None):
    with contextlib.redirect_stdout(io.StringIO()):
        data = ReflectionData(verbose=0).load_mtz(str(mtz_dir / f"{name}.mtz"))
    keep = data.masks()
    if d_cut is not None:
        keep = keep & (data.resolution >= d_cut)
    eps = data.spacegroup.epsilon(data.hkl).to(data.F)[keep]
    if data.I is not None:
        return fit_wilson_b(data.I[keep], data.resolution[keep], epsilon=eps)
    return fit_wilson_b(
        data.F[keep],
        data.resolution[keep],
        sigma=data.F_sigma[keep],
        epsilon=eps,
        amplitudes=True,
    )


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(MEASURED))
def test_deposited_data(name, mtz_dir):
    full = _fit_deposited(name, mtz_dir)
    assert full.B == pytest.approx(MEASURED[name], abs=2.0)
    # Cutting the data at 3.5 Å moves B by at most 9 Å² on these six (6G9X,
    # the largest), where the uncorrected fit fell back to a fixed 200 Å².
    cut = _fit_deposited(name, mtz_dir, d_cut=3.5)
    assert cut is not None
    assert cut.B == pytest.approx(full.B, abs=10.0)
    assert math.isfinite(cut.sigma_B)
