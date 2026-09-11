"""The overall-anisotropy fit, on data whose anisotropy is known by construction.

Synthetic Wilson intensities: acentric reflections are exponentially
distributed about their shell mean, centric ones follow a chi-squared with one
degree of freedom, and the anisotropy enters as ``exp(-2 pi^2 s.U.s)`` on the
mean. Feeding that in and asking for U back is the only way to separate a
correct fit from one that merely returns something plausible.

The first test is the one that matters: **isotropic data must give back zero
anisotropy**. Fitting the same relation in log space instead is biased --
``E[ln(I/<I>)]`` is ``-gamma``, not zero -- and with no constant term that
offset can only be absorbed by the quadratic form. It comes back as tens of
square Angstrom of anisotropy that is not in the data.
"""

import math

import pytest
import torch

from torchref.experimental.alignment.sh import (
    assign_shells,
    equal_count_shell_edges,
    fit_overall_anisotropy,
)

pytestmark = pytest.mark.unit

B_PER_U = 8.0 * math.pi ** 2


def _synthetic(U_true, n=40000, seed=0, centric_fraction=0.0):
    """Wilson intensities carrying exactly ``U_true``, on a 4-15 A shell."""
    g = torch.Generator().manual_seed(seed)
    smag = 1.0 / (4.0 + 11.0 * torch.rand(n, generator=g, dtype=torch.float64))
    ct = 2 * torch.rand(n, generator=g, dtype=torch.float64) - 1
    phi = 2 * math.pi * torch.rand(n, generator=g, dtype=torch.float64)
    st = (1 - ct * ct).clamp(min=0).sqrt()
    s = torch.stack([smag * st * torch.cos(phi),
                     smag * st * torch.sin(phi),
                     smag * ct], dim=-1)

    # Shell mean falls off with resolution, times the anisotropic term.
    sigma_shell = torch.exp(-20.0 * smag * smag)
    aniso = torch.exp(-2.0 * (math.pi ** 2) * torch.einsum(
        "ni,ij,nj->n", s, U_true.to(torch.float64), s))
    mean_I = sigma_shell * aniso

    centric = torch.rand(n, generator=g, dtype=torch.float64) < centric_fraction
    # Acentric: I = mean * Exp(1). Centric: I = mean * chi^2_1.
    e = -torch.log(torch.rand(n, generator=g, dtype=torch.float64).clamp(min=1e-300))
    z = torch.randn(n, generator=g, dtype=torch.float64) ** 2
    I = mean_I * torch.where(centric, z, e)
    F = I.clamp(min=0).sqrt()

    edges, _ = equal_count_shell_edges(smag, 20)
    return F, s, assign_shells(smag, edges), centric


def _spread_B(U):
    ev = torch.linalg.eigvalsh(U.to(torch.float64)) * B_PER_U
    return float(ev[2] - ev[0])


#: Reflections used by the isotropic-data tests. The estimator's own scatter on
#: the B spread is about 7 A^2 at this count, and falls as 1/sqrt(n).
_N_ISO = 40000

#: Threshold for "no anisotropy detected" at ``_N_ISO``: above the estimator's
#: measured scatter (median 6.9, max 8.2 over six seeds) with margin. This is a
#: noise floor, not a bias tolerance -- see the scaling test below.
_ISO_TOLERANCE_B = 12.0


@pytest.mark.parametrize("centric_fraction", [0.0, 0.15])
def test_isotropic_data_gives_no_anisotropy(centric_fraction):
    """Zero anisotropy in, nothing but estimation noise out."""
    F, s, idx, cen = _synthetic(
        torch.zeros(3, 3), n=_N_ISO, seed=1, centric_fraction=centric_fraction)
    U = fit_overall_anisotropy(F, s, idx, cen, P=20)
    assert _spread_B(U) < _ISO_TOLERANCE_B, (
        f"isotropic data produced {_spread_B(U):.1f} A^2 of B anisotropy, "
        f"beyond this estimator's scatter at n={_N_ISO}"
    )


def test_the_isotropic_residual_is_noise_not_bias():
    """The spurious spread must shrink as 1/sqrt(n), not plateau.

    This is the real test of the fit's centring. A biased estimator -- for
    instance one regressing ``ln(I/<I>)`` with no constant term, where
    ``E[ln(I/<I>)] = -gamma`` has to be absorbed by the quadratic form -- gives
    a spread that stays put as reflections are added. An unbiased one averages
    it away.
    """
    def median_spread(n):
        vals = []
        for k in range(3):
            F, s, idx, cen = _synthetic(torch.zeros(3, 3), n=n, seed=100 + k)
            vals.append(_spread_B(fit_overall_anisotropy(F, s, idx, cen, P=20)))
        return sorted(vals)[1]

    coarse, fine = median_spread(20000), median_spread(320000)
    # 16x the reflections should buy about 4x, i.e. well over 2x even allowing
    # for the scatter of a 3-seed median.
    assert fine < coarse / 2.0, (
        f"B spread went {coarse:.2f} -> {fine:.2f} A^2 for 16x the reflections; "
        f"an unbiased fit should shrink roughly 4x, a biased one not at all"
    )


def test_a_known_tensor_is_recovered():
    """Uniaxial anisotropy of a realistic size comes back to within a few A^2."""
    B_true = torch.diag(torch.tensor([-15.0, -15.0, 30.0], dtype=torch.float64))
    U_true = B_true / B_PER_U
    F, s, idx, cen = _synthetic(U_true, n=60000, seed=2)
    U = fit_overall_anisotropy(F, s, idx, cen, P=20)
    B_fit = U.to(torch.float64) * B_PER_U
    # The isotropic part is a gauge -- the per-shell normalisation removes it --
    # so compare the deviatoric parts.
    dev = lambda M: M - torch.eye(3, dtype=torch.float64) * torch.diagonal(M).mean()
    err = (dev(B_fit) - dev(B_true)).abs().max().item()
    assert err < 6.0, f"recovered B off by {err:.1f} A^2:\n{B_fit}"


def test_zero_amplitudes_do_not_dominate():
    """A handful of vanishing amplitudes must not steer the fit.

    The earlier version clamped them to 1e-30 and took a logarithm, turning each
    into a residual of about -69 in an unweighted least squares.
    """
    F, s, idx, cen = _synthetic(torch.zeros(3, 3), seed=3)
    clean = fit_overall_anisotropy(F, s, idx, cen, P=20)
    F2 = F.clone()
    F2[::500] = 0.0
    spiked = fit_overall_anisotropy(F2, s, idx, cen, P=20)
    assert abs(_spread_B(spiked) - _spread_B(clean)) < 3.0, (
        f"zeroing 0.2% of amplitudes moved the fit from "
        f"{_spread_B(clean):.2f} to {_spread_B(spiked):.2f} A^2"
    )


def test_too_few_reflections_returns_zero():
    F, s, idx, cen = _synthetic(torch.zeros(3, 3), n=60, seed=4)
    U = fit_overall_anisotropy(F, s, idx, cen, P=20, min_count=20)
    assert torch.equal(U, torch.zeros(3, 3, dtype=U.dtype))
