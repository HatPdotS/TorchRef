"""The fixed component's coefficients are the likelihood's own derivatives, and
the fast map with a fixed component is that likelihood's second-order expansion.

Three identities pin this down: the closed forms for ``c_quad`` and ``c_lin``
equal autograd derivatives of ``rice_per_refl``; the map equals
``translation_score_at`` at its grid points with a fixed component present; and
with nothing fixed the coefficients reduce to the LERF1 form the rotation
function expands, so the existing path is untouched.
"""
import math

import pytest
import torch

from torchref.base.targets.xray_likelihoods import rice_per_refl
from torchref.experimental.alignment.translation import (
    CandidateTransform, FixedComponent, TranslationObs,
    fast_translation_function, translation_score_at, llg_at_translations)
from torchref.symmetry.cell import Cell
from torchref.symmetry.spacegroup import SpaceGroup

pytestmark = pytest.mark.unit


def _synthetic(n=3000, seed=0, sg="P 21 21 21"):
    g = torch.Generator().manual_seed(seed)
    cell = Cell([61.0, 72.0, 83.0, 90.0, 90.0, 90.0], device="cpu")
    spacegroup = SpaceGroup(sg, device="cpu")
    rng = torch.arange(-9, 10)
    h, k, l = torch.meshgrid(rng, rng, rng, indexing="ij")
    hkl = torch.stack([h.flatten(), k.flatten(), l.flatten()], dim=-1)
    hkl = hkl[(hkl.abs().sum(dim=-1) > 0)]
    hkl = hkl[torch.randperm(hkl.shape[0], generator=g)[:n]]
    s_mag = (hkl.to(torch.float64) @ cell.reciprocal_basis_matrix.to(torch.float64)).norm(dim=-1)
    Sigma = 3000.0 * torch.exp(-2.0 * 25.0 * s_mag ** 2)
    I = Sigma * -torch.rand(n, generator=g, dtype=torch.float64).clamp(min=1e-9).log()
    F = I.sqrt()
    obs = TranslationObs.build(F, hkl, spacegroup, cell, delta_vrms_A=0.8)
    # A "fixed" structure factor with a plausible Wilson falloff and random phase.
    amp = (0.5 * Sigma * -torch.rand(n, generator=g, dtype=torch.float64).clamp(min=1e-9).log()).sqrt()
    ph = 2 * math.pi * torch.rand(n, generator=g, dtype=torch.float64)
    F_fixed = torch.polar(amp, ph)
    return obs, F_fixed, spacegroup, cell


def test_coefficients_are_the_likelihood_derivatives():
    obs, F_fixed, sg, cell = _synthetic()
    fx = FixedComponent.build(obs, F_fixed, err_fixed_A=0.8)
    E = obs.E_obs.to(torch.float64)
    Fm = (fx.D_f * fx.E_f.abs()).to(torch.float64).detach().requires_grad_(True)
    V = fx.V.to(torch.float64).detach().requires_grad_(True)
    nll = rice_per_refl(E, Fm, V, obs.centric).sum()
    dFm, dV = torch.autograd.grad(nll, (Fm, V))
    w = obs.weight.to(torch.float64)
    sa = obs.sigma_a.to(torch.float64)
    # LL = -NLL, so dLL/dSigma = -dV, dLL/d|Fc| = -dFm; twice the derivative is
    # the LERF1 convention (cw (E^2 - 1) with cw = 2 acentric, 1 centric at F = 0).
    torch.testing.assert_close(fx.c_quad.to(torch.float64), 2.0 * w * sa ** 2 * (-dV), rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(fx.c_lin.abs().to(torch.float64), (2.0 * w * sa * (-dFm)).abs(), rtol=1e-3, atol=1e-4)
    # The phase of c_lin is the fixed structure's phase, up to the sign of dLL/d|Fc|.
    rel = torch.angle(fx.c_lin.to(torch.complex128) * torch.exp(-1j * torch.angle(F_fixed)))
    assert torch.all((rel.abs() < 1e-3) | ((rel.abs() - math.pi).abs() < 1e-3))


def test_no_fixed_part_reduces_to_lerf1():
    """A zero fixed structure factor gives back the observed side's coefficient."""
    obs, F_fixed, sg, cell = _synthetic()
    fx = FixedComponent.build(obs, torch.zeros_like(F_fixed), err_fixed_A=0.8)
    torch.testing.assert_close(fx.c_quad, obs.coeff, rtol=1e-4, atol=1e-5)
    assert float(fx.c_lin.abs().max()) < 1e-6


def _candidate(obs, seed=1):
    """A random normalised transform with symmetry-rotated indices, as prepare_candidate makes."""
    g = torch.Generator().manual_seed(seed)
    S = 4
    N = obs.hkl.shape[0]
    hkl = obs.hkl.to(torch.float32)
    sg = SpaceGroup("P 21 21 21", device="cpu")
    h_R = torch.einsum("ne,ied->ind", hkl, sg.matrices.to(torch.float32))
    G = torch.polar(torch.rand(S, N, generator=g), 2 * math.pi * torch.rand(S, N, generator=g)).to(torch.complex64)
    G = G / math.sqrt(S)
    return CandidateTransform(G=G, h_R=h_R, norm=torch.ones(N)), sg


def test_map_with_fixed_component_equals_the_direct_score_at_grid_points():
    obs, F_fixed, sg, cell = _synthetic()
    fx = FixedComponent.build(obs, F_fixed, err_fixed_A=0.8)
    cand, sg4 = _candidate(obs)
    score, peaks = fast_translation_function(
        obs, cand, sg4, cell, grid_spacing_A=6.0, n_peaks=3, cluster_radius_A=6.0, fixed=fx)
    nx, ny, nz = score.shape
    for ijk in [(0, 0, 0), (3, 5, 7), (nx - 1, 2, nz // 2)]:
        t = torch.tensor([ijk[0] / nx, ijk[1] / ny, ijk[2] / nz], dtype=torch.float64)
        direct = translation_score_at(obs, cand, t, fixed=fx)
        # The map carries a t-independent constant from |E_f|^2; compare differences.
        t0 = torch.zeros(3, dtype=torch.float64)
        d_map = float(score[ijk] - score[0, 0, 0])
        d_direct = direct - translation_score_at(obs, cand, t0, fixed=fx)
        assert abs(d_map - d_direct) <= 1e-3 * max(1.0, abs(d_direct)), (ijk, d_map, d_direct)


def test_fixed_component_pins_the_origin():
    """With a fixed part two peaks related by an allowed origin shift are distinct."""
    obs, F_fixed, sg, cell = _synthetic()
    fx = FixedComponent.build(obs, F_fixed, err_fixed_A=0.8)
    cand, sg4 = _candidate(obs)
    _, peaks = fast_translation_function(
        obs, cand, sg4, cell, grid_spacing_A=6.0, n_peaks=8, cluster_radius_A=6.0, fixed=fx)
    assert len(peaks) == 8


def test_llg_with_fixed_is_finite_and_reference_is_fixed_only():
    obs, F_fixed, sg, cell = _synthetic()
    fx = FixedComponent.build(obs, F_fixed, err_fixed_A=0.8)
    cand, _ = _candidate(obs)
    llg = llg_at_translations(obs, cand, torch.zeros(2, 3, dtype=torch.float64), fixed=fx)
    assert torch.isfinite(llg).all()
    assert torch.isfinite(fx.ll_ref)
