"""The observed side of the translation search, and what it guarantees.

``TranslationObs`` exists so that "what is E_obs here" has exactly one answer
across the whole run. Three separate answers used to live in this path -- one in
the coarse search, one in the likelihood, one hand-rolled in the pipeline -- and
they disagreed, which meant the stage that chose a peak and the stage that
re-ranked it were not scoring the same quantity.

These tests pin the properties that claim rests on: the normalisation is the
shared Wilson fit, the weight is a real per-reflection weight rather than a
per-shell one (per-shell weights cancel in a correlation), and the whole object
is invariant to the units the amplitudes arrive in.
"""
import math

import pytest
import torch

from torchref.experimental.alignment.translation import TranslationObs
from torchref.scaling import WilsonNormaliser
from torchref.symmetry.cell import Cell
from torchref.symmetry.spacegroup import SpaceGroup


def _case(n=4000, seed=0, sg="P 21 21 21"):
    """A synthetic reflection set with a realistic Wilson falloff."""
    g = torch.Generator().manual_seed(seed)
    # Pinned to CPU: this host has an accelerator, and Cell/SpaceGroup would
    # land there while the synthetic hkl below stays on the host.
    cell = Cell([61.0, 72.0, 83.0, 90.0, 90.0, 90.0], device="cpu")
    spacegroup = SpaceGroup(sg, device="cpu")
    # Miller indices on a coarse block, origin removed.
    rng = torch.arange(-9, 10)
    h, k, l = torch.meshgrid(rng, rng, rng, indexing="ij")
    hkl = torch.stack([h.flatten(), k.flatten(), l.flatten()], dim=-1)
    hkl = hkl[(hkl.abs().sum(dim=-1) > 0)]
    hkl = hkl[torch.randperm(hkl.shape[0], generator=g)[:n]]

    s_mag = (hkl.to(torch.float64)
             @ cell.reciprocal_basis_matrix.to(torch.float64)).norm(dim=-1)
    # Exponential intensities on a Wilson curve, so <E^2> = 1 is reachable.
    Sigma = 3000.0 * torch.exp(-2.0 * 25.0 * s_mag ** 2)
    I = Sigma * -torch.rand(n, generator=g, dtype=torch.float64).clamp(min=1e-9).log()
    F = I.sqrt()
    sig_F = 0.05 * F + 0.01 * F.mean()
    return F, sig_F, hkl, spacegroup, cell, s_mag


@pytest.mark.unit
def test_normalisation_is_the_shared_wilson_fit():
    """E_obs is WilsonNormaliser's E, not a private per-shell mean."""
    F, _, hkl, sg, cell, _ = _case()
    obs = TranslationObs.build(F, hkl, sg, cell)

    direct = WilsonNormaliser(
        obs.F_obs * obs.F_obs, obs.s_mag, eps=obs.eps, centric=obs.centric,
        n_coeff=6,
    )
    torch.testing.assert_close(obs.E_obs, direct.E.to(torch.float64))


@pytest.mark.unit
def test_mean_e_squared_is_one():
    """<E^2> = 1 is an identity of the Gamma fit, so it holds to fit precision.

    k-weighted, because that is the score equation the intercept solves:
    sum_h k_h (I_h/mu_h - 1) = 0 with k = 1 acentric, 1/2 centric.
    """
    F, _, hkl, sg, cell, _ = _case()
    obs = TranslationObs.build(F, hkl, sg, cell)
    k = torch.where(obs.centric, 0.5, 1.0).to(torch.float64)
    mean_e2 = (k * obs.E_obs ** 2).sum() / k.sum()
    assert abs(float(mean_e2) - 1.0) < 1e-6, mean_e2


@pytest.mark.unit
def test_e_obs_is_invariant_to_the_amplitude_scale():
    """Rescaling every amplitude must not change E. It is an ABSOLUTE normaliser.

    Exact in the model -- a common factor lands entirely in Sigma's intercept --
    but the fit is IRLS, so the tolerance is its convergence floor rather than
    machine epsilon. Measured spread is ~2e-8 over 4000 reflections; 1e-6 catches
    a genuine scale dependence without chasing the solver.
    """
    F, sig_F, hkl, sg, cell, _ = _case()
    base = TranslationObs.build(F, hkl, sg, cell, sig_F=sig_F)
    scaled = TranslationObs.build(7.5 * F, hkl, sg, cell, sig_F=7.5 * sig_F)
    torch.testing.assert_close(base.E_obs, scaled.E_obs, rtol=1e-6, atol=1e-6)
    # F/sigma is unchanged by a common factor, so the weight must be too.
    torch.testing.assert_close(base.weight, scaled.weight, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_weight_varies_within_a_shell():
    """The part of the weight that is not gauge is the part that varies within a shell.

    A weight constant inside a resolution shell is a per-shell weight, and a
    correlation absorbs those -- which is the whole reason twelve E conventions
    moved the rotation function's truth rank by nothing. So this asserts the
    thing that makes weighting worth doing at all, not merely that a weight
    exists.
    """
    F, sig_F, hkl, sg, cell, _ = _case()
    # Give two reflections at the SAME resolution very different sigmas.
    obs = TranslationObs.build(F, hkl, sg, cell, sig_F=sig_F)

    within = []
    for shell in range(obs.n_shells):
        w = obs.weight[obs.shell_idx == shell]
        if w.numel() > 20:
            within.append(float(w.std() / w.mean().clamp(min=1e-30)))
    assert within, "no populated shells"
    assert min(within) > 1e-3, (
        f"weight is effectively constant within shells (max rel. spread "
        f"{max(within):.2e}); it would be absorbed by the correlation"
    )


@pytest.mark.unit
def test_weight_is_uniform_without_sigmas():
    """No sigmas, no weight. The varying half of the weight IS the measurement term."""
    F, _, hkl, sg, cell, _ = _case()
    obs = TranslationObs.build(F, hkl, sg, cell, sig_F=None)
    assert torch.allclose(obs.weight, torch.ones_like(obs.weight))


@pytest.mark.unit
def test_weight_is_normalised_to_mean_one():
    """So the score's scale does not depend on how the sigmas happened to be scaled."""
    F, sig_F, hkl, sg, cell, _ = _case()
    obs = TranslationObs.build(F, hkl, sg, cell, sig_F=sig_F)
    assert abs(float(obs.weight.mean()) - 1.0) < 1e-9


@pytest.mark.unit
@pytest.mark.parametrize("sg_name", ["P 1", "P 21 21 21", "C 1 2 1", "P 43 21 2"])
def test_epsilon_and_centricity_come_from_the_spacegroup(sg_name):
    """Both reach the fit, and epsilon is the friedel=False count.

    Wilson's <I> = eps*Sigma counts the operations mapping h to itself, which add
    coherently and set the mean. The Friedel-folded count answers a different
    question -- it describes the distribution's shape, which enters as the Gamma
    shape via centricity, separately. Applying the wrong one here shifts the
    normalisation of every axial reflection.
    """
    F, _, hkl, sg, cell, _ = _case(sg=sg_name)
    obs = TranslationObs.build(F, hkl, sg, cell)
    hkl_l = obs.hkl.round().to(torch.int64)
    torch.testing.assert_close(
        obs.eps, sg.epsilon(hkl_l, friedel=False).to(torch.float64).clamp(min=1.0),
    )
    torch.testing.assert_close(obs.centric, sg.is_centric(hkl_l).to(torch.bool))


@pytest.mark.unit
def test_shell_binning_is_equal_count_and_shared():
    """One binning, used by both the sigma_A fit and the likelihood.

    They used to derive their own from the same |s| -- one rank-based, one
    value-based -- which put boundary reflections in different shells depending
    on which stage asked.
    """
    F, _, hkl, sg, cell, _ = _case()
    obs = TranslationObs.build(F, hkl, sg, cell, n_shells=10)
    counts = torch.bincount(obs.shell_idx, minlength=obs.n_shells)
    assert obs.n_shells == 10
    assert int(counts.min()) > 0
    # Equal-count binning: no shell should be wildly larger than the mean.
    assert float(counts.max()) < 3.0 * float(counts.to(torch.float64).mean())
    # And it must be monotone in |s|: shells partition resolution, not noise.
    hi = torch.stack([obs.s_mag[obs.shell_idx == b].max()
                      for b in range(obs.n_shells)])
    assert bool((hi[1:] >= hi[:-1]).all())
