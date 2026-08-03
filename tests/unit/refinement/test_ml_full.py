"""Regression guards for the full-form MLF target (``xray_ml_full``).

These lock in what the one-off screening study in ``sigma_a_rework/quad_screen.py``
established. Several of them exist because the corresponding bug was actually made
and caught during development, which is noted where relevant.
"""

import math

import numpy as np
import pytest
import torch

from torchref.base.targets import xray_ml_full as F
from torchref.base.targets.xray_likelihoods import rice_marginal_math, rice_per_refl

DT = torch.float64


@pytest.fixture(autouse=True)
def _no_compile():
    """Keep this module eager. ``torch.compile`` costs ~10-25 s the first time and
    these tests are about the maths, not the fusion; the compiled path gets its own
    test below, which asserts it agrees with eager."""
    import torchref.config as cfg

    prev = cfg.compile_targets.value
    cfg.compile_targets.value = False
    yield
    cfg.compile_targets.value = prev


def _t(*vals):
    return [torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=DT) for v in vals]


def _grid(r_sig, r_fc, r_fo, Sigma=1.0):
    """Dimensionless grid. NLL is exactly equivariant under
    ``F -> lam F, Sigma -> lam^2 Sigma`` (``NLL -> NLL + log lam``), so fixing
    ``Sigma = 1`` loses no generality -- see ``test_scale_equivariance``."""
    out = []
    for rs in r_sig:
        for fc in r_fc:
            for rf in r_fo:
                out.append((rf * rs, rs, fc, Sigma))
    a = np.asarray(out, dtype=np.float64)
    return _t(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


# ---------------------------------------------------------------------------
# 1. the load-bearing test: quadrature machinery vs an exact closed form
# ---------------------------------------------------------------------------


def test_quadrature_reproduces_exact_centric_closed_form():
    """Run the *acentric* quadrature machinery on the *centric* integrand and
    compare against the exact ``Phi`` expression.

    This is the strongest single check available: it validates the window, the node
    count and the log-sum-exp against an independent analytic result, on the same
    code path, with no external reference to trust. It is what caught the window bug
    (unioning the Laplace window with ``F_obs +- n sigma`` starved the grid of
    resolution: max|dNLL| ~ 1e1..1e2, flat in ``n_quad``).
    """
    F_obs, sig, Fc, Sigma = _grid(
        [1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0],
        [0.0, 0.1, 1.0, 5.0, 20.0, 50.0],
        [0.0, 0.5, 2.0, 10.0, 200.0],
    )
    exact = F.centric_nll(F_obs, sig, Fc, Sigma)

    # centric integrand through the generic quadrature path
    def log_h_centric(t):
        inv_S = 1.0 / Sigma
        return (
            0.5 * torch.log(2.0 / (math.pi * Sigma))
            - (t * t + Fc * Fc) * (0.5 * inv_S)
            + F._log_cosh(t * Fc * inv_S)
            - 0.5 * (F.LOG_2PI + 2.0 * torch.log(sig))
            - (F_obs - t) ** 2 / (2.0 * sig**2)
        )

    # centric asymptote is log cosh(x) ~ |x| with x = t Fc / Sigma, i.e. HALF the
    # acentric slope; using the acentric centre here misplaces the peak.
    a = 0.5 * (1.0 / Sigma + 1.0 / sig**2)
    t0 = (F_obs / sig**2 + Fc / Sigma) / (2.0 * a)
    s = 1.0 / torch.sqrt(2.0 * a)
    lo = torch.clamp(t0 - F.N_SIGMA * s, min=0.0)
    hi = t0 + F.N_SIGMA * s
    half, mid = (hi - lo) * 0.5, (hi + lo) * 0.5
    shift = torch.maximum(
        log_h_centric(torch.clamp(t0, min=1e-30)),
        torch.maximum(
            log_h_centric(torch.clamp(F_obs, min=1e-30)),
            log_h_centric(torch.clamp(Fc, min=1e-30)),
        ),
    )
    x, w = F._gl_nodes(F.N_QUAD, DT, F_obs.device)
    acc = torch.zeros_like(F_obs)
    for k in range(F.N_QUAD):
        acc = acc + w[k] * torch.exp(
            log_h_centric(torch.clamp(mid + half * x[k], min=1e-30)) - shift
        )
    quad = -(torch.log(acc) + torch.log(half) + shift)

    err = (quad - exact).abs().max().item()
    assert err < 1e-6, f"quadrature vs exact centric closed form: max |dNLL| = {err:.3e}"


# ---------------------------------------------------------------------------
# 2. limit contract
# ---------------------------------------------------------------------------


def test_sigma_obs_to_zero_reduces_to_the_ml_target():
    """As ``sigma_obs -> 0`` the full form must collapse onto the current ``ml``
    likelihood (Rice / folded normal with variance ``epsilon*beta``) *exactly*.

    ``N(F_obs; t, sigma) -> delta(t - F_obs)``, so the integral returns
    ``p_m(F_obs)`` and the measurement normaliser -- which sits *inside* the
    integral -- cancels completely. There is no leftover ``-log sigma`` offset.

    The limit needs ``sigma << min(F_obs, sqrt(Sigma))``, so ``F_obs`` is set in
    absolute terms here rather than as a multiple of ``sigma``. At ``F_obs ~ sigma``
    the delta approximation legitimately fails: the Rice's linear-in-``t`` factor
    varies strongly across the Gaussian and the Gaussian is truncated at ``t = 0``.
    That is a real property of the likelihood, not a defect of the quadrature.
    """
    a = np.array(
        [(fo, 1e-6, fc, 1.0) for fo in (0.5, 1.0, 3.0, 20.0)
         for fc in (0.0, 0.5, 2.0, 10.0, 40.0)],
        dtype=np.float64,
    )
    F_obs, sig, Fc, Sigma = _t(a[:, 0], a[:, 1], a[:, 2], a[:, 3])
    for centric in (False, True):
        cen = torch.full(F_obs.shape, centric, dtype=torch.bool)
        got = F.ml_full_nll_per_refl(F_obs, sig, Fc, Sigma, cen, li0=F.log_i0_exact)
        want = rice_per_refl(F_obs, Fc, Sigma, cen)
        d = (got - want).abs().max().item()
        assert d < 1e-4, f"centric={centric}: sigma->0 limit off by {d:.3e}"


def test_beta_to_zero_reduces_to_a_pure_gaussian_on_the_amplitude():
    """With a perfect model (``beta -> 0``) only measurement error remains, so the
    likelihood must become ``N(F_obs; |F_calc|, sigma_obs)``."""
    F_obs, sig, Fc, _ = _grid([1.0], [5.0, 20.0], [0.5, 1.0, 2.0])
    Sigma = torch.full_like(F_obs, 1e-8)
    cen = torch.zeros_like(F_obs, dtype=torch.bool)
    got = F.ml_full_nll_per_refl(F_obs, sig, Fc, Sigma, cen, li0=F.log_i0_exact)
    want = 0.5 * (F.LOG_2PI + 2.0 * torch.log(sig)) + (F_obs - Fc) ** 2 / (2.0 * sig**2)
    assert (got - want).abs().max().item() < 1e-3


def test_scale_equivariance():
    """``F -> lam F, Sigma -> lam^2 Sigma`` must give ``NLL -> NLL + log lam`` exactly
    (the density has dimension 1/F). Guards against a lost Jacobian."""
    base = _t([50.0], [5.0], [45.0], [100.0])
    cen = torch.zeros(1, dtype=torch.bool)
    n0 = F.ml_full_nll_per_refl(*base, cen, li0=F.log_i0_exact)
    for lam in (1e-3, 1e3):
        sc = [base[0] * lam, base[1] * lam, base[2] * lam, base[3] * lam**2]
        n1 = F.ml_full_nll_per_refl(*sc, cen, li0=F.log_i0_exact)
        assert (n1 - (n0 + math.log(lam))).abs().item() < 1e-9


# ---------------------------------------------------------------------------
# 3. the physics this whole change is about
# ---------------------------------------------------------------------------


def test_epsilon_scales_the_model_variance_only_not_sigma_obs():
    """``epsilon`` is a symmetry enhancement of the *expected intensity*, so it
    multiplies the phase-carrying model variance ``beta`` and must leave the
    amplitude-only measurement error untouched.

    Checked by construction: ``ml_full(eps, beta)`` must equal
    ``ml_full(1, eps*beta)`` -- and must NOT equal a form in which ``sigma_obs`` is
    also inflated by ``eps``.
    """
    F_obs, sig, Fc, beta = _grid([0.3, 1.0], [1.0, 8.0], [0.5, 2.0])
    cen = torch.zeros_like(F_obs, dtype=torch.bool)
    for e in (2.0, 4.0):
        eps = torch.full_like(F_obs, e)
        via_eps = F.ml_full_nll_per_refl(F_obs, sig, Fc, beta, cen, eps, li0=F.log_i0_exact)
        via_beta = F.ml_full_nll_per_refl(F_obs, sig, Fc, e * beta, cen, li0=F.log_i0_exact)
        assert torch.allclose(via_eps, via_beta, atol=1e-10)

        # sigma_obs must NOT pick up epsilon
        wrong = F.ml_full_nll_per_refl(
            F_obs, sig * math.sqrt(e), Fc, e * beta, cen, li0=F.log_i0_exact
        )
        assert not torch.allclose(via_eps, wrong, atol=1e-6)


def test_log_i0_is_even():
    """``I0`` is even, so ``log I0`` must be too.

    Both implementations previously undid ``i0e = exp(-|z|) I0`` with ``+z`` instead
    of ``+|z|``, which returns ``log I0(z) - 2|z|`` for negative argument. Production
    never sees ``z < 0``, so the value path was unaffected -- but it broke the
    evenness of ``NLL(F_calc)`` and made every finite-difference gradient check
    report a spurious non-zero derivative at ``F_calc = 0``.
    """
    z = torch.linspace(0.0, 200.0, 4001, dtype=DT)
    for fn in (F.log_i0, F.log_i0_exact):
        assert torch.allclose(fn(z), fn(-z), atol=0.0, rtol=0.0)


def test_nll_is_even_in_f_calc_and_flat_at_zero():
    """``NLL`` depends on ``F_calc`` only through ``|F_calc|`` and ``F_calc**2``, so it
    is even and its derivative vanishes at ``F_calc = 0``."""
    F_obs, sig, _, Sigma = _grid([0.01, 1.0], [0.0], [0.0, 2.0])
    cen = torch.zeros_like(F_obs, dtype=torch.bool)
    for fc in (0.05, 0.5):
        p = F.ml_full_nll_per_refl(F_obs, sig, torch.full_like(F_obs, fc), Sigma, cen,
                                   li0=F.log_i0_exact)
        m = F.ml_full_nll_per_refl(F_obs, sig, torch.full_like(F_obs, -fc), Sigma, cen,
                                   li0=F.log_i0_exact)
        assert torch.allclose(p, m, atol=1e-12)

    Fc = torch.zeros_like(F_obs).requires_grad_(True)
    out = F.ml_full_nll_per_refl(F_obs, sig, Fc, Sigma, cen, li0=F.log_i0_exact).sum()
    (g,) = torch.autograd.grad(out, Fc)
    assert g.abs().max().item() < 1e-10


# ---------------------------------------------------------------------------
# 4. gradients
# ---------------------------------------------------------------------------


def test_gradient_matches_a_converged_reference():
    """``dNLL/d|F_calc|`` at the production node count must match a much finer rule.

    Self-convergence rather than finite differences: the quadrature gradient was
    verified to agree between N=32 and N=128 to ~1e-15 relative, whereas an FD
    reference is limited to ~1e-8 and is the noisier of the two.
    """
    F_obs, sig, Fc, Sigma = _grid([0.01, 0.3, 3.0, 100.0], [0.0, 1.0, 20.0], [0.0, 2.0, 60.0])
    cen = torch.zeros_like(F_obs, dtype=torch.bool)
    grads = []
    for n in (F.N_QUAD, 128):
        fc = Fc.clone().requires_grad_(True)
        out = F.ml_full_nll_per_refl(F_obs, sig, fc, Sigma, cen, n_quad=n,
                                    li0=F.log_i0_exact).sum()
        (g,) = torch.autograd.grad(out, fc)
        grads.append(g)
    scale = torch.clamp(grads[1].abs(), min=1e-3)
    assert ((grads[0] - grads[1]).abs() / scale).max().item() < 1e-6


def test_gradients_are_finite_including_the_unused_parity_branch():
    """Gathering per parity (rather than ``torch.where``) means a branch that is not
    evaluated cannot emit NaN. Exercise all-acentric and all-centric inputs."""
    F_obs, sig, Fc, Sigma = _grid([1e-3, 1.0, 100.0], [0.0, 30.0], [0.0, 5.0])
    for centric in (False, True):
        cen = torch.full(F_obs.shape, centric, dtype=torch.bool)
        fc = Fc.clone().requires_grad_(True)
        out = F.ml_full_nll_per_refl(F_obs, sig, fc, Sigma, cen, li0=F.log_i0_exact).sum()
        (g,) = torch.autograd.grad(out, fc)
        assert torch.isfinite(out).all() and torch.isfinite(g).all()


def test_log_ndtr_lower_branch_stays_finite():
    """The centric ``s = -1`` branch drives ``Phi``'s argument deep negative;
    ``log(ndtr(.))`` underflows to ``-inf`` there and ``log_ndtr`` does not."""
    F_obs, sig, Fc, Sigma = _grid([1e-3, 1e-2], [50.0, 200.0], [1.0])
    fc = Fc.clone().requires_grad_(True)
    out = F.centric_nll(F_obs, sig, fc, Sigma)
    assert torch.isfinite(out).all()
    (g,) = torch.autograd.grad(out.sum(), fc)
    assert torch.isfinite(g).all()


# ---------------------------------------------------------------------------
# 5. mechanics
# ---------------------------------------------------------------------------


def test_parity_gather_matches_elementwise_evaluation():
    """The gather/scatter dispatch must agree with evaluating each parity alone."""
    F_obs, sig, Fc, Sigma = _grid([0.1, 2.0], [0.5, 10.0], [0.5, 3.0])
    cen = torch.zeros_like(F_obs, dtype=torch.bool)
    cen[::2] = True
    mixed = F.ml_full_nll_per_refl(F_obs, sig, Fc, Sigma, cen, li0=F.log_i0_exact)
    all_a = F.acentric_nll(F_obs, sig, Fc, Sigma, li0=F.log_i0_exact)
    all_c = F.centric_nll(F_obs, sig, Fc, Sigma)
    want = torch.where(cen, all_c, all_a)
    assert torch.allclose(mixed, want, atol=1e-12)


def test_masked_sum_and_empty_parity():
    F_obs, sig, Fc, Sigma = _grid([1.0], [2.0], [1.0, 3.0])
    cen = torch.zeros_like(F_obs, dtype=torch.bool)  # no centrics at all
    per = F.ml_full_nll_per_refl(F_obs, sig, Fc, Sigma, cen, li0=F.log_i0_exact)
    tot = rice_marginal_math(F_obs, Fc, Sigma, sig, cen, li0=F.log_i0_exact)
    assert torch.allclose(tot, per.sum(), atol=1e-10)
    mask = torch.zeros_like(F_obs)
    mask[0] = 1.0
    part = rice_marginal_math(F_obs, Fc, Sigma, sig, cen, mask=mask, li0=F.log_i0_exact)
    assert torch.allclose(part, per[0], atol=1e-10)


def test_log_i0_fast_accuracy_budget():
    """The hand-rolled polynomial replaces ``i0e`` (25x ``exp``); pin its error."""
    z = torch.cat([
        torch.linspace(0.0, 3.75, 20001, dtype=DT),
        torch.logspace(math.log10(3.75), 4.0, 20001, dtype=DT),
    ])
    err = (F.log_i0(z) - F.log_i0_exact(z)).abs().max().item()
    assert err < 2e-6, f"log_i0 polynomial error {err:.3e}"


def test_float32_runs_and_tracks_float64():
    """float32 is the production dtype. Its error is set by the magnitude of the
    per-reflection NLL, not by the quadrature; keep the realistic range honest."""
    F_obs, sig, Fc, Sigma = _grid([0.1, 1.0, 10.0], [0.5, 5.0], [0.5, 2.0])
    cen = torch.zeros_like(F_obs, dtype=torch.bool)
    v64 = F.ml_full_nll_per_refl(F_obs, sig, Fc, Sigma, cen)
    v32 = F.ml_full_nll_per_refl(
        *[x.to(torch.float32) for x in (F_obs, sig, Fc, Sigma)], cen
    )
    assert torch.isfinite(v32).all()
    d = (v32.to(DT) - v64).abs().max().item()
    assert d < 1e-2, f"f32 vs f64 {d:.3e}"


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_degenerate_sizes(n):
    """Empty and size-1 inputs must work: a dataset can have very few centrics, and
    size 1 is also where torch.compile's 0/1 specialisation would bite."""
    F_obs, sig, Fc, Sigma = (torch.full((n,), v, dtype=DT) for v in (5.0, 1.0, 4.0, 2.0))
    cen = torch.zeros(n, dtype=torch.bool)
    out = F.ml_full_nll_per_refl(F_obs, sig, Fc, Sigma, cen, li0=F.log_i0_exact)
    assert out.shape == (n,)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 6. the compiled path
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_compiled_path_matches_eager_and_does_not_recompile_per_size():
    """The fused kernel must agree with eager, and one compilation must serve every
    reflection count.

    Only the leading dimension varies (``n_quad``, dtype and device are fixed), so
    ``dynamic=True`` is sufficient and no chunking/padding layer is needed. This
    pins that: ``unique_graphs`` must not grow as sizes change.
    """
    import torchref.config as cfg
    from torch._dynamo.utils import counters

    torch.manual_seed(0)

    def inputs(n):
        g = torch.Generator().manual_seed(n)
        return (
            torch.rand(n, generator=g) * 100 + 1,
            torch.rand(n, generator=g) * 5 + 0.5,
            torch.rand(n, generator=g) * 100 + 1,
            torch.rand(n, generator=g) * 500 + 10,
        )

    cfg.compile_targets.value = False
    eager = {n: F.acentric_nll(*inputs(n)) for n in (37, 1000, 4096)}

    F._COMPILED.clear()
    counters.clear()
    cfg.compile_targets.value = True
    try:
        seen = []
        for n in (37, 1000, 4096):
            got = F.acentric_nll(*inputs(n))
            seen.append(counters["stats"].get("unique_graphs", 0))
            assert torch.allclose(got, eager[n], atol=1e-4, rtol=1e-4), f"n={n}"
        assert seen[-1] == seen[0], (
            f"recompiled as the reflection count changed: unique_graphs {seen}"
        )
    finally:
        cfg.compile_targets.value = False
        F._COMPILED.clear()


def test_compile_switch_is_respected_and_float64_stays_eager():
    """float64 must bypass the compiled float32 kernel (it is the reference path)."""
    import torchref.config as cfg

    x = [torch.rand(64, dtype=torch.float64) * 10 + 1 for _ in range(4)]
    cfg.compile_targets.value = True
    try:
        F._COMPILED.clear()
        F.acentric_nll(*x)
        assert not F._COMPILED, "float64 should not have triggered a compile"
    finally:
        cfg.compile_targets.value = False
        F._COMPILED.clear()


# ---------------------------------------------------------------------------
# 7. the Luzzati mean coupling alpha
# ---------------------------------------------------------------------------


def test_alpha_enters_only_as_the_rice_mean():
    """``alpha`` must be exactly equivalent to scaling ``|F_calc|``.

    It appears in the model term solely as ``alpha*Fc`` -- via ``(alpha*Fc)**2`` and
    ``2 t (alpha*Fc)`` -- so folding it into ``Fc`` is not an approximation. This
    pins that, and would catch any future attempt to give alpha its own term.
    """
    F_obs, sig, Fc, Sigma = _grid([0.1, 1.0, 10.0], [0.5, 2.0, 20.0], [0.5, 2.0])
    for centric in (False, True):
        cen = torch.full(F_obs.shape, centric, dtype=torch.bool)
        for a in (0.5, 0.9, 1.3):
            alpha = torch.full_like(F_obs, a)
            via_alpha = F.ml_full_nll_per_refl(
                F_obs, sig, Fc, Sigma, cen, alpha=alpha, li0=F.log_i0_exact
            )
            via_fc = F.ml_full_nll_per_refl(
                F_obs, sig, a * Fc, Sigma, cen, li0=F.log_i0_exact
            )
            assert torch.allclose(via_alpha, via_fc, atol=1e-12), f"alpha={a}"


def test_alpha_one_and_alpha_none_agree():
    """``alpha=1`` must reproduce the alpha-free path bit-for-bit, so enabling the
    machinery cannot perturb the existing behaviour."""
    F_obs, sig, Fc, Sigma = _grid([0.01, 1.0, 100.0], [0.0, 5.0], [0.0, 3.0])
    cen = torch.zeros_like(F_obs, dtype=torch.bool)
    cen[::2] = True
    none = F.ml_full_nll_per_refl(F_obs, sig, Fc, Sigma, cen, li0=F.log_i0_exact)
    ones = F.ml_full_nll_per_refl(
        F_obs, sig, Fc, Sigma, cen, alpha=torch.ones_like(F_obs), li0=F.log_i0_exact
    )
    assert torch.allclose(none, ones, atol=0.0, rtol=0.0)


def test_estimator_alpha_satisfies_the_second_moment_identity():
    """``alpha**2 Sigma_P + beta_model + S2 == B`` must hold EXACTLY, per shell.

    alpha and both betas are now derived from one bounded ``sigma_A``, so the identity
    is algebraic rather than approximate. The previous implementation fitted alpha on
    the raw moment ``B`` and subtracted the measurement variance from beta afterwards,
    which left the shipped pair implying ``sigma_A**2 = (B - beta_raw)/(B - S2) > 1``
    whenever ``beta_raw < S2``. This test is the reason that cannot recur.

    Note the old version of this test asserted the OPPOSITE of the last check below --
    that alpha is "identical with and without" sigma_obs. That was pinning the defect:
    if supplying sigma_obs moves beta onto the ``Sigma_N = B - S2`` surface while alpha
    stays on the raw-``B`` surface, the two are no longer describing the same model.
    """
    from torchref.refinement.model_error_estimation.sigma_a import estimate_beta

    g = torch.Generator().manual_seed(0)
    n = 4000
    Fc = torch.rand(n, generator=g, dtype=DT) * 100 + 1
    # alpha_true = 0.8, beta_true = 400, plus measurement noise
    t = 0.8 * Fc + torch.randn(n, generator=g, dtype=DT) * 20
    sig = torch.full((n,), 5.0, dtype=DT)
    F_obs = (t + torch.randn(n, generator=g, dtype=DT) * sig).abs()
    cen = torch.zeros(n, dtype=torch.bool)
    eps = torch.ones(n, dtype=DT)
    dss = torch.linspace(0.01, 0.3, n, dtype=DT)
    free = torch.zeros(n, dtype=torch.bool)
    free[::2] = True

    sh = estimate_beta(F_obs, Fc, cen, eps, dss, free)
    assert (sh.alpha > 0).all() and torch.isfinite(sh.alpha).all()
    # recovered alpha should sit near the truth, not at 1
    assert 0.5 < float(sh.alpha.median()) < 1.2
    # with no sigma_obs, S2 == 0 and the two variances coincide
    assert float(sh.S2.abs().max()) == 0.0
    assert torch.equal(sh.beta, sh.beta_model)

    sh_s = estimate_beta(F_obs, Fc, cen, eps, dss, free, sigma_obs=sig)
    tol = 1e-10 if DT == torch.float64 else 1e-5
    for s in (sh, sh_s):
        resid = (s.alpha**2 * s.Sigma_P + s.beta_model + s.S2 - s.B).abs() / s.B
        assert float(resid.max()) < tol, f"identity violated by {float(resid.max()):.2e}"
        # sigma_A <= 1 is structural, so alpha can never exceed its physical bound
        bound = (s.Sigma_N / s.Sigma_P).sqrt()
        assert (s.alpha <= bound * (1 + 1e-5)).all()

    # supplying sigma_obs strips the measurement variance out of beta_model, and the
    # difference is exactly S2 -- not an unrecoverable clamped subtraction.
    assert (sh_s.beta_model <= sh_s.beta + 1e-9).all()
    assert torch.allclose(sh_s.beta - sh_s.beta_model, sh_s.S2, atol=tol * 10)
    assert float(sh_s.S2.min()) > 0.0, "sigma_obs was supplied, so S2 must be positive"
