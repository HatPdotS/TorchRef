"""Tests for ``nll_beta``: Gaussian amplitude NLL with ``ml``'s per-shell variance.

This target exists to separate two things that ``ml`` conflates -- the *variance
model* (per-shell beta) and the *distribution shape* (Rice/Bessel). It completes a
2x2 with ``gaussian`` (Gaussian + sigma_obs) and ``rice`` (Rice + sigma_obs).

The load-bearing test is `test_gradient_matches_rice_in_the_strong_signal_limit`:
the whole construction is only meaningful if the Gaussian variance is the correct
large-signal limit of the MLF. A wrong factor (``Sigma`` instead of ``Sigma/2`` for
acentrics) rescales every x-ray gradient by 2, which is indistinguishable from an
x-ray weight change and would silently confound the head-to-head.
"""


import pytest
import torch

from torchref.base.targets.xray_likelihoods import (
    amplitude_var_from_complex,
    complex_var_from_beta,
    inflate_with_sigma_obs,
    nll_math,
    rice_math,
    rice_per_refl,
)

DT = torch.float64


def _rice(Fo, Fc, Sigma, cen):
    """``ml`` / ``ml_noalpha``: Rice at the COMPLEX variance."""
    return rice_math(Fo, Fc, Sigma, cen)


def _nll_beta(Fo, Fc, Sigma, cen):
    """``nll_beta``: the same variance, converted to an AMPLITUDE variance first.

    That conversion (``Sigma/2`` acentric, ``Sigma`` centric) is the large-signal limit
    being tested below, so it must be the real builder rather than an inline factor.
    """
    return nll_math(Fo, Fc, amplitude_var_from_complex(Sigma, cen))


def _grad_wrt_fc(loss_of_sigma, Fo, Fc0, beta, cen, eps, sigma_obs=None):
    """``d(loss)/d|Fc|`` for a ``loss_of_sigma(Fo, Fc, Sigma, cen) -> scalar``.

    Builds ``Sigma`` here so both arms provably see the SAME complex variance -- the point
    of the comparison is the distribution shape, so a difference in variance construction
    would confound it.
    """
    Sigma = complex_var_from_beta(beta, eps)
    if sigma_obs is not None:
        Sigma = inflate_with_sigma_obs(Sigma, sigma_obs, cen)
    Fc = Fc0.clone().requires_grad_(True)
    loss_of_sigma(Fo, Fc, Sigma, cen).backward()
    return Fc.grad


@pytest.mark.parametrize("centric", [False, True])
def test_gradient_matches_rice_in_the_strong_signal_limit(centric):
    """d(NLL)/d|Fc| must agree with the MLF where 2*Fo*Fc/Sigma >> 1.

    Gradients are what drive refinement, so this is the meaningful equivalence --
    the losses themselves differ by additive terms that carry no gradient.
    """
    n = 2000
    g = torch.Generator().manual_seed(0)
    # strong signal: Sigma small relative to Fo*Fc
    Fo = 50 + 20 * torch.rand(n, generator=g, dtype=DT)
    Fc = 50 + 20 * torch.rand(n, generator=g, dtype=DT)
    beta = 1 + torch.rand(n, generator=g, dtype=DT)
    eps = torch.ones(n, dtype=DT)
    cen = torch.full((n,), centric, dtype=torch.bool)

    g_rice = _grad_wrt_fc(_rice, Fo, Fc, beta, cen, eps)
    g_gauss = _grad_wrt_fc(_nll_beta, Fo, Fc, beta, cen, eps)

    rel = ((g_gauss - g_rice).abs() / g_rice.abs().clamp(min=1e-12)).median()
    assert float(rel) < 0.02, (
        f"centric={centric}: median relative gradient mismatch {float(rel):.4f} -- "
        f"the Gaussian variance is not the MLF's large-signal limit"
    )


def test_wrong_parity_factor_would_be_caught():
    """Guard the factor itself: using Sigma (not Sigma/2) for acentrics doubles the
    gradient, so the test above genuinely constrains it rather than passing trivially.
    """
    n = 500
    g = torch.Generator().manual_seed(1)
    Fo = 50 + 20 * torch.rand(n, generator=g, dtype=DT)
    Fc = 50 + 20 * torch.rand(n, generator=g, dtype=DT)
    beta = torch.ones(n, dtype=DT)
    eps = torch.ones(n, dtype=DT)
    acen = torch.zeros(n, dtype=torch.bool)

    good = _grad_wrt_fc(_nll_beta, Fo, Fc, beta, acen, eps)
    # beta doubled == using Sigma instead of Sigma/2
    bad = _grad_wrt_fc(_nll_beta, Fo, Fc, 2 * beta, acen, eps)
    ratio = (good / bad).median()
    assert abs(float(ratio) - 2.0) < 0.05, f"expected 2x, got {float(ratio):.3f}"


def test_diverges_from_rice_in_the_low_information_regime():
    """The targets MUST differ somewhere, or the 2x2 is vacuous. This pins WHERE.

    The gradients w.r.t. |Fc| are::

        Rice:     2 Fc/Sigma - (2 Fo/Sigma) * m(z),   m = I1/I0(z), z = 2 Fo Fc/Sigma
        Gaussian: 2 Fc/Sigma - (2 Fo/Sigma)

    so they differ by ``(2 Fo/Sigma) * (1 - m(z))``. Note what that implies:

    * ``Fo -> 0`` does NOT separate them -- the difference carries a factor ``Fo``,
      and both targets simply pull ``|Fc| -> 0``. (Measured: they agree to 1.4% at
      ``Fo ~ 0.1, Fc ~ 7, Sigma ~ 75``. An earlier version of this test asserted
      divergence there and was simply wrong about the physics.)
    * Separation needs ``Fo`` LARGE and ``z`` SMALL, i.e. ``Fo ~ Fc`` with
      ``Sigma >> Fo*Fc``: a well-measured reflection the model cannot explain. That
      is where Rice's phase-error annulus differs from a symmetric Gaussian.
    """
    n = 2000
    g = torch.Generator().manual_seed(2)
    Fo = 4 + 2 * torch.rand(n, generator=g, dtype=DT)
    Fc = 4 + 2 * torch.rand(n, generator=g, dtype=DT)  # Fo ~ Fc
    beta = 400 + 200 * torch.rand(n, generator=g, dtype=DT)  # Sigma >> Fo*Fc
    eps = torch.ones(n, dtype=DT)
    cen = torch.zeros(n, dtype=torch.bool)

    g_rice = _grad_wrt_fc(_rice, Fo, Fc, beta, cen, eps)
    g_gauss = _grad_wrt_fc(_nll_beta, Fo, Fc, beta, cen, eps)
    # Ratio of medians, NOT median of ratios: the Gaussian gradient passes through
    # zero at Fo == Fc, so per-element ratios are unbounded there.
    rel = float((g_gauss - g_rice).abs().median() / g_rice.abs().median())
    assert rel > 0.2, (
        f"low-information gradients differ by only {rel:.4f} -- then nll_beta is "
        f"not actually a different target and the 2x2 is vacuous"
    )


def test_sigma_obs_term_matches_the_ml_convention():
    """``sigma_in_variance`` must inflate Sigma the same way in both targets."""
    n = 400
    g = torch.Generator().manual_seed(3)
    Fo = 10 + 5 * torch.rand(n, generator=g, dtype=DT)
    Fc = 10 + 5 * torch.rand(n, generator=g, dtype=DT)
    beta = 1 + torch.rand(n, generator=g, dtype=DT)
    eps = torch.randint(1, 4, (n,), generator=g).to(DT)
    sig = 0.5 + torch.rand(n, generator=g, dtype=DT)
    cen = torch.rand(n, generator=g) < 0.3
    parity = torch.where(cen, torch.ones_like(sig), torch.full_like(sig, 2.0))

    def _nll_beta(b, sigma_obs=None):
        Sigma = complex_var_from_beta(b, eps)
        if sigma_obs is not None:
            Sigma = inflate_with_sigma_obs(Sigma, sigma_obs, cen)
        return nll_math(Fo, Fc, amplitude_var_from_complex(Sigma, cen))

    direct = _nll_beta(beta, sigma_obs=sig)
    absorbed = _nll_beta(beta + parity * sig**2 / eps)
    assert torch.allclose(direct, absorbed, atol=1e-9)


def test_rice_sigma_obs_follows_refmacs_inflation_convention():
    """``_ml_beta_nll_per_refl(sigma_obs=...)`` is Refmac's ``ll_amp`` variance.

No target uses it -- there is no Rice-with-sigma_obs mode, and ``_specs.py`` explains why
    on physics grounds. ``inflate_with_sigma_obs`` is kept solely so the equivalence with
    ``servalcat/src/amplitude.cpp::ll_amp`` stays reproducible, which makes it dead weight
    unless something actually checks it. Hence this test. (It used to be an unreachable
    *argument* on two loss functions; it is now a named builder, which is why this test
    calls it directly.)

    Two properties of Refmac's convention are easy to get backwards and are both pinned:

    * the parity factor is ``(3 - c)`` -- **2** for acentrics, **1** for centrics, i.e.
      the opposite way round from most epsilon-like factors;
    * ``sigma_obs**2`` is **not** scaled by ``epsilon``. Refmac folds epsilon into ``S``
      only ("S: must include epsilon"), so the measurement term is added flat.
    """
    n = 400
    g = torch.Generator().manual_seed(4)
    Fo = 10 + 5 * torch.rand(n, generator=g, dtype=DT)
    Fc = 10 + 5 * torch.rand(n, generator=g, dtype=DT)
    beta = 1 + torch.rand(n, generator=g, dtype=DT)
    eps = torch.randint(1, 4, (n,), generator=g).to(DT)
    sig = 0.5 + torch.rand(n, generator=g, dtype=DT)
    cen = torch.rand(n, generator=g) < 0.3
    parity = torch.where(cen, torch.ones_like(sig), torch.full_like(sig, 2.0))

    direct = rice_per_refl(
        Fo, Fc, inflate_with_sigma_obs(complex_var_from_beta(beta, eps), sig, cen), cen
    )
    # Same Sigma reached by hand: Sigma = eps*beta + (3-c)*sigma**2, so the equivalent
    # beta is beta + (3-c)*sigma**2/eps. If the epsilon scaling of the sigma term were
    # wrong, this identity would fail on the eps != 1 reflections only.
    absorbed = rice_per_refl(
        Fo, Fc, complex_var_from_beta(beta + parity * sig**2 / eps, eps), cen
    )
    assert torch.allclose(direct, absorbed, atol=1e-9)
    assert bool((eps != 1).any()), "vacuous: needs eps != 1 to test the scaling"


def test_each_mode_has_its_own_class():
    """One class per selectable mode -- the 2026-08 refactor's thesis, as an assertion.

    Replaces a test that pinned the spec table's ``(distribution, variance, mean)`` columns.
    Those columns existed only to let ONE class serve four rows by branching on them at
    runtime; the rows are now separate classes and the columns are gone, so asserting them
    would be asserting documentation. The table itself enforces uniqueness at import, so this
    checks the mapping rather than re-deriving it.
    """
    from torchref.refinement.targets.xray import (
        LeastSquaresXrayTarget,
        MLFullXrayTarget,
        MLNoAlphaXrayTarget,
        MLXrayTarget,
        NLLBetaXrayTarget,
        NLLXrayTarget,
        UnitWeightK1XrayTarget,
    )
    from torchref.refinement.targets.xray._specs import XRAY_TARGETS

    expected = {
        "ml": MLXrayTarget,
        "ml_noalpha": MLNoAlphaXrayTarget,
        "ml_full": MLFullXrayTarget,
        "nll_beta": NLLBetaXrayTarget,
        "nll": NLLXrayTarget,
        "ls": LeastSquaresXrayTarget,
        "ls_wunit_k1": UnitWeightK1XrayTarget,
    }
    assert set(expected) == set(XRAY_TARGETS.names), "table and test disagree on the rows"
    for name, cls in expected.items():
        assert XRAY_TARGETS.by_name(name).target_cls is cls, name

    seen = {}
    for name in XRAY_TARGETS.names:
        cls = XRAY_TARGETS.by_name(name).target_cls
        assert cls not in seen, (
            f"{name} and {seen[cls]} share {cls.__name__}: a class serving two modes has to "
            f"branch on something at runtime, which is what this refactor removed"
        )
        seen[cls] = name


def test_only_the_estimator_backed_rows_own_an_estimator():
    """``nll`` must not pay for a model-error estimate it does not use.

    Was ``spec.needs_estimator``, a field whose only reader was the factory's dispatch.
    Now it is a property of the class hierarchy, so ask the hierarchy.
    """
    from torchref.refinement.targets.xray import (
        NLLXrayTarget,
        SigmaAXrayTarget,
    )
    from torchref.refinement.targets.xray._specs import XRAY_TARGETS

    for name in ("ml", "ml_noalpha", "nll_beta", "ml_full"):
        cls = XRAY_TARGETS.by_name(name).target_cls
        assert issubclass(cls, SigmaAXrayTarget), name
    assert not issubclass(NLLXrayTarget, SigmaAXrayTarget)
    assert not hasattr(NLLXrayTarget(), "_sigma_a"), (
        "nll built an estimator it never reads"
    )


def test_which_likelihood_each_row_evaluates(monkeypatch):
    """Behavioural replacement for the spec's ``distribution`` column.

    Each row's ``_loss`` must reach exactly one of the three primitives. Checked by
    capturing stubs rather than by reading source text, so it cannot go vacuous when the
    call moves.
    """
    import torchref.refinement.targets.xray.ml_full as ml_full_mod
    import torchref.refinement.targets.xray.ml_noalpha as ml_noalpha_mod
    import torchref.refinement.targets.xray.nll_beta as nll_beta_mod
    from torchref.refinement.targets.xray import (
        MLFullXrayTarget,
        MLNoAlphaXrayTarget,
        MLXrayTarget,
        NLLBetaXrayTarget,
    )

    fired = []

    def _stub(tag):
        def f(*a, **kw):
            fired.append((tag, kw.get("sigma_obs", "positional-or-absent")))
            return torch.zeros((), dtype=DT)
        return f

    monkeypatch.setattr(ml_noalpha_mod, "rice_math", _stub("rice"))
    monkeypatch.setattr(nll_beta_mod, "nll_math", _stub("nll"))
    monkeypatch.setattr(ml_full_mod, "rice_marginal_math", _stub("rice_marginal"))

    n = 8
    ctx = _fake_ctx(n)
    for cls, want in (
        (MLXrayTarget, "rice"),
        (MLNoAlphaXrayTarget, "rice"),
        (NLLBetaXrayTarget, "nll"),
        (MLFullXrayTarget, "rice_marginal"),
    ):
        fired.clear()
        cls()._loss(ctx)
        assert [t for t, _ in fired] == [want], f"{cls.__name__}: {fired}"


def _fake_ctx(n):
    """A minimal :class:`SigmaALossInputs` for exercising ``_loss`` in isolation."""
    from torchref.refinement.targets.xray import SigmaALossInputs

    class _Sub:
        centric = torch.zeros(n, dtype=torch.bool)

        def select(self, t):
            return t

    ones = torch.ones(n, dtype=DT)
    return SigmaALossInputs(
        F_obs=ones,
        F_calc=ones,
        Sigma=ones,
        centric=torch.zeros(n, dtype=torch.bool),
        sigma_obs_full=ones,
        est=None,
        sub=_Sub(),
    )


def test_rice_with_sigma_obs_is_not_offered():
    """`rice` was removed on physical grounds and must not come back silently.

    sigma_obs is a 1-DOF amplitude error with no phase component, so it cannot be the
    Sigma of a Rice likelihood (which comes from a 2-D isotropic complex Gaussian
    marginalised over phase). Empirically it was the worst target measured -- bond RMSZ
    28.0 where every other target sat near 1.3.
    """
    import pytest

    from torchref.refinement.targets.xray._specs import XRAY_TARGETS

    assert "rice" not in XRAY_TARGETS.names
    with pytest.raises(ValueError, match="Unknown X-ray target mode"):
        XRAY_TARGETS.by_name("rice")
    # And no surviving ROW maps to the Rice-at-sigma_obs class. That class still exists --
    # `RiceXrayTarget`, kept private for the MR aligner, which has no test coverage of its
    # own so repointing it would be an untested numerical change -- but it must not be
    # selectable. This replaces a check on the spec table's `distribution`/`variance`
    # columns, which existed only to drive dispatch and were removed with it.
    from torchref.refinement.targets.xray import RiceXrayTarget

    for spec in XRAY_TARGETS.specs:
        assert spec.target_cls is not RiceXrayTarget, (
            f"{spec.name} makes the private Rice-at-sigma_obs target selectable again"
        )
        assert not issubclass(spec.target_cls, RiceXrayTarget), spec.name


def test_finite_and_positive_variance_on_degenerate_input():
    """beta -> 0 and Fo -> 0 must not produce inf/nan."""
    Fo = torch.tensor([0.0, 1.0, 0.0], dtype=DT)
    Fc = torch.tensor([0.0, 0.0, 5.0], dtype=DT)
    beta = torch.tensor([0.0, 1e-30, 1.0], dtype=DT)
    eps = torch.ones(3, dtype=DT)
    cen = torch.tensor([False, True, False])
    v = nll_math(
        Fo, Fc,
        amplitude_var_from_complex(complex_var_from_beta(beta, eps), cen),
    )
    assert torch.isfinite(v), v


def test_mean_centring_is_intrinsic_to_the_spec_not_a_flag():
    """alpha is a property of the target, not a toggle.

    History: `use_alpha` was a CLI flag on every sigma_A target. It was also one of five
    flags that never reached the target (a double-build in LBFGSRefinement reset them),
    so the "mean centring adds nothing" measurement was never actually made. Rather than
    fix the flag, alpha became intrinsic: `ml_full` centres on `alpha*|F_c|` by
    definition and the others on `|F_c|`. That is expressible in the spec, so there is
    nothing left to get out of sync.
    """
    import inspect

    from torchref.refinement.targets.xray import (
        MLFullXrayTarget,
        MLNoAlphaXrayTarget,
        MLXrayTarget,
        NLLBetaXrayTarget,
        NLLXrayTarget,
        SigmaAXrayTarget,
    )

    # No target may take a mode selector of any kind: that is the whole point of the split.
    for cls in (MLXrayTarget, MLNoAlphaXrayTarget, MLFullXrayTarget, NLLBetaXrayTarget,
                NLLXrayTarget, SigmaAXrayTarget):
        params = set(inspect.signature(cls.__init__).parameters)
        assert "use_alpha" not in params, f"{cls.__name__}: use_alpha came back as a flag"
        assert not ({"spec", "mode", "distribution", "variance", "mean"} & params), (
            f"{cls.__name__} takes a mode selector: {sorted(params)}"
        )

    # BEHAVIOURAL, not source-text: `_mean` must actually apply alpha for the two rows that
    # centre on it, and must not for the others. Strictly stronger than the string check it
    # replaces -- this catches a subclass that FORGOT the override, which a grep could not.
    class _Est:
        alpha = torch.full((4,), 2.0, dtype=DT)

    class _Sub:
        def select(self, t):
            return t

    F_calc = torch.arange(1, 5, dtype=DT)
    for cls in (MLXrayTarget, MLFullXrayTarget):
        got = cls()._mean(F_calc, _Est(), _Sub())
        assert torch.equal(got, 2.0 * F_calc), f"{cls.__name__} does not centre on alpha"
    for cls in (MLNoAlphaXrayTarget, NLLBetaXrayTarget):
        got = cls()._mean(F_calc, _Est(), _Sub())
        assert torch.equal(got, F_calc), f"{cls.__name__} centres on alpha but must not"
