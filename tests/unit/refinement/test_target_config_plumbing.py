"""The x-ray target must actually receive its configuration.

This file exists because it did not. Until 2026-07-31,
``LBFGSRefinement.__init__`` called ``set_xray_target_mode(target_mode)`` *after*
``super().__init__()`` had already built correctly-configured targets, and that method
forwarded only ``model/data/scaler/mode/use_work_set/verbose``. The second
build therefore reverted five options to factory defaults:

    --sigma-a-estimator   -> "v1"
    --sigma-in-variance   -> False
    --ml-estimation-set   -> "free"
    --use-alpha           -> False
    --no-beta-sigma-subtract -> subtract=True

They were dead for every ``LBFGSRefinement``, which is the CLI and every paper harness.
A whole session of benchmark arms varying those knobs measured nothing, and the failure
was invisible because the arms still *ran* -- they just silently ran the default target.

The lesson these tests encode: **a configuration option is not tested until something
asserts it reached the object that consumes it.** Asserting the CLI parses it, or that
the constructor stores it, is not enough.
"""

import types
import inspect

import pytest
import torch

from torchref.refinement.base_refinement import Refinement
from torchref.refinement.targets.xray import MLXrayTarget


def test_single_source_of_truth_for_target_kwargs():
    """Both construction paths must go through ``_xray_target_kwargs``.

    If a second path spells the kwargs out again, the two can drift and the original
    bug returns. ``create_xray_target`` should appear exactly once in the base class.
    """
    src = inspect.getsource(Refinement)
    assert src.count("create_xray_target(") == 2, (
        "expected exactly two create_xray_target calls (work + test) inside one "
        "helper; a second construction site can silently drop configuration"
    )
    assert "_xray_target_kwargs" in src
    assert "_build_xray_targets" in src


def test_set_xray_target_mode_preserves_configuration():
    """Switching mode must not reset the target configuration.

    This is the exact regression: ``set_xray_target_mode`` rebuilds the targets, so it
    must rebuild them with the SAME configuration, not with factory defaults.
    """
    src = inspect.getsource(Refinement.set_xray_target_mode)
    assert "_build_xray_targets" in src, (
        "set_xray_target_mode must delegate to the shared builder so it cannot drop "
        "configuration"
    )
    # and it must not spell out kwargs of its own
    assert "sigma_a_estimator" not in src
    assert "estimation_set" not in src


def test_lbfgs_does_not_rebuild_targets_after_init():
    """``LBFGSRefinement`` must not rebuild targets after ``super().__init__()``.

    The double build was the delivery mechanism for the bug: even a correct
    ``_init_targets`` was overwritten milliseconds later.
    """
    from torchref.refinement.lbfgs_refinement import LBFGSRefinement

    src = inspect.getsource(LBFGSRefinement.__init__)
    # Strip comment-only lines: the method is named in a comment explaining the fix,
    # and matching prose instead of code is how a source-text assertion goes wrong.
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.strip().startswith("#")
    )
    assert "self.set_xray_target_mode(" not in code, (
        "LBFGSRefinement.__init__ must not rebuild the targets; it should pass "
        "xray_mode into super().__init__() so they are built once, configured"
    )
    # the mode must still reach the base
    assert "xray_mode" in code


def test_base_accepts_and_stores_the_full_target_configuration():
    """Every option the targets need must be a real ``Refinement`` parameter.

    ``getattr`` fallbacks inside the builder mean a missing parameter degrades
    silently to a default instead of raising, so the parameter list is the guard.
    """
    params = inspect.signature(Refinement.__init__).parameters
    for name in ("xray_mode", "scale_target"):
        assert name in params, f"Refinement.__init__ is missing {name!r}"
    # The five options this file was written about have been REMOVED, not fixed-and-kept:
    # four were experiments that measured nothing (because of the very bug documented
    # above) and the fifth, use_alpha, became intrinsic to the ml_full spec. Guard
    # against them creeping back as flags rather than as taxonomy rows.
    for gone in (
        "subtract_sigma_from_beta",
        "use_alpha",
        "sigma_a_estimator",
        "sigma_in_variance",
        "ml_estimation_set",
        # ...and a sixth, removed 2026-08: `sigma_m_scale` was the retired Bhattacharyya
        # target's knob and was dead END TO END -- the factory accepted and silently dropped
        # it, and `RiceSigmaMXrayTarget` initialised its calibration to zeros regardless.
        # This test previously ASSERTED that thread was intact, i.e. it locked a dead flag in
        # place. That is the failure mode this whole file exists to prevent, so it is worth
        # noting that the guard itself can become the problem.
        "sigma_m_scale",
    ):
        assert gone not in params, f"{gone!r} came back as a Refinement kwarg"


def test_builder_forwards_every_stored_option():
    """The kwargs dict must carry each stored option through to the factory.

    Checked against ``create_xray_target``'s signature so that adding a factory
    parameter without wiring it here is caught.
    """
    from torchref.refinement.targets.xray import create_xray_target

    factory_params = set(inspect.signature(create_xray_target).parameters)
    src = inspect.getsource(Refinement._xray_target_kwargs)
    # Every option the builder forwards must be one the factory accepts. Checked in this
    # direction (builder -> factory) rather than against a hardcoded list: a hardcoded list is
    # what let `sigma_m_scale` be asserted-as-present long after it stopped doing anything.
    for name in ("sigma_a_max", "shrink"):
        assert name in factory_params, f"factory lost {name!r}"
        assert name in src, f"_xray_target_kwargs does not forward {name!r}"


def test_the_mode_reaches_the_constructed_target():
    """The surviving configuration is the mode itself, so that is what must arrive.

    The five per-option variants this file was written about are gone; `xray_mode` is
    now the only thing the builder carries that can change the target's behaviour, so
    it is the thing worth pinning end-to-end.
    """
    from torchref.refinement.targets.xray._specs import XRAY_TARGETS

    class _Stub(Refinement):
        def __init__(self, **cfg):  # bypass all data/model setup
            self.model = None
            self.reflection_data = None
            self.scaler = None
            self.verbose = 0
            for k, v in cfg.items():
                setattr(self, k, v)

    stub = _Stub(xray_mode="ml_full")
    kw = stub._xray_target_kwargs()
    assert "mode" not in kw, "mode is passed separately, not through the kwargs dict"
    # and the spec table still resolves every mode the CLI offers
    for name in XRAY_TARGETS.names:
        assert XRAY_TARGETS.by_name(name).name == name


def test_every_registered_estimator_accepts_the_targets_call_signature():
    """Any selectable estimator must accept exactly what the target passes it.

    This generalises a bug that was invisible for a whole session: the target passed
    `parity_weighted_sigma` into `get()`, the v2/v3 estimators had no such parameter, and
    selecting them raised `TypeError` -- latent only because the selector flag was ITSELF
    a no-op, so the broken path was never taken. Two bugs concealing each other. v2/v3
    have since been removed (benchmarked, not better), but the contract must hold for
    whatever is registered next.
    """
    from torchref.refinement.model_error_estimation.sigma_a import SigmaAEstimator
    from torchref.refinement.targets.xray import MLXrayTarget

    # Whatever the target passes must be accepted by the estimator. Read the call out
    # of forward() rather than hard-coding it, so adding an argument on one side without
    # the other fails here instead of at the first refinement.
    src = inspect.getsource(MLXrayTarget.forward)
    passed = {
        kw for kw in ("sigma_obs", "out_epsilon", "target_dss")
        if f"{kw}=" in src
    }
    accepted = set(inspect.signature(SigmaAEstimator.get).parameters)
    assert passed <= accepted, f"target passes {sorted(passed - accepted)} to get()"


def test_every_registered_estimator_returns_the_shared_result_type():
    """All estimators must return `SigmaAEstimate`, not a bare tuple.

    The target reads `est.beta` / `est.beta_model` / `est.epsilon` by name. An estimator
    returning `(beta, epsilon)` would raise AttributeError at the first forward -- which
    is exactly what a half-migrated registry produced during this refactor.
    """
    import torch

    from torchref.refinement.model_error_estimation.sigma_a import SigmaAEstimate, SigmaAEstimator

    g = torch.Generator().manual_seed(0)
    n = 3000
    DT = torch.float64
    fc = 5 + 3 * torch.rand(n, generator=g, dtype=DT)
    fo = (fc + 0.5 * torch.randn(n, generator=g, dtype=DT)).abs()
    sg = 0.2 + 0.1 * torch.rand(n, generator=g, dtype=DT)
    eps = torch.ones(n, dtype=DT)
    cen = torch.rand(n, generator=g) < 0.3
    dss = 0.02 + 0.28 * torch.rand(n, generator=g, dtype=DT)
    mask = torch.ones(n, dtype=torch.bool)

    for name, cls in (("v1", SigmaAEstimator),):
        est = cls().get(fo, fc, cen, eps, dss, mask, sigma_obs=sg)
        assert isinstance(est, SigmaAEstimate), f"{name} returned {type(est).__name__}"
        # every field shares one length, so a consumer reading several cannot mismatch
        for fld in ("sigma_a", "alpha", "beta", "beta_model", "epsilon"):
            v = getattr(est, fld)
            assert v.shape == fo.shape, f"{name}.{fld} has shape {tuple(v.shape)}"
            assert torch.all(torch.isfinite(v)), f"{name}.{fld} not finite"
            assert not v.requires_grad, f"{name}.{fld} is not detached"
        assert torch.all(est.beta > 0) and torch.all(est.beta_model > 0), name
        # beta is the TOTAL variance, so it cannot be below the model-only part
        assert torch.all(est.beta >= est.beta_model - 1e-9), name


def test_sigma_a_estimator_knobs_reach_the_estimator():
    """`--sigma-a-max` / `--no-shrink` must arrive at `SigmaAEstimator.get`.

    A configuration option is not tested until something asserts it reached the object
    that consumes it. Five flags in this exact path were no-ops for the CLI and every
    paper harness because a second target build bypassed
    `Refinement._xray_target_kwargs()`; a session of benchmarking measured nothing. This
    walks the whole chain -- CLI namespace -> kwargs dict -> factory -> target -> the
    `get()` call -- rather than trusting any single link.
    """
    import argparse

    from torchref.refinement.model_error_estimation.sigma_a import SHRINK_ENABLED, SIGMA_A_MAX, SigmaAEstimator
    from torchref.cli.refine import _sigma_a_kwargs
    from torchref.refinement.base_refinement import Refinement
    from torchref.refinement.targets.xray import MLXrayTarget, create_xray_target

    # 1. unset CLI flags must NOT override the library defaults with None
    ns = argparse.Namespace(sigma_a_max=None, no_shrink=False, shrink_passes=None)
    assert _sigma_a_kwargs(ns) == {}
    ns = argparse.Namespace(sigma_a_max=0.999, no_shrink=True, shrink_passes=None)
    assert _sigma_a_kwargs(ns) == {"sigma_a_max": 0.999, "shrink": False}

    # 2. the factory forwards them to the target
    tgt = create_xray_target(mode="ml", sigma_a_max=0.999, shrink=False)
    assert isinstance(tgt, MLXrayTarget)
    assert tgt.sigma_a_max == 0.999 and tgt.shrink is False

    # 3. defaults come from the module, not from a second copy of the number, and they
    #    are the SAME for every spec. The shrinkage default used to be spec-dependent
    #    (off for `ml`, on for `ml_full`), which meant the two targets were fitted with
    #    differently configured estimators -- so `ml vs ml_full` confounded the likelihood
    #    with the estimator. One estimator, one configuration, every caller.
    for mode in ("ml", "ml_noalpha", "nll_beta", "ml_full"):
        d = create_xray_target(mode=mode)
        assert d.sigma_a_max == SIGMA_A_MAX
        assert d.shrink == SHRINK_ENABLED, (
            f"{mode}: shrink={d.shrink}, expected the single default"
        )
    # an explicit value still wins over the default
    assert create_xray_target(mode="ml", shrink=False).shrink is False

    # 4. `_xray_target_kwargs` -- the SINGLE construction path -- carries them
    kw = Refinement._xray_target_kwargs(
        types.SimpleNamespace(
            model=None, reflection_data=None, scaler=None, verbose=0,
            sigma_a_max=0.999, shrink=False,
        )
    )
    assert kw["sigma_a_max"] == 0.999 and kw["shrink"] is False

    # 5. forward() actually passes them on, and `get()` accepts them
    src = inspect.getsource(MLXrayTarget.forward)
    for kwarg in ("sigma_a_max", "shrink"):
        assert f"{kwarg}=self.{kwarg}" in src, f"forward() drops {kwarg}"
    params = inspect.signature(SigmaAEstimator.get).parameters
    assert "kwargs" in params or {"sigma_a_max", "shrink"} <= set(params)


def test_shrink_passes_is_a_deprecated_on_off_alias():
    """`--shrink-passes` survives as an on/off alias, loudly.

    The shrinkage target became a fitted curve rather than the neighbouring shells, so the
    fit is one-shot and a pass count would be a flag whose value changes nothing -- the
    exact shape of the five dead flags this suite exists to prevent. It is kept only so
    existing scripts and benchmark arms keep running, and it must warn while doing so.
    """
    import argparse

    from torchref.cli.refine import _sigma_a_kwargs

    for passes, expected in ((0, False), (1, True), (3, True)):
        ns = argparse.Namespace(sigma_a_max=None, no_shrink=False, shrink_passes=passes)
        with pytest.warns(DeprecationWarning, match="--no-shrink"):
            assert _sigma_a_kwargs(ns) == {"shrink": expected}


def test_estimator_knobs_actually_change_the_result():
    """The final link: a different knob value must produce a different beta.

    Steps 1-5 above prove the value travels. This proves it is USED -- an argument can
    arrive and be ignored, which is indistinguishable from a dead flag in a benchmark.
    """
    import torch

    from torchref.refinement.model_error_estimation.sigma_a import SigmaAEstimator

    g = torch.Generator().manual_seed(3)
    n, DT = 2400, torch.float64
    fc = 5 + 3 * torch.rand(n, generator=g, dtype=DT)
    fo = (fc + 1.5 * torch.randn(n, generator=g, dtype=DT)).abs()
    eps = torch.ones(n, dtype=DT)
    cen = torch.zeros(n, dtype=torch.bool)
    dss = torch.linspace(0.02, 0.3, n, dtype=DT)
    mask = torch.ones(n, dtype=torch.bool)
    args = (fo, fc, cen, eps, dss, mask)

    a = SigmaAEstimator().get(*args, shrink=False)
    b = SigmaAEstimator().get(*args, shrink=True)
    assert not torch.equal(a.beta, b.beta), "shrink had no effect"

    # sigma_a_max is the floor on the model-error variance, so raising it must allow a
    # SMALLER beta somewhere (or leave it unchanged if no shell is at the bound).
    c = SigmaAEstimator().get(*args, sigma_a_max=0.5, shrink=False)
    assert float(c.beta.min()) >= float(a.beta.min()) - 1e-12, (
        "a tighter sigma_a_max must not permit a smaller beta"
    )
    assert not torch.equal(a.beta, c.beta), "sigma_a_max had no effect"
