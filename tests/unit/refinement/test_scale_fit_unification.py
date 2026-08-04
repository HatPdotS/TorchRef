"""There is exactly one scale fit, and its objective never centres on ``alpha``.

This file exists because there were three. Until 2026-08-04:

* ``LBFGSRefinement.refine_scaler`` optimised the scaler against the **body** x-ray target
  (guarded by a ``use_lossstate_scaler`` flag no caller ever set to False),
* ``Refinement.get_scales`` used ``self.scale_target``, a different objective,
* ``_refine_everything_lbfgs_single_cycle`` called ``scaler.refine_lbfgs()`` bare, silently
  discarding ``self.scale_target``.

The body target ``ml`` centres its likelihood on ``alpha*|F_calc|`` with ``alpha <= 1``, and
``alpha`` is degenerate with the very scale being fitted -- so minimising it over the scale
made the scale absorb ``1/alpha``. Every reported R-factor, computed from ``k*|F_calc|``,
inflated by the same factor, and because sigma_A was re-estimated at the drifted scale it
compounded: median reported R-work rose +0.093 (``ml``) and +0.130 (``ml_full``) from cycle 3
to cycle 10 over 756 structures, while the two non-alpha-centred rows *fell* by 0.014. The
refined models were fine -- the external scorers used the other path -- so nothing failed
loudly. On 1A0F the run log said 0.3418/0.3719 where a fresh score of the same file gave
0.2204/0.2554.

The lesson these tests encode: **an in-run reported number is not trustworthy until something
asserts it agrees with an independent measurement of the same state.** Asserting the fit
converges, or that R is finite and in (0,1), would not have caught any of this.
"""

import ast
import inspect
import textwrap

import pytest
import torch

from torchref.refinement.base_refinement import Refinement
from torchref.refinement.lbfgs_refinement import LBFGSRefinement
from torchref.refinement.targets.xray._specs import XRAY_TARGETS
from torchref.refinement.targets.xray.sigma_a import SigmaAXrayTarget
from torchref.scaling.scaler_base import DEFAULT_SCALE_TARGET, SCALE_TARGETS

# Collection-time guard: the scale targets are x-ray registry rows, so a rename in the
# registry must fail here rather than at the first scale fit of a production run.
_MISSING = [n for n in SCALE_TARGETS if n not in XRAY_TARGETS.names]
if _MISSING:
    raise RuntimeError(
        f"SCALE_TARGETS names absent from XRAY_TARGETS: {_MISSING}. "
        f"Available rows: {XRAY_TARGETS.names}"
    )


def _assert_refine_lbfgs_rejects(scale_target):
    """``ScalerBase.refine_lbfgs`` must refuse ``scale_target`` before doing any work.

    Validation lives at the point of use rather than in a helper, so exercise it there.
    ``ScalerBase()`` needs no data for this: the check precedes every data access.
    """
    from torchref.scaling.scaler_base import ScalerBase

    with pytest.raises(ValueError, match="scale_target must be one of"):
        ScalerBase().refine_lbfgs(torch.zeros(1, dtype=torch.complex64),
                                  scale_target=scale_target)


# =============================================================================
# The constraint: no scale-fit objective may centre on alpha
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize("name", SCALE_TARGETS)
def test_scale_target_does_not_centre_on_alpha(name):
    """The whole bug in one assertion.

    A sigma_A row centres on ``alpha*|F_calc|`` by overriding ``_mean``; the base class
    default is ``|F_calc|``. Any row selectable as a scale target must therefore leave
    ``_mean`` alone. Non-sigma_A rows (``nll``) have no ``alpha`` to begin with.
    """
    cls = XRAY_TARGETS.by_name(name).target_cls
    if not issubclass(cls, SigmaAXrayTarget):
        return
    assert cls._mean is SigmaAXrayTarget._mean, (
        f"{name} overrides _mean, so it centres on something other than |F_calc| -- "
        f"almost certainly alpha, which is degenerate with the scale being fitted"
    )


@pytest.mark.unit
@pytest.mark.parametrize("name", ["ml", "ml_full"])
def test_alpha_centred_rows_are_not_selectable_as_scale_targets(name):
    """``ml`` and ``ml_full`` are the two alpha-centred rows and must be rejected.

    Pinned by name as well as by the property above: this is the specific pair that shipped
    a +0.09 R-work drift, and a future row is caught by
    :func:`test_scale_target_does_not_centre_on_alpha`.
    """
    assert name in XRAY_TARGETS.names, "guard is stale -- row renamed or removed"
    assert name not in SCALE_TARGETS
    _assert_refine_lbfgs_rejects(name)


@pytest.mark.unit
def test_alpha_centred_rows_really_do_centre_on_alpha():
    """The negative control for the test above.

    If ``ml`` stopped overriding ``_mean``, ``test_scale_target_does_not_centre_on_alpha``
    would pass vacuously for every row and this file would assert nothing.
    """
    ml = XRAY_TARGETS.by_name("ml").target_cls
    assert ml._mean is not SigmaAXrayTarget._mean, (
        "ml no longer overrides _mean; the alpha-centring guard has become vacuous"
    )
    assert "_alpha_centred" in inspect.getsource(ml._mean)


# =============================================================================
# The name surface
# =============================================================================


@pytest.mark.unit
def test_default_scale_target_is_selectable():
    assert DEFAULT_SCALE_TARGET in SCALE_TARGETS


@pytest.mark.unit
def test_unknown_scale_target_rejected():
    _assert_refine_lbfgs_rejects("no_such_objective")


@pytest.mark.unit
def test_cli_offers_exactly_the_selectable_objectives():
    """``--scale-target``'s choices are the table, with nothing extra and nothing missing.

    Built from ``SCALE_TARGETS`` rather than repeated, so adding a row to the table cannot
    leave the CLI rejecting it -- and an alpha-centred row cannot appear here.
    """
    action = next(
        a for a in _refine_parser()._actions if "--scale-target" in (a.option_strings or [])
    )
    assert tuple(action.choices) == tuple(SCALE_TARGETS)
    assert action.default == DEFAULT_SCALE_TARGET


def _refine_parser():
    """``refine.py``'s parser, captured without invoking a refinement.

    Same trick as ``paper/check_submitter_flags.py``: intercept ``parse_args`` rather than
    refactoring the CLI to expose its parser.
    """
    import argparse as _argparse

    import torchref.cli.refine as refine_cli

    class _Captured(Exception):
        def __init__(self, parser):
            self.parser = parser

    def grab(self, *a, **kw):
        raise _Captured(self)

    original = _argparse.ArgumentParser.parse_args
    _argparse.ArgumentParser.parse_args = grab
    try:
        refine_cli.main()
    except _Captured as captured:
        return captured.parser
    finally:
        _argparse.ArgumentParser.parse_args = original
    raise RuntimeError("refine.main() returned without calling parse_args")


# =============================================================================
# One implementation, and it is not the body target
# =============================================================================


@pytest.mark.unit
def test_only_one_scale_fit_entry_point():
    """``refine_scaler`` lives on the base and is not overridden by the LBFGS driver.

    Two implementations is what let the objectives diverge; the override is where the
    alpha-centred one lived.
    """
    assert "refine_scaler" not in vars(LBFGSRefinement), (
        "LBFGSRefinement re-defines refine_scaler; the scale fit must have one "
        "implementation, on Refinement"
    )
    assert LBFGSRefinement.refine_scaler is Refinement.refine_scaler


@pytest.mark.unit
def test_refine_scaler_does_not_minimise_the_body_loss():
    """The regression guard. The body loss carries the alpha-centred mean."""
    src = inspect.getsource(Refinement.refine_scaler)
    for forbidden in ("complete_loss_state", "xray_target_work", "loss_state"):
        assert forbidden not in src, (
            f"refine_scaler references {forbidden!r}: fitting the scale against the body "
            f"loss is the 2026-08 bug, not a refactor of it"
        )
    assert "scale_target" in src, "the scale fit must read self.scale_target"


@pytest.mark.unit
def test_get_scales_is_a_cold_start_over_the_same_fit():
    """``get_scales`` must add ``initialize()`` and then delegate, not fit independently."""
    src = inspect.getsource(Refinement.get_scales)
    assert "initialize()" in src, "get_scales is the cold-start entry point"
    assert "refine_scaler()" in src, (
        "get_scales must delegate to the one scale-fit implementation rather than "
        "calling refine_lbfgs itself"
    )
    assert "refine_lbfgs" not in src


@pytest.mark.unit
def test_every_driver_routes_through_refine_scaler():
    """No driver may call ``scaler.refine_lbfgs`` directly.

    ``_refine_everything_lbfgs_single_cycle`` did, without ``scale_target``, so
    ``--scale-target`` was silently inert under ``--mode everything``.

    Matched on the AST rather than the source text, so the prose documenting the removal
    does not satisfy the test that guards it.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(LBFGSRefinement)))
    direct = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "refine_lbfgs"
    ]
    assert not direct, (
        f"{len(direct)} driver call(s) to refine_lbfgs at line(s) "
        f"{[n.lineno for n in direct]} bypass refine_scaler and (historically) drop "
        f"scale_target"
    )


@pytest.mark.unit
def test_refine_fits_the_scale_before_reporting_after_scaling():
    """``after_scaling`` must be recorded *after* the fit, or the label is false.

    It used to be recorded after only a solvent refresh and an outlier re-flag, with the fit
    at the end of the cycle -- so the field reported the previous cycle's scaler. That field
    is what the per-cycle benchmark figure plots.
    """
    src = inspect.getsource(LBFGSRefinement.refine)
    fit = src.index("self.refine_scaler()")
    label = src.index('cycle_dict["after_scaling"]')
    assert fit < label, (
        "refine() records after_scaling before fitting the scale; the label describes "
        "the previous cycle's state"
    )


# =============================================================================
# The invariant that was missing: in-run R == an independent score of the same state
# =============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("scale_target", SCALE_TARGETS)
def test_in_run_rfactor_matches_a_fresh_score(pdb_dir, mtz_dir, scale_target):
    """Refine with the alpha-centred body target, then re-measure the same model state.

    The reported R and a cold-started score must agree: they are the same model and the same
    data, so any gap is the scaler disagreeing with itself. Pre-fix this gap was ~0.12.
    """
    pdb, mtz = pdb_dir / "1DAW.pdb", mtz_dir / "1DAW.mtz"
    if not (pdb.exists() and mtz.exists()):
        pytest.skip("1DAW fixture not present")

    ref = LBFGSRefinement(
        data_file=str(mtz), pdb=str(pdb), verbose=0,
        target_mode="ml", scale_target=scale_target,
    )
    history = ref.refine(macro_cycles=2)

    in_work, in_free = (float(x) for x in ref.get_rfactor())
    ref.get_scales()  # independent cold-start measurement of the same coordinates
    fresh_work, fresh_free = (float(x) for x in ref.get_rfactor())

    assert in_work == pytest.approx(fresh_work, abs=5e-3), (
        f"in-run R_work {in_work:.4f} disagrees with a fresh score {fresh_work:.4f}; "
        f"the scale the loss saw is not the scale a scorer would fit"
    )
    assert in_free == pytest.approx(fresh_free, abs=5e-3)

    # And the reported trajectory must not run away from the truth as cycles accumulate.
    reported = [
        e["after_scaling"]["rwork"]
        for seg in history.values()
        if isinstance(seg, list)
        for e in seg
        if "rwork" in e.get("after_scaling", {})
    ]
    assert len(reported) == 2
    assert max(reported) < 0.6, f"reported R-work diverged: {reported}"


@pytest.mark.integration
def test_scale_fit_leaves_the_body_parameters_alone(pdb_dir, mtz_dir):
    """The scale fit must move only scaler parameters.

    ``LossState.run`` disables ``requires_grad`` outside the optimizer's intent set, so this
    holds by construction -- but the construction is what changed, so assert it.
    """
    pdb, mtz = pdb_dir / "1DAW.pdb", mtz_dir / "1DAW.mtz"
    if not (pdb.exists() and mtz.exists()):
        pytest.skip("1DAW fixture not present")

    ref = LBFGSRefinement(data_file=str(mtz), pdb=str(pdb), verbose=0, target_mode="ml")
    before = {n: p.detach().clone() for n, p in ref.model.named_parameters()}
    ref.refine_scaler()
    for name, param in ref.model.named_parameters():
        assert torch.equal(param.detach(), before[name]), (
            f"the scale fit moved model parameter {name!r}"
        )
