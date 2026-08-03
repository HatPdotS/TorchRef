"""Per-mode golden loss, gradient and R-factor. The instrument for the target refactor.

This file exists to make a *behaviour-preserving* restructuring of
``torchref/refinement/targets/xray/`` provable rather than plausible. That refactor splits
one spec-switching class into five independent target classes, consolidates five loss
modules into three primitives, and moves the sigma_A/sigma_M estimators into
``torchref/refinement/model_error_estimation/``. All of it is code motion: same operations,
same order, same dtypes.

So the assertions are **bitwise**, on ``float.hex()``. A tolerance would let a reordered
cast or a moved clamp through, which is precisely the regression class this guards.

Two things had to be got right for this to test anything at all, both of them ways a
"parity" suite can look green while proving nothing:

1. **The F_calc cache must be refreshed with a live tensor before each mode.**
   ``XrayTarget.get_rfactor`` recomputes ``F_calc`` under ``torch.no_grad()`` and
   *overwrites* ``ModelFT``'s cache with a **detached** tensor. Every forward after that --
   on the same target or a brand-new one -- then yields a loss with a ``grad_fn`` but no
   path to the model parameters, so ``backward()`` leaves every ``p.grad`` at ``None``.
   The first draft of this file recorded ``|grad| = 0`` for 13 of 14 mode x set
   combinations and passed. ``recalc=True`` does **not** recover (the stale detached entry
   survives it) and ``retain_graph=True`` does not either (the cache is detached, not
   merely freed) -- ``model.reset_cache()`` is the only recovery, and is what
   ``LossState.optimize`` calls after every optimizer step, which is why production never
   sees this. Hence :func:`_loss_grad_rfactor` resets, and asserts a gradient arrived.

2. **The scaler is seeded.** ``ScalerBase.initialize`` draws ``U_aniso`` from an unseeded
   ``torch.normal(0, 0.001, (6,))``, and ``get_scales()`` then runs LBFGS plus discrete
   outlier rejection on top. Without ``torch.manual_seed`` before the refinement is built,
   nothing here is reproducible. Verified by measuring twice in one process.

Goldens measured 2026-08-03 on 1DAW (23356 reflections, 22056 work / 1201 free),
float64/CPU, with ``ls`` already unit-weighted (see ``ls_sigma`` below).
"""

import inspect

import pytest
import torch

from torchref.refinement.targets.xray import (
    LeastSquaresXrayTarget,
    create_xray_target,
)

MODES = ["nll", "nll_beta", "ml", "ml_noalpha", "ml_full", "ls", "ls_wunit_k1"]

#: ``mode/use_set -> (loss.hex(), grad_l2.hex())``.
#:
#: ``ls_sigma`` is **not a mode**. It is the retired sigma-weighted least squares, kept
#: here as the evidence for why ``ls`` became unit-weighted: with ``w = 1/sigma**2`` the LS
#: loss is ``0.5*sum d**2/sigma**2`` and ``nll`` is the same plus
#: ``sum(log sigma + 0.5*log 2pi)`` over the *identical* ``median(sigma)*0.1`` clamp. The
#: extra term is parameter-independent, so the two rows had **bit-identical gradients** --
#: note ``ls_sigma`` and ``nll`` share a gradient hex below, in both sets -- and therefore
#: produced the same refinement trajectory while reporting different numbers.
GOLDEN = {
    "nll/work":          ("0x1.3299de01d50f5p+19", "0x1.3e618134a95b3p+16"),
    "nll_beta/work":     ("0x1.0ef35c36bcd12p+16", "0x1.8720843e33638p+9"),
    "ml/work":           ("0x1.044d67cc45869p+16", "0x1.3974eaf6f9622p+9"),
    "ml_noalpha/work":   ("0x1.07215294f29f0p+16", "0x1.c46f99e00851ep+9"),
    "ml_full/work":      ("0x1.06ef29c354f5ep+16", "0x1.36cf700200f97p+9"),
    "ls/work":           ("0x1.ab46e444d39dcp+18", "0x1.69d68fc002455p+15"),
    "ls_wunit_k1/work":  ("0x1.a9a053df7f2a0p+18", "0x1.72f0d22673fbap+15"),
    "ls_sigma/work":     ("0x1.25fed17a95fa4p+19", "0x1.3e618134a95b3p+16"),
    "nll/free":          ("0x1.c8dec8d995879p+15", "0x1.7d5f4df65840fp+14"),
    "nll_beta/free":     ("0x1.f9a3b9afd8e14p+11", "0x1.cb221138b5adap+7"),
    "ml/free":           ("0x1.e46de48525030p+11", "0x1.67e1527ad77ffp+7"),
    "ml_noalpha/free":   ("0x1.ed3b0967b2f64p+11", "0x1.c594f5ae15b38p+7"),
    "ml_full/free":      ("0x1.e82023c6b2135p+11", "0x1.5d7e9eae2f404p+7"),
    "ls/free":           ("0x1.60a4814af7b06p+15", "0x1.f304147bea63fp+13"),
    "ls_wunit_k1/free":  ("0x1.608f8336e64e8p+15", "0x1.f0cc92488e320p+13"),
    "ls_sigma/free":     ("0x1.bdc959a368eecp+15", "0x1.7d5f4df65840fp+14"),
}

#: ``(R_work, R_free)`` to 6 dp. Identical for every mode except ``ls_wunit_k1``, which is
#: the only target overriding ``_scaled_F_calc_full`` (it reports on its own work-fit
#: closed-form scale rather than the scaler's). That contrast is the point: it is the one
#: path the least-squares class split can silently break.
GOLDEN_R = {"ls_wunit_k1": (0.204327, 0.270047)}
DEFAULT_R = (0.204091, 0.269681)


@pytest.fixture(scope="module")
def bundle():
    """A seeded, scaled 1DAW refinement in float64/CPU, built once for the module.

    Not the shared ``model_and_data``/``initialized_scaler`` fixtures: those hand back a
    bare ``Model``, which has no ``forward`` and so cannot compute F_calc, and a scaler
    that has not been fit. A real ``LBFGSRefinement`` is what wires the SF engine.
    """
    from torchref.config import device as _device, dtypes as _dtypes
    from torchref.refinement.lbfgs_refinement import LBFGSRefinement

    f0, c0, d0 = _dtypes.float, _dtypes.complex, _device.current
    _dtypes.float, _dtypes.complex = torch.float64, torch.complex128
    _device.current = torch.device("cpu")
    try:
        torch.manual_seed(0)  # see the module docstring -- the scaler is stochastic
        ref = LBFGSRefinement(
            data_file="tests/files/mtz/1DAW.mtz",
            pdb="tests/files/cif/1DAW.cif",
            device=torch.device("cpu"),
            target_mode="ml",
            verbose=0,
        )
        ref.get_scales()
        yield ref
    finally:
        _dtypes.float, _dtypes.complex, _device.current = f0, c0, d0


def _build(ref, key, use_set):
    mode = key.split("/")[0]
    kw = dict(data=ref.reflection_data, model=ref.model, scaler=ref.scaler,
              use_set=use_set, verbose=0)
    if mode == "ls_sigma":
        return LeastSquaresXrayTarget(weighting="sigma", **kw)
    return create_xray_target(mode=mode, **kw)


def _loss_grad_rfactor(ref, tgt):
    """``(loss, |grad|_2, R_work, R_free)``, with the cache hazard handled.

    Order matters: ``reset_cache()`` first, ``get_rfactor()`` **last**. See the module
    docstring.
    """
    model = ref.model
    model.reset_cache()
    for p in model.parameters():
        p.grad = None
    loss = tgt.forward()
    loss.backward()
    grads = [p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]
    assert grads, (
        "no parameter received a gradient -- the F_calc cache is detached, so this "
        "measurement is vacuous rather than merely wrong"
    )
    gl2 = float(torch.linalg.vector_norm(torch.cat(grads).double()))
    rwork, rfree = tgt.get_rfactor()
    return float(loss), gl2, rwork, rfree


@pytest.mark.unit
@pytest.mark.parametrize("use_set", ["work", "free"])
@pytest.mark.parametrize("mode", MODES + ["ls_sigma"])
def test_loss_and_gradient_are_bit_identical(bundle, mode, use_set):
    key = f"{mode}/{use_set}"
    loss, gl2, _rw, _rf = _loss_grad_rfactor(bundle, _build(bundle, key, use_set))
    want_loss, want_grad = GOLDEN[key]
    assert loss.hex() == want_loss, f"{key}: loss moved to {loss!r} ({loss.hex()})"
    assert gl2.hex() == want_grad, f"{key}: |grad| moved to {gl2!r} ({gl2.hex()})"


@pytest.mark.unit
@pytest.mark.parametrize("mode", MODES)
def test_rfactor_is_unchanged(bundle, mode):
    """``get_rfactor`` per mode. Only ``ls_wunit_k1`` differs -- it is the sole target
    overriding ``_scaled_F_calc_full``, and that override is what the least-squares class
    split can break without touching any loss value."""
    _l, _g, rwork, rfree = _loss_grad_rfactor(
        bundle, _build(bundle, f"{mode}/work", "work")
    )
    want = GOLDEN_R.get(mode, DEFAULT_R)
    assert (round(rwork, 6), round(rfree, 6)) == want, f"{mode}: R moved"


@pytest.mark.unit
def test_sigma_weighted_ls_has_the_same_gradient_as_nll(bundle):
    """Why ``ls`` is unit-weighted, asserted rather than asserted-in-a-comment.

    ``0.5*sum d**2/sigma**2`` and ``0.5*sum d**2/sigma**2 + sum(log sigma + 0.5*log 2pi)``
    differ by a constant in the parameters, so their gradients agree exactly while their
    losses do not. If a future change makes these two gradients differ, either ``nll``'s
    sigma clamp or the LS sigma branch has moved, and the de-duplication argument for
    ``ls`` being unit-weighted no longer holds.
    """
    _, g_sigma, _, _ = _loss_grad_rfactor(bundle, _build(bundle, "ls_sigma/work", "work"))
    l_nll, g_nll, _, _ = _loss_grad_rfactor(bundle, _build(bundle, "nll/work", "work"))
    l_sigma, _, _, _ = _loss_grad_rfactor(
        bundle, _build(bundle, "ls_sigma/work", "work")
    )
    assert g_sigma.hex() == g_nll.hex(), "the constant-offset identity has broken"
    assert l_nll > l_sigma, "nll must exceed sigma-weighted ls by the log-sigma term"
    # and unit-weighted `ls` must NOT share that gradient -- that is the whole point
    _, g_unit, _, _ = _loss_grad_rfactor(bundle, _build(bundle, "ls/work", "work"))
    assert g_unit.hex() != g_nll.hex(), "unit-weighted ls still duplicates nll"


@pytest.mark.unit
def test_the_estimate_is_computed_once_per_maintenance_block(bundle, monkeypatch):
    """beta/alpha are refreshed once per optimizer-step block, NOT per iteration.

    LBFGS evaluates the closure many times per step (strong-Wolfe line search). Estimating
    inside that loop would both pay for a solve per evaluation and make the objective
    non-stationary during the search, so ``SigmaAEstimator.get`` returns a cache until
    ``maintenance()`` clears it and ``LossState.optimize`` calls ``maintenance()`` *after*
    the step loop. Nothing asserted this before; a refactor that moved the estimator
    construction into ``forward`` would be silently correct-but-ruinous.
    """
    import torchref.refinement.model_error_estimation.sigma_a as sig

    calls = {"n": 0}
    real = sig.estimate_beta

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(sig, "estimate_beta", counting)

    tgt = _build(bundle, "ml/work", "work")
    for _ in range(5):
        bundle.model.reset_cache()
        tgt.forward()
    assert calls["n"] == 1, f"re-estimated inside the block ({calls['n']} solves for 5 forwards)"

    tgt.maintenance()
    tgt.forward()
    assert calls["n"] == 2, "maintenance() did not invalidate the estimate"


@pytest.mark.unit
def test_the_estimator_is_constructed_once_not_per_forward(bundle):
    """The companion structural check: ``forward`` must not build a ``SigmaAEstimator``.

    A per-forward construction would give a fresh empty cache every call, so the caching
    contract above would hold *within* a forward and be worthless across one. Cheaper and
    more direct than counting constructions at runtime.
    """
    from torchref.refinement.targets.xray import SigmaAXrayTarget

    src = inspect.getsource(SigmaAXrayTarget.forward)
    assert "SigmaAEstimator(" not in src, "forward() constructs an estimator"
    assert "SigmaAEstimator(" in inspect.getsource(
        SigmaAXrayTarget.__init__
    ), "__init__ no longer constructs the estimator"
