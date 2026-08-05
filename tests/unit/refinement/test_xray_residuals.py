"""The per-reflection seam: ``XrayTarget.residuals`` and the ``_per_refl`` hook.

``forward`` and ``residuals`` are the summed and unsummed forms of one expression, so
the invariant that matters is that they cannot encode different objectives. For most
rows that is structural -- ``forward`` *is* ``_masked_sum(_per_refl(...))`` -- but
``nll``, ``ls`` and ``ls_wunit_k1`` keep a fused Triton kernel in ``forward`` that
never materialises the per-reflection tensor, so there the two are written out
separately and only a test holds them together. That is what
:func:`test_residuals_sum_matches_forward` is for.

The rest pins the three properties ``residuals`` promises that ``forward`` does not:
full size, masks not applied, non-finite values not substituted.
"""

import pytest
import torch

from torchref.base.targets.xray_likelihoods import _masked_sum
from torchref.refinement.targets.xray._specs import XRAY_TARGETS
from torchref.refinement.targets.xray.factory import create_xray_target

#: Every selectable row. ``rice`` is deliberately absent from the table (it is private),
#: and is covered through :class:`RiceXrayTarget` directly below.
ALL_MODES = list(XRAY_TARGETS.names)


@pytest.fixture(scope="module")
def refinement(pdb_dir, mtz_dir):
    """A scaled 1DAW refinement. Module-scoped: the scale fit is the expensive part and
    nothing here mutates the model."""
    pdb = pdb_dir / "1DAW.pdb"
    mtz = mtz_dir / "1DAW.mtz"
    if not (pdb.exists() and mtz.exists()):
        pytest.skip("1DAW fixture not present")
    from torchref import LBFGSRefinement

    ref = LBFGSRefinement(
        data_file=str(mtz), pdb=str(pdb), target_mode="ml", verbose=0
    )
    ref.get_scales()
    return ref


def _target(refinement, mode, use_set="work"):
    return create_xray_target(
        data=refinement.reflection_data,
        model=refinement.model,
        scaler=refinement.scaler,
        mode=mode,
        use_set=use_set,
    )


# =====================================================================
# The seam itself
# =====================================================================


@pytest.mark.integration
@pytest.mark.parametrize("mode", ALL_MODES)
def test_residuals_are_full_size(refinement, mode):
    """One entry per reflection in ``data.hkl``, in storage order.

    This is what makes the array addressable from outside the target -- comparable
    against masks, resolution or the work/free split without an index map.
    """
    t = _target(refinement, mode)
    with torch.no_grad():
        res = t.residuals()
    assert res.shape == (len(refinement.reflection_data.hkl),)


@pytest.mark.integration
@pytest.mark.parametrize("use_set", ["work", "free"])
@pytest.mark.parametrize("mode", ALL_MODES)
def test_residuals_sum_matches_forward(refinement, mode, use_set):
    """Summing the residuals over a target's own subset reproduces its ``forward``.

    The one test standing between the eager ``_per_refl`` twins and the fused Triton
    ``forward`` kernels of ``nll`` / ``ls`` / ``ls_wunit_k1``: nothing else would notice
    the two drifting into different objectives.

    ``ls_wunit_k1`` on the free set is the documented exception -- see
    :func:`test_ls_wunit_k1_forward_refits_its_scale_on_the_scored_set`.
    """
    if mode == "ls_wunit_k1" and use_set == "free":
        pytest.skip("ls_wunit_k1 free-set forward refits its own scale; see its own test")

    t = _target(refinement, mode, use_set=use_set)
    sub = t._subset()
    with torch.no_grad():
        fwd = t.forward()
        summed = _masked_sum(t.residuals().index_select(0, sub.indices))

    assert torch.isfinite(fwd)
    torch.testing.assert_close(summed, fwd, rtol=1e-5, atol=1e-5)


@pytest.mark.integration
@pytest.mark.parametrize("mode", ALL_MODES)
def test_masked_out_reflections_still_get_a_residual(refinement, mode):
    """The only mask property ``residuals`` promises: everything gets a value.

    A masked-out reflection must still be scored -- not absent, zero-filled or NaN --
    because the point of a full-size residual is to be able to ask *why* a reflection was
    excluded, which is impossible if exclusion also removes the evidence.

    Note this is *not* "the residuals are mask-invariant": see
    :func:`test_fitted_nuisance_parameters_do_depend_on_the_masks`.
    """
    data = refinement.reflection_data
    t = _target(refinement, mode)

    keep = torch.ones(len(data.hkl), dtype=torch.bool, device=data.device)
    dropped = slice(0, max(1, len(keep) // 10))
    keep[dropped] = False
    data.masks["probe"] = keep
    try:
        with torch.no_grad():
            res = t.residuals()
        assert res.shape == (len(data.hkl),)
        assert torch.isfinite(res[dropped]).all(), "masked-out reflections scored as NaN"
        assert (res[dropped] != 0).any(), "masked-out reflections were zero-filled"
    finally:
        del data.masks["probe"]


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["nll", "ls"])
def test_residual_of_a_reflection_is_independent_of_its_own_mask_bit(refinement, mode):
    """For the rows with no fitted nuisance parameter of their own, masking is invisible.

    ``nll`` and ``ls`` read only the frozen scaler, so their residuals are a pure function
    of ``(F_obs, sigma, F_calc)`` per reflection. Restricted to these two rows on purpose --
    the general statement is false and the next test says why.
    """
    data = refinement.reflection_data
    t = _target(refinement, mode)
    with torch.no_grad():
        before = t.residuals().clone()

    keep = torch.ones(len(data.hkl), dtype=torch.bool, device=data.device)
    keep[: max(1, len(keep) // 10)] = False
    data.masks["probe"] = keep
    try:
        with torch.no_grad():
            after = t.residuals()
        torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)
    finally:
        del data.masks["probe"]


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["ml", "ls_wunit_k1"])
def test_fitted_nuisance_parameters_do_depend_on_the_masks(refinement, mode):
    """Rows that FIT something move when the trusted set changes -- by design, and a
    constraint on any criterion built on top of these residuals.

    ``ml`` estimates ``beta`` on the free set; ``ls_wunit_k1`` fits its closed-form ``c``
    on the work set. Both sets are intersected with ``masks()``, so rejecting reflections
    shifts the nuisance parameter and therefore *every* residual, not only the rejected
    ones. That is correct -- one does not fit a scale against data one has judged
    untrustworthy -- but it means rejection feeds back into the statistic that drives it.
    The loop is convergent rather than one-way (unlike a latch), and characterising it is
    a Phase-2 question, not something to assert away here.
    """
    data = refinement.reflection_data
    t = _target(refinement, mode)
    with torch.no_grad():
        before = t.residuals().clone()

    keep = torch.ones(len(data.hkl), dtype=torch.bool, device=data.device)
    keep[: max(1, len(keep) // 4)] = False
    data.masks["probe"] = keep
    try:
        t.maintenance()  # drop the cached estimate so it refits on the changed free set
        with torch.no_grad():
            after = t.residuals()
        assert not torch.allclose(after, before, rtol=1e-4, atol=1e-4), (
            f"{mode}'s residuals did not move when the trusted set changed -- either the "
            "nuisance fit stopped seeing the masks, or a cache is not being invalidated"
        )
    finally:
        del data.masks["probe"]
        t.maintenance()


@pytest.mark.integration
def test_residuals_do_not_substitute_non_finite_values(refinement):
    """``forward`` replaces a NaN with ``1e6`` so it cannot poison a gradient;
    ``residuals`` must not, because there a NaN is the finding.
    """
    from torchref.refinement.targets.xray.ml_noalpha import MLNoAlphaXrayTarget

    class _NaNRow(MLNoAlphaXrayTarget):
        def _per_refl(self, ctx):
            out = super()._per_refl(ctx)
            out = out.clone()
            out[0] = float("nan")
            return out

    t = _NaNRow(
        data=refinement.reflection_data,
        model=refinement.model,
        scaler=refinement.scaler,
        use_set="work",
    )
    with torch.no_grad():
        assert torch.isnan(t.residuals()).any(), "residuals swallowed a NaN"
        assert torch.isfinite(t.forward()), "forward let a NaN through"


@pytest.mark.integration
def test_residuals_are_differentiable(refinement):
    """Left differentiable on purpose: it is an ordinary target expression, and a caller
    wanting a diagnostic wraps it in ``no_grad`` itself."""
    t = _target(refinement, "ml")
    xyz = refinement.model.xyz.refinable_params  # the leaf; `.xyz` is a MixedTensor
    res = t.residuals()
    assert res.requires_grad
    grads = torch.autograd.grad(res.sum(), xyz, allow_unused=True)[0]
    assert grads is not None and torch.isfinite(grads).all()


# =====================================================================
# Row-specific hazards
# =====================================================================


@pytest.mark.integration
def test_ls_wunit_k1_residuals_use_the_work_fit_scale(refinement):
    """``ls_wunit_k1`` fits its own closed-form scale, and ``_per_refl`` must take it from
    :meth:`_scaled_F_calc_full` -- the work-set fit applied to every reflection.

    Fitting it on the view being evaluated instead would, for the full-reflection view,
    absorb the free set into the scale: neither what the loss saw nor what ``get_rfactor``
    reports.
    """
    t = _target(refinement, "ls_wunit_k1", use_set="free")
    data = refinement.reflection_data
    with torch.no_grad():
        base = t.get_F_calc_scaled(data.hkl_for_sf(), recalc=False)
        c_from_residual_path = (t._scaled_F_calc_full() / base.clamp(min=1e-12)).median()
        work = data.work
        c_work = t._binwise_scale(
            work.select(base), work.F, work.select(t._get_bins_cached())
        ).squeeze()
    torch.testing.assert_close(c_from_residual_path, c_work, rtol=1e-5, atol=1e-6)


@pytest.mark.integration
def test_ls_wunit_k1_forward_refits_its_scale_on_the_scored_set(refinement):
    """Pins a PRE-EXISTING inconsistency, so a later fix is a deliberate change.

    ``UnitWeightK1XrayTarget.forward`` scales through ``_scaled_amplitudes``, which refits
    the closed-form ``c`` on whatever subset the target is bound to. On the free-set target
    that means the reported ``xray_test`` loss is computed under a scale fit to the very
    reflections being scored, while ``get_rfactor`` uses the work-set fit. ``residuals``
    follows ``get_rfactor``, so the two disagree here and only here.
    """
    t = _target(refinement, "ls_wunit_k1", use_set="free")
    sub = t._subset()
    with torch.no_grad():
        fwd = t.forward()
        summed = _masked_sum(t.residuals().index_select(0, sub.indices))
    assert not torch.isclose(summed, fwd, rtol=1e-3), (
        "ls_wunit_k1's free-set forward now agrees with the work-fit scale -- if that was "
        "intentional, delete this test and un-skip the free case in "
        "test_residuals_sum_matches_forward"
    )


@pytest.mark.integration
def test_ml_full_parity_cache_serves_both_views(refinement):
    """``ml_full`` caches its parity split per subset. A single-slot cache would make
    ``forward`` and ``residuals`` evict each other and pay a device sync every call.
    """
    t = _target(refinement, "ml_full")
    with torch.no_grad():
        t.forward()
        t.residuals()
    assert set(t._parity_cache) == {
        ("work", refinement.reflection_data.work.n),
        ("all", len(refinement.reflection_data.hkl)),
    }


@pytest.mark.integration
def test_private_rice_row_has_residuals(refinement):
    """``rice`` is not in the taxonomy but is still constructed directly by
    ``experimental/alignment/rigid_body.py``, so it carries the seam too."""
    from torchref.refinement.targets.xray.rice import RiceXrayTarget

    t = RiceXrayTarget(
        data=refinement.reflection_data,
        model=refinement.model,
        scaler=refinement.scaler,
        use_set="work",
    )
    sub = t._subset()
    with torch.no_grad():
        fwd = t.forward()
        summed = _masked_sum(t.residuals().index_select(0, sub.indices))
    torch.testing.assert_close(summed, fwd, rtol=1e-5, atol=1e-5)


# =====================================================================
# What residuals() stands on: the "all" view
# =====================================================================


@pytest.mark.integration
def test_all_view_is_every_reflection_regardless_of_masks(refinement):
    """``data.all`` is the odd one out: not intersected with ``masks()``. That asymmetry
    is the whole reason it exists."""
    data = refinement.reflection_data
    n = len(data.hkl)
    assert data.all.n == n
    torch.testing.assert_close(
        data.all.indices, torch.arange(n, device=data.all.indices.device)
    )
    assert bool(data.all.mask.all())

    keep = torch.ones(n, dtype=torch.bool, device=data.device)
    keep[: n // 10] = False
    data.masks["probe"] = keep
    try:
        assert data.all.n == n, "'all' must not be narrowed by a mask"
        assert data.work.n < int(keep.sum()) + 1, "'work' must still be narrowed by it"
    finally:
        del data.masks["probe"]


@pytest.mark.integration
def test_all_view_select_is_value_identity(refinement):
    """``select`` on the ``all`` view is a copy, not the identity object -- but must be
    value-identical, or every full-size array silently reorders."""
    data = refinement.reflection_data
    torch.testing.assert_close(data.all.select(data.F), data.F)
    torch.testing.assert_close(data.all.hkl, data.hkl)
