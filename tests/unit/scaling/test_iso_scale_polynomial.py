"""The isotropic scale is a Chebyshev polynomial in ``sin(theta)/lambda``.

These pin the properties the parameterisation is chosen for, none of which a
per-bin step scale had:

* it is **continuous** -- neighbouring reflections cannot differ by a bin-boundary jump;
* ``n_iso_coeff=1`` is exactly one global constant, which is what rigid-body
  ``ls_wunit_k1`` relies on to avoid double-scaling against its own closed-form scale;
* low orders span the classical physical forms, so ``k*exp(-B s^2/4)`` is representable;
* the per-reflection clamp still bounds a polynomial that is unbounded by construction.

``F_calc`` is synthesised rather than computed from the model: these are unit tests of the
scale parameterisation, and a real ``F_calc`` needs the whole FFT stack.
"""

import pytest
import torch


def _fake_fcalc(scaler, damping: float = 30.0):
    """Complex amplitudes on the same scale as ``F_obs``, with a known B-like trend."""
    fobs = scaler._data.get_corrected_data()[0]
    amp = fobs * torch.exp(-damping * scaler._s_half_sq) / 2.5
    return amp.to(torch.complex64)


@pytest.fixture
def scaler(model_and_data):
    from torchref.scaling.scaler import Scaler

    return Scaler(
        model_and_data["model"], model_and_data["data"], nbins=10, verbose=0
    )


@pytest.mark.unit
def test_design_matrix_shape_and_chebyshev_range(scaler):
    design = scaler._iso_design
    assert design.shape == (scaler.s.shape[0], scaler.n_iso_coeff)
    # T_0 == 1 everywhere; every T_k is bounded by 1 on [-1, 1].
    assert torch.allclose(design[:, 0], torch.ones_like(design[:, 0]))
    assert design.abs().max() <= 1.0 + 1e-5


@pytest.mark.unit
def test_single_coefficient_is_one_global_scale(model_and_data):
    """``n_iso_coeff=1`` must give the SAME scale to every reflection."""
    from torchref.scaling.scaler import Scaler

    sc = Scaler(
        model_and_data["model"], model_and_data["data"], nbins=10,
        n_iso_coeff=1, verbose=0,
    )
    sc.calc_initial_scale(_fake_fcalc(sc))
    assert sc.c_iso.shape == (1,)
    with torch.no_grad():
        log_k = sc.iso_log_scale()
    assert log_k.shape[0] == sc.s.shape[0]
    assert torch.allclose(log_k, log_k[0].expand_as(log_k), atol=1e-6)


@pytest.mark.unit
def test_low_orders_span_the_classical_scale_forms(scaler):
    """The basis is Chebyshev in ``x = sin(theta)/lambda``, so a polynomial of degree
    ``d`` in ``x`` needs ``d + 1`` terms and must then be representable exactly. Degree 2
    is scale-plus-overall-B, the classical form.
    """
    x = torch.sqrt(scaler._s_half_sq.clamp(min=0))
    for n_terms, target in ((2, 1.7 - 3.0 * x), (3, 1.7 - 3.0 * x**2)):
        design = scaler._iso_design[:, :n_terms]
        coeff = torch.linalg.lstsq(design, target.unsqueeze(1)).solution.squeeze(1)
        residual = (design @ coeff - target).abs().max()
        assert residual < 1e-3 * target.abs().max(), (
            f"{n_terms} terms cannot represent a degree-{n_terms - 1} polynomial "
            f"(residual {residual:.3e})"
        )


@pytest.mark.unit
def test_scale_is_continuous_in_resolution(scaler):
    """No bin-boundary jumps: reflections adjacent in ``s`` get adjacent scales."""
    scaler.calc_initial_scale(_fake_fcalc(scaler))
    with torch.no_grad():
        order = torch.argsort(scaler._s_half_sq)
        log_k = scaler.iso_log_scale()[order]
        ds = scaler._s_half_sq[order].diff()
        jumps = log_k.diff().abs()
    # Where two reflections sit at effectively the same resolution the scale must be
    # effectively the same; a step model breaks exactly this at a bin edge.
    coincident = ds < 1e-8
    assert coincident.any(), "fixture has no coincident-resolution reflections"
    assert jumps[coincident].max() < 1e-5


@pytest.mark.unit
def test_per_reflection_clamp_bounds_the_polynomial(scaler):
    """A polynomial is unbounded at the ends of its range; the clamp is what stops a
    single extreme reflection carrying an arbitrary scale."""
    scaler.calc_initial_scale(_fake_fcalc(scaler))
    with torch.no_grad():
        scaler.c_iso.fill_(50.0)
        log_k = scaler.iso_log_scale()
    assert log_k.max() <= 10.0 + 1e-6
    assert log_k.min() >= -10.0 - 1e-6


@pytest.mark.unit
def test_seed_recovers_the_closed_form_scale_curve(scaler):
    """``calc_initial_scale`` projects the per-bin closed-form ratio onto the basis, so
    the fit starts from the curve a binned model would have started from."""
    fcalc = _fake_fcalc(scaler)
    scaler.calc_initial_scale(fcalc)
    fobs = scaler._data.get_corrected_data()[0]
    amp = torch.abs(fcalc).to(fobs.dtype)
    with torch.no_grad():
        log_k = scaler.iso_log_scale()
        work = scaler._data.work.mask.to(torch.bool)
        bins = scaler.bins.to(torch.int64)
        checked = 0
        for b in range(scaler.nbins):
            m = work & (bins == b)
            if m.sum() < 50:
                continue
            closed_form = torch.log(
                ((fobs[m] * amp[m]).sum() / (amp[m] ** 2).sum()).clamp(min=1e-6)
            )
            assert abs(float(log_k[m].mean() - closed_form)) < 0.15
            checked += 1
    assert checked >= 5


@pytest.mark.unit
def test_forward_applies_the_polynomial_scale(scaler):
    """``forward`` must scale by ``exp(iso_log_scale())``, per reflection."""
    fcalc = _fake_fcalc(scaler)
    scaler.calc_initial_scale(fcalc)
    with torch.no_grad():
        scaled = scaler.forward(fcalc)
        expected = torch.exp(scaler.iso_log_scale()) * fcalc
    assert torch.allclose(torch.abs(scaled), torch.abs(expected), rtol=1e-4)


@pytest.mark.unit
def test_n_iso_coeff_survives_a_state_dict_round_trip(scaler):
    from torchref.scaling.scaler import Scaler

    scaler.calc_initial_scale(_fake_fcalc(scaler))
    state = scaler.state_dict()
    assert state["n_iso_coeff"] == scaler.n_iso_coeff

    fresh = Scaler(verbose=0)
    fresh.set_data(scaler._data.module)
    fresh.c_iso = torch.nn.Parameter(torch.zeros_like(scaler.c_iso))
    fresh.load_state_dict(state, strict=False)
    assert fresh.n_iso_coeff == scaler.n_iso_coeff
    assert torch.allclose(fresh.c_iso, scaler.c_iso)
