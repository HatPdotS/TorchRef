"""The rotation search takes its working precision and its device from config.

Two properties, both easy to lose silently:

* **Precision is `torchref.config`'s, not an argument's.** The expansion used to
  be handed ``compute_dtype=torch.complex64`` by one call site, which made the
  fused CPU Legendre kernel reachable only through that argument -- its
  ``BackendTable`` row gates on ``dtypes=(torch.float32,)``, so dropping the
  argument silently routed to the portable path. Now the default *is* float32,
  so the gate matches by construction and flipping the config flips the engine.
* **The device is resolved from both inputs**, not read off whichever one the
  code happens to touch first. Model and data on different devices used to
  either cross-device or throw depending on which line ran.
"""

import pytest
import torch

from torchref.config import dtypes

pytestmark = pytest.mark.unit


def _tiny_reflections(n=4000, seed=0):
    """A P1 shell of reflections wide enough to bin into 20 Wilson shells."""
    g = torch.Generator().manual_seed(seed)
    s_mag = 0.07 + 0.18 * torch.rand(n, generator=g, dtype=torch.float64)
    theta = torch.acos(2 * torch.rand(n, generator=g, dtype=torch.float64) - 1)
    phi = 6.283185307179586 * torch.rand(n, generator=g, dtype=torch.float64)
    s_vec = torch.stack(
        [s_mag * torch.sin(theta) * torch.cos(phi),
         s_mag * torch.sin(theta) * torch.sin(phi),
         s_mag * torch.cos(theta)], dim=-1,
    )
    F = torch.randn(n, generator=g, dtype=torch.float64).abs() + 0.1
    centric = torch.zeros(n, dtype=torch.bool)
    return s_vec, F, centric


@pytest.mark.parametrize("float_dtype,want_complex", [
    (torch.float32, torch.complex64),
    (torch.float64, torch.complex128),
])
def test_expansion_follows_the_configured_float_dtype(float_dtype, want_complex):
    from torchref.experimental.alignment.frf.data_mr import bessel_sh_expand

    s_vec, F, _ = _tiny_reflections()
    original = dtypes.float, dtypes.complex
    try:
        dtypes.float = float_dtype
        dtypes.complex = want_complex
        # `s_vec` stays float64 on purpose -- the clustering keys need that
        # resolution -- so this also pins that the OUTPUT dtype is not inherited
        # from the input.
        out = bessel_sh_expand(s_vec, F, L=12, bessel_h_scale=40.0)
    finally:
        dtypes.float, dtypes.complex = original
    assert out.coeffs.dtype == want_complex, (
        f"expansion returned {out.coeffs.dtype} with dtypes.float={float_dtype}"
    )


def test_the_back_half_accumulates_one_step_wider():
    """The tail is deliberately wider than the expansion, and must stay so.

    Narrowing it looks like free memory -- complex128 buffers holding
    complex64-accurate content -- and it is not. The radial sum and the Wigner
    contraction are both oscillatory, so they cancel, and single-precision
    accumulation was measured to move scores by 1e-4 to 1.4e-3 relative and
    leave only 1 of 500 candidate slots holding the same orientation. The
    expansion's own working precision is config's; this accumulation is not.
    """
    from torchref.experimental.alignment.frf.data_mr import (
        bessel_sh_expand, cross_correlate_xi,
    )
    from torchref.experimental.alignment.frf.sitelist_ang import (
        build_dense_map_per_beta,
    )
    from torchref.experimental.alignment.frf.wigner_d import (
        wigner_contraction_per_beta,
    )

    s_vec, F, _ = _tiny_reflections()
    original = dtypes.float, dtypes.complex
    try:
        dtypes.float, dtypes.complex = torch.float32, torch.complex64
        c = bessel_sh_expand(s_vec, F, L=12, bessel_h_scale=40.0)
    finally:
        dtypes.float, dtypes.complex = original

    assert c.coeffs.dtype == torch.complex64, "expansion should be at config dtype"
    xi = cross_correlate_xi(c, c)
    assert xi.dtype == torch.complex128, (
        f"the radial accumulation narrowed to {xi.dtype}; see the docstring on "
        f"cross_correlate_xi for what that costs"
    )
    # Everything downstream follows xi rather than re-deciding.
    betas = torch.linspace(0.0, 3.0, 5, dtype=torch.float64)
    S = wigner_contraction_per_beta(xi, betas)
    assert S.dtype == xi.dtype, f"Wigner did not follow xi: {S.dtype}"
    M = build_dense_map_per_beta(xi, betas, fft_size=48)
    assert M.dtype == xi.dtype, f"FFT did not follow xi: {M.dtype}"


def test_wigner_blocks_carry_no_float64_to_the_device():
    """The eigendecomposition is precision-critical but belongs on the host.

    What reaches the device is the ``d^l`` blocks, bounded in [-1, 1], at the
    working precision -- so an accelerator without float64 is not excluded.
    """
    from torchref.experimental.alignment.frf import wigner_d

    wigner_d.clear_wigner_d_cache()
    betas = torch.linspace(0.0, 3.0, 4, dtype=torch.float64)
    blocks = wigner_d._wigner_d_blocks(
        8, betas, torch.device("cpu"), torch.float32,
    )
    assert all(b.dtype == torch.float32 for b in blocks)
    # d^l(0) = I, the cheapest correctness check on the blocks themselves.
    for l, b in enumerate(blocks, start=1):
        eye = torch.eye(2 * l + 1, dtype=torch.float32)
        assert torch.allclose(b[0], eye, atol=1e-5), f"d^{l}(0) is not I"


def test_anisotropy_fit_runs_on_the_host_in_double():
    """It is 7 parameters over ~1e4 reflections; precision there is worth more
    than locality, and keeping it on the host removes a float64 requirement."""
    from alignment_lab.lab.benchmark import load_case
    from torchref.experimental.alignment.rotation_search import (
        ANISO_FIT_WINDOW_A, fit_anisotropy,
    )

    _, data = load_case("1DAW")[:2]
    d_max, d_min = ANISO_FIT_WINDOW_A
    U = fit_anisotropy(data, d_min=d_min, d_max=d_max)
    assert U.device.type == "cpu"
    assert U.dtype == torch.float64
    assert U.shape == (3, 3)
    assert torch.allclose(U, U.T, atol=1e-12), "U must be symmetric"


def test_device_is_resolved_from_both_inputs():
    """Reading one input's device is what let model and data disagree."""
    from alignment_lab.lab.benchmark import load_case
    from torchref.utils import resolve_device

    model, data = load_case("1DAW")[:2]
    # Data-first precedence, per the convention in torchref/maps/map.py.
    resolved = resolve_device(data, model)
    assert resolved == data.hkl.device or resolved.type == data.hkl.device.type
    assert model.xyz().device.type == resolved.type
