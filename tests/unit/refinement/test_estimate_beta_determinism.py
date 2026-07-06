"""Determinism / device-parity / float32-stability of ``estimate_beta``.

Regression for the GPU refinement blow-up: ``estimate_beta`` used a CUDA
atomic ``scatter_add`` for its per-shell moments and a non-stable ``argsort``
for binning, and computed ``wi = A*B - C**2`` (a catastrophic cancellation as
the shell correlation rho -> 1). In float32 that made beta non-deterministic
across GPU processes (bins snapping to the 1.0 floor at random), which froze a
bad beta into ``refine_lbfgs`` and collapsed the per-bin scale.

The fix: stable ``argsort``, an atomic-free contiguous ``segment_reduce``, and
sum-of-squares/centered reformulations of ``wi`` and ``OMEGA``. These tests pin
that beta is (a) bit-identical across repeated GPU calls, (b) matches CPU, and
(c) matches a float64 reference (i.e. the float32 path is accurate) — using an
input engineered to be the hard case: tied ``d_star_sq`` and high-correlation
shells.
"""

import pytest
import torch

from torchref.base.targets.xray_ml_sigmaa import estimate_beta


def _hard_inputs(dtype=torch.float32):
    """Free-set inputs that trigger the failure mode: many tied d_star_sq
    (so binning is argsort-tie-sensitive) and near-perfectly-correlated shells
    (so wi = A*B - C^2 catastrophically cancels)."""
    g = torch.Generator().manual_seed(1234)
    n = 2000
    # heavily tied resolutions: only 40 distinct d_star_sq values over n refl
    dss_levels = torch.linspace(0.02, 0.25, 40)
    idx = torch.randint(0, 40, (n,), generator=g)
    d_star_sq = dss_levels[idx]
    # amplitudes: larger at low resolution
    base = 200.0 * torch.exp(-8.0 * d_star_sq) + 1.0
    F_obs = base * (0.5 + torch.rand(n, generator=g))
    # F_calc almost perfectly correlated with F_obs -> rho -> 1 (cancellation)
    F_calc = F_obs * 0.95 + 0.01 * base * torch.randn(n, generator=g)
    F_calc = F_calc.abs()
    centric = torch.rand(n, generator=g) < 0.1
    epsilon = torch.ones(n)
    free_mask = torch.ones(n, dtype=torch.bool)
    to = lambda t: t.to(dtype) if t.is_floating_point() else t
    return (
        to(F_obs), to(F_calc), centric, to(epsilon), to(d_star_sq), free_mask,
    )


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestEstimateBetaGPUDeterminism:
    def test_gpu_bit_identical_across_calls(self):
        """Same input -> identical beta on every GPU call (was: 15x spread)."""
        args = [t.cuda() for t in _hard_inputs()]
        betas = [estimate_beta(*args)[1].detach().cpu() for _ in range(8)]
        for b in betas[1:]:
            assert torch.equal(b, betas[0]), "estimate_beta non-deterministic on GPU"

    def test_gpu_matches_cpu(self):
        """GPU beta matches CPU beta to float32 rounding (was: totally different)."""
        cpu_args = _hard_inputs()
        gpu_args = [t.cuda() for t in cpu_args]
        b_cpu = estimate_beta(*cpu_args)[1]
        b_gpu = estimate_beta(*gpu_args)[1].cpu()
        torch.testing.assert_close(b_gpu, b_cpu, rtol=1e-4, atol=1e-4)
        # no shell should be spuriously floored while CPU has a finite value
        assert (b_gpu.min() > 1.0 + 1e-6) == (b_cpu.min() > 1.0 + 1e-6)


@pytest.mark.unit
class TestEstimateBetaFloat32Accuracy:
    def test_float32_matches_float64(self):
        """The float32 path (stable wi/OMEGA) tracks a float64 reference — i.e.
        the reformulation is accurate, not just deterministic."""
        args32 = _hard_inputs(torch.float32)
        args64 = _hard_inputs(torch.float64)
        b32 = estimate_beta(*args32)[1].double()
        b64 = estimate_beta(*args64)[1]
        torch.testing.assert_close(b32, b64, rtol=2e-3, atol=1e-2)

    def test_beta_non_negative_and_finite(self):
        _, bbin, _ = estimate_beta(*_hard_inputs())
        assert torch.isfinite(bbin).all()
        assert (bbin > 0).all()
