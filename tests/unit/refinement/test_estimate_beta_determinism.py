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

from torchref.refinement.model_error_estimation.sigma_a import estimate_beta

#: Tolerance for comparing ``beta``/``sigma_A`` across dtypes or devices. See the note on
#: ``test_sigma_a_solve.GRID_STEP_RTOL``: the solve is a discrete argmin over a grid whose
#: final stage steps ``beta`` by 0.38%, and near a flat optimum two dtypes or two devices
#: can land on neighbouring candidates. Bounded by one step (measured worst case 3.8e-3),
#: against a ~14% sampling sd on the same quantity.
#:
#: This does NOT weaken what this module pins. Same-device repeatability is still asserted
#: bit-exactly (``torch.equal``), which is the property whose loss caused the GPU blow-up;
#: only the cross-dtype/cross-device comparisons carry this tolerance.
GRID_STEP_RTOL = 1e-2


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
@pytest.mark.cuda
class TestEstimateBetaGPUDeterminism:
    def test_gpu_bit_identical_across_calls(self):
        """Same input -> identical beta on every GPU call (was: 15x spread)."""
        args = [t.cuda() for t in _hard_inputs()]
        betas = [estimate_beta(*args).beta.detach().cpu() for _ in range(8)]
        for b in betas[1:]:
            assert torch.equal(b, betas[0]), "estimate_beta non-deterministic on GPU"

    def test_gpu_matches_cpu(self):
        """GPU beta matches CPU beta to the grid step (was: totally different)."""
        cpu_args = _hard_inputs()
        gpu_args = [t.cuda() for t in cpu_args]
        b_cpu = estimate_beta(*cpu_args).beta
        b_gpu = estimate_beta(*gpu_args).beta.cpu()
        torch.testing.assert_close(
            b_gpu, b_cpu, rtol=GRID_STEP_RTOL, atol=GRID_STEP_RTOL
        )
        # no shell should be spuriously floored while CPU has a finite value
        assert (b_gpu.min() > 1.0 + 1e-6) == (b_cpu.min() > 1.0 + 1e-6)


@pytest.mark.unit
class TestEstimateBetaFloat32Accuracy:
    def test_float32_matches_float64(self):
        """The float32 path (stable wi/OMEGA) tracks a float64 reference — i.e.
        the reformulation is accurate, not just deterministic."""
        args32 = _hard_inputs(torch.float32)
        args64 = _hard_inputs(torch.float64)
        b32 = estimate_beta(*args32).beta.double()
        b64 = estimate_beta(*args64).beta
        torch.testing.assert_close(b32, b64, rtol=GRID_STEP_RTOL, atol=1e-2)

    def test_beta_non_negative_and_finite(self):
        bbin = estimate_beta(*_hard_inputs()).beta
        assert torch.isfinite(bbin).all()
        assert (bbin > 0).all()


def _real_inputs(mtz_dir, pdb_dir, dtype=torch.float32):
    """1DAW observations against its own deposited model.

    The synthetic ``_hard_inputs`` fixture above does not reproduce the conditioning
    regime the fit actually meets: it was built to stress ``wi``/``OMEGA``, not the Rice
    kernel, and a synthetic scene can badly over- or under-state the cancellation. Real
    amplitudes against a real model is what settles whether the float32 path is accurate
    enough to ship.
    """
    from torchref.io.datasets.reflection_data import ReflectionData
    from torchref.model.model_ft import ModelFT
    from torchref.refinement.model_error_estimation.sigma_a import epsilon_from_hkl

    d = ReflectionData(verbose=0)
    d.load_mtz(str(mtz_dir / "1DAW.mtz"))
    m = ModelFT(verbose=0, max_res=2.0)
    m.load_pdb(str(pdb_dir / "1DAW.pdb"))
    with torch.no_grad():
        fc = d.structure_factors(m)
    cast = lambda t: t.detach().cpu().to(dtype)  # noqa: E731
    return dict(
        F_obs=cast(d.F),
        F_calc=cast(fc.abs()),
        centric=d.centric.detach().cpu(),
        epsilon=cast(epsilon_from_hkl(d.hkl, d.spacegroup)),
        d_star_sq=cast(1.0 / d.resolution**2),
        free_mask=~d.rfree_flags.detach().cpu().bool(),
        sigma_obs=cast(d.F_sigma),
    )


@pytest.mark.unit
class TestFloat32OnRealData:
    """The float32 path on a deposited structure.

    ``estimate_beta`` used to force float64 internally, which made it unrunnable on MPS
    (no float64 there at all). The cancellation-free Rice kernel is what makes float32
    accurate enough to drop that; these pin the accuracy claim on real data.
    """

    def test_float32_matches_float64_on_1daw(self, mtz_dir, pdb_dir):
        a = estimate_beta(**_real_inputs(mtz_dir, pdb_dir, torch.float32))
        b = estimate_beta(**_real_inputs(mtz_dir, pdb_dir, torch.float64))
        torch.testing.assert_close(
            a.sigma_a.double(), b.sigma_a, rtol=GRID_STEP_RTOL, atol=GRID_STEP_RTOL
        )
        torch.testing.assert_close(
            a.beta.double(), b.beta, rtol=GRID_STEP_RTOL, atol=GRID_STEP_RTOL
        )

    def test_second_moment_identity_survives_float32(self, mtz_dir, pdb_dir):
        """The identity is algebraic, so float32 should cost only float32 rounding."""
        sh = estimate_beta(**_real_inputs(mtz_dir, pdb_dir, torch.float32))
        resid = (
            sh.alpha**2 * sh.Sigma_P + sh.beta_model + sh.S2 - sh.B
        ).abs() / sh.B
        assert float(resid.max()) < 1e-5, f"identity off by {float(resid.max()):.2e}"


@pytest.mark.mps
def test_estimate_beta_runs_on_mps_and_matches_cpu(mtz_dir, pdb_dir):
    """Regression: the fit was unrunnable on MPS for two independent reasons.

    A hardcoded ``torch.float64`` (MPS has none) and ``torch.segment_reduce`` (not
    implemented for MPS). The dtype raised first, so the second blocker stayed hidden
    until the first was fixed -- hence a test that asserts it *runs*, not just that the
    numbers agree.
    """
    args = _real_inputs(mtz_dir, pdb_dir, torch.float32)
    cpu = estimate_beta(**args)
    gpu = estimate_beta(
        **{k: (v.to("mps") if torch.is_tensor(v) else v) for k, v in args.items()}
    )
    assert gpu.sigma_a.device.type == "mps"
    torch.testing.assert_close(
        gpu.sigma_a.cpu(), cpu.sigma_a, rtol=GRID_STEP_RTOL, atol=GRID_STEP_RTOL
    )
    torch.testing.assert_close(
        gpu.beta.cpu(), cpu.beta, rtol=GRID_STEP_RTOL, atol=GRID_STEP_RTOL
    )
