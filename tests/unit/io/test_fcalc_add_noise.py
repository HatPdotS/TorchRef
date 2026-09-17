"""Simulated intensities must keep their negatives.

``add_noise`` draws two independent noisy half-datasets and returns their mean. The
amplitude it derives has to be clamped -- an amplitude cannot be negative -- but the
*intensity* must not be, and both the intensity and its sigma have to survive on the
returned dataset.

Why this is not a detail: clamping ``I_mean`` at zero puts a **positive bias** on exactly
the weak reflections where the noise dominates. That bias is smooth, positive, and largest
where the signal is weakest -- the same signature as a genuine positive perturbation of the
merged intensity. Any study of an effect at the 1e-3 level built on clamped simulated data
would be measuring its own generator.
"""

import pytest
import torch


@pytest.fixture
def fcalc_scene():
    """A small P1 scene with a deliberately wide dynamic range.

    The weak tail is the point: with strong reflections only, noise never pushes an
    intensity negative and nothing below can distinguish clamped from unclamped.
    """
    from torchref.io.datasets import FcalcDataset

    dataset = FcalcDataset.from_cell_and_resolution(
        cell=[30.0, 32.0, 34.0, 90.0, 90.0, 90.0],
        spacegroup="P 1",
        d_min=3.0,
        device=torch.device("cpu"),
    )
    n = len(dataset.hkl)
    gen = torch.Generator().manual_seed(17)
    # Amplitudes spanning three orders of magnitude, so I spans six.
    amp = 10.0 ** (torch.rand(n, generator=gen) * 3.0 - 1.0)
    phase = torch.rand(n, generator=gen) * 6.283
    dataset.set_fcalc((amp * torch.exp(1j * phase)).to(torch.complex64))
    return dataset


@pytest.mark.unit
class TestNegativesSurvive:
    def test_some_intensities_come_out_negative(self, fcalc_scene):
        noisy = fcalc_scene.add_noise(sigma_mul=0.5, seed=3, verbose=False)
        assert noisy.I is not None, "add_noise did not retain intensities"
        assert bool((noisy.I < 0).any()), (
            "no negative intensities at 50% multiplicative noise -- either the scene has "
            "no weak reflections or the intensities were clamped"
        )

    def test_the_amplitude_is_clamped_but_the_intensity_is_not(self, fcalc_scene):
        """The asymmetry is deliberate, so it is asserted rather than assumed."""
        noisy = fcalc_scene.add_noise(sigma_mul=0.5, seed=3, verbose=False)
        assert bool((noisy.fcalc_amp >= 0).all())
        negative = noisy.I < 0
        assert bool(negative.any())
        # Where the intensity is negative the amplitude is floored at zero, so the two
        # cannot agree -- which is exactly the information a clamp would have destroyed.
        assert torch.allclose(
            noisy.fcalc_amp[negative], torch.zeros(int(negative.sum()))
        )

    @staticmethod
    def _weak(noisy, truth):
        """The subset a clamp at zero can touch: reflections within 2 sigma of zero.

        Measuring over the whole list instead would drown the effect -- the strong
        reflections contribute nothing to the bias but dominate its standard error, so
        the very reflections the clamp distorts are the ones averaged away.

        Note this needs a noise model whose sigma does **not** scale with the intensity.
        Under purely multiplicative noise ``sigma = f * I``, so no reflection is ever weak
        relative to its own sigma and this subset is empty; the Poisson-like ``sigma_lin``
        term below gives ``sigma ~ sqrt(I)`` and therefore a genuine weak tail.
        """
        return truth < 2.0 * noisy.I_sigma

    def test_the_intensity_is_unbiased_on_the_weak_reflections(self, fcalc_scene):
        """The property the clamp breaks, as a bound on the mean of the weak tail.

        The tolerance is the standard error over that subset, not a percentage.
        """
        noisy = fcalc_scene.add_noise(
            sigma_lin=200.0, sigma_mul=0.0, seed=5, verbose=False
        )
        truth = fcalc_scene.fcalc_amp**2
        weak = self._weak(noisy, truth)
        assert int(weak.sum()) > 50, "too few weak reflections to say anything"

        residual = (noisy.I - truth)[weak]
        sem = float(noisy.I_sigma[weak].pow(2).sum().sqrt() / int(weak.sum()))
        bias = float(residual.mean())
        assert abs(bias) < 4.0 * sem, (
            f"weak-reflection intensity bias {bias:.4g} exceeds 4 sigma "
            f"({4 * sem:.4g}); the intensities are being clamped or otherwise skewed"
        )

    def test_clamping_would_be_detectable_on_this_scene(self, fcalc_scene):
        """Anti-vacuity: quantify what the defect would have looked like here.

        Without this, the test above could pass simply because the scene has no
        reflections weak enough for a clamp to reach.

        Measured on this scene, the clamp bias runs only about one to two times the
        statistical error on the same mean, and needs a high noise level to stand clear
        of it at all. That is not a reason to tolerate it: the bias is **systematic**, so
        it repeats identically across datasets and survives averaging, while the error it
        is being compared against shrinks as 1/sqrt(N). It is the accumulation, not the
        size on any one dataset, that would corrupt a calibration curve.
        """
        noisy = fcalc_scene.add_noise(
            sigma_lin=200.0, sigma_mul=0.0, seed=5, verbose=False
        )
        truth = fcalc_scene.fcalc_amp**2
        weak = self._weak(noisy, truth)
        assert int(weak.sum()) > 50

        honest = float((noisy.I - truth)[weak].mean())
        clamped = float((noisy.I.clamp(min=0.0) - truth)[weak].mean())
        sem = float(noisy.I_sigma[weak].pow(2).sum().sqrt() / int(weak.sum()))

        assert clamped > honest, "clamping did not raise the mean on this scene"
        assert clamped > 4.0 * sem, (
            f"the clamped bias ({clamped:.4g}) would fall inside the noise "
            f"({4 * sem:.4g}) here, so this scene cannot demonstrate the defect"
        )


@pytest.mark.unit
class TestSigmaAndHalves:
    def test_sigma_of_the_mean_is_the_single_draw_sigma_over_root_two(self, fcalc_scene):
        a = fcalc_scene.add_noise(sigma_mul=0.2, seed=11, verbose=False)
        b = fcalc_scene.add_noise(sigma_mul=0.2, seed=12, verbose=False)
        # Same model, same noise scale: the reported sigma is a property of the model,
        # not of the draw, so it must be identical across seeds.
        assert torch.allclose(a.I_sigma, b.I_sigma)

        expected = torch.sqrt(
            torch.tensor(0.2) ** 2 * fcalc_scene.fcalc_amp**4
        ) / (2.0**0.5)
        assert torch.allclose(a.I_sigma, expected, rtol=1e-5)

    def test_amplitude_sigma_uses_the_true_amplitude(self, fcalc_scene):
        """Propagating against the noisy amplitude would warp sigma per draw and break
        inverse-variance weighting downstream."""
        a = fcalc_scene.add_noise(sigma_mul=0.2, seed=11, verbose=False)
        b = fcalc_scene.add_noise(sigma_mul=0.2, seed=99, verbose=False)
        assert torch.allclose(a.fobs_sigma, b.fobs_sigma)

    def test_different_seeds_give_different_draws(self, fcalc_scene):
        a = fcalc_scene.add_noise(sigma_mul=0.2, seed=1, verbose=False)
        b = fcalc_scene.add_noise(sigma_mul=0.2, seed=2, verbose=False)
        assert not torch.allclose(a.I, b.I)

    def test_the_same_seed_reproduces(self, fcalc_scene):
        a = fcalc_scene.add_noise(sigma_mul=0.2, seed=7, verbose=False)
        b = fcalc_scene.add_noise(sigma_mul=0.2, seed=7, verbose=False)
        assert torch.equal(a.I, b.I)

    def test_phases_are_untouched(self, fcalc_scene):
        """Only the amplitude is perturbed; the phase is the model's."""
        noisy = fcalc_scene.add_noise(sigma_mul=0.2, seed=4, verbose=False)
        strong = fcalc_scene.fcalc_amp > 1.0
        assert torch.allclose(
            noisy.fcalc_phase[strong], fcalc_scene.fcalc_phase[strong], atol=1e-5
        )

    def test_the_source_dataset_is_not_modified(self, fcalc_scene):
        before = fcalc_scene.fcalc_amp.clone()
        fcalc_scene.add_noise(sigma_mul=0.3, seed=8, verbose=False)
        assert torch.equal(fcalc_scene.fcalc_amp, before)
        assert fcalc_scene.I is None


@pytest.mark.unit
class TestReferenceDriven:
    def test_sigmas_are_grafted_from_the_reference(self, fcalc_scene, mtz_dir):
        """The path that matters for real data: per-reflection sigmas from a measured
        dataset rather than a parametric model."""
        from torchref import ReflectionData
        from torchref.io.datasets import FcalcDataset

        mtz = mtz_dir / "1DAW.mtz"
        if not mtz.exists():
            pytest.skip("1DAW fixture not present")
        ref = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))
        if ref.I is None:
            pytest.skip("1DAW loaded without intensities")

        # Build on the reference's own HKL list, which is what makes grafting 1:1.
        scene = FcalcDataset(
            hkl=ref.hkl.clone(), cell=fcalc_scene.cell,
            spacegroup=fcalc_scene.spacegroup, device=torch.device("cpu"),
        )
        gen = torch.Generator().manual_seed(2)
        amp = torch.rand(len(ref.hkl), generator=gen) * 100.0
        scene.set_fcalc((amp + 0j).to(torch.complex64))

        noisy = scene.add_noise(reference=ref, seed=1, verbose=False)
        assert torch.allclose(noisy.I_sigma, ref.I_sigma / (2.0**0.5))

    def test_a_mismatched_reference_is_rejected(self, fcalc_scene, mtz_dir):
        from torchref import ReflectionData

        mtz = mtz_dir / "1DAW.mtz"
        if not mtz.exists():
            pytest.skip("1DAW fixture not present")
        ref = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))

        with pytest.raises(ValueError, match="does not match"):
            fcalc_scene.add_noise(reference=ref, verbose=False)

    def test_a_reference_without_sigmas_is_rejected(self, fcalc_scene):
        from types import SimpleNamespace

        bad = SimpleNamespace(I_sigma=None, hkl=fcalc_scene.hkl)
        with pytest.raises(ValueError, match="I_sigma is None"):
            fcalc_scene.add_noise(reference=bad, verbose=False)
