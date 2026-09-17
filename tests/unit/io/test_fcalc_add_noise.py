"""Unbiased noisy intensities, uncertainty propagation and reproducible draws."""

import pytest
import torch


@pytest.fixture
def fcalc_scene():
    """A small P1 scene with a deliberately wide dynamic range."""
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
        """The subset a clamp at zero can touch: reflections within 2 sigma of zero."""
        return truth < 2.0 * noisy.I_sigma

    def test_the_intensity_is_unbiased_on_the_weak_reflections(self, fcalc_scene):
        """The property the clamp breaks, as a bound on the mean of the weak tail."""
        noisy = fcalc_scene.add_noise(
            sigma_lin=200.0, sigma_mul=0.0, seed=5, verbose=False
        )
        truth = fcalc_scene.fcalc_amp**2
        weak = self._weak(noisy, truth)
        assert int(weak.sum()) > 50, "too few weak reflections to say anything"

        assert (noisy.I[weak] < 0).any()
        residual = (noisy.I - truth)[weak]
        sem = float(noisy.I_sigma[weak].pow(2).sum().sqrt() / int(weak.sum()))
        bias = float(residual.mean())
        assert abs(bias) < 4.0 * sem, (
            f"weak-reflection intensity bias {bias:.4g} exceeds 4 sigma "
            f"({4 * sem:.4g}); the intensities are being clamped or otherwise skewed"
        )


@pytest.mark.unit
class TestSigmaAndHalves:
    def test_sigma_of_the_mean_is_the_single_draw_sigma_over_root_two(
        self, fcalc_scene
    ):
        a = fcalc_scene.add_noise(sigma_mul=0.2, seed=11, verbose=False)
        b = fcalc_scene.add_noise(sigma_mul=0.2, seed=12, verbose=False)
        # Same model, same noise scale: the reported sigma is a property of the model,
        # not of the draw, so it must be identical across seeds.
        assert torch.allclose(a.I_sigma, b.I_sigma)

        expected = torch.sqrt(torch.tensor(0.2) ** 2 * fcalc_scene.fcalc_amp**4) / (
            2.0**0.5
        )
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
            hkl=ref.hkl.clone(),
            cell=fcalc_scene.cell,
            spacegroup=fcalc_scene.spacegroup,
            device=torch.device("cpu"),
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
