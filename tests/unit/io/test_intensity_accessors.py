"""Scaled intensities must carry the amplitude scale **squared**.

``get_corrected_data`` applies ``corr(s, U) * exp(log_scale)`` to amplitudes. The
intensity counterpart has to apply the square of that, because the correction is defined
on amplitudes. Getting it wrong leaves a smooth, resolution-dependent error in the
intensities that is indistinguishable from a scale or overall-B mismatch -- so it is
pinned here as an exact ratio rather than checked by eye.

The subset views are also pinned: ``.F``/``.sigF`` are corrected and ``.F_raw``/``.sigF_raw``
are not, and ``.I``/``.sigI`` now follow the same rule. A view where ``.F`` was scaled and
``.I`` was not is the shape of bug that makes an amplitude target and an intensity target
disagree about which dataset they are fitting.
"""

import pytest
import torch


@pytest.fixture(scope="module")
def with_intensities(mtz_dir):
    """1DAW -- the only fixture carrying both I/SIGI and FP/SIGFP."""
    mtz = mtz_dir / "1DAW.mtz"
    if not mtz.exists():
        pytest.skip("1DAW fixture not present")

    from torchref import ReflectionData

    data = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))
    if data.I is None:
        pytest.skip("1DAW loaded without intensities")
    return data


@pytest.fixture(scope="module")
def without_intensities(mtz_dir):
    """3GR5 -- amplitudes only."""
    mtz = mtz_dir / "3GR5.mtz"
    if not mtz.exists():
        pytest.skip("3GR5 fixture not present")

    from torchref import ReflectionData

    data = ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz))
    if data.I is not None:
        pytest.skip("3GR5 unexpectedly carries intensities")
    return data


def _perturb(data, dlog=0.3):
    """Give the dataset a non-trivial scale and anisotropy, restored on exit."""

    class _Ctx:
        def __enter__(self):
            self.log_scale = data.log_scale.detach().clone()
            self.U = data.U_aniso.detach().clone()
            with torch.no_grad():
                data.log_scale += dlog
                data.U_aniso += torch.tensor([0.01, -0.005, 0.008, 0.002, 0.0, 0.0])
            return data

        def __exit__(self, *exc):
            with torch.no_grad():
                data.log_scale.copy_(self.log_scale)
                data.U_aniso.copy_(self.U)
            data._corrected_fp = None
            data._corrected_I_fp = None

    return _Ctx()


@pytest.mark.unit
class TestSquaredScale:
    def test_intensity_factor_is_the_square_of_the_amplitude_factor(
        self, with_intensities
    ):
        """The exact relationship, as a per-reflection ratio.

        Independent of how I and F relate in the file (French-Wilson, not I == F**2),
        because it compares each quantity against its own unscaled self.
        """
        data = with_intensities
        with _perturb(data):
            F_scaled, _ = data.get_corrected_data()
            I_scaled, _ = data.get_corrected_intensities()

            keep = data.masks().to(torch.bool) & (data.F.abs() > 1e-6) & (
                data.I.abs() > 1e-6
            )
            amp_factor = (F_scaled[keep] / data.F[keep]) ** 2
            int_factor = I_scaled[keep] / data.I[keep]

            rel = ((int_factor - amp_factor).abs() / amp_factor.abs()).max()
            assert rel < 1e-5, (
                f"intensity scale factor is not the square of the amplitude one "
                f"(max rel error {rel:.2e})"
            )

    def test_sigma_scales_with_the_same_factor_as_the_intensity(
        self, with_intensities
    ):
        data = with_intensities
        with _perturb(data):
            I_scaled, sig_scaled = data.get_corrected_intensities()
            keep = (data.I.abs() > 1e-6) & (data.I_sigma.abs() > 1e-6)

            ratio_I = I_scaled[keep] / data.I[keep]
            ratio_s = sig_scaled[keep] / data.I_sigma[keep]
            assert torch.allclose(ratio_I, ratio_s, rtol=1e-6)

    def test_the_perturbation_actually_changes_the_intensities(
        self, with_intensities
    ):
        """Anti-vacuity: at log_scale 0 and U 0 every factor above is 1."""
        data = with_intensities
        before, _ = data.get_corrected_intensities()
        with _perturb(data):
            after, _ = data.get_corrected_intensities()
            assert not torch.allclose(before, after)

    def test_a_pure_scale_change_squares_into_the_intensities(
        self, with_intensities
    ):
        """A doubling of the amplitude scale must quadruple the intensities."""
        data = with_intensities
        base, _ = data.get_corrected_intensities()
        original = data.log_scale.detach().clone()
        try:
            with torch.no_grad():
                data.log_scale += float(torch.log(torch.tensor(2.0)))
            data._corrected_I_fp = None
            doubled, _ = data.get_corrected_intensities()
        finally:
            with torch.no_grad():
                data.log_scale.copy_(original)
            data._corrected_I_fp = None

        keep = base.abs() > 1e-6
        ratio = (doubled[keep] / base[keep])
        assert torch.allclose(ratio, torch.full_like(ratio, 4.0), rtol=1e-5)


@pytest.mark.unit
class TestSubsetViews:
    def test_amplitudes_and_intensities_are_both_corrected(self, with_intensities):
        data = with_intensities
        with _perturb(data):
            work = data.work
            assert not torch.allclose(work.F, work.F_raw)
            assert not torch.allclose(work.I, work.I_raw), (
                "subset.I returned raw intensities while subset.F was scaled"
            )
            assert not torch.allclose(work.sigI, work.sigI_raw)

    def test_raw_views_match_the_parent_tensors(self, with_intensities):
        data = with_intensities
        work = data.work
        idx = work.indices
        assert torch.equal(work.I_raw, data.I.index_select(0, idx))
        assert torch.equal(work.sigI_raw, data.I_sigma.index_select(0, idx))

    def test_subset_intensities_match_the_full_size_scaled_array(
        self, with_intensities
    ):
        data = with_intensities
        with _perturb(data):
            I_scaled, sig_scaled = data.get_corrected_intensities()
            for kind in ("work", "free"):
                sub = getattr(data, kind)
                idx = sub.indices
                assert torch.equal(sub.I, I_scaled.index_select(0, idx))
                assert torch.equal(sub.sigI, sig_scaled.index_select(0, idx))

    def test_cache_follows_a_scale_change(self, with_intensities):
        """The fingerprint must invalidate, or a refinement would fit stale data."""
        data = with_intensities
        first = data.work.I.clone()
        original = data.log_scale.detach().clone()
        try:
            with torch.no_grad():
                data.log_scale += 0.5
            second = data.work.I
            assert not torch.allclose(first, second)
        finally:
            with torch.no_grad():
                data.log_scale.copy_(original)


@pytest.mark.unit
class TestNoIntensities:
    def test_get_corrected_intensities_raises_with_an_actionable_message(
        self, without_intensities
    ):
        with pytest.raises(ValueError, match="no I/SIGI columns|No intensities"):
            without_intensities.get_corrected_intensities()

    def test_subset_views_return_none_rather_than_raising(self, without_intensities):
        work = without_intensities.work
        assert work.I is None
        assert work.sigI is None
        assert work.I_raw is None
        assert work.sigI_raw is None
