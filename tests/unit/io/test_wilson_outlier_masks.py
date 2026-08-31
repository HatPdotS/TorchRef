"""ReflectionData-level wiring of Wilson outlier rejection.

Two different questions reach ``masks`` from the same statistics, and the point
of these tests is that they stay distinguishable:

- ``french_wilson_valid`` -- can this intensity be integrated at all? Installed by
  ``load`` on the intensity path, from French-Wilson's own guard;
- ``wilson_valid`` -- is this observation an outlier? Two-tailed, computed
  against a robust anisotropic ``Sigma``;
- ``sanity_F`` -- is there a measurement here at all?

Conflating them is how a file with 44% absent reflections came to report a 44%
outlier rate.
"""

import pytest
import torch

from torchref.io.datasets.reflection_data import ReflectionData

CELL = (50.0, 60.0, 70.0, 90.0, 90.0, 90.0)


def _synthetic(F, F_sigma, hkl=None, device="cpu"):
    """A P1 dataset of len(F) reflections with distinct Miller indices."""
    n = len(F)
    if hkl is None:
        hkl = torch.stack(
            [
                torch.arange(1, n + 1),
                torch.zeros(n, dtype=torch.long),
                torch.zeros(n, dtype=torch.long),
            ],
            dim=-1,
        ).to(torch.int32)
    return ReflectionData.from_tensors(
        hkl,
        torch.as_tensor(F, dtype=torch.float32),
        torch.as_tensor(F_sigma, dtype=torch.float32),
        CELL,
        "P 1",
        rfree_flags=torch.ones(n, dtype=torch.bool),
        device=device,
        verbose=0,
        friedel_merged=True,
    )


def _wilson_grid(half_width=16, Sigma=500.0, seed=5):
    """Miller indices and Wilson-distributed amplitudes on a 3D grid."""
    axis = torch.arange(-half_width, half_width + 1)
    hkl = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    hkl = hkl.reshape(-1, 3)
    hkl = hkl[hkl.abs().sum(dim=1) > 0].to(torch.int32)

    generator = torch.Generator().manual_seed(seed)
    w = -torch.log(torch.rand(len(hkl), generator=generator).clamp(min=1e-12))
    I = Sigma * w
    F = torch.sqrt(I)
    return hkl, F, 0.02 * F + 0.5


def _plant_zingers(data, n, seed, factor=50.0, d_max=4.0):
    """Multiply ``n`` measured reflections by ``factor`` in intensity, re-flag.

    Corruption has to happen after construction: ``from_tensors`` canonicalises,
    which reorders the list, so indices chosen beforehand mean nothing
    afterwards. It also has to touch whichever observable the criterion reads --
    intensities when the file carries them, amplitudes otherwise -- and stay
    inside the resolution range the criterion tests, or the recall being measured
    is really the fraction of the dataset finer than ``d_max``.
    """
    generator = torch.Generator().manual_seed(seed)
    eligible = data.masks["sanity_F"].cpu() & (data.resolution.cpu() < d_max)
    measured = torch.nonzero(eligible, as_tuple=True)[0]
    planted = measured[torch.randperm(len(measured), generator=generator)[:n]]
    planted = planted.to(data.F.device)

    if data.I is not None:
        data.I[planted] *= factor
    data.F[planted] *= factor**0.5

    del data.masks[ReflectionData.WILSON_MASK_KEY]
    data.flag_wilson_outliers()
    flagged = torch.nonzero(
        ~data.masks[ReflectionData.WILSON_MASK_KEY], as_tuple=True
    )[0]
    return set(planted.tolist()), set(flagged.tolist())


# =============================================================================
# The masks stay separate
# =============================================================================


@pytest.mark.unit
def test_outlier_mask_is_installed_at_load():
    hkl, F, F_sigma = _wilson_grid()

    data = _synthetic(F, F_sigma, hkl=hkl)

    assert ReflectionData.WILSON_MASK_KEY in data.masks


@pytest.mark.unit
def test_clean_wilson_data_is_not_rejected():
    """Data drawn from the distribution the criterion assumes must survive it."""
    hkl, F, F_sigma = _wilson_grid()

    data = _synthetic(F, F_sigma, hkl=hkl)

    rejected = int((~data.masks[ReflectionData.WILSON_MASK_KEY]).sum())
    assert rejected <= 5, f"{rejected} of {len(F)} clean reflections rejected"


@pytest.mark.unit
def test_planted_zingers_are_rejected():
    hkl, F, F_sigma = _wilson_grid()
    data = _synthetic(F, F_sigma, hkl=hkl)

    planted, flagged = _plant_zingers(data, n=100, seed=11)

    assert len(flagged & planted) > 50
    # Precision matters as much as recall: this discards real measurements.
    assert len(flagged - planted) < 15


@pytest.mark.unit
def test_absent_measurements_are_sanity_not_outliers(mtz_dir):
    """A row with no measurement is not an improbable observation, and counting
    it as one is what made the old report meaningless."""
    data = ReflectionData(verbose=0).load_mtz(str(mtz_dir / "6G9X.mtz"))

    absent = ~data.masks["sanity_F"]
    assert int(absent.sum()) > 20000, "6G9X carries a large absent population"

    outliers = ~data.masks[ReflectionData.WILSON_MASK_KEY]
    assert int(outliers.sum()) < 100
    assert not bool((outliers & absent).any())


@pytest.mark.unit
def test_intensity_path_keeps_french_wilsons_guard_under_its_own_key(mtz_dir):
    data = ReflectionData(verbose=0).load_mtz(str(mtz_dir / "4BX9.mtz"))

    assert data.I is not None, "4BX9 should load via the intensity path"
    assert ReflectionData.FRENCH_WILSON_MASK_KEY in data.masks
    torch.testing.assert_close(
        data.masks[ReflectionData.FRENCH_WILSON_MASK_KEY].sum(),
        data._FrenchWilson.valid_mask.sum(),
    )
    # And the outlier test still ran on top of it, rather than being skipped
    # because a mask was already present.
    assert ReflectionData.WILSON_MASK_KEY in data.masks


# =============================================================================
# Deposited data
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    ["1DAW", "2DQ6", "3A5V", "3E98", "3GR5", "3K7M", "3VRJ", "4BX9", "5BOV", "6G9X"],
)
def test_deposited_structures_lose_almost_nothing(name, mtz_dir):
    """Deposited data has already been through processing and merging; a
    criterion that rejects percent-level populations of it is mis-calibrated,
    not perceptive."""
    data = ReflectionData(verbose=0).load_mtz(str(mtz_dir / f"{name}.mtz"))

    measured = int(data.masks["sanity_F"].sum())
    rejected = int((~data.masks[ReflectionData.WILSON_MASK_KEY]).sum())

    assert rejected / measured < 0.005, f"{name}: rejected {rejected}/{measured}"


@pytest.mark.unit
def test_flagged_reflections_show_no_directional_bias(mtz_dir):
    """The regression that catches a lost anisotropy correction.

    1DAW diffracts about four times more strongly along ``h*`` than ``l*``. A
    shell-averaged Sigma over-normalises the weak direction and under-normalises
    the strong one, so the flagged set piles up along ``h*`` -- 52 of 56 with
    mean ``|h|`` nearly twice the dataset's, before the correction existed.
    """
    data = ReflectionData(verbose=0).load_mtz(str(mtz_dir / "1DAW.mtz"))

    _, flagged = _plant_zingers(data, n=400, seed=13)
    assert len(flagged) > 100, "the planted population must be found first"

    index = torch.tensor(sorted(flagged), device=data.hkl.device)
    hkl = data.hkl.abs().to(torch.float32)
    bias = hkl[index].mean(dim=0) / hkl[data.masks["sanity_F"]].mean(dim=0)

    assert bool(((bias > 0.6) & (bias < 1.6)).all()), f"directional bias {bias}"


# =============================================================================
# Refusals and knobs
# =============================================================================


@pytest.mark.unit
def test_alpha_controls_how_much_is_rejected():
    hkl, F, F_sigma = _wilson_grid()
    data = _synthetic(F, F_sigma, hkl=hkl)
    _, flagged = _plant_zingers(data, n=200, seed=17)

    del data.masks[ReflectionData.WILSON_MASK_KEY]
    data.flag_wilson_outliers(alpha=1e-12)
    loose = int((~data.masks[ReflectionData.WILSON_MASK_KEY]).sum())

    assert loose < len(flagged)


@pytest.mark.unit
def test_refuses_to_fall_back_when_everything_is_rejected(monkeypatch):
    """An all-False mask means broken data or a failed Sigma, not 'therefore
    keep everything'."""
    hkl, F, F_sigma = _wilson_grid(half_width=10)
    data = _synthetic(F, F_sigma, hkl=hkl)

    def reject_everything(*args, **kwargs):
        n = args[0].shape[0]
        return torch.zeros(n, dtype=torch.bool, device=args[0].device), {
            "n_tested": n,
            "n_strong": n,
            "n_weak": 0,
            "log_p_threshold": 0.0,
            "h_min": 0.0,
            "U": torch.zeros(6),
        }

    monkeypatch.setattr(
        "torchref.base.wilson_outliers.wilson_outlier_mask", reject_everything
    )
    del data.masks[ReflectionData.WILSON_MASK_KEY]

    with pytest.raises(ValueError, match="reject all"):
        data.flag_wilson_outliers()

    # The bad mask was not installed, so the dataset is left usable.
    assert ReflectionData.WILSON_MASK_KEY not in data.masks


@pytest.mark.unit
def test_french_wilson_guard_refuses_an_all_false_mask():
    hkl, F, F_sigma = _wilson_grid(half_width=8)
    data = _synthetic(F, F_sigma, hkl=hkl)

    with pytest.raises(ValueError, match="rejected all"):
        data._set_french_wilson_mask(torch.zeros(len(data.hkl), dtype=torch.bool))


@pytest.mark.unit
def test_suspicious_sigma_is_no_longer_run_at_load():
    hkl, F, F_sigma = _wilson_grid(half_width=8)
    data = _synthetic(F, F_sigma, hkl=hkl)
    assert "flagged_sigma" not in data.masks

    # Still available for diagnostics, and still writes its own key.
    data.flag_suspicious_sigma()
    assert "flagged_sigma" in data.masks


@pytest.mark.unit
def test_too_few_reflections_are_left_alone():
    """Wilson statistics cannot be estimated from a handful of reflections, and
    guessing at them would reject real data."""
    data = _synthetic(torch.linspace(10.0, 200.0, 120), torch.full((120,), 5.0))

    keep = data.masks.get(ReflectionData.WILSON_MASK_KEY)
    assert keep is None or bool(keep.all())


@pytest.mark.unit
def test_french_wilson_records_its_own_mask_full_size():
    from torchref.base.french_wilson import FrenchWilson

    hkl = torch.tensor([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]])
    fw = FrenchWilson(hkl, torch.tensor(CELL), "P 1", verbose=0)
    assert fw.valid_mask is None

    I = torch.tensor([100.0, 50.0, -5.0, float("nan")])
    sigma_I = torch.tensor([10.0, 8.0, 7.0, 5.0])
    fw(I, sigma_I)

    assert fw.valid_mask is not None
    assert fw.valid_mask.shape == I.shape
    assert fw.valid_mask.dtype == torch.bool
    # Well-measured reflections survive; the NaN row never converted, so it is
    # not kept on the strength of a comparison that was never made.
    assert fw.valid_mask[:2].all()
    assert not bool(fw.valid_mask[3])
