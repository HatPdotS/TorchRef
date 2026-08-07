"""ReflectionData-level wiring of Wilson-probability outlier rejection.

Two paths reach the same guard from different information:

- intensities present -> ``load`` installs French-Wilson's *own* mask, computed
  on the true intensities (``FrenchWilson.valid_mask``);
- amplitudes only -> ``flag_wilson_outliers`` reconstructs intensities, which
  cannot recover negative-intensity evidence and so only catches absurd sigmas.

Also pins the refusal to fall back when every reflection is rejected.
"""

import pytest
import torch

from torchref.io.datasets.reflection_data import ReflectionData

CELL = (50.0, 60.0, 70.0, 90.0, 90.0, 90.0)


def _synthetic(F, F_sigma, device="cpu"):
    """A P1 dataset of len(F) reflections with distinct Miller indices."""
    n = len(F)
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


@pytest.mark.unit
def test_wilson_mask_is_installed_on_the_amplitude_path():
    n = 200
    F = torch.linspace(10.0, 200.0, n)
    data = _synthetic(F, F * 0.1)

    assert ReflectionData.WILSON_MASK_KEY in data.masks
    # Ordinary well-measured amplitudes are all explainable.
    assert bool(data.masks[ReflectionData.WILSON_MASK_KEY].all())


@pytest.mark.unit
def test_absurd_sigma_is_flagged_on_the_amplitude_path():
    """The one thing the reconstruction *can* still see."""
    n = 200
    F = torch.full((n,), 100.0)
    F_sigma = F * 0.1
    F_sigma[7] = 5.0e4  # sigma dwarfing the shell mean intensity

    data = _synthetic(F, F_sigma)
    keep = data.masks[ReflectionData.WILSON_MASK_KEY]

    assert not bool(keep[7])
    assert int((~keep).sum()) == 1


@pytest.mark.unit
def test_suspicious_sigma_is_no_longer_run_at_load():
    data = _synthetic(torch.linspace(10.0, 200.0, 100), torch.full((100,), 5.0))
    assert "flagged_sigma" not in data.masks

    # Still available for diagnostics, and still writes its own key.
    data.flag_suspicious_sigma()
    assert "flagged_sigma" in data.masks


@pytest.mark.unit
def test_refuses_to_fall_back_when_everything_is_rejected():
    """An all-False mask means broken data, not 'therefore keep everything'."""
    data = _synthetic(torch.linspace(10.0, 200.0, 50), torch.full((50,), 1.0))

    with pytest.raises(ValueError, match="reject all"):
        data._set_wilson_mask(torch.zeros(50, dtype=torch.bool))

    # The bad mask was not installed, so the dataset is left usable.
    assert bool(data.masks[ReflectionData.WILSON_MASK_KEY].any())


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


@pytest.mark.unit
def test_intensity_path_uses_french_wilsons_own_mask(tmp_path):
    """On a real intensity-bearing file the mask comes from load(), not from
    the amplitude reconstruction."""
    data = ReflectionData(verbose=0).load_mtz("tests/files/mtz/4BX9.mtz")

    assert data.I is not None, "4BX9 should load via the intensity path"
    assert ReflectionData.WILSON_MASK_KEY in data.masks
    torch.testing.assert_close(
        data.masks[ReflectionData.WILSON_MASK_KEY].sum(),
        data._FrenchWilson.valid_mask.sum(),
    )
