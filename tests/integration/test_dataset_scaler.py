"""Joint observed-data scaling and live dataset access on deposited reflections."""

import copy
import math

import pytest
import torch

from torchref import DatasetCollection, ScaledDataset
from torchref.base.targets.dataset_scaling import dataset_scaling_loss
from torchref.config import get_default_device, get_int_dtype
from torchref.scaling import DatasetScaler

pytestmark = pytest.mark.integration


def clone(data):
    return data.__select__(
        torch.arange(len(data), device=data.device, dtype=get_int_dtype())
    )


def collection(data, factors):
    dc = DatasetCollection(device=data.device, verbose=0)
    for i, factor in enumerate(factors):
        raw = clone(data)
        raw.F *= factor
        raw.F_sigma *= factor
        if raw.I is not None:
            raw.I *= factor**2
            raw.I_sigma *= factor**2
        dc.add_dataset(str(i), raw)
    return dc


@pytest.mark.parametrize("factors", [(1, 1), (1, 1000), (1, 2, 8)])
def test_known_scales_are_centered_and_sources_stay_raw(
    loaded_reflection_data, factors
):
    before = loaded_reflection_data.F.clone()
    dc = collection(loaded_reflection_data, factors).scale(nsteps=2)
    expected = torch.tensor(
        factors, dtype=dc.scaler.raw_parameters.dtype, device=get_default_device()
    ).log()
    expected = expected.mean() - expected
    torch.testing.assert_close(
        dc.scaler.corrections[:, 0], expected, atol=1e-4, rtol=1e-4
    )
    assert dc.scaler.corrections[:, 1:].abs().max() < 1e-4
    assert torch.equal(loaded_reflection_data.F, before)
    assert not hasattr(loaded_reflection_data, "parameters")
    assert not hasattr(loaded_reflection_data, "log_scale")
    assert not hasattr(loaded_reflection_data, "U_aniso")
    for key, _ in dc:
        assert isinstance(dc[key], ScaledDataset)
        torch.testing.assert_close(dc[key].F, dc["0"].F, rtol=1e-4, atol=1e-4)


def test_known_anisotropy_is_recovered(loaded_reflection_data):
    dc = collection(loaded_reflection_data, (1, 1, 1))
    reference = DatasetScaler(dc.datasets)
    truth = torch.tensor(
        [
            [0.3, 0.2, -0.1, 0.1, 0.03, -0.02, 0.05],
            [-0.2, -0.1, 0.2, -0.05, -0.02, 0.01, -0.03],
            [-0.1, -0.1, -0.1, -0.05, -0.01, 0.01, -0.02],
        ],
        dtype=reference.raw_parameters.dtype,
        device=get_default_device(),
    )
    for i, ds in enumerate(dc.values()):
        distortion = (reference.design(ds.hkl) @ truth[i]).exp()
        ds.F /= distortion
        ds.F_sigma /= distortion
    dc.scale(nsteps=4)
    torch.testing.assert_close(dc.scaler.corrections, truth, atol=2e-3, rtol=2e-3)


def test_live_access_scales_both_sigmas_and_all_entrypoints(loaded_reflection_data):
    dc = collection(loaded_reflection_data, (1, 1)).scale(nsteps=1)
    data = dc["0"]
    with torch.no_grad():
        dc.scaler.raw_parameters[0, 0] = 2 * math.log(2)
        dc.scaler.raw_parameters[0, 1] = 0.1
    correction = 2 * torch.exp(
        0.05 * (data.hkl[:, 0] / dc.scaler.hkl_scale[0]).square()
    )
    data.generate_validation_set(val_fraction_of_free=0.5, seed=0)
    for name, power, subset_attr, stack in [
        ("F", 1, "F", dc.stack_F_obs),
        ("F_sigma", 1, "sigF", dc.stack_F_sigma),
        ("I", 2, "I", dc.stack_I_obs),
        ("I_sigma", 2, "sigI", dc.stack_I_sigma),
    ]:
        raw = getattr(data, name + "_raw")
        actual = getattr(data, name)
        torch.testing.assert_close(actual, raw * correction**power)
        torch.testing.assert_close(stack()[0], actual)
        for kind in ("work", "free", "validation"):
            subset = getattr(data, kind)
            torch.testing.assert_close(
                getattr(subset, subset_attr), actual[subset.mask]
            )
            torch.testing.assert_close(
                getattr(subset, subset_attr + "_raw"), raw[subset.mask]
            )
    torch.testing.assert_close(dc(mask=False)["0"][1], data.F)
    torch.testing.assert_close(data.get_corrected_data(), (data.F, data.F_sigma))
    torch.testing.assert_close(data.get_corrected_intensities(), (data.I, data.I_sigma))
    dc.scaler.requires_grad_(True)
    for _ in range(2):
        dc.scaler.zero_grad()
        data.work.F.sum().backward()
        assert dc.scaler.raw_parameters.grad[1].abs().sum() > 0
    dc.scaler.requires_grad_(False)


def test_selection_copy_alignment_and_independent_collections(loaded_reflection_data):
    a = collection(loaded_reflection_data, (1, 4)).scale(nsteps=1)
    b = collection(loaded_reflection_data, (1, 9)).scale(nsteps=1)
    view = a["0"]
    selected = view.__select__(
        torch.arange(
            0, len(view), 3, device=get_default_device(), dtype=get_int_dtype()
        )
    )
    torch.testing.assert_close(selected.F, view.F[::3])
    assert selected.scaler is a.scaler
    assert copy.deepcopy(view).scaler is a.scaler
    aligned = view.copy().validate_hkl(view.hkl.flip(0))
    torch.testing.assert_close(aligned.F, view.F.flip(0))
    assert a.scaler is not b.scaler
    assert not torch.allclose(a["0"].F, b["0"].F)
    scaler = a.scaler
    a.scale(nsteps=1)
    assert a.scaler is scaler
    a.add_dataset("extra", loaded_reflection_data)
    assert a.scaler is None
    a.scale(nsteps=1)
    assert a.scaler is not scaler
    torch.testing.assert_close(view.F, view.F_raw * 2)


@pytest.mark.usefixtures("double_cpu")
def test_two_dataset_loss_matches_propagated_variance_and_gradients(
    loaded_reflection_data,
):
    f = torch.stack(
        (loaded_reflection_data.F[:128], loaded_reflection_data.F[:128] * 1.1)
    ).double()
    sigma = torch.stack(
        (loaded_reflection_data.F_sigma[:128], loaded_reflection_data.F_sigma[:128] * 3)
    ).double()
    log_k = torch.zeros_like(f, requires_grad=True)
    mask = torch.isfinite(f) & torch.isfinite(sigma) & (sigma > 0)
    actual = dataset_scaling_loss(f, sigma, log_k, mask)
    common = mask.all(dim=0)
    expected = 0.5 * ((f[0] - f[1]).square() / sigma.square().sum(dim=0))[common].sum()
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.gradcheck(
        lambda p: dataset_scaling_loss(f, sigma, p - p.mean(dim=0), mask),
        (log_k,),
        fast_mode=True,
    )
    noisier = dataset_scaling_loss(f, sigma * 2, log_k, mask)
    torch.testing.assert_close(noisier, actual / 4)


def test_missing_and_invalid_observations_have_finite_gradients(loaded_reflection_data):
    f = torch.stack(
        (loaded_reflection_data.F[:128], loaded_reflection_data.F[:128] * 1.1)
    )
    sigma = torch.stack(
        (loaded_reflection_data.F_sigma[:128], loaded_reflection_data.F_sigma[:128])
    )
    mask = torch.isfinite(f) & torch.isfinite(sigma) & (sigma > 0)
    mask[:, :10] = False
    f[:, :10] = float("nan")
    sigma[:, :10] = 0
    log_k = torch.zeros_like(f, requires_grad=True)
    loss = dataset_scaling_loss(f, sigma, log_k, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(log_k.grad).all()
    assert torch.equal(log_k.grad[:, :10], torch.zeros_like(log_k.grad[:, :10]))


def test_free_and_validation_changes_do_not_affect_fit(loaded_reflection_data):
    results = []
    for filler in (3, 900):
        dc = collection(loaded_reflection_data, (1, 2))
        ds = dc["1"]
        mismatched = ds.work.indices[:60]
        ds.rfree_flags[mismatched[:30]] = False
        ds.validation_flags = torch.zeros_like(ds.rfree_flags, dtype=torch.bool)
        ds.validation_flags[mismatched[30:]] = True
        held_out = ~ds.work.mask
        ds.F[held_out] = filler
        ds.F_sigma[held_out] = filler
        dc.scale(nsteps=2)
        assert torch.equal(dc.hkl, dc.scaler.hkl)
        assert not dc.scaler.fit_mask[:, mismatched].any()
        results.append(dc.scaler.raw_parameters.detach().clone())
    assert torch.equal(*results)


def test_permutation_and_partial_overlap_chain(loaded_reflection_data):
    n = len(loaded_reflection_data)
    sources = {
        "a": clone(loaded_reflection_data).__select__(
            torch.arange(n // 2, device=get_default_device(), dtype=get_int_dtype())
        ),
        "b": clone(loaded_reflection_data),
        "c": clone(loaded_reflection_data).__select__(
            torch.arange(n // 2, n, device=get_default_device(), dtype=get_int_dtype())
        ),
    }
    sources["b"].F *= 2
    sources["b"].F_sigma *= 2
    sources["c"].F *= 4
    sources["c"].F_sigma *= 4
    results = []
    for order in [("a", "b", "c"), ("c", "a", "b")]:
        dc = DatasetCollection(device=get_default_device(), verbose=0)
        for key in order:
            dc.add_dataset(key, sources[key])
        dc.scale(nsteps=2)
        results.append(
            {k: dc.scaler.corrections[dc.scaler.keys.index(k)] for k in order}
        )
    for key in sources:
        torch.testing.assert_close(
            results[0][key], results[1][key], atol=1e-4, rtol=1e-4
        )
    with pytest.raises(ValueError, match="disconnected"):
        DatasetScaler({k: sources[k] for k in ("a", "c")})
    with pytest.raises(ValueError, match="identify"):
        DatasetScaler(
            {
                "a": loaded_reflection_data.__select__(
                    torch.arange(6, device=get_default_device(), dtype=get_int_dtype())
                ),
                "b": loaded_reflection_data.__select__(
                    torch.arange(6, device=get_default_device(), dtype=get_int_dtype())
                ),
            }
        )


def test_checkpoint_and_mtz_export_preserve_observations(
    loaded_reflection_data, tmp_path
):
    import reciprocalspaceship as rs

    dc = collection(loaded_reflection_data, (1, 4)).scale(nsteps=1)
    path = tmp_path / "collection.pt"
    dc.save_state(path)
    restored = DatasetCollection.load_state(path, device=get_default_device())
    assert restored["0"].scaler is restored["1"].scaler is restored.scaler
    torch.testing.assert_close(restored.stack_F_obs(), dc.stack_F_obs())
    path = tmp_path / "view.pt"
    dc["0"].save_state(path)
    restored_view = ScaledDataset.load_state(path, device=get_default_device())
    torch.testing.assert_close(restored_view.I, dc["0"].I)
    path = tmp_path / "scaled.mtz"
    dc["0"].write_mtz(str(path))
    exported = rs.read_mtz(str(path))
    assert len(exported) == len(dc["0"])
    import numpy as np

    for column, attribute in [
        ("F-obs", "F"),
        ("SIGF-obs", "F_sigma"),
        ("I-obs", "I"),
        ("SIGI-obs", "I_sigma"),
    ]:
        np.testing.assert_allclose(
            exported[column].to_numpy(dtype=float),
            getattr(dc["0"], attribute).detach().cpu().numpy(),
            rtol=1e-6,
        )


def test_bijvoet_observations_keep_distinct_identities(loaded_reflection_data):
    """Canonical duplicate HKLs retain separate signed observations and sigmas."""
    data = loaded_reflection_data.__select__(
        torch.arange(
            256, device=get_default_device(), dtype=get_int_dtype()
        ).repeat_interleave(2)
    )
    data.friedel_merged = False
    data.friedel_flags = (
        torch.arange(len(data), device=get_default_device(), dtype=get_int_dtype()) % 2
        == 1
    )
    data.hkl_anomalous = torch.where(data.friedel_flags[:, None], -data.hkl, data.hkl)
    data.F[data.friedel_flags] *= 1.25
    dc = collection(data, (1, 4)).scale(nsteps=1)
    assert len(dc) == len(data)
    lookup = {tuple(h.tolist()): f for h, f in zip(data.hkl_anomalous, data.F)}
    expected = torch.stack([lookup[tuple(h.tolist())] for h in dc["0"].hkl_anomalous])
    torch.testing.assert_close(dc["0"].F_raw, expected)
    torch.testing.assert_close(dc["0"].F, dc["1"].F, rtol=1e-4, atol=1e-4)
    assert dc["0"].friedel_flags.sum() == 256
    selected = (
        dc["0"]
        .copy()
        .validate_hkl(dc.hkl.flip(0), identity_hkl=dc["0"].hkl_anomalous.flip(0))
    )
    torch.testing.assert_close(selected.F, dc["0"].F.flip(0))


def test_raw_dataset_excludes_deprecated_interfaces(loaded_reflection_data):
    """Observation storage exposes subset views without optimization methods."""
    for name in (
        "log_scale",
        "U_aniso",
        "parameters",
        "setup_scale",
        "setup_anisotropy",
        "compute_e_values",
        "get_radial_shells",
        "get_work_set",
        "get_test_set",
        "get_rfree_masks",
        "_masked_unpack",
        "get_mask",
    ):
        assert not hasattr(loaded_reflection_data, name)
    assert not callable(loaded_reflection_data)


def test_ded_context_consumes_collection_scaled_views(
    loaded_reflection_data, monkeypatch
):
    """DED preparation uses corrected amplitudes after installing scaled members."""
    from torchref.cli import validate_ded

    dark, light = clone(loaded_reflection_data), clone(loaded_reflection_data)
    light.F *= 4
    light.F_sigma *= 4
    inputs = {"dark": dark, "light": light}
    monkeypatch.setattr(
        validate_ded, "load_reflection_data", lambda path, **kwargs: inputs[path]
    )
    context = validate_ded.setup_ded_context(
        "dark", "light", dmin=2.2, device=get_default_device()
    )
    dc = context["collection"]
    assert context["data_dark"] is dc["dark"]
    assert context["data_light"] is dc["light"]
    torch.testing.assert_close(dc["dark"].F, dc["light"].F, rtol=1e-4, atol=1e-4)
    relative_difference = (
        context["w_dfo"].norm() / dc["dark"].F[context["refl_mask"]].norm()
    )
    assert relative_difference < 1e-5


def test_missing_intensity_access(loaded_reflection_data):
    """Raw and scaled views expose missing columns and reject intensity-only reads."""
    dc = collection(loaded_reflection_data, (1, 2))
    dc["0"].I = dc["0"].I_sigma = None
    raw = dc["0"]
    dc.scale(nsteps=1)
    for data in (raw, dc["0"]):
        with pytest.raises(ValueError, match="No intensities"):
            data.get_corrected_intensities()
        for name in ("I", "sigI", "I_raw", "sigI_raw"):
            assert getattr(data.work, name) is None
    with pytest.raises(ValueError, match="'0'"):
        dc.stack_I_obs()
