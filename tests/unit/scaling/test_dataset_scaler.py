"""Joint observed-data scaling and live dataset access on deposited reflections."""

import copy
import math

import pytest
import torch

from torchref import DatasetCollection, ReflectionData, ScaledDataset
from torchref.base.targets.dataset_scaling import dataset_scaling_loss
from torchref.scaling import DatasetScaler


@pytest.fixture(scope="module")
def deposited(mtz_dir):
    """Measured 1DAW amplitudes, intensities, uncertainties and partitions."""
    return ReflectionData(device="cpu", verbose=0).load_mtz(str(mtz_dir / "1DAW.mtz"))


def clone(data):
    return data.__select__(torch.arange(len(data), device=data.device))


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
def test_known_scales_are_centered_and_sources_stay_raw(deposited, factors):
    before = deposited.F.clone()
    dc = collection(deposited, factors).scale(nsteps=2)
    expected = torch.tensor(factors, dtype=dc.scaler.raw_parameters.dtype).log()
    expected = expected.mean() - expected
    torch.testing.assert_close(
        dc.scaler.corrections[:, 0], expected, atol=1e-4, rtol=1e-4
    )
    assert dc.scaler.corrections[:, 1:].abs().max() < 1e-4
    assert torch.equal(deposited.F, before)
    assert not hasattr(deposited, "parameters")
    assert not hasattr(deposited, "log_scale")
    assert not hasattr(deposited, "U_aniso")
    for key, _ in dc:
        assert isinstance(dc[key], ScaledDataset)
        torch.testing.assert_close(dc[key].F, dc["0"].F, rtol=1e-4, atol=1e-4)


def test_known_anisotropy_is_recovered(deposited):
    dc = collection(deposited, (1, 1, 1))
    reference = DatasetScaler(dc.datasets)
    truth = torch.tensor(
        [
            [0.3, 0.2, -0.1, 0.1, 0.03, -0.02, 0.05],
            [-0.2, -0.1, 0.2, -0.05, -0.02, 0.01, -0.03],
            [-0.1, -0.1, -0.1, -0.05, -0.01, 0.01, -0.02],
        ],
        dtype=reference.raw_parameters.dtype,
    )
    for i, ds in enumerate(dc.values()):
        distortion = (reference.design(ds.hkl) @ truth[i]).exp()
        ds.F /= distortion
        ds.F_sigma /= distortion
    dc.scale(nsteps=4)
    torch.testing.assert_close(dc.scaler.corrections, truth, atol=2e-3, rtol=2e-3)


def test_live_access_scales_both_sigmas_and_all_entrypoints(deposited):
    dc = collection(deposited, (1, 1)).scale(nsteps=1)
    data = dc["0"]
    with torch.no_grad():
        dc.scaler.raw_parameters[0, 0] = 2 * math.log(2)
    for name, power in [("F", 1), ("F_sigma", 1), ("I", 2), ("I_sigma", 2)]:
        torch.testing.assert_close(
            getattr(data, name), getattr(data, name + "_raw") * 2**power
        )
    torch.testing.assert_close(data.work.F, data.F[data.work.mask])
    torch.testing.assert_close(data.work.sigF, data.F_sigma[data.work.mask])
    torch.testing.assert_close(data.work.sigI, data.I_sigma[data.work.mask])
    torch.testing.assert_close(data.work.sigI_raw, data.I_sigma_raw[data.work.mask])
    torch.testing.assert_close(dc.stack_F_obs()[0], data.F)
    torch.testing.assert_close(dc(mask=False)["0"][1], data.F)
    torch.testing.assert_close(dc.stack_I_sigma()[0], data.I_sigma)
    dc.scaler.requires_grad_(True)
    for _ in range(2):
        dc.scaler.zero_grad()
        data.work.F.sum().backward()
        assert dc.scaler.raw_parameters.grad[1].abs().sum() > 0
    dc.scaler.requires_grad_(False)


def test_selection_copy_alignment_and_independent_collections(deposited):
    a = collection(deposited, (1, 4)).scale(nsteps=1)
    b = collection(deposited, (1, 9)).scale(nsteps=1)
    view = a["0"]
    selected = view.__select__(torch.arange(0, len(view), 3))
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
    a.add_dataset("extra", deposited)
    assert a.scaler is None
    a.scale(nsteps=1)
    assert a.scaler is not scaler
    torch.testing.assert_close(view.F, view.F_raw * 2)


def test_two_dataset_loss_matches_propagated_variance_and_gradients(deposited):
    f = torch.stack((deposited.F[:128], deposited.F[:128] * 1.1)).double()
    sigma = torch.stack((deposited.F_sigma[:128], deposited.F_sigma[:128] * 3)).double()
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


def test_missing_and_invalid_observations_have_finite_gradients(deposited):
    f = torch.stack((deposited.F[:128], deposited.F[:128] * 1.1))
    sigma = torch.stack((deposited.F_sigma[:128], deposited.F_sigma[:128]))
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


def test_free_and_validation_changes_do_not_affect_fit(deposited):
    results = []
    for filler in (3, 900):
        dc = collection(deposited, (1, 2))
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


def test_permutation_and_partial_overlap_chain(deposited):
    n = len(deposited)
    sources = {
        "a": clone(deposited).__select__(torch.arange(n // 2)),
        "b": clone(deposited),
        "c": clone(deposited).__select__(torch.arange(n // 2, n)),
    }
    sources["b"].F *= 2
    sources["b"].F_sigma *= 2
    sources["c"].F *= 4
    sources["c"].F_sigma *= 4
    results = []
    for order in [("a", "b", "c"), ("c", "a", "b")]:
        dc = DatasetCollection(device="cpu", verbose=0)
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
                "a": deposited.__select__(torch.arange(6)),
                "b": deposited.__select__(torch.arange(6)),
            }
        )


def test_checkpoint_and_mtz_export_preserve_observations(deposited, tmp_path):
    import reciprocalspaceship as rs

    dc = collection(deposited, (1, 4)).scale(nsteps=1)
    path = tmp_path / "collection.pt"
    dc.save_state(path)
    restored = DatasetCollection.load_state(path, device="cpu")
    assert restored["0"].scaler is restored["1"].scaler is restored.scaler
    torch.testing.assert_close(restored.stack_F_obs(), dc.stack_F_obs())
    path = tmp_path / "view.pt"
    dc["0"].save_state(path)
    restored_view = ScaledDataset.load_state(path, device="cpu")
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
            getattr(dc["0"], attribute).detach().numpy(),
            rtol=1e-6,
        )


def test_bijvoet_observations_keep_distinct_identities(deposited):
    """Canonical duplicate HKLs retain separate signed observations and sigmas."""
    data = deposited.__select__(torch.arange(256).repeat_interleave(2))
    data.friedel_merged = False
    data.friedel_flags = torch.arange(len(data)) % 2 == 1
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


def test_raw_dataset_excludes_deprecated_interfaces(deposited):
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
        assert not hasattr(deposited, name)
    assert not callable(deposited)


def test_ded_context_consumes_collection_scaled_views(deposited, monkeypatch):
    """DED preparation uses corrected amplitudes after installing scaled members."""
    from torchref.cli import validate_ded

    dark, light = clone(deposited), clone(deposited)
    light.F *= 4
    light.F_sigma *= 4
    inputs = {"dark": dark, "light": light}
    monkeypatch.setattr(
        validate_ded, "load_reflection_data", lambda path, **kwargs: inputs[path]
    )
    context = validate_ded.setup_ded_context("dark", "light", dmin=2.2, device="cpu")
    dc = context["collection"]
    assert context["data_dark"] is dc["dark"]
    assert context["data_light"] is dc["light"]
    torch.testing.assert_close(dc["dark"].F, dc["light"].F, rtol=1e-4, atol=1e-4)
    relative_difference = (
        context["w_dfo"].norm() / dc["dark"].F[context["refl_mask"]].norm()
    )
    assert relative_difference < 1e-5
