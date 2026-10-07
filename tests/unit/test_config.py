"""Unit tests for ``torchref.config``: CUDA auto-selection gates the GPU it returns."""

import types
import warnings

import pytest
import torch

from torchref import config

pytestmark = pytest.mark.unit

_GB = 1024**3


def _fake_gpus(monkeypatch, gpus, current):
    """Present ``gpus`` (``(capability, total_memory)`` per index) as the visible GPUs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_arch_list", lambda: ["sm_70", "sm_80", "sm_90"]
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(gpus))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: gpus[i][0])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: types.SimpleNamespace(total_memory=gpus[i][1]),
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


@pytest.mark.parametrize(
    "gpus",
    [
        [((6, 1), 4 * _GB), ((8, 0), 40 * _GB)],
        [((8, 0), 8 * _GB), ((8, 0), 80 * _GB)],
    ],
    ids=["old-arch-current", "small-vram-current"],
)
def test_unfit_current_gpu_falls_back_although_another_qualifies(monkeypatch, gpus):
    """``cuda`` resolves to the current GPU, so that is the one the gates must pass."""
    _fake_gpus(monkeypatch, gpus, current=0)

    with pytest.warns(UserWarning, match="cuda:0"):
        assert config._auto_detect_device() == torch.device("cpu")


def test_fit_current_gpu_is_selected(monkeypatch):
    """A current GPU that passes the gates is auto-selected without a warning."""
    _fake_gpus(monkeypatch, [((6, 1), 4 * _GB), ((8, 0), 40 * _GB)], current=1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        selected = config._auto_detect_device()

    assert config.canonical_device(selected) == torch.device("cuda", 1)
