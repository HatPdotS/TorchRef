"""Unit tests for the density sigma-cutoff config and the per-atom radius policy.

These replace the old per-structure ``calibrate_radius`` auto-radius tests: the
splat radius is now per-atom (``N_sigma * sigma_eff``) with ``N_sigma`` set by
``torchref.sigma_cutoff_ed``.
"""

import math

import pytest
import torch

import torchref
from torchref.config import SigmaCutoffConfig, get_sigma_cutoff_ed
from torchref.base.electron_density.radius_policy import (
    R_HI,
    R_LO,
    per_atom_radius_aniso,
    per_atom_radius_iso,
    sigma_eff_iso,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_sigma_cutoff_default_and_getter():
    assert torchref.sigma_cutoff_ed.value == get_sigma_cutoff_ed()
    assert torchref.sigma_cutoff_ed.value > 0


def test_sigma_cutoff_set_roundtrip():
    prev = torchref.sigma_cutoff_ed.value
    try:
        torchref.sigma_cutoff_ed.value = 4.25
        assert torchref.sigma_cutoff_ed.value == 4.25
        assert get_sigma_cutoff_ed() == 4.25
    finally:
        torchref.sigma_cutoff_ed.value = prev


@pytest.mark.parametrize("bad", [0, -1.0, "x", None, True])
def test_sigma_cutoff_rejects_invalid(bad):
    cfg = SigmaCutoffConfig()
    with pytest.raises((ValueError, TypeError)):
        cfg.value = bad


def test_sigma_cutoff_env(monkeypatch):
    monkeypatch.setenv("TORCHREF_SIGMA_CUTOFF_ED", "3.0")
    assert SigmaCutoffConfig().value == 3.0
    monkeypatch.setenv("TORCHREF_SIGMA_CUTOFF_ED", "bogus")
    with pytest.raises(ValueError):
        SigmaCutoffConfig()


# ---------------------------------------------------------------------------
# Radius policy
# ---------------------------------------------------------------------------
def _iso_inputs(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    adp = torch.rand(n, generator=g) * 40 + 2
    B_widths = torch.rand(n, 5, generator=g) * 20 + 2
    return adp, B_widths


def test_iso_radius_shape_and_clamp():
    adp, B = _iso_inputs()
    r = per_atom_radius_iso(adp, B, n_sigma=3.5)
    assert r.shape == (adp.shape[0],)
    assert torch.all(r >= R_LO) and torch.all(r <= R_HI)
    # quantized to 0.25
    assert torch.allclose(r, torch.round(r / 0.25) * 0.25)


def test_iso_radius_monotonic_in_B():
    """Larger ADP -> wider density -> larger (or equal, after clamp) radius."""
    B = torch.full((3, 5), 5.0)
    adp = torch.tensor([1.0, 50.0, 150.0])
    r = per_atom_radius_iso(adp, B, n_sigma=3.5)
    assert r[0] <= r[1] <= r[2]


def test_iso_radius_grows_with_nsigma():
    adp, B = _iso_inputs()
    r3 = per_atom_radius_iso(adp, B, n_sigma=3.0)
    r5 = per_atom_radius_iso(adp, B, n_sigma=5.0)
    assert torch.all(r5 >= r3)
    assert r5.sum() > r3.sum()  # not all clamped


def test_sigma_eff_formula():
    adp = torch.tensor([10.0])
    B = torch.tensor([[2.0, 4.0, 8.0, 1.0, 3.0]])  # broadest = 8
    expected = math.sqrt((8.0 + 10.0) / (8 * math.pi**2))
    assert abs(float(sigma_eff_iso(adp, B)) - expected) < 1e-6


def test_aniso_radius_bounding_sphere():
    """Anisotropic radius uses the largest U eigenvalue; clamped + quantized."""
    n = 6
    g = torch.Generator().manual_seed(1)
    B = torch.rand(n, 5, generator=g) * 20 + 2
    u = torch.zeros(n, 6)
    u[:, :3] = torch.rand(n, 3, generator=g) * 0.15 + 0.02  # positive-definite-ish
    r = per_atom_radius_aniso(B, u, n_sigma=3.5)
    assert r.shape == (n,)
    assert torch.all(r >= R_LO) and torch.all(r <= R_HI)
    # a fatter ellipsoid (bigger U) gives a >= radius
    u_big = u.clone(); u_big[:, :3] *= 2.0
    r_big = per_atom_radius_aniso(B, u_big, n_sigma=3.5)
    assert torch.all(r_big >= r)
