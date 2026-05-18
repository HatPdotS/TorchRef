"""
Unit tests for Model.rotate.
"""
import math
from pathlib import Path

import pytest
import torch

from torchref.model import Model


TEST_PDB = Path(__file__).resolve().parents[2] / "files" / "pdb" / "1DAW.pdb"


@pytest.fixture
def loaded_model():
    return Model().load_pdb(str(TEST_PDB))


def _Rz(angle_rad: float) -> torch.Tensor:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.unit
def test_rotate_identity_preserves_coords(loaded_model):
    """Rotation by identity is a no-op (within FP)."""
    rotated = loaded_model.rotate(torch.eye(3))
    delta = (loaded_model.xyz() - rotated.xyz()).abs().max().item()
    assert delta < 1e-5


@pytest.mark.unit
def test_rotate_returns_new_instance(loaded_model):
    """rotate() must NOT mutate the original."""
    xyz_before = loaded_model.xyz().clone()
    rotated = loaded_model.rotate(_Rz(math.radians(30.0)))
    xyz_after = loaded_model.xyz()
    assert (xyz_before - xyz_after).abs().max().item() < 1e-9, \
        "Original model coords should be unchanged"
    # Rotated model has different coords
    delta = (loaded_model.xyz() - rotated.xyz()).abs().max().item()
    assert delta > 1e-3


@pytest.mark.unit
def test_rotate_preserves_centroid(loaded_model):
    """Rotation around centroid (default) preserves the centroid."""
    R = _Rz(math.radians(60.0))
    rotated = loaded_model.rotate(R)
    c_before = loaded_model.xyz().mean(dim=0)
    c_after = rotated.xyz().mean(dim=0)
    assert (c_before - c_after).norm().item() < 1e-4


@pytest.mark.unit
def test_rotate_preserves_pairwise_distances(loaded_model):
    """Rotation is rigid: pairwise distances are preserved."""
    R = _Rz(math.radians(45.0))
    rotated = loaded_model.rotate(R)
    xyz0 = loaded_model.xyz()
    xyz1 = rotated.xyz()
    # Sample 50 random pairs of atoms
    g = torch.Generator().manual_seed(0)
    n = xyz0.shape[0]
    i = torch.randint(0, n, (50,), generator=g)
    j = torch.randint(0, n, (50,), generator=g)
    d0 = (xyz0[i] - xyz0[j]).norm(dim=-1)
    d1 = (xyz1[i] - xyz1[j]).norm(dim=-1)
    assert (d0 - d1).abs().max().item() < 1e-4


@pytest.mark.unit
def test_rotate_two_rotations_compose(loaded_model):
    """rotate(R2) ∘ rotate(R1) ≈ rotate(R2 @ R1) (centroid-preserving)."""
    R1 = _Rz(math.radians(30.0))
    R2 = _Rz(math.radians(50.0))
    composed = loaded_model.rotate(R1).rotate(R2)
    single = loaded_model.rotate(R2 @ R1)
    delta = (composed.xyz() - single.xyz()).abs().max().item()
    assert delta < 1e-4


@pytest.mark.unit
def test_rotate_around_explicit_center(loaded_model):
    """Rotation around an explicit center: that point is fixed."""
    R = _Rz(math.radians(90.0))
    center = torch.tensor([5.0, -3.0, 1.0])
    rotated = loaded_model.rotate(R, center=center)
    # The center, if it were a model atom, would be invariant. We can check by
    # taking any atom and asserting xyz_new = R · (xyz_old − center) + center.
    xyz0 = loaded_model.xyz()
    expected = (xyz0 - center) @ R.T.to(xyz0.dtype) + center
    delta = (rotated.xyz() - expected).abs().max().item()
    assert delta < 1e-4
