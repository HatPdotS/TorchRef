"""Local-frame placement: exact inverse, exact gradients, and a safe fallback.

A riding hydrogen is stored as an offset in a frame built from three atoms. What has
to hold is that the offset round-trips through placement exactly, that a force on the
placed point reaches only the three frame atoms, and that a frame the geometry cannot
define is recognised rather than used.
"""

import pytest
import torch

from torchref.base.coordinates.local_frame import (
    frame_is_degenerate,
    local_frame_coordinates,
    place_local_frame,
)


def _frames(n: int, dtype=torch.float64):
    generator = torch.Generator().manual_seed(7)
    p = torch.rand(n, 3, generator=generator, dtype=dtype) * 10
    n1 = p + torch.randn(n, 3, generator=generator, dtype=dtype)
    n2 = p + torch.randn(n, 3, generator=generator, dtype=dtype)
    point = p + torch.randn(n, 3, generator=generator, dtype=dtype)
    return p, n1, n2, point


@pytest.mark.unit
def test_local_coordinates_invert_placement():
    """A point expressed in its frame and placed again lands where it started."""
    p, n1, n2, point = _frames(64)
    valid = torch.ones(64, dtype=torch.bool)
    local = local_frame_coordinates(p, n1, n2, point)
    back = place_local_frame(p, n1, n2, local, valid, point - p)
    assert torch.allclose(back, point, atol=1e-12)


@pytest.mark.unit
def test_offset_is_invariant_under_rigid_motion():
    """Rotating and translating the three frame atoms carries the point along."""
    p, n1, n2, point = _frames(16)
    local = local_frame_coordinates(p, n1, n2, point)
    angle = torch.tensor(0.7, dtype=torch.float64)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    shift = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    moved = [x @ rotation.T + shift for x in (p, n1, n2, point)]
    valid = torch.ones(16, dtype=torch.bool)
    placed = place_local_frame(moved[0], moved[1], moved[2], local, valid, point - p)
    assert torch.allclose(placed, moved[3], atol=1e-12)


@pytest.mark.unit
def test_gradients_are_exact_and_reach_only_the_frame_atoms():
    """Autograd through the frame matches finite differences."""
    p, n1, n2, point = _frames(6)
    local = local_frame_coordinates(p, n1, n2, point)
    valid = torch.ones(6, dtype=torch.bool)
    rigid = point - p

    def place(pp, a, b):
        return place_local_frame(pp, a, b, local, valid, rigid)

    leaves = tuple(x.clone().requires_grad_() for x in (p, n1, n2))
    assert torch.autograd.gradcheck(place, leaves, eps=1e-6, atol=1e-6)


@pytest.mark.unit
def test_invalid_frames_fall_back_to_rigid_translation():
    """Where the frame is flagged invalid the point simply follows its parent."""
    p, n1, n2, point = _frames(8)
    local = torch.zeros(8, 3, dtype=torch.float64)
    valid = torch.zeros(8, dtype=torch.bool)
    rigid = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(8, 3)
    placed = place_local_frame(p, n1, n2, local, valid, rigid)
    assert torch.allclose(placed, p + rigid)


@pytest.mark.unit
def test_degenerate_frames_are_detected():
    """Collinear or collapsed reference bonds are flagged, healthy frames are not."""
    p = torch.zeros(3, 3, dtype=torch.float64)
    n1 = torch.tensor([[1.0, 0.0, 0.0]] * 3, dtype=torch.float64)
    n2 = torch.tensor(
        [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [1e-5, 0.0, 0.0]], dtype=torch.float64
    )
    flagged = frame_is_degenerate(p, n1, n2)
    assert flagged.tolist() == [False, True, True]
