"""Rotation-function peaks are one per orientation, not one per symmetry mate.

The greedy SO(3) suppression treats ``R`` and its point-group mates ``R R_g`` as
the same peak when the Cartesian symmetry rotations are supplied. The group
composes on the **right** -- measured on real peak lists, where the left orbit
finds no coincident pairs and the right orbit finds every mate -- so the test
pins that side too: a left-composed copy must survive as a distinct peak.
"""
import math

import pytest
import torch

from torchref.experimental.alignment.frf.peak_finder import find_rotation_peaks
from torchref.experimental.alignment.frf.rotation_utils import (
    edmonds_euler_from_rotation_matrix,
    rotation_matrix_from_edmonds_euler,
)
from torchref.experimental.alignment.frf.types import AdaptiveRotationFunction

pytestmark = pytest.mark.unit


def _rz(deg: float) -> torch.Tensor:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def _arf(rotations, values) -> AdaptiveRotationFunction:
    eul = torch.tensor([edmonds_euler_from_rotation_matrix(R) for R in rotations],
                       dtype=torch.float64)
    n = eul.shape[0]
    return AdaptiveRotationFunction(
        alphas=eul[:, 0], betas=eul[:, 1], gammas=eul[:, 2],
        values=torch.tensor(values, dtype=torch.float64),
        beta_starts=torch.tensor([0, n]), beta_grid=torch.tensor([0.0]),
        grid_sampling_deg=3.0,
    )


def test_symmetry_mates_collapse_to_one_peak_and_the_side_is_right():
    sym_cart = torch.stack([_rz(0.0), _rz(90.0), _rz(180.0), _rz(270.0)])  # 4 about z
    R1 = rotation_matrix_from_edmonds_euler(0.3, 0.7, 1.1)
    R_right = R1 @ _rz(90.0)            # a mate: the group acts on the right
    R_left = _rz(90.0) @ R1             # not a mate of R1 for a generic R1
    R3 = rotation_matrix_from_edmonds_euler(2.0, 1.3, 0.4)
    arf = _arf([R1, R_right, R_left, R3], [10.0, 9.0, 8.5, 8.0])

    plain = find_rotation_peaks(arf, n_peaks=10, sigma_threshold=-50.0, nms_radius_deg=6.0)
    assert len(plain) == 4, "without symmetry every sample is its own peak"

    dedup = find_rotation_peaks(arf, n_peaks=10, sigma_threshold=-50.0,
                                nms_radius_deg=6.0, sym_cart=sym_cart)
    scores = sorted(p.score for p in dedup)
    assert scores == [8.0, 8.5, 10.0], (
        "the right-composed mate (9.0) must be suppressed and the "
        f"left-composed copy (8.5) kept; got {scores}"
    )
