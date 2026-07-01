"""Tests for superpose_vectors_robust_torch.

Guards the early-return bug where ``return best_matrix`` sat inside the loop,
so the function exited after iteration 0 and ``max_iterations`` was silently
ignored. The return now lives after the loop; the recovered transform must
still align a rotated/translated copy of the reference back onto it.
"""
import numpy as np
import pytest
import torch

from torchref.base.alignment.superposition import (
    apply_transformation,
    superpose_vectors_robust_torch,
)


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )


@pytest.mark.unit
@pytest.mark.parametrize("max_iterations", [1, 10])
def test_recovers_known_transform(max_iterations):
    torch.manual_seed(0)
    ref = torch.rand(40, 3, dtype=torch.float32) * 10.0
    R = _rot_z(0.7)
    t = torch.tensor([3.0, -2.0, 1.5], dtype=torch.float32)
    mov = ref @ R.T + t  # mov is ref rotated+translated

    M = superpose_vectors_robust_torch(ref, mov, max_iterations=max_iterations)
    assert M.shape == (3, 4)

    aligned = apply_transformation(mov, M)
    rmsd = torch.sqrt(torch.mean(torch.sum((aligned - ref) ** 2, dim=1)))
    assert rmsd < 1e-3


@pytest.mark.unit
def test_return_is_outside_loop():
    """With max_iterations>1 the result must be at least as good as a single step."""
    torch.manual_seed(1)
    ref = torch.rand(30, 3, dtype=torch.float32) * 5.0
    mov = ref @ _rot_z(-0.4).T + torch.tensor([1.0, 1.0, -1.0])

    def _rmsd(n):
        M = superpose_vectors_robust_torch(ref, mov, max_iterations=n)
        aligned = apply_transformation(mov, M)
        return torch.sqrt(torch.mean(torch.sum((aligned - ref) ** 2, dim=1)))

    assert _rmsd(10) <= _rmsd(1) + 1e-5
