"""Compare restraint kernels with host references on deposited Cartesian coordinates."""

import math

import numpy as np
import pytest
import torch

from torchref.base.targets._common import EPS
from torchref.base.targets.adp import adp_simu_math
from torchref.base.targets.angle import angle_math
from torchref.base.targets.bond import bond_math
from torchref.base.targets.chiral import chiral_math
from torchref.base.targets.planarity import planarity_math
from torchref.base.targets.xray_ls import ls_xray_loss_math
from torchref.config import get_default_device, get_float_dtype, get_int_dtype

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def deposited_atoms(sample_cif_file):
    """Return detached Cartesian coordinates (Å) and isotropic B-factors (Å²)."""
    from torchref.model import Model

    model = Model(verbose=0)
    model.load_cif(str(sample_cif_file))
    return model.xyz().detach().clone(), model.adp().detach().clone()


def _indices(rows, device):
    return torch.tensor(rows, dtype=get_int_dtype(), device=device)


def _gaussian_sum(residual, sigma):
    return np.sum(
        0.5 * (residual / sigma) ** 2 + np.log(sigma) + 0.5 * math.log(2 * math.pi)
    )


def test_bond_value(deposited_atoms) -> None:
    """Bond lengths enter a summed Gaussian NLL in Å."""
    xyz, _ = deposited_atoms
    host = xyz[:4].cpu().numpy().astype(np.float64)
    idx = _indices([[0, 1], [2, 3]], xyz.device)
    refs = xyz.new_tensor([1.4, 1.5])
    sigma = xyz.new_tensor([0.1, 0.2])
    # The kernel regularizes squared distance to keep coincident-atom gradients finite.
    distance = np.sqrt(np.sum((host[[0, 2]] - host[[1, 3]]) ** 2, axis=1) + EPS)
    expected = _gaussian_sum(distance - refs.cpu().numpy(), sigma.cpu().numpy())
    torch.testing.assert_close(
        bond_math(xyz, idx, refs, sigma), xyz.new_tensor(expected)
    )


def test_angle_value(deposited_atoms) -> None:
    """Angles and their restraint sigmas enter the NLL in radians."""
    import gemmi

    xyz, _ = deposited_atoms
    positions = [gemmi.Position(*row) for row in xyz[:4].cpu().tolist()]
    angles = np.array(
        [gemmi.calculate_angle(*positions[:3]), gemmi.calculate_angle(*positions[1:4])]
    )
    idx = _indices([[0, 1, 2], [1, 2, 3]], xyz.device)
    refs = xyz.new_tensor([1.8, 2.0])
    sigma = xyz.new_tensor([0.1, 0.2])
    expected = _gaussian_sum(angles - refs.cpu().numpy(), sigma.cpu().numpy())
    torch.testing.assert_close(
        angle_math(xyz, idx, refs, sigma), xyz.new_tensor(expected)
    )


def test_chiral_value(deposited_atoms) -> None:
    """The signed scalar triple product, without a 1/6 factor, sets chirality."""
    xyz, _ = deposited_atoms
    host = xyz[:4].cpu().numpy().astype(np.float64)
    volume = np.linalg.det(host[1:] - host[0])
    idx = _indices([[0, 1, 2, 3]], xyz.device)
    refs = xyz.new_tensor([2.0])
    sigma = xyz.new_tensor([0.5])
    expected = _gaussian_sum(volume - 2.0, 0.5)
    torch.testing.assert_close(
        chiral_math(xyz, idx, refs, sigma), xyz.new_tensor(expected)
    )


def test_planarity_value(deposited_atoms) -> None:
    """The plane penalty sums signed-distance Gaussian NLLs over its atoms."""
    xyz, _ = deposited_atoms
    host = xyz[:5].cpu().numpy().astype(np.float64)
    centered = host - host.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    distances = centered @ vh[-1]
    idx = _indices([[0, 1, 2, 3, 4]], xyz.device)
    sigma = xyz.new_full((1, 5), 0.2)
    expected = _gaussian_sum(distances, 0.2)
    torch.testing.assert_close(
        planarity_math(xyz, [(idx, sigma)]), xyz.new_tensor(expected)
    )


def test_simu_value(deposited_atoms) -> None:
    """SIMU penalizes differences of deposited isotropic B-factors in Å²."""
    _, b = deposited_atoms
    host = b[:4].cpu().numpy().astype(np.float64)
    idx = _indices([[0, 1], [2, 3]], b.device)
    expected = _gaussian_sum(host[[0, 2]] - host[[1, 3]], 2.0)
    torch.testing.assert_close(
        adp_simu_math(b, idx, b.new_tensor(2.0)), b.new_tensor(expected)
    )


@pytest.mark.parametrize("weighting, expected", [("sigma", 6.5), ("unit", 20.0)])
def test_least_squares_value_and_mask(weighting: str, expected: float) -> None:
    """Least squares sums half squared amplitude errors using the selected weights."""
    obs = torch.tensor(
        [10.0, 20.0, 30.0], dtype=get_float_dtype(), device=get_default_device()
    )
    calc = -obs - obs.new_tensor([2.0, 6.0, 50.0])
    sigma = obs.new_tensor([1.0, 2.0, 5.0])
    mask = torch.tensor([True, True, False], device=obs.device)
    loss = ls_xray_loss_math(obs, calc, sigma, mask, weighting=weighting)
    torch.testing.assert_close(loss, obs.new_tensor(expected))
    torch.testing.assert_close(
        ls_xray_loss_math(obs, obs, sigma, weighting=weighting), obs.new_zeros(())
    )
