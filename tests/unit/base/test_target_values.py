"""Compare restraint kernels with host references on deposited Cartesian coordinates."""

import math

import numpy as np
import pytest
import torch

from torchref.base.targets._common import EPS, torsions_from_xyz
from torchref.base.targets.adp import adp_simu_math
from torchref.base.targets.angle import angle_math
from torchref.base.targets.bond import bond_math
from torchref.base.targets.chiral import chiral_math
from torchref.base.targets.planarity import planarity_math
from torchref.base.targets.ramachandran import ramachandran_math
from torchref.base.targets.xray_ls import ls_xray_loss_math
from torchref.config import get_default_device, get_float_dtype, get_int_dtype

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def deposited_model(sample_cif_file):
    """The sample structure (1DAW) as deposited."""
    from torchref.model import Model

    model = Model(verbose=0)
    model.load_cif(str(sample_cif_file))
    return model


@pytest.fixture(scope="module")
def deposited_atoms(deposited_model):
    """Return detached Cartesian coordinates (Å) and isotropic B-factors (Å²)."""
    return (
        deposited_model.xyz().detach().clone(),
        deposited_model.adp().detach().clone(),
    )


def _indices(rows, device):
    return torch.tensor(rows, dtype=get_int_dtype(), device=device)


def _gemmi_dihedrals(xyz: torch.Tensor, rows) -> np.ndarray:
    """``gemmi.calculate_dihedral`` in degrees for each atom quadruple in ``rows``."""
    import gemmi

    host = xyz.detach().cpu().double().numpy()
    return np.degrees(
        [
            gemmi.calculate_dihedral(*(gemmi.Position(*host[i]) for i in row))
            for row in rows
        ]
    )


def _wrapped(degrees: np.ndarray) -> np.ndarray:
    return (degrees + 180.0) % 360.0 - 180.0


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


def test_dihedral_sign_is_iupac() -> None:
    """Viewed along B→C, a far bond turned clockwise from the near bond is positive."""
    c, s = math.cos(math.radians(60.0)), math.sin(math.radians(60.0))
    xyz = torch.tensor(
        [[1.0, 0, 0], [0, 0, 0], [0, 0, 1.0], [c, s, 1.0], [c, -s, 1.0]],
        dtype=get_float_dtype(),
        device=get_default_device(),
    )
    # Looking along +z, +x -> (c, s) is a clockwise turn. Reversing the atom order
    # leaves a dihedral unchanged; the mirror image negates it.
    idx = _indices([[0, 1, 2, 3], [3, 2, 1, 0], [0, 1, 2, 4]], xyz.device)
    torch.testing.assert_close(
        torsions_from_xyz(xyz, idx), xyz.new_tensor([60.0, 60.0, -60.0])
    )


def test_dihedral_matches_gemmi(deposited_atoms) -> None:
    """The eager dihedral is gemmi's, on deposited atoms and on random quadruples."""
    xyz, _ = deposited_atoms
    generator = torch.Generator().manual_seed(0)
    scattered = (3.0 * torch.randn(400, 3, generator=generator)).to(xyz)
    for points in (xyz[:400], scattered):
        rows = [[i, i + 1, i + 2, i + 3] for i in range(len(points) - 3)]
        ours = torsions_from_xyz(points, _indices(rows, points.device))
        delta = _wrapped(ours.cpu().double().numpy() - _gemmi_dihedrals(points, rows))
        assert np.abs(delta).max() < 1e-3


def test_ramachandran_reads_surfaces_at_iupac_phi_psi(deposited_model) -> None:
    """The Ramachandran NLL is the surface interpolated at gemmi's phi and psi.

    The surfaces are tabulated in the IUPAC convention and are not symmetric under
    (phi, psi) -> (-phi, -psi): read at the mirror image, 1DAW scores several times
    higher, which this comparison would not survive.
    """
    restraints = deposited_model.restraints
    xyz = deposited_model.xyz().detach()
    phi_idx, psi_idx = restraints._rama_phi_indices, restraints._rama_psi_indices
    surfaces, kind = restraints._rama_surfaces, restraints._rama_surface_type

    phi = (_gemmi_dihedrals(xyz, phi_idx.tolist()) + 180.0) % 360.0
    psi = (_gemmi_dihedrals(xyz, psi_idx.tolist()) + 180.0) % 360.0
    grid = surfaces.cpu().double().numpy()[kind.cpu().numpy()]
    i0, j0 = np.floor(phi).astype(int) % 360, np.floor(psi).astype(int) % 360
    i1, j1 = (i0 + 1) % 360, (j0 + 1) % 360
    u, v = phi - np.floor(phi), psi - np.floor(psi)
    n = np.arange(len(phi))
    expected = np.sum(
        (1 - u) * (1 - v) * grid[n, i0, j0]
        + (1 - u) * v * grid[n, i0, j1]
        + u * (1 - v) * grid[n, i1, j0]
        + u * v * grid[n, i1, j1]
    )
    torch.testing.assert_close(
        ramachandran_math(xyz, phi_idx, psi_idx, surfaces, kind),
        xyz.new_tensor(expected),
        rtol=1e-5,
        atol=1e-3,
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
