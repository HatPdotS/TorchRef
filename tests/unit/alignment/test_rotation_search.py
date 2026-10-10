"""The rotation search's public contract: three inputs, and one convention.

``rotation_search(model, data, model_error_A)`` is the whole surface. What a
caller most easily gets wrong is not the arguments but the *sense* of the
returned rotation -- whether to apply ``R`` or ``R.T`` to the coordinates. The
round-trip test below settles that operationally rather than by reading a
docstring: it rotates a model by a known matrix, searches, applies the inverse
of a returned solution, and requires the model back where it started, modulo the
crystal's rotational symmetry.

These run on real data (1DAW, C2) because the search needs a real Patterson;
1DAW is the small fast case.
"""

import math

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.slow]

#: The search is scored on whether truth is inside the candidate window the
#: placement search carries, not on being rank 0.
TOP_N = 20


@pytest.fixture(scope="module")
def case(pdb_dir, mtz_dir):
    from torchref.io.datasets.reflection_data import ReflectionData
    from torchref.model import ModelFT

    pdb, mtz = pdb_dir / "1DAW.pdb", mtz_dir / "1DAW.mtz"
    if not (pdb.exists() and mtz.exists()):
        pytest.skip("1DAW not available")
    model = ModelFT(verbose=0).load_pdb(str(pdb))
    data = ReflectionData(verbose=0).load_mtz(str(mtz))
    return model, data


def _rotation(seed: int) -> torch.Tensor:
    """Haar-uniform SO(3) via QR with the sign correction."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def _angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().cpu().to(torch.float64)
    b = b.detach().cpu().to(torch.float64)
    tr = torch.diagonal(a @ b.T).sum().item()
    return math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))


def _sym_cartesian(data) -> torch.Tensor:
    """Space-group rotations as Cartesian operators, on the CPU.

    ``rotations`` is always CPU float64 while the data's tensors may sit on an
    accelerator, and the Miller-index matrices have to be carried into the
    Cartesian basis before they can be compared with a rotation of coordinates.
    """
    from torchref.experimental.alignment.sh import hkl_symops_to_cartesian

    # `.cpu()` before the widening: the data may sit on an accelerator that
    # cannot hold a float64 tensor at all, and the widening is exact on the host.
    return hkl_symops_to_cartesian(
        data.spacegroup.matrices.cpu().to(torch.float64),
        data.cell.reciprocal_basis_matrix.cpu().to(torch.float64),
    )


def _kabsch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rotation taking ``a`` onto ``b``, both centred. CPU float64."""
    a = a.detach().cpu().to(torch.float64)
    b = b.detach().cpu().to(torch.float64)
    x = a - a.mean(0)
    y = b - b.mean(0)
    u, _, vt = torch.linalg.svd(y.T @ x)
    d = torch.sign(torch.linalg.det(vt.T @ u.T))
    return vt.T @ torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.float64)) @ u.T


def _search_rotated(case, seed, model_error_A=0.8, n_peaks=200):
    from torchref.experimental.alignment import rotation_search

    model, data = case
    R_true = _rotation(seed)
    rotated = model.copy().rotate(
        R_true.to(model.dtype_float), center=model.xyz().mean(0),
    )
    return rotated, data, R_true, rotation_search(
        rotated, data, model_error_A, n_peaks=n_peaks,
    )


def test_returns_the_documented_shapes(case):
    from torchref.experimental.alignment import RotationSolutions

    _, _, _, sol = _search_rotated(case, seed=11, n_peaks=50)
    assert isinstance(sol, RotationSolutions)
    n = len(sol)
    assert n > 0
    assert sol.rotations.shape == (n, 3, 3)
    assert sol.rotations.dtype == torch.float64
    for name in ("scores", "z_scores"):
        assert getattr(sol, name).shape == (n,)
    assert sol.euler_zyz.shape == (n, 3)
    assert sol.lmax > 0 and sol.d_min > 0
    assert sol.model_error_A == pytest.approx(0.8)
    # Best first, on the standardised scale.
    z = sol.z_scores
    assert torch.all(z[:-1] >= z[1:] - 1e-9)


def test_rotations_are_rotations(case):
    _, _, _, sol = _search_rotated(case, seed=11, n_peaks=50)
    R = sol.rotations
    eye = torch.eye(3, dtype=torch.float64).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-9)
    assert torch.allclose(torch.linalg.det(R),
                          torch.ones(len(sol), dtype=torch.float64), atol=1e-9)


def test_euler_and_matrix_agree(case):
    """``euler_zyz`` is the same orientation as ``rotations``, not a variant."""
    from torchref.experimental.alignment.frf.rotation_utils import (
        rotation_matrix_from_edmonds_euler,
    )

    _, _, _, sol = _search_rotated(case, seed=11, n_peaks=20)
    for i in range(min(5, len(sol))):
        a, b, g = sol.euler_zyz[i].tolist()
        assert torch.allclose(rotation_matrix_from_edmonds_euler(a, b, g),
                              sol.rotations[i], atol=1e-12)


def test_a_solution_inverts_the_applied_rotation(case):
    """The convention, settled by algebra: ``rotations[i]`` is ``S . R_true``.

    ``R_true`` is the rotation applied to the model's coordinates to build the
    search model, so a returned solution composed with its inverse must leave a
    symmetry operator behind. That fixes the sense of the returned matrix
    without appealing to the docstring. If it ever inverts, the placement stage
    silently searches the wrong orientation.
    """
    _, data, R_true, sol = _search_rotated(case, seed=11)
    sym_cart = _sym_cartesian(data)
    best = min(
        min(_angle_deg(sol.rotations[i] @ R_true.T, S) for S in sym_cart)
        for i in range(min(TOP_N, len(sol)))
    )
    assert best < 5.0, (
        f"no solution in the top {TOP_N} composes with R_true^-1 to within 5 "
        f"degrees of a symmetry operator; closest was {best:.2f} degrees. "
        f"Either the search failed on 1DAW or the convention flipped."
    )


def test_applying_the_transpose_places_the_model(case):
    """The documented usage, at the level of coordinates.

    ``model.rotate(rotations[i].T)`` is what the docstring tells a caller to do.
    Doing it to the search model must superpose it back onto the unrotated model
    up to a symmetry operation -- checked through the actual rotate() call, so
    the test would catch a mismatch between the docstring and the maths.
    """
    rotated, data, _, sol = _search_rotated(case, seed=11)
    model, _ = case
    reference = model.xyz()
    centre = rotated.xyz().mean(0)
    sym_cart = _sym_cartesian(data)

    best = None
    for i in range(min(TOP_N, len(sol))):
        placed = rotated.copy().rotate(
            sol.rotations[i].T.to(rotated.dtype_float).contiguous(), center=centre,
        )
        residual = _kabsch(placed.xyz(), reference)
        ang = min(_angle_deg(residual, S) for S in sym_cart)
        best = ang if best is None else min(best, ang)
    assert best is not None and best < 5.0, (
        f"applying rotations[i].T did not superpose the model onto the "
        f"unrotated reference for any of the top {TOP_N}; closest residual was "
        f"{best:.2f} degrees from a symmetry operator"
    )


def test_model_error_changes_the_result(case):
    """``model_error_A`` must reach the engine.

    It sets the sigma_A fall-off, so it decides how much the high-resolution
    terms count. The previous entry point accepted a coordinate error and then
    overwrote it with an empirical estimate from the atom count, so the caller's
    value was silently discarded; this asserts it is not.
    """
    _, _, _, tight = _search_rotated(case, seed=11, model_error_A=0.2, n_peaks=20)
    _, _, _, loose = _search_rotated(case, seed=11, model_error_A=2.5, n_peaks=20)
    assert tight.model_error_A != loose.model_error_A
    assert not torch.allclose(tight.z_scores[:5], loose.z_scores[:5], atol=1e-6), (
        "model_error_A did not change the rotation function"
    )


def test_uninitialised_model_is_rejected():
    from torchref.experimental.alignment import rotation_search
    from torchref.model import ModelFT

    with pytest.raises(RuntimeError, match="no coordinates"):
        rotation_search(ModelFT(verbose=0), None, 0.8)
