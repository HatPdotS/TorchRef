"""
Integration test for ``align_model_to_data``: end-to-end rotation search
returning a re-oriented ModelFT.

Setup:
- F_obs: real 1DAW.mtz at its native C2 spacegroup.
- Search model: P1 copy of 1DAW.pdb whose atomic coordinates have been
  rotated by a random R_true.

Acceptance: after ``align_model_to_data``, the returned model's atom coordinates are
within 8° rotation distance (modulo C2 symmetry of F_obs) of the un-rotated
canonical orientation, for 5/5 random trials.
"""
from pathlib import Path

import pytest
import torch

from torchref.experimental.alignment import align_model_to_data
from torchref.experimental.alignment.frf.rotation_utils import rotation_angular_distance_deg
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT


TEST_FILES = Path(__file__).resolve().parents[2] / "files"
PDB_1DAW = TEST_FILES / "pdb" / "1DAW.pdb"
MTZ_1DAW = TEST_FILES / "mtz" / "1DAW.mtz"


def _load_p1_search_model() -> ModelFT:
    """Load 1DAW and force spacegroup to P1 via the proper setter."""
    m = ModelFT().load_pdb(str(PDB_1DAW))
    m.spacegroup = "P 1"
    return m


@pytest.fixture(scope="module")
def real_setup():
    """Real C2 F_obs + P1 search model factory."""
    data = ReflectionData().load_mtz(str(MTZ_1DAW))
    return data, _load_p1_search_model


def _random_rotation(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _best_alignment_rotation(xyz_a: torch.Tensor, xyz_b: torch.Tensor) -> torch.Tensor:
    """
    Kabsch: return R that minimises ||xyz_a - xyz_b @ R.T|| (with both centred).
    Used here to recover the effective rotation between two atom sets.
    """
    a = xyz_a - xyz_a.mean(0)
    b = xyz_b - xyz_b.mean(0)
    H = b.T @ a
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=H.dtype))
    R = Vt.T @ D @ U.T
    return R


def _cartesian_symops(data) -> torch.Tensor:
    """Point-group rotations as Cartesian matrices, ``B S B^-1``.

    ``spacegroup.matrices`` act on fractional coordinates. A Kabsch rotation is
    Cartesian, and comparing the two directly is only right when the cell is
    orthogonal and the operator diagonal -- in a trigonal cell two of the six
    mates of a correct placement read as 30 and 21 degrees.
    """
    B = data.cell.fractional_matrix.to(torch.float64)
    S = data.spacegroup.matrices.to(torch.float64)
    return B @ S @ torch.linalg.inv(B)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("trial", range(5))
def test_fit_to_data_real_1daw(real_setup, trial):
    """
    Apply a random R_true to a P1 search model, align it to the real data, and
    verify the returned model is within 8° rotation distance of the
    canonical orientation (modulo C2 symmetry of F_obs).
    """
    data, make_model = real_setup
    sym_mats = _cartesian_symops(data)

    canonical = make_model()
    xyz_canonical = canonical.xyz().clone()
    centroid = xyz_canonical.mean(0)

    R_true = _random_rotation(seed=5000 + trial)
    search = canonical.rotate(R_true.to(canonical.dtype_float), center=centroid)

    aligned = align_model_to_data(
        search,
        data,
        d_min=4.0, d_max=15.0,
        n_shells=20,
        n_rotation_peaks=200,
        do_translation=False,  # this test only checks rotation accuracy
        verbose=0,
    )

    # The effective rotation between aligned.xyz() and xyz_canonical should
    # be a C2 symmetry operator (i.e. nearly identity or 2-fold along b).
    R_residual = _best_alignment_rotation(
        aligned.xyz().to(torch.float64), xyz_canonical.to(torch.float64),
    )
    # Compare R_residual to identity / each C2 op.
    best_err = float("inf")
    for k in range(sym_mats.shape[0]):
        err = rotation_angular_distance_deg(R_residual, sym_mats[k])
        if err < best_err:
            best_err = err
    assert best_err < 8.0, (
        f"trial {trial}: aligned model is {best_err:.2f}° from a C2-equivalent "
        f"canonical orientation"
    )
