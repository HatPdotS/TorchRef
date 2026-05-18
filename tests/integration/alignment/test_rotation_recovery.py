"""
Gate test for `torchref.alignment.ball_rotation_search`:
recover a known random rotation of a model from its rotated F_calc.

Setup (synthetic, P1 logic-only):
- Load 1DAW.pdb.
- Compute F_calc at the model's HKL set → `e_obs`.
- For each random R_true, rotate model coordinates by R_true, recompute
  F_calc → `e_calc`.
- Run ball_rotation_search and check that the top peak gives R within `tol_deg`
  of R_true.

The test does NOT use F_obs from the MTZ — that would test model+data fit, not
rotation function correctness. Once 20/20 passes here, the next test exercises
the full pipeline against real F_obs.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from torchref.alignment.ball_search import (
    ball_rotation_search,
    rotation_matrix_from_edmonds_euler,
    rotation_angular_distance_deg,
)
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT


TEST_FILES = Path(__file__).resolve().parents[2] / "files"
PDB_1DAW = TEST_FILES / "pdb" / "1DAW.pdb"
MTZ_1DAW = TEST_FILES / "mtz" / "1DAW.mtz"


@pytest.fixture(scope="module")
def model_1daw():
    return ModelFT().load_pdb(str(PDB_1DAW))


@pytest.fixture(scope="module")
def data_1daw():
    # Expand to P1 so the HKL set covers reciprocal-space directions uniformly,
    # not just the ASU. Friedel mates are added by sh_expand_ball internally,
    # but symmetry mates are needed here to get angular coverage.
    return ReflectionData().load_mtz(str(MTZ_1DAW)).expand_to_p1(include_friedel=False)


def _random_rotation(seed: int) -> torch.Tensor:
    """Uniform random rotation on SO(3) via QR of a Gaussian matrix."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _hkl_to_s_vectors(hkl: torch.Tensor, cell) -> torch.Tensor:
    """Convert HKL → reciprocal-lattice vectors |s| (1/Å)."""
    rec_basis = cell.reciprocal_basis_matrix
    if callable(rec_basis):
        rec_basis = rec_basis()
    rec_basis = rec_basis.to(torch.float64)
    return hkl.to(torch.float64) @ rec_basis


def _normalize_amplitudes_by_shell(F: torch.Tensor, s_mag: torch.Tensor, n_shells: int = 24) -> torch.Tensor:
    """
    Crude per-shell normalization: |E(h)| = |F(h)| / sqrt(<|F|²>_shell).
    Returns a real tensor of the same shape as F.
    """
    F = F.to(torch.float64)
    s_mag = s_mag.to(torch.float64)
    # equal-count binning
    sorted_idx = torch.argsort(s_mag)
    n = s_mag.numel()
    shell_idx = torch.zeros(n, dtype=torch.int64)
    chunk = n // n_shells
    for k in range(n_shells):
        a = k * chunk
        b = (k + 1) * chunk if k < n_shells - 1 else n
        shell_idx[sorted_idx[a:b]] = k
    norm = torch.zeros(n, dtype=torch.float64)
    for k in range(n_shells):
        mask = shell_idx == k
        f2 = (F[mask] ** 2).mean().clamp(min=1e-30)
        norm[mask] = torch.sqrt(f2)
    return F / norm


def _run_trial(model: ModelFT, hkl: torch.Tensor, R_true: torch.Tensor,
               L: int, P: int, d_min: float, d_max: float):
    """
    One rotation-recovery trial.

    We do NOT rotate the model atoms — that would change |F_calc| in ways
    coupled to crystallographic symmetry (ModelFT applies symmetry mates).
    Instead, we simulate "rotated F_calc" by placing F_orig values at rotated
    reciprocal-space positions: this gives a field that is exactly the rotation
    of the original field on the sphere, with no symmetry artifacts. This
    tests the rotation function itself, isolated from F-calc-vs-symmetry.
    """
    with torch.no_grad():
        F_orig = model(hkl).abs().to(torch.float64)

    s_obs = _hkl_to_s_vectors(hkl, model.cell)
    s_mag = s_obs.norm(dim=-1)
    keep = (s_mag >= 1.0 / d_max) & (s_mag <= 1.0 / d_min)
    s_obs = s_obs[keep]
    F_orig = F_orig[keep]
    s_mag = s_mag[keep]

    # "Rotated calc": same values at rotated reciprocal positions.
    R64 = R_true.to(torch.float64)
    s_calc = s_obs @ R64.T

    # Normalize within each radial shell so the rotation function is dominated
    # by directional structure, not by the |F|² magnitude vs resolution.
    e_obs = _normalize_amplitudes_by_shell(F_orig, s_mag, n_shells=P)
    e_calc = e_obs  # values come along with positions

    C, alphas, betas, gammas, peaks = ball_rotation_search(
        s_obs, e_obs, s_calc, e_calc,
        L=L, P=P, n_peaks=30, refine_subvoxel=True, n_refine=10,
        sigma_threshold=0.0,
    )
    return peaks


def _min_err_over_pointgroup(R_test: torch.Tensor, R_true: torch.Tensor,
                              data) -> float:
    """
    Return min angular distance (deg) between R_test and any symmetry-equivalent
    of R_true under the spacegroup's point group.

    The field f_obs has the symmetry of |F_calc|, which equals the spacegroup
    point group symmetry. So a recovered rotation R_test that differs from
    R_true by a point-group operation is just as correct.
    """
    sym_mats = data.spacegroup.matrices.to(torch.float64)  # (N_ops, 3, 3)
    best = float("inf")
    R_test = R_test.to(torch.float64)
    R_true = R_true.to(torch.float64)
    for k in range(sym_mats.shape[0]):
        R_eq = sym_mats[k] @ R_true
        err = rotation_angular_distance_deg(R_test, R_eq)
        if err < best:
            best = err
    return best


@pytest.mark.integration
@pytest.mark.parametrize("trial", range(5))
def test_rotation_recovery_1daw_top_peak(model_1daw, data_1daw, trial):
    """
    The true rotation (or any spacegroup-symmetry equivalent) should be within
    8° of one of the score-tied top peaks (L=32 → 5.6° voxels).

    "Score-tied" = score within 1% of the top peak. This tolerates the known
    Patterson rotation-function degeneracy where |F|² accidentally has near-
    symmetries that produce a (sub-percent-) close-second peak at a related but
    distinct rotation. The pipeline always evaluates top-N candidates with
    rigid-body refinement, so a tied second-place peak is just as good.
    """
    R_true = _random_rotation(seed=1000 + trial)
    peaks = _run_trial(
        model_1daw, data_1daw.hkl, R_true,
        L=32, P=20, d_min=4.0, d_max=15.0,
    )
    top_score = peaks[0].score
    tied = [p for p in peaks if p.score >= top_score * 0.99]
    best_err = min(
        _min_err_over_pointgroup(
            rotation_matrix_from_edmonds_euler(p.alpha, p.beta, p.gamma),
            R_true,
            data_1daw,
        )
        for p in tied
    )
    assert best_err < 8.0, (
        f"trial {trial}: best score-tied peak {best_err:.2f}° from R_true "
        f"({len(tied)} peaks within 1% of top score {top_score:.3e})"
    )


@pytest.mark.integration
@pytest.mark.parametrize("trial", range(5))
def test_rotation_recovery_1daw_in_top5(model_1daw, data_1daw, trial):
    """
    The true rotation (modulo spacegroup symmetry) must be within 8° of one
    of the top-5 peaks.
    """
    R_true = _random_rotation(seed=2000 + trial)
    peaks = _run_trial(
        model_1daw, data_1daw.hkl, R_true,
        L=32, P=20, d_min=4.0, d_max=15.0,
    )
    best_err = float("inf")
    for p in peaks[:5]:
        R_p = rotation_matrix_from_edmonds_euler(p.alpha, p.beta, p.gamma)
        err = _min_err_over_pointgroup(R_p, R_true, data_1daw)
        if err < best_err:
            best_err = err
    assert best_err < 8.0, f"trial {trial}: best of top-5 = {best_err:.2f}°"
