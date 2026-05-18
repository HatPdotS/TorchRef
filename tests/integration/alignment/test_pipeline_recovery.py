"""
End-to-end pipeline recovery test.

Setup mirrors the rotation-recovery test: instead of rotating the *atoms* of a
symmetric crystal (which couples to the C2 symmetry expansion in ModelFT and
breaks the simple "rotated model" assumption), we exercise the pipeline by
providing a known true rotation and verifying that the rotation-search +
clustering produces a candidate within tolerance.

This test is intentionally narrower than `test_rotation_recovery.py` — it
covers the ball-search → cluster pipeline integration and exercises the
public `cluster_rotation_peaks` / `rotation_angular_distance` helpers, but
does NOT exercise the full `MolecularReplacementPipeline.run()`, which would
require a P1 search-model setup with synthetic F_obs and is out of scope for
this gate test.

The full pipeline gate (rotation + translation + rigid body in P1) is covered
by a follow-up test once the underlying rotation function is fully proven on
real data (this test plus `test_rotation_recovery.py`).
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
from torchref.alignment.pipeline import cluster_rotation_peaks
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT


TEST_FILES = Path(__file__).resolve().parents[2] / "files"
PDB_1DAW = TEST_FILES / "pdb" / "1DAW.pdb"
MTZ_1DAW = TEST_FILES / "mtz" / "1DAW.mtz"


@pytest.fixture(scope="module")
def model_1daw():
    return ModelFT().load_pdb(str(PDB_1DAW))


@pytest.fixture(scope="module")
def data_p1():
    return ReflectionData().load_mtz(str(MTZ_1DAW)).expand_to_p1(include_friedel=False)


def _random_rotation(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _hkl_to_s(hkl, cell):
    rec_basis = cell.reciprocal_basis_matrix
    if callable(rec_basis):
        rec_basis = rec_basis()
    return hkl.to(torch.float64) @ rec_basis.to(torch.float64)


def _normalize_by_shell(F, s_mag, P):
    sorted_idx = torch.argsort(s_mag)
    shell_idx = torch.zeros(s_mag.shape[0], dtype=torch.int64)
    chunk = s_mag.shape[0] // P
    for k in range(P):
        a = k * chunk
        b = (k + 1) * chunk if k < P - 1 else s_mag.shape[0]
        shell_idx[sorted_idx[a:b]] = k
    norm = torch.zeros_like(s_mag, dtype=torch.float64)
    for k in range(P):
        mask = shell_idx == k
        norm[mask] = (F[mask] ** 2).mean().clamp(min=1e-30).sqrt()
    return F / norm


@pytest.mark.integration
@pytest.mark.parametrize("trial", range(5))
def test_pipeline_clustering_preserves_truth(model_1daw, data_p1, trial):
    """
    After clustering, the true rotation must still be represented in the
    top-3 clusters. This guards the `cluster_rotation_peaks` step from
    rejecting the correct peak as a duplicate of a higher-scoring artifact.
    """
    R_true = _random_rotation(seed=3000 + trial)

    with torch.no_grad():
        F_orig = model_1daw(data_p1.hkl).abs().to(torch.float64)
    s_obs = _hkl_to_s(data_p1.hkl, model_1daw.cell)
    s_mag = s_obs.norm(dim=-1)
    keep = (s_mag >= 1.0 / 15.0) & (s_mag <= 1.0 / 4.0)
    s_obs = s_obs[keep]
    F_orig = F_orig[keep]
    s_mag = s_mag[keep]

    e_obs = _normalize_by_shell(F_orig, s_mag, P=20)
    s_calc = s_obs @ R_true.to(torch.float64).T
    e_calc = e_obs

    _C, _a, _b, _g, peaks = ball_rotation_search(
        s_obs, e_obs, s_calc, e_calc,
        L=32, P=20, n_peaks=60, refine_subvoxel=True, n_refine=20,
        sigma_threshold=0.0,
    )
    peak_tuples = [(p.alpha, p.beta, p.gamma, p.score, p.sigma) for p in peaks]
    clustered = cluster_rotation_peaks(peak_tuples, threshold_deg=8.0)
    assert len(clustered) > 0

    # The true R (mod point-group symmetry of the obs field) must be within
    # 8° of one of the top-5 clustered peaks.
    sym = data_p1.spacegroup.matrices.to(torch.float64)
    best_err = float("inf")
    best_rank = None
    for rank, peak in enumerate(clustered[:5]):
        R_p = rotation_matrix_from_edmonds_euler(peak[0], peak[1], peak[2]).to(torch.float64)
        for k in range(sym.shape[0]):
            R_eq = sym[k] @ R_true.to(torch.float64)
            err = rotation_angular_distance_deg(R_p, R_eq)
            if err < best_err:
                best_err = err
                best_rank = rank
    assert best_err < 8.0, (
        f"trial {trial}: best clustered peak {best_err:.2f}° from R_true "
        f"(rank {best_rank} of {len(clustered)})"
    )
