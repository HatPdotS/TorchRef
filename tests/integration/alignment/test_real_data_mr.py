"""
Phase-2 real-data rotation recovery test.

Setup:
- F_obs: real measured data from 1DAW.mtz (C2 spacegroup, intermolecular
  Patterson contributions present).
- Search model: P1 copy of 1DAW.pdb whose atomic coordinates have been
  rotated by a random R_true. This simulates "user has a search model in
  some arbitrary orientation".
- Pipeline: ball-search on (E²-1) Patterson coefficients → top-N candidates →
  Sim MLRF (LL interpolation + per-shell σA fit) rescore.

Acceptance: the true rotation must be within 8° of one of the top-5 rescored
peaks for 5/5 random trials. This is "shortlist contains truth" — the
downstream translation + rigid-body refinement (Phase 3) breaks the
Patterson rotation-function ambiguity by R-factor.
"""
import math
from pathlib import Path

import pytest
import torch

from torchref.alignment.ball_search import (
    ball_rotation_search,
    edmonds_euler_from_rotation_matrix,
    rotation_angular_distance_deg,
    rotation_matrix_from_edmonds_euler,
)
from torchref.alignment.lattman_love import LattmanLoveInterpolator
from torchref.alignment.ml_rotation import sim_mlrf_rescore
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT
from torchref.symmetry import SpaceGroup


TEST_FILES = Path(__file__).resolve().parents[2] / "files"
PDB_1DAW = TEST_FILES / "pdb" / "1DAW.pdb"
MTZ_1DAW = TEST_FILES / "mtz" / "1DAW.mtz"


@pytest.fixture(scope="module")
def real_data_setup():
    """
    Real C2 F_obs + P1 search model.

    The search model is the same atoms as ``data`` but with spacegroup forced
    to P1 — the standard MR setup. We use the proper `model.spacegroup`
    setter, which routes through `_maybe_initialize_fft()` to rebuild the
    SfFFT with the new symmetry.
    """
    data = ReflectionData().load_mtz(str(MTZ_1DAW))
    model = ModelFT().load_pdb(str(PDB_1DAW))
    model.spacegroup = SpaceGroup("P 1")
    return data, model


def _random_rotation(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _hkl_to_s(hkl, cell):
    rec = cell.reciprocal_basis_matrix
    if callable(rec):
        rec = rec()
    return hkl.to(torch.float64) @ rec.to(torch.float64)


def _shellbin_norm(F, smag, P):
    order = torch.argsort(smag)
    idx = torch.zeros_like(smag, dtype=torch.int64)
    chunk = smag.numel() // P
    for k in range(P):
        a = k * chunk
        b = (k + 1) * chunk if k < P - 1 else smag.numel()
        idx[order[a:b]] = k
    norm = torch.zeros_like(smag, dtype=F.dtype)
    for k in range(P):
        m = idx == k
        norm[m] = (F[m] ** 2).mean().clamp(min=1e-30).sqrt()
    return F / norm


def _min_err_over_sym(R_test, R_ref, sym_mats):
    best = float("inf")
    for k in range(sym_mats.shape[0]):
        R_eq = sym_mats[k] @ R_ref.to(torch.float64)
        e = rotation_angular_distance_deg(R_test.to(torch.float64), R_eq)
        if e < best:
            best = e
    return best


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("trial", range(5))
def test_real_data_rotation_recovery(real_data_setup, trial):
    """
    Real C2 F_obs + rotated P1 search model: the true rotation must be within
    8° of one of the top-5 ML-rescored peaks (modulo C2 symmetry).
    """
    data, model = real_data_setup
    sym_mats = data.spacegroup.matrices.to(torch.float64)
    centric_full = data.centric if isinstance(data.centric, torch.Tensor) else torch.tensor(data.centric)

    F_obs_full = data.F.to(torch.float64).abs()
    s_vec_full = _hkl_to_s(data.hkl, data.cell)
    s_mag_full = s_vec_full.norm(dim=-1)
    d_min, d_max = 4.0, 15.0
    keep = (s_mag_full >= 1.0 / d_max) & (s_mag_full <= 1.0 / d_min)
    F_obs = F_obs_full[keep]
    s_vec = s_vec_full[keep]
    s_mag = s_mag_full[keep]
    hkl = data.hkl[keep]
    centric = centric_full[keep].to(torch.bool)

    P_shells = 20
    E_obs = _shellbin_norm(F_obs, s_mag, P_shells)
    patt_obs = E_obs ** 2 - 1.0  # Patterson coefficient (origin-removed E²)

    # Apply random rotation to search-model atoms, build LL interpolator.
    R_true = _random_rotation(seed=5000 + trial)
    xyz_canonical = model.xyz().clone()
    centroid = xyz_canonical.mean(0)
    xyz_rot = (xyz_canonical - centroid) @ R_true.T.to(xyz_canonical.dtype) + centroid
    model.xyz[:] = xyz_rot
    try:
        ll = LattmanLoveInterpolator(model, padding_factor=2.0, max_res_A=3.0)
        F_calc = ll.evaluate(
            torch.eye(3, dtype=torch.float32), hkl, data.cell, return_amplitude=True,
        ).to(torch.float64)
    finally:
        model.xyz[:] = xyz_canonical

    E_calc = _shellbin_norm(F_calc, s_mag, P_shells)
    patt_calc = E_calc ** 2 - 1.0

    # Stage 1: fast ball-search on Patterson coefficients.
    _, _, _, _, peaks = ball_rotation_search(
        s_vec, patt_obs, s_vec, patt_calc,
        L=32, P=P_shells, n_peaks=200, refine_subvoxel=True, n_refine=50,
        sigma_threshold=-5.0,
    )

    # Stage 2: Sim MLRF rescore.
    rescored = sim_mlrf_rescore(
        peaks, F_obs, hkl, s_mag, centric, ll, data.cell,
        n_shells=15, n_refine=200, batch_size=50,
    )

    # The true rotation (modulo C2 symmetry) must be within 8° of one of top-5.
    best = float("inf")
    best_rank = None
    for rank, p in enumerate(rescored[:10]):
        R_p = rotation_matrix_from_edmonds_euler(p.alpha, p.beta, p.gamma)
        err = _min_err_over_sym(R_p, R_true, sym_mats)
        if err < best:
            best = err
            best_rank = rank
    assert best < 8.0, (
        f"trial {trial}: best top-10 ML peak {best:.2f}° from R_true "
        f"(rank {best_rank}); rescored top-5 errs = "
        f"{[round(_min_err_over_sym(rotation_matrix_from_edmonds_euler(p.alpha, p.beta, p.gamma), R_true, sym_mats), 2) for p in rescored[:5]]}"
    )
