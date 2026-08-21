"""
Unit tests for `amplitude_translation_search`.

The function does a coarse-grid Pearson correlation between |F_obs|² and
|F_calc(h, t)|² over fractional translations. With the search model placed at
canonical positions and `F_obs` derived from a translated copy of the same
model, the top correlation peak (or one of the top-3) must land at `-t_true`
modulo an allowed origin shift of the spacegroup — i.e. the translation that
would bring the search model into agreement with the observed data.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from torchref.experimental.alignment.translation import amplitude_translation_search
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT


TEST_FILES = Path(__file__).resolve().parents[2] / "files"
PDB_1DAW = TEST_FILES / "pdb" / "1DAW.pdb"
MTZ_1DAW = TEST_FILES / "mtz" / "1DAW.mtz"


class _ModelEvaluator:
    """Thin evaluator: returns model_p1(hkl) at integer HKL."""

    def __init__(self, model_p1):
        self._model = model_p1
        self.device = model_p1.xyz().device

    def evaluate(self, R, hkl, real_cell, return_amplitude=False):
        hkl_int = hkl.round().to(torch.int64).to(self.device)
        with torch.no_grad():
            f = self._model(hkl_int)
        return f.abs() if return_amplitude else f


def _wrap_frac(t: np.ndarray) -> np.ndarray:
    return (t + 0.5) % 1.0 - 0.5


@pytest.fixture(scope="module")
def setup():
    canonical = ModelFT().load_pdb(str(PDB_1DAW))
    data = ReflectionData().load_mtz(str(MTZ_1DAW))
    mask = data.get_valid_mask()
    return canonical, data, mask


@pytest.mark.unit
@pytest.mark.slow
def test_amplitude_tf_zero_translation(setup):
    """Un-translated model: top-1 peak at the origin (modulo C-centering)."""
    canonical, data, mask = setup
    with torch.no_grad():
        F_obs = canonical(data.hkl[mask]).abs().to(torch.float64)
    model_p1 = canonical.copy()
    model_p1.spacegroup = "P 1"
    evaluator = _ModelEvaluator(model_p1)

    R_id = torch.eye(3, dtype=torch.float64)
    _, _, peaks = amplitude_translation_search(
        F_obs=F_obs, interpolator=evaluator, R_rotation=R_id,
        hkl=data.hkl[mask], spacegroup=data.spacegroup, real_cell=data.cell,
        grid_steps=12, n_peaks=10, cluster_radius=0.05,
    )
    assert len(peaks) > 0
    # C2 + C-centering allowed origins: (0, *, 0) and (1/2, *, 1/2)
    def origin_dist(t):
        xz0 = np.linalg.norm(_wrap_frac(np.array([t[0], t[2]])))
        xz1 = np.linalg.norm(_wrap_frac(np.array([t[0] - 0.5, t[2] - 0.5])))
        return min(xz0, xz1)
    best_dist = min(origin_dist(p.translation) for p in peaks[:3])
    assert best_dist < 0.10, (
        f"top-3 peaks miss origin-equivalent by {best_dist:.3f}; "
        f"peaks: {[p.translation.tolist() for p in peaks[:3]]}"
    )


@pytest.mark.unit
@pytest.mark.slow
def test_amplitude_tf_recovers_known_translation(setup):
    """A model translated by t_true: top-3 peaks include -t_true (mod origins)."""
    canonical, data, mask = setup
    t_true = np.array([0.18, -0.07, 0.23])
    # F_obs from canonical (un-translated); search model is canonical_p1
    # translated by t_true (so the recovered TF peak should be at -t_true mod
    # the allowed origin shifts).
    with torch.no_grad():
        F_obs = canonical(data.hkl[mask]).abs().to(torch.float64)
    model_p1 = canonical.copy()
    model_p1.spacegroup = "P 1"
    model_p1 = model_p1.translate(
        torch.tensor(t_true, dtype=canonical.dtype_float), fractional=True,
    )
    evaluator = _ModelEvaluator(model_p1)

    R_id = torch.eye(3, dtype=torch.float64)
    _, _, peaks = amplitude_translation_search(
        F_obs=F_obs, interpolator=evaluator, R_rotation=R_id,
        hkl=data.hkl[mask], spacegroup=data.spacegroup, real_cell=data.cell,
        grid_steps=12, n_peaks=10, cluster_radius=0.05,
    )
    assert len(peaks) > 0
    # Expected: t_peak ≡ -t_true (mod allowed origin). C2 allowed origins
    # along x and z: (0, *, 0) and (1/2, *, 1/2). y is polar.
    def xz_dist(t):
        d_origin = np.linalg.norm(
            _wrap_frac(np.array([t[0] + t_true[0], t[2] + t_true[2]]))
        )
        d_cshift = np.linalg.norm(
            _wrap_frac(np.array([t[0] + t_true[0] - 0.5, t[2] + t_true[2] - 0.5]))
        )
        return min(d_origin, d_cshift)
    best_dist = min(xz_dist(p.translation) for p in peaks[:3])
    assert best_dist < 0.10, (
        f"top-3 peaks don't bracket -t_true (best xz_dist {best_dist:.3f}); "
        f"peaks: {[p.translation.tolist() for p in peaks[:3]]}"
    )
