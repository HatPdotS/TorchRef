"""Unit tests for the fast translation function.

``fast_translation_function`` accumulates the Crowther-Blow coefficients of a
normalised, weighted ``E_obs^2`` against the candidate's normalised
``|E_calc(h, t)|^2`` and inverts one FFT. With the search model at canonical
positions and ``F_obs`` derived from a translated copy of the same model, the
top peak (or one of the top three) must land at ``-t_true`` modulo an allowed
origin shift of the space group -- the translation that would bring the search
model into agreement with the observed data -- and the likelihood must prefer
that peak.

``TranslationObs`` carries the observed side. It is built once here, as the
pipeline builds it once per run, because normalisation and weighting are
properties of the observations and do not change when the model moves.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from torchref.experimental.alignment.translation import (
    TranslationObs,
    fast_translation_function,
    llg_at_translations,
    prepare_candidate,
)
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT


TEST_FILES = Path(__file__).resolve().parents[2] / "files"
PDB_1DAW = TEST_FILES / "pdb" / "1DAW.pdb"
MTZ_1DAW = TEST_FILES / "mtz" / "1DAW.mtz"


@pytest.fixture(scope="module")
def setup():
    canonical = ModelFT().load_pdb(str(PDB_1DAW))
    data = ReflectionData().load_mtz(str(MTZ_1DAW))
    # The pipeline's default window: the rotation search's 15-4 A.
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64)
    s = (data.hkl.to(torch.float64) @ rec).norm(dim=-1)
    mask = data.get_valid_mask() & (s >= 1.0 / 15.0) & (s <= 1.0 / 4.0)
    return canonical, data, mask


def _search(canonical, data, mask, t_true):
    """Peaks and their likelihoods for a search model translated by ``t_true``."""
    with torch.no_grad():
        F_obs = canonical(data.hkl[mask]).abs()
    model_p1 = canonical.copy()
    model_p1.max_res = 4.0 / 1.5
    model_p1.spacegroup = "P 1"
    if t_true is not None:
        model_p1 = model_p1.translate(
            torch.tensor(t_true, dtype=canonical.dtype_float), fractional=True,
        )
    obs = TranslationObs.build(F_obs, data.hkl[mask], data.spacegroup, data.cell)
    cand = prepare_candidate(model_p1, obs, data.spacegroup, data.cell)
    _, peaks = fast_translation_function(
        obs, cand, data.cell, grid_spacing_A=4.0 / 3.0, n_peaks=3,
        cluster_radius_A=4.0,
    )
    assert len(peaks) > 0
    llg = llg_at_translations(
        obs, cand,
        torch.as_tensor(np.stack([p.translation for p in peaks]), dtype=torch.float64),
    )
    return peaks, llg


def _xz_dist_to_origin_class(t: np.ndarray, t_true: np.ndarray) -> float:
    """Distance of ``t + t_true`` from an allowed origin in C2, x and z only.

    C2's origin is free along y; the centring makes (1/2, 1/2, 0) a lattice
    vector and (0, *, 1/2) an allowed shift, so x and z are each determined
    only modulo 1/2.
    """
    d = np.array([t[0] + t_true[0], t[2] + t_true[2]])
    d = (d + 0.25) % 0.5 - 0.25
    return float(np.linalg.norm(d))


@pytest.mark.unit
@pytest.mark.slow
def test_fast_tf_zero_translation(setup):
    """Un-translated model: the likelihood's pick sits at an origin-equivalent."""
    canonical, data, mask = setup
    peaks, llg = _search(canonical, data, mask, None)
    best = peaks[int(llg.argmax())]
    dist = _xz_dist_to_origin_class(best.translation, np.zeros(3))
    assert dist < 0.03, (
        f"likelihood pick misses an origin-equivalent by {dist:.3f}; "
        f"peaks: {[p.translation.round(3).tolist() for p in peaks]}"
    )


@pytest.mark.unit
@pytest.mark.slow
def test_fast_tf_recovers_known_translation(setup):
    """A model translated by t_true: the likelihood's pick is at -t_true (mod origins)."""
    canonical, data, mask = setup
    t_true = np.array([0.18, -0.07, 0.23])
    peaks, llg = _search(canonical, data, mask, t_true)
    best = peaks[int(llg.argmax())]
    dist = _xz_dist_to_origin_class(best.translation, t_true)
    assert dist < 0.03, (
        f"likelihood pick misses -t_true by {dist:.3f}; "
        f"peaks: {[p.translation.round(3).tolist() for p in peaks]}"
    )
    # The fast map's own top peak should already be the right one here; the
    # likelihood is the arbiter when it is not.
    assert _xz_dist_to_origin_class(peaks[0].translation, t_true) < 0.03


@pytest.mark.unit
@pytest.mark.slow
def test_e_calc_is_normalised(setup):
    """``<E_calc^2>`` is one to within the fit's tolerance, for a placed candidate."""
    canonical, data, mask = setup
    with torch.no_grad():
        F_obs = canonical(data.hkl[mask]).abs()
    model_p1 = canonical.copy()
    model_p1.max_res = 4.0 / 1.5
    model_p1.spacegroup = "P 1"
    obs = TranslationObs.build(F_obs, data.hkl[mask], data.spacegroup, data.cell)
    cand = prepare_candidate(model_p1, obs, data.spacegroup, data.cell)
    # The normalisation already carries eps: E is per unit of eps*Sigma_calc.
    E2 = cand.e_calc(torch.zeros(3, dtype=torch.float64)) ** 2
    mean_e2 = float(E2.mean())
    assert abs(mean_e2 - 1.0) < 0.15, mean_e2
