"""The observation set the rotation search hands its engine.

Reflections the data flag invalid (``ReflectionData.masks()``) stay out of both
the anisotropy fit and the rotation function, as they stay out of the
translation search. The anisotropy correction is applied to ``F_obs`` and
``sigma(F_obs)`` alike, so ``F/sigma`` -- which sets the inverse-variance
observation weight -- is the data's own.
"""

import importlib
from pathlib import Path

import pytest
import torch

from torchref.config import get_complex_dtype, get_float_dtype
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT

pytestmark = pytest.mark.unit

rs = importlib.import_module("torchref.experimental.alignment.rotation_search")

TEST_FILES = Path(__file__).resolve().parents[2] / "files"
PDB_1DAW = TEST_FILES / "pdb" / "1DAW.pdb"
MTZ_1DAW = TEST_FILES / "mtz" / "1DAW.mtz"


class _Stop(Exception):
    pass


@pytest.fixture
def data():
    d = ReflectionData().load_mtz(str(MTZ_1DAW))
    invalid = torch.zeros(d.hkl.shape[0], dtype=torch.bool, device=d.hkl.device)
    invalid[::7] = True
    d.masks["test_invalid"] = ~invalid
    return d


def _window(data, d_min, d_max):
    real = get_float_dtype()
    rec = data.cell.reciprocal_basis_matrix.to(real)
    s = (data.hkl.to(real) @ rec).norm(dim=-1)
    return (s >= 1.0 / d_max) & (s <= 1.0 / d_min)


def test_anisotropy_fit_skips_invalid_reflections(data, monkeypatch):
    seen = {}

    def fake_fit(F_obs, *args, **kwargs):
        seen["n"] = int(F_obs.shape[0])
        return torch.zeros(3, 3, dtype=F_obs.dtype, device=F_obs.device)

    monkeypatch.setattr(rs, "fit_overall_anisotropy", fake_fit)
    rs.fit_anisotropy(data, d_min=4.0, d_max=15.0)

    expected = _window(data, 4.0, 15.0) & data.masks().to(torch.bool)
    assert seen["n"] == int(expected.sum())


def test_rotation_function_sees_valid_rows_and_corrected_sigmas(data, monkeypatch):
    model = ModelFT(verbose=0).load_pdb(str(PDB_1DAW))
    U = rs.fit_anisotropy(data, d_min=4.0, d_max=15.0)
    assert float(U.abs().max()) > 0.0, "1DAW should carry some anisotropy"
    seen = {}

    class FakeEngine:
        def __init__(self, s_obs, F_obs, centric, sym_mats, **kwargs):
            seen.update(F_obs=F_obs, sig_F=kwargs["sig_F_obs"], **kwargs)
            raise _Stop

    def fake_dense(*args, **kwargs):
        return torch.zeros(1, 3), torch.zeros(1, dtype=get_complex_dtype())

    monkeypatch.setattr(
        "torchref.experimental.alignment.frf.api.FastRotationFunction", FakeEngine
    )
    monkeypatch.setattr(
        "torchref.experimental.alignment.frf.dense_calc.dense_calc_via_box", fake_dense
    )
    with pytest.raises(_Stop):
        rs.search_peaks(model, data, 0.8, U_aniso=U, n_peaks=5)

    keep = _window(data, seen["d_min"], seen["d_max"]) & data.masks().to(torch.bool)
    assert seen["F_obs"].shape[0] == int(keep.sum())
    ratio_raw = data.F[keep].abs() / data.F_sigma[keep]
    ratio_seen = seen["F_obs"] / seen["sig_F"]
    assert torch.allclose(
        ratio_seen.cpu(), ratio_raw.cpu().to(ratio_seen.dtype), rtol=1e-4
    )
