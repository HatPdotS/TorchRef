"""The MR pipeline works in the data's unit cell, whatever the search model's CRYST1.

A search model from another crystal form (or a predicted model with a
placeholder ``CRYST1 1 1 1``) carries a cell that is not the crystal's. The
translation template, the rotation-only model and the placed model must all be
in the data's cell, or fractional placement and the template's structure
factors are computed in the wrong lattice.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from torchref.experimental.alignment.frf.types import RotationPeak
from torchref.experimental.alignment.pipeline import (
    MolecularReplacementPipeline,
    MRSolution,
)
from torchref.experimental.alignment.translation import prepare_candidate
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT
from torchref.symmetry import Cell

pytestmark = pytest.mark.unit

TEST_FILES = Path(__file__).resolve().parents[2] / "files"
PDB_1DAW = TEST_FILES / "pdb" / "1DAW.pdb"
MTZ_1DAW = TEST_FILES / "mtz" / "1DAW.mtz"


@pytest.fixture(scope="module")
def setup():
    data = ReflectionData().load_mtz(str(MTZ_1DAW))
    model = ModelFT(verbose=0).load_pdb(str(PDB_1DAW))
    a, b, c, al, be, ga = model.cell.key
    model.cell = Cell(
        [a * 1.25, b * 1.1, c * 0.9, al, be, ga],
        dtype=model.dtype_float, device=model.device,
    )
    model.spacegroup = "P 1"
    return data, model


def test_placed_model_is_in_the_data_cell(setup):
    data, model = setup
    pipe = MolecularReplacementPipeline(data, model)
    t = np.array([0.5, 0.0, 0.0])
    sol = MRSolution(
        rotation=np.eye(3), translation=t, rotation_score=0.0,
        translation_score=0.0, r_factor=0.0,
    )
    placed = pipe.place(sol)

    assert placed.cell == data.cell
    assert placed.spacegroup.number == data.spacegroup.number
    shift = (placed.xyz() - model.xyz()).mean(dim=0).cpu().to(torch.float64)
    expected = 0.5 * data.cell.fractional_matrix[:, 0].cpu().to(torch.float64)
    assert torch.allclose(shift, expected, atol=1e-3), (shift, expected)
    assert model.cell != data.cell, "the search model's own cell was overwritten"


def test_rotation_only_model_is_in_the_data_cell(setup):
    data, model = setup
    pipe = MolecularReplacementPipeline(data, model)
    peak = RotationPeak(alpha=0.3, beta=0.7, gamma=1.1, score=1.0, sigma=1.0)
    rotated, _ = pipe._make_rotated(peak)
    assert rotated.cell == data.cell
    assert rotated.spacegroup.number == data.spacegroup.number


def test_translation_template_is_p1_in_the_data_cell(setup):
    data, model = setup
    pipe = MolecularReplacementPipeline(data, model)
    pipe._prepare_translation_arrays()
    assert pipe._p1.cell == data.cell
    assert pipe._p1.spacegroup.number == 1


def test_prepare_candidate_rejects_a_foreign_cell(setup):
    data, model = setup
    pipe = MolecularReplacementPipeline(data, model)
    pipe._prepare_translation_arrays()
    with pytest.raises(ValueError, match="cell"):
        prepare_candidate(model, pipe._obs, data.spacegroup, data.cell)
