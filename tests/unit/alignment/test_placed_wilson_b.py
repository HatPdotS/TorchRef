"""The placed model leaves the pipeline at the data's Wilson B level."""
import pytest
import torch

from torchref.config import get_float_dtype

pytestmark = pytest.mark.alignment


@pytest.fixture(scope="module")
def placed_and_data(pdb_dir, mtz_dir):
    from torchref.experimental.alignment import MolecularReplacementPipeline
    from torchref.io.datasets.reflection_data import ReflectionData
    from torchref.model import ModelFT

    pdb, mtz = pdb_dir / "1DAW.pdb", mtz_dir / "1DAW.mtz"
    if not (pdb.exists() and mtz.exists()):
        pytest.skip("1DAW not available")
    data = ReflectionData(verbose=0).load_mtz(str(mtz))
    search = ModelFT(verbose=0).load_pdb(str(pdb))
    search.spacegroup = "P 1"
    with torch.no_grad():
        # A confidence-style B column: flat and far below the data's level.
        search.adp.set(torch.full_like(search.adp(), 6.0), torch.isfinite(search.adp()))
    pipe = MolecularReplacementPipeline(data, search, d_min=4.0, d_max=15.0, n_shells=20,
                                        n_rotation_candidates=2)
    sol = pipe.run(do_translation=True)[0]
    return sol.model, data, search


def _relative_wilson_b(model, data):
    from torchref.experimental.alignment.frf.preprocessing import fit_relative_wilson_b

    real = get_float_dtype()
    valid = data.get_valid_mask()
    hkl = data.hkl[valid]
    with torch.no_grad():
        F_calc = model(hkl.to(model.xyz().device)).abs()
    s_mag = (hkl.to(real) @ data.cell.reciprocal_basis_matrix.to(dtype=real, device=hkl.device)).norm(dim=-1)
    return fit_relative_wilson_b(data.F[valid].to(F_calc.device), F_calc, s_mag.to(F_calc.device))


def test_placed_model_sits_at_the_data_wilson_b(placed_and_data):
    placed, data, search = placed_and_data
    shift = placed.last_alignment_wilson_b_shift
    assert shift > 5.0, "a 6 A^2 model against real data needs a positive shift"
    assert abs(_relative_wilson_b(placed, data)) < 3.0
    assert torch.allclose(placed.adp() - search.adp(), torch.full_like(search.adp(), shift), atol=1e-2)


def test_search_model_b_is_untouched(placed_and_data):
    _, _, search = placed_and_data
    assert torch.allclose(search.adp(), torch.full_like(search.adp(), 6.0))
