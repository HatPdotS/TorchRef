"""The ``hydrogens_in_xray`` flag: what it gates, and where it has to survive.

It decides only whether hydrogen rows reach the structure-factor gathers. Restraints
never consult it, the solvent mask never includes hydrogens, and the setting must
follow the model through copies, selections, strips and state dicts.
"""

import numpy as np
import pytest
import torch

from torchref.model.context import ModelContext
from torchref.model.model import Model
from torchref.model.model_ft import ModelFT


@pytest.fixture(scope="module")
def with_hydrogens(pdb_dir):
    """7L84 keeps its deposited hydrogens."""
    model = Model(verbose=0)
    model.load_pdb(str(pdb_dir / "7L84.pdb"))
    return model


def _n_h(model):
    return int((model.pdb["element"].str.strip() == "H").sum())


@pytest.mark.unit
def test_default_is_on():
    assert ModelContext().hydrogens_in_xray is True
    assert Model(verbose=0).hydrogens_in_xray is True


@pytest.mark.unit
def test_partition_covers_hydrogens_only_when_on(with_hydrogens):
    """Off drops exactly the hydrogen rows from the isotropic gather."""
    model = with_hydrogens
    n_atoms, n_h = len(model.pdb), _n_h(model)
    assert n_h > 0

    def n_in_fcalc():
        return model.get_iso()[0].shape[0] + model.get_aniso()[0].shape[0]

    model.hydrogens_in_xray = True
    assert n_in_fcalc() == n_atoms
    model.hydrogens_in_xray = False
    assert n_in_fcalc() == n_atoms - n_h
    model.hydrogens_in_xray = True
    assert n_in_fcalc() == n_atoms


@pytest.mark.unit
def test_off_matches_a_stripped_model(pdb_dir):
    """Excluding hydrogens from Fcalc equals computing Fcalc without them."""
    full = ModelFT(verbose=0, max_res=2.5, hydrogens_in_xray=False)
    full.load_pdb(str(pdb_dir / "7L84.pdb"))
    heavy = ModelFT(verbose=0, max_res=2.5, strip_H=True)
    heavy.load_pdb(str(pdb_dir / "7L84.pdb"))
    grid = torch.arange(-3, 4)
    hkl = torch.cartesian_prod(grid, grid, grid)
    hkl = hkl[(hkl != 0).any(dim=1)].to(full.device)
    with torch.no_grad():
        f_full = full(hkl)
        f_heavy = heavy(hkl.to(heavy.device))
    assert torch.allclose(f_full, f_heavy, rtol=1e-4, atol=1e-3)


@pytest.mark.unit
def test_setting_survives_copy_select_and_strip(with_hydrogens):
    model = with_hydrogens.copy()
    model.hydrogens_in_xray = False
    assert model.copy().hydrogens_in_xray is False
    assert model.select("chain A").hydrogens_in_xray is False
    assert model.strip_hydrogens().hydrogens_in_xray is False


@pytest.mark.unit
@pytest.mark.parametrize("model_class", [Model, ModelFT])
def test_setting_round_trips_through_state_dict(pdb_dir, model_class):
    kwargs = {"max_res": 3.0} if model_class is ModelFT else {}
    model = model_class(verbose=0, hydrogens_in_xray=False, **kwargs)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    state = model.state_dict()
    restored = model_class.create_from_state_dict(state, device=model.device)
    assert restored.hydrogens_in_xray is False
    state.pop("hydrogens_in_xray", None)
    legacy = model_class.create_from_state_dict(state, device=model.device)
    assert legacy.hydrogens_in_xray is True


@pytest.mark.unit
def test_deprecated_alias_is_inverted_and_warns():
    model = Model(verbose=0)
    with pytest.warns(DeprecationWarning):
        model.exclude_H_from_sf = True
    assert model.hydrogens_in_xray is False
    with pytest.warns(DeprecationWarning):
        assert model.exclude_H_from_sf is True


@pytest.mark.unit
def test_solvent_mask_ignores_hydrogens(pdb_dir):
    """The bulk-solvent mask is the same with and without hydrogen rows."""
    from torchref.scaling.solvent import SolventModel

    full = ModelFT(verbose=0, max_res=2.5)
    full.load_pdb(str(pdb_dir / "7L84.pdb"))
    heavy = ModelFT(verbose=0, max_res=2.5, strip_H=True)
    heavy.load_pdb(str(pdb_dir / "7L84.pdb"))
    assert _n_h(full) > 0 and _n_h(heavy) == 0
    mask_full = SolventModel(full, verbose=0).get_solvent_mask()
    mask_heavy = SolventModel(heavy, verbose=0).get_solvent_mask()
    assert torch.equal(mask_full, mask_heavy)
