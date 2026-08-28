"""Hydrogens are present by default: kept where the file has them, generated where not.

The interesting cases are the partially-hydrogenated file, which has to be topped up per
parent rather than left alone, and the per-atom buffers that are cached lazily and go
stale the moment the atom set grows.
"""

import numpy as np
import pytest

from torchref.model.model import Model


def _elements(model):
    return model.pdb["element"].astype(str).str.strip().values


def _counts(model):
    elements = _elements(model)
    n_h = int((elements == "H").sum())
    return len(model.pdb), n_h


@pytest.mark.unit
def test_a_file_without_hydrogens_gets_them(pdb_dir):
    """1DAW ships none, so every hydrogen here is generated."""
    model = Model(verbose=0)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))

    total, n_h = _counts(model)
    assert n_h > 0
    heavy = total - n_h
    assert (
        0.7 < n_h / heavy < 1.3
    ), f"{n_h} hydrogens on {heavy} heavy atoms is not a plausible ratio"


@pytest.mark.unit
def test_a_partially_hydrogenated_file_is_topped_up(pdb_dir):
    """1AK5 ships 675 hydrogens on 2582 heavy atoms, where full is roughly 2500.

    Generation is decided per parent -- the plan proposes only a hydrogen the template
    names and the model lacks -- so a file that already has some still gets the rest. A
    does-the-table-contain-any test would have left this structure as deposited.
    """
    kept = Model(verbose=0, add_hydrogens=False)
    kept.load_pdb(str(pdb_dir / "1AK5_with_H.pdb"))
    _, n_kept = _counts(kept)

    topped = Model(verbose=0)
    topped.load_pdb(str(pdb_dir / "1AK5_with_H.pdb"))
    _, n_topped = _counts(topped)

    assert n_kept > 0, "1AK5_with_H is supposed to ship some hydrogens"
    assert (
        n_topped > n_kept * 2
    ), f"only {n_topped} hydrogens after top-up, against {n_kept} in the file"


@pytest.mark.unit
def test_strip_H_still_removes_everything(pdb_dir):
    """The opt-out is unaffected: no hydrogen survives, generated or deposited."""
    for name in ("1DAW.pdb", "7L84.pdb"):
        model = Model(verbose=0, strip_H=True)
        model.load_pdb(str(pdb_dir / name))
        _, n_h = _counts(model)
        assert n_h == 0, f"{name} kept {n_h} hydrogens under strip_H"


@pytest.mark.unit
def test_add_hydrogens_false_keeps_the_file_as_it_is(pdb_dir):
    """Generation off, stripping off: exactly what the reader produced."""
    model = Model(verbose=0, add_hydrogens=False)
    model.load_pdb(str(pdb_dir / "7L84.pdb"))
    total, n_h = _counts(model)
    assert n_h > 0, "7L84 ships hydrogens, so they should have been kept"

    generated = Model(verbose=0)
    generated.load_pdb(str(pdb_dir / "7L84.pdb"))
    assert _counts(generated)[0] >= total


@pytest.mark.unit
def test_per_atom_buffers_are_rebuilt_for_the_new_atom_set(pdb_dir):
    """Every lazily-cached per-atom buffer matches the table after generation.

    These are guarded by ``hasattr`` and returned as-is once built, which was safe only
    while an atom-set change always produced a fresh model. Generating hydrogens in
    place left the van der Waals radii at the heavy-atom count while the pair list
    indexed the full set, and the non-bonded build raised ``IndexError``.
    """
    model = Model(verbose=0)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    n_atoms = len(model.pdb)

    assert model.get_vdw_radii().shape[0] == n_atoms
    assert model.Z.shape[0] == n_atoms

    radii = model.get_vdw_radii().detach().cpu().numpy()
    assert np.isfinite(radii).all()
    is_h = _elements(model) == "H"
    assert is_h.any()
    assert np.allclose(radii[is_h], 1.20), "hydrogens did not get a hydrogen radius"


@pytest.mark.unit
def test_restraints_build_over_the_hydrogenated_model(pdb_dir):
    """Restraints cover the hydrogens, and each carries exactly one bond."""
    model = Model(verbose=0)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    restraints = model.restraints

    elements = _elements(model)
    is_h = elements == "H"
    bonds = restraints.restraints["bond"]["all"]["indices"].cpu().numpy()
    involves_h = is_h[bonds[:, 0]] | is_h[bonds[:, 1]]
    assert int(involves_h.sum()) == int(is_h.sum())

    vdw = restraints.restraints["vdw"]["indices"]
    assert int(vdw.max()) < len(
        model.pdb
    ), "the non-bonded pair list indexes past the end of the atom table"


@pytest.mark.unit
def test_riding_hydrogens_are_not_placed_when_real_ones_exist(pdb_dir):
    """The riding stand-in goes quiet once the model carries hydrogens.

    Riding hydrogens approximate the sterics of hydrogens the model does not have.
    Placing them alongside real ones would put phantom atoms in the structure that push
    real ones around -- and they would not even be the hydrogens the generator declined,
    because the riding builder counts bonded neighbours by distance while the generator
    reads them off the bond graph.
    """
    model = Model(verbose=0)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    restraints = model.restraints

    assert restraints.h_topo is not None
    assert restraints.h_topo.n_hydrogens == 0

    stripped = Model(verbose=0, strip_H=True)
    stripped.load_pdb(str(pdb_dir / "1DAW.pdb"))
    assert (
        stripped.restraints.h_topo.n_hydrogens > 0
    ), "with hydrogens absent the riding stand-in should still be built"
