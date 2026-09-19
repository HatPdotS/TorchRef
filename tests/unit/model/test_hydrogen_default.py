"""Keep deposited hydrogens by default and generate missing ones only on request.

The interesting cases are the partially-hydrogenated file, which has to be topped up per
parent rather than left alone, and the per-atom buffers that are cached lazily and go
stale the moment the atom set grows.
"""

from pathlib import Path

import numpy as np
import pytest

from torchref.model.model import Model
from torchref.model.model_ft import ModelFT


def _elements(model):
    return model.pdb["element"].astype(str).str.strip().values


def _counts(model):
    elements = _elements(model)
    n_h = int((elements == "H").sum())
    return len(model.pdb), n_h


@pytest.mark.unit
@pytest.mark.parametrize("model_class", [Model, ModelFT])
@pytest.mark.parametrize("filename", ["1DAW.pdb", "1AK5_with_H.pdb", "7L84.pdb"])
def test_default_preserves_deposited_atoms(
    pdb_dir: Path,
    filename: str,
    monkeypatch: pytest.MonkeyPatch,
    model_class: type[Model],
) -> None:
    """Default loading neither generates hydrogens nor removes deposited ones."""
    from torchref.io.pdb import PDBReader

    path = pdb_dir / filename
    deposited, _, _ = PDBReader(verbose=0).read(str(path))()

    def unexpected_generation(self: Model) -> None:
        pytest.fail("Default loading must not generate hydrogens")

    monkeypatch.setattr(Model, "_add_missing_hydrogens", unexpected_generation)
    model = model_class(verbose=0).load_pdb(str(path))

    np.testing.assert_array_equal(_elements(model), deposited["element"].str.strip())


@pytest.mark.unit
def test_default_cif_load_does_not_generate_hydrogens(
    cif_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mmCIF loading also leaves missing hydrogens absent by default."""

    def unexpected_generation(self: Model) -> None:
        pytest.fail("Default mmCIF loading must not generate hydrogens")

    monkeypatch.setattr(Model, "_add_missing_hydrogens", unexpected_generation)
    model = Model(verbose=0).load_cif(str(cif_dir / "1DAW.cif"))
    total, n_h = _counts(model)
    assert total > 0
    assert n_h == 0


@pytest.mark.unit
def test_context_defaults_to_no_hydrogen_generation() -> None:
    """A standalone model context leaves hydrogen generation disabled."""
    from torchref.model.context import ModelContext

    assert ModelContext().add_hydrogens is False


@pytest.mark.unit
def test_a_file_without_hydrogens_gets_them(pdb_dir):
    """1DAW ships none, so every hydrogen here is generated."""
    model = Model(verbose=0, add_hydrogens=True)
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

    topped = Model(verbose=0, add_hydrogens=True)
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
        model = Model(verbose=0, strip_H=True, add_hydrogens=True)
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

    generated = Model(verbose=0, add_hydrogens=True)
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
    model = Model(verbose=0, add_hydrogens=True)
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
    model = Model(verbose=0, add_hydrogens=True)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    restraints = model.restraints

    elements = _elements(model)
    is_h = elements == "H"
    assert is_h.any()
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
    model = Model(verbose=0, add_hydrogens=True)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    restraints = model.restraints

    assert restraints.h_topo is not None
    assert restraints.h_topo.n_hydrogens == 0

    stripped = Model(verbose=0, strip_H=True)
    stripped.load_pdb(str(pdb_dir / "1DAW.pdb"))
    assert (
        stripped.restraints.h_topo.n_hydrogens > 0
    ), "with hydrogens absent the riding stand-in should still be built"


# --- The user's restraint dictionary is the one that hydrogenates ------------------

RENAMED_GLU_H = {"HAX", "HBX", "HBY", "HGX", "HGY"}


@pytest.fixture
def renamed_glu_cif(test_files_dir):
    """A GLU dictionary whose side-chain hydrogens carry names the library lacks."""
    return str(test_files_dir / "restraints" / "GLU_renamed.cif")


def _glu_hydrogen_names(model):
    pdb = model.pdb
    glu_h = (pdb["resname"].astype(str).str.strip() == "GLU") & (
        pdb["element"].astype(str).str.strip() == "H"
    )
    return set(pdb.loc[glu_h, "name"].astype(str).str.strip())


@pytest.mark.unit
def test_generation_reads_the_cif_given_at_construction(pdb_dir, renamed_glu_cif):
    """A dictionary passed to the constructor overrides the library for generation.

    The names prove which dictionary was read, and the bond degree proves the generated
    hydrogens are the ones the restraints know: a hydrogen generated from one
    dictionary and restrained by another has no bond edge at all.
    """
    model = Model(verbose=0, add_hydrogens=True, cif_path=renamed_glu_cif)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    assert model.ctx.cif_path == renamed_glu_cif
    names = _glu_hydrogen_names(model)
    assert RENAMED_GLU_H <= names
    assert not {"HA", "HB2", "HB3", "HG2", "HG3"} & names

    atoms = model.restraints.topology.atoms
    is_h = atoms.is_hydrogen.cpu().numpy()
    degree = atoms.degree().cpu().numpy()
    assert (degree[is_h] > 0).all(), "generated hydrogens without a bond restraint"


@pytest.mark.unit
def test_derived_models_keep_the_restraint_cif(pdb_dir, renamed_glu_cif):
    """hydrogenate, strip_hydrogens and select all carry the dictionary along."""
    model = Model(verbose=0, add_hydrogens=False, cif_path=renamed_glu_cif)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))

    hydrogenated = model.hydrogenate()
    assert hydrogenated.ctx.cif_path == renamed_glu_cif
    assert RENAMED_GLU_H <= _glu_hydrogen_names(hydrogenated)
    assert "GLU" in hydrogenated.restraints.cif_dict
    template_h = set(
        hydrogenated.restraints.cif_dict["GLU"]["atoms"]["atom_id"].astype(str).str.strip()
    )
    assert RENAMED_GLU_H <= template_h

    assert hydrogenated.strip_hydrogens().ctx.cif_path == renamed_glu_cif
    assert model.select("resname GLU").ctx.cif_path == renamed_glu_cif


@pytest.mark.unit
def test_state_dict_round_trips_the_restraint_cif(pdb_dir, renamed_glu_cif):
    model = Model(verbose=0, cif_path=renamed_glu_cif)
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    restored = Model.create_from_state_dict(model.state_dict(), verbose=0)
    assert restored.ctx.cif_path == renamed_glu_cif


@pytest.mark.unit
def test_load_model_registers_the_cif_before_loading(pdb_dir, renamed_glu_cif):
    """The shared CLI loader generates from the user dictionary too."""
    from torchref.cli._common import load_model

    model = load_model(
        str(pdb_dir / "1DAW.pdb"), verbose=0, cif=renamed_glu_cif, add_hydrogens=True
    )
    assert model.ctx.cif_path == renamed_glu_cif
    assert RENAMED_GLU_H <= _glu_hydrogen_names(model)


@pytest.mark.unit
def test_generation_reads_every_compound_of_a_multi_block_cif(pdb_dir, test_files_dir):
    """A dictionary with several ``data_comp_`` blocks hydrogenates each of its compounds.

    Multi-compound dictionaries once restrained only their last block; generation now
    reads the same dictionary, so both renamed sets must appear and every generated
    hydrogen must carry a bond edge.
    """
    cif = test_files_dir / "restraints" / "GLU_ASP_renamed.cif"
    blocks = [l for l in cif.read_text().splitlines() if l.startswith("data_comp_")]
    assert len(blocks) == 3, blocks  # comp_list + GLU + ASP: the fixture is really multi-block

    model = Model(verbose=0, add_hydrogens=True, cif_path=str(cif))
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))
    pdb = model.pdb
    is_h = pdb["element"].astype(str).str.strip() == "H"
    resname = pdb["resname"].astype(str).str.strip()
    names = pdb["name"].astype(str).str.strip()
    assert RENAMED_GLU_H <= set(names[is_h & (resname == "GLU")])
    assert {"HBQ", "HBR"} <= set(names[is_h & (resname == "ASP")])
    assert not {"HB2", "HB3"} & set(names[is_h & (resname == "ASP")])

    atoms = model.restraints.topology.atoms
    degree = atoms.degree().cpu().numpy()
    assert (degree[atoms.is_hydrogen.cpu().numpy()] > 0).all()
