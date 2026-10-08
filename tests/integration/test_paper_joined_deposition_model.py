"""The paper's dark/light deposition merge keeps each atom's own occupancy."""

import importlib.util

import gemmi
import pytest

pytestmark = pytest.mark.integration


def _occupancies(structure):
    return {
        (chain.name, str(residue.seqid), atom.name, atom.altloc): atom.occ
        for chain in structure[0]
        for residue in chain
        for atom in residue
    }


def test_merge_scales_each_atoms_occupancy_by_its_state(project_root, pdb_dir):
    """3GR5's HOH A224 sits on a special position at 0.50 in both states."""
    path = project_root / "paper" / "figure4_difference_refinement"
    spec = importlib.util.spec_from_file_location(
        "create_joined_model_for_deposition",
        path / "create_joined_model_for_deposition.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = str(pdb_dir / "3GR5.pdb")
    own = _occupancies(gemmi.read_structure(source))

    merged = _occupancies(
        module.merge_structures(
            gemmi.read_structure(source), gemmi.read_structure(source), 0.7, 0.3
        )
    )

    assert merged[("A", "224", "O", "A")] == pytest.approx(0.35)
    for (chain, seqid, name, _), occ in own.items():
        assert merged[(chain, seqid, name, "A")] == pytest.approx(0.7 * occ)
        assert merged[(chain, seqid, name, "B")] == pytest.approx(0.3 * occ)
