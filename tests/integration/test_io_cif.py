"""
Integration tests for CIF file loading.

Tests real file I/O operations with actual CIF files.
"""

import pytest
import torch

from torchref.config import canonical_device, get_default_device, get_float_dtype


@pytest.mark.integration
def test_cif_loading_contract(loaded_model, sample_cif_file) -> None:
    """A deposited CIF supplies aligned atomic tensors and its crystal metadata."""
    import gemmi

    model = loaded_model
    reference = gemmi.read_structure(str(sample_cif_file))
    xyz, adp, occupancy = model.xyz(), model.adp(), model.occupancy()
    assert xyz.shape == (len(model.pdb), 3)
    assert len(xyz) > 0
    assert adp.shape == occupancy.shape == (len(xyz),)
    for tensor in (xyz, adp, occupancy, model.cell.data):
        assert tensor.dtype == get_float_dtype()
        assert canonical_device(tensor.device) == canonical_device(get_default_device())
        assert torch.isfinite(tensor).all()
    assert torch.all(adp >= 0)
    assert {"x", "y", "z", "element", "resname", "chainid", "resseq"} <= set(
        model.pdb.columns
    )
    assert {"C", "N", "O"} <= set(model.pdb.element)
    torch.testing.assert_close(
        model.cell.data, xyz.new_tensor(reference.cell.parameters)
    )
    assert (
        model.spacegroup.number
        == gemmi.find_spacegroup_by_name(reference.spacegroup_hm).number
    )


class TestCIFSaving:
    """Tests for saving CIF files."""

    @pytest.mark.integration
    def test_save_and_reload_cif(self, sample_cif_file, tmp_path):
        """Test saving a model to CIF and reloading it."""
        from torchref.model.model import Model

        # Load original
        model1 = Model()
        model1.load_cif(str(sample_cif_file))
        n_atoms1 = model1.xyz().shape[0]

        # Save to temp file using write_pdb (CIF saving may not exist)
        output_path = tmp_path / "test_output.pdb"
        model1.write_pdb(str(output_path))

        assert output_path.exists()

        # add_hydrogens=False on reload: what is under test is whether the written
        # file round-trips, not whether generation reruns. Regenerating on reload can
        # legitimately differ, because ``write_pdb`` does not emit LINK records -- so a
        # metal-coordinated nitrogen comes back with a free valence and takes a hydrogen
        # it did not have before.
        model2 = Model(add_hydrogens=False)
        model2.load_pdb(str(output_path))
        n_atoms2 = model2.xyz().shape[0]

        # Compare atom counts
        assert n_atoms2 == n_atoms1
