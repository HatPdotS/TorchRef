"""Inter-residue bonds are written from the model topology.

``Model.write_pdb`` writes the topology's disulfides as SSBOND and its other
inter-residue bonds as LINK records; ``Model.write_cif`` writes both as
``_struct_conn``. Each test reads the output back with gemmi, an independent reader,
and compares its connections with gemmi's reading of the deposited file. 3A5V carries
four disulfides (one to a cysteine in two conformations) and twelve glycan LINKs; 1DAW
carries fourteen magnesium links.
"""

import gemmi
import pytest

from torchref.io.metadata import RefinementMetadata
from torchref.model.model import Model


def _connections(path):
    """``{(type, frozenset of (chain, seqnum, icode, atom))}`` as gemmi reads ``path``."""
    st = gemmi.read_structure(str(path))
    return {
        (
            con.type.name,
            frozenset(
                (p.chain_name, p.res_id.seqid.num, p.res_id.seqid.icode, p.atom_name)
                for p in (con.partner1, con.partner2)
            ),
        )
        for con in st.connections
    }


@pytest.fixture(scope="module")
def glycoprotein(pdb_dir):
    model = Model()
    model.load_pdb(str(pdb_dir / "3A5V.pdb"))
    return model


@pytest.fixture(scope="module")
def magnesium(cif_dir):
    model = Model()
    model.load_cif(str(cif_dir / "1DAW.cif"))
    return model


@pytest.mark.unit
def test_connections_come_from_the_topology(glycoprotein):
    """Disulfides per atom pair (the two-conformer SG twice), links as covalent."""
    table = glycoprotein.connections()
    assert table["conn_type"].value_counts().to_dict() == {"covale": 12, "disulf": 5}
    assert table["length"].between(1.3, 2.2).all()


@pytest.mark.unit
@pytest.mark.parametrize("suffix", [".pdb", ".cif"])
def test_written_bonds_match_the_deposited_file(
    glycoprotein, pdb_dir, tmp_path, suffix
):
    out = tmp_path / f"out{suffix}"
    write = glycoprotein.write_pdb if suffix == ".pdb" else glycoprotein.write_cif
    write(str(out))

    expected = _connections(pdb_dir / "3A5V.pdb")
    assert len(expected) == 16
    assert _connections(out) == expected


@pytest.mark.unit
@pytest.mark.parametrize("suffix", [".pdb", ".cif"])
def test_metal_links_are_metal_coordination(magnesium, tmp_path, suffix):
    out = tmp_path / f"out{suffix}"
    write = magnesium.write_pdb if suffix == ".pdb" else magnesium.write_cif
    write(str(out))

    back = _connections(out)
    assert len(back) == 14
    if suffix == ".cif":
        assert {kind for kind, _ in back} == {"MetalC"}


@pytest.mark.unit
def test_pdb_metadata_does_not_repeat_the_bonds(glycoprotein, pdb_dir, tmp_path):
    """The topology's records replace the SSBOND and LINK records metadata carries."""
    meta = RefinementMetadata.from_pdb_file(str(pdb_dir / "3A5V.pdb"))
    assert meta.passthrough_pdb_records.get("LINK")
    out = tmp_path / "out.pdb"
    glycoprotein.write_pdb(str(out), metadata=meta)

    records = [line[:6] for line in out.read_text().splitlines()]
    assert records.count("SSBOND") == 4
    assert records.count("LINK  ") == 12
    assert _connections(out) == _connections(pdb_dir / "3A5V.pdb")


@pytest.mark.unit
def test_cif_metadata_does_not_repeat_the_bonds(glycoprotein, cif_dir, tmp_path):
    meta = RefinementMetadata.from_cif_file(str(cif_dir / "3A5V.cif"))
    assert "_struct_conn" in meta.passthrough_cif_categories
    out = tmp_path / "out.cif"
    glycoprotein.write_cif(str(out), metadata=meta)

    block = gemmi.cif.read(str(out)).sole_block()
    assert len(block.find_values("_struct_conn.id")) == 17
    assert out.read_text().count("_struct_conn.id") == 1


@pytest.mark.unit
def test_header_connectivity_replaces_the_inputs(pdb_dir):
    """An empty list is the model having no bonds, not "keep the input's"."""
    meta = RefinementMetadata.from_pdb_file(str(pdb_dir / "3A5V.pdb"))
    kept = meta.render_pdb_header().splitlines()
    replaced = meta.render_pdb_header(connectivity=[]).splitlines()

    assert any(line.startswith(("SSBOND", "LINK")) for line in kept)
    assert not any(line.startswith(("SSBOND", "LINK")) for line in replaced)
