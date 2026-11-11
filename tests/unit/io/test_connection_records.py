"""Inter-residue bonds are written from the model topology.

Each output is read back with gemmi and its connections compared with gemmi's reading
of the deposited file: 3A5V has four disulfides (one to a two-conformer cysteine) and
twelve glycan LINKs, 1DAW fourteen magnesium links. With the input's metadata passed,
its own SSBOND/LINK/``_struct_conn`` must be replaced, not repeated.
"""

import gemmi
import pytest

from torchref.io.metadata import RefinementMetadata
from torchref.model.model import Model


def _connections(path):
    """``(type, partner atoms)`` per connection, duplicates kept, altlocs ignored."""
    return [
        (
            con.type.name,
            frozenset(
                (p.chain_name, p.res_id.seqid.num, p.atom_name)
                for p in (con.partner1, con.partner2)
            ),
        )
        for con in gemmi.read_structure(str(path)).connections
    ]


@pytest.mark.unit
@pytest.mark.parametrize("entry", ["3A5V", "1DAW"])
def test_written_bonds_match_the_deposited_file(entry, pdb_dir, tmp_path):
    source = pdb_dir / f"{entry}.pdb"
    model = Model()
    model.load_pdb(str(source))
    meta = RefinementMetadata.from_pdb_file(str(source))
    expected = _connections(source)

    for name, write, metadata in [
        ("plain.pdb", model.write_pdb, None),
        ("meta.pdb", model.write_pdb, meta),
        ("plain.cif", model.write_cif, None),
        ("meta.cif", model.write_cif, meta),
    ]:
        write(str(tmp_path / name), metadata=metadata)
        back = _connections(tmp_path / name)
        assert set(back) == set(expected), name
        # mmCIF keeps 3A5V's disulfide to each conformer of Cys 53; SSBOND cannot.
        extra = int(entry == "3A5V" and name.endswith(".cif"))
        assert len(back) == len(expected) + extra, name


@pytest.mark.unit
def test_input_bonds_to_symmetry_mates_are_kept(pdb_dir, cif_dir, tmp_path):
    """No topology holds a bond to a symmetry mate, so the input's record is kept."""
    model = Model()
    model.load_pdb(str(pdb_dir / "1DAW.pdb"))

    mate = (
        "LINK         OE2AGLU A  40                 CD  LYS A  49     1555   4556  2.39"
    )
    meta = RefinementMetadata.from_pdb_file(str(pdb_dir / "1DAW.pdb"))
    meta.passthrough_pdb_records["LINK"].append(mate)
    model.write_pdb(str(tmp_path / "out.pdb"), metadata=meta)
    assert mate in (tmp_path / "out.pdb").read_text().splitlines()

    meta = RefinementMetadata.from_cif_file(str(cif_dir / "1DAW.cif"))
    retarget = {
        "conn_type_id": "covale",
        "ptnr1_label_comp_id": "GLU",
        "ptnr1_auth_seq_id": "40",
        "ptnr1_label_atom_id": "OE2",
        "pdbx_ptnr1_label_alt_id": "A",
        "ptnr2_label_comp_id": "LYS",
        "ptnr2_auth_seq_id": "49",
        "ptnr2_label_atom_id": "CD",
        "ptnr2_symmetry": "4_556",
    }
    for tag, values in meta.passthrough_cif_categories["_struct_conn"].items():
        values.append(retarget.get(tag.split(".")[1], values[0]))
    model.write_cif(str(tmp_path / "out.cif"), metadata=meta)
    block = gemmi.cif.read(str(tmp_path / "out.cif")).sole_block()
    table = block.find("_struct_conn.", ["ptnr2_label_atom_id", "ptnr2_symmetry"])
    assert [list(row) for row in table].count(["CD", "4_556"]) == 1
    assert len(table) == 15
