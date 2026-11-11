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
