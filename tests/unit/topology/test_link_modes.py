"""The two link modes that take coordinates out of the restraint graph.

``peptide_links="sequence"`` bonds sequence neighbours whatever their separation, and
``disulfides="open"`` bonds no cysteines but lets their sulfurs approach to the bond
length, so a bridge can close under the data.
"""

import math

import numpy as np
import pytest
import torch

from torchref.model import Model
from torchref.topology import Restraints, Topology
from torchref.topology.build import _template_is_l
from torchref.topology.nonbonded import BridgeContacts, contact_distances

CPU = torch.device("cpu")


def _restraints(model, xyz, **modes):
    ctx = model.ctx
    return Restraints(
        topology=ctx.topology,
        xyz=xyz,
        cell=ctx.cell,
        spacegroup=ctx.spacegroup,
        links=ctx.links,
        **modes,
    )


def _rows(topology, name):
    columns = topology.columns()
    return np.flatnonzero((columns["resname"] == "CYS") & (columns["name"] == name))


@pytest.fixture(scope="module")
def bridged(pdb_dir):
    """3GR5: one disulfide, Cys 136 to Cys 155."""
    return Model(device=CPU).load_pdb(str(pdb_dir / "3GR5.pdb"))


@pytest.fixture(scope="module")
def protein(pdb_dir):
    model = Model(device=CPU).load_pdb(str(pdb_dir / "1DAW.pdb"))
    return model.select("not resname HOH and not resname MG and not resname ANP")


@pytest.mark.unit
def test_sequence_links_do_not_depend_on_the_coordinates(protein):
    """Coordinates scattered by 1 Å break most C-N distances; the sequence still links
    every neighbour, and the bonds are the ones the deposited coordinates give."""
    xyz = protein.xyz().detach()
    torch.manual_seed(0)
    scattered = xyz + torch.randn_like(xyz)

    reference = _restraints(protein, xyz, nonbonded=False).topology
    by_distance = _restraints(protein, scattered, nonbonded=False).topology
    by_sequence = _restraints(
        protein, scattered, nonbonded=False, peptide_links="sequence"
    ).topology

    n_links = len(reference.residues.links_of_kind("TRANS"))
    assert len(by_distance.residues.links_of_kind("TRANS")) < n_links
    assert len(by_sequence.residues.links_of_kind("TRANS")) == n_links
    assert by_sequence.atoms.bonds.tuple_set() == reference.atoms.bonds.tuple_set()


@pytest.mark.unit
def test_sequence_links_stop_at_a_numbering_gap_and_at_non_peptides():
    full = Topology.from_sequence({"A": "AGAGA", "W": ["HOH", "HOH"]})
    keep = np.flatnonzero(full.columns()["resseq"] != 3)
    gapped = full.gather(
        keep[(full.columns()["chain"][keep] == "A") | (keep >= full.n_atoms - 2)]
    )
    restraints = Restraints(
        topology=gapped, peptide_links="sequence", disulfides="open"
    )

    links = restraints.topology.residues.links_of_kind("TRANS")
    resseq = restraints.topology.residues.resseq
    assert sorted((int(resseq[a]), int(resseq[b])) for a, b in links) == [
        (1, 2),
        (4, 5),
    ]


@pytest.mark.unit
def test_open_disulfides_are_not_bonded_but_may_close(bridged):
    xyz = bridged.xyz().detach()
    detected = _restraints(bridged, xyz)
    opened = _restraints(bridged, xyz, disulfides="open")

    assert len(detected.topology.residues.links_of_kind("disulf")) == 1
    assert len(opened.topology.residues.links_of_kind("disulf")) == 0
    for kind in ("bond", "angle", "torsion"):
        assert "disulfide" in detected.restraints[kind]
        assert "disulfide" not in opened.restraints[kind]

    sulfurs, carbons = _rows(opened.topology, "SG"), _rows(opened.topology, "CB")
    disulf = opened.link_dict["disulf"]
    bond = float(disulf["bonds"]["value"].iloc[0])
    angle = math.radians(float(disulf["angles"]["value"].iloc[0]))
    third = math.sqrt(bond**2 + 1.812**2 - 2 * bond * 1.812 * math.cos(angle))

    def contact(restraints, rows_a, rows_b):
        pairs = restraints.restraints["vdw"]["indices"].cpu().numpy()
        hit = (np.isin(pairs[:, 0], rows_a) & np.isin(pairs[:, 1], rows_b)) | (
            np.isin(pairs[:, 0], rows_b) & np.isin(pairs[:, 1], rows_a)
        )
        return restraints.restraints["vdw"]["min_distances"][torch.as_tensor(hit)]

    # Bonded, the bridge's 1-2 and 1-3 pairs are excluded from the pair list; open,
    # they are in it at the distances a closed bridge puts them.
    assert len(contact(detected, sulfurs, sulfurs)) == 0
    assert len(contact(detected, sulfurs, carbons)) == 0
    assert contact(opened, sulfurs, sulfurs).tolist() == pytest.approx([bond])
    assert contact(opened, sulfurs, carbons).tolist() == pytest.approx(
        [third, third], abs=1e-3
    )
    assert third < 3.1 and float((xyz[sulfurs[0]] - xyz[sulfurs[1]]).norm()) > bond


@pytest.mark.unit
def test_open_disulfides_leave_every_other_contact_alone(bridged):
    xyz = bridged.xyz().detach()
    detected = _restraints(bridged, xyz).restraints["vdw"]
    opened = _restraints(bridged, xyz, disulfides="open").restraints["vdw"]
    cysteine = np.concatenate(
        [_rows(bridged.ctx.topology, name) for name in ("SG", "CB", "CA")]
    )

    def table(vdw):
        keys = torch.cat(
            [vdw["indices"], vdw["symop_indices"][:, None], vdw["cell_offsets"]], dim=1
        )
        return {
            tuple(key): distance
            for key, distance in zip(keys.tolist(), vdw["min_distances"].tolist())
        }

    before, after = table(detected), table(opened)
    assert before.keys() <= after.keys()
    assert all(after[key] == distance for key, distance in before.items())
    added = after.keys() - before.keys()
    assert added and all(set(key[:2]) <= set(cysteine.tolist()) for key in added)


@pytest.mark.unit
def test_riding_hydrogens_are_not_scored_across_an_open_bridge(bridged):
    """The hydrogens riding on a cysteine's SG and CB sit where the other sulfur
    arrives; scored against it they would hold the bridge open."""
    opened = _restraints(bridged, bridged.xyz().detach(), disulfides="open")
    hydrogens = opened.h_topo
    sulfurs, carbons = _rows(opened.topology, "SG"), _rows(opened.topology, "CB")
    parent = hydrogens.h_parent_idx.cpu().numpy()
    rider = hydrogens.cand_idx_i.cpu().numpy() - opened.topology.n_atoms
    partner = hydrogens.cand_idx_j.cpu().numpy()

    on_bridge = np.isin(parent[rider], np.concatenate([sulfurs, carbons]))
    assert on_bridge.any()
    assert not (on_bridge & np.isin(partner, sulfurs)).any()


@pytest.mark.unit
def test_bridge_contacts_replace_the_radius_sum_by_depth():
    radii = torch.tensor([1.8, 1.7, 1.8, 1.7, 1.5])
    depth = torch.tensor([0, 1, 0, 1, -1])
    pairs = torch.tensor([[0, 2], [0, 3], [1, 2], [1, 3], [0, 4], [1, 4]])
    bridges = BridgeContacts(depth=depth, distances=(2.03, 3.03))

    plain = contact_distances(radii, None, pairs)
    across = contact_distances(radii, None, pairs, bridges)

    assert across.tolist() == pytest.approx([2.03, 3.03, 3.03, 3.4, 3.3, 3.2])
    assert plain.tolist() == pytest.approx([3.6, 3.5, 3.5, 3.4, 3.3, 3.2])


@pytest.mark.unit
def test_a_build_without_coordinates_needs_both_modes(bridged):
    for modes in ({}, {"peptide_links": "sequence"}, {"disulfides": "open"}):
        with pytest.raises(ValueError, match="need the coordinates"):
            _restraints(bridged, None, **modes)
    with pytest.raises(ValueError, match="peptide_links must be one of"):
        _restraints(bridged, None, peptide_links="guess")


@pytest.mark.unit
def test_without_coordinates_the_hand_is_the_templates():
    def chirals(order, sign):
        return {
            "center": np.array(["CA"]),
            "atom1": np.array([order[0]]),
            "atom2": np.array([order[1]]),
            "atom3": np.array([order[2]]),
            "volume_sign": np.array([sign]),
        }

    assert _template_is_l(chirals(("N", "C", "CB"), 1.0))
    assert _template_is_l(chirals(("C", "N", "CB"), -1.0))
    assert not _template_is_l(chirals(("N", "C", "CB"), -1.0))
    assert not _template_is_l(chirals(("N", "C", "CB"), 0.0))
    assert not _template_is_l(chirals(("N", "C", "HA"), 1.0))
    assert not _template_is_l(None)
