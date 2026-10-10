"""Torsion restraints measure the IUPAC dihedral the monomer-library references use.

Every amino-acid torsion reference is symmetric under negation for its period, so a
protein cannot tell the two signs apart. Sugar rings and their substituents can: 3A5V
carries NAG, MAN and BMA, whose AceDRG references are period-1 or otherwise
sign-sensitive, and under the opposite sign they are restrained toward the mirror image.
"""

import numpy as np
import pytest
import torch

from torchref.model.model import Model
from torchref.topology.ramachandran import TYPE_CIS_PROLINE, TYPE_TRANS_PROLINE

pytestmark = pytest.mark.unit

_SUGARS = ("NAG", "MAN", "BMA")


def _deposited(path):
    model = Model(verbose=0)
    model.load_pdb(str(path))
    return model.xyz().detach(), model.restraints


@pytest.fixture(scope="module")
def glycoprotein(pdb_dir):
    """3A5V coordinates (Å) and restraints, as deposited."""
    return _deposited(pdb_dir / "3A5V.pdb")


def _gemmi_dihedrals(xyz: torch.Tensor, rows) -> np.ndarray:
    import gemmi

    host = xyz.cpu().double().numpy()
    return np.degrees(
        [
            gemmi.calculate_dihedral(*(gemmi.Position(*host[i]) for i in row))
            for row in rows
        ]
    )


def _rms(values: torch.Tensor) -> float:
    return float(values.square().mean().sqrt())


def test_restraint_torsions_match_gemmi(glycoprotein):
    """``Restraints.torsions`` equals ``gemmi.calculate_dihedral`` on every restraint."""
    xyz, restraints = glycoprotein
    idx = restraints.restraints["torsion"]["all"]["indices"]
    ours = restraints.torsions(idx, xyz).cpu().double().numpy()
    delta = (ours - _gemmi_dihedrals(xyz, idx.tolist()) + 180.0) % 360.0 - 180.0
    assert np.abs(delta).max() < 1e-3


def test_sugar_torsions_sit_near_their_references(glycoprotein):
    """Deposited sugars score rms z of a few; their mirror images score above 15."""
    xyz, restraints = glycoprotein
    group = restraints.restraints["torsion"]["all"]
    deviations, sigmas_deg = restraints.torsion_deviations_with_sigmas(xyz)
    sigmas = torch.deg2rad(sigmas_deg)
    mirror = -restraints.torsions(group["indices"], xyz)
    mirrored = restraints._wrap_torsion_periodicity(
        torch.deg2rad(mirror - group["references"]), group["periods"]
    )
    resname = np.asarray(restraints.topology.columns()["resname"]).astype(str)
    owner = resname[group["indices"][:, 1].cpu().numpy()]

    for sugar in _SUGARS:
        mask = torch.as_tensor(owner == sugar, device=deviations.device)
        assert int(mask.sum()) > 0, sugar
        assert _rms(deviations[mask] / sigmas[mask]) < 4.0, sugar
        assert _rms(mirrored[mask] / sigmas[mask]) > 15.0, sugar


def test_cis_proline_takes_the_cis_surface(pdb_dir):
    """A proline reads the cis or trans surface by the |omega| < 90° of its peptide."""
    xyz, restraints = _deposited(pdb_dir / "1DAW.pdb")
    omega_idx = restraints.restraints["torsion"]["omega"]["indices"].tolist()
    # omega CA-C-N-CA ends on the C(i-1), N, CA that open phi C(i-1)-N-CA-C.
    omega_by_tail = {tuple(row[1:]): row for row in omega_idx}
    kind = restraints._rama_surface_type.cpu()
    proline = (kind == TYPE_CIS_PROLINE) | (kind == TYPE_TRANS_PROLINE)
    phi_rows = restraints._rama_phi_indices.cpu()[proline].tolist()

    omega = _gemmi_dihedrals(xyz, [omega_by_tail[tuple(row[:3])] for row in phi_rows])
    is_cis = kind[proline].numpy() == TYPE_CIS_PROLINE
    assert is_cis.any() and not is_cis.all()
    np.testing.assert_array_equal(is_cis, np.abs(omega) < 90.0)


def _surface_type(restraints, resseq):
    """Ramachandran surface type of chain A's residue ``resseq``, found by its phi."""
    columns = restraints.topology.columns()
    (ca,) = np.flatnonzero(
        (columns["chain"] == "A")
        & (columns["resseq"] == resseq)
        & (columns["name"] == "CA")
    )
    phi_ca = restraints._rama_phi_indices.cpu()[:, 2]
    (row,) = torch.nonzero(phi_ca == int(ca)).flatten().tolist()
    return int(restraints._rama_surface_type[row])


def test_proline_without_omega_defaults_to_trans(pdb_dir, tmp_path):
    """A proline whose peptide has no omega reads the trans surface.

    Without PHE232's CA neither the 231-232 nor the 232-233 peptide has an omega,
    while the one measured just before them, GLU230-PRO231, is cis.
    """
    lines = []
    for line in (pdb_dir / "1DAW.pdb").read_text().splitlines():
        if line.startswith("ATOM") and line[21] == "A":
            if line[22:26] == " 232" and line[12:16] == " CA ":
                continue
            if line[22:26] == " 233":
                line = line[:17] + "PRO" + line[20:]
        lines.append(line)
    path = tmp_path / "no_omega.pdb"
    path.write_text("\n".join(lines) + "\n")
    _, restraints = _deposited(path)
    assert _surface_type(restraints, 233) == TYPE_TRANS_PROLINE


def test_degenerate_omega_defaults_to_trans(pdb_dir, tmp_path):
    """A proline whose omega is undefined reads the trans surface.

    With ARG19's CA on its C, the ARG19-PRO20 omega has no defined value, which the
    dihedral reads as 0°.
    """
    lines = (pdb_dir / "1DAW.pdb").read_text().splitlines()
    arg19 = {
        line[12:16]: i
        for i, line in enumerate(lines)
        if line.startswith("ATOM") and line[17:26] == "ARG A  19"
    }
    ca, c = arg19[" CA "], arg19[" C  "]
    lines[ca] = lines[ca][:30] + lines[c][30:54] + lines[ca][54:]
    path = tmp_path / "collapsed.pdb"
    path.write_text("\n".join(lines) + "\n")
    _, restraints = _deposited(path)
    assert _surface_type(restraints, 20) == TYPE_TRANS_PROLINE


def _nucleotide(code):
    """One nucleotide at its dictionary's ideal coordinates.

    Returns the atom table, the restraint dictionary with each torsion's
    ``_chem_comp_tor.id`` (which names the sugar-pucker set it belongs to), and
    ``{torsion id: (atom rows, ideal value)}``.
    """
    from pathlib import Path

    import gemmi
    import pandas as pd

    from torchref import PATH_TORCHREF_DATA
    from torchref.topology.monomer.cif import read_cif

    path = Path(PATH_TORCHREF_DATA, "monomer_library", code[0].lower(), f"{code}.cif")
    block = gemmi.cif.read(str(path)).find_block(f"comp_{code}")
    atoms = block.find("_chem_comp_atom.", ["atom_id", "type_symbol", "x", "y", "z"])
    names = [gemmi.cif.as_string(row[0]) for row in atoms]
    table = pd.DataFrame(
        {
            "name": names,
            "element": [row[1] for row in atoms],
            "x": [float(row[2]) for row in atoms],
            "y": [float(row[3]) for row in atoms],
            "z": [float(row[4]) for row in atoms],
        }
    ).assign(chainid="A", resseq=1, resname=code)

    tags = ["id", "atom_id_1", "atom_id_2", "atom_id_3", "atom_id_4", "value_angle"]
    torsions = {
        row[0]: (
            tuple(names.index(gemmi.cif.as_string(row[k])) for k in (1, 2, 3, 4)),
            float(row[5]),
        )
        for row in block.find("_chem_comp_tor.", tags)
    }
    cif_dict = read_cif(str(path))
    return table, cif_dict, torsions


@pytest.mark.parametrize("code, pucker", [("DA", "C2e"), ("A", "C3e")])
def test_each_nucleotide_keeps_one_sugar_pucker(code, pucker):
    """One sugar-pucker torsion set per residue, the one its starting geometry has.

    The dictionaries restrain the same ring torsions to C2'-endo and to C3'-endo
    values; DNA's ideal coordinates are C2'-endo and RNA's C3'-endo.
    """
    from torchref.topology.build import build_topology_with_values
    from torchref.topology.topology import Topology

    table, cif_dict, torsions = _nucleotide(code)
    xyz = torch.as_tensor(table[["x", "y", "z"]].values)
    topology, values, _ = build_topology_with_values(
        Topology.from_table(table), cif_dict, xyz
    )
    references = {}
    for row, reference in zip(
        topology.atoms.torsions.origin("intra").tolist(),
        values["torsion"]["intra"]["references"].tolist(),
    ):
        references.setdefault(tuple(row), []).append(reference)

    ring = [tid for tid in torsions if tid.startswith(f"{pucker}-nyu")]
    assert len(ring) == 5
    for tid in ring:
        atoms, value = torsions[tid]
        assert references[atoms] == pytest.approx([value], abs=1e-3), tid
    assert all(len(found) == 1 for found in references.values())
