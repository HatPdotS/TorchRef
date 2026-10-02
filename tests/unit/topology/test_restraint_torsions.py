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
