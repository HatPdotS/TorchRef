"""Ramachandran NLL surfaces for backbone restraints.

Pre-computed NLL = -log P(phi, psi | residue_type) from MolProbity/cctbx
distributions, held as one (6, 360, 360) tensor at 1° resolution in
``torchref/data/rama_nll_surfaces.pt``. The first axis indexes the six residue
types; see the ``TYPE_*`` constants below for the mapping and
:func:`classify_residue` for the selection rules.
"""

from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Surface type constants
# ---------------------------------------------------------------------------

#: Surface-type index for standard amino acids (not GLY/PRO/ILE/VAL/pre-PRO).
TYPE_GENERAL = 0
#: Surface-type index for glycine.
TYPE_GLYCINE = 1
#: Surface-type index for proline with a cis peptide bond.
TYPE_CIS_PROLINE = 2
#: Surface-type index for proline with a trans peptide bond.
TYPE_TRANS_PROLINE = 3
#: Surface-type index for a residue preceding proline (pre-proline).
TYPE_PRE_PROLINE = 4
#: Surface-type index for isoleucine or valine.
TYPE_ILE_VAL = 5

#: Number of Ramachandran surface types (size of the first tensor axis).
N_TYPES = 6
#: Number of 1° bins per axis, covering [-180, +180).
GRID_SIZE = 360

_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "rama_nll_surfaces.pt"


def load_nll_surfaces(device: torch.device) -> torch.Tensor:
    """Load pre-computed Ramachandran NLL surfaces.

    Parameters
    ----------
    device : torch.device
        Target device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Shape ``(6, 360, 360)`` with NLL values.
        Index ``[type, i, j]`` corresponds to phi = (i - 180)° and
        psi = (j - 180)°.
    """
    return torch.load(_DATA_FILE, map_location=device, weights_only=True)


def classify_residue(
    resname: str,
    next_resname: str,
    omega_deg: float = 180.0,
) -> int:
    """Determine the Ramachandran surface type for a residue.

    Parameters
    ----------
    resname : str
        Three-letter residue name of the current residue.
    next_resname : str
        Three-letter residue name of the next residue in sequence.
    omega_deg : float, optional
        Omega torsion angle in degrees, used for cis/trans proline detection.
        A proline is classified as cis when ``|omega_deg| < 90``. Default
        ``180.0`` (trans).

    Returns
    -------
    int
        One of TYPE_GENERAL .. TYPE_ILE_VAL.
    """
    if resname == "GLY":
        return TYPE_GLYCINE
    if resname == "PRO":
        # cis if |omega| < 90°
        if abs(omega_deg) < 90.0:
            return TYPE_CIS_PROLINE
        return TYPE_TRANS_PROLINE
    if next_resname == "PRO":
        return TYPE_PRE_PROLINE
    if resname in ("ILE", "VAL"):
        return TYPE_ILE_VAL
    return TYPE_GENERAL
