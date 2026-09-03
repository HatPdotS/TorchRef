"""Does a placed search model sit on top of the chains already placed?

The likelihood penalises overlap -- density counted twice is density the data
do not have -- but not sharply enough to rely on. On a homodimer the site of the
chain already placed is the strongest signal in the map, and a second copy is
happy to land on it. So a candidate placement is checked against the symmetry
images of the fixed chains and rejected outright when more than a small fraction
of its C-alpha atoms are within contact distance, which is what Phaser's packing
function does.

C-alpha atoms only: a clash between backbones is a clash, and side chains are
where a slightly wrong placement disagrees with its neighbours legitimately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...model.model_ft import ModelFT

#: Two C-alpha atoms closer than this are one clash. Adjacent residues sit at
#: 3.8 A; nothing legitimate is closer.
CLASH_CUTOFF_A = 3.0

#: Placements with more than this fraction of their C-alpha atoms clashing are
#: rejected. A few percent tolerates a loop or a terminus genuinely disordered
#: in the crystal.
MAX_CLASH_FRACTION = 0.05


def calpha_mask(model: "ModelFT") -> torch.Tensor:
    """Boolean mask over the model's atoms selecting the C-alpha atoms."""
    names = model.pdb["name"].astype(str).str.strip().values
    return torch.as_tensor(names == "CA", dtype=torch.bool)


def fixed_images_frac(fixed_xyz_frac: torch.Tensor, spacegroup) -> torch.Tensor:
    """Every symmetry image of the fixed atoms, fractional, ``(n_ops * N, 3)``.

    ``x' = S x + t`` for each operation; lattice translations are left to the
    minimum-image step in :func:`clash_fraction`, which is where they belong.
    """
    return spacegroup.expand_positions(fixed_xyz_frac).reshape(-1, 3)


def clash_fraction(
    moving_xyz_frac: torch.Tensor,
    fixed_images: torch.Tensor,
    real_cell,
    cutoff_A: float = CLASH_CUTOFF_A,
) -> float:
    """Fraction of ``moving`` atoms within ``cutoff_A`` of any fixed image.

    Per-axis minimum image: the fractional difference to each image is wrapped
    into ``[-1/2, 1/2)`` before it is converted to Angstrom. For a 3 A question
    that is exact whenever the cell's shortest axis is longer than 6 A, which
    is every crystal there is.

    Parameters
    ----------
    moving_xyz_frac : torch.Tensor
        ``(M, 3)`` fractional coordinates of the candidate's C-alpha atoms.
    fixed_images : torch.Tensor
        ``(P, 3)`` fractional coordinates, from :func:`fixed_images_frac`.
    real_cell : Cell
        Converts fractional differences to Angstrom.
    """
    if moving_xyz_frac.numel() == 0 or fixed_images.numel() == 0:
        return 0.0
    dev = moving_xyz_frac.device
    images = fixed_images.to(dev).to(moving_xyz_frac.dtype)
    B = real_cell.fractional_matrix.to(dev).to(moving_xyz_frac.dtype)
    cutoff2 = float(cutoff_A) ** 2
    hit = torch.zeros(moving_xyz_frac.shape[0], dtype=torch.bool, device=dev)
    # Chunk over the moving atoms so the (M, P, 3) difference stays small.
    chunk = max(1, int(2e6 // max(1, images.shape[0])))
    for a in range(0, moving_xyz_frac.shape[0], chunk):
        d = moving_xyz_frac[a:a + chunk].unsqueeze(1) - images.unsqueeze(0)   # (m, P, 3)
        d = d - torch.round(d)
        d2 = (d @ B.T).pow(2).sum(dim=-1)                                       # (m, P)
        hit[a:a + chunk] = (d2 < cutoff2).any(dim=1)
    return float(hit.to(torch.float32).mean())
