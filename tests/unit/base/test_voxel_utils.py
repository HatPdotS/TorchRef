"""``find_relevant_voxels`` offers every voxel within the given radius of an atom."""

import pytest
import torch

from torchref.base.electron_density.solvent_mask import add_to_solvent_mask
from torchref.base.electron_density.voxel_utils import find_relevant_voxels
from torchref.base.fourier import get_real_grid
from torchref.io.pdb import PDBReader
from torchref.symmetry.cell import Cell

pytestmark = pytest.mark.unit


def test_atom_mask_matches_brute_force_on_a_monoclinic_cell(pdb_dir):
    """The candidates plus ``add_to_solvent_mask``'s exact cutoff give the true mask.

    1DAW's cell (C2, beta = 103.56 deg) on a 216x90x69 grid, 20 seeded atoms, r = 2.5 A:
    the mask must hold every voxel whose minimum-image distance to an atom is within r
    and no other, against a brute-force pass over the whole grid. Voxels within 1e-4 A
    of a sphere are float32 ties and are not judged.
    """
    _, cell, _ = PDBReader().read(str(pdb_dir / "1DAW.pdb"))()
    cell = Cell(cell, device="cpu")
    frac, inv_frac = cell.fractional_matrix, cell.inv_fractional_matrix
    dims, r = (216, 90, 69), 2.5
    g = torch.Generator().manual_seed(0)
    xyz_frac = torch.rand(20, 3, generator=g, dtype=torch.float64)
    xyz = (xyz_frac @ frac.double().T).to(frac.dtype)

    grid = get_real_grid(fractional_matrix=frac, gridsize=dims)
    coords, idx = find_relevant_voxels(grid, xyz, r, inv_frac_matrix=inv_frac)
    mask = add_to_solvent_mask(
        coords, idx, torch.zeros(dims, dtype=torch.bool), xyz, r, inv_frac, frac
    )

    axes = [torch.arange(n, dtype=torch.float64) / n for n in dims]
    voxel_frac = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    atom_frac = xyz.double() @ inv_frac.double().T
    nearest = torch.full((voxel_frac.shape[0],), float("inf"), dtype=torch.float64)
    for a in atom_frac:
        d = voxel_frac - a
        d = d - torch.round(d)
        nearest = torch.minimum(nearest, (d @ frac.double().T).norm(dim=-1))
    nearest = nearest.view(dims)

    missed = int((~mask & (nearest <= r - 1e-4)).sum())
    spurious = int((mask & (nearest > r + 1e-4)).sum())
    in_radius = int((nearest <= r).sum())
    assert missed == 0, f"{missed} of {in_radius} in-radius voxels were never offered"
    assert spurious == 0, f"{spurious} voxels outside the radius were marked"
