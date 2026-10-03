"""The two VDW pair searches: the periodic grid (accelerators) and the k-d tree (CPU).

Both take the same symmetry-image table and return ``(i, j, combo_j)`` under one
convention, so the rest of ``build_vdw_restraints_gpu`` cannot tell them apart. The k-d
tree is checked against brute force; the grid is checked against the k-d tree, which
also keeps it exercised on a CPU-only CI now that CPU builds no longer call it.

The grid's cells are ``cell_length / cutoff`` along each axis, which in an oblique cell
leaves them narrower than the cutoff perpendicular to a face. Pairs between that width
and the cutoff can then sit two cells apart and are missed. Both structures here are
oblique (1BYW hexagonal, 3E98 monoclinic), so that shortfall is asserted rather than
assumed away.
"""

import numpy as np
import pytest
import torch

from torchref.config import dtypes
from torchref.model.model import Model
from torchref.topology import nonbonded as nb

CUTOFF = 6.0


def _image_table(path):
    """Steps 1-2 of ``build_vdw_restraints_gpu`` for one model, on CPU.

    Pinned to CPU, not ``TORCHREF_DEVICE``: the k-d tree is the CPU search, and the
    accelerator runners would otherwise put the cell on the GPU next to CPU tensors.
    """
    model = Model(verbose=0, device=torch.device("cpu"))
    model.load_pdb(str(path))
    cell, sg = model.ctx.cell, model.ctx.spacegroup
    xyz = model.xyz().detach().to(dtypes.float)
    op_indices, offsets = nb.prefilter_symop_offsets(cell, sg, xyz, CUTOFF)
    identity = ((op_indices == 0) & (offsets == 0).all(dim=1)).nonzero()[0].item()
    lengths = torch.stack([cell.a, cell.b, cell.c]).to(dtypes.float)
    grid_dims = torch.clamp((lengths / CUTOFF).long(), min=1)
    flat_cell, atom_idx, combo_idx, cart_pos = nb.assign_to_grid(
        xyz, cell, sg, op_indices, offsets, grid_dims
    )
    # Perpendicular width of one grid cell along each axis: lattice-plane spacing
    # V / |face| over the number of cells.
    basis = cell.fractional_to_cartesian(torch.eye(3, dtype=dtypes.float))
    volume = torch.linalg.det(basis).abs()
    faces = [(basis[1], basis[2]), (basis[2], basis[0]), (basis[0], basis[1])]
    widths = [
        (volume / torch.linalg.cross(u, v).norm() / grid_dims[k]).item()
        for k, (u, v) in enumerate(faces)
    ]
    return dict(
        n_atoms=xyz.shape[0], n_combos=len(op_indices), identity=identity,
        grid_dims=grid_dims, flat_cell=flat_cell, atom_idx=atom_idx,
        combo_idx=combo_idx, cart_pos=cart_pos, min_width=min(widths),
    )


def _keys(i, j, c, t):
    """``(i, j, combo_j)`` as one integer each, for set comparison."""
    return ((i * t["n_atoms"] + j) * t["n_combos"] + c).cpu().numpy()


def _distance(keys, t):
    n, m = t["n_atoms"], t["n_combos"]
    pos = t["cart_pos"].reshape(n, m, 3).double()
    keys = torch.as_tensor(keys)
    i, j, c = keys // (n * m), (keys // m) % n, keys % m
    return (pos[i, t["identity"]] - pos[j, c]).norm(dim=1).numpy()


@pytest.fixture(scope="module", params=["1BYW_af.pdb", "3E98.pdb"])
def table(request, pdb_dir):
    return _image_table(pdb_dir / request.param)


def test_kdtree_matches_brute_force(table):
    t = table
    n, m = t["n_atoms"], t["n_combos"]
    # The image table is atom-major; the distance lookup relies on it.
    assert torch.equal(t["atom_idx"], torch.arange(n).repeat_interleave(m))
    assert torch.equal(t["combo_idx"], torch.arange(m).repeat(n))

    pos = t["cart_pos"].reshape(n, m, 3)
    asu = pos[:, t["identity"]]
    images = pos.reshape(n * m, 3)
    expected = []
    for start in range(0, n, 256):
        d = torch.cdist(asu[start:start + 256].double(), images.double(), compute_mode="donot_use_mm_for_euclid_dist")
        i, e = (d < CUTOFF).nonzero(as_tuple=True)
        i = i + start
        j, c = e // m, e % m
        keep = (c != t["identity"]) | (i < j)
        expected.append(_keys(i[keep], j[keep], c[keep], t))
    expected = np.concatenate(expected)

    got = _keys(*nb.find_pairs_kdtree(
        t["cart_pos"], t["atom_idx"], t["combo_idx"], CUTOFF, t["identity"]
    ), t)
    assert np.array_equal(np.sort(expected), np.sort(got))


def test_kdtree_output_convention(table):
    t = table
    i, j, c = nb.find_pairs_kdtree(
        t["cart_pos"], t["atom_idx"], t["combo_idx"], CUTOFF, t["identity"]
    )
    keys = _keys(i, j, c, t)
    assert np.all(np.diff(keys) > 0), "sorted by (i, j, combo_j), no duplicates"
    intra = c == t["identity"]
    assert bool((i[intra] < j[intra]).all()), "intra-ASU pairs once, i < j, no self"
    assert i.dtype == j.dtype == c.dtype == torch.int64


def test_grid_is_the_kdtree_minus_its_cell_width_shortfall(table):
    t = table
    order, cells, starts, lookup = nb.build_cell_list(t["flat_cell"], int(t["grid_dims"].prod()))
    grid = np.unique(_keys(*nb.find_pairs_periodic_grid_v2(
        t["cart_pos"][order], t["atom_idx"][order], t["combo_idx"][order],
        cells, starts, lookup, t["grid_dims"], CUTOFF, t["identity"],
    ), t))
    tree = _keys(*nb.find_pairs_kdtree(
        t["cart_pos"], t["atom_idx"], t["combo_idx"], CUTOFF, t["identity"]
    ), t)

    # Allow for the grid's matmul cdist rounding right at the cutoff.
    only_grid = np.setdiff1d(grid, tree)
    assert np.all(np.abs(_distance(only_grid, t) - CUTOFF) < 1e-4)
    only_tree = np.setdiff1d(tree, grid)
    assert t["min_width"] < CUTOFF, "both test cells are oblique enough to show it"
    assert np.all(_distance(only_tree, t) > t["min_width"] - 1e-4)
    assert len(only_tree) < 0.01 * len(tree)
