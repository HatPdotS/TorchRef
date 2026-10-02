"""``symmetry_image_positions`` and ``is_symmetry_image``, the one image construction.

The non-bonded pair builder forms the images it searches with this function and the
non-bonded scoring places the stored pairs with it. These pin the contract both rely
on: ``B (R B^-1 x + t + n)`` for every entry, the identity included, with a pure
lattice translation kept as a real image.
"""

import gemmi
import pytest
import torch

from torchref.base.coordinates import is_symmetry_image, symmetry_image_positions
from torchref.config import get_float_dtype, get_int_dtype
from torchref.symmetry import SpaceGroup
from torchref.symmetry.cell import Cell

pytestmark = pytest.mark.unit

# Tolerance on a position in Å. The coordinates reach ~100 Å, so float32 rounding of
# the fractional round trip is ~1e-5 Å.
_ATOL = 1e-4


@pytest.fixture(scope="module")
def crystal(pdb_dir):
    """1DAW (C 1 2 1, monoclinic, so B is not diagonal): coordinates, cell, group."""
    st = gemmi.read_structure(str(pdb_dir / "1DAW.pdb"))
    xyz = torch.tensor(
        [[a.pos.x, a.pos.y, a.pos.z] for ch in st[0] for r in ch for a in r][:200],
        dtype=get_float_dtype(),
    )
    c = st.cell
    cell = Cell([c.a, c.b, c.c, c.alpha, c.beta, c.gamma])
    sg = SpaceGroup(st.spacegroup_hm)
    tables = (
        sg.matrices,
        sg.translations,
        cell.fractional_matrix,
        cell.inv_fractional_matrix,
    )
    return xyz, cell, sg, tables


def _per_point(n_points, op, offset):
    ops = torch.full((n_points,), op, dtype=get_int_dtype())
    offsets = torch.tensor(offset, dtype=get_int_dtype()).expand(n_points, 3)
    return ops, offsets


def test_matches_the_space_group_expansion(crystal):
    xyz, cell, sg, tables = crystal
    expanded = sg.expand_positions(cell.cartesian_to_fractional(xyz))
    for op in range(sg.n_ops):
        for offset in ([0, 0, 0], [1, 0, -2], [-3, 2, 1]):
            ops, offsets = _per_point(len(xyz), op, offset)
            got = symmetry_image_positions(xyz, ops, offsets, *tables)
            shift = torch.tensor(offset, dtype=xyz.dtype)
            want = cell.fractional_to_cartesian(expanded[op] + shift)
            torch.testing.assert_close(got, want, atol=_ATOL, rtol=0)


def test_a_pure_lattice_translation_is_an_image(crystal):
    xyz, cell, _, tables = crystal
    ops, offsets = _per_point(len(xyz), 0, [2, -1, 1])
    got = symmetry_image_positions(xyz, ops, offsets, *tables)
    shift = cell.fractional_to_cartesian(offsets[0].to(xyz.dtype))
    torch.testing.assert_close(got, xyz + shift, atol=_ATOL, rtol=0)
    assert bool(is_symmetry_image(ops, offsets).all())


def test_the_identity_returns_the_point(crystal):
    xyz, _, _, tables = crystal
    ops, offsets = _per_point(len(xyz), 0, [0, 0, 0])
    got = symmetry_image_positions(xyz, ops, offsets, *tables)
    torch.testing.assert_close(got, xyz, atol=_ATOL, rtol=0)
    assert not bool(is_symmetry_image(ops, offsets).any())
    torch.testing.assert_close(
        symmetry_image_positions(xyz, ops, None, *tables), got, atol=0, rtol=0
    )


def test_is_symmetry_image_looks_at_operation_and_offset():
    ops = torch.tensor([0, 0, 1, 1], dtype=get_int_dtype())
    offsets = torch.tensor(
        [[0, 0, 0], [0, -1, 0], [0, 0, 0], [1, 0, 0]], dtype=get_int_dtype()
    )
    assert is_symmetry_image(ops, offsets).tolist() == [False, True, True, True]


def test_broadcasting_matches_one_entry_per_point(crystal):
    xyz, _, sg, tables = crystal
    ops = torch.tensor([0, 1, 0, sg.n_ops - 1], dtype=get_int_dtype())
    offsets = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [-1, 2, 0], [1, 1, 1]], dtype=get_int_dtype()
    )
    grid = symmetry_image_positions(xyz[:, None, :], ops, offsets, *tables)
    assert grid.shape == (len(xyz), len(ops), 3)
    flat = symmetry_image_positions(
        xyz.repeat_interleave(len(ops), dim=0),
        ops.repeat(len(xyz)),
        offsets.repeat(len(xyz), 1),
        *tables,
    )
    torch.testing.assert_close(grid.reshape(-1, 3), flat, atol=_ATOL, rtol=0)


def test_gradient_is_the_cartesian_rotation(crystal):
    xyz, cell, sg, tables = crystal
    op = 1
    ops, offsets = _per_point(1, op, [1, 0, -1])

    def image(x):
        return symmetry_image_positions(x[None, :], ops, offsets, *tables)[0]

    jacobian = torch.autograd.functional.jacobian(image, xyz[0].clone())
    B, B_inv = cell.fractional_matrix, cell.inv_fractional_matrix
    want = B @ sg.matrices[op].to(B.dtype) @ B_inv
    torch.testing.assert_close(jacobian, want, atol=1e-5, rtol=0)
