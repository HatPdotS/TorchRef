"""Copy placement, periodic boxes and copy presence of a crystal layout.

Placement is pure tensor arithmetic and must be differentiable; the box must satisfy
OpenMM's reduced-form rules for every real cell, the hexagonal boundary case included;
presence must keep one copy per distinct site of a molecule on a symmetry element and
every copy of a molecule in a general position.
"""

import numpy as np
import pytest
import torch

from torchref.experimental.mm.layout import CrystalLayout, reduce_box
from torchref.symmetry import Cell, SpaceGroup


def _cubic(spacegroup="P 1", a=10.0):
    return Cell([a, a, a, 90.0, 90.0, 90.0]), SpaceGroup(spacegroup)


@pytest.mark.unit
def test_isolated_returns_the_coordinates():
    """The isolated layout passes coordinates through untouched."""
    layout = CrystalLayout.isolated()
    xyz = torch.randn(5, 3)
    out = layout.positions(xyz)
    assert out.shape == (1, 5, 3)
    assert torch.equal(out[0], xyz)
    assert not layout.periodic


@pytest.mark.unit
def test_quasi_crystal_tiles_along_a():
    """Member d·N_sym + j sits in cell d along a, placed by operation j."""
    cell, sg = _cubic("P 1 2 1")
    layout = CrystalLayout.quasi_crystal(cell, sg, n_disorder=3)
    assert layout.n_copies == 6 and layout.n_sources == 6
    one = torch.tensor([[1.0, 2.0, 3.0]])
    out = layout.positions(one.expand(6, 1, 3).contiguous())
    # P2 operation 1 is (-x, y, -z); cell d adds d·a.
    expected = torch.tensor(
        [[1.0, 2, 3], [-1, 2, -3], [11, 2, 3], [9, 2, -3], [21, 2, 3], [19, 2, -3]]
    )
    assert torch.allclose(out[:, 0], expected, atol=1e-5)
    assert np.allclose(np.diag(layout.box), [30.0, 10.0, 10.0])


@pytest.mark.unit
def test_translation_is_fractional():
    """A screw axis translation of 1/2 along c moves half a cell length."""
    cell, sg = _cubic("P 1 1 21")
    layout = CrystalLayout.unit_cell(cell, sg, cutoff=4.0)
    out = layout.positions(torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.allclose(out[1, 0], torch.tensor([-1.0, -2.0, 8.0]), atol=1e-5)


@pytest.mark.unit
def test_gradient_reaches_every_source():
    """Each copy's gradient returns to its own coordinate set, rotated back."""
    cell, sg = _cubic("P 1 2 1")
    layout = CrystalLayout.quasi_crystal(cell, sg, n_disorder=2)
    xyz = torch.randn(4, 3, 3, requires_grad=True)
    weights = torch.randn(4, 3, 3)
    (layout.positions(xyz) * weights).sum().backward()
    rotation = torch.as_tensor(layout.rotation, dtype=xyz.dtype)
    expected = torch.einsum("cji,cnj->cni", rotation, weights)
    assert torch.allclose(xyz.grad, expected, atol=1e-6)


@pytest.mark.unit
def test_unit_cell_repeats_until_the_cutoff_fits():
    """Every box height is at least twice the cutoff."""
    cell, sg = _cubic("P 21 21 21", a=12.0)
    layout = CrystalLayout.unit_cell(cell, sg, cutoff=10.0)
    assert np.allclose(np.diag(layout.box), [24.0, 24.0, 24.0])
    assert layout.n_copies == 4 * 8
    assert set(layout.source.tolist()) == {0}


@pytest.mark.unit
def test_hexagonal_cell_reduces():
    """3GR5's hexagonal cell puts b_x on the boundary; the reduced box is strict."""
    cell = Cell([85.7, 85.7, 141.4, 90.0, 90.0, 120.0])
    layout = CrystalLayout.unit_cell(cell, SpaceGroup("P 65 2 2"), cutoff=10.0)
    a, b, c = layout.box.T
    assert a[1] == a[2] == b[2] == 0.0
    assert abs(b[0]) < a[0] / 2 and abs(c[0]) < a[0] / 2 and abs(c[1]) < b[1] / 2
    assert layout.n_copies == 12
    assert np.isclose(abs(np.linalg.det(layout.box)), float(cell.volume), rtol=1e-6)


@pytest.mark.unit
def test_reduction_keeps_the_lattice():
    """Reducing a skewed box subtracts lattice vectors only."""
    box = np.array([[10.0, 8.0, 7.0], [0.0, 9.0, 6.0], [0.0, 0.0, 11.0]])
    reduced = reduce_box(box)
    integer = np.linalg.solve(box, reduced)
    assert np.allclose(integer, np.round(integer))
    assert np.isclose(np.linalg.det(reduced), np.linalg.det(box))


@pytest.mark.unit
def test_to_source_frame_inverts_positions():
    """Placing and taking back returns the source coordinates for every copy."""
    cell = Cell([30.0, 35.0, 40.0, 90.0, 100.0, 90.0])
    layout = CrystalLayout.unit_cell(cell, SpaceGroup("C 1 2 1"), cutoff=5.0)
    xyz = torch.randn(7, 3, generator=torch.Generator().manual_seed(0)) * 5
    back = layout.to_source_frame(layout.positions(xyz).numpy())
    scale = float(xyz.abs().max())
    assert np.allclose(back, xyz.numpy()[None], rtol=1e-4, atol=1e-4 * scale)


@pytest.mark.unit
def test_special_position_molecule_is_kept_once_per_site():
    """An ion on a two-fold axis is in one of its two coincident copies."""
    cell, sg = _cubic("P 1 2 1")
    layout = CrystalLayout.unit_cell(cell, sg, cutoff=4.0)
    # Molecule 0 on the two-fold (x = z = 0), molecule 1 in a general position.
    xyz = np.array([[[0.0, 3.0, 0.0], [2.0, 3.0, 4.0]]])
    present = layout.presence(xyz, np.array([0, 1]), np.array([True, True]), cutoff=1.5)
    assert present.tolist() == [[True, True], [False, True]]


@pytest.mark.unit
def test_overlap_is_found_across_the_box_boundary():
    """Coinciding copies one lattice vector apart are the same site."""
    cell, sg = _cubic("P 1 2 1")
    layout = CrystalLayout.unit_cell(cell, sg, cutoff=4.0)
    # (5, y, 5) maps to (-5, y, -5), which is (5, y, 5) one cell over.
    xyz = np.array([[[5.0, 1.0, 5.0]]])
    present = layout.presence(xyz, np.array([0]), np.array([True]), cutoff=1.5)
    assert present[:, 0].tolist() == [True, False]


@pytest.mark.unit
def test_hydrogens_do_not_decide_presence():
    """Only heavy atoms count: a water's hydrogens may differ between its copies."""
    cell, sg = _cubic("P 1 2 1")
    layout = CrystalLayout.unit_cell(cell, sg, cutoff=4.0)
    xyz = np.array([[[0.0, 3.0, 0.0], [0.9, 3.3, 0.2], [-0.3, 3.3, 0.9]]])
    present = layout.presence(
        xyz, np.zeros(3, dtype=np.int64), np.array([True, False, False]), cutoff=1.5
    )
    assert present[:, 0].tolist() == [True, False]


@pytest.mark.unit
def test_presence_off_and_non_periodic_keep_everything():
    cell, sg = _cubic("P 1 2 1")
    xyz = np.array([[[0.0, 3.0, 0.0]]])
    periodic = CrystalLayout.unit_cell(cell, sg, cutoff=4.0)
    assert periodic.presence(xyz, np.array([0]), np.array([True]), 0.0).all()
    isolated = CrystalLayout.isolated()
    assert isolated.presence(xyz, np.array([0]), np.array([True]), 1.5).all()


@pytest.mark.unit
def test_wrong_source_count_raises():
    cell, sg = _cubic("P 1 2 1")
    layout = CrystalLayout.quasi_crystal(cell, sg, n_disorder=2)
    with pytest.raises(ValueError, match=r"Expected \(4, n_atoms, 3\)"):
        layout.positions(torch.randn(3, 2, 3))
    with pytest.raises(ValueError, match="n_disorder"):
        CrystalLayout.quasi_crystal(cell, sg, n_disorder=0)


@pytest.mark.unit
def test_overlap_of_different_molecules_is_reported():
    """A second ion placed on the symmetry copy of the first is named, not merged."""
    cell, sg = _cubic("P 1 2 1")
    layout = CrystalLayout.unit_cell(cell, sg, cutoff=4.0)
    # (-2, 3, -4) is where operation 1 puts (2, 3, 4): the two ions coincide.
    xyz = np.array([[[2.0, 3.0, 4.0], [-2.0, 3.0, -4.0], [0.0, 0.0, 3.0]]])
    molecules = np.array([0, 1, 2])
    heavy = np.ones(3, dtype=bool)
    present = layout.presence(xyz, molecules, heavy, cutoff=1.5)
    assert present.all()
    assert layout.overlaps(xyz, molecules, heavy, present, cutoff=1.5) == [
        (0, 0, 1, 1),
        (1, 0, 0, 1),
    ]


@pytest.mark.unit
def test_overlap_report_sees_across_the_boundary():
    cell, sg = _cubic("P 1")
    layout = CrystalLayout.unit_cell(cell, sg, cutoff=4.0)
    xyz = np.array([[[0.2, 5.0, 5.0], [9.9, 5.0, 5.0]]])
    molecules = np.array([0, 1])
    heavy = np.ones(2, dtype=bool)
    present = layout.presence(xyz, molecules, heavy, cutoff=1.5)
    assert layout.overlaps(xyz, molecules, heavy, present, cutoff=1.5) == [(0, 0, 0, 1)]
