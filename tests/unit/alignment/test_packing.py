"""The packing check: a copy on top of a fixed chain clashes, a copy elsewhere does not.

Symmetry images count -- a candidate sitting on a symmetry mate of the fixed
chain is on top of real atoms -- and so do lattice translations, which the
per-axis minimum image supplies.
"""
import pandas as pd
import pytest
import torch

from torchref.experimental.alignment.packing import (
    calpha_mask, clash_fraction, fixed_images_frac)
from torchref.symmetry.cell import Cell
from torchref.symmetry.spacegroup import SpaceGroup

pytestmark = pytest.mark.unit


@pytest.fixture
def crystal():
    cell = Cell([61.0, 72.0, 83.0, 90.0, 90.0, 90.0], device="cpu")
    sg = SpaceGroup("P 21 21 21", device="cpu")
    g = torch.Generator().manual_seed(3)
    # A compact blob of 150 "C-alpha" atoms near (0.2, 0.3, 0.4).
    frac = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64) \
        + 0.06 * (torch.rand(150, 3, generator=g, dtype=torch.float64) - 0.5)
    return cell, sg, frac


def test_a_copy_on_the_fixed_chain_clashes_completely(crystal):
    cell, sg, frac = crystal
    images = fixed_images_frac(frac, sg)
    assert clash_fraction(frac, images, cell) == pytest.approx(1.0)


def test_a_copy_far_away_does_not_clash(crystal):
    cell, sg, frac = crystal
    images = fixed_images_frac(frac, sg)
    # Shift by a quarter cell along each axis: 15-20 A from every image.
    moved = frac + torch.tensor([0.25, 0.25, 0.25], dtype=torch.float64)
    assert clash_fraction(moved, images, cell) == 0.0


def test_a_copy_on_a_symmetry_mate_clashes(crystal):
    """The candidate at S x + t of the fixed chain is on top of real atoms."""
    cell, sg, frac = crystal
    images = fixed_images_frac(frac, sg)
    S = sg.matrices.to(torch.float64)[1]
    t = sg.translations.to(torch.float64)[1]
    mate = frac @ S.T + t
    assert clash_fraction(mate, images, cell) == pytest.approx(1.0)


def test_lattice_translations_are_the_same_site(crystal):
    cell, sg, frac = crystal
    images = fixed_images_frac(frac, sg)
    assert clash_fraction(frac + torch.tensor([1.0, -2.0, 3.0]), images, cell) == pytest.approx(1.0)


def test_calpha_mask_reads_the_atom_names():
    class _M:
        pdb = pd.DataFrame({"name": [" N  ", " CA ", " C  ", " O  ", " CA "]})
    assert calpha_mask(_M()).tolist() == [False, True, False, False, True]
