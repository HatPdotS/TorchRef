"""generate_possible_hkl returns the full resolution sphere in every cell.

The reference is a brute-force box twice as wide as the bound |h| <= a / d_min
(likewise k by b, l by c), filtered by d-spacing. In oblique cells 1 / (|a*| d_min)
is smaller than a / d_min, so a box sized by the reciprocal axis lengths misses
part of the sphere; the hexagonal and rhombohedral cells here exercise that.
"""

import math

import pytest
import torch

from torchref.base.reciprocal.hkl import generate_possible_hkl, get_d_spacing

pytestmark = pytest.mark.unit

CELLS = {
    "orthorhombic": (40.0, 50.0, 60.0, 90.0, 90.0, 90.0),
    "monoclinic": (50.0, 60.0, 70.0, 90.0, 105.0, 90.0),
    "hexagonal": (80.0, 80.0, 120.0, 90.0, 90.0, 120.0),
    "rhombohedral_acute": (40.0, 40.0, 40.0, 60.0, 60.0, 60.0),
    "triclinic_skewed": (30.0, 40.0, 50.0, 70.0, 80.0, 60.0),
}
D_MIN = 5.0


def _keys(hkl):
    return set(map(tuple, hkl.tolist()))


def _sphere(cell, d_min):
    n = 2 * math.ceil(max(cell[:3]) / d_min)
    r = torch.arange(-n, n + 1)
    box = torch.cartesian_prod(r, r, r)
    box = box[(box != 0).any(dim=1)]
    inside = box[get_d_spacing(box, cell) >= d_min]
    assert inside.abs().max() < n, "reference box too small"
    return inside


@pytest.mark.parametrize("name", CELLS)
def test_full_sphere(name):
    cell = torch.tensor(CELLS[name], dtype=torch.float64)
    hkl = generate_possible_hkl(cell, D_MIN)

    assert len(_keys(hkl)) == len(hkl)
    assert _keys(hkl) == _keys(_sphere(cell, D_MIN))


@pytest.mark.parametrize("name", CELLS)
def test_friedel_closed_without_origin(name):
    cell = torch.tensor(CELLS[name], dtype=torch.float64)
    hkl = generate_possible_hkl(cell, D_MIN)

    assert (0, 0, 0) not in _keys(hkl)
    assert _keys(hkl) == _keys(-hkl)
    assert bool((get_d_spacing(hkl, cell) >= D_MIN).all())
