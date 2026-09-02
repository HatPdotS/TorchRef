"""The origin freedom a space group leaves a molecular-replacement solution.

Two placements that differ by an allowed origin shift, or by any displacement
along a polar axis, give identical ``|F|``. The translation search must not
return them as different peaks and the pose metric must not count them as
different answers, so both take the group's own account of that freedom.
"""
import pytest
import torch

from torchref.symmetry.spacegroup import SpaceGroup

pytestmark = pytest.mark.unit


def _shifts(name):
    d, p = SpaceGroup(name, device="cpu").origin_shifts()
    return {tuple(round(float(x), 4) for x in row) for row in d}, p.shape[1]


def test_p1_is_all_polar():
    d, n_polar = _shifts("P 1")
    assert n_polar == 3
    assert d == {(0.0, 0.0, 0.0)}, "every shift is polar, so one representative"


def test_p21_has_a_polar_axis_and_half_shifts_in_the_other_two():
    d, n_polar = _shifts("P 1 21 1")
    assert n_polar == 1
    assert len(d) == 4, "one representative per shift, not one per y position"
    xz = {(u[0], u[2]) for u in d}
    assert xz == {(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)}


def test_p212121_family_has_eight_shifts_and_no_polar_axis():
    d, n_polar = _shifts("P 21 21 2")
    assert n_polar == 0
    assert d == {(a, b, c) for a in (0.0, 0.5) for b in (0.0, 0.5) for c in (0.0, 0.5)}


def test_trigonal_321_allows_only_the_half_shift_along_c():
    d, n_polar = _shifts("P 31 2 1")
    assert n_polar == 0
    assert d == {(0.0, 0.0, 0.0), (0.0, 0.0, 0.5)}


def test_cubic_432_allows_the_body_diagonal_half_shift():
    d, n_polar = _shifts("P 4 3 2")
    assert n_polar == 0
    assert d == {(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)}


def test_centred_lattice_counts_its_centring_as_a_lattice_vector():
    """C2: b is polar; x and z are each free modulo 1/2 once (1/2, 1/2, 0) is a
    lattice vector, so (1/2, *, 0) is allowed as well as (0, *, 1/2)."""
    d, n_polar = _shifts("C 1 2 1")
    assert n_polar == 1
    xz = {(u[0], u[2]) for u in d}
    assert xz == {(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)}


def test_zero_shift_comes_first():
    d, _ = SpaceGroup("P 21 21 2", device="cpu").origin_shifts()
    assert torch.all(d[0] == 0)
