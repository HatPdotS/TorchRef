"""Value identity of :class:`~torchref.symmetry.spacegroup.SpaceGroup` and the
setting-preserving round trip through gemmi."""

import gemmi
import pytest

from torchref.symmetry import SpaceGroup


@pytest.mark.unit
def test_same_number_different_setting_are_unequal():
    a = SpaceGroup("P 1 21 1")
    b = SpaceGroup("P 1 1 21")
    assert a.number == b.number == 4
    assert a != b
    assert hash(a) != hash(b)
    # The reason the identity must be setting-aware: the screw axis moves.
    assert a.grid_requirements() != b.grid_requirements()


@pytest.mark.unit
def test_same_setting_compares_and_hashes_equal():
    a = SpaceGroup("P 21 21 21")
    b = SpaceGroup(19)
    assert a == b
    assert hash(a) == hash(b)
    assert a.key == b.key == a.xhm


@pytest.mark.unit
def test_copy_is_equal():
    a = SpaceGroup("C 1 2 1")
    assert a.copy() == a
    assert hash(a.copy()) == hash(a)


@pytest.mark.unit
def test_equality_with_gemmi_spacegroup_is_setting_aware():
    a = SpaceGroup("P 1 21 1")
    assert a == gemmi.find_spacegroup_by_name("P 1 21 1")
    assert a != gemmi.find_spacegroup_by_name("P 1 1 21")
    assert (a == "P 1 21 1") is False


@pytest.mark.unit
@pytest.mark.parametrize("xhm", ["R 3:R", "R 3:H", "P 4/n:1", "P 4/n:2"])
def test_rewrapping_preserves_the_setting(xhm):
    sg = SpaceGroup(xhm)
    assert sg.xhm == xhm
    assert SpaceGroup(sg).xhm == xhm
    assert sg._gemmi.xhm() == xhm
    assert SpaceGroup(sg).n_ops == sg.n_ops
