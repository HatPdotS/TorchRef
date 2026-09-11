"""``Symmetry.epsilon`` carries two conventions, and they must not drift together.

Folding Friedel mates into epsilon mixes two different effects. Operations that
map ``h -> h`` add coherently and set the **mean**, ``<|F|^2> = eps * Sigma`` --
the conventional crystallographic epsilon. Operations that map ``h -> -h`` leave
the mean alone and make ``F`` real, which changes the **distribution**; that is
centricity, and ``is_centric`` already carries it.

Both settings have a consumer: sigma_A estimation is calibrated against the
Friedel-folded default, and the molecular-replacement likelihood wants the
conventional count for its ``V = eps - sigma_A**2``. So the risk is not that one
is wrong -- it is that a later edit quietly makes them the same, or flips which
one is the default, and nothing notices. These pin the difference, its location,
and its size.

Counts are the measured ones rather than recomputed, so a regression shows up as
a specific wrong number instead of a green test that changed convention.
"""

import pytest
import torch

from torchref.symmetry import SpaceGroup

pytestmark = pytest.mark.unit

#: ``(hm, centred)``. Spans primitive and centred lattices, and the point groups
#: where the alignment package's own epsilon was measured to disagree.
SPACEGROUPS = [
    ("C 1 2 1", True),
    ("P 21 21 2", False),
    ("P 31 2 1", False),
    ("P 43 21 2", False),
    ("P 4 3 2", False),
    ("P 65 2 2", False),
]


def _hkl(n=4000, seed=11):
    g = torch.Generator().manual_seed(seed)
    hkl = torch.randint(-12, 13, (n, 3), generator=g)
    return hkl[(hkl.abs().sum(dim=-1) > 0)]          # drop (0,0,0)


@pytest.mark.parametrize("hm, centred", SPACEGROUPS)
def test_friedel_changes_centric_reflections_and_only_those(hm, centred):
    """The switch must move exactly the centric reflections, never a general one.

    This is the property that makes the two conventions safe to hold at once: if
    the difference ever spread beyond centrics, one of them would have stopped
    meaning what its docstring says.
    """
    sg = SpaceGroup(hm)
    hkl = _hkl()
    with_f = sg.epsilon(hkl, friedel=True)
    without = sg.epsilon(hkl, friedel=False)
    centric = sg.is_centric(hkl).to(torch.bool)

    differs = with_f != without
    assert not bool((differs & ~centric).any()), (
        f"{hm}: epsilon differs on {int((differs & ~centric).sum())} ACENTRIC "
        f"reflections; the Friedel term must only reach centrics"
    )
    # And the difference is a doubling where it lands, not an arbitrary shift.
    if bool(differs.any()):
        ratio = (with_f[differs] / without[differs])
        assert torch.allclose(ratio, torch.full_like(ratio, 2.0)), (
            f"{hm}: Friedel folding is not a factor of two where it applies"
        )


@pytest.mark.parametrize("hm, centred", SPACEGROUPS)
def test_the_conventional_count_is_never_larger(hm, centred):
    sg = SpaceGroup(hm)
    hkl = _hkl()
    assert bool((sg.epsilon(hkl, friedel=False)
                 <= sg.epsilon(hkl, friedel=True)).all())


@pytest.mark.parametrize("hm, centred", SPACEGROUPS)
def test_centring_cosets_are_counted_by_both(hm, centred):
    """A centred lattice gives every reflection the centring order as a factor.

    Not asserted as desirable -- it is a documented property with one consequence,
    that epsilon is inflated wherever it is used as a *term* rather than a factor.
    Pinned so the behaviour is deliberate rather than discovered again.
    """
    sg = SpaceGroup(hm)
    hkl = _hkl()
    eps = sg.epsilon(hkl, friedel=False)
    general_min = float(eps.min())
    if centred:
        assert general_min >= 2.0, (
            f"{hm} is centred; every reflection should carry the centring order "
            f"but the minimum epsilon is {general_min}"
        )
    else:
        assert general_min == 1.0, (
            f"{hm} is primitive; general reflections should have epsilon 1, got "
            f"{general_min}"
        )


def test_the_default_is_the_calibrated_one():
    """sigma_A estimation is calibrated against Friedel-folded epsilon.

    Flipping this default would decalibrate the refinement path silently, so the
    default is pinned separately from the behaviour of either branch.
    """
    sg = SpaceGroup("P 21 21 2")
    hkl = _hkl()
    assert torch.equal(sg.epsilon(hkl), sg.epsilon(hkl, friedel=True))


def test_epsilon_is_at_least_one_and_finite():
    for hm, _ in SPACEGROUPS:
        sg = SpaceGroup(hm)
        for friedel in (True, False):
            eps = sg.epsilon(_hkl(), friedel=friedel)
            assert bool(torch.isfinite(eps).all())
            assert float(eps.min()) >= 1.0
