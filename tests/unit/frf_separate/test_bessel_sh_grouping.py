"""What the SH-Bessel expansion's grouping is allowed to cost.

`bessel_sh_expand` does not sum over reflections one at a time. It groups them
by ``(|s|, cos theta)`` -- both factors are constant within a group -- and sums
over groups, which is what makes the cost tractable at L = 100. Two properties
make that safe, and both are cheap to lose silently:

* the negative-m half of the result is redundant, exactly, so only half is
  computed and the rest mirrored;
* the group representative is the group *mean*, so the grouping's error stays
  far below what the rest of the chain contributes.

The reference for the second is an expansion with a grouping key so fine that
every reflection is its own group -- i.e. the ungrouped sum. Comparing against
the previous implementation instead would only show that two approximations
agree with each other.
"""

import math

import pytest
import torch

import torchref.experimental.alignment.frf.data_mr as dm
from torchref.experimental.alignment.frf.data_mr import bessel_sh_expand

pytestmark = pytest.mark.unit

#: A key this fine puts every reflection in its own group: the exact sum.
_UNGROUPED = 10 ** 16

#: Phaser buckets cos(theta) at 1e-3 (`lib/sphericalY.h:43`) and evaluates the
#: Legendre polynomials once per bucket, which costs it about 2e-5 relative on
#: these coefficients. Staying two orders inside that is ample; the threshold is
#: set from the measured 1.2e-8 to 4.8e-8 with headroom, not from taste.
_GROUPING_TOLERANCE = 5e-7


def _random_set(seed, n, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(n, 3, generator=g, dtype=dtype)
    s = s / s.norm(dim=-1, keepdim=True) * (
        0.07 + 0.18 * torch.rand(n, 1, generator=g, dtype=dtype))
    return s, torch.randn(n, generator=g, dtype=dtype)


def _grid_set(k=8, step=0.013):
    """A lattice, where |s| degeneracy is exact and the grouping pays most."""
    idx = torch.arange(-k, k + 1, dtype=torch.float64)
    a, b, c = torch.meshgrid(idx, idx, idx, indexing="ij")
    s = torch.stack([a.reshape(-1), b.reshape(-1), c.reshape(-1)], dim=-1) * step
    s = s[s.norm(dim=-1) > 1e-9]
    g = torch.Generator().manual_seed(4)
    return s, torch.randn(s.shape[0], generator=g, dtype=torch.float64)


@pytest.fixture
def ungrouped():
    """Run the expansion with grouping effectively disabled."""
    def run(s, I, **kw):
        ks, kc = dm._GROUP_SCALE_S, dm._GROUP_SCALE_COS
        dm._GROUP_SCALE_S = dm._GROUP_SCALE_COS = _UNGROUPED
        try:
            return bessel_sh_expand(s, I, **kw).coeffs
        finally:
            dm._GROUP_SCALE_S, dm._GROUP_SCALE_COS = ks, kc
    return run


@pytest.mark.parametrize("L,scale", [(17, 30.0), (33, 48.0), (65, 64.0)])
def test_grouping_error_stays_far_below_the_reference_implementation(
        L, scale, ungrouped):
    """The grouped sum must track the ungrouped one."""
    s, I = _random_set(seed=21, n=5000)
    ref = ungrouped(s, I, L=L, bessel_h_scale=scale)
    got = bessel_sh_expand(s, I, L=L, bessel_h_scale=scale).coeffs
    rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-300)
    assert rel < _GROUPING_TOLERANCE, (
        f"L={L}: grouping cost {rel:.2e} relative, over the {_GROUPING_TOLERANCE:.0e} "
        f"budget. A coarser key, or a group representative that is not the mean, "
        f"will do this."
    )


def test_a_lattice_groups_without_loss(ungrouped):
    """On a lattice the degeneracy is exact, so the grouping is free."""
    s, I = _grid_set()
    ref = ungrouped(s, I, L=65, bessel_h_scale=64.0)
    got = bessel_sh_expand(s, I, L=65, bessel_h_scale=64.0).coeffs
    rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-300)
    assert rel < 1e-12, f"lattice grouping is not loss-free: {rel:.2e}"


@pytest.mark.parametrize("L", [17, 33, 65])
def test_negative_m_is_the_conjugate_of_positive_m(L):
    """``c[n,l,-m] = (-1)^m conj(c[n,l,+m])``, which is why only half is summed.

    The intensity, the radial weight and the Legendre factor are all real, and
    P_{l,|m|} does not distinguish +m from -m, so m enters only through the
    azimuthal phase. If this ever fails, the mirrored half of the array is
    wrong and the rotation function is being fed a non-Hermitian Patterson.
    """
    s, I = _random_set(seed=5, n=3000)
    c = bessel_sh_expand(s, I, L=L, bessel_h_scale=48.0).coeffs
    for m in range(1, L):
        pos = c[:, :, (L - 1) + m]
        neg = c[:, :, (L - 1) - m]
        assert torch.equal(neg, ((-1.0) ** m) * pos.conj()), f"m={m} mirror broken"


def test_the_group_representative_is_the_mean_not_a_member():
    """A scatter assignment leaves an arbitrary member; the mean is centred.

    Two reflections inside one key bin, placed either side of the bin centre:
    with a mean representative the expansion is symmetric under swapping which
    one comes first in the array, with a last-writer-wins representative it is
    not.
    """
    L, scale = 33, 48.0
    eps = 0.4 / dm._GROUP_SCALE_S          # comfortably inside one bin
    base = torch.tensor([[0.10, 0.03, 0.05]], dtype=torch.float64)
    unit = base / base.norm()
    r = base.norm()
    a = unit * (r - eps)
    b = unit * (r + eps)
    I = torch.tensor([1.0, 1.0], dtype=torch.float64)

    fwd = bessel_sh_expand(torch.cat([a, b]), I, L=L, bessel_h_scale=scale).coeffs
    rev = bessel_sh_expand(torch.cat([b, a]), I, L=L, bessel_h_scale=scale).coeffs
    rel = (fwd - rev).abs().max().item() / max(fwd.abs().max().item(), 1e-300)
    assert rel < 1e-13, (
        f"reordering two reflections in the same bin changed the result by "
        f"{rel:.2e}; the representative is order-dependent, so it is not the mean"
    )
