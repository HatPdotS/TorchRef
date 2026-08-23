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

The claims about the grouping being *loss-free* are only meaningful at a
precision finer than the loss being ruled out, so the tests that assert
1e-10-and-below take ``double_cpu``. At the working precision (float32) the
floor is float32 epsilon times the accumulation depth -- measured 1.4e-06 and
3.6e-07 on these cases -- which says nothing about the grouping.
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


def test_a_lattice_groups_without_loss(ungrouped, double_cpu):
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


def _reference_expansion(s_vec, intensity, L, bessel_h_scale):
    """A slow, obvious expansion, built from independently-checked parts.

    Sums over reflections one at a time, taking the Legendre factor from
    ``sh._bar_legendre_recurrence`` -- which ``tests/unit/alignment/test_sh.py``
    pins against ``scipy`` -- and the radial factor from
    ``spherical_bessel_table``. No grouping, no shells, no fused loop.

    This exists because the other tests in this file compare
    ``bessel_sh_expand`` against *itself* at a different grouping resolution, so
    a term dropped from the sum appears identically on both sides and cancels.
    One did: a refactor formed the products that feed the shell accumulation
    before writing the sectoral (m = l) entry of the Legendre row, silently
    losing that entry for every even l. Every test here passed, and the only
    signal was a benchmark truth rank moving from 8 to 13.
    """
    from torchref.experimental.alignment.frf.data_mr import spherical_bessel_table
    from torchref.experimental.alignment.sh import _bar_legendre_recurrence

    s_vec = torch.cat([s_vec, -s_vec], dim=0)          # enforce_friedel
    intensity = torch.cat([intensity, intensity], dim=0)

    lmax = L - 1
    lmax_even = lmax if lmax % 2 == 0 else lmax - 1
    N_radial = (lmax_even - 2) // 2 + 1
    u_max = lmax_even + 1

    smag = s_vec.norm(dim=-1).clamp(min=1e-30)
    cos_t = (s_vec[:, 2] / smag).clamp(-1.0, 1.0)
    sin_t = (1.0 - cos_t * cos_t).clamp(min=0.0).sqrt()
    phi = torch.atan2(s_vec[:, 1], s_vec[:, 0])

    barP = _bar_legendre_recurrence(cos_t, sin_t, L)     # (M, L, L)
    x = (bessel_h_scale * smag).clamp(min=1e-30)
    j = spherical_bessel_table(x, u_max)                # (M, u_max+1)

    out = torch.zeros((N_radial, L, 2 * L - 1), dtype=torch.complex128)
    for l in range(2, lmax_even + 1, 2):
        for n in range((lmax_even - l) // 2 + 1):
            u = l + 2 * n + 1
            radial = math.sqrt(2 * u + 1) * j[:, u] / x
            for m in range(-l, l + 1):
                # Y_lm = barP_{l,|m|} * C(m, phi), with C = (-1)^m e^{i m phi}
                # for m >= 0 and e^{i m phi} for m < 0; the expansion uses the
                # conjugate.
                sign = (-1.0) ** m if m >= 0 else 1.0
                phase = torch.polar(torch.ones_like(phi), -m * phi)
                term = (intensity * radial * barP[:, l, abs(m)] * sign) * phase
                out[n, l, (L - 1) + m] = term.sum()
    return out


@pytest.mark.parametrize("L", [9, 13])
def test_matches_an_independent_direct_summation(L, double_cpu):
    """The whole expansion, against a reference that shares no code with it."""
    s, I = _random_set(seed=31, n=120)
    ref = _reference_expansion(s, I, L, 24.0)
    got = bessel_sh_expand(s, I, L=L, bessel_h_scale=24.0).coeffs
    assert got.shape == ref.shape
    rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-300)
    assert rel < 1e-10, f"L={L}: differs from a direct summation by {rel:.2e}"
