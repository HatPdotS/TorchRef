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

Everything here runs at the configured precision on the configured device, so
the expansion is exercised where it actually runs. That fixes what the budgets
can say. At float32 the difference between two groupings is dominated by the
accumulation's own rounding, not by the grouping, and that floor is not
portable: the three grouping cases cost 5.5e-08 to 1.5e-07 on CPU, 4.2e-07 to
7.4e-07 on this machine's MPS, and up to 2.1e-06 on the project's MPS CI,
because the backends reduce in a different order and MPS varies further by GPU
and torch version. So a single working-precision budget covers them all --
see ``_WORKING_PRECISION_TOLERANCE`` for where the number comes from and what
it costs.
"""

import math

import pytest
import torch

import torchref.experimental.alignment.frf.data_mr as dm
from torchref.config import get_default_device, get_float_dtype
from torchref.experimental.alignment.frf.data_mr import bessel_sh_expand

pytestmark = pytest.mark.unit

#: A key this fine puts every reflection in its own group: the exact sum.
_UNGROUPED = 10 ** 16

#: What any of these comparisons is allowed to cost at the working precision.
#:
#: Measured, worst case over the assertions below: 1.9e-06 on CPU float32 and
#: 1.4e-06 on this machine's MPS, with the grouping cases reaching 2.1e-06 on
#: the project's MPS CI. 1e-04 clears the worst observed figure by ~50x, which is
#: ample for the backend spread -- the same case differs 3x between two MPS
#: devices, and CPU is worse than MPS on the lattice.
#:
#: It still sits far below the accuracy of what feeds it. The sphere/voxel
#: discretisation upstream of the structure factors is itself good to about
#: 5e-03 rel L2 (``tests/unit/base/test_canonical_sphere_cpu.py``), so a budget
#: here of 1e-04 is ~50x tighter than the input it is expanding. Downstream is
#: the same story: rotation-function scores moved 1.8e-07 to 2.5e-04 relative
#: across the float32 migration and the peak lists on 1DAW, 3K7M, 2DQ6 and 4BX9
#: came back in the same order.
#:
#: What it gives up, and this is the real cost: at this width the grouping
#: comparison notices neither a 10x coarser key (1.6e-06) nor a 100x one
#: (4.1e-05). Only 1000x (1.3e-03) trips it. These are smoke tests for gross
#: breakage on the device the code actually runs on, not tripwires for a
#: degraded key -- ``test_matches_an_independent_direct_summation`` is the one
#: that would still catch a dropped term, because it compares against a
#: reference rather than against the expansion at another grouping. Given the
#: 5e-03 upstream, a key error of 4.1e-05 is not a correctness problem anyway;
#: it would be a performance-versus-accuracy choice made by accident.
_WORKING_PRECISION_TOLERANCE = 1e-4

#: Alias kept for the grouping cases, which is what the assertion message names.
_GROUPING_TOLERANCE = _WORKING_PRECISION_TOLERANCE

#: The direct-summation check compares against a host-double reference rather
#: than against the expansion at another grouping, so it carries the working
#: precision's whole error. Measured 8.1e-07; same budget, same reasoning.
_DIRECT_SUM_TOLERANCE = _WORKING_PRECISION_TOLERANCE


def _to_working(s, I, dtype=None, device=None):
    """Place a host-double set at the configured precision and device.

    Drawn in double and cast once, rather than generated at the working dtype,
    so the *set* is the same whatever precision it is evaluated in -- otherwise
    a tolerance measured at one dtype is not comparable with the same tolerance
    at another, because the sample moved too.
    """
    dtype = get_float_dtype() if dtype is None else dtype
    device = get_default_device() if device is None else device
    return (s.to(device=device, dtype=dtype), I.to(device=device, dtype=dtype))


def _random_set(seed, n, dtype=None, device=None):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(n, 3, generator=g, dtype=torch.float64)
    s = s / s.norm(dim=-1, keepdim=True) * (
        0.07 + 0.18 * torch.rand(n, 1, generator=g, dtype=torch.float64))
    I = torch.randn(n, generator=g, dtype=torch.float64)
    return _to_working(s, I, dtype, device)


def _grid_set(k=8, step=0.013, dtype=None, device=None):
    """A lattice, where |s| degeneracy is exact and the grouping pays most."""
    idx = torch.arange(-k, k + 1, dtype=torch.float64)
    a, b, c = torch.meshgrid(idx, idx, idx, indexing="ij")
    s = torch.stack([a.reshape(-1), b.reshape(-1), c.reshape(-1)], dim=-1) * step
    s = s[s.norm(dim=-1) > 1e-9]
    g = torch.Generator().manual_seed(4)
    I = torch.randn(s.shape[0], generator=g, dtype=torch.float64)
    return _to_working(s, I, dtype, device)


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
    # "Loss-free" is a float64 statement: on a lattice the |s| degeneracy is
    # exact, so in double this lands at 1e-15. At float32 the accumulation floor
    # is 1.9e-06 (CPU) / 1.4e-06 (MPS) and swamps it, so what is asserted here
    # is that the lattice case is no worse than any other -- the exactness claim
    # is only visible in double.
    assert rel < _WORKING_PRECISION_TOLERANCE, (
        f"lattice grouping is not loss-free: {rel:.2e}"
    )


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

    ab, I_w = _to_working(torch.cat([a, b]), I)
    ba, _ = _to_working(torch.cat([b, a]), I)
    fwd = bessel_sh_expand(ab, I_w, L=L, bessel_h_scale=scale).coeffs
    rev = bessel_sh_expand(ba, I_w, L=L, bessel_h_scale=scale).coeffs
    rel = (fwd - rev).abs().max().item() / max(fwd.abs().max().item(), 1e-300)
    # Not loosened with the others: the mean of two numbers does not depend on
    # their order in floating point either, so this measures 0.0 exactly at
    # float32 and float64 alike. A budget here would only hide a real failure.
    assert rel == 0.0, (
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
def test_matches_an_independent_direct_summation(L):
    """The whole expansion, against a reference that shares no code with it.

    The reference stays in host double while the expansion runs at the working
    precision, so this is the one test here that measures the expansion's true
    accuracy rather than its self-consistency. That is also why its budget is
    the widest: it carries the working precision's whole error, not just the
    grouping's share of it.
    """
    s64, I64 = _random_set(seed=31, n=120, dtype=torch.float64, device="cpu")
    ref = _reference_expansion(s64, I64, L, 24.0)
    got = bessel_sh_expand(*_random_set(seed=31, n=120),
                           L=L, bessel_h_scale=24.0).coeffs
    assert got.shape == ref.shape
    # `.cpu()` before widening: complex128 cannot be materialised on a backend
    # that has no double.
    got = got.cpu().to(ref.dtype)
    rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-300)
    assert rel < _DIRECT_SUM_TOLERANCE, (
        f"L={L}: differs from a direct summation by {rel:.2e}"
    )


def test_the_antipodal_copy_would_only_double_the_result():
    """Why the expansion no longer concatenates ``-s``.

    Only even ``l`` are computed, and ``Y_lm(-s_hat) = (-1)^l Y_lm(s_hat)``, so
    for even ``l`` the antipodal copy contributes exactly what the original does.
    The intensity is duplicated verbatim and ``|s|`` is unchanged, so the whole
    coefficient array doubles and nothing about its *shape* changes -- which is
    why removing it rescales the rotation function by four and reorders nothing.

    Pinned here rather than argued in a comment: if a future change made odd ``l``
    carry signal, this equality would break and the removal would need revisiting.
    """
    s, I = _random_set(seed=17, n=300)
    single = bessel_sh_expand(s, I, L=17, bessel_h_scale=30.0).coeffs
    doubled = bessel_sh_expand(
        torch.cat([s, -s], dim=0), torch.cat([I, I], dim=0),
        L=17, bessel_h_scale=30.0,
    ).coeffs
    scale = single.abs().max().clamp(min=1e-300)
    rel = ((doubled - 2.0 * single).abs().max() / scale).item()
    assert rel < _WORKING_PRECISION_TOLERANCE, (
        f"the antipodal copy is not an exact factor of two: {rel:.2e} relative. "
        f"Removing it was justified on that being exact."
    )
