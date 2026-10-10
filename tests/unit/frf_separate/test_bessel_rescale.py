"""What the Bessel ladder's on-the-fly rescaling is allowed to cost: nothing.

Miller's downward recurrence for ``j_u(x)`` seeds at an arbitrary magnitude and
renormalises at the end, so the intermediate ladder is inflated by whatever the
true ``j_{n_start}(x)`` happens to be -- 1e157 at the FRF's low-resolution end.
That is fine in float64 and overflows float32 for every ``x`` below about 35,
which is most of the resolution range, so the recurrence rescales as it goes.

The rescale factor is a power of two on purpose: dividing by one decrements the
exponent and leaves the mantissa alone, so the table must come out **bitwise**
equal to the un-rescaled ladder rather than merely close to it. That is the
property these tests pin, against two independent references -- the previous
implementation, transcribed inline, and ``scipy.special.spherical_jn``.
"""

import math

import pytest
import torch

from torchref.experimental.alignment.frf.data_mr import (
    _BESSEL_RESCALE_EXP,
    spherical_bessel_table,
)

pytestmark = pytest.mark.unit

#: Bessel arguments spanning the FRF's range. ``bessel_h_scale = lmax_even *
#: d_min_eff``, so ``x`` runs from ``bessel_h_scale / d_max`` (~1.3 at
#: d_max = 100 A) up to exactly ``lmax_even`` at the high-resolution limit.
_X_VALUES = [1.257, 1.885, 3.0, 5.0, 10.0, 20.0, 40.0, 64.0]

_U_MAX = 65          # lmax_even + 1 at the shipped LMAX_CAP = 64
_N_EXTRA = 25


def _unrescaled_reference(x, u_max, n_extra=_N_EXTRA):
    """The recurrence as it stood before rescaling, transcribed verbatim.

    Kept as a literal copy rather than a call into the module: the point is to
    compare against the *previous* arithmetic, so it must not track any later
    edit to the production function.
    """
    x64 = x.to(torch.float64)
    safe_x = x64.clamp(min=1e-30)
    inv_x = 1.0 / safe_x
    n_start = max(u_max + n_extra, u_max + 2)
    j_high = torch.zeros_like(x64)
    j_mid = torch.ones_like(x64)
    j_table = torch.zeros((u_max + 1, *x64.shape), dtype=torch.float64)
    peak = torch.zeros_like(x64)
    for n in range(n_start, 0, -1):
        j_low = (2.0 * n + 1.0) * inv_x * j_mid - j_high
        if n - 1 <= u_max:
            j_table[n - 1] = j_low
        peak = torch.maximum(peak, j_low.abs())
        j_high = j_mid
        j_mid = j_low
    true_j0 = torch.sin(x64) * inv_x
    true_j0 = torch.where(x64 < 1e-30, torch.ones_like(x64), true_j0)
    computed_j0 = j_table[0]
    safe_j0 = torch.where(
        computed_j0.abs() < 1e-30, torch.ones_like(computed_j0), computed_j0,
    )
    j_table = j_table * (true_j0 / safe_j0).unsqueeze(0)
    perm = list(range(1, j_table.dim())) + [0]
    return j_table.permute(*perm).contiguous(), peak


def test_rescaling_is_bit_identical_to_the_unrescaled_ladder():
    x = torch.tensor(_X_VALUES, dtype=torch.float64)
    ref, _ = _unrescaled_reference(x, _U_MAX)
    got = spherical_bessel_table(x, _U_MAX)
    assert got.shape == ref.shape
    assert torch.equal(got, ref), (
        "rescaling perturbed the table; max relative deviation "
        f"{((got - ref).abs() / ref.abs().clamp(min=1e-300)).max().item():.3e}"
    )


def test_the_rescale_branch_is_actually_exercised():
    """Guard against a vacuous equality test.

    If the ladder never crossed the threshold the comparison above would pass
    while testing nothing, so assert the un-rescaled ladder really does run away
    -- and past float32's ceiling, which is the reason the rescaling exists.
    """
    x = torch.tensor(_X_VALUES, dtype=torch.float64)
    _, peak = _unrescaled_reference(x, _U_MAX)
    threshold = 2.0 ** _BESSEL_RESCALE_EXP
    f32_max = torch.finfo(torch.float32).max
    crossed = int((peak > threshold).sum())
    over_f32 = int((peak > f32_max).sum())
    assert crossed >= len(_X_VALUES) - 2, (
        f"only {crossed} of {len(_X_VALUES)} arguments cross 2**"
        f"{_BESSEL_RESCALE_EXP}; the rescale path is nearly dead"
    )
    assert over_f32 >= 5, (
        f"only {over_f32} arguments overflow float32 (peaks: "
        f"{[f'{v:.2e}' for v in peak.tolist()]})"
    )


def test_matches_scipy_spherical_jn():
    """Independent oracle, over the range where float64 carries the answer."""
    scipy_special = pytest.importorskip("scipy.special")
    x = torch.tensor(_X_VALUES, dtype=torch.float64)
    got = spherical_bessel_table(x, _U_MAX)
    for i, xv in enumerate(_X_VALUES):
        for u in range(0, _U_MAX + 1):
            want = float(scipy_special.spherical_jn(u, xv))
            # Below ~1e-290 the reference itself is at the edge of float64, and
            # the FRF flushes anything under float32's smallest normal anyway.
            if abs(want) < 1e-30:
                continue
            mine = float(got[i, u])
            assert math.isclose(mine, want, rel_tol=1e-9, abs_tol=1e-300), (
                f"j_{u}({xv}) = {mine:.12e}, scipy says {want:.12e}"
            )


def test_batched_shape_and_dtype_round_trip():
    x = torch.rand(7, 3, dtype=torch.float64) * 60.0 + 1.3
    got = spherical_bessel_table(x, 20)
    assert got.shape == (7, 3, 21)
    assert got.dtype == torch.float64
    got32 = spherical_bessel_table(x.to(torch.float32), 20)
    assert got32.dtype == torch.float32
