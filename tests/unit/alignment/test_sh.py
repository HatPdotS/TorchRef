"""Leaf mathematics for the rotation function: shell binning and the Legendre reference.

Two independent things, both pinned because something downstream trusts them.

**Shell binning.** Two consumers deriving their own equal-count edges from the
same ``|s|`` is how boundary reflections end up in different shells depending on
which stage asked. The edges are computed once and the index passed down; these
tests pin the round trip that makes that safe.

**``_bar_legendre_recurrence``.** Production never calls it -- the Bessel-SH
expansion runs the same recurrence inside its kernels. It exists as the
*independent* implementation that
``tests/unit/frf_separate/test_bessel_sh_grouping.py`` builds a slow reference
expansion on, to check the fused one against something other than itself. That
only works if the reference is itself trustworthy, which is what the scipy
comparison here is for: it used to reach this recurrence through ``evaluate_ylm``,
and that wrapper is gone.
"""
import math

import numpy as np
import pytest
import torch

from torchref.experimental.alignment.sh import (
    _bar_legendre_recurrence,
    assign_shells,
    equal_count_shell_edges,
)


# ---------------------------------------------------------------------------
# The Legendre reference the FRF expansion is checked against
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("L", [4, 9])
def test_bar_legendre_matches_scipy(L):
    """bar_P_l^m(x) = sqrt[(2l+1)/(4pi) (l-m)!/(l+m)!] |P_l^m(x)|, against scipy.

    scipy's ``lpmv`` carries the Condon-Shortley phase and this recurrence does
    not, so the comparison is on magnitude -- which is the convention the
    docstring states and the kernels implement.
    """
    scipy_special = pytest.importorskip("scipy.special")

    theta = torch.tensor([0.3, 1.1, 2.0, 2.9], dtype=torch.float64)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    bar_P = _bar_legendre_recurrence(cos_t, sin_t, L)          # (M, L, L)

    x = cos_t.numpy()
    for l in range(L):
        for m in range(l + 1):
            norm = math.sqrt(
                (2 * l + 1) / (4 * math.pi)
                * math.factorial(l - m) / math.factorial(l + m)
            )
            expected = norm * np.abs(scipy_special.lpmv(m, l, x))
            np.testing.assert_allclose(
                bar_P[:, l, m].abs().numpy(), expected, atol=1e-11, rtol=1e-11,
                err_msg=f"l={l}, m={m}",
            )


@pytest.mark.unit
def test_bar_legendre_pole_values():
    """At the north pole (cos theta = 1): zero for m > 0, sqrt((2l+1)/4pi) at m = 0.

    The pole is where the recurrence is most fragile -- sin(theta) = 0 kills the
    sectoral seed, and everything above it comes from the vertical step.
    """
    L = 5
    theta = torch.tensor([0.0], dtype=torch.float64)
    bar_P = _bar_legendre_recurrence(torch.cos(theta), torch.sin(theta), L)
    for l in range(L):
        for m in range(1, l + 1):
            assert abs(bar_P[0, l, m].item()) < 1e-14, f"l={l}, m={m}"
        np.testing.assert_allclose(
            bar_P[0, l, 0].item(), math.sqrt((2 * l + 1) / (4 * math.pi)),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Shell binning
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_shell_assignment_round_trip():
    """The bins really are equal-count, and every reflection lands in one.

    ``equal_count_shell_edges`` returns ``(edges, centers)`` -- the second value
    is the shell mid-points, not the occupancies -- so equal-count is asserted
    on the assignment rather than read off the return.
    """
    torch.manual_seed(0)
    s = torch.rand(1000, dtype=torch.float64) * 0.4 + 0.05
    n_shells = 12
    edges, centers = equal_count_shell_edges(s, n_shells)
    idx = assign_shells(s, edges)

    assert edges.shape == (n_shells + 1,)
    assert centers.shape == (n_shells,)
    assert int(idx.min()) >= 0 and int(idx.max()) < n_shells
    counts = torch.bincount(idx, minlength=n_shells)
    assert int(counts.sum()) == s.numel(), "a reflection fell outside every shell"
    # 1000 into 12 cannot divide evenly; equal-count means within one of ideal.
    assert int(counts.max()) - int(counts.min()) <= 1, counts
    # Centres must sit inside their own shell.
    assert bool(((centers > edges[:-1]) & (centers < edges[1:])).all())


@pytest.mark.unit
def test_shells_are_monotone_in_resolution():
    """A shell is a resolution range, so the bins must not interleave."""
    torch.manual_seed(1)
    s = torch.rand(500, dtype=torch.float64) * 0.3 + 0.02
    edges, _ = equal_count_shell_edges(s, 8)
    idx = assign_shells(s, edges)
    hi = torch.stack([s[idx == b].max() for b in range(8) if (idx == b).any()])
    assert bool((hi[1:] >= hi[:-1]).all())


@pytest.mark.unit
def test_assignment_is_stable_under_a_subset():
    """Slicing rows must not move a reflection's shell; rebuilding edges would.

    This is the property the "assign once, pass it down" rule rests on: given
    the SAME edges, a subset of the reflections lands in the same bins it did in
    the full set.
    """
    torch.manual_seed(2)
    s = torch.rand(600, dtype=torch.float64) * 0.3 + 0.02
    edges, _ = equal_count_shell_edges(s, 10)
    full = assign_shells(s, edges)
    sub = torch.arange(0, 600, 3)
    torch.testing.assert_close(assign_shells(s[sub], edges), full[sub])
