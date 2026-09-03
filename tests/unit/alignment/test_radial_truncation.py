"""Pin the SH-Bessel radial band to Phaser's per-``l`` size.

Phaser allocates the ``Elmn`` array with ``nmax = (lmax - l + 2) / 2`` radial
terms for each even ``l`` and runs ``n`` from 1 to ``nmax``
(``DataMR.cc:894-896``). The band therefore narrows as ``l`` rises -- for
``lmax = 76`` it is 38 terms at ``l = 2`` and a single term at ``l = 76`` -- so
the high-``l`` bands cannot carry more radial detail than the reflection set
supports.

``bessel_sh_expand`` allocates a flat ``(N_radial, L, 2L-1)`` array, where
``N_radial`` is Phaser's *widest* band (the one at ``l = 2``). That shape is
easy to misread as "every ``l`` carries ``N_radial`` radial terms"; it does not.
The ``(l, n) -> u = l + 2n + 1`` index build populates only
``n_l = (lmax_even - l)//2 + 1`` terms per ``l``, which is exactly Phaser's
``nmax``, so the truncation is already in force through the allocated support.

These tests pin that invariant against the formula, so the agreement is
asserted rather than inferred from the array shape.

They run under ``double_cpu`` because the band's *width* and its *populated
extent* only coincide in double precision. At the working precision the radial
weight ``sqrt(2u+1) j_u(x)/x`` underflows for high ``u`` at small ``x`` -- see
``test_the_working_precision_truncates_the_band_further`` -- so counting
non-zeros there measures float32's exponent range, not Phaser's formula.
"""

import pytest
import torch

from torchref.experimental.alignment.frf.data_mr import bessel_sh_expand


def _expand(L: int, n_points: int = 900):
    """Expand a fixed pseudo-random point set; returns ``(N_radial, L, 2L-1)``."""
    g = torch.Generator().manual_seed(5)
    s = torch.randn(n_points, 3, generator=g, dtype=torch.float64)
    s = s / s.norm(dim=-1, keepdim=True) * (
        0.05 + 0.15 * torch.rand(n_points, 1, generator=g, dtype=torch.float64)
    )
    intensity = torch.randn(n_points, generator=g, dtype=torch.float64)
    out = bessel_sh_expand(s, intensity, L=L, bessel_h_scale=30.0)
    return out.coeffs.detach().cpu()


def _phaser_nmax(lmax_even: int, l: int) -> int:
    """``(lmax - l + 2) / 2`` -- Phaser's radial band size for one ``l``."""
    return (lmax_even - l + 2) // 2


@pytest.mark.parametrize("L", [21, 41, 67])
def test_radial_band_matches_phaser_width(L, double_cpu):
    """Non-zero radial indices at each ``l`` must stop at Phaser's ``nmax``.

    Our ``n`` index is 0-based against Phaser's 1-based, so the condition is
    ``n < nmax(l)``.
    """
    c = _expand(L)
    N_radial, _, _ = c.shape
    lmax_even = L - 1 if (L - 1) % 2 == 0 else L - 2

    for l in range(2, lmax_even + 1, 2):
        nz = (c[:, l, :].abs() > 0).any(dim=-1).nonzero().flatten()
        if nz.numel() == 0:
            continue                      # legitimately empty band
        allowed = _phaser_nmax(lmax_even, l)
        assert int(nz.max()) < allowed, (
            f"L={L} l={l}: radial index {int(nz.max())} exceeds Phaser's "
            f"nmax={allowed}"
        )


@pytest.mark.parametrize("L", [21, 41, 67])
def test_band_narrows_to_a_single_term_at_lmax(L, double_cpu):
    """The top band keeps exactly one radial term, the widest keeps them all."""
    lmax_even = L - 1 if (L - 1) % 2 == 0 else L - 2
    assert _phaser_nmax(lmax_even, lmax_even) == 1
    c = _expand(L)
    assert _phaser_nmax(lmax_even, 2) == c.shape[0], (
        "N_radial should equal Phaser's widest band, at l=2"
    )
    top = (c[:, lmax_even, :].abs() > 0).any(dim=-1).nonzero().flatten()
    if top.numel():
        assert int(top.max()) == 0, "l=lmax must retain only the n=0 term"


@pytest.mark.parametrize("L", [21, 41, 67])
def test_band_is_exactly_phasers_width_not_merely_bounded(L, double_cpu):
    """The populated band must *equal* Phaser's ``nmax(l)``, not just fit inside.

    A bound alone would also pass for an expansion that silently drops radial
    terms it should keep, which would cost resolution at low ``l``.
    """
    c = _expand(L)
    lmax_even = L - 1 if (L - 1) % 2 == 0 else L - 2
    for l in range(2, lmax_even + 1, 2):
        nz = (c[:, l, :].abs() > 0).any(dim=-1).nonzero().flatten()
        assert nz.numel(), f"L={L} l={l}: band is empty"
        assert int(nz.max()) + 1 == _phaser_nmax(lmax_even, l), (
            f"L={L} l={l}: {int(nz.max()) + 1} radial terms, "
            f"Phaser has {_phaser_nmax(lmax_even, l)}"
        )


def test_allocated_width_exceeds_the_populated_band(double_cpu):
    """The array is wider than the support at every l above the first.

    This is the fact that makes the array shape misleading, and the reason the
    tests above assert the support rather than ``coeffs.shape``.
    """
    L = 67
    lmax_even = L - 1 if (L - 1) % 2 == 0 else L - 2
    c = _expand(L)
    N_radial = c.shape[0]
    assert N_radial == _phaser_nmax(lmax_even, 2)
    assert _phaser_nmax(lmax_even, lmax_even) == 1
    assert N_radial > _phaser_nmax(lmax_even, lmax_even), (
        "allocated width should exceed the top band's single term"
    )


def test_the_working_precision_truncates_the_band_further():
    """At float32 the tail of the radial band is not small, it is *absent*.

    ``j_u(x)`` for ``u >> x`` is genuinely negligible -- at ``x = 1.9`` every
    ``u >= 33`` is below float32's smallest normal -- so at the working precision
    the populated band is narrower than Phaser's ``nmax(l)`` wherever the shell's
    Bessel argument is small. That is a property of the engine as shipped, not a
    defect, and it is pinned here so it is not rediscovered as a regression: the
    structural tests above deliberately run in double precision, which would
    otherwise leave the shipped configuration untested.
    """
    L = 67
    c = _expand(L)                      # working precision, no double_cpu
    lmax_even = L - 1 if (L - 1) % 2 == 0 else L - 2
    populated = (c[:, 2, :].abs() > 0).any(dim=-1).nonzero().flatten()
    assert populated.numel(), "l=2 band is entirely empty"
    got = int(populated.max()) + 1
    phaser = _phaser_nmax(lmax_even, 2)
    assert got < phaser, (
        f"expected the float32 tail to underflow: got {got} terms, Phaser has "
        f"{phaser}. If this now matches, the working precision widened and the "
        f"structural tests above no longer need double_cpu."
    )
