"""Invariants every E convention has to hold, whatever it does inside.

The conformance harness in `alignment_lab` reports on all of these and more, as
a table, over real data. These are the subset that must never break: a failure
here is a bug rather than a trade-off, so they belong in the gate rather than in
a report someone has to read.

The epsilon check is here because it already caught one. `WilsonShellE` divided
the shell mean by ``eps`` without dividing the intensity by it, which left
``<|E|**2> = <eps>`` -- 1 on a primitive lattice and 2 on a centred one, so a
normaliser whose absolute scale depended on the space group. The rotation
function never saw it (it passes no ``eps``), which is exactly why it survived:
a defect only reachable through an argument nobody was passing yet.
"""

import functools

import pytest
import torch

from torchref.experimental.alignment.e_values import (
    CalcGlobalE, CalcShellE, FrenchWilsonE, SmoothSigmaE, WilsonShellE,
    WilsonShellEpsE,
)

pytestmark = pytest.mark.unit

CONVENTIONS = [
    WilsonShellE, WilsonShellEpsE, CalcShellE, CalcGlobalE, FrenchWilsonE,
    functools.partial(SmoothSigmaE, n_coeff=6),
]


def _name(c):
    return getattr(getattr(c, "func", c), "__name__", str(c))


def _wilson_data(n=20000, seed=1, centric_frac=0.1):
    """Amplitudes drawn from the distribution the conventions assume."""
    g = torch.Generator().manual_seed(seed)
    s = torch.rand(n, generator=g, dtype=torch.float64) * 0.4 + 0.05
    # |E|^2 ~ Exp(1) scaled by a resolution-dependent Sigma, so there is a real
    # trend for a per-shell or smooth normaliser to have to remove.
    sigma = torch.exp(-40.0 * s * s) * 2500.0 + 1.0
    E2 = -torch.log(torch.rand(n, generator=g, dtype=torch.float64).clamp(min=1e-12))
    F = (E2 * sigma).sqrt()
    sig_F = F * 0.05 + 0.5
    centric = torch.zeros(n, dtype=torch.bool)
    centric[: int(n * centric_frac)] = True
    return F, s, centric, sig_F


def _build(cls, F, s, centric, sig_F, eps=None, n_shells=20):
    kw = {"sig_F": sig_F} if getattr(cls, "uses_sigma_f", False) else {}
    return cls(F, s, centric, eps=eps, n_shells=n_shells, **kw)


@pytest.mark.parametrize("cls", CONVENTIONS, ids=_name)
def test_epsilon_reaches_both_sides_of_the_ratio_or_neither(cls):
    """A convention's scale must not depend on the lattice centring.

    ``eps`` is a constant 2 here, which is what a centred lattice gives every
    reflection. Dividing the shell mean by it and not the intensity would show
    up as ``<|E|**2>`` doubling -- the bug this pins.
    """
    F, s, centric, sig_F = _wilson_data()
    eps = torch.full_like(F, 2.0)
    without = _build(cls, F, s, centric, sig_F).E
    with_eps = _build(cls, F, s, centric, sig_F, eps=eps).E
    m_without = float((without * without).mean())
    m_with = float((with_eps * with_eps).mean())
    assert m_with == pytest.approx(m_without, rel=1e-6), (
        f"{_name(cls)}: <|E|^2> moves from {m_without:.4f} to {m_with:.4f} when "
        f"a uniform eps=2 is supplied. A uniform multiplicity cancels out of "
        f"E**2 = (F**2/eps) / <F**2/eps>; a change means eps reached only one "
        f"side of that ratio."
    )


@pytest.mark.parametrize("cls", CONVENTIONS, ids=_name)
def test_invariant_to_a_global_rescale_of_F(cls):
    """E is a ratio, so multiplying every amplitude must change nothing.

    This is the property that lets the rotation function ignore scale entirely,
    and the one a convention with any absolute constant in it would break.
    """
    F, s, centric, sig_F = _wilson_data()
    base = _build(cls, F, s, centric, sig_F).E
    for c in (1e-3, 1e3):
        scaled = _build(cls, F * c, s, centric, sig_F * c).E
        assert torch.allclose(scaled, base, rtol=1e-6, atol=1e-9), (
            f"{_name(cls)}: scaling F by {c:g} moved E by up to "
            f"{float((scaled - base).abs().max()):.3e}"
        )


@pytest.mark.parametrize("cls", CONVENTIONS, ids=_name)
def test_the_normaliser_removes_the_resolution_trend(cls):
    """<|E|**2> must not drift with resolution -- that trend IS the weighting."""
    F, s, centric, sig_F = _wilson_data()
    E = _build(cls, F, s, centric, sig_F).E
    order = torch.argsort(s)
    means = [float((E[order[k::10]] ** 2).mean()) for k in range(10)]
    lo, hi = min(means), max(means)
    assert hi / max(lo, 1e-12) < 1.35, (
        f"{_name(cls)}: <|E|^2> ranges {lo:.3f}..{hi:.3f} across resolution "
        f"deciles; the normaliser is leaving a trend behind"
    )


def test_a_sigma_f_convention_names_a_calc_companion():
    """There is no measurement error on a calc set, so it needs a stand-in."""
    assert FrenchWilsonE.uses_sigma_f
    companion = FrenchWilsonE.for_calc()
    assert companion is not FrenchWilsonE
    assert not getattr(companion, "uses_sigma_f", False)


def test_french_wilson_refuses_to_run_without_sigmas():
    """Silently degrading to plain Wilson would hide the whole difference."""
    F, s, centric, _ = _wilson_data(n=2000)
    with pytest.raises(ValueError, match="sig_F"):
        FrenchWilsonE(F, s, centric, n_shells=20)


def test_the_frf_calc_path_is_bit_identical_to_wilson_normalise():
    """The seam must be inert while the default convention is in place.

    Everything downstream of ``E_calc`` is untouched, so bit-identity here is
    what makes the peak list unchanged rather than merely similar.
    """
    from torchref.experimental.alignment.frf.preprocessing import (
        wilson_normalise,
    )

    F, s, centric, _ = _wilson_data()
    old, _ = wilson_normalise(F, s, 20)
    new = WilsonShellE(F, s, centric, n_shells=20).E
    assert torch.equal(new, old), (
        f"max deviation {float((new - old).abs().max()):.3e}"
    )


def test_a_partial_is_a_usable_convention():
    """Configuration rides in as ``functools.partial``, so lookups must survive it.

    A partial forwards ``__call__`` but not class attributes, so asking one for
    ``for_calc`` or ``uses_sigma_f`` directly raises ``AttributeError``. The FRF
    asks for both on every run -- which is why all three ``SmoothSigmaE`` arms
    of the first convention panel failed on all 50 cells without producing a
    single number.
    """
    from torchref.experimental.alignment.e_values import (
        convention_class, convention_for_calc, convention_uses_sigma_f,
    )

    plain = functools.partial(SmoothSigmaE, n_coeff=6)
    assert convention_class(plain) is SmoothSigmaE
    assert convention_uses_sigma_f(plain) is False
    # Its own companion, so the configuration has to survive the round trip.
    assert convention_for_calc(plain) is plain

    fw = functools.partial(FrenchWilsonE)
    assert convention_uses_sigma_f(fw) is True
    # A different class is named, so its keywords are not this one's.
    assert convention_for_calc(fw) is WilsonShellE

    F, s, centric, sig_F = _wilson_data(n=4000)
    assert convention_for_calc(plain)(F, s, centric, n_shells=20).E.shape == F.shape
