"""Phase-shift sign convention for reciprocal-space symmetry operations.

These tests pin the phase contract of :func:`expand_hkl`, :func:`canonicalize_hkl`
and :func:`reduce_hkl` against an *independent* reference: a direct structure-factor
summation over an explicitly symmetry-expanded atom set. That is the definition of a
structure factor, so it cannot drift in step with the implementation the way a
path-vs-path comparison can.

Why this file exists
--------------------
All three functions computed ``+2π h·t`` where the correct shift is ``-2π h·t``
(and, for the Friedel-flipped rows of ``canonicalize_hkl``, ``+2π h·t`` -- the sign
is *not* uniform there; see that function's comment). The bug survived because the
residual error of the wrong sign is ``4π h·t mod 2π``, which is **exactly zero** for
2₁ screw axes and for centring translations -- and the pre-existing phase test
(``test_canonicalize_hkl.py::test_phase_roundtrip``) parametrised only over
``P21, P212121, C2, P4``, every one of which falls in that blind spot.

The parametrisation below deliberately spans all three regimes:

===================================  ===========================
translation                          error if the sign is flipped
===================================  ===========================
2₁ screws, centring (P21, C2, ...)   0        (invisible)
4₁/4₃ screws (P41, P43, P43212)      π
3₁/6₁ screws (P31, P61, P3121, ...)  2π/3
===================================  ===========================

Do not narrow this list. ``test_wrong_sign_would_be_detected`` guards against it
becoming vacuous.
"""

import numpy as np
import pytest
import torch

from torchref.config import get_float_dtype
from torchref.symmetry.reciprocal_symmetry import (
    canonicalize_hkl,
    expand_hkl,
    reduce_hkl,
)
from torchref.symmetry.spacegroup import SpaceGroup

# Groups spanning the three regimes above. P1 is the degenerate control (no
# translations at all); the screw-axis groups are the ones with real signal.
SPACE_GROUPS = [
    "P 1",
    "P 21",
    "P 21 21 21",
    "C 1 2 1",
    "P 41",
    "P 43",
    "P 31",
    "P 61",
    "P 43 21 2",
    "P 31 2 1",
    "P 65 2 2",
    "I 41",
]

# Groups where a flipped sign produces a non-zero phase error. Kept separate so the
# anti-vacuity test can assert the suite is actually sensitive to the regression.
SENSITIVE_GROUPS = ["P 41", "P 43", "P 31", "P 61", "P 43 21 2", "P 31 2 1", "P 65 2 2"]

# Comfortably tighter than the smallest real error (2π/3 ≈ 2.09 rad) yet loose
# enough for the float32 default dtype the functions compute in (~1e-6 observed).
ATOL_RAD = 1e-4


def _wrap(a):
    """Wrap angles into (-π, π] so 2π-equivalent phases compare equal."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _reference_model(spacegroup, n_atoms=10, seed=1):
    """A symmetry-obeying atom set plus an exact ``F(h)`` evaluator.

    The atom list is the asymmetric unit expanded by every symmetry operation, so
    the resulting structure genuinely has the space-group symmetry and its
    structure factors must obey ``arg F(hR) - arg F(h) = -2π h·t``. Evaluated in
    float64 regardless of the library's configured dtype -- this is the reference.
    """
    rng = np.random.default_rng(seed)
    sg = SpaceGroup(spacegroup, dtype=get_float_dtype(), device=torch.device("cpu"))
    rot = sg.matrices.cpu().numpy().astype(np.float64)
    trans = sg.translations.cpu().numpy().astype(np.float64)

    xyz_asu = rng.random((n_atoms, 3))
    f_asu = rng.uniform(4.0, 10.0, n_atoms)
    xyz = np.concatenate([xyz_asu @ rot[i].T + trans[i] for i in range(len(rot))])
    f = np.concatenate([f_asu] * len(rot))

    def structure_factor(hkl):
        hkl = np.atleast_2d(np.asarray(hkl, dtype=np.float64))
        return (f[None, :] * np.exp(2j * np.pi * (hkl @ xyz.T))).sum(axis=1)

    return structure_factor


def _random_hkl(n=8, seed=3):
    """Random non-zero Miller indices."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        h = tuple(int(x) for x in rng.integers(-6, 7, 3))
        if h != (0, 0, 0):
            out.append(h)
    return torch.tensor(np.array(out), dtype=torch.int32)


@pytest.mark.unit
@pytest.mark.parametrize("spacegroup", SPACE_GROUPS)
def test_expand_hkl_phase_contract(spacegroup):
    """``phi_expanded = phi_orig[indices] + phase_shifts`` must be the true phase.

    This is the contract stated in :func:`expand_hkl`'s own docstring. Checked with
    ``include_friedel=False``: the Friedel half cannot be expressed as an additive
    shift at all (``phi(-h) = -phi(h)`` is a conjugation, not an offset), so the
    documented contract only applies to the pure-rotation expansion.
    """
    fcalc = _reference_model(spacegroup)
    hkl = _random_hkl()
    phi_in = np.angle(fcalc(hkl.numpy()))

    hkl_p1, indices, shifts = expand_hkl(
        hkl, spacegroup, include_friedel=False, remove_absences=True
    )
    got = phi_in[indices.cpu().numpy()] + shifts.cpu().numpy()
    expected = np.angle(fcalc(hkl_p1.numpy()))

    err = np.abs(_wrap(got - expected)).max()
    assert err < ATOL_RAD, (
        f"{spacegroup}: expand_hkl phase contract violated by {err:.4f} rad. "
        f"Expected arg F(hR) - arg F(h) == -2*pi*h.t."
    )


@pytest.mark.unit
@pytest.mark.parametrize("spacegroup", SPACE_GROUPS)
def test_canonicalize_hkl_phase_contract(spacegroup):
    """``phi_canonical = where(friedel, -phi_in, phi_in) + phase_shifts``.

    This is the form :meth:`ReflectionData._canonicalize_in_place` applies. Note the
    sign of the shift differs between the Friedel and non-Friedel halves, so a
    uniform sign is necessarily wrong for one of them.
    """
    fcalc = _reference_model(spacegroup)
    # Feed a full-sphere, deliberately non-canonical set so real work is required.
    hkl_all, _, _ = expand_hkl(
        _random_hkl(), spacegroup, include_friedel=True, remove_absences=True
    )
    phi_in = np.angle(fcalc(hkl_all.numpy()))

    canonical, shifts, friedel, sort_idx = canonicalize_hkl(
        hkl_all, spacegroup, include_friedel=True
    )
    f = friedel.cpu().numpy()
    base = np.where(f, -phi_in[sort_idx.cpu().numpy()], phi_in[sort_idx.cpu().numpy()])
    got = base + shifts.cpu().numpy()
    expected = np.angle(fcalc(canonical.numpy()))

    err = np.abs(_wrap(got - expected)).max()
    assert err < ATOL_RAD, (
        f"{spacegroup}: canonicalize_hkl phase contract violated by {err:.4f} rad "
        f"({f.sum()} of {len(f)} rows were Friedel-flipped)."
    )


@pytest.mark.unit
@pytest.mark.parametrize("spacegroup", SPACE_GROUPS)
def test_reduce_hkl_phase_contract(spacegroup):
    """Every equivalent must reconstruct the ASU phase, not just the first.

    ``reduce_hkl`` returns one row per ASU reflection and a column per equivalent.
    Column 0 is typically the identity operation (zero shift), so checking only that
    column would pass regardless of the sign -- iterate over all of them.
    """
    fcalc = _reference_model(spacegroup)
    hkl_p1, _, _ = expand_hkl(
        _random_hkl(), spacegroup, include_friedel=False, remove_absences=True
    )
    phi_p1 = np.angle(fcalc(hkl_p1.numpy()))

    hkl_asu, red_idx, red_shift = reduce_hkl(
        hkl_p1, spacegroup, include_friedel=False
    )
    phi_asu = np.angle(fcalc(hkl_asu.numpy()))
    idx, sh = red_idx.cpu().numpy(), red_shift.cpu().numpy()

    errs, n_checked = [], 0
    for row in range(idx.shape[0]):
        for col in range(idx.shape[1]):
            src = idx[row, col]
            if src < 0:
                continue
            n_checked += 1
            errs.append(_wrap(phi_p1[src] + sh[row, col] - phi_asu[row]))

    assert n_checked >= idx.shape[0], "no equivalents were checked"
    err = np.abs(errs).max()
    assert err < ATOL_RAD, (
        f"{spacegroup}: reduce_hkl phase contract violated by {err:.4f} rad "
        f"over {n_checked} equivalents."
    )


@pytest.mark.unit
@pytest.mark.parametrize("spacegroup", SENSITIVE_GROUPS)
def test_wrong_sign_would_be_detected(spacegroup):
    """Anti-vacuity guard: the tests above must be *able* to fail.

    Re-runs the ``expand_hkl`` check with the shift negated and asserts it breaks. If
    someone narrows ``SPACE_GROUPS`` back to only 2₁/centring groups, or a refactor
    makes the shifts identically zero, the contract tests would pass trivially and
    this test is what notices.
    """
    fcalc = _reference_model(spacegroup)
    hkl = _random_hkl()
    phi_in = np.angle(fcalc(hkl.numpy()))

    hkl_p1, indices, shifts = expand_hkl(
        hkl, spacegroup, include_friedel=False, remove_absences=True
    )
    expected = np.angle(fcalc(hkl_p1.numpy()))
    flipped = phi_in[indices.cpu().numpy()] - shifts.cpu().numpy()

    err = np.abs(_wrap(flipped - expected)).max()
    assert err > 0.5, (
        f"{spacegroup}: negating the phase shift changed nothing (max err "
        f"{err:.2e} rad), so this space group cannot detect a sign regression "
        f"and does not belong in SENSITIVE_GROUPS."
    )
