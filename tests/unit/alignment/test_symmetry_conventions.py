"""Guard the reciprocal-space symmetry convention used by the FRF.

Reciprocal space transforms as ``h' = h·R`` (equivalently ``Rᵀ·h``), the rule
:meth:`SpaceGroup.apply_to_hkl` implements. The alignment package re-derives
symmetry expansions inline in several places, and each of those inline copies
once used ``R·h`` instead.

The reason that survived so long is worth stating, because it is what makes
these tests necessary rather than merely nice: **the two conventions agree
whenever the symmetry matrices are orthogonal**, which they are in every
monoclinic, orthorhombic, tetragonal and cubic setting. Only in a trigonal or
hexagonal basis, where ``S·Sᵀ ≠ I``, do they diverge -- and there the wrong
convention silently builds an orbit that mixes non-equivalent reflections,
writing conflicting ``|F|`` onto one Miller index.

So every test here is parametrised over a space group whose matrices are *not*
orthogonal. A test that only covers P2₁2₁2₁ or P4₃2₁2 cannot fail.
"""

from pathlib import Path

import pytest
import torch

from torchref.experimental.alignment.frf.preprocessing import (
    compute_epsilon,
    epsilon_aware_unroll,
)
from torchref.experimental.alignment.sh import (
    hkl_symops_to_cartesian,
    symmetrize_anisotropy,
)
from torchref.symmetry import SpaceGroup


#: Space groups spanning both regimes. ``non_orthogonal`` flags the settings
#: whose rotation matrices are not orthogonal in their own basis -- the only
#: ones that can discriminate ``h·R`` from ``R·h``.
SPACEGROUPS = [
    pytest.param("P 31 2 1", True, id="P3121-trigonal"),
    pytest.param("P 65 2 2", True, id="P6522-hexagonal"),
    pytest.param("P 63", True, id="P63-hexagonal"),
    pytest.param("P 4 3 2", False, id="P432-cubic"),
    pytest.param("P 21 21 2", False, id="P21212-orthorhombic"),
    pytest.param("C 1 2 1", False, id="C2-monoclinic"),
]


def _cell_for(hm: str):
    """A cell consistent with the space group's lattice constraints."""
    if hm.startswith("P 3") or hm.startswith("P 6"):
        return (60.0, 60.0, 95.0, 90.0, 90.0, 120.0)
    if hm.startswith("P 4 3") or hm.startswith("P 21 21"):
        return (70.0, 70.0, 70.0, 90.0, 90.0, 90.0) if "4 3" in hm else (
            45.0, 55.0, 65.0, 90.0, 90.0, 90.0)
    return (80.0, 40.0, 60.0, 90.0, 104.0, 90.0)


def _reciprocal_basis(cell):
    """``B`` from TorchRef's own :class:`Cell`, so the convention matches by
    construction rather than by a hand-rolled duplicate."""
    from torchref.symmetry import Cell

    return Cell(list(cell)).reciprocal_basis_matrix.detach().cpu().to(torch.float64)


@pytest.mark.parametrize("hm, non_orthogonal", SPACEGROUPS)
def test_symop_orthogonality_flags_the_hard_cases(hm, non_orthogonal):
    """The premise of every other test here: which settings can discriminate.

    If this ever reports a trigonal/hexagonal group as orthogonal, the tests
    below stop testing anything.
    """
    S = SpaceGroup(hm).matrices.detach().cpu().to(torch.float64)
    eye = torch.eye(3, dtype=torch.float64)
    err = max(float((S[k] @ S[k].T - eye).abs().max()) for k in range(S.shape[0]))
    assert (err > 0.5) == non_orthogonal, (
        f"{hm}: max|S·Sᵀ-I| = {err:.3f}, expected "
        f"{'non-orthogonal' if non_orthogonal else 'orthogonal'} matrices"
    )


@pytest.mark.parametrize("hm, non_orthogonal", SPACEGROUPS)
def test_cartesian_symops_are_rotations(hm, non_orthogonal):
    """``hkl_symops_to_cartesian`` must return genuine rotations.

    A Cartesian symmetry operator is orthogonal with determinant +1 by
    definition. Conjugating with the untransposed ``S`` does not produce one in
    a non-orthogonal basis -- it returned matrices with orthogonality error 5.33
    for P 3₁ 2 1 and P 6₅ 2 2, which is what let a wrong anisotropy
    symmetrisation through undetected.
    """
    del non_orthogonal
    S = SpaceGroup(hm).matrices.detach().cpu().to(torch.float64)
    B = _reciprocal_basis(_cell_for(hm))
    R = hkl_symops_to_cartesian(S, B).detach().cpu().to(torch.float64)

    eye = torch.eye(3, dtype=torch.float64)
    for k in range(R.shape[0]):
        # 1e-6 is float64 noise through the matrix inverse; the defect this
        # guards against produced an error of 5.33.
        assert torch.allclose(R[k] @ R[k].T, eye, atol=1e-6), (
            f"{hm} op {k} is not orthogonal:\n{R[k]}"
        )
        assert float(torch.det(R[k])) == pytest.approx(1.0, abs=1e-6), (
            f"{hm} op {k} has determinant {float(torch.det(R[k])):.6f}, not +1"
        )


@pytest.mark.parametrize("hm, non_orthogonal", SPACEGROUPS)
def test_unroll_matches_apply_to_hkl(hm, non_orthogonal):
    """Any inline symmetry expansion must agree with the shared helper.

    ``apply_to_hkl`` is the single definition of the convention; the einsum
    contraction ``"kji,nj->kni"`` is the same operation with the operator axis
    first. ``"kij,nj->kni"`` is the bug and differs here for the non-orthogonal
    settings.
    """
    del non_orthogonal
    sg = SpaceGroup(hm)
    S = sg.matrices.detach().cpu().to(torch.float64)
    g = torch.Generator().manual_seed(7)
    hkl = torch.randint(-12, 13, (200, 3), generator=g).to(torch.float64)

    reference = sg.apply_to_hkl(hkl).detach().cpu().to(torch.float64)          # (N, 3, ops)
    unrolled = torch.einsum("kji,nj->kni", S, hkl)              # (ops, N, 3)
    assert torch.allclose(unrolled.permute(1, 2, 0), reference, atol=1e-9), (
        f"{hm}: inline unroll disagrees with SpaceGroup.apply_to_hkl"
    )


@pytest.mark.parametrize("hm, non_orthogonal", SPACEGROUPS)
def test_wrong_convention_changes_the_orbit(hm, non_orthogonal):
    """``R·h`` yields a different *orbit* from ``h·R`` iff ``S`` is non-orthogonal.

    Per operator the two always differ unless ``S`` is symmetric, so the honest
    statement is about the orbit as a set. For an orthogonal group ``Sᵀ = S⁻¹``
    is itself a group member, so the set is unchanged and the bug is invisible;
    in a hexagonal basis ``Sᵀ`` leaves the group and the orbit genuinely moves.

    This is what gives the other tests their teeth: it asserts the defect is
    observable at all for these settings, so a regression cannot hide behind a
    benchmark built only from orthogonal lattices.
    """
    S = SpaceGroup(hm).matrices.detach().cpu().to(torch.float64)
    g = torch.Generator().manual_seed(11)
    hkl = torch.randint(-12, 13, (200, 3), generator=g).to(torch.float64)

    def orbit_keys(contraction):
        o = torch.einsum(contraction, S, hkl).round().to(torch.long)   # (ops, N, 3)
        k = ((o[..., 0] + 64) * 256 + (o[..., 1] + 64)) * 256 + (o[..., 2] + 64)
        return torch.sort(k, dim=0).values                             # per-reflection set

    same_orbits = torch.equal(orbit_keys("kji,nj->kni"), orbit_keys("kij,nj->kni"))
    assert (not same_orbits) == non_orthogonal, (
        f"{hm}: expected the orbits to "
        f"{'differ' if non_orthogonal else 'coincide'} between conventions"
    )


@pytest.mark.parametrize("hm", ["P 31 2 1", "P 65 2 2"])
def test_symmetrised_anisotropy_obeys_the_lattice(hm):
    """With a 3- or 6-fold along c, Cartesian ``U`` must be ``diag(a, a, c)``.

    The previous conjugation returned a ``U`` *more* anisotropic than its input
    (diag 0.90/1.30/0.60 became 1.58/3.70/0.60 with off-diagonals of 1.83),
    which no averaging over a point group can do.
    """
    S = SpaceGroup(hm).matrices.detach().cpu().to(torch.float64)
    B = _reciprocal_basis(_cell_for(hm))
    R = hkl_symops_to_cartesian(S, B).detach().cpu().to(torch.float64)

    U = torch.tensor([[0.90, 0.10, 0.05],
                      [0.10, 1.30, -0.07],
                      [0.05, -0.07, 0.60]], dtype=torch.float64)
    U_sym = symmetrize_anisotropy(U, R).detach().cpu().to(torch.float64)

    off = U_sym - torch.diag(U_sym.diagonal())
    assert float(off.abs().max()) < 1e-6, f"{hm}: U not diagonal:\n{U_sym}"
    assert float(U_sym[0, 0]) == pytest.approx(float(U_sym[1, 1]), abs=1e-6), (
        f"{hm}: U11 != U22 ({U_sym[0,0]:.6f} vs {U_sym[1,1]:.6f})"
    )
    # c is unconstrained by the axis, so it must survive untouched.
    assert float(U_sym[2, 2]) == pytest.approx(0.60, abs=1e-6)
    # An average cannot exceed the input's range.
    assert float(U_sym[0, 0]) == pytest.approx(1.10, abs=1e-6)


@pytest.mark.parametrize("hm, non_orthogonal", SPACEGROUPS)
def test_epsilon_uses_the_row_vector_convention(hm, non_orthogonal):
    """``compute_epsilon`` counts ops fixing ``h``, which needs ``h·R``.

    Reflections on a symmetry axis must come out with multiplicity > 1; with
    the wrong convention the wrong reflections are flagged.
    """
    del non_orthogonal
    sg = SpaceGroup(hm)
    S = sg.matrices.detach().cpu().to(torch.float64)
    g = torch.Generator().manual_seed(3)
    hkl = torch.randint(-9, 10, (400, 3), generator=g)

    eps = compute_epsilon(hkl, S).detach().cpu()
    assert int(eps.min()) >= 1
    # Recompute independently through the shared helper.
    ref = sg.apply_to_hkl(hkl.to(torch.float64)).detach().cpu()                # (N, 3, ops)
    same = (ref.round() == hkl.to(torch.float64).unsqueeze(-1)).all(dim=1)
    assert torch.equal(eps.to(torch.long), same.sum(dim=-1).clamp(min=1)), (
        f"{hm}: compute_epsilon disagrees with apply_to_hkl"
    )


@pytest.mark.parametrize("hm, non_orthogonal", SPACEGROUPS)
def test_epsilon_aware_unroll_stays_within_the_true_orbit(hm, non_orthogonal):
    """Every emitted position must be a genuine symmetry mate of its input.

    This exercises a real call site rather than the contraction in isolation.
    Under the wrong convention the emitted positions leave the true orbit for a
    non-orthogonal setting, which is what let two inequivalent reflections land
    on one Miller index carrying different ``|F|``.
    """
    del non_orthogonal
    sg = SpaceGroup(hm)
    S = sg.matrices.detach().cpu().to(torch.float64)
    g = torch.Generator().manual_seed(19)
    hkl = torch.randint(-9, 10, (150, 3), generator=g)

    unrolled, asu_idx = epsilon_aware_unroll(hkl, S)
    unrolled = unrolled.detach().cpu().to(torch.long)
    asu_idx = asu_idx.detach().cpu().to(torch.long)

    # true orbit of each parent, via the shared helper
    orbit = sg.apply_to_hkl(hkl.to(torch.float64)).detach().cpu().round().to(torch.long)
    for i in range(0, unrolled.shape[0], 17):          # stride: full sweep is redundant
        parent = int(asu_idx[i])
        mates = orbit[parent].T                        # (ops, 3)
        assert (mates == unrolled[i]).all(dim=-1).any(), (
            f"{hm}: emitted {unrolled[i].tolist()} is not in the orbit of "
            f"{hkl[parent].tolist()}"
        )


def test_no_column_convention_survives_in_the_alignment_package():
    """No ``S·h`` symmetry contraction may reappear in the alignment package.

    The four defects this module guards were four *copies* of one rule. The
    other tests pin the rule; this one pins the absence of new copies, which is
    the failure mode that actually occurred -- the shared helper was corrected
    while the inline duplicates were not.
    """
    import re

    pkg = Path(__file__).resolve().parents[3] / "torchref" / "experimental" / "alignment"
    # `kij` contracted against an hkl-like index is the column convention.
    pattern = re.compile(r'einsum\(\s*["\']k?ij,\s*nj->')
    offenders = []
    for path in sorted(pkg.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(pkg.parent.parent.parent)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "reciprocal-space symmetry expansion must use h·R (\"kji,nj->\"), not "
        "R·h (\"kij,nj->\"):\n  " + "\n  ".join(offenders)
    )
