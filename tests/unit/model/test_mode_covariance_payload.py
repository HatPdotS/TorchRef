"""The mode-covariance node payload, of which TLS is one member.

A node stores the covariance of the displacement modes it represents rather than an ADP,
so the ADP an atom receives depends on where it sits inside the node's region. Two
properties have to hold before any of it is worth measuring: the rigid mode set must
reproduce the textbook TLS expression exactly, and every mode set must stay
positive-semidefinite at every displacement, because an indefinite U makes the
structure-factor FFT return NaN.
"""

import math

import pytest
import torch

from torchref.model.disorder_field import (
    MODE_SETS,
    AnisotropicPayload,
    ModeCovariancePayload,
)
from torchref.model.parameter_wrappers import (
    chol_param_count,
    psd_to_raw,
    raw6_to_u6,
    raw_to_cholesky,
    u6_to_matrix,
)

DTYPE = torch.float64


def _u6_to_mat(u6):
    return u6_to_matrix(u6)


def _random_sigma(q, k=1, scale=0.05, seed=0):
    """A random PD ``(k, q, q)`` covariance."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(k, q, q, generator=g, dtype=DTYPE) * scale
    return A @ A.transpose(-1, -2) + 1e-3 * torch.eye(q, dtype=DTYPE)


# ----------------------------------------------------------------------------------
# Sizes.
# ----------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode_set,q,width",
    [("constant", 3, 6), ("rigid", 6, 21), ("rigid_dilation", 7, 28), ("affine", 12, 78)],
)
def test_mode_set_sizes(mode_set, q, width):
    """The ladder is 6 / 21 / 28 / 78 parameters; TLS is the 21 (20 determinable)."""
    p = ModeCovariancePayload(mode_set)
    assert p.q == q
    assert p.width == width == chol_param_count(q)
    assert p.out_width == 6


@pytest.mark.unit
def test_unknown_mode_set_is_rejected():
    with pytest.raises(ValueError, match="Unknown mode set"):
        ModeCovariancePayload("librational_whimsy")


@pytest.mark.unit
def test_mode_sets_are_nested():
    """Each rung must contain the one below, or the ladder is not a ladder."""
    keys = ["constant", "rigid", "rigid_dilation", "affine"]
    for lo, hi in zip(keys, keys[1:]):
        assert set(MODE_SETS[lo]).issubset(set(MODE_SETS[hi]))
        assert ModeCovariancePayload(lo).q < ModeCovariancePayload(hi).q


# ----------------------------------------------------------------------------------
# The TLS identity. If this fails nothing downstream is trustworthy.
# ----------------------------------------------------------------------------------


@pytest.mark.unit
def test_rigid_mode_set_is_textbook_tls():
    """``Psi Sigma Psi^T`` with translations and rotations IS ``T + AS + S^T A^T + A L A^T``.

    ``A`` is the matrix whose columns are ``e_i x r``, so ``A lambda = lambda x r``. With
    ``Sigma`` blocked as ``[[T, S^T], [S, L]]`` over ``c = (t, lambda)`` the expansion is
    the classical TLS expression, which is the claim that makes this payload a
    generalisation of TLS rather than something merely similar.
    """
    payload = ModeCovariancePayload("rigid")
    sigma = _random_sigma(6, k=1, seed=3)[0]
    T, L, S = sigma[:3, :3], sigma[3:, 3:], sigma[3:, :3]

    torch.manual_seed(11)
    for r in torch.randn(20, 3, dtype=DTYPE) * 5.0:
        # Columns of A are e_i x r.
        A = torch.stack(
            [
                torch.tensor([0.0, -r[2], r[1]], dtype=DTYPE),
                torch.tensor([r[2], 0.0, -r[0]], dtype=DTYPE),
                torch.tensor([-r[1], r[0], 0.0], dtype=DTYPE),
            ],
            dim=1,
        )
        expected = T + A @ S + (A @ S).T + A @ L @ A.T

        Psi = payload.modes(r)  # (3, 6)
        got = Psi @ sigma @ Psi.T
        assert torch.allclose(got, expected, atol=1e-10), f"r={r.tolist()}"


@pytest.mark.unit
def test_rotation_generators_are_antisymmetric():
    """``A`` must be antisymmetric, which is what makes the rigid set a rigid motion."""
    payload = ModeCovariancePayload("rigid")
    r = torch.tensor([1.3, -2.1, 0.7], dtype=DTYPE)
    A = payload.modes(r)[:, 3:]
    assert torch.allclose(A, -A.T, atol=1e-12)


@pytest.mark.unit
def test_trace_s_is_the_flat_direction():
    """TLS has exactly one unobservable combination: adding to ``tr S`` must be free.

    Shifting ``S -> S + c I`` leaves ``U(r)`` unchanged for every ``r``, because
    ``A(cI) + (A cI)^T = c(A + A^T) = 0`` for antisymmetric ``A``.
    """
    payload = ModeCovariancePayload("rigid")
    sigma = _random_sigma(6, k=1, seed=5)[0]
    shifted = sigma.clone()
    shifted[3:, :3] += 0.01 * torch.eye(3, dtype=DTYPE)
    shifted[:3, 3:] += 0.01 * torch.eye(3, dtype=DTYPE)

    torch.manual_seed(2)
    for r in torch.randn(10, 3, dtype=DTYPE) * 4.0:
        Psi = payload.modes(r)
        assert torch.allclose(Psi @ sigma @ Psi.T, Psi @ shifted @ Psi.T, atol=1e-12)


# ----------------------------------------------------------------------------------
# Positive-semidefiniteness, the property the whole construction exists for.
# ----------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("mode_set", sorted(MODE_SETS))
def test_psd_at_extreme_displacement(mode_set):
    """PSD for any ``Sigma`` and any ``r``, including far outside the node's region.

    This is what an arbitrary polynomial in ``r`` cannot promise: it goes indefinite
    somewhere, and somewhere is the edge of the region where the weights have not yet
    decayed.
    """
    payload = ModeCovariancePayload(mode_set)
    torch.manual_seed(7)
    raw = torch.randn(4, payload.width, dtype=DTYPE) * 3.0
    L = raw_to_cholesky(raw, payload.q, payload.epsilon)
    sigma = L @ L.transpose(-1, -2)

    for scale in (0.0, 1e-3, 1.0, 50.0, 1000.0):
        r = torch.randn(6, 3, dtype=DTYPE) * scale
        Psi = payload.modes(r)                       # (6, 3, q)
        U = Psi @ sigma[:, None] @ Psi.transpose(-1, -2)[None]  # (4, 6, 3, 3)
        ev = torch.linalg.eigvalsh(U)
        assert float(ev.min()) >= -1e-9, f"{mode_set} indefinite at |r|~{scale}"


@pytest.mark.unit
@pytest.mark.parametrize("scale", [0.1, 1.0, 3.0, 6.0])
def test_sigma_is_pd_for_any_parameter_value(scale):
    """Cholesky storage: no parameter value can make the covariance indefinite.

    Judged against the matrix norm, not against zero. The diagonal is ``exp(x)``, so a
    wide spread of parameters gives ``Sigma`` a huge dynamic range and ``eigvalsh``
    returns the small eigenvalues with an error set by the large ones -- a float64
    property of the eigensolver, not of the parametrisation.
    """
    payload = ModeCovariancePayload("affine")
    torch.manual_seed(1)
    raw = torch.randn(8, payload.width, dtype=DTYPE) * scale
    ev = torch.linalg.eigvalsh(payload.sigma(raw))
    tol = 1e-10 * ev.abs().max(dim=-1, keepdim=True).values
    assert bool((ev > -tol).all()), f"min eigenvalue {float(ev.min()):.3e} at scale {scale}"


# ----------------------------------------------------------------------------------
# Agreement with the payload it generalises.
# ----------------------------------------------------------------------------------


@pytest.mark.unit
def test_constant_mode_set_matches_anisotropic_payload():
    """``"constant"`` is the same model as :class:`AnisotropicPayload`.

    Both store a 3x3 Cholesky factor and hand every atom the same U, so given the same
    raw parameters they must produce the same tensor. That makes the constant rung the
    inertness guard: a change here that moved it would be a change to the existing
    anisotropic field.
    """
    eps = 1e-3
    mode = ModeCovariancePayload("constant", epsilon=eps)
    aniso = AnisotropicPayload(epsilon=eps)
    torch.manual_seed(4)
    raw = torch.randn(5, 6, dtype=DTYPE)

    xyz = torch.randn(9, 3, dtype=DTYPE) * 10.0
    node_pos = torch.randn(5, 3, dtype=DTYPE) * 10.0
    nl = torch.randint(0, 5, (9, 3))

    got = mode.contributions(raw, xyz, node_pos, nl)
    expected = aniso.contributions(raw, xyz, node_pos, nl)
    assert got.shape == expected.shape == (9, 3, 6)
    assert torch.allclose(got, expected, atol=1e-12)


@pytest.mark.unit
def test_raw_to_cholesky_matches_the_unrolled_three_by_three():
    """The general helper must agree with the unrolled 3x3 pair it generalises."""
    eps = 1e-3
    torch.manual_seed(6)
    raw = torch.randn(12, 6, dtype=DTYPE)
    L = raw_to_cholesky(raw, 3, eps)
    got = L @ L.transpose(-1, -2)
    expected = _u6_to_mat(raw6_to_u6(raw, eps))
    assert torch.allclose(got, expected, atol=1e-12)


@pytest.mark.unit
def test_cholesky_round_trip():
    """``psd_to_raw`` inverts ``raw_to_cholesky`` for a PD matrix."""
    eps = 1e-4
    for q in (3, 6, 12):
        sigma = _random_sigma(q, k=5, scale=0.3, seed=q)
        raw = psd_to_raw(sigma, eps)
        L = raw_to_cholesky(raw, q, eps)
        assert torch.allclose(L @ L.transpose(-1, -2), sigma, atol=1e-8), f"q={q}"


# ----------------------------------------------------------------------------------
# Fit and magnitude.
# ----------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("mode_set", sorted(MODE_SETS))
def test_fit_seeds_translations_and_floors_the_rest(mode_set):
    """Entering the parametrisation must start as the equivalent constant-U field."""
    payload = ModeCovariancePayload(mode_set, epsilon=1e-3)
    torch.manual_seed(8)
    n_atoms, k_nodes = 40, 4
    xyz = torch.randn(n_atoms, 3, dtype=DTYPE) * 8.0
    node_pos = torch.randn(k_nodes, 3, dtype=DTYPE) * 8.0
    nl = torch.randint(0, k_nodes, (n_atoms, 2))
    w_dense = torch.rand(n_atoms, k_nodes, dtype=DTYPE)
    w_dense = w_dense / w_dense.sum(dim=1, keepdim=True)
    target_b = 10.0 + 20.0 * torch.rand(n_atoms, dtype=DTYPE)

    raw = payload.fit(target_b, w_dense, 1e-3, xyz, node_pos, nl)
    assert raw.shape == (k_nodes, payload.width)
    sigma = payload.sigma(raw)

    # Translation block carries the fit; every gradient mode sits at the floor.
    assert float(sigma[:, :3, :3].diagonal(dim1=-2, dim2=-1).min()) > 1e-4
    if payload.q > 3:
        grad = sigma[:, 3:, 3:]
        assert float(grad.abs().max()) < 1e-5, "gradient modes did not start at the floor"


@pytest.mark.unit
def test_log_magnitude_is_b_eq_of_the_translation_block():
    payload = ModeCovariancePayload("affine")
    torch.manual_seed(9)
    raw = torch.randn(6, payload.width, dtype=DTYPE) * 0.5
    T = payload.sigma(raw)[:, :3, :3]
    expected = torch.log(
        ((8.0 * math.pi**2 / 3.0) * T.diagonal(dim1=-2, dim2=-1).sum(-1)).clamp(min=1e-6)
    )
    assert torch.allclose(payload.log_magnitude(raw), expected, atol=1e-12)


# ----------------------------------------------------------------------------------
# Gradients.
# ----------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("mode_set", ["rigid", "affine"])
def test_gradcheck_contributions(mode_set):
    """Gradient w.r.t. the node parameters and the coordinates.

    ``node_pos`` is held constant here rather than differentiated: the conditioning
    length scale is a detached median over node positions, so gradcheck's numerical
    derivative would pick up a path the analytic one deliberately cuts. That cut is the
    subject of :func:`test_length_scale_carries_no_gradient`; the gradient that reaches
    a node's position through the displacement ``r`` is covered here by ``xyz``, which
    enters the same subtraction with the opposite sign.
    """
    payload = ModeCovariancePayload(mode_set)
    torch.manual_seed(10)
    xyz = (torch.randn(7, 3, dtype=DTYPE) * 4.0).requires_grad_(True)
    node_pos = torch.randn(3, 3, dtype=DTYPE) * 4.0
    nl = torch.randint(0, 3, (7, 2))
    raw = (torch.randn(3, payload.width, dtype=DTYPE) * 0.3).requires_grad_(True)

    assert torch.autograd.gradcheck(
        lambda p_, x: payload.contributions(p_, x, node_pos, nl),
        (raw, xyz),
        eps=1e-6,
        atol=1e-7,
    )


@pytest.mark.unit
def test_length_scale_carries_no_gradient():
    """The conditioning length scale is a stop-gradient, on purpose.

    It is the median nearest-neighbour node distance, whose true derivative is supported
    on whichever single node pair happens to be at the median -- an artifact of the
    layout, not a direction any optimiser should follow. Cutting it also stops the
    optimiser rescaling its own modes by spreading the nodes out. Node position still
    gets its real gradient through the displacement ``r``.
    """
    payload = ModeCovariancePayload("affine")
    torch.manual_seed(12)
    xyz = torch.randn(20, 3, dtype=DTYPE) * 5.0
    node_pos = (torch.randn(4, 3, dtype=DTYPE) * 5.0).requires_grad_(True)
    nl = torch.randint(0, 4, (20, 3))
    raw = torch.randn(4, payload.width, dtype=DTYPE) * 0.3

    payload.contributions(raw, xyz, node_pos, nl).sum().backward()
    assert node_pos.grad is not None
    assert float(node_pos.grad.abs().sum()) > 0, "no gradient reaches node positions at all"

    # The scale itself must be a plain float, not something carrying a graph.
    lam = payload._length_scale(node_pos)
    assert isinstance(lam, float)
