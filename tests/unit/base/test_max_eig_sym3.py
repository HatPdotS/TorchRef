"""Closed-form largest eigenvalue of a symmetric 3x3 vs torch.linalg.eigvalsh.

``_max_eig_sym3`` replaces ``eigvalsh`` in the anisotropic splat-radius policy so
it runs natively on MPS; it must agree with the reference eigendecomposition,
including on diagonal and near-degenerate matrices.
"""

import pytest
import torch

from torchref.base.electron_density.radius_policy import (
    _max_eig_sym3,
    _u6_to_u3,
    per_atom_radius_aniso,
)

pytestmark = pytest.mark.unit


def _sym(n, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, 3, 3, generator=g) * scale
    return (M + M.transpose(1, 2)) / 2


def test_matches_eigvalsh_random():
    S = _sym(4000, 0)
    ref = torch.linalg.eigvalsh(S).max(dim=1).values
    got = _max_eig_sym3(S)
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)


def test_matches_eigvalsh_diagonal():
    # purely diagonal (off-diagonals zero) -> largest diagonal entry
    d = torch.tensor([[3.0, -1.0, 2.0], [5.0, 5.0, 5.0], [0.1, 0.2, 0.05]])
    S = torch.diag_embed(d)
    ref = torch.linalg.eigvalsh(S).max(dim=1).values
    got = _max_eig_sym3(S)
    assert torch.allclose(got, ref, atol=1e-5)
    assert torch.allclose(got, d.max(dim=1).values, atol=1e-5)


def test_matches_eigvalsh_near_degenerate():
    # near-isotropic (all eigenvalues nearly equal) stresses the p2->0 branch
    S = torch.eye(3).expand(50, 3, 3).clone()
    S += _sym(50, 7, scale=1e-4)
    ref = torch.linalg.eigvalsh(S).max(dim=1).values
    got = _max_eig_sym3(S)
    assert torch.allclose(got, ref, atol=1e-4)


def test_radius_matches_eigvalsh_path():
    # The (quantized) aniso radius must be identical to the eigvalsh-based one.
    g = torch.Generator().manual_seed(3)
    n = 500
    u = torch.zeros(n, 6)
    u[:, :3] = torch.rand(n, 3, generator=g) * 0.12 + 0.02
    u[:, 3:] = (torch.rand(n, 3, generator=g) - 0.5) * 0.02
    B = torch.rand(n, 5, generator=g) * 20 + 2

    rad_closed = per_atom_radius_aniso(B, u, n_sigma=3.0)

    # Reference radius using eigvalsh directly.
    from torchref.base.electron_density.radius_policy import (
        EIGHT_PI2, _ceil_round, R_LO, R_HI,
    )
    b_form = B.max(dim=1).values
    lam = torch.linalg.eigvalsh(_u6_to_u3(u)).max(dim=1).values
    sigma = torch.sqrt((b_form / EIGHT_PI2 + lam).clamp(min=1e-6))
    rad_ref = _ceil_round(3.0 * sigma).clamp(min=R_LO, max=R_HI)

    assert torch.equal(rad_closed, rad_ref)
