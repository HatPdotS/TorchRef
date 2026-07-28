"""Shared cells, atom sets and map metrics for the accelerator splat tests.

``test_variable_radius_gpu.py`` (Triton) and ``test_variable_radius_mps.py``
(Metal) compare an accelerator density splat against the portable reference.
They had byte-identical private copies of everything below, which is how the two
degeneracies fixed here survived in both at once:

* every atom had ``occ == 1``, so the kernels' ``grad_occ = grad_sum / occ``
  scaling was only ever exercised as a division by one;
* anisotropic ``u`` had **zero off-diagonals**, i.e. every ellipsoid was
  axis-aligned. That left the cross-term arithmetic completely uncovered --
  the ``p01``/``p02``/``p12`` entries of the inverted 3x3, and the backward's
  off-diagonal U gradients, which carry a ``4*pi^2`` factor where the diagonal
  ones carry ``2*pi^2``.

Both are now non-degenerate, following the shapes already used by
``_sf_leaves`` / ``_rand_u6`` in ``tests/unit/test_gradient_correctness.py``.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "cell_orthorhombic",
    "cell_monoclinic",
    "iso_atoms",
    "aniso_atoms",
    "cos_sim",
    "rel_map",
    "to_device",
]


def cell_orthorhombic():
    """Orthorhombic P1 cell + grid. Returns ``(abc, grid, frac, inv_frac, voxel, rsg)``."""
    a, b, c = 30.0, 25.0, 20.0
    nx, ny, nz = 60, 50, 40
    frac = torch.diag(torch.tensor([a, b, c]))
    inv_frac = torch.diag(torch.tensor([1 / a, 1 / b, 1 / c]))
    voxel = torch.tensor([a / nx, b / ny, c / nz])
    ii, jj, kk = torch.meshgrid(
        torch.arange(nx), torch.arange(ny), torch.arange(nz), indexing="ij"
    )
    rsg = (torch.stack([ii / nx, jj / ny, kk / nz], -1).to(torch.float32)) @ frac.T
    return (a, b, c), (nx, ny, nz), frac.float(), inv_frac.float(), voxel.float(), rsg


def cell_monoclinic():
    """Non-orthogonal (beta ~ 100 deg) cell: exercises the per-axis bounding box
    and the off-diagonal coordinate math (the c voxel step gains an x-component)."""
    a, b, c = 30.0, 25.0, 20.0
    nx, ny, nz = 60, 50, 40
    beta = math.radians(100.0)
    frac = torch.tensor(
        [[a, 0.0, c * math.cos(beta)],
         [0.0, b, 0.0],
         [0.0, 0.0, c * math.sin(beta)]], dtype=torch.float64)
    inv_frac = torch.linalg.inv(frac)
    voxel = (frac.norm(dim=0) / torch.tensor([nx, ny, nz], dtype=torch.float64)).float()
    ii, jj, kk = torch.meshgrid(
        torch.arange(nx), torch.arange(ny), torch.arange(nz), indexing="ij"
    )
    fc = torch.stack([ii / nx, jj / ny, kk / nz], -1).double()
    rsg = (fc @ frac.T).float()
    return (a, b, c), (nx, ny, nz), frac.float(), inv_frac.float(), voxel, rsg


def iso_atoms(cell, n=60, seed=0):
    """Isotropic atoms. Returns ``(xyz, adp, occ, A, B)``, all float32.

    Occupancies are in ``[0.6, 1.0)``, never exactly 1: the kernels divide the
    accumulated gradient by ``occ`` to recover ``d/d occ``, and at ``occ == 1``
    that division is a no-op that hides a wrong scaling.
    """
    g = torch.Generator().manual_seed(seed)
    a, b, c = cell
    xyz = torch.rand(n, 3, generator=g) * torch.tensor([a, b, c])
    adp = torch.rand(n, generator=g) * 35 + 3
    occ = torch.rand(n, generator=g) * 0.4 + 0.6
    A = torch.rand(n, 5, generator=g) * 5
    B = torch.rand(n, 5, generator=g) * 20 + 2
    return [t.float() for t in (xyz, adp, occ, A, B)]


def aniso_atoms(cell, n=30, seed=1):
    """Anisotropic atoms. Returns ``(xyz, u, occ, A, B)``, all float32.

    ``u`` is ``[U11, U22, U33, U12, U13, U23]`` with a positive diagonal and
    **signed, non-zero off-diagonals** (magnitudes chosen to keep every U
    comfortably positive-definite, as the shader inverts ``M_g`` analytically
    without a positive-definiteness guard). Occupancies are non-unit, as above.
    """
    g = torch.Generator().manual_seed(seed)
    a, b, c = cell
    xyz = torch.rand(n, 3, generator=g) * torch.tensor([a, b, c])
    u = torch.zeros(n, 6)
    u[:, :3] = torch.rand(n, 3, generator=g) * 0.12 + 0.02
    u[:, 3:] = (torch.rand(n, 3, generator=g) - 0.5) * 0.02
    occ = torch.rand(n, generator=g) * 0.4 + 0.6
    A = torch.rand(n, 5, generator=g) * 5
    B = torch.rand(n, 5, generator=g) * 20 + 2
    return [t.float() for t in (xyz, u, occ, A, B)]


def cos_sim(a, b):
    """Cosine similarity of two maps, flattened, in float64.

    Moves to CPU first: the accumulation wants float64, and MPS has no float64,
    so a caller who forgot ``.cpu()`` would otherwise get a confusing dtype
    error from the backend rather than a comparison.
    """
    a, b = a.detach().cpu().reshape(-1).double(), b.detach().cpu().reshape(-1).double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def rel_map(x, y):
    """Max absolute map difference, relative to the reference's peak."""
    x, y = x.detach().cpu(), y.detach().cpu()
    return float((x - y).abs().max() / (y.abs().max() + 1e-8))


def to_device(dev, *ts):
    """Move every tensor in ``ts`` to ``dev``."""
    return [t.to(dev) for t in ts]
