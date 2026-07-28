"""GPU variable-radius density path: Triton (wq_grid / wq_grid_aniso) vs the CPU
grouped splat, which share the identical per-atom radius policy.

Both backends truncate each atom at its own ``N_sigma * sigma_eff`` radius, so the
forward maps must agree to float32 + analytic-vs-gathered-coord tolerance and the
gradients (xyz / adp / u / occ) must agree in direction *and* magnitude.

The candidate side runs under ``Engine.TRITON``, which raises rather than
degrading to the portable splat -- that is what keeps this file able to fail.

Cells, atom sets and metrics come from ``tests/helpers/kernel_cases.py``, shared
with the MPS/Metal equivalent. They used to be a private copy here, which is how
two coverage gaps survived in both files at once: unit occupancies (hiding the
``grad / occ`` rescaling) and zero off-diagonal ``u`` (hiding every ellipsoid
cross-term). Both are non-degenerate now.

Markers: ``@pytest.mark.cuda`` -- the sole gate, see ``conftest.py``'s
``pytest_collection_modifyitems``; this file deliberately does no availability
checking of its own. Also ``integration``.
"""

import pytest
import torch

from tests.helpers.grad_asserts import assert_grads_agree
from tests.helpers.kernel_cases import (
    aniso_atoms,
    cell_monoclinic,
    cell_orthorhombic,
    cos_sim,
    iso_atoms,
    rel_map,
    to_device,
)
from torchref.base.electron_density.main import build_electron_density
from torchref.utils import Engine, use_engine

pytestmark = [pytest.mark.cuda, pytest.mark.integration]

# Left at the values this file has always used. Unlike the MPS tolerances, these
# have not been re-measured against the new non-degenerate atom sets -- that
# needs a CUDA host. Tighten only with measurements in hand.
_REL_TOL = 2e-2
_COS_TOL = 0.9995
_GRAD_KW = dict(min_cos=0.999, ratio_tol=1e-2)


def _empty_iso(device):
    """Zero-length isotropic arguments, for an aniso-only structure."""
    return (
        torch.zeros(0, 3, device=device),
        torch.zeros(0, device=device),
        torch.zeros(0, device=device),
        torch.zeros(0, 5, device=device),
        torch.zeros(0, 5, device=device),
    )


def _run_iso(engine, device, cell_fn, atoms):
    _, _, frac, inv_frac, voxel, rsg = cell_fn()
    xyz, adp, occ, A, B = to_device(device, *atoms)
    rsg, frac, inv_frac, voxel = to_device(device, rsg, frac, inv_frac, voxel)
    with use_engine(engine):
        return build_electron_density(
            rsg, xyz, adp, occ, A, B, inv_frac, frac, voxel, dtype=torch.float32
        )


def _run_aniso(engine, device, cell_fn, atoms):
    _, _, frac, inv_frac, voxel, rsg = cell_fn()
    xyz, u, occ, A, B = to_device(device, *atoms)
    rsg, frac, inv_frac, voxel = to_device(device, rsg, frac, inv_frac, voxel)
    xi, ai, oi, Ai, Bi = _empty_iso(device)
    with use_engine(engine):
        return build_electron_density(
            rsg, xi, ai, oi, Ai, Bi, inv_frac, frac, voxel,
            xyz_aniso=xyz, u_aniso=u, occ_aniso=occ, A_aniso=A, B_aniso=B,
            dtype=torch.float32,
        )


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def test_iso_triton_matches_cpu_grouped(cuda_device):
    cell_fn = cell_orthorhombic
    atoms = iso_atoms(cell_fn()[0])
    cpu = _run_iso(Engine.AUTO, "cpu", cell_fn, atoms)
    gpu = _run_iso(Engine.TRITON, cuda_device, cell_fn, atoms)
    assert rel_map(gpu, cpu) < _REL_TOL
    assert cos_sim(gpu, cpu) > _COS_TOL


def test_iso_triton_matches_eager_monoclinic(cuda_device):
    """Guard the per-axis (inv-frac-norm) box + direct coords on a sheared cell:
    TRITON (CUDA) vs the portable EAGER plain-scatter reference (independent box)."""
    cell_fn = cell_monoclinic
    atoms = iso_atoms(cell_fn()[0])
    ref = _run_iso(Engine.EAGER, "cpu", cell_fn, atoms)
    gpu = _run_iso(Engine.TRITON, cuda_device, cell_fn, atoms)
    assert rel_map(gpu, ref) < _REL_TOL
    assert cos_sim(gpu, ref) > _COS_TOL


@pytest.mark.parametrize(
    "cell_fn", [cell_orthorhombic, cell_monoclinic], ids=["orthorhombic", "monoclinic"]
)
def test_aniso_triton_matches_cpu_grouped(cuda_device, cell_fn):
    """Both cells, so the ellipsoid cross-terms are exercised on a sheared cell
    too -- previously only the isotropic path saw a non-orthogonal cell."""
    atoms = aniso_atoms(cell_fn()[0])
    cpu = _run_aniso(Engine.AUTO, "cpu", cell_fn, atoms)
    gpu = _run_aniso(Engine.TRITON, cuda_device, cell_fn, atoms)
    assert rel_map(gpu, cpu) < _REL_TOL
    assert cos_sim(gpu, cpu) > _COS_TOL


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


def _iso_grads(engine, device, cell_fn, atoms, weights):
    xyz, adp, occ, A, B = to_device(device, *atoms)
    xyz, adp, occ = (t.clone().requires_grad_() for t in (xyz, adp, occ))
    _, _, frac, inv_frac, voxel, rsg = cell_fn()
    rsg, frac, inv_frac, voxel = to_device(device, rsg, frac, inv_frac, voxel)
    with use_engine(engine):
        dm = build_electron_density(
            rsg, xyz, adp, occ, A, B, inv_frac, frac, voxel, dtype=torch.float32
        )
    (dm * weights.to(device)).sum().backward()
    return {"xyz": xyz.grad, "adp": adp.grad, "occ": occ.grad}


def _aniso_grads(engine, device, cell_fn, atoms, weights):
    xyz, u, occ, A, B = to_device(device, *atoms)
    xyz, u, occ = (t.clone().requires_grad_() for t in (xyz, u, occ))
    _, _, frac, inv_frac, voxel, rsg = cell_fn()
    rsg, frac, inv_frac, voxel = to_device(device, rsg, frac, inv_frac, voxel)
    xi, ai, oi, Ai, Bi = _empty_iso(device)
    with use_engine(engine):
        dm = build_electron_density(
            rsg, xi, ai, oi, Ai, Bi, inv_frac, frac, voxel,
            xyz_aniso=xyz, u_aniso=u, occ_aniso=occ, A_aniso=A, B_aniso=B,
            dtype=torch.float32,
        )
    (dm * weights.to(device)).sum().backward()
    return {"xyz": xyz.grad, "u": u.grad, "occ": occ.grad}


def test_iso_gradients_match_cpu_grouped(cuda_device):
    cell_fn = cell_orthorhombic
    atoms = iso_atoms(cell_fn()[0])
    w = torch.randn(cell_fn()[1], generator=torch.Generator().manual_seed(3))
    cpu = _iso_grads(Engine.AUTO, "cpu", cell_fn, atoms, w)
    gpu = _iso_grads(Engine.TRITON, cuda_device, cell_fn, atoms, w)
    assert_grads_agree(gpu, cpu, ctx="iso ", **_GRAD_KW)


def test_aniso_gradients_match_cpu_grouped(cuda_device):
    """Covers the two paths the old zero-off-diagonal / unit-occupancy atoms
    could not reach: the off-diagonal U gradients and the ``grad / occ``
    rescaling, which at ``occ == 1`` is a division by one."""
    cell_fn = cell_orthorhombic
    atoms = aniso_atoms(cell_fn()[0])
    w = torch.randn(cell_fn()[1], generator=torch.Generator().manual_seed(4))
    cpu = _aniso_grads(Engine.AUTO, "cpu", cell_fn, atoms, w)
    gpu = _aniso_grads(Engine.TRITON, cuda_device, cell_fn, atoms, w)
    assert_grads_agree(gpu, cpu, ctx="aniso ", **_GRAD_KW)

    # The off-diagonal block must actually be exercised, else this test could
    # pass on an all-zero comparison.
    assert gpu["u"][:, 3:].abs().max() > 0, "off-diagonal U gradients are all zero"
