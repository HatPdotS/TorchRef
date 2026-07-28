"""MPS variable-radius density path: native Metal kernels vs the portable
plain-scatter reference, on the same MPS device.

Both truncate each atom at its own per-atom radius, so the forward maps must
agree to float32 + truncation-shape tolerance and the gradients (xyz / adp / u /
occ) must agree in direction *and* magnitude.

The candidate side runs under ``Engine.METAL``, never ``Engine.AUTO``. That is
load-bearing, not stylistic: under AUTO a Metal kernel that fails to dispatch
falls back to the very same ``add_isotropic_plain_var`` the reference uses, so
every assertion here would pass with ``rel_map == 0.0`` and ``cos == 1.0`` while
measuring nothing at all. ``Engine.METAL`` raises instead of degrading, which is
what makes this file able to fail.

``dtype=torch.float32`` is passed explicitly rather than inherited from the
ambient ``TORCHREF_DTYPE_FLOAT``: under a float64 config the Metal gate never
fires, which is a second, independent way for this file to go vacuous.

Markers: ``@pytest.mark.mps`` (the sole gate -- see ``conftest.py``'s
``pytest_collection_modifyitems``; this file deliberately does no availability
checking of its own), ``integration``.
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

pytestmark = [pytest.mark.mps, pytest.mark.integration]

# Measured Metal-vs-portable agreement on an M-series GPU with the
# non-degenerate atom sets (non-unit occupancy, signed off-diagonal U):
#
#   iso   orthorhombic  rel_map 5.9e-3   cos 0.9999838
#   iso   monoclinic    rel_map 5.2e-3   cos 0.9999727
#   aniso orthorhombic  rel_map 7.7e-4   cos 0.9999990
#   aniso monoclinic    rel_map 1.0e-3   cos 0.9999990
#
# The iso path is an order of magnitude looser than the aniso one because its
# per-atom cutoff is quantized to a whole number of voxels, so a voxel shell can
# land exactly on the ``r2 <= r2cut`` boundary and be included by one
# implementation and not the other. Those are low-density tail voxels, which is
# why the cosine stays tight while the max-relative figure does not. Tolerances
# are ~3-5x the measured maxima; do not tighten without re-measuring.
_ISO_REL_TOL = 2e-2
_ANISO_REL_TOL = 5e-3
_COS_TOL = 0.9999

# Gradient thresholds: the values ``tests/unit/test_gradient_correctness.py``
# uses for "float32 + distinct kernel arithmetic". The ratio check is what a bare
# cosine cannot do -- a kernel returning ``2 * grad`` is perfectly parallel.
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


def _run_iso(engine, mps_device, cell_fn, atoms):
    _, _, frac, inv_frac, voxel, rsg = cell_fn()
    xyz, adp, occ, A, B = to_device(mps_device, *atoms)
    rsg, frac, inv_frac, voxel = to_device(mps_device, rsg, frac, inv_frac, voxel)
    with use_engine(engine):
        return build_electron_density(
            rsg, xyz, adp, occ, A, B, inv_frac, frac, voxel, dtype=torch.float32
        )


def _run_aniso(engine, mps_device, cell_fn, atoms):
    _, _, frac, inv_frac, voxel, rsg = cell_fn()
    xyz, u, occ, A, B = to_device(mps_device, *atoms)
    rsg, frac, inv_frac, voxel = to_device(mps_device, rsg, frac, inv_frac, voxel)
    xi, ai, oi, Ai, Bi = _empty_iso(mps_device)
    with use_engine(engine):
        return build_electron_density(
            rsg, xi, ai, oi, Ai, Bi, inv_frac, frac, voxel,
            xyz_aniso=xyz, u_aniso=u, occ_aniso=occ, A_aniso=A, B_aniso=B,
            dtype=torch.float32,
        )


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cell_fn", [cell_orthorhombic, cell_monoclinic], ids=["orthorhombic", "monoclinic"]
)
def test_iso_metal_matches_plain(mps_device, cell_fn):
    atoms = iso_atoms(cell_fn()[0])
    ref = _run_iso(Engine.EAGER, mps_device, cell_fn, atoms)
    got = _run_iso(Engine.METAL, mps_device, cell_fn, atoms)
    assert rel_map(got, ref) < _ISO_REL_TOL
    assert cos_sim(got, ref) > _COS_TOL


@pytest.mark.parametrize(
    "cell_fn", [cell_orthorhombic, cell_monoclinic], ids=["orthorhombic", "monoclinic"]
)
def test_aniso_metal_matches_plain(mps_device, cell_fn):
    """Both cells, so the ellipsoid cross-terms are exercised on a sheared cell
    too -- previously only the isotropic path saw a non-orthogonal cell."""
    atoms = aniso_atoms(cell_fn()[0])
    ref = _run_aniso(Engine.EAGER, mps_device, cell_fn, atoms)
    got = _run_aniso(Engine.METAL, mps_device, cell_fn, atoms)
    assert rel_map(got, ref) < _ANISO_REL_TOL
    assert cos_sim(got, ref) > _COS_TOL


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


def _iso_grads(engine, mps_device, cell_fn, atoms, weights):
    xyz, adp, occ, A, B = to_device(mps_device, *atoms)
    xyz, adp, occ = (t.clone().requires_grad_() for t in (xyz, adp, occ))
    _, _, frac, inv_frac, voxel, rsg = cell_fn()
    rsg, frac, inv_frac, voxel = to_device(mps_device, rsg, frac, inv_frac, voxel)
    with use_engine(engine):
        dm = build_electron_density(
            rsg, xyz, adp, occ, A, B, inv_frac, frac, voxel, dtype=torch.float32
        )
    (dm * weights).sum().backward()
    return {"xyz": xyz.grad, "adp": adp.grad, "occ": occ.grad}


def _aniso_grads(engine, mps_device, cell_fn, atoms, weights):
    xyz, u, occ, A, B = to_device(mps_device, *atoms)
    xyz, u, occ = (t.clone().requires_grad_() for t in (xyz, u, occ))
    _, _, frac, inv_frac, voxel, rsg = cell_fn()
    rsg, frac, inv_frac, voxel = to_device(mps_device, rsg, frac, inv_frac, voxel)
    xi, ai, oi, Ai, Bi = _empty_iso(mps_device)
    with use_engine(engine):
        dm = build_electron_density(
            rsg, xi, ai, oi, Ai, Bi, inv_frac, frac, voxel,
            xyz_aniso=xyz, u_aniso=u, occ_aniso=occ, A_aniso=A, B_aniso=B,
            dtype=torch.float32,
        )
    (dm * weights).sum().backward()
    return {"xyz": xyz.grad, "u": u.grad, "occ": occ.grad}


def test_iso_gradients_match_plain(mps_device):
    cell_fn = cell_orthorhombic
    atoms = iso_atoms(cell_fn()[0])
    w = torch.randn(cell_fn()[1], generator=torch.Generator().manual_seed(3)).to(
        mps_device
    )
    ref = _iso_grads(Engine.EAGER, mps_device, cell_fn, atoms, w)
    got = _iso_grads(Engine.METAL, mps_device, cell_fn, atoms, w)
    assert_grads_agree(got, ref, ctx="iso ", **_GRAD_KW)


def test_aniso_gradients_match_plain(mps_device):
    """Covers the two paths the old zero-off-diagonal / unit-occupancy atoms
    could not reach: the off-diagonal U gradients (which carry a ``4*pi^2``
    factor where the diagonal ones carry ``2*pi^2``) and the ``grad / occ``
    rescaling, which at ``occ == 1`` is a division by one."""
    cell_fn = cell_orthorhombic
    atoms = aniso_atoms(cell_fn()[0])
    w = torch.randn(cell_fn()[1], generator=torch.Generator().manual_seed(4)).to(
        mps_device
    )
    ref = _aniso_grads(Engine.EAGER, mps_device, cell_fn, atoms, w)
    got = _aniso_grads(Engine.METAL, mps_device, cell_fn, atoms, w)
    assert_grads_agree(got, ref, ctx="aniso ", **_GRAD_KW)

    # The off-diagonal block must actually be exercised, else this test could
    # pass on an all-zero comparison.
    assert got["u"][:, 3:].abs().max() > 0, "off-diagonal U gradients are all zero"


# ---------------------------------------------------------------------------
# Dispatch provenance and strict-mode behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", [Engine.AUTO, Engine.METAL])
def test_engine_dispatches_to_metal_kernel(mps_device, monkeypatch, engine):
    """On MPS float32, both AUTO and METAL must actually call the Metal kernel.

    Verified with a call recorder rather than by comparing maps. Comparing is not
    sound here: the portable reference uses ``scatter_add`` on MPS, whose
    accumulation order is not reproducible, so two portable runs differ at ~1e-7
    and ``torch.equal`` against one proves nothing either way.

    The patch target is the **package** attribute, not a ``main`` global:
    ``_add_isotropic`` does a function-local ``from ...kernels.mps import
    add_isotropic_mps_var`` on every call, so it re-reads the package each time.
    Patching ``main.add_isotropic_mps_var`` would silently no-op and leave this
    test as vacuous as the bug it guards.
    """
    from torchref.base.electron_density import kernels

    real = kernels.mps.add_isotropic_mps_var
    calls = []

    def recording(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(kernels.mps, "add_isotropic_mps_var", recording)

    cell_fn = cell_orthorhombic
    _run_iso(engine, mps_device, cell_fn, iso_atoms(cell_fn()[0]))
    assert calls, f"{engine} did not dispatch to the Metal kernel"


def test_metal_raises_when_shader_unavailable(mps_device, monkeypatch):
    """``Engine.METAL`` must raise, never degrade, when the shader is missing.

    ``monkeypatch`` reverts the module globals at teardown, so this needs no
    ``try``/``finally`` -- and must not have one, since a swallowed failure here
    is exactly the behaviour under test.
    """
    from torchref.base.electron_density.kernels.mps import compile as mps_compile

    monkeypatch.setattr(mps_compile, "_lib", None)
    monkeypatch.setattr(mps_compile, "_lib_failed", True)
    monkeypatch.setattr(mps_compile, "_lib_error", ("forced test failure", ""))

    cell_fn = cell_orthorhombic
    atoms = iso_atoms(cell_fn()[0])
    with pytest.raises(RuntimeError, match="forced test failure"):
        _run_iso(Engine.METAL, mps_device, cell_fn, atoms)

    # ...while AUTO still degrades gracefully to the portable splat. Compared
    # loosely, not bitwise: see the note in the provenance test above.
    auto = _run_iso(Engine.AUTO, mps_device, cell_fn, atoms)
    eager = _run_iso(Engine.EAGER, mps_device, cell_fn, atoms)
    assert rel_map(auto, eager) < 1e-5
