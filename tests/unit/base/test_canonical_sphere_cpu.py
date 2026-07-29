"""One truncation contract, checked on CPU with no accelerator required.

Every production density kernel truncates each atom at the same spherical cutoff:

    voxel v gets atom i's density iff ||w||^2 <= r_i^2, with w the minimum-image
    Cartesian atom->voxel vector (sphere centred on the ATOM, not on its anchor
    node) and r_i the raw radius_policy radius,

enumerated over the per-axis box ceil(r * n_axis * ||inv_frac row_axis||). This file
pins that contract down where it can actually run in CI on a laptop.

Why this file exists
--------------------
Until now every variable-radius cross-check was gated on hardware --
``test_variable_radius_gpu.py`` needs CUDA, ``test_variable_radius_mps.py`` needs
Apple silicon -- so on a CPU-only machine *nothing* tested the splat geometry. That
is how three different cutoffs came to coexist: the fast CPU path splatted a cube,
the portable CPU path used a node-centred diagonal metric at a voxel-rounded
radius, and Metal inflated its radius to a whole voxel. On a beta=115 deg cell those
differed by ~5e-3 rel L2 -- more than the 1.7e-3 truncation error the cutoff exists
to deliver -- and the MPS test absorbed it in a 2e-2 tolerance.

The reference here is a direct O(atoms x voxels) evaluation of the contract as
written above (``_brute_iso`` / ``_brute_aniso``), not another kernel: comparing two
kernels to each other cannot catch a shared misreading of the geometry.

Non-orthogonal cells are the point. An orthogonal cell hides a diagonal-metric
error entirely, so every geometry assertion runs at beta = 90, 100 and 115 degrees.
"""

import math

import pytest
import torch

from torchref.base.electron_density.kernels.cpu import sphere_splat
from torchref.base.electron_density.kernels.cpu.variable_radius import (
    add_anisotropic_plain_var,
    add_isotropic_plain_var,
)
from torchref.base.electron_density.main import build_electron_density
from torchref.base.electron_density.radius_policy import (
    per_atom_radius_aniso,
    per_atom_radius_iso,
)
from torchref.base.scattering.scattering_table import get_scattering_params_by_z
from torchref.utils import Engine, use_engine

pytestmark = pytest.mark.unit

# float32 agreement floor. The fused kernel uses a fast 2^x exp (the CPU analogue of
# the metal::fast::exp the Metal kernels already use), measured at 2e-5 rel L2
# against std::exp; the portable splat uses torch.exp. Both are far below the 7.9e-4
# amplitude-truncation error at the default 3 sigma, so this tolerance bounds
# arithmetic noise while still failing on any geometry disagreement (the smallest of
# which, node- vs atom-centring on an orthogonal cell, is 9.9e-4).
_F32_TOL = 2e-4
_BETAS = (90.0, 100.0, 115.0)


def _cell(beta_deg, dtype=torch.float32, dims=(48, 40, 34), abc=(28.0, 24.0, 20.0)):
    """Monoclinic-family P1 cell; beta=90 degenerates to orthorhombic."""
    a, b, c = abc
    beta = math.radians(beta_deg)
    f64 = torch.tensor(
        [[a, 0.0, c * math.cos(beta)],
         [0.0, b, 0.0],
         [0.0, 0.0, c * math.sin(beta)]], dtype=torch.float64)
    return f64.to(dtype), torch.linalg.inv(f64).to(dtype), dims, f64


def _voxel_size(f64, dims):
    """``voxel_size`` as ``sf_fft`` derives it; unused by the kernels, still in the
    ``build_electron_density`` signature."""
    return (f64.norm(dim=0) / torch.tensor(dims, dtype=torch.float64)).float()


def _iso_atoms(f64, n=36, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.tensor([6, 7, 8, 16]).repeat(n // 4 + 1)[:n]
    A, B = get_scattering_params_by_z(z, dtype=dtype)
    xyz = (torch.rand(n, 3, generator=g, dtype=torch.float64) @ f64.T).to(dtype)
    adp = (torch.rand(n, generator=g) * 35 + 8).to(dtype)
    # never exactly 1.0: the kernels recover d/d_occ by dividing the accumulated
    # gradient by occ, and at occ == 1 a wrong scaling is invisible
    occ = (torch.rand(n, generator=g) * 0.4 + 0.6).to(dtype)
    return xyz, adp, occ, A, B


def _aniso_atoms(f64, n=24, dtype=torch.float32, seed=1):
    g = torch.Generator().manual_seed(seed)
    z = torch.tensor([6, 7, 8, 16]).repeat(n // 4 + 1)[:n]
    A, B = get_scattering_params_by_z(z, dtype=dtype)
    xyz = (torch.rand(n, 3, generator=g, dtype=torch.float64) @ f64.T).to(dtype)
    u = torch.zeros(n, 6)
    u[:, :3] = torch.rand(n, 3, generator=g) * 0.12 + 0.02
    # signed, non-zero off-diagonals: the p01/p02/p12 entries of the analytically
    # inverted 3x3 and the 4*pi^2 off-diagonal U gradients are dead code without them
    u[:, 3:] = (torch.rand(n, 3, generator=g) - 0.5) * 0.02
    occ = (torch.rand(n, generator=g) * 0.4 + 0.6).to(dtype)
    return xyz, u.to(dtype), occ, A, B


def _frac_grid(dims, dtype):
    """Voxel fractional coordinates, computed *in* ``dtype``.

    ``torch.arange(n) / n`` divides an integer tensor by a Python int, which lands in
    the default float32 and would cap a float64 reference's accuracy at ~1e-7.
    """
    ii, jj, kk = torch.meshgrid(
        *[torch.arange(n, dtype=dtype) for n in dims], indexing="ij")
    return torch.stack([ii / dims[0], jj / dims[1], kk / dims[2]], -1).reshape(-1, 3)


def _brute_iso(dims, xyz, adp, occ, A, B, inv_frac, frac, r):
    """Direct evaluation of the canonical contract, atom by atom."""
    dtype = xyz.dtype
    fc = _frac_grid(dims, dtype)
    out = torch.zeros(dims[0] * dims[1] * dims[2], dtype=dtype)
    xyz_frac = xyz @ inv_frac.T
    B_total = ((B + adp[:, None]) * 0.25).clamp(min=0.1)
    A_norm = A * occ[:, None] * (math.pi ** 1.5) / (B_total * torch.sqrt(B_total))
    for i in range(xyz.shape[0]):
        d = fc - (xyz_frac[i] % 1.0)
        d = d - torch.round(d)                       # minimum image
        w = d @ frac.T
        r2 = (w * w).sum(-1)
        m = r2 <= r[i] * r[i]
        e = torch.exp(-(math.pi ** 2) * r2[m][:, None] / B_total[i][None, :])
        out[m] = out[m] + (e * A_norm[i][None, :]).sum(-1)
    return out.view(*dims)


def _brute_aniso(dims, xyz, u, occ, A, B, inv_frac, frac, r):
    """As ``_brute_iso``, with the Mahalanobis density and a Euclidean cutoff."""
    dtype = xyz.dtype
    fc = _frac_grid(dims, dtype)
    out = torch.zeros(dims[0] * dims[1] * dims[2], dtype=dtype)
    xyz_frac = xyz @ inv_frac.T
    U3 = torch.zeros(u.shape[0], 3, 3, dtype=dtype)
    U3[:, 0, 0], U3[:, 1, 1], U3[:, 2, 2] = u[:, 0], u[:, 1], u[:, 2]
    U3[:, 0, 1] = U3[:, 1, 0] = u[:, 3]
    U3[:, 0, 2] = U3[:, 2, 0] = u[:, 4]
    U3[:, 1, 2] = U3[:, 2, 1] = u[:, 5]
    M = (B[:, :, None, None] * torch.eye(3, dtype=dtype)
         + 8 * math.pi ** 2 * U3[:, None]) / 4.0
    Minv = torch.linalg.inv(M)
    A_norm = (A * occ[:, None] * (math.pi ** 1.5)
              / torch.sqrt(torch.linalg.det(M).clamp(min=1e-10)))
    for i in range(xyz.shape[0]):
        d = fc - (xyz_frac[i] % 1.0)
        d = d - torch.round(d)
        w = d @ frac.T
        m = (w * w).sum(-1) <= r[i] * r[i]           # cutoff is the Euclidean sphere
        ww = w[m]
        q = torch.einsum("vi,gij,vj->vg", ww, Minv[i], ww)
        out[m] = out[m] + (A_norm[i][None, :] * torch.exp(-(math.pi ** 2) * q)).sum(-1)
    return out.view(*dims)


def _rel_l2(x, y):
    return float((x - y).double().norm() / y.double().norm())


def _empty_iso(dtype):
    """Zero-length isotropic arguments, for an aniso-only structure."""
    return (torch.zeros(0, 3, dtype=dtype), torch.zeros(0, dtype=dtype),
            torch.zeros(0, dtype=dtype), torch.zeros(0, 5, dtype=dtype),
            torch.zeros(0, 5, dtype=dtype))


# ===========================================================================
# The contract: each kernel vs a direct evaluation of the spec
# ===========================================================================

@pytest.mark.parametrize("beta", _BETAS)
def test_fused_iso_matches_contract(beta):
    if not sphere_splat.sphere_splat_available():
        pytest.skip(f"fused CPU splat unavailable: {sphere_splat.last_error()}")
    frac, inv_frac, dims, f64 = _cell(beta)
    xyz, adp, occ, A, B = _iso_atoms(f64)
    r = per_atom_radius_iso(adp, B, n_sigma=3.0)
    got = sphere_splat.add_isotropic_cpu_sphere_var(
        torch.zeros(dims), xyz, adp, occ, A, B, inv_frac, frac, r)
    assert _rel_l2(got, _brute_iso(dims, xyz, adp, occ, A, B, inv_frac, frac, r)) < _F32_TOL


@pytest.mark.parametrize("beta", _BETAS)
def test_fused_aniso_matches_contract(beta):
    if not sphere_splat.sphere_splat_available():
        pytest.skip(f"fused CPU splat unavailable: {sphere_splat.last_error()}")
    frac, inv_frac, dims, f64 = _cell(beta)
    xyz, u, occ, A, B = _aniso_atoms(f64)
    r = per_atom_radius_aniso(B, u, n_sigma=3.0)
    got = sphere_splat.add_anisotropic_cpu_sphere_var(
        torch.zeros(dims), xyz, u, occ, A, B, inv_frac, frac, r)
    assert _rel_l2(got, _brute_aniso(dims, xyz, u, occ, A, B, inv_frac, frac, r)) < _F32_TOL


@pytest.mark.parametrize("beta", _BETAS)
def test_portable_iso_matches_contract(beta):
    """The portable splat used a node-centred diagonal metric at a voxel-rounded
    radius; on beta=115 that alone was 5e-3 rel L2."""
    frac, inv_frac, dims, f64 = _cell(beta)
    xyz, adp, occ, A, B = _iso_atoms(f64)
    r = per_atom_radius_iso(adp, B, n_sigma=3.0)
    got = add_isotropic_plain_var(
        torch.zeros(dims), xyz, adp, occ, A, B, inv_frac, frac, r)
    assert _rel_l2(got, _brute_iso(dims, xyz, adp, occ, A, B, inv_frac, frac, r)) < _F32_TOL


@pytest.mark.parametrize("beta", _BETAS)
def test_portable_aniso_matches_contract(beta):
    """The portable aniso splat used a full cube -- ~2.3x the sphere's voxels."""
    frac, inv_frac, dims, f64 = _cell(beta)
    xyz, u, occ, A, B = _aniso_atoms(f64)
    r = per_atom_radius_aniso(B, u, n_sigma=3.0)
    got = add_anisotropic_plain_var(
        torch.zeros(dims), xyz, u, occ, A, B, inv_frac, frac, r)
    assert _rel_l2(got, _brute_aniso(dims, xyz, u, occ, A, B, inv_frac, frac, r)) < _F32_TOL


def test_fused_float64_is_exact():
    """float64 uses std::exp, so only fp rounding separates it from the reference."""
    if not sphere_splat.sphere_splat_available():
        pytest.skip("fused CPU splat unavailable")
    frac, inv_frac, dims, f64 = _cell(100.0, dtype=torch.float64, dims=(32, 28, 24))
    xyz, adp, occ, A, B = _iso_atoms(f64, dtype=torch.float64)
    r = per_atom_radius_iso(adp, B, n_sigma=3.0)
    got = sphere_splat.add_isotropic_cpu_sphere_var(
        torch.zeros(dims, dtype=torch.float64), xyz, adp, occ, A, B, inv_frac, frac, r)
    want = _brute_iso(dims, xyz, adp, occ, A, B, inv_frac, frac, r)
    assert _rel_l2(got, want) < 1e-13


# ===========================================================================
# AUTO vs EAGER through the real dispatch: no accelerator needed
# ===========================================================================

def _build(engine, dims, frac, inv_frac, voxel, dtype, iso=None, aniso=None):
    rsg = torch.zeros(*dims, 3, dtype=dtype)  # shape only; no kernel reads its values
    xi, ai, oi, Ai, Bi = iso if iso is not None else _empty_iso(dtype)
    kw = {}
    if aniso is not None:
        xa, ua, oa, Aa, Ba = aniso
        kw = dict(xyz_aniso=xa, u_aniso=ua, occ_aniso=oa, A_aniso=Aa, B_aniso=Ba)
    with use_engine(engine):
        return build_electron_density(
            rsg, xi, ai, oi, Ai, Bi, inv_frac, frac, voxel, dtype=dtype, **kw)


@pytest.mark.parametrize("beta", _BETAS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64],
                         ids=["float32", "float64"])
def test_auto_matches_eager_iso(beta, dtype):
    frac, inv_frac, dims, f64 = _cell(beta, dtype=dtype)
    voxel = _voxel_size(f64, dims)
    atoms = _iso_atoms(f64, dtype=dtype)
    ref = _build(Engine.EAGER, dims, frac, inv_frac, voxel, dtype, iso=atoms)
    got = _build(Engine.AUTO, dims, frac, inv_frac, voxel, dtype, iso=atoms)
    tol = _F32_TOL if dtype is torch.float32 else 1e-12
    assert _rel_l2(got, ref) < tol


@pytest.mark.parametrize("beta", _BETAS)
def test_auto_matches_eager_aniso(beta):
    frac, inv_frac, dims, f64 = _cell(beta)
    voxel = _voxel_size(f64, dims)
    atoms = _aniso_atoms(f64)
    ref = _build(Engine.EAGER, dims, frac, inv_frac, voxel, torch.float32, aniso=atoms)
    got = _build(Engine.AUTO, dims, frac, inv_frac, voxel, torch.float32, aniso=atoms)
    assert _rel_l2(got, ref) < _F32_TOL


@pytest.mark.parametrize("kind", ["iso", "aniso"])
def test_auto_matches_eager_gradients(kind):
    """Direction *and* magnitude: a kernel returning ``2 * grad`` is perfectly
    parallel, so cosine alone cannot catch it."""
    frac, inv_frac, dims, f64 = _cell(100.0, dtype=torch.float64, dims=(32, 28, 24))
    voxel = _voxel_size(f64, dims)
    w = torch.randn(dims, generator=torch.Generator().manual_seed(7),
                    dtype=torch.float64)
    if kind == "iso":
        xyz, p, occ, A, B = _iso_atoms(f64, dtype=torch.float64)
    else:
        xyz, p, occ, A, B = _aniso_atoms(f64, dtype=torch.float64)

    def grads(engine):
        x, pp, o = (t.clone().requires_grad_() for t in (xyz, p, occ))
        pack = (x, pp, o, A, B)
        dm = _build(engine, dims, frac, inv_frac, voxel, torch.float64,
                    **({"iso": pack} if kind == "iso" else {"aniso": pack}))
        (dm * w).sum().backward()
        return x.grad, pp.grad, o.grad

    for g_auto, g_eager in zip(grads(Engine.AUTO), grads(Engine.EAGER)):
        assert torch.allclose(g_auto, g_eager, rtol=1e-9, atol=0)


def test_cutoff_is_grid_independent():
    """The cutoff must not be requantized to the grid.

    Metal used to round the radius up to a whole voxel, which made the *same*
    ``sigma_cutoff_ed`` truncate at different radii on different grid samplings. With
    the radius used raw, refining the grid must converge the map rather than move the
    cutoff: the total mass a coarse and a fine grid assign is set by the same sphere.
    """
    frac, inv_frac, _, f64 = _cell(100.0)
    xyz, adp, occ, A, B = _iso_atoms(f64, n=24)
    r = per_atom_radius_iso(adp, B, n_sigma=3.0)
    masses = []
    for dims in ((40, 34, 28), (60, 51, 42)):
        dm = add_isotropic_plain_var(
            torch.zeros(dims), xyz, adp, occ, A, B, inv_frac, frac, r)
        cell_vol = float(torch.linalg.det(f64))
        masses.append(float(dm.double().sum()) * cell_vol / (dims[0] * dims[1] * dims[2]))
    # Same truncation sphere, so the integrated mass agrees to grid-sampling error.
    assert abs(masses[1] - masses[0]) / masses[0] < 5e-3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64],
                         ids=["float32", "float64"])
def test_auto_actually_dispatches_the_fused_kernel(dtype):
    """Guard against the AUTO-vs-EAGER tests going vacuous.

    If ``should_use_sphere_splat`` ever stopped firing, AUTO would fall through to
    the very same portable splat EAGER uses and every equivalence assertion above
    would pass at ``rel_l2 == 0`` while measuring nothing -- the exact failure mode
    ``test_variable_radius_mps.py`` documents for its own Metal gate. float64 is
    parametrized because the old dispatch was float32-only, so a regression to a
    dtype gate would be invisible on float32 alone.

    ``main`` is patched, not the kernel module: the dispatch site resolves the name
    from its own module globals at import time.
    """
    import torchref.base.electron_density.main as main_mod

    if not sphere_splat.sphere_splat_available():
        pytest.skip("fused CPU splat unavailable")
    frac, inv_frac, dims, f64 = _cell(100.0, dtype=dtype)
    calls = []
    real = main_mod.add_isotropic_cpu_sphere_var

    def recording(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    main_mod.add_isotropic_cpu_sphere_var = recording
    try:
        _build(Engine.AUTO, dims, frac, inv_frac, _voxel_size(f64, dims), dtype,
               iso=_iso_atoms(f64, dtype=dtype))
    finally:
        main_mod.add_isotropic_cpu_sphere_var = real
    assert calls, f"Engine.AUTO did not reach the fused CPU splat for {dtype}"


def test_empty_atom_sets():
    """A structure with no isotropic (or no anisotropic) atoms must not crash."""
    frac, inv_frac, dims, f64 = _cell(90.0)
    voxel = _voxel_size(f64, dims)
    only_aniso = _build(Engine.AUTO, dims, frac, inv_frac, voxel, torch.float32,
                        aniso=_aniso_atoms(f64))
    assert torch.isfinite(only_aniso).all() and float(only_aniso.abs().sum()) > 0
    both_empty = _build(Engine.AUTO, dims, frac, inv_frac, voxel, torch.float32)
    assert float(both_empty.abs().sum()) == 0.0


def test_density_map_accumulates_not_overwrites():
    """Both passes add into one map, so the aniso pass must not clobber the iso one."""
    frac, inv_frac, dims, f64 = _cell(90.0)
    voxel = _voxel_size(f64, dims)
    iso, aniso = _iso_atoms(f64), _aniso_atoms(f64)
    a = _build(Engine.AUTO, dims, frac, inv_frac, voxel, torch.float32, iso=iso)
    b = _build(Engine.AUTO, dims, frac, inv_frac, voxel, torch.float32, aniso=aniso)
    both = _build(Engine.AUTO, dims, frac, inv_frac, voxel, torch.float32,
                  iso=iso, aniso=aniso)
    assert _rel_l2(both, a + b) < 1e-6
