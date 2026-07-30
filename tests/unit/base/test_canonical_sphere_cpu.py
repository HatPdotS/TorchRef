"""One truncation contract, checked on CPU with no accelerator required.

Every production density kernel truncates each atom at the same spherical cutoff:

    voxel v gets atom i's density iff ||w||^2 <= r_i^2, with w the minimum-image
    Cartesian atom->voxel vector (sphere centred on the ATOM, not on its anchor
    node) and r_i the raw radius_policy radius,

enumerated over the per-axis box ceil(r * n_axis * ||inv_frac row_axis||). This file
pins that contract down where it can actually run in CI on a laptop.

Why this file exists
--------------------
Every variable-radius cross-check used to be gated on hardware, so on a CPU-only
machine *nothing* tested the splat geometry. That is how three different cutoffs came to
coexist: the fast CPU path splatted a cube, the portable CPU path used a node-centred
diagonal metric at a voxel-rounded radius, and Metal inflated its radius to a whole
voxel. On a beta=115 deg cell those differed by ~5e-3 rel L2 -- more than the 1.7e-3
truncation error the cutoff exists to deliver -- and the accelerator tests absorbed it in
a 2e-2 kernel-vs-kernel tolerance.

Accuracy against an independent reference now lives in ``tests/unit/structure_factor/``,
which checks every production kernel on every device against direct summation. This file
remains the *geometry* contract: a from-spec brute force, cheap, CPU-only, and
independent of both kernels.

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
def test_auto_actually_dispatches_the_fused_kernel(dtype, monkeypatch):
    """Guard against the AUTO-vs-EAGER tests going vacuous.

    If the ``cpu_sphere`` row ever stopped matching, AUTO would fall through to the very
    same portable splat EAGER uses and every equivalence assertion above would pass at
    ``rel_l2 == 0`` while measuring nothing. float64 is parametrized because the dispatch
    was once float32-only, so a regression to a dtype gate would be invisible on float32
    alone.

    The kernel's **defining module** is patched -- the same rule as every other provenance
    test, since dispatch resolves each kernel by ``(module_path, attr)`` at call time. This
    used to patch ``main`` instead, because the ladder there resolved the name from its own
    globals.
    """
    if not sphere_splat.sphere_splat_available():
        pytest.skip("fused CPU splat unavailable")
    frac, inv_frac, dims, f64 = _cell(100.0, dtype=dtype)
    calls = []
    real = sphere_splat.add_isotropic_cpu_sphere_var

    def recording(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(sphere_splat, "add_isotropic_cpu_sphere_var", recording)
    _build(Engine.AUTO, dims, frac, inv_frac, _voxel_size(f64, dims), dtype,
           iso=_iso_atoms(f64, dtype=dtype))
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


# =========================================================================
# Ported from now-deleted tests of orphaned kernels
# =========================================================================
# ``tests/unit/base/test_aniso_map_building.py`` and ``test_cpu_scatter.py`` covered the
# legacy fixed-radius splat and the C++ structured scatter. The dispatch routes CPU to the
# fused sphere splat or the portable one, so that whole chain -- the grouped-separable and
# cube splats, the separable density core, the structured scatter -- became unreachable
# from ``build_electron_density`` and has since been deleted. The two checks worth keeping
# are re-pointed at the live path here.


@pytest.mark.parametrize("beta", _BETAS)
@pytest.mark.parametrize("engine", [Engine.AUTO, Engine.EAGER], ids=["auto", "eager"])
def test_aniso_reduces_to_isotropic(beta, engine):
    """``U = b/(8*pi^2) I`` must reproduce the isotropic splat, on the live dispatch.

    An analytic identity rather than a comparison against another implementation: a
    spherical anisotropic tensor *is* an isotropic B factor, so the two code paths must
    agree exactly whatever the cell. It constrains the ``8*pi^2`` conversion and the
    diagonal handling of the Mahalanobis form in one assertion, and it fails for a whole
    class of index and factor errors that cross-backend parity cannot see because both
    sides would share them.

    Ported from ``test_aniso_map_building.py::test_aniso_reduces_to_isotropic``, which
    asserted the same identity against ``vectorized_add_to_map_aniso`` -- a kernel no
    longer on any dispatch path.
    """
    frac, inv_frac, dims, f64 = _cell(beta, dtype=torch.float64)
    xyz, adp, occ, A, B = _iso_atoms(f64, n=24, dtype=torch.float64)
    voxel = _voxel_size(f64, dims)
    grid = torch.zeros(*dims, 3, dtype=torch.float64)

    u_sph = torch.zeros(xyz.shape[0], 6, dtype=torch.float64)
    u_sph[:, :3] = (adp / (8.0 * math.pi**2)).unsqueeze(1)

    with use_engine(engine):
        iso_map = build_electron_density(
            grid, xyz, adp, occ, A, B, inv_frac, frac, voxel, dtype=torch.float64
        )
        aniso_map = build_electron_density(
            grid,
            xyz[:0], adp[:0], occ[:0], A[:0], B[:0],
            inv_frac, frac, voxel,
            xyz_aniso=xyz, u_aniso=u_sph, occ_aniso=occ, A_aniso=A, B_aniso=B,
            dtype=torch.float64,
        )

    rel = _rel_l2(aniso_map, iso_map)
    assert rel < 1e-12, (
        f"beta={beta}, {engine.value}: a spherical U does not reproduce the iso splat "
        f"(rel L2 {rel:.3e}). Suspect the 8*pi^2 conversion or the diagonal of the "
        f"Mahalanobis form."
    )
    assert float(iso_map.abs().sum()) > 0, "both maps are empty; the identity is vacuous"


@pytest.mark.parametrize("n_threads", [1, 2, 4])
def test_fused_kernel_is_thread_invariant(n_threads):
    """The fused splat must give the same map at any thread count.

    It partitions the *output* across threads via ``at::parallel_for`` rather than using
    atomics, so a race or a partition-boundary error would show up as a thread-count
    dependence. Deliberately non-orthogonal and dense enough that many atoms' spheres
    overlap, since a partitioning bug is invisible when no two atoms touch the same voxel.

    **Weak on macOS.** ``_cpp_build.py`` omits ``-fopenmp`` there, and the extension
    measurably does not parallelize on this host -- 3000 atoms on a 120x108x96 grid take
    132/131/131 ms at 1/2/4 threads. So this passes locally largely because there is only
    one thread to disagree with. It still earns its place: it is real coverage wherever the
    extension is built with OpenMP (Linux, CI), and it costs milliseconds.

    Bit-exactness is the right assertion and is not merely aspirational -- verified to hold
    on this scene and on a 6x denser one at 2, 4 and 8 threads. Output partitioning means
    each voxel is accumulated by one thread over a fixed atom order, so a nonzero
    difference is an ordering change worth investigating, not float noise. (When checking
    this by hand, compare the maps, not ``dm.sum()``: ``Tensor.sum`` uses a
    thread-count-dependent tree reduction and will show a spurious difference of its own.)

    Ported from the ``n_threads``-parametrized thread-safety test in
    ``test_cpu_scatter.py``, which exercised the C++ structured scatter -- no longer
    reachable from the dispatch.
    """
    if not sphere_splat.sphere_splat_available():
        pytest.skip(f"fused CPU sphere splat unavailable: {sphere_splat.last_error()}")

    frac, inv_frac, dims, f64 = _cell(115.0, dtype=torch.float32)
    xyz, adp, occ, A, B = _iso_atoms(f64, n=96, dtype=torch.float32, seed=7)
    voxel = _voxel_size(f64, dims)
    grid = torch.zeros(*dims, 3, dtype=torch.float32)

    original = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with use_engine(Engine.AUTO):
            ref = build_electron_density(
                grid, xyz, adp, occ, A, B, inv_frac, frac, voxel, dtype=torch.float32
            )
        torch.set_num_threads(n_threads)
        with use_engine(Engine.AUTO):
            got = build_electron_density(
                grid, xyz, adp, occ, A, B, inv_frac, frac, voxel, dtype=torch.float32
            )
    finally:
        torch.set_num_threads(original)

    # Bit-exact: output partitioning means each voxel is summed by exactly one thread in
    # the same order regardless of thread count. Any drift is a real ordering change.
    assert torch.equal(got, ref), (
        f"{n_threads} threads changed the map (max abs diff "
        f"{float((got - ref).abs().max()):.3e}); the fused splat should partition the "
        "output, so results must not depend on thread count"
    )


def test_fused_gate_requires_one_shared_dtype():
    """The fused kernel takes a *uniform* dtype, not any mix of f32 and f64.

    The C++ selects one ``scalar_t`` from the output map via
    ``AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), ...)`` and then reads every other
    tensor through a raw pointer of that type. So a float64 map beside float32 atoms would
    reinterpret the coordinate buffer as doubles -- garbage values and a 2x out-of-bounds
    read -- and the gate has to refuse it.

    Written down because the rule is easy to get wrong when it is restated as a set of
    permitted dtypes: "each tensor's dtype is in {f32, f64}" *admits* the mixed case, while
    the actual requirement is "all tensors share one dtype drawn from {f32, f64}". The two
    read almost identically and only one is memory-safe. In the table that difference is the
    ``require_uniform_dtype`` flag, and this asserts it against the row that ships.
    """
    from torchref.base.electron_density._backends import DENSITY_BACKENDS

    row = DENSITY_BACKENDS.by_name("cpu_sphere")
    f64 = torch.zeros(4, dtype=torch.float64)
    f32 = torch.zeros(4, dtype=torch.float32)
    six = lambda *ts: list(ts) + [ts[-1]] * (6 - len(ts))  # noqa: E731

    # Uniform sets satisfy the device/dtype contract.
    assert row.mismatch(six(f32, f32, f32)) is None
    assert row.mismatch(six(f64, f64, f64)) is None

    # Mixed sets never do, and the message says why.
    for mixed in (six(f64, f32, f32), six(f32, f64, f32)):
        why = row.mismatch(mixed)
        assert why is not None and "single dtype" in why, why


def test_fused_extension_compiles():
    """The fused sphere splat must actually build. Fails rather than skipping.

    Every other test in this file -- and in ``tests/unit/structure_factor`` -- calls
    ``pytest.skip`` when ``sphere_splat_available()`` is False, which is right for them:
    they are testing numerics, and without the extension there is nothing to test. But if
    *every* test skips, a build that has stopped working produces an all-green run while
    the CPU production path has silently degraded to the portable eager splat. The engine
    dispatch is designed to degrade quietly under ``Engine.AUTO`` (see
    ``main.py::_add_isotropic``), which is correct for users and dangerous for CI.

    So exactly one test asserts the extension builds, and reports the captured diagnostic
    plus environment when it does not -- the same stance, and most of the same diagnostic
    surface, as the ``TestCompilation`` class in the now-deleted ``test_cpu_scatter.py``.
    That guard previously protected the C++ structured scatter, a helper; it now protects
    the production CPU splat, so it matters more than it did.
    """
    if sphere_splat.sphere_splat_available():
        return

    import os
    import shutil
    import sys

    err = sphere_splat.last_error()
    env_info = (
        f"  python:    {sys.executable}\n"
        f"  ninja:     {shutil.which('ninja')}\n"
        f"  CXX env:   {os.environ.get('CXX', '<unset>')}\n"
        f"  CC env:    {os.environ.get('CC', '<unset>')}\n"
        f"  PATH head: {os.environ.get('PATH', '').split(':')[:5]}\n"
        f"  TORCH_EXTENSIONS_DIR: "
        f"{os.environ.get('TORCH_EXTENSIONS_DIR', '<unset>')}\n"
    )
    pytest.fail(
        "The fused CPU sphere-splat extension failed to build, so Engine.AUTO on CPU is "
        "silently falling back to the portable eager splat for every density "
        "calculation.\n"
        f"Error: {err}\n\n"
        f"Environment:\n{env_info}"
    )

