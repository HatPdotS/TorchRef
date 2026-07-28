"""MPS variable-radius density path: native Metal kernels (Engine.AUTO) vs the
portable plain-scatter reference (Engine.EAGER), on the same MPS device.

Both truncate each atom at its own per-atom radius, so the forward maps must
agree to float32 + truncation-shape tolerance and the gradients (xyz / adp / u /
occ) must be parallel. Requires an Apple-silicon GPU (MPS); skipped elsewhere.

Markers: ``@pytest.mark.mps`` (skipped without ``--run-gpu``), ``integration``.
"""

import math

import pytest
import torch

from torchref.base.electron_density.kernels.mps import mps_kernels_available
from torchref.base.electron_density.main import build_electron_density
from torchref.utils import Engine, use_engine

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.fixture
def mps_device(gpu_device):
    if gpu_device.type != "mps":
        pytest.skip("MPS-specific test (Metal kernels)")
    if not mps_kernels_available():
        pytest.skip("Metal splat kernels failed to compile")
    return gpu_device


def _cell():
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


def _cell_monoclinic():
    """Non-orthogonal (beta ~ 100 deg) cell: exercises the per-axis box and the
    off-diagonal coordinate math (u_c gains an x-component)."""
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


def _iso_atoms(cell, n=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    a, b, c = cell
    xyz = torch.rand(n, 3, generator=g) * torch.tensor([a, b, c])
    adp = torch.rand(n, generator=g) * 35 + 3
    occ = torch.ones(n)
    A = torch.rand(n, 5, generator=g) * 5
    B = torch.rand(n, 5, generator=g) * 20 + 2
    return [t.float() for t in (xyz, adp, occ, A, B)]


def _aniso_atoms(cell, n=30, seed=1):
    g = torch.Generator().manual_seed(seed)
    a, b, c = cell
    xyz = torch.rand(n, 3, generator=g) * torch.tensor([a, b, c])
    u = torch.zeros(n, 6)
    u[:, :3] = torch.rand(n, 3, generator=g) * 0.12 + 0.02
    occ = torch.ones(n)
    A = torch.rand(n, 5, generator=g) * 5
    B = torch.rand(n, 5, generator=g) * 20 + 2
    return [t.float() for t in (xyz, u, occ, A, B)]


def _cos(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _rel_map(x, y):
    return float((x - y).abs().max() / (y.abs().max() + 1e-8))


def _to(dev, *ts):
    return [t.to(dev) for t in ts]


def test_iso_metal_matches_plain(mps_device):
    _, grid, frac, inv_frac, voxel, rsg = _cell()
    xyz, adp, occ, A, B = _to(mps_device, *_iso_atoms(_cell()[0]))
    rsg, frac, inv_frac, voxel = _to(mps_device, rsg, frac, inv_frac, voxel)

    def run(engine):
        with use_engine(engine):
            return build_electron_density(rsg, xyz, adp, occ, A, B, inv_frac, frac, voxel)

    ref = run(Engine.EAGER)   # portable plain splat
    got = run(Engine.AUTO)    # Metal
    assert _rel_map(got.cpu(), ref.cpu()) < 2e-2
    assert _cos(got.cpu(), ref.cpu()) > 0.9995


def test_iso_metal_matches_plain_monoclinic(mps_device):
    _, grid, frac, inv_frac, voxel, rsg = _cell_monoclinic()
    xyz, adp, occ, A, B = _to(mps_device, *_iso_atoms(_cell_monoclinic()[0]))
    rsg, frac, inv_frac, voxel = _to(mps_device, rsg, frac, inv_frac, voxel)

    def run(engine):
        with use_engine(engine):
            return build_electron_density(rsg, xyz, adp, occ, A, B, inv_frac, frac, voxel)

    ref = run(Engine.EAGER)
    got = run(Engine.AUTO)
    assert _rel_map(got.cpu(), ref.cpu()) < 2e-2
    assert _cos(got.cpu(), ref.cpu()) > 0.9995


def test_aniso_metal_matches_plain(mps_device):
    _, grid, frac, inv_frac, voxel, rsg = _cell()
    xa, ua, oa, Aa, Ba = _to(mps_device, *_aniso_atoms(_cell()[0]))
    rsg, frac, inv_frac, voxel = _to(mps_device, rsg, frac, inv_frac, voxel)
    empty = torch.zeros(0, 3, device=mps_device)
    z0 = torch.zeros(0, device=mps_device)
    z05 = torch.zeros(0, 5, device=mps_device)

    def run(engine):
        with use_engine(engine):
            return build_electron_density(
                rsg, empty, z0, z0, z05, z05, inv_frac, frac, voxel,
                xyz_aniso=xa, u_aniso=ua, occ_aniso=oa, A_aniso=Aa, B_aniso=Ba,
            )

    ref = run(Engine.EAGER)
    got = run(Engine.AUTO)
    assert _rel_map(got.cpu(), ref.cpu()) < 2e-2
    assert _cos(got.cpu(), ref.cpu()) > 0.9995


def test_iso_gradients_match_plain(mps_device):
    _, grid, frac, inv_frac, voxel, rsg = _cell()
    xyz0, adp0, occ0, A, B = _iso_atoms(_cell()[0])
    rsg, frac, inv_frac, voxel, A, B = _to(mps_device, rsg, frac, inv_frac, voxel, A, B)
    w = torch.randn(grid, device=mps_device)

    def run(engine):
        xx = xyz0.to(mps_device).clone().requires_grad_()
        aa = adp0.to(mps_device).clone().requires_grad_()
        oo = occ0.to(mps_device).clone().requires_grad_()
        with use_engine(engine):
            dm = build_electron_density(rsg, xx, aa, oo, A, B, inv_frac, frac, voxel)
        (dm * w).sum().backward()
        return xx.grad, aa.grad, oo.grad

    gx_r, ga_r, go_r = run(Engine.EAGER)
    gx_m, ga_m, go_m = run(Engine.AUTO)
    assert _cos(gx_m.cpu(), gx_r.cpu()) > 0.999
    assert _cos(ga_m.cpu(), ga_r.cpu()) > 0.999
    assert _cos(go_m.cpu(), go_r.cpu()) > 0.999


def test_aniso_gradients_match_plain(mps_device):
    _, grid, frac, inv_frac, voxel, rsg = _cell()
    xa0, ua0, oa0, Aa, Ba = _aniso_atoms(_cell()[0])
    rsg, frac, inv_frac, voxel, Aa, Ba = _to(mps_device, rsg, frac, inv_frac, voxel, Aa, Ba)
    empty = torch.zeros(0, 3, device=mps_device)
    z0 = torch.zeros(0, device=mps_device)
    z05 = torch.zeros(0, 5, device=mps_device)
    w = torch.randn(grid, device=mps_device)

    def run(engine):
        xx = xa0.to(mps_device).clone().requires_grad_()
        uu = ua0.to(mps_device).clone().requires_grad_()
        with use_engine(engine):
            dm = build_electron_density(
                rsg, empty, z0, z0, z05, z05, inv_frac, frac, voxel,
                xyz_aniso=xx, u_aniso=uu, occ_aniso=oa0.to(mps_device),
                A_aniso=Aa, B_aniso=Ba,
            )
        (dm * w).sum().backward()
        return xx.grad, uu.grad

    gx_r, gu_r = run(Engine.EAGER)
    gx_m, gu_m = run(Engine.AUTO)
    assert _cos(gx_m.cpu(), gx_r.cpu()) > 0.999
    assert _cos(gu_m.cpu(), gu_r.cpu()) > 0.999
