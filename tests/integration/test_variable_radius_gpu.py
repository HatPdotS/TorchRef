"""GPU variable-radius density path: Triton (wq_grid / wq_grid_aniso) vs the CPU
grouped splat, which share the identical per-atom radius policy.

Both backends truncate each atom at its own ``N_sigma * sigma_eff`` radius, so the
forward maps must agree to float32 + analytic-vs-gathered-coord tolerance and the
gradients (xyz / adp / u / occ) must be parallel. Requires CUDA + Triton.

Markers: ``@pytest.mark.gpu`` (skipped without ``--run-gpu``), ``integration``.
"""

import math

import pytest
import torch

from torchref.base.electron_density.main import build_electron_density
from torchref.utils import Engine, use_engine

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


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
    """Non-orthogonal (beta ~ 100 deg) cell to exercise the inv-frac-norm per-axis
    box and the off-diagonal coordinate math (u_c gains an x-component)."""
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


def test_iso_triton_matches_cpu_grouped(gpu_device):
    cell, grid, frac, inv_frac, voxel, rsg = _cell()
    xyz, adp, occ, A, B = _iso_atoms(cell)

    with use_engine(Engine.AUTO):
        cpu = build_electron_density(
            rsg, xyz, adp, occ, A, B, inv_frac, frac, voxel
        )
    with use_engine(Engine.TRITON):
        gpu = build_electron_density(
            rsg.to(gpu_device), xyz.to(gpu_device), adp.to(gpu_device), occ.to(gpu_device),
            A.to(gpu_device), B.to(gpu_device), inv_frac.to(gpu_device), frac.to(gpu_device), voxel.to(gpu_device),
        )
    assert _rel_map(gpu.cpu(), cpu) < 2e-2
    assert _cos(gpu.cpu(), cpu) > 0.9995


def test_aniso_triton_matches_cpu_grouped(gpu_device):
    cell, grid, frac, inv_frac, voxel, rsg = _cell()
    xa, ua, oa, Aa, Ba = _aniso_atoms(cell)
    empty = torch.zeros(0, 3)

    with use_engine(Engine.AUTO):
        cpu = build_electron_density(
            rsg, empty, torch.zeros(0), torch.zeros(0), torch.zeros(0, 5),
            torch.zeros(0, 5), inv_frac, frac, voxel,
            xyz_aniso=xa, u_aniso=ua, occ_aniso=oa, A_aniso=Aa, B_aniso=Ba,
        )
    with use_engine(Engine.TRITON):
        gpu = build_electron_density(
            rsg.to(gpu_device), empty.to(gpu_device), torch.zeros(0, device=gpu_device),
            torch.zeros(0, device=gpu_device), torch.zeros(0, 5, device=gpu_device),
            torch.zeros(0, 5, device=gpu_device), inv_frac.to(gpu_device), frac.to(gpu_device),
            voxel.to(gpu_device),
            xyz_aniso=xa.to(gpu_device), u_aniso=ua.to(gpu_device), occ_aniso=oa.to(gpu_device),
            A_aniso=Aa.to(gpu_device), B_aniso=Ba.to(gpu_device),
        )
    assert _rel_map(gpu.cpu(), cpu) < 2e-2
    assert _cos(gpu.cpu(), cpu) > 0.9995


def test_iso_triton_matches_eager_monoclinic(gpu_device):
    """Guard the per-axis (inv-frac-norm) box + direct coords on a sheared cell:
    TRITON (CUDA) vs the portable EAGER plain-scatter reference (independent box)."""
    cell, grid, frac, inv_frac, voxel, rsg = _cell_monoclinic()
    xyz, adp, occ, A, B = _iso_atoms(cell)

    with use_engine(Engine.EAGER):
        ref = build_electron_density(
            rsg, xyz, adp, occ, A, B, inv_frac, frac, voxel
        )
    with use_engine(Engine.TRITON):
        gpu = build_electron_density(
            rsg.to(gpu_device), xyz.to(gpu_device), adp.to(gpu_device), occ.to(gpu_device),
            A.to(gpu_device), B.to(gpu_device), inv_frac.to(gpu_device), frac.to(gpu_device),
            voxel.to(gpu_device),
        )
    assert _rel_map(gpu.cpu(), ref) < 2e-2
    assert _cos(gpu.cpu(), ref) > 0.9995


def test_iso_gradients_match_cpu_grouped(gpu_device):
    cell, grid, frac, inv_frac, voxel, rsg = _cell()
    xyz, adp, occ, A, B = _iso_atoms(cell)
    w = torch.randn(grid)

    def run(device, engine):
        xx = xyz.to(device).clone().requires_grad_()
        aa = adp.to(device).clone().requires_grad_()
        oo = occ.to(device).clone().requires_grad_()
        with use_engine(engine):
            dm = build_electron_density(
                rsg.to(device), xx, aa, oo, A.to(device), B.to(device),
                inv_frac.to(device), frac.to(device), voxel.to(device),
            )
        (dm * w.to(device)).sum().backward()
        return xx.grad, aa.grad, oo.grad

    gx_c, ga_c, go_c = run("cpu", Engine.AUTO)
    gx_g, ga_g, go_g = run(gpu_device, Engine.TRITON)
    assert _cos(gx_g.cpu(), gx_c) > 0.999
    assert _cos(ga_g.cpu(), ga_c) > 0.999
    assert _cos(go_g.cpu(), go_c) > 0.999


def test_aniso_gradients_match_cpu_grouped(gpu_device):
    cell, grid, frac, inv_frac, voxel, rsg = _cell()
    xa, ua, oa, Aa, Ba = _aniso_atoms(cell)
    empty = torch.zeros(0, 3)
    w = torch.randn(grid)

    def run(device, engine):
        xx = xa.to(device).clone().requires_grad_()
        uu = ua.to(device).clone().requires_grad_()
        with use_engine(engine):
            dm = build_electron_density(
                rsg.to(device), empty.to(device), torch.zeros(0, device=device),
                torch.zeros(0, device=device), torch.zeros(0, 5, device=device),
                torch.zeros(0, 5, device=device), inv_frac.to(device),
                frac.to(device), voxel.to(device),
                xyz_aniso=xx, u_aniso=uu, occ_aniso=oa.to(device),
                A_aniso=Aa.to(device), B_aniso=Ba.to(device),
            )
        (dm * w.to(device)).sum().backward()
        return xx.grad, uu.grad

    gx_c, gu_c = run("cpu", Engine.AUTO)
    gx_g, gu_g = run(gpu_device, Engine.TRITON)
    assert _cos(gx_g.cpu(), gx_c) > 0.999
    assert _cos(gu_g.cpu(), gu_c) > 0.999
