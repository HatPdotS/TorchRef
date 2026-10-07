"""The voxel-list density path of ``jit_reference``: lazy scripting and its cache.

The cache directory is read from ``TORCHREF_COMPILE_CACHE`` when the module is imported,
so these tests run ``torchref`` in a subprocess with the variable set.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import torchref

pytestmark = pytest.mark.unit

_REPO = Path(torchref.__file__).resolve().parents[1]


class _StaleKernel(torch.nn.Module):
    """A cached kernel from another build: same signature, wrong numbers."""

    def forward(
        self,
        coords_frac: torch.Tensor,
        voxel_indices: torch.Tensor,
        density_map: torch.Tensor,
        xyz: torch.Tensor,
        b: torch.Tensor,
        inv_frac_matrix: torch.Tensor,
        G: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        occ: torch.Tensor,
    ) -> torch.Tensor:
        return density_map + 1000.0


def _run_torchref(code: str, cache_dir: Path) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter with ``TORCHREF_COMPILE_CACHE=cache_dir``."""
    env = dict(os.environ, TORCHREF_COMPILE_CACHE=str(cache_dir))
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO), os.environ.get("PYTHONPATH")) if p
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO,
    )


def test_import_does_not_create_the_kernel_cache(tmp_path):
    """``import torchref`` succeeds when the JIT cache directory cannot be created."""
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the cache directory's parent should be")
    proc = _run_torchref("import torchref", blocker / "cache")
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_unversioned_cache_file_is_not_loaded(tmp_path, pdb_dir):
    """A ``jit_cpu_kernel.pt`` written by another build is ignored, not executed.

    The CPU kernel is scripted afresh on 1DAW (3051 atoms, voxels within 3 A) and
    must match a freshly scripted ``_CpuDensityKernel`` exactly; its cache file is
    keyed on the torch version and the kernel source.
    """
    torch.jit.save(
        torch.jit.script(_StaleKernel()), str(tmp_path / "jit_cpu_kernel.pt")
    )
    code = f"""
import torch
from torchref.base.electron_density.kernels.cpu import jit_reference as J
from torchref.base.electron_density.voxel_utils import find_relevant_voxels
from torchref.base.fourier import get_real_grid
from torchref.base.scattering.scattering_table import (
    elements_to_z,
    get_scattering_params_by_z,
)
from torchref.io.pdb import PDBReader
from torchref.symmetry.cell import Cell

df, cell, _ = PDBReader().read({str(pdb_dir / "1DAW.pdb")!r})()
cell = Cell(cell)
frac, inv_frac = cell.fractional_matrix, cell.inv_fractional_matrix
xyz = torch.tensor(df[["x", "y", "z"]].to_numpy(), dtype=frac.dtype)
b = torch.tensor(df["tempfactor"].to_numpy(), dtype=frac.dtype)
occ = torch.tensor(df["occupancy"].to_numpy(), dtype=frac.dtype)
z = elements_to_z(df["element"].tolist())
A, B = get_scattering_params_by_z(z, dtype=frac.dtype)
grid = get_real_grid(fractional_matrix=frac, gridsize=(216, 90, 69))
coords, idx = find_relevant_voxels(grid, xyz, 3.0, inv_frac_matrix=inv_frac)

got = J.vectorized_add_to_map(
    coords, idx, torch.zeros(grid.shape[:3], dtype=frac.dtype),
    xyz, b, inv_frac, frac, A, B, occ,
)
want = torch.jit.script(J._CpuDensityKernel())(
    J.precompute_fractional_coords(coords, inv_frac), idx,
    torch.zeros(grid.shape[:3], dtype=frac.dtype),
    xyz, b, inv_frac, J.compute_metric_tensor(frac), A, B, occ,
)
assert torch.equal(got, want), f"max |diff| {{(got - want).abs().max().item()}}"
"""
    proc = _run_torchref(code, tmp_path)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert len(list(tmp_path.glob("jit_cpu_kernel-*.pt"))) == 1


def _split_1daw(pdb_path, device):
    """1DAW as two atom groups of voxel-list splat arguments on ``device``, float32."""
    from torchref.base.electron_density.voxel_utils import find_relevant_voxels
    from torchref.base.fourier import get_real_grid
    from torchref.base.scattering.scattering_table import (
        elements_to_z,
        get_scattering_params_by_z,
    )
    from torchref.io.pdb import PDBReader
    from torchref.symmetry.cell import Cell

    df, cell, _ = PDBReader().read(str(pdb_path))()
    cell = Cell(cell)
    f32 = torch.float32
    frac = cell.fractional_matrix.to(device, f32)
    inv_frac = cell.inv_fractional_matrix.to(device, f32)
    xyz = torch.tensor(df[["x", "y", "z"]].to_numpy(), dtype=f32, device=device)
    b = torch.tensor(df["tempfactor"].to_numpy(), dtype=f32, device=device)
    occ = torch.tensor(df["occupancy"].to_numpy(), dtype=f32, device=device)
    z = elements_to_z(df["element"].tolist()).to(device)
    A, B = get_scattering_params_by_z(z, dtype=f32)
    grid = get_real_grid(fractional_matrix=frac, gridsize=(108, 45, 35))
    coords, idx = find_relevant_voxels(grid, xyz, 3.0, inv_frac_matrix=inv_frac)
    half = xyz.shape[0] // 2
    groups = [
        (coords[s], idx[s], xyz[s], b[s], A[s], B[s], occ[s])
        for s in (slice(0, half), slice(half, None))
    ]
    return groups, inv_frac, frac, grid.shape[:3]


def _first_group_xyz_grad(groups, inv_frac, frac, dims, weight):
    """d/d xyz_1 of ``sum(weight * map)``, the map built by two chained splats."""
    from torchref.base.electron_density.kernels.cpu.jit_reference import (
        vectorized_add_to_map,
    )

    (c1, i1, x1, b1, A1, B1, o1), (c2, i2, x2, b2, A2, B2, o2) = groups
    x1 = x1.clone().requires_grad_(True)
    density = torch.zeros(dims, dtype=frac.dtype, device=frac.device)
    density = vectorized_add_to_map(c1, i1, density, x1, b1, inv_frac, frac, A1, B1, o1)
    density = vectorized_add_to_map(c2, i2, density, x2, b2, inv_frac, frac, A2, B2, o2)
    (grad,) = torch.autograd.grad((weight * density).sum(), x1, allow_unused=True)
    return grad


@pytest.mark.cuda
def test_fused_cuda_splat_passes_the_gradient_to_its_input_map(pdb_dir):
    """The fused Triton splat returns ``density_map + splat``, so its backward must hand
    the incoming gradient to ``density_map``.

    1DAW split into two atom groups: the first group reaches the loss only through the
    second call's ``density_map``, so its coordinate gradient must match the CPU JIT
    path's.
    """
    from torchref.base.targets._dispatch import use_triton

    cuda = torch.device("cuda")
    groups, inv_frac, frac, dims = _split_1daw(pdb_dir / "1DAW.pdb", cuda)
    if not use_triton(groups[0][2]):
        pytest.skip("the fused Triton splat is not selected on this host")
    weight = torch.randn(dims, generator=torch.Generator().manual_seed(0))

    got = _first_group_xyz_grad(groups, inv_frac, frac, dims, weight.to(cuda))
    cpu_groups = [tuple(t.cpu() for t in g) for g in groups]
    want = _first_group_xyz_grad(cpu_groups, inv_frac.cpu(), frac.cpu(), dims, weight)

    assert got is not None, "no gradient reached the first splat's atoms"
    rel = float((got.cpu() - want).norm() / want.norm())
    assert rel < 1e-3, f"CUDA vs CPU first-group xyz gradient rel L2 {rel:.2e}"
