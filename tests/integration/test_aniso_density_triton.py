"""Triton-vs-eager equivalence for the anisotropic electron-density kernel.

The fused anisotropic Triton kernel (float32) is compared against the eager
``vectorized_add_to_map_aniso`` (forward + xyz/u/occ gradients), and end-to-end
through ``ModelFT`` on an all-ANISOU structure (AUTO/triton vs EAGER).

Markers: ``@pytest.mark.gpu`` (skipped without ``--run-gpu``) and
``@pytest.mark.integration``. Requires CUDA.
"""

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

_ATOL = 1e-2
_RTOL = 1e-3
_PDB = "tests/files/pdb/7L84.pdb"  # all-anisotropic


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the anisotropic Triton kernel")
    return torch.device("cuda")


def _rel(a, b):
    return ((a - b).abs().max() / (b.abs().max() + 1e-8)).item()


def _model(cuda):
    from torchref.model.model_ft import ModelFT

    m = ModelFT(max_res=2.5, verbose=0)
    m.load_pdb(_PDB)
    return m.to(cuda)


def test_aniso_triton_vs_eager_fwd_bwd(cuda):
    from torchref.base.kernels.triton_kernel import aniso_fused_find_and_place_atoms
    from torchref.base.electron_density.voxel_utils import find_relevant_voxels
    from torchref.base.electron_density.map_building import vectorized_add_to_map_aniso

    m = _model(cuda)
    fft = m.fft
    grid = fft.real_space_grid.detach()
    inv_frac = fft.inv_fractional_matrix.detach()
    frac = fft.fractional_matrix.detach()
    vox = fft.voxel_size
    rad = fft.radius_angstrom

    xyz0, u0, occ0, A, B = m.get_aniso()
    n = 40
    xyz0 = xyz0[:n].detach().float()
    u0 = u0[:n].detach().float()
    occ0 = occ0[:n].detach().float()
    A = A[:n].detach().float()
    B = B[:n].detach().float()

    torch.manual_seed(0)
    base = torch.zeros(grid.shape[:3], dtype=torch.float32, device=cuda)
    gout = torch.randn_like(base)  # asymmetric upstream gradient

    def run(kind):
        xyz = xyz0.clone().requires_grad_()
        u = u0.clone().requires_grad_()
        occ = occ0.clone().requires_grad_()
        if kind == "triton":
            dm = aniso_fused_find_and_place_atoms(
                grid, base, xyz, u, inv_frac, frac, A, B, occ, rad, vox
            )
        else:
            surr, idx = find_relevant_voxels(
                grid, xyz, radius_angstrom=rad, inv_frac_matrix=inv_frac
            )
            dm = vectorized_add_to_map_aniso(
                surr, idx, base.clone(), xyz, u, inv_frac, frac, A, B, occ
            )
        (dm * gout).sum().backward()
        return dm.detach(), xyz.grad, u.grad, occ.grad

    t = run("triton")
    e = run("eager")
    assert _rel(t[0], e[0]) < _RTOL          # density map
    assert _rel(t[1], e[1]) < _RTOL          # grad xyz
    assert _rel(t[2], e[2]) < _RTOL          # grad U (the new code)
    assert _rel(t[3], e[3]) < _RTOL          # grad occ


def test_aniso_modelft_engine_toggle(cuda):
    """End-to-end F_calc on 7L84 (all-ANISOU): AUTO(triton) vs EAGER."""
    from torchref.utils import Engine, use_engine

    m = _model(cuda)
    assert m.get_aniso()[0].shape[0] > 0 and m.get_iso()[0].shape[0] == 0

    torch.manual_seed(0)
    hkl = torch.randint(-12, 13, (400, 3), device=cuda)
    hkl = hkl[(hkl.abs().sum(1) > 0)].float()

    def fcalc(engine):
        with use_engine(engine):
            if hasattr(m, "reset_cache"):
                m.reset_cache()
            return m.forward(hkl, apply_anomalous=False).detach()

    Ft = fcalc(Engine.AUTO)
    Fe = fcalc(Engine.EAGER)
    assert _rel(Ft, Fe) < _RTOL
    amp_corr = np.corrcoef(Ft.abs().cpu().numpy(), Fe.abs().cpu().numpy())[0, 1]
    assert amp_corr > 0.999
