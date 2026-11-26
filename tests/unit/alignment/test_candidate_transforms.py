"""The two routes to a search model's oriented transforms agree.

``candidate_transforms`` either orients one P1 copy per rotation and runs the FFT
structure factors (``method="fft"``), or evaluates every orientation in one direct
summation at ``s^T R`` (``method="direct"``), which turns anisotropic ADPs
implicitly. Agreement at a general orientation on a model with anisotropic atoms
therefore pins that the FFT route turns ``U`` with the coordinates.
"""

from types import SimpleNamespace

import pytest
import torch

from torchref.experimental.alignment.translation_batched import candidate_transforms

pytestmark = pytest.mark.unit

#: The FFT route's alias blur and grid, as a fraction of the transforms' rms.
_TOL = 5e-3


def test_fft_and_direct_routes_agree_for_anisotropic_atoms(pdb_dir):
    from torchref.model.model_ft import ModelFT

    model = ModelFT()
    model.load_pdb(str(pdb_dir / "7L84.pdb"))
    assert model.aniso_flag.any()
    dtype, dev = model.dtype_float, model.device
    cell, sg = model.cell, model.spacegroup

    r = torch.arange(-20, 21)
    hkl = torch.cartesian_prod(r, r, r).to(dev)
    rec = cell.reciprocal_basis_matrix.detach().to(device=dev, dtype=dtype)
    s_mag = (hkl.to(dtype) @ rec).norm(dim=-1)
    keep = (s_mag > 1.0 / 15.0) & (s_mag < 1.0 / 4.0)
    g = torch.Generator().manual_seed(0)
    pick = torch.randperm(int(keep.sum()), generator=g)[:1500].to(dev)
    hkl, s_mag = hkl[keep][pick], s_mag[keep][pick]
    obs = SimpleNamespace(hkl=hkl, s_mag=s_mag, eps=torch.ones_like(s_mag))

    # dtype-ok: QR of a 3x3 seed matrix, cast to the model's dtype below
    Q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    rots = torch.stack([torch.eye(3, dtype=torch.float64), Q * torch.linalg.det(Q)])
    rots = rots.to(device=dev, dtype=dtype)

    with torch.no_grad():
        fft = candidate_transforms(model, obs, sg, cell, rots, method="fft")
        direct = candidate_transforms(model, obs, sg, cell, rots, method="direct")
    for k in range(rots.shape[0]):
        diff = (fft[k] - direct[k]).abs().pow(2).mean().sqrt()
        err = float(diff / direct[k].abs().pow(2).mean().sqrt())
        assert err < _TOL, f"rotation {k}: {err:.2e}"
