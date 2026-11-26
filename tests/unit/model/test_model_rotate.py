"""``Model.rotate`` turns anisotropic ADPs with the coordinates.

Pinned physically rather than against the formula: the structure factors of the
rotated model at ``s`` must equal the original's at ``R^T s`` (both about the
rotation centre), which holds for an anisotropic atom only if ``U' = R U R^T``.
Uses 7L84, a deposited model with both isotropic and anisotropic atoms.
"""

import pytest
import torch

from torchref.base.direct_summation import ds_aniso, ds_iso

pytestmark = pytest.mark.unit


@pytest.fixture
def mixed_model(pdb_dir):
    from torchref.model.model_ft import ModelFT

    m = ModelFT()
    m.load_pdb(str(pdb_dir / "7L84.pdb"))
    return m


def _rotation(seed, dtype):
    """A random proper rotation."""
    g = torch.Generator().manual_seed(seed)
    # dtype-ok: QR of a 3x3 seed matrix, cast to the model's dtype below
    Q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    return (Q * torch.linalg.det(Q)).to(dtype)


def _transform(model, s, center):
    """Direct-summation ``F(s)`` about ``center``; ``s`` Cartesian in Å⁻¹."""
    xyz, adp, occ, A, B = model.get_iso()
    F = ds_iso(s, s.norm(dim=-1), (xyz - center).contiguous(), occ, adp, A, B)
    xyz, u, occ, A, B = model.get_aniso()
    return F + ds_aniso(s, s, (xyz - center).contiguous(), occ, u, A, B)


def test_rotate_turns_anisotropic_adps_with_the_coordinates(mixed_model):
    model = mixed_model
    flag = model.aniso_flag.clone()
    assert flag.any() and not flag.all()
    dtype, dev = model.dtype_float, model.device
    R = _rotation(0, dtype).to(dev)
    center = model.get_centroid().detach()
    rotated = model.copy().rotate(R, center=center)

    g = torch.Generator().manual_seed(1)
    s = ((torch.rand(400, 3, generator=g) - 0.5) * 0.8).to(device=dev, dtype=dtype)
    with torch.no_grad():
        got = _transform(rotated, s, center)
        # Row vectors: (R^T s)^T = s^T R.
        want = _transform(model, s @ R, center)
    err = float((got - want).abs().max() / want.abs().max())
    assert err < 1e-4, f"{err:.2e}"
    assert torch.equal(rotated.aniso_flag, flag)
    assert torch.isnan(rotated.u()[~flag]).all()
