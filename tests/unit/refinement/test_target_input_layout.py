"""Restraint targets score a strided view of the coordinates as they score a copy.

A coordinate wrapper may return a view whose row stride is not 3, for example one
member of an ``(n_members, n_atoms, 3)`` ensemble laid out by NumPy. The CUDA (Triton)
kernels address their inputs as packed row-major arrays, so each target must give the
same loss and the same gradient for such a view as for a packed copy of it. Checked on
1DAW with its own restraints, on CPU and, where present, on CUDA.
"""

import pytest
import torch

from torchref.base.targets.adp import adp_simu_math
from torchref.base.targets.bond import bond_math
from torchref.config import get_int_dtype
from torchref.model.model import Model
from torchref.refinement.targets.geometry import (
    AngleTarget,
    BondTarget,
    ChiralTarget,
    NonBondedHTarget,
    NonBondedTarget,
    PlanarityTarget,
    RamachandranTarget,
    TorsionTarget,
)

pytestmark = pytest.mark.unit

DEVICES = [
    pytest.param(torch.device("cpu"), id="cpu"),
    pytest.param(torch.device("cuda", 0), id="cuda", marks=pytest.mark.cuda),
]

TARGETS = [
    BondTarget,
    AngleTarget,
    TorsionTarget,
    ChiralTarget,
    PlanarityTarget,
    RamachandranTarget,
    NonBondedTarget,
    NonBondedHTarget,
]


def _strided_leaf(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A leaf holding ``values`` twice, and the view of its first copy.

    The view equals ``values`` but its first-dimension stride is doubled, so it is not
    contiguous; its storage offset is zero, which a kernel would read correctly.
    """
    base = torch.stack([values, values], dim=1).requires_grad_()
    return base, base.select(1, 0)


def _strided(values: torch.Tensor) -> torch.Tensor:
    """``values`` as a non-contiguous view with the first-dimension stride doubled."""
    return torch.stack([values, values], dim=1).select(1, 0)


class _Coordinates(torch.nn.Module):
    """Stands in for the model's coordinate wrapper and returns a given tensor."""

    def __init__(self, xyz: torch.Tensor):
        super().__init__()
        self.value = xyz

    def forward(self) -> torch.Tensor:
        return self.value


@pytest.fixture(scope="module", params=DEVICES)
def model(request, sample_cif_file) -> Model:
    """1DAW as deposited, on the parametrized device."""
    model = Model(device=request.param)
    model.load_cif(str(sample_cif_file))
    return model


@pytest.mark.parametrize("target_cls", TARGETS, ids=lambda cls: cls.__name__)
def test_target_scores_a_strided_xyz_view_as_a_packed_copy(
    model, target_cls, monkeypatch
):
    """Loss and coordinate gradient agree for packed and strided ``model.xyz()``."""
    target = target_cls(model)
    values = model.xyz().detach().clone()

    packed = values.clone().requires_grad_()
    monkeypatch.setitem(model._modules, "xyz", _Coordinates(packed))
    expected = target()
    expected.backward()

    base, view = _strided_leaf(values)
    assert torch.equal(view, values) and not view.is_contiguous()
    monkeypatch.setitem(model._modules, "xyz", _Coordinates(view))
    loss = target()
    loss.backward()

    assert torch.isfinite(expected)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(base.grad[:, 0], packed.grad)
    assert not base.grad[:, 1].any()


def test_bond_math_reads_strided_indices_and_parameters(model):
    """Strided index, reference and sigma tensors give the packed bond NLL."""
    bond = model.restraints.restraints["bond"]["all"]
    xyz = model.xyz().detach()
    args = (bond["indices"], bond["references"], bond["sigmas"])

    expected = bond_math(xyz, *args)
    strided = [_strided(t) for t in args]
    assert not any(t.is_contiguous() for t in strided)
    torch.testing.assert_close(bond_math(xyz, *strided), expected)


def test_adp_simu_scores_a_strided_b_view_as_a_packed_copy(model):
    """SIMU loss and B gradient agree for packed and strided isotropic B (Å²)."""
    b = model.adp().detach().clone()
    n = b.shape[0]
    pairs = torch.stack([torch.arange(n - 1), torch.arange(1, n)], dim=1).to(
        device=b.device, dtype=get_int_dtype()
    )
    sigma = b.new_tensor(2.0)

    packed = b.clone().requires_grad_()
    expected = adp_simu_math(packed, pairs, sigma)
    expected.backward()

    base, view = _strided_leaf(b)
    loss = adp_simu_math(view, pairs, sigma)
    loss.backward()

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(base.grad[:, 0], packed.grad)
