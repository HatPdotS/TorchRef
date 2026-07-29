"""Regression tests for kernel/dispatch/dtype fixes.

Covers the issues found during gradient-correctness stress testing (see
the plan / memory ``kernel_stress_findings.md``):

* Fix 5 — eager SF helpers compute scattering factors from A/B even when
  ``max_memory_gb=None`` and ``scattering_factors=None``.
* Fix 2 — runtime ``dtypes.float``/``dtypes.complex`` switch is honored by
  freshly-constructed models (no frozen default-arg dtype).
* Fix 3 — eager geometry restraints produce finite gradients at degenerate
  geometry (NaN-safe, matching the Triton kernels).
* Fix 4 — a clear error is raised on model-boundary dtype mismatch.
"""

import itertools
import tempfile

import pytest
import torch

from torchref.config import device, dtypes

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fix 5 — None scattering factors on the non-batched eager path
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Fix 3 — eager geometry math is NaN-safe at degenerate geometry
# ---------------------------------------------------------------------------
def test_angle_eager_finite_grad_collinear():
    """Collinear angle (a-b-c on a line) -> finite eager gradient, not NaN."""
    from torchref.base.targets.angle import _angle_math_eager

    xyz = torch.tensor([[1.0, 0, 0], [0.0, 0, 0], [2.0, 0, 0]], requires_grad=True)
    idx = torch.tensor([[0, 1, 2]])
    loss = _angle_math_eager(xyz, idx, torch.tensor([1.9]), torch.tensor([0.05]))
    (grad,) = torch.autograd.grad(loss, xyz)
    assert torch.isfinite(loss).all()
    assert torch.isfinite(grad).all()


def test_bond_eager_finite_grad_coincident():
    """Coincident atoms (zero-length bond) -> finite eager gradient, not NaN."""
    from torchref.base.targets.bond import _bond_math_eager

    xyz = torch.zeros(2, 3, requires_grad=True)  # both at the origin
    idx = torch.tensor([[0, 1]])
    loss = _bond_math_eager(xyz, idx, torch.tensor([1.5]), torch.tensor([0.02]))
    (grad,) = torch.autograd.grad(loss, xyz)
    assert torch.isfinite(loss).all()
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize(
    "coords",
    [
        [[0.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]],  # zero central bond
        [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]],  # fully collinear
    ],
)
def test_torsion_eager_finite_grad_degenerate(coords):
    """Degenerate dihedral (zero |b2| / collinear) -> finite eager gradient."""
    from torchref.base.targets._common import torsions_from_xyz

    xyz = torch.tensor(coords, requires_grad=True)
    idx = torch.tensor([[0, 1, 2, 3]])
    loss = torsions_from_xyz(xyz, idx).sum()
    (grad,) = torch.autograd.grad(loss, xyz)
    assert torch.isfinite(grad).all()


# ---------------------------------------------------------------------------
# Fix 2 — runtime dtype switch is honored by freshly-constructed objects
# ---------------------------------------------------------------------------
def test_runtime_dtype_switch_model(double_cpu):
    """A runtime ``dtypes.float = float64`` reaches new Model/ModelFT/SF objects.

    Regression for the frozen default-arg dtype bug: constructors used
    ``dtype = get_float_dtype()`` as a default, evaluated once at import, so a
    later ``dtypes.float`` change was ignored.
    """
    from torchref.model.model import Model
    from torchref.model.model_ft import ModelFT
    from torchref.symmetry import Cell, SpaceGroup

    assert Model().dtype_float == torch.float64
    assert ModelFT().dtype_float == torch.float64
    assert Cell([20.0, 20, 20, 90, 90, 90]).data.dtype == torch.float64
    assert SpaceGroup("P 1").matrices.dtype == torch.float64


def test_runtime_dtype_switch_forward_complex128(double_cpu, tmp_path):
    """End-to-end: float64 model -> ModelFT.forward returns complex128."""
    import itertools

    from torchref.model.model_ft import ModelFT

    pdb = tmp_path / "p1.pdb"
    g = torch.Generator().manual_seed(0)
    lines = ["CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1"]
    for i in range(8):
        x, y, z = (torch.rand(3, generator=g) * 15 + 2.5).tolist()
        lines.append(
            f"ATOM  {i + 1:5d}  C   GLY A{i + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        )
    lines.append("END")
    pdb.write_text("\n".join(lines) + "\n")

    model = ModelFT(max_res=1.5, verbose=0)
    model.load_pdb(str(pdb))
    assert model.xyz.refinable_params.dtype == torch.float64
    hkl = torch.tensor(
        [h for h in itertools.product(range(-2, 3), repeat=3) if any(h)],
        dtype=torch.float64,
    )
    assert model(hkl).dtype == torch.complex128


# ---------------------------------------------------------------------------
# Fix 4 — clear error on a model-boundary dtype mismatch
# ---------------------------------------------------------------------------
def _tiny_p1_model(tmp_path, max_res=1.5):
    import itertools

    from torchref.model.model_ft import ModelFT

    pdb = tmp_path / "p1.pdb"
    g = torch.Generator().manual_seed(0)
    lines = ["CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1"]
    for i in range(8):
        x, y, z = (torch.rand(3, generator=g) * 15 + 2.5).tolist()
        lines.append(
            f"ATOM  {i + 1:5d}  C   GLY A{i + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        )
    lines.append("END")
    pdb.write_text("\n".join(lines) + "\n")
    model = ModelFT(max_res=max_res, verbose=0)
    model.load_pdb(str(pdb))
    hkl_int = torch.tensor(
        [h for h in itertools.product(range(-2, 3), repeat=3) if any(h)]
    )
    return model, hkl_int


def test_forward_dtype_mismatch_raises(tmp_path):
    """float64 hkl into a float32 model -> clear TypeError, not a cryptic crash."""
    from torchref.config import device, dtypes

    f0, d0 = dtypes.float, device.current
    dtypes.float = torch.float32
    device.current = torch.device("cpu")
    try:
        model, hkl_int = _tiny_p1_model(tmp_path)
        # integer + matching-float hkl are accepted
        assert model(hkl_int).dtype == torch.complex64
        assert model(hkl_int.to(torch.float32)).dtype == torch.complex64
        # float64 hkl into a float32 model is rejected with a clear message
        with pytest.raises(TypeError, match="hkl has dtype torch.float64"):
            model(hkl_int.to(torch.float64))
    finally:
        dtypes.float = f0
        device.current = d0


