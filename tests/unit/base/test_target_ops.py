"""The target-operator machinery, exercised without a GPU.

``target_op`` turns a forward/backward kernel pair into ``torch.ops.torchref`` operators
and an ``autograd.Function``. The shipped kernels are Triton (CUDA only), so here the
eager bond math is registered as a *CPU* kernel for the duration of a test, through a
scoped library that is removed afterwards. That isolates the plumbing -- argument order,
saved tensors, the gradient routing and the first-order guard -- from the kernels, which
``tests/integration/test_triton_vs_eager_targets.py`` checks on CUDA.
"""

import pytest
import torch

from torchref.base.targets._common import EPS
from torchref.base.targets.bond import _bond_math_eager
from torchref.base.targets.ops import BondNLL

pytestmark = pytest.mark.unit


def _eager_fwd(xyz, idx, references, sigmas):
    return _bond_math_eager(xyz, idx, references, sigmas)


def _eager_bwd(grad, xyz, idx, references, sigmas):
    # Analytic, like the Triton backward: dispatcher kernels run below autograd, so an
    # ``autograd.grad`` here would see no graph.
    diff = xyz[idx[:, 1]] - xyz[idx[:, 0]]
    length = (diff.pow(2).sum(-1) + EPS).sqrt()
    coef = ((length - references) / sigmas**2 / length).unsqueeze(-1) * diff * grad
    out = torch.zeros_like(xyz)
    out.index_add_(0, idx[:, 1], coef)
    out.index_add_(0, idx[:, 0], -coef)
    return out


@pytest.fixture
def cpu_bond_kernels():
    with torch.library._scoped_library("torchref", "IMPL") as lib:
        lib.impl("bond_nll_fwd", _eager_fwd, "CPU")
        lib.impl("bond_nll_bwd", _eager_bwd, "CPU")
        yield


def _bonds(n_atoms=20, n_bonds=30, dtype=torch.float64):
    g = torch.Generator().manual_seed(5)
    xyz = 3.0 * torch.randn(n_atoms, 3, generator=g, dtype=dtype)
    idx = torch.randint(0, n_atoms, (n_bonds, 2), generator=g)
    ref = 1.2 + 0.5 * torch.rand(n_bonds, generator=g, dtype=dtype)
    sig = 0.02 + 0.03 * torch.rand(n_bonds, generator=g, dtype=dtype)
    return xyz, idx, ref, sig


def test_target_function_matches_eager_value_and_gradient(cpu_bond_kernels):
    xyz, idx, ref, sig = _bonds()
    x1 = xyz.clone().requires_grad_(True)
    x2 = xyz.clone().requires_grad_(True)
    loss_op = BondNLL.apply(x1, idx, ref, sig)
    loss_eager = _bond_math_eager(x2, idx, ref, sig)
    (3.0 * loss_op).backward()
    (3.0 * loss_eager).backward()
    torch.testing.assert_close(loss_op, loss_eager, rtol=1e-12, atol=0)
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-12, atol=1e-12)


def test_target_function_refuses_second_order(cpu_bond_kernels):
    xyz, idx, ref, sig = _bonds()
    x = xyz.clone().requires_grad_(True)
    loss = BondNLL.apply(x, idx, ref, sig) + (x ** 2).sum()
    with pytest.raises(RuntimeError, match=r"BondNLL\.backward: the second derivative"):
        torch.autograd.grad(loss, x, create_graph=True)


def test_target_ops_pass_opcheck(cpu_bond_kernels):
    xyz, idx, ref, sig = _bonds()
    torch.library.opcheck(BondNLL.fwd, (xyz, idx, ref, sig))
    torch.library.opcheck(BondNLL.bwd, (torch.tensor(2.0, dtype=xyz.dtype), xyz, idx, ref, sig))


def test_target_op_has_no_cpu_kernel_of_its_own():
    """Outside the fixture only CUDA is registered: a CPU call is a dispatcher error.

    The call sites never make one -- ``use_triton`` routes CPU tensors to the eager path --
    so this pins that the operator does not silently grow a CPU fallback.
    """
    xyz, idx, ref, sig = _bonds(dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="bond_nll_fwd"):
        BondNLL.fwd(xyz, idx, ref, sig)
