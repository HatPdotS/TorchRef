"""``torchref.utils.matmul.matmul`` is exact where it should be and right on MPS.

The MPS cases poison the caching allocator with blocks full of junk first: on an M1/M2
GPU a long-reduction ``a @ b`` sums that junk into its result, so a helper that fell
back to one kernel call would come out wrong, not merely imprecise.
"""

import pytest
import torch

from torchref.utils.matmul import matmul

pytestmark = pytest.mark.unit


def _poison_mps_allocator(value: float) -> None:
    for k in range(6, 24):
        junk = torch.full((2**k,), value, device="mps")
        torch.mps.synchronize()
        del junk


def test_cpu_is_one_plain_matmul():
    """Off MPS the helper is ``a @ b`` itself, bit for bit, at any reduction length."""
    g = torch.Generator().manual_seed(0)
    a = torch.randn(40000, 17, generator=g)
    w = torch.randn(40000, generator=g)
    assert torch.equal(matmul(a.T, a), a.T @ a)
    assert torch.equal(matmul(a.T, w), a.T @ w)


@pytest.mark.mps
@pytest.mark.parametrize("value", [float("nan"), 7.0])
@pytest.mark.parametrize("k", [32767, 50001, 68468])
@pytest.mark.parametrize("n", [16, 17, 64])
def test_long_reduction_is_exact_on_mps(k, n, value):
    """A Gram matrix of ones over ``k`` rows is ``k`` in every entry, whatever the
    allocator held before."""
    a = torch.ones(k, n, device="mps")
    _poison_mps_allocator(value)
    gram = matmul(a.T, a).cpu()
    assert torch.equal(gram, torch.full((n, n), float(k)))


@pytest.mark.mps
def test_mps_matches_cpu_for_vectors_and_batches():
    """Matrix-vector, vector-vector and batched products agree with the CPU."""
    g = torch.Generator().manual_seed(1)
    a = torch.randn(3, 68468, 17, generator=g)
    v = torch.randn(68468, generator=g)
    am, vm = a.to("mps"), v.to("mps")
    _poison_mps_allocator(float("nan"))
    cases = [
        (matmul(am[0].T, vm), a[0].T @ v),
        (matmul(vm, vm), v @ v),
        (matmul(am.mT, am), a.mT @ a),
    ]
    for got, want in cases:
        torch.testing.assert_close(got.cpu(), want, rtol=1e-4, atol=1e-2)


@pytest.mark.mps
def test_gradients_flow_through_the_chunks():
    """The chunked product has the same gradient as the plain one on the CPU."""
    g = torch.Generator().manual_seed(2)
    a = torch.randn(50001, 16, generator=g)
    am = a.to("mps").requires_grad_(True)
    ac = a.clone().requires_grad_(True)
    matmul(am.T, am).sum().backward()
    (ac.T @ ac).sum().backward()
    torch.testing.assert_close(am.grad.cpu(), ac.grad, rtol=1e-4, atol=1e-3)
