"""``first_order_only`` refuses a second derivative through a graph-less backward.

The kernel-backed autograd Functions return gradients that carry no graph, so a
``create_graph=True`` backward through them would drop their curvature. The guard
must raise for every such backward, including the two forms ``once_differentiable``
lets through, and leave first-order backward alone.
"""

import pytest
import torch

from torchref.base.targets._dispatch import first_order_only

pytestmark = pytest.mark.unit


class _CubeSum(torch.autograd.Function):
    """``sum(x**3)`` with a graph-less backward, as a kernel-backed Function has."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x.detach() ** 3).sum()

    @staticmethod
    @first_order_only
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * 3.0 * x.detach() ** 2


def test_first_order_backward_is_unchanged():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    (grad,) = torch.autograd.grad(_CubeSum.apply(x), x)
    torch.testing.assert_close(grad, torch.tensor([3.0, 12.0]))


def test_backward_under_no_grad_is_unchanged():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    _CubeSum.apply(x).backward()
    torch.testing.assert_close(x.grad, torch.tensor([3.0, 12.0]))


@pytest.mark.filterwarnings("ignore:Using backward\\(\\) with create_graph=True")
@pytest.mark.parametrize("second_pass", ["autograd.grad", "backward"])
def test_create_graph_raises_naming_the_function(second_pass):
    """A loss linear in the Function's output, the case ``once_differentiable``
    misses, raises on the ``create_graph=True`` pass itself."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    loss = _CubeSum.apply(x) + (x**2).sum()
    with pytest.raises(
        RuntimeError, match=r"_CubeSum\.backward: the second derivative"
    ):
        if second_pass == "autograd.grad":
            torch.autograd.grad(loss, x, create_graph=True)
        else:
            loss.backward(create_graph=True)
