"""Pin the RMS gradient norm across one or several parameter tensors."""

import math

import pytest
import torch

from torchref.config import get_default_device, get_float_dtype
from torchref.utils.gradnorm import gradnorm

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("split", [False, True], ids=["single", "multiple"])
def test_gradnorm_rms(split: bool) -> None:
    """The norm weights individual gradient elements, not parameter tensors."""
    values = torch.tensor(
        [1.0, 2.0, 3.0], dtype=get_float_dtype(), device=get_default_device()
    )
    chunks = (values[:1], values[1:]) if split else (values,)
    params = [chunk.clone().requires_grad_() for chunk in chunks]
    loss = sum((param.square().sum() for param in params))
    expected = values.new_tensor(math.sqrt(56.0 / 3.0))
    torch.testing.assert_close(gradnorm(loss, iter(params)), expected)


def test_gradnorm_zero_gradient() -> None:
    """A connected loss with zero derivative has zero RMS gradient."""
    param = torch.ones(
        3, dtype=get_float_dtype(), device=get_default_device(), requires_grad=True
    )
    torch.testing.assert_close(
        gradnorm((param * 0).sum(), [param]), param.new_zeros(())
    )
