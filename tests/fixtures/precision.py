"""Scope numerical reference configuration and expose comparison tolerances."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch

import torchref
from torchref.config import device, dtypes


@contextmanager
def cpu_double_precision() -> Iterator[None]:
    """Temporarily select CPU float64/complex128 for numerical references.

    Notes
    -----
    Mutate process-wide TorchRef defaults, not PyTorch factory defaults. Restore
    float/complex dtype, device, and density cutoff even when the body raises.
    Objects allocated inside the context retain their own dtype and device.
    """
    original = dtypes.float, dtypes.complex, device.current
    cutoff = torchref.sigma_cutoff_ed.value
    dtypes.float = torch.float64
    dtypes.complex = torch.complex128
    device.current = torch.device("cpu")
    try:
        yield
    finally:
        dtypes.float, dtypes.complex, device.current = original
        torchref.sigma_cutoff_ed.value = cutoff


@pytest.fixture
def double_cpu() -> Iterator[None]:
    """Use CPU double precision for one test and restore configuration afterward."""
    with cpu_double_precision():
        yield


@pytest.fixture
def rtol() -> float:
    """Relative tolerance for floating point comparisons."""
    return 1e-5


@pytest.fixture
def atol() -> float:
    """Absolute tolerance for floating point comparisons."""
    return 1e-8
