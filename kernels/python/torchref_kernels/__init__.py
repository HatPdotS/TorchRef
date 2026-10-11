"""Prebuilt CPU kernels for TorchRef.

Low-level entry points taking ``(data_ptr, numel)`` pairs of CPU tensors. They are not
meant to be called directly: TorchRef's wrappers (``torchref.utils.native``) validate
device, dtype, contiguity and shapes, own the autograd, and choose the thread count.
Nothing here imports torch.
"""

from torchref_kernels._native import ABI_VERSION, build_info
from torchref_kernels import _native

__version__ = build_info()["version"]

__all__ = ["ABI_VERSION", "build_info", "native", "__version__"]


def native():
    """The compiled extension module holding the kernel entry points."""
    return _native
