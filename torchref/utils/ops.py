"""TorchRef's kernels as PyTorch operators: ``torch.ops.torchref.*``.

Each accelerated kernel family is one operator with one implementation per device,
registered with PyTorch's dispatcher, which then picks the device kernel from the inputs.
Operators come in forward/backward pairs (``<name>_fwd`` / ``<name>_bwd``) of plain,
non-differentiable kernels; autograd stays in one ``torch.autograd.Function`` per family
that calls the pair. That split is deliberate: ``torch.library.custom_op`` with
``register_autograd`` costs about twice the per-call overhead of an ``autograd.Function``
over a dispatcher operator, which matters for the small geometry kernels called hundreds
of times per optimiser step.

What the dispatcher does *not* decide stays in :mod:`torchref.utils.backends`: whether an
accelerated row applies at all (dtype, availability, ``force_portable``) and what happens
when it fails. The portable reference kernels are plain PyTorch and are never operators,
so their autograd -- including double backward -- is untouched.

Every operator also gets a fake (shape-only) implementation, so ``torch.compile`` can
trace through it, and can be checked with ``torch.library.opcheck``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Callable, Dict, Tuple

import torch

__all__ = ["define_op", "late"]

#: The ``torchref`` operator namespace; ``define_op`` adds to it.
_LIB = torch.library.Library("torchref", "DEF")

_DISPATCH_KEY = {"cpu": "CPU", "cuda": "CUDA", "mps": "MPS"}


def late(module: str, attr: str) -> Callable:
    """A callable that resolves ``module.attr`` on every call.

    Kernels are registered through this rather than as function objects so that importing
    an operator never imports Triton or compiles a Metal library, and so a kernel stays
    monkeypatchable where it is defined.
    """

    def call(*args, **kwargs):
        return getattr(import_module(module), attr)(*args, **kwargs)

    call.__name__ = f"late_{attr}"
    call.__qualname__ = f"late({module}.{attr})"
    return call


def define_op(
    schema: str,
    impls: Dict[str, Tuple[str, str]],
    fake: Callable,
) -> torch._ops.OpOverloadPacket:
    """Define ``torchref::<name>`` with one kernel per device.

    Parameters
    ----------
    schema : str
        Operator schema without the namespace, e.g.
        ``"bond_nll_fwd(Tensor xyz, Tensor idx, Tensor ref, Tensor sigma) -> Tensor"``.
        Kernels must not mutate their inputs unless the schema says so.
    impls : dict
        Device type (``"cpu"``, ``"cuda"``, ``"mps"``) to ``(module, attr)`` of the kernel,
        resolved per call (see :func:`late`). A device left out raises
        ``NotImplementedError`` from the dispatcher if the operator is called there; the
        backend tables never route such a call to the operator.
    fake : Callable
        Shape/dtype-only implementation, used by ``torch.compile`` and ``opcheck``.

    Returns
    -------
    OpOverloadPacket
        ``torch.ops.torchref.<name>``.
    """
    name = schema.split("(", 1)[0].strip()
    _LIB.define(schema)
    for device, (module, attr) in impls.items():
        _LIB.impl(name, late(module, attr), _DISPATCH_KEY[device])
    torch.library.register_fake(f"torchref::{name}", fake, lib=_LIB)
    return getattr(torch.ops.torchref, name)
