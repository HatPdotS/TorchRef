"""Restraint and X-ray target kernels as PyTorch operators.

A target kernel ``<name>`` is a pair ``torch.ops.torchref.<name>_fwd`` (inputs to a scalar
loss) and ``<name>_bwd`` (upstream scalar gradient to ``d loss / d xyz``), registered for
the devices that have a kernel -- today CUDA (Triton). :func:`target_op` builds the pair
and the ``autograd.Function`` that composes them; the call site decides, through
:func:`~torchref.base.targets._dispatch.use_triton`, whether the operator or the eager
reference runs.

The backwards return gradients with no graph, so every target operator is first order
only: under ``create_graph=True`` the Function raises instead of returning a second
derivative silently missing its term; :func:`torchref.utils.use_portable` selects the
double-differentiable eager path.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from torchref.base.targets._dispatch import second_order_error
from torchref.utils.ops import define_op

_TRITON = "torchref.base.targets.triton"


def _scalar_fake(xyz, *_):
    return xyz.new_empty(())


def _grad_fake(grad, xyz, *_):
    return torch.empty_like(xyz)


def target_op(name: str, function_name: str, static_schema: str, impls: Dict[str, Tuple[str, str]]):
    """Define ``<name>_fwd`` / ``<name>_bwd`` and the ``autograd.Function`` over them.

    Parameters
    ----------
    name : str
        Operator stem; the kernels are ``impls[device] = (module, stem)`` resolved as
        ``module.<stem>_fwd`` and ``module.<stem>_bwd``.
    function_name : str
        Name of the returned Function, as it appears in error messages.
    static_schema : str
        Schema of the arguments after ``xyz``, which get no gradient, e.g.
        ``"Tensor idx, Tensor references, Tensor sigmas"``.
    impls : dict
        Device type to ``(module, stem)``.

    Returns
    -------
    type
        A ``torch.autograd.Function``: ``apply(xyz, *static)`` returns the scalar loss.
    """
    fwd = define_op(
        f"{name}_fwd(Tensor xyz, {static_schema}) -> Tensor",
        {d: (m, f"{stem}_fwd") for d, (m, stem) in impls.items()},
        _scalar_fake,
    )
    bwd = define_op(
        f"{name}_bwd(Tensor grad, Tensor xyz, {static_schema}) -> Tensor",
        {d: (m, f"{stem}_bwd") for d, (m, stem) in impls.items()},
        _grad_fake,
    )

    class Target(torch.autograd.Function):
        @staticmethod
        def forward(ctx, xyz, *static):
            ctx.save_for_backward(xyz, *static)
            return fwd(xyz, *static)

        @staticmethod
        def backward(ctx, grad_out):
            if torch.is_grad_enabled():
                raise second_order_error(f"{function_name}.backward")
            xyz, *static = ctx.saved_tensors
            return (bwd(grad_out.contiguous(), xyz, *static),) + (None,) * len(static)

    Target.__name__ = Target.__qualname__ = function_name
    Target.fwd, Target.bwd = fwd, bwd
    return Target


#: Bond-length Gaussian NLL, ``BondNLL.apply(xyz, idx, references, sigmas)``.
BondNLL = target_op(
    "bond_nll",
    "BondNLL",
    "Tensor idx, Tensor references, Tensor sigmas",
    {"cuda": (f"{_TRITON}.bond", "bond_nll")},
)
