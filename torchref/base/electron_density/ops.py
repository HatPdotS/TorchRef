"""The fused density splat as PyTorch operators, with one autograd rule for every device.

``torch.ops.torchref.splat_{iso,aniso}_{fwd,bwd}`` each have a CPU (``torchref-kernels``),
CUDA (Triton) and MPS (Metal) kernel; PyTorch's dispatcher picks one from the inputs'
device. All kernels share one contract:

* ``*_fwd(density_map, xyz, adp_or_u, occ, A, B, r2cut, inv_frac, frac)`` returns a new
  map ``density_map + splat``; ``density_map`` is not modified.
* ``*_bwd(grad, xyz, adp_or_u, occ, A, B, r2cut, inv_frac, frac)`` returns
  ``(d xyz, d adp_or_u, d occ)`` of ``sum(grad * splat)``.

Every tensor must share the device; the dtype is the kernel row's to check (CUDA and Metal
are float32 only), which is why the calls are made only after
:data:`~torchref.base.electron_density._backends.DENSITY_BACKENDS` has selected a row.

:class:`SplatIso` / :class:`SplatAniso` own the autograd. Second order differs by device:
on CPU a ``create_graph=True`` backward is re-derived through the portable splat
(``sphere_splat._double_backward_vjp``); the CUDA and Metal backwards raise instead of
returning a gradient with no graph.
"""

from __future__ import annotations

from importlib import import_module

import torch

from torchref.base.targets._dispatch import second_order_error
from torchref.utils.ops import define_op

_CPU = "torchref.base.electron_density.kernels.cpu.sphere_splat"
_CUDA = "torchref.base.electron_density.kernels.cuda.variable_radius"
_MPS = "torchref.base.electron_density.kernels.mps.variable_radius"
_PORTABLE = "torchref.base.electron_density.kernels.cpu.variable_radius"

_ARGS = (
    "Tensor xyz, Tensor {adp}, Tensor occ, Tensor A, Tensor B, Tensor r2cut, "
    "Tensor inv_frac, Tensor frac"
)


def _fake_fwd(density_map, *_):
    return torch.empty_like(density_map)


def _fake_bwd(grad, xyz, adp, occ, *_):
    return torch.empty_like(xyz), torch.empty_like(adp), torch.empty_like(occ)


def _impls(name):
    return {"cpu": (_CPU, name), "cuda": (_CUDA, name), "mps": (_MPS, name)}


splat_iso_fwd = define_op(
    f"splat_iso_fwd(Tensor density_map, {_ARGS.format(adp='adp')}) -> Tensor",
    _impls("iso_fwd"),
    _fake_fwd,
)
splat_iso_bwd = define_op(
    f"splat_iso_bwd(Tensor grad, {_ARGS.format(adp='adp')}) -> (Tensor, Tensor, Tensor)",
    _impls("iso_bwd"),
    _fake_bwd,
)
splat_aniso_fwd = define_op(
    f"splat_aniso_fwd(Tensor density_map, {_ARGS.format(adp='u')}) -> Tensor",
    _impls("aniso_fwd"),
    _fake_fwd,
)
splat_aniso_bwd = define_op(
    f"splat_aniso_bwd(Tensor grad, {_ARGS.format(adp='u')}) -> (Tensor, Tensor, Tensor)",
    _impls("aniso_bwd"),
    _fake_bwd,
)


def _splat_function(name, fwd, bwd, plain_attr):
    class Splat(torch.autograd.Function):
        @staticmethod
        def forward(ctx, density_map, xyz, adp, occ, A, B, r2cut, inv_frac, frac):
            out = fwd(
                density_map.contiguous(), xyz.contiguous(), adp.contiguous(),
                occ.contiguous(), A.contiguous(), B.contiguous(), r2cut.contiguous(),
                inv_frac.contiguous(), frac.contiguous(),
            )
            ctx.save_for_backward(xyz, adp, occ, A, B, r2cut, inv_frac, frac)
            ctx.grid_shape = tuple(density_map.shape)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            xyz, adp, occ, A, B, r2cut, inv_frac, frac = ctx.saved_tensors
            # Autograd runs backward with grad mode on exactly when create_graph=True.
            if torch.is_grad_enabled():
                if grad_out.device.type != "cpu":
                    raise second_order_error(f"{name}.backward")
                return import_module(_CPU)._double_backward_vjp(
                    getattr(import_module(_PORTABLE), plain_attr), ctx, grad_out,
                    (xyz, adp, occ), (A, B, inv_frac, frac), r2cut,
                )
            g_xyz, g_adp, g_occ = bwd(
                grad_out.contiguous(), xyz.contiguous(), adp.contiguous(),
                occ.contiguous(), A.contiguous(), B.contiguous(), r2cut.contiguous(),
                inv_frac.contiguous(), frac.contiguous(),
            )
            # out = density_map + splat, so the gradient wrt density_map is the identity.
            return (grad_out, g_xyz, g_adp, g_occ, None, None, None, None, None)

    Splat.__name__ = Splat.__qualname__ = name
    return Splat


#: Isotropic splat with autograd: ``SplatIso.apply(density_map, xyz, adp, occ, A, B,
#: r2cut, inv_frac, frac)`` returns ``density_map + splat``.
SplatIso = _splat_function(
    "SplatIso", splat_iso_fwd, splat_iso_bwd, "add_isotropic_plain_var"
)
#: Anisotropic splat with autograd; as :data:`SplatIso` with ``u`` ``(n, 6)`` for ``adp``.
SplatAniso = _splat_function(
    "SplatAniso", splat_aniso_fwd, splat_aniso_bwd, "add_anisotropic_plain_var"
)
