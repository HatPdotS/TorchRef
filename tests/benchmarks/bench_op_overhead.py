"""Per-call overhead of the ``torch.ops.torchref`` operator path versus the alternatives.

Times one forward + backward of a small kernel, where Python-side dispatch rather than
the GPU dominates, for each way of wiring the same kernels into autograd:

* ``eager``       -- the plain-PyTorch reference (no custom kernel)
* ``function``    -- an ``autograd.Function`` calling the kernels directly
* ``library_op``  -- what TorchRef ships: ``autograd.Function`` over ``torch.library``
                     operators (``torch.ops.torchref.*``)
* ``custom_op``   -- ``torch.library.custom_op`` with ``register_autograd``

for the bond target (CUDA only: its kernel is Triton) and the isotropic density splat
(CUDA, MPS or CPU). The ``library_op - function`` difference is the cost of going through
the dispatcher; multiply by ~300-500 target calls per L-BFGS step for the per-step cost.

    python tests/benchmarks/bench_op_overhead.py [--device cuda] [--iters 2000]
"""

from __future__ import annotations

import argparse
import time

import torch

from torchref.base.electron_density import ops as ed_ops
from torchref.base.electron_density.kernels.cpu.variable_radius import (
    add_isotropic_plain_var,
)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _time(fn, device, iters, warmup=50):
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters * 1e6


def _report(title, rows):
    print(f"\n{title}")
    base = rows.get("function")
    for name, us in rows.items():
        extra = "" if base is None or name == "function" else f"   ({us - base:+.1f} us vs function)"
        print(f"  {name:<11} {us:9.1f} us / fwd+bwd{extra}")


def bench_bond(device, iters, n_atoms=1600, n_bonds=1650):
    from torchref.base.targets.bond import _bond_math_eager
    from torchref.base.targets.ops import BondNLL
    from torchref.base.targets.triton import bond as tb

    g = torch.Generator().manual_seed(0)
    xyz0 = (3.0 * torch.randn(n_atoms, 3, generator=g)).to(device)
    idx = torch.randint(0, n_atoms, (n_bonds, 2), generator=g).to(device)
    ref = (1.2 + 0.5 * torch.rand(n_bonds, generator=g)).to(device)
    sig = (0.02 + 0.03 * torch.rand(n_bonds, generator=g)).to(device)

    class Direct(torch.autograd.Function):
        @staticmethod
        def forward(ctx, xyz, idx, ref, sig):
            ctx.save_for_backward(xyz, idx, ref, sig)
            return tb.bond_nll_fwd(xyz, idx, ref, sig)

        @staticmethod
        def backward(ctx, grad):
            if torch.is_grad_enabled():
                raise RuntimeError("first order only")
            xyz, idx, ref, sig = ctx.saved_tensors
            return tb.bond_nll_bwd(grad.contiguous(), xyz, idx, ref, sig), None, None, None

    @torch.library.custom_op("torchref_bench::bond_fwd", mutates_args=())
    def c_fwd(xyz: torch.Tensor, idx: torch.Tensor, ref: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        return tb.bond_nll_fwd(xyz, idx, ref, sig)

    @c_fwd.register_fake
    def _(xyz, idx, ref, sig):
        return xyz.new_empty(())

    @torch.library.custom_op("torchref_bench::bond_bwd", mutates_args=())
    def c_bwd(grad: torch.Tensor, xyz: torch.Tensor, idx: torch.Tensor, ref: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        return tb.bond_nll_bwd(grad.contiguous(), xyz, idx, ref, sig)

    @c_bwd.register_fake
    def _(grad, xyz, idx, ref, sig):
        return torch.empty_like(xyz)

    def setup(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    def backward(ctx, grad):
        xyz, idx, ref, sig = ctx.saved_tensors
        return c_bwd(grad, xyz, idx, ref, sig), None, None, None

    c_fwd.register_autograd(backward, setup_context=setup)

    def step(f):
        def run():
            x = xyz0.detach().requires_grad_(True)
            f(x, idx, ref, sig).backward()
        return run

    rows = {
        "eager": _time(step(_bond_math_eager), device, iters),
        "function": _time(step(Direct.apply), device, iters),
        "library_op": _time(step(BondNLL.apply), device, iters),
        "custom_op": _time(step(c_fwd), device, iters),
    }
    _report(f"bond target, {n_bonds} bonds, {device}", rows)


def bench_splat(device, iters, n_atoms=1500, grid=(72, 128, 66)):
    from importlib import import_module

    mod = import_module(
        {"cuda": "torchref.base.electron_density.kernels.cuda.variable_radius",
         "mps": "torchref.base.electron_density.kernels.mps.variable_radius",
         "cpu": "torchref.base.electron_density.kernels.cpu.sphere_splat"}[device.type]
    )
    g = torch.Generator().manual_seed(1)
    frac = torch.tensor([[35.0, 0.0, -9.1], [0.0, 64.0, 0.0], [0.0, 0.0, 31.7]])
    inv = frac.inverse()
    xyz0 = torch.rand(n_atoms, 3, generator=g) @ frac.T
    t = [x.to(device) for x in (
        xyz0, 15 + 25 * torch.rand(n_atoms, generator=g),
        0.5 + 0.5 * torch.rand(n_atoms, generator=g),
        0.5 + 2 * torch.rand(n_atoms, 5, generator=g),
        0.3 + 30 * torch.rand(n_atoms, 5, generator=g),
        (2.0 + 2.0 * torch.rand(n_atoms, generator=g)) ** 2, inv, frac,
    )]
    xyz, adp, occ, A, B, r2, im, fm = t
    dm = torch.zeros(grid, device=device)

    class Direct(torch.autograd.Function):
        @staticmethod
        def forward(ctx, dm, xyz, adp, occ, A, B, r2, im, fm):
            ctx.save_for_backward(xyz, adp, occ, A, B, r2, im, fm)
            return mod.iso_fwd(dm, xyz, adp, occ, A, B, r2, im, fm)

        @staticmethod
        def backward(ctx, grad):
            s = ctx.saved_tensors
            gx, gb, go = mod.iso_bwd(grad.contiguous(), *s)
            return (grad, gx, gb, go) + (None,) * 5

    def step(f, plain=False):
        def run():
            x = xyz.detach().requires_grad_(True)
            if plain:
                out = f(dm, x, adp, occ, A, B, im, fm, r2.sqrt())
            else:
                out = f(dm, x, adp, occ, A, B, r2, im, fm)
            out.sum().backward()
        return run

    rows = {
        "eager": _time(step(add_isotropic_plain_var, plain=True), device, 3, warmup=1),
        "function": _time(step(Direct.apply), device, iters // 10),
        "library_op": _time(step(ed_ops.SplatIso.apply), device, iters // 10),
    }
    _report(f"isotropic splat, {n_atoms} atoms, grid {grid}, {device}", rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--device", default=default)
    ap.add_argument("--iters", type=int, default=2000)
    opts = ap.parse_args()
    device = torch.device(opts.device)
    print(f"torch {torch.__version__}, device {device}")
    if device.type == "cuda":
        bench_bond(device, opts.iters)
    else:
        print("\n(bond target skipped: its kernel is Triton, CUDA only)")
    bench_splat(device, opts.iters)


if __name__ == "__main__":
    main()
