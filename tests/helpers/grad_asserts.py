"""Gradient-agreement assertions shared across kernel comparison tests.

Comparing two kernels' gradients needs **two** metrics, not one. Cosine
similarity alone passes a systematic scale error -- a kernel that returns
``2 * grad`` is perfectly parallel to the reference -- so every assertion here
checks direction *and* magnitude.

Extracted from ``tests/unit/test_gradient_correctness.py`` so the accelerator
kernel tests (``test_variable_radius_gpu.py`` / ``test_variable_radius_mps.py``)
can reuse them instead of hand-rolling a bare cosine check.
"""

from __future__ import annotations

import torch

__all__ = [
    "cosine_similarity",
    "gradnorm_ratio",
    "assert_grads_agree",
    "got_ref_pairs",
    "rel_error",
    "central_fd_grad",
    "hvp",
    "hvp_central_fd",
]


def _flat_real(t: torch.Tensor) -> torch.Tensor:
    """Flatten to a real 1-D CPU float64 vector, splitting complex into (re, im).

    The CPU hop is required, not cosmetic: these helpers accumulate in float64
    and MPS has no float64, so comparing accelerator gradients in place would
    fail in the backend instead of in the assertion.
    """
    t = t.detach().cpu()
    if t.is_complex():
        return torch.view_as_real(t).reshape(-1).double()
    return t.reshape(-1).double()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two gradients (direction agreement)."""
    return torch.nn.functional.cosine_similarity(
        _flat_real(a), _flat_real(b), dim=0
    ).item()


def gradnorm_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    """Ratio of gradient norms ``||a|| / ||b||`` (magnitude agreement)."""
    nb = _flat_real(b).norm()
    return (_flat_real(a).norm() / nb).item()


def got_ref_pairs(got, ref):
    """Yield ``(name, got_grad, ref_grad)`` triples for a dict or sequence."""
    if isinstance(got, dict):
        for k in got:
            yield str(k), got[k], ref[k]
    else:
        for i, (g, r) in enumerate(zip(got, ref)):
            yield str(i), g, r


def assert_grads_agree(got, ref, *, min_cos=0.9999, ratio_tol=1e-3, ctx=""):
    """Assert per-leaf cosine ≈ 1 and gradnorm ratio ≈ 1 for matched grads."""
    for name, g, r in got_ref_pairs(got, ref):
        cos = cosine_similarity(g, r)
        ratio = gradnorm_ratio(g, r)
        assert cos >= min_cos, f"{ctx}{name}: cosine {cos:.8f} < {min_cos}"
        assert (
            abs(ratio - 1.0) <= ratio_tol
        ), f"{ctx}{name}: gradnorm ratio {ratio:.8f} off 1 by > {ratio_tol}"


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error ``||a - b|| / ||b||`` in float64, device-agnostic."""
    fa, fb = _flat_real(a), _flat_real(b)
    return ((fa - fb).norm() / fb.norm()).item()


# ---------------------------------------------------------------------------
# Finite differences and Hessian-vector products
# ---------------------------------------------------------------------------
# Promoted here from three divergent private copies -- ``test_gradient_correctness.py``
# (``_central_fd_grad``), ``test_triton_vs_eager_targets.py`` (``_hvp_vs_fd``) and
# ``test_kernel_fixes.py`` (inline, twice). They disagreed on eps and on whether to
# detach, which made their tolerances incomparable.
#
# Caveat on where these are valid: a central difference is only a sound reference for a
# smooth function. The electron-density map route is hard-truncated at a per-atom radius
# and the cull surface moves with the atom, so a voxel crossing it contributes a fixed
# jump to the difference. Use finite differences against direct summation, which is
# analytic, and use direct summation as the reference for the map route.
def central_fd_grad(loss_fn, x0: torch.Tensor, eps: float) -> torch.Tensor:
    """Central-difference gradient of a scalar ``loss_fn(x)`` at ``x0``.

    Costs ``2 * x0.numel()`` forward evaluations, so keep ``x0`` small.
    """
    flat = x0.detach().reshape(-1).clone()
    out = torch.zeros_like(flat)
    for i in range(flat.numel()):
        plus, minus = flat.clone(), flat.clone()
        plus[i] += eps
        minus[i] -= eps
        with torch.no_grad():
            hi = loss_fn(plus.view_as(x0))
            lo = loss_fn(minus.view_as(x0))
        out[i] = (hi - lo) / (2.0 * eps)
    return out.view_as(x0)


def hvp(loss_fn, x0: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Hessian-vector product by double backward: ``d/dx (grad(loss) . v)``."""
    x = x0.detach().clone().requires_grad_(True)
    (g1,) = torch.autograd.grad(loss_fn(x), x, create_graph=True)
    (out,) = torch.autograd.grad((g1 * v).sum(), x)
    return out.detach()


def hvp_central_fd(loss_fn, x0: torch.Tensor, v: torch.Tensor, eps: float):
    """Hessian-vector product by central-differencing the *gradient* along ``v``.

    Two gradient evaluations rather than ``2 * numel`` forwards, so this stays cheap even
    on realistic scenes.
    """

    def grad_at(xv):
        x = xv.detach().clone().requires_grad_(True)
        (g,) = torch.autograd.grad(loss_fn(x), x)
        return g.detach()

    return (grad_at(x0 + eps * v) - grad_at(x0 - eps * v)) / (2.0 * eps)
