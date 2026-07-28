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
