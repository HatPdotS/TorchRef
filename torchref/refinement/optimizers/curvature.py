"""Hessian-diagonal estimation for seeding :class:`SeededLBFGS`.

Provides a stochastic (Hutchinson) estimate of the loss Hessian's diagonal and the
inverse-curvature preconditioner ``1/(|diag|+lambda)`` derived from it, via *analytic*
double-backward Hessian-vector products::

    (g,)   = torch.autograd.grad(loss, params, create_graph=True)   # once
    (Hv,)  = torch.autograd.grad((g * v).sum(), params)             # per probe

That is only correct on the pure-torch electron-density path, so everything runs under
``use_portable()``. Finite-difference HVPs are deliberately NOT used: the production
Triton/CUDA kernels emit float32 gradients, so central differences catastrophically
cancel, whereas the analytic HVP does not and float32 is adequate for a preconditioner.

The returned diagonal is flat and aligned to ``cat(p.reshape(-1) for p in params)`` --
the ``_gather_flat_grad`` layout -- so it feeds straight into
:meth:`SeededLBFGS.set_init_hess_diag`.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

from torchref.utils import use_portable


def _sample_probe(
    numel: int,
    probe: str,
    generator: Optional[torch.Generator],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Draw one Hutchinson probe vector of length ``numel``."""
    if probe == "rademacher":
        r = torch.randint(
            0, 2, (numel,), generator=generator, device=device, dtype=torch.int64  # dtype-ok: randint {0,1} bernoulli draw, immediately cast to float dtype; width irrelevant
        )
        return r.to(dtype).mul_(2.0).sub_(1.0)  # {0,1} -> {-1,+1}
    if probe == "gaussian":
        return torch.randn(numel, generator=generator, device=device, dtype=dtype)
    raise ValueError(f"unknown probe type {probe!r}")


def _flatten(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def hessian_diagonal(
    aggregate: Callable[[], torch.Tensor],
    params: Sequence[torch.nn.Parameter],
    *,
    n_probes: int = 16,
    generator: Optional[torch.Generator] = None,
    probe: str = "rademacher",
    basis_max_numel: int = 4096,
) -> torch.Tensor:
    """Estimate ``diag(H)`` of ``aggregate()`` w.r.t. ``params``, flat and detached.

    Runs the double-backward under :func:`torchref.utils.use_portable`, the only kernel path
    that composes correctly under ``create_graph=True``.

    Parameters
    ----------
    aggregate : callable
        Zero-argument callable returning the scalar loss (e.g. ``LossState.aggregate``).
        Called once, with the graph retained so every probe reuses the first-order backward.
    params : sequence of nn.Parameter
        The leaves to differentiate w.r.t., in the SAME order the optimizer gets them, so the
        flat result matches ``_gather_flat_grad``.
    n_probes : int
        Number of Hutchinson probes (ignored when ``probe="basis"``).
    generator : torch.Generator, optional
        For deterministic probes.
    probe : {"rademacher", "gaussian", "basis"}
        ``"basis"`` computes the EXACT diagonal in ``numel`` HVPs -- verification and small
        problems only, capped by ``basis_max_numel``.

    Returns
    -------
    torch.Tensor
        Flat detached estimate of ``diag(H)``, aligned to
        ``cat(p.reshape(-1) for p in params)``.

    Raises
    ------
    ValueError
        If any parameter is complex.
    """
    params = list(params)
    if any(torch.is_complex(p) for p in params):
        raise ValueError("hessian_diagonal does not support complex parameters")
    # Differentiate only through leaves that require grad; frozen leaves (e.g.
    # occupancy held fixed, or aniso U on an isotropic model) get zero curvature
    # -> a bounded 1/lam seed. Harmless: their gradient is zero, so the optimizer
    # never moves them regardless of the seed value. The flat layout still spans
    # the FULL params list so it matches the optimizer's _gather_flat_grad order.
    active = [p for p in params if p.requires_grad]
    if not active:
        raise ValueError("no params require grad; nothing to differentiate")

    def _assemble_full(active_parts):
        """Map a per-active-param list back to a full-length flat tensor in
        ``params`` order, filling detached zeros for frozen params."""
        it = iter(active_parts)
        pieces = []
        for p in params:
            if p.requires_grad:
                pieces.append(next(it).reshape(-1))
            else:
                pieces.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        return torch.cat(pieces)

    with use_portable():
        loss = aggregate()
        # allow_unused: an active leaf may still not enter the loss.
        grads = torch.autograd.grad(
            loss, active, create_graph=True, allow_unused=True
        )
        grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, active)
        ]
        g_flat = _assemble_full(grads)  # graph-attached for active slices
        numel = g_flat.numel()
        device, dtype = g_flat.device, g_flat.dtype

        def _hvp(v: torch.Tensor, retain: bool) -> torch.Tensor:
            parts = torch.autograd.grad(
                (g_flat * v).sum(),
                active,
                retain_graph=retain,
                create_graph=False,
                allow_unused=True,
            )
            parts = [
                pt if pt is not None else torch.zeros_like(p)
                for pt, p in zip(parts, active)
            ]
            return _assemble_full(parts)

        diag = torch.zeros(numel, device=device, dtype=dtype)

        if probe == "basis":
            if numel > basis_max_numel:
                raise ValueError(
                    f"probe='basis' needs {numel} HVPs; exceeds basis_max_numel "
                    f"({basis_max_numel}). Use rademacher/gaussian for large problems."
                )
            for i in range(numel):
                v = torch.zeros(numel, device=device, dtype=dtype)
                v[i] = 1.0
                diag[i] = _hvp(v, retain=(i < numel - 1))[i]
            return _sanitize_diag(diag.detach())

        for k in range(n_probes):
            v = _sample_probe(numel, probe, generator, device, dtype)
            diag.add_(v * _hvp(v, retain=(k < n_probes - 1)))
        diag.div_(n_probes)
        return _sanitize_diag(diag.detach())


def _sanitize_diag(diag: torch.Tensor) -> torch.Tensor:
    """Replace sporadic non-finite diagonal entries with the max finite |diag|
    (→ stiffest, i.e. smallest, safest seeded step for those directions). If NO
    entry is finite (the whole HVP blew up), leave it non-finite so the caller
    can detect it and skip seeding this cycle.
    """
    finite = torch.isfinite(diag)
    if finite.all() or not finite.any():
        return diag
    fill = diag[finite].abs().max()
    return torch.where(finite, diag, fill)


def preconditioner_from_diagonal(
    diag: torch.Tensor, lam: float = 1e-2, group_sizes=None
) -> torch.Tensor:
    """Turn a Hessian diagonal into a positive inverse-curvature preconditioner.

    ``precond = 1 / max(|diag|, lam * max|diag|)``. The absolute value makes negative
    (indefinite) curvature still give a descent direction, and the **relative** floor caps
    the preconditioner ratio at ``1/lam``. That relative floor is what keeps the seeded
    first step safe far from the minimum: with an absolute floor a near-flat direction
    gets an unbounded preconditioner and the Newton-scaled ``t=1`` step blows it to
    non-finite geometry.

    Parameters
    ----------
    group_sizes : sequence of int, optional
        Contiguous group lengths summing to ``diag.numel()`` (e.g. one per
        ``nn.Parameter``). Given these, the relative floor is computed **per group** rather
        than globally, which matters when groups differ hugely in curvature -- the few scaler
        params each touch all reflections, so a global floor would crush the body group to a
        near-zero step.
    """
    if group_sizes is None:
        return _precond_slice(diag, lam)
    if int(sum(group_sizes)) != diag.numel():
        raise ValueError(
            f"group_sizes sum {int(sum(group_sizes))} != diag numel {diag.numel()}"
        )
    out = torch.empty_like(diag)
    off = 0
    for n in group_sizes:
        out[off:off + n] = _precond_slice(diag[off:off + n], lam)
        off += n
    return out


def _precond_slice(diag: torch.Tensor, lam: float) -> torch.Tensor:
    if diag.numel() == 0:  # empty group (e.g. absent aniso U / occupancy leaf)
        return diag
    a = diag.abs()
    amax = a.max()
    # abs backstop for the degenerate all-flat case (amax == 0)
    floor = torch.clamp(lam * amax, min=torch.finfo(a.dtype).eps)
    return 1.0 / torch.clamp(a, min=floor)


def hessian_diagonal_preconditioner(
    aggregate: Callable[[], torch.Tensor],
    params: Sequence[torch.nn.Parameter],
    *,
    n_probes: int = 16,
    lam: float = 1e-2,
    per_group: bool = False,
    generator: Optional[torch.Generator] = None,
    probe: str = "rademacher",
) -> torch.Tensor:
    """:func:`hessian_diagonal` then :func:`preconditioner_from_diagonal`.

    ``per_group=True`` floors each ``params`` entry by its own ``max|diag|``; use it when
    the params span groups of very different curvature magnitude (e.g. body + scaler).
    """
    params = list(params)
    diag = hessian_diagonal(
        aggregate,
        params,
        n_probes=n_probes,
        generator=generator,
        probe=probe,
    )
    group_sizes = [p.numel() for p in params] if per_group else None
    return preconditioner_from_diagonal(diag, lam=lam, group_sizes=group_sizes)
