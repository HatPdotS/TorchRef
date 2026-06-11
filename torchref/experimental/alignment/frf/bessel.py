"""Spherical Bessel functions ``j_u(x)`` for the Bessel-SH expansion.

Phaser source: Phaser uses Miller downward recurrence in its own
implementation (``phaser/lib/jiffy.h``) for stability. We use the same
recurrence here because ``scipy.special.spherical_jn`` is not vectorised
across torch tensors and goes via NumPy round-trip — too slow for the
``(N_radial × N_obs)`` table required by ``bessel_sh_expand``.

Numerical reference for Miller downward recurrence:
    Abramowitz & Stegun §10.1.19 — start at high u, recur downward, then
    rescale by the analytic ``j_0(x) = sin(x)/x``.
"""
from __future__ import annotations

import torch


def spherical_bessel_table(
    x: torch.Tensor,
    u_max: int,
) -> torch.Tensor:
    """Tabulate ``j_u(x)`` for u ∈ [0, u_max] over a 1-D tensor of x.

    Returns
    -------
    j : torch.Tensor, shape (u_max + 1, x.numel())
        ``j[u, i] = j_u(x[i])``. Always float64 internally.
    """
    if x.ndim != 1:
        raise ValueError(f"expected 1-D x, got shape {tuple(x.shape)}")
    if u_max < 0:
        raise ValueError(f"u_max must be >= 0, got {u_max}")

    device = x.device
    x64 = x.to(torch.float64)
    n = x64.numel()
    out = torch.zeros((u_max + 1, n), dtype=torch.float64, device=device)

    # Handle x == 0 separately (j_0(0)=1, j_u(0)=0 for u>=1).
    zero_mask = x64 == 0
    nz_mask = ~zero_mask
    out[0, zero_mask] = 1.0
    if not nz_mask.any():
        return out

    xnz = x64[nz_mask]
    nnz = xnz.numel()

    # Direct formulas for u = 0, 1 — accurate for all x > 0.
    j0 = torch.sin(xnz) / xnz
    j1 = (torch.sin(xnz) - xnz * torch.cos(xnz)) / xnz**2

    if u_max == 0:
        out[0, nz_mask] = j0
        return out
    if u_max == 1:
        out[0, nz_mask] = j0
        out[1, nz_mask] = j1
        return out

    # Forward recurrence is unstable when u >> x; switch to Miller downward
    # recurrence for those entries. Cutoff u_fwd_safe = ceil(x) is generous;
    # Phaser uses similar logic in jiffy.h.
    u_fwd_safe = torch.clamp(torch.ceil(xnz).to(torch.int64), min=2)
    # Allocate per-x downward recurrence with a generous starting index
    # (u_max + a few extra terms) — Miller convention.
    u_start = u_max + 15
    f_curr = torch.zeros(nnz, dtype=torch.float64, device=device)
    f_next = torch.ones(nnz, dtype=torch.float64, device=device)
    table = torch.zeros((u_max + 1, nnz), dtype=torch.float64, device=device)

    # Downward: j_{u-1}(x) = (2u+1)/x · j_u(x) - j_{u+1}(x)
    for u in range(u_start, -1, -1):
        f_prev = (2 * u + 1) / xnz * f_next - f_curr
        if u <= u_max:
            table[u] = f_prev
        f_curr = f_next
        f_next = f_prev

    # Rescale so table[0] matches analytic j_0(x) = sin(x)/x.
    scale = j0 / table[0]
    table = table * scale.unsqueeze(0)

    # Override with the forward direct values where they are stable
    # (small u, small x). Forward recurrence: j_{u+1} = (2u+1)/x · j_u - j_{u-1}.
    fwd = torch.zeros((u_max + 1, nnz), dtype=torch.float64, device=device)
    fwd[0] = j0
    fwd[1] = j1
    for u in range(1, u_max):
        fwd[u + 1] = (2 * u + 1) / xnz * fwd[u] - fwd[u - 1]
    # Per-x, use forward where u <= u_fwd_safe; otherwise Miller-rescaled.
    u_idx = torch.arange(u_max + 1, device=device).unsqueeze(1)  # (u_max+1, 1)
    use_fwd = u_idx <= u_fwd_safe.unsqueeze(0)                    # (u_max+1, nnz)
    table_combined = torch.where(use_fwd, fwd, table)

    out[:, nz_mask] = table_combined
    return out
