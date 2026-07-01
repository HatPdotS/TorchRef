"""Separable Gaussian box-splat for isotropic atoms (CPU / MPS shared core).

Factorizes exp(-alpha * r^T G r) into 1D Gaussians per axis with 2D cross-term
corrections for non-orthogonal cells, keeping peak memory low. ``_separable_density``
is the shared core reused by the variable-radius CPU splat
(``cpu/variable_radius.py::add_isotropic_cpu_separable_var``).
"""

import math

import torch

from torchref.config import dtypes

# Peak-element budget for the triclinic general path's batched (C, N_comp, n^3)
# exp intermediate. Atoms are sub-batched so this is the memory ceiling
# regardless of the caller's chunk size. 16M floats ~= 64 MB.
_GENERAL_BUDGET = 16_000_000


def _separable_density(
    d_frac: torch.Tensor,
    alpha: torch.Tensor,
    A_norm: torch.Tensor,
    G: torch.Tensor,
    has_ab: bool,
    has_ac: bool,
    has_bc: bool,
) -> torch.Tensor:
    """Separable Gaussian density evaluation.

    Factorizes exp(-alpha * r^T G r) into 1D Gaussians per axis with 2D
    cross-term corrections for non-orthogonal cells. Batches all corrections
    across the 5 ITC92 components and uses einsum where possible.

    For non-orthogonal cells, cross-term exponents are combined with the
    relevant diagonal exponents before taking exp() to avoid float32 overflow
    (exp(-big) * exp(+big) = 0 * inf = NaN).  Each combined 2D block
    exponent corresponds to a principal sub-matrix of G (positive definite),
    guaranteeing the exponent is always <= 0 and exp() is in (0, 1].

    Dispatch by crystal system for optimal performance:
    - Orthogonal (no cross terms): separable 1D products + einsum
    - Hexagonal  (ab only):  combined ab exponent + einsum with e_c
    - Monoclinic (ac only):  combined ac exponent + einsum with e_b
    - General    (bc, or multiple cross terms): full 3D exponent per component

    Parameters
    ----------
    d_frac : (C, 3, n_axis) — fractional distances per axis, PBC-wrapped.
    alpha : (C, N_comp) — pi^2 / B_total.
    A_norm : (C, N_comp) — weighted amplitudes.
    G : (3, 3) — metric tensor frac_matrix.T @ frac_matrix.
    has_ab : bool — whether G[0,1] cross-term is significant.
    has_ac : bool — whether G[0,2] cross-term is significant.
    has_bc : bool — whether G[1,2] cross-term is significant.

    Returns
    -------
    (C, n_axis, n_axis, n_axis) density cube.
    """
    # --- Convert fractional → Cartesian per-axis ---
    cell_lengths = torch.sqrt(torch.diagonal(G))
    d_cart = d_frac * cell_lengths[None, :, None]  # (C, 3, n)

    # --- 1D exponents (always <= 0) ---
    da2 = d_cart[:, 0, :] ** 2
    db2 = d_cart[:, 1, :] ** 2
    dc2 = d_cart[:, 2, :] ** 2

    log_a = -alpha.unsqueeze(2) * da2.unsqueeze(1)  # (C, Nc, n)
    log_b = -alpha.unsqueeze(2) * db2.unsqueeze(1)
    log_c = -alpha.unsqueeze(2) * dc2.unsqueeze(1)

    if not (has_ab or has_ac or has_bc):
        # ---- Orthogonal cells: pure separable, all exp() args <= 0 ----
        e_a = torch.exp(log_a)
        e_b = torch.exp(log_b)
        e_c = torch.exp(log_c)
        e_ab = e_a.unsqueeze(3) * e_b.unsqueeze(2)
        return torch.einsum("cg,cgij,cgk->cijk", A_norm, e_ab, e_c)

    # --- Cross-term coefficients ---
    cos_gamma = G[0, 1] / (cell_lengths[0] * cell_lengths[1])
    cos_beta = G[0, 2] / (cell_lengths[0] * cell_lengths[2])
    cos_alpha = G[1, 2] / (cell_lengths[1] * cell_lengths[2])

    da = d_cart[:, 0, :]
    db = d_cart[:, 1, :]
    dc = d_cart[:, 2, :]
    alpha_4d = alpha[:, :, None, None]  # (C, Nc, 1, 1)

    if has_ab and not has_ac and not has_bc:
        # ---- Hexagonal / trigonal: only ab cross-term ----
        # Combined 2D exponent: -alpha*(da2 + db2 + 2*cos_gamma*da*db)
        # = -alpha * d_ab^T G_ab d_ab <= 0 (G_ab positive definite)
        prod_ab = da.unsqueeze(2) * db.unsqueeze(1)
        log_ab = (
            log_a[:, :, :, None]
            + log_b[:, :, None, :]
            + (-2.0 * alpha_4d * cos_gamma * prod_ab[:, None, :, :])
        )
        slice_ab = torch.exp(log_ab)  # (C, Nc, n, n), all in (0, 1]
        e_c = torch.exp(log_c)
        return torch.einsum("cg,cgij,cgk->cijk", A_norm, slice_ab, e_c)

    if has_ac and not has_ab and not has_bc:
        # ---- Monoclinic (beta != 90): only ac cross-term ----
        # Combined 2D exponent: -alpha*(da2 + dc2 + 2*cos_beta*da*dc)
        # = -alpha * d_ac^T G_ac d_ac <= 0 (G_ac positive definite)
        prod_ac = da.unsqueeze(2) * dc.unsqueeze(1)
        log_ac = (
            log_a[:, :, :, None]
            + log_c[:, :, None, :]
            + (-2.0 * alpha_4d * cos_beta * prod_ac[:, None, :, :])
        )
        e_ac = torch.exp(log_ac)  # (C, Nc, n_a, n_c), all in (0, 1]
        e_b = torch.exp(log_b)
        return torch.einsum("cg,cgj,cgik->cijk", A_norm, e_b, e_ac)

    # ---- General path (triclinic, or multiple cross-terms) ----
    # The per-component exponent is -alpha[:, g] * Q, where the quadratic form
    #   Q = da^2 + db^2 + dc^2 + 2*cos_gamma*da*db + 2*cos_beta*da*dc + 2*cos_alpha*db*dc
    # is the cell geometry (r^T G r in cell-length-normalized units) and is
    # INDEPENDENT of the ITC92 component g (alpha[:, g] factors out entirely).
    # So build Q ONCE (it is positive semidefinite -> -alpha*Q <= 0, no overflow),
    # then evaluate exp(-alpha[:, g] * Q) for all components in a SINGLE batched
    # exp + einsum rather than a per-component loop with repeated geometry rebuilds.
    # The batched (C, N_comp, n, n, n) intermediate is 5x the cube, so atoms are
    # processed in sub-batches sized to keep peak memory under _GENERAL_BUDGET
    # floats (the result is identical to the loop, just fewer/larger exp calls --
    # ~1.5-2x faster on the triclinic path which is dominated by the exp).
    C = d_frac.shape[0]
    n = d_frac.shape[2]
    Nc = alpha.shape[1]
    sub = max(1, _GENERAL_BUDGET // max(1, Nc * n * n * n))
    parts = []
    for s in range(0, C, sub):
        e = min(s + sub, C)
        Q = (
            da2[s:e, :, None, None]
            + db2[s:e, None, :, None]
            + dc2[s:e, None, None, :]
        )  # (c, n, n, n) — broadcasting materializes a full contiguous cube
        if has_ab:
            Q = Q + (2.0 * cos_gamma * da[s:e].unsqueeze(2) * db[s:e].unsqueeze(1)).unsqueeze(3)
        if has_ac:
            Q = Q + (2.0 * cos_beta * da[s:e].unsqueeze(2) * dc[s:e].unsqueeze(1)).unsqueeze(2)
        if has_bc:
            Q = Q + (2.0 * cos_alpha * db[s:e].unsqueeze(2) * dc[s:e].unsqueeze(1)).unsqueeze(1)
        # (c, N_comp, n, n, n): one exp over all components, then weighted sum.
        comp = torch.exp(-alpha[s:e, :, None, None, None] * Q.unsqueeze(1))
        parts.append(torch.einsum("cg,cgijk->cijk", A_norm[s:e], comp))

    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
