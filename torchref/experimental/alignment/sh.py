"""Leaf mathematics the rotation function needs: Legendre seeds, shells, anisotropy.

Three unrelated things share this module because they share one consumer.

* **Legendre recurrence coefficients** and their seed, for the fully-normalised
  associated Legendre ``bar_P_l^m(cos theta)``, which the Bessel-SH expansion in
  :mod:`~torchref.experimental.alignment.frf.data_mr` and its compiled kernels
  build on. Normalised so ``(2l)!`` is never formed explicitly.
* **Equal-count resolution shells** (:func:`equal_count_shell_edges`,
  :func:`assign_shells`). Assigned once and passed down: two consumers deriving
  their own edges from the same ``|s|`` disagree about the reflections sitting on
  a boundary.
* **Overall anisotropy** -- fit in intensity space, projected onto the point
  group, applied to amplitudes with the half exponent. The projection is
  load-bearing: an unconstrained six-component fit can return a tensor the
  lattice forbids.

This module used to also carry a full spherical-harmonic expansion
(``evaluate_ylm``, ``sh_expand_ball``). Nothing called it -- the FRF's own
expansion superseded it -- so it went. ``_bar_legendre_recurrence`` survived it:
production does not call that either, but the FRF expansion's only *independent*
test reference is built on it.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def legendre_recurrence_coefficients(L: int, dtype, device):
    """Coefficient tables for the fully-normalised Legendre recurrence.

    Returns ``(a, b, sect)`` with ``a``, ``b`` of shape ``(L, L)`` indexed
    ``[l, m]`` and ``sect`` of shape ``(L,)``::

        a_l^m = sqrt((2l-1)(2l+1) / ((l-m)(l+m)))
        b_l^m = sqrt((2l+1)(l+m-1)(l-m-1) / ((l-m)(l+m)(2l-3)))
        sect_m = sqrt((2m+1) / (2m))

    ``a`` and ``b`` are **zero wherever m >= l**, which lets a caller run the
    vertical recurrence at full width instead of slicing to ``[:l]``: the
    out-of-support entries come out zero on their own. ``b`` also vanishes at
    l = m+1 through its ``(l-m-1)`` factor, so that case needs no special
    handling.

    Split out of :func:`_bar_legendre_recurrence` because the rotation function
    runs this recurrence itself, fused with its own accumulation, and two copies
    of these formulae would be two chances to get them subtly different.
    """
    # Small integers, exact in any float dtype; the results are cast to `dtype`.
    ll = torch.arange(L, dtype=dtype, device=device).view(L, 1)
    mm = torch.arange(L, dtype=dtype, device=device).view(1, L)
    valid = ll > mm
    denom = (ll - mm) * (ll + mm)
    denom_safe = torch.where(valid, denom, torch.ones_like(denom))
    a = torch.sqrt((2.0 * ll - 1.0) * (2.0 * ll + 1.0) / denom_safe)
    b_num = (2.0 * ll + 1.0) * (ll + mm - 1.0) * (ll - mm - 1.0)
    b_den = denom_safe * (2.0 * ll - 3.0)
    b = torch.sqrt(torch.clamp(
        b_num / torch.where(b_den == 0, torch.ones_like(b_den), b_den), min=0.0))
    a = torch.where(valid, a, torch.zeros_like(a)).to(dtype)
    b = torch.where(valid, b, torch.zeros_like(b)).to(dtype)
    m_arange = torch.arange(L, dtype=dtype, device=device)
    sect = torch.sqrt(
        (2.0 * m_arange + 1.0) / (2.0 * m_arange).clamp(min=1.0)).to(dtype)
    return a, b, sect


#: bar_P_0^0.
LEGENDRE_SEED = 1.0 / math.sqrt(4.0 * math.pi)


def _bar_legendre_recurrence(
    cos_theta: torch.Tensor,
    sin_theta: torch.Tensor,
    L: int,
    keep_l: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fully-normalised associated Legendre ``bar_P_l^m(cos theta)``, all l < L, m <= l.

    **Kept for its test, deliberately.** Production does not call this: the
    Bessel-SH expansion runs the same recurrence inside its compiled kernels,
    from :func:`legendre_recurrence_coefficients` and :data:`LEGENDRE_SEED`.
    That is exactly why this stays -- it is a second, independent, pure-torch
    implementation, and ``tests/unit/frf_separate/test_bessel_sh_grouping.py``
    builds a slow reference expansion on it to check the fused one. The other
    tests there compare ``bessel_sh_expand`` against *itself* at a different
    grouping, so a dropped term cancels; one did, and only this reference caught
    it. Deleting it would leave the expansion checked only against itself.

    Definition:
        bar_P_l^m(x) = √[(2l+1)/(4π) · (l-m)!/(l+m)!] · P_l^m(x)
    where P_l^m is the *unsigned* associated Legendre (no Condon-Shortley phase).

    The recurrence needs only levels ``l-1`` and ``l-2``, so it carries two
    rolling rows rather than reading back out of the full table. That matters at
    the sizes the rotation function uses: an all-l table is ``(batch, L, L)``,
    which for 1e5 batch entries at L=65 is several GB touched once.

    Parameters
    ----------
    cos_theta, sin_theta : torch.Tensor
        Matching real tensors of any batch shape.
    L : int
        Bandwidth; l runs over [0, L).
    keep_l : torch.Tensor, optional
        Which ``l`` rows to return, as an increasing index tensor. ``None``
        returns all of them. Passing only the rows the caller needs -- the even
        ones, for a centrosymmetric Patterson -- halves the output.

    Returns
    -------
    bar_P : torch.Tensor, real
        Shape ``(..., L, L)``, or ``(..., len(keep_l), L)`` when ``keep_l`` is
        given. Entries with m > l are zero.
    """
    batch_shape = cos_theta.shape
    dtype = cos_theta.dtype
    device = cos_theta.device

    inv_sqrt_4pi = LEGENDRE_SEED

    a_coef, b_coef, sect = legendre_recurrence_coefficients(
        L, dtype, device)

    cos_e = cos_theta.unsqueeze(-1)  # (..., 1)
    sin_e = sin_theta.unsqueeze(-1)

    if keep_l is None:
        rows = torch.arange(L, device=device)
    else:
        rows = keep_l.to(device=device, dtype=torch.long)  # dtype-ok: index tensor; index_add_/gather need int64
    # l -> its position in the output, or -1 when it is not kept.
    where = torch.full((L,), -1, dtype=torch.long, device=device)  # dtype-ok: index tensor; index_add_/gather need int64
    where[rows] = torch.arange(rows.numel(), device=device)
    where_list = where.tolist()

    out = torch.zeros((*batch_shape, rows.numel(), L), dtype=dtype, device=device)
    prev2 = torch.zeros((*batch_shape, L), dtype=dtype, device=device)
    prev1 = torch.zeros((*batch_shape, L), dtype=dtype, device=device)
    prev1[..., 0] = inv_sqrt_4pi                             # bar_P_0^0
    if where_list[0] >= 0:
        out[..., where_list[0], :] = prev1

    # One loop over l, updating every m in [0, l] at once. The vertical
    # recurrence (m < l) and the sectoral diagonal (m = l) read only levels l-1
    # and l-2, which is what the two rolling rows hold.
    for l in range(1, L):
        cur = torch.zeros_like(prev1)
        cur[..., :l] = (
            a_coef[l, :l] * cos_e * prev1[..., :l] - b_coef[l, :l] * prev2[..., :l]
        )
        cur[..., l] = sect[l] * sin_e[..., 0] * prev1[..., l - 1]
        pos = where_list[l]
        if pos >= 0:
            out[..., pos, :] = cur
        prev2, prev1 = prev1, cur

    return out


def get_axis_order(sym_mats: torch.Tensor, axis: int) -> int:
    """
    Order of the highest-multiplicity proper rotation about a principal axis.

    `sym_mats` is the spacegroup rotation matrices (n_ops, 3, 3). `axis` is
    0/1/2 for x/y/z. Returns the largest n such that some R in `sym_mats` is a
    rotation by 2π/n around that axis. For non-rotational operations (or
    rotations not aligned with the axis), the spacegroup element is skipped.
    Returns 1 if no proper rotation around the axis exists.

    Used by the Phaser-style m-symmetry filter on the spherical-harmonic
    coefficients: the Patterson is invariant under the spacegroup rotations,
    so m-values that violate the highest-order axis symmetry are pure noise.
    """
    a = torch.zeros(3, dtype=torch.float64, device=sym_mats.device)  # dtype-ok: 3x3 rotation algebra in double on the host
    a[axis] = 1.0
    max_order = 1
    n_ops = sym_mats.shape[0]
    for k in range(n_ops):
        R = sym_mats[k].to(torch.float64)  # dtype-ok: 3x3 rotation algebra in double on the host
        # Axis must be invariant under R (proper or improper rotation about it).
        if (R @ a - a).norm().item() > 1e-3:
            continue
        # Trace of a rotation by angle θ about the preserved axis is 1+2cosθ.
        tr = R.diagonal().sum().item()
        cos_a = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
        # Identity (angle ~0) → order 1.
        if cos_a >= 1.0 - 1e-6:
            continue
        angle = math.acos(cos_a)
        n = round(2 * math.pi / angle)
        if n > max_order:
            max_order = n
    return max_order


def get_high_order_axis(sym_mats: torch.Tensor) -> Tuple[int, int]:
    """
    Return (axis, zsymm) where `axis` ∈ {0, 1, 2} (x/y/z) maximises
    `get_axis_order`, with z preferred on ties (matches Phaser's
    `highOrderAxis()` in SpaceGroup.cc).
    """
    orders = [get_axis_order(sym_mats, a) for a in (0, 1, 2)]
    # Phaser: axis=3 (z); axis=2 if orderY > orderZ; axis=1 if orderX > both.
    if orders[0] > orders[1] and orders[0] > orders[2]:
        axis = 0
    elif orders[1] > orders[2]:
        axis = 1
    else:
        axis = 2
    return axis, orders[axis]


def equal_count_shell_edges(
    s_magnitudes: torch.Tensor,
    P: int,
    s_min: Optional[float] = None,
    s_max: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute equal-count radial shell edges for a list of |s| magnitudes.

    Each of the P shells receives (approximately) the same number of reflections.
    Reflections outside [s_min, s_max] (if given) are excluded.

    Returns
    -------
    edges : torch.Tensor, real, shape (P+1,)
        Shell boundaries, increasing.
    centers : torch.Tensor, real, shape (P,)
        Mid-points of each shell.
    """
    s = s_magnitudes
    if s_min is not None:
        s = s[s >= s_min]
    if s_max is not None:
        s = s[s <= s_max]
    s_sorted, _ = torch.sort(s)
    N = s_sorted.numel()
    # quantile-based partition
    idx = torch.linspace(0, N - 1, P + 1, dtype=s.dtype, device=s.device).round().long()
    edges = s_sorted[idx]
    # nudge endpoints so the data is fully covered (avoid floating-point miss)
    if s_min is not None:
        edges[0] = min(edges[0].item(), s_min)
    else:
        edges[0] = edges[0] - 1e-6
    if s_max is not None:
        edges[-1] = max(edges[-1].item(), s_max)
    else:
        edges[-1] = edges[-1] + 1e-6
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def fit_overall_anisotropy(
    F_obs: torch.Tensor,
    s_vectors: torch.Tensor,
    shell_idx: torch.Tensor,
    centric: torch.Tensor,
    P: int,
    min_count: int = 20,
    n_iter: int = 12,
) -> torch.Tensor:
    """
    Fit the overall anisotropy tensor U from F_obs alone (no model needed).

    The Popov-Bourenkov correction models the observed intensities as a
    per-shell isotropic Wilson piece modulated by an overall anisotropic
    Debye-Waller term::

        E[ I(h) / <I>_shell ] = c * exp(-2 pi^2 s.U.s)

    That expectation is exact in **intensity** space, which is where this fits
    it: a free constant ``c`` absorbs the overall scale, the weights come from
    ``Var(I/<I>)`` -- 1 for acentric reflections and 2 for centric ones -- and
    non-positive or non-finite amplitudes are dropped. Gauss-Newton from
    ``U = 0``.

    Fitting the same relation in log space instead is what the earlier version
    did, and it is biased: ``E[ln(I/<I>)]`` is ``-gamma = -0.577`` for acentric
    and ``-gamma - ln 2`` for centric reflections, not zero. Without a constant
    term that offset can only be absorbed by the quadratic form, so U comes back
    with a large spurious component -- and because centric reflections lie on
    the zones perpendicular to the symmetry axes, the bias is
    direction-dependent rather than a harmless overall scale.

    The returned U is the correction to *apply* in the form::

        F_obs_corrected(h) = F_obs(h) * exp(+pi^2 s.U.s)

    so the corrected amplitudes have the same mean square in every direction.
    Project it onto the point group with :func:`symmetrize_anisotropy` before
    applying it: an unconstrained six-component fit can return a tensor the
    lattice forbids.

    Parameters
    ----------
    F_obs : (N,) real
    s_vectors : (N, 3) real, reciprocal-space Cartesian (1/Angstrom)
    shell_idx : (N,) int64 -- shell of each reflection, in [0, P); negative
        entries are excluded
    centric : (N,) bool
    P : int, number of shells
    min_count : int, optional
        Shells with fewer reflections than this are dropped, since their mean
        intensity is too noisy to normalise against.
    n_iter : int, optional
        Gauss-Newton iterations.

    Returns
    -------
    U : (3, 3) symmetric real tensor (Angstrom squared). Zero if too few
        reflections survive to constrain seven parameters.
    """
    valid = shell_idx >= 0
    F = F_obs[valid].to(torch.float64)  # dtype-ok: seven-parameter Gauss-Newton fit in double on the host, once per search
    s = s_vectors[valid].to(torch.float64)  # dtype-ok: seven-parameter Gauss-Newton fit in double on the host, once per search
    idx = shell_idx[valid]
    cen = centric[valid].bool()

    ok = torch.isfinite(F) & (F > 0)
    F, s, idx, cen = F[ok], s[ok], idx[ok], cen[ok]

    I = F * F
    count = torch.zeros(P, dtype=torch.int64, device=F.device)  # dtype-ok: index tensor; index_add_/gather need int64
    total = torch.zeros(P, dtype=torch.float64, device=F.device)  # dtype-ok: seven-parameter Gauss-Newton fit in double on the host, once per search
    count.index_add_(0, idx, torch.ones_like(idx))
    total.index_add_(0, idx, I)
    mean_I = (total / count.clamp(min=1).to(torch.float64)).clamp(min=1e-30)  # dtype-ok: seven-parameter Gauss-Newton fit in double on the host, once per search

    keep = (count >= min_count)[idx]
    if int(keep.sum()) < 50:
        return torch.zeros((3, 3), dtype=F_obs.dtype, device=F_obs.device)
    ratio = I[keep] / mean_I[idx[keep]]
    sk, cenk = s[keep], cen[keep]

    x, y, z = sk[:, 0], sk[:, 1], sk[:, 2]
    # s.U.s = Uxx sx^2 + Uyy sy^2 + Uzz sz^2
    #         + 2 Uxy sx sy + 2 Uxz sx sz + 2 Uyz sy sz
    quad = torch.stack([x * x, y * y, z * z,
                        2 * x * y, 2 * x * z, 2 * y * z], dim=1)
    # Column 0 is the free constant ln(c); the rest carry -2 pi^2 s.U.s.
    A = torch.cat([torch.ones_like(x).unsqueeze(1),
                   -2.0 * (torch.pi ** 2) * quad], dim=1)
    w = torch.where(cenk, torch.full_like(ratio, 0.5), torch.ones_like(ratio))

    theta = torch.zeros(7, dtype=torch.float64, device=F.device)  # dtype-ok: seven-parameter Gauss-Newton fit in double on the host, once per search
    for _ in range(n_iter):
        model = torch.exp((A @ theta).clamp(min=-20.0, max=20.0))
        J = model.unsqueeze(1) * A
        Jw = J * w.unsqueeze(1)
        H = J.transpose(0, 1) @ Jw
        grad = Jw.transpose(0, 1) @ (ratio - model)
        H = H + torch.eye(7, dtype=H.dtype, device=H.device) * 1e-12 * float(
            torch.diagonal(H).abs().max().clamp(min=1e-30))
        theta = theta + torch.linalg.solve(H, grad)

    u = theta[1:]
    return torch.tensor(
        [[u[0], u[3], u[4]], [u[3], u[1], u[5]], [u[4], u[5], u[2]]],
        dtype=F_obs.dtype, device=F_obs.device,
    )


def hkl_symops_to_cartesian(
    sg_mats: torch.Tensor,
    rec_basis: torch.Tensor,
) -> torch.Tensor:
    """
    Convert spacegroup symmetry operators that act on integer Miller indices
    into the equivalent rotation operators acting on Cartesian reciprocal-
    space vectors (`s = h @ rec_basis`, column-vector form: `s = M @ h` with
    `M = rec_basis^T`).

    For column vectors: `s' = M · S · M⁻¹ · s` so `P_cart = M · S · M⁻¹`.

    For orthogonal cells (orthorhombic+) M is diagonal and `P_cart == S`
    exactly. For non-orthogonal cells (monoclinic, hex/trig with γ=120°,
    triclinic), the Cartesian form differs and matters for any operation
    that mixes the axes (e.g. averaging tensors over the point group).

    Parameters
    ----------
    sg_mats : torch.Tensor, shape (n_ops, 3, 3)
        Integer Miller-index symops (data.spacegroup.matrices).
    rec_basis : torch.Tensor, shape (3, 3)
        Reciprocal basis matrix such that `s = h @ rec_basis` (row-vector
        convention used throughout torchref).

    Returns
    -------
    sym_mats_cart : torch.Tensor, shape (n_ops, 3, 3), real
    """
    dtype = torch.float64  # dtype-ok: 3x3 rotation algebra in double on the host
    M = rec_basis.to(dtype).transpose(-1, -2)            # (3, 3)
    M_inv = torch.linalg.inv(M)
    S = sg_mats.to(dtype)                                 # (n_ops, 3, 3)
    # S^T, not S: reciprocal space transforms as h' = h.S, so the operator
    # acting on Cartesian s as a column vector is (B^-1 S B)^T = M S^T M^-1
    # with M = B^T. Using S here returns matrices that are not rotations at all
    # in a non-orthogonal basis -- measured orthogonality error 5.33 for
    # P 3_1 2 1 and P 6_5 2 2, versus 2e-7 with the transpose. The two agree
    # whenever the symmetry matrices are orthogonal, i.e. everywhere except
    # trigonal/hexagonal, which is why this survived.
    return torch.einsum("ij,klj,lm->kim", M, S, M_inv)


def symmetrize_anisotropy(
    U: torch.Tensor,
    sym_mats_cart: torch.Tensor,
) -> torch.Tensor:
    """
    Project a symmetric 3×3 tensor `U` onto the point-group-invariant
    subspace of the spacegroup by averaging over its Cartesian rotation
    operators:

        U_sym = (1/n) Σ_k  P_k · U · P_k^T

    This mirrors Phaser's `site_symmetry.average_u_star()` and the
    `RefineANO.cc:116-142` constraint construction. After symmetrisation the
    tensor automatically satisfies the crystal's point-group symmetry:

        - cubic:        U_sym = (trace U / 3) · I              (1 DOF)
        - tetragonal:   U_sym = diag(λ, λ, μ)                  (2 DOF)
        - orthorhombic: U_sym = diag(λ, μ, ν)                  (3 DOF)
        - monoclinic:   diagonal + one off-diagonal             (4 DOF)
        - triclinic:    unchanged                               (6 DOF)

    This is the structural fix that prevents an unconstrained 6-component
    regression from producing physically impossible anisotropy in
    high-symmetry cells (e.g. fitting a 70 Å² eigenvalue on a cubic dataset
    where every eigenvalue must be equal by symmetry).

    Parameters
    ----------
    U : torch.Tensor, shape (3, 3), symmetric
    sym_mats_cart : torch.Tensor, shape (n_ops, 3, 3)
        Output of `hkl_symops_to_cartesian` — Cartesian-space rotation
        operators of the spacegroup.

    Returns
    -------
    U_sym : torch.Tensor, shape (3, 3), symmetric, same dtype/device as U
    """
    dtype = U.dtype
    device = U.device
    sym = sym_mats_cart.to(device=device, dtype=dtype)
    # Vectorised: U_avg = mean_k  P_k · U · P_k^T
    U_avg = torch.einsum("kij,jl,knl->in", sym, U, sym) / sym.shape[0]
    # Enforce symmetry (numerical hygiene; pure rotation averaging preserves
    # symmetry exactly, but FP drift can leave 1e-15 asymmetry).
    return 0.5 * (U_avg + U_avg.transpose(-1, -2))


def apply_overall_anisotropy(
    F: torch.Tensor,
    s_vectors: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the inverse of the anisotropy tensor to amplitudes:
        F_corrected(h) = F(h) · exp(+π²·s·U·s)
    (Inverse direction = +π² to undo the Debye-Waller effect.)
    """
    device = F.device
    dtype = F.dtype
    s = s_vectors.to(device).to(dtype)
    U_t = U.to(device).to(dtype)
    s_dot_U = s @ U_t                                       # (N, 3)
    arg = (torch.pi ** 2) * (s_dot_U * s).sum(dim=-1)
    return F * torch.exp(arg.clamp(min=-10.0, max=10.0))


def compute_patterson_shell_variance(
    patt: torch.Tensor,
    shell_idx: torch.Tensor,
    P: int,
    min_count: int = 8,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Empirical per-shell variance of a Patterson coefficient (typically E²−1).

    For each shell p, returns Var_p = <patt²>_p - <patt>_p².

    Shells with fewer than `min_count` reflections inherit the variance of the
    nearest shell that does meet the count threshold (search outward, prefer
    higher-resolution / larger-index shells first). The result is clamped at
    `eps` so the inverse-sqrt weight `1/√Var_p` stays bounded.

    Parameters
    ----------
    patt : torch.Tensor, shape (N,)
        Per-reflection Patterson coefficient (e.g. E² − 1).
    shell_idx : torch.Tensor, shape (N,), int64
        Shell assignment in [0, P). Reflections with index -1 are ignored.
    P : int
        Number of shells.
    min_count : int, default 8
        Shells with fewer reflections than this borrow from a neighbour.
    eps : float, default 1e-3
        Lower bound on returned variance.

    Returns
    -------
    var : torch.Tensor, shape (P,), same dtype as `patt`
    """
    device = patt.device
    dtype = patt.dtype
    valid = shell_idx >= 0
    patt_v = patt[valid]
    idx_v = shell_idx[valid]
    count = torch.zeros(P, dtype=torch.int64, device=device)  # dtype-ok: index tensor; index_add_/gather need int64
    count.index_add_(0, idx_v, torch.ones_like(idx_v))
    sum1 = torch.zeros(P, dtype=dtype, device=device)
    sum2 = torch.zeros(P, dtype=dtype, device=device)
    sum1.index_add_(0, idx_v, patt_v)
    sum2.index_add_(0, idx_v, patt_v * patt_v)
    safe_count = count.clamp(min=1).to(dtype)
    mean = sum1 / safe_count
    var = sum2 / safe_count - mean * mean
    var = var.clamp(min=eps)

    counts_l = count.tolist()
    var_l = var.tolist()
    good = [i for i, c in enumerate(counts_l) if c >= min_count]
    if not good:
        return torch.full_like(var, max(eps, 1.0))
    for i in range(P):
        if counts_l[i] >= min_count:
            continue
        nearest = min(good, key=lambda g, ii=i: (abs(g - ii), -g))
        var_l[i] = var_l[nearest]
    return torch.tensor(var_l, dtype=dtype, device=device).clamp(min=eps)


def assign_shells(
    s_magnitudes: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """
    Assign each reflection to a shell index in [0, P).

    Reflections strictly outside the edges get index -1 (caller may filter).
    """
    # bucketize returns indices in [0, P+1]; we want shell indices in [0, P).
    # Reflections with s == edges[0] go to bucket 0 (use right=True trick).
    idx = torch.bucketize(s_magnitudes.contiguous(), edges.contiguous(), right=True) - 1
    P = edges.shape[0] - 1
    invalid = (idx < 0) | (idx >= P)
    idx = idx.clamp(min=0, max=P - 1)
    idx = torch.where(invalid, torch.full_like(idx, -1), idx)
    return idx
