"""Triton kernels for the batched Bessel x spherical-harmonic expansion.

Three kernels, driven by
:class:`torchref.experimental.alignment.frf.batched.ExpansionPlan` -- the first once per
reflection geometry, the other two once per set of intensities:

1. ``_bessel_table_kernel`` -- ``j_u(x)`` for every resolution shell, by the same
   rescaled Miller downward recurrence as
   :func:`~torchref.experimental.alignment.frf.data_mr.spherical_bessel_table`, one lane
   per shell. The torch version needs a host sync per recurrence step to decide whether
   to rescale; here each lane carries its own rescale count, so there is none.
2. ``_legendre_shell_kernel`` -- ``T[s, l, p] = sum_{h in s} bar_P_l^p(cos theta_h)
   (-1)^p e^{-i p phi_h} I_h`` for even ``l``, one program per (shell, block of ``p``).
   The recurrence runs in registers; a program owns its shell's rows of ``T``.
3. ``_radial_kernel`` -- ``c[e, n, l, p] = sum_{s in e} rho_{l,n}(x_s) T[s, l, p]``,
   one program per (expansion, even ``l``, block of ``p``), the radial weights built
   on the fly from the Bessel table.

Every program writes only rows it owns, so there are no atomics and the result is
bit-reproducible. float32 and CUDA only; the dispatch table sends everything else to
the portable path.
"""

from __future__ import annotations

import math

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

#: Power-of-two exponent of the Bessel ladder's rescale; the same value as
#: ``data_mr._BESSEL_RESCALE_EXP`` so both paths rescale at the same points.
RESCALE_EXP = tl.constexpr(100)
RESCALE_THRESH = tl.constexpr(float(2**100))
RESCALE_INV = tl.constexpr(float(2**-100))
INV_TWO_PI = tl.constexpr(1.0 / (2.0 * math.pi))
TWO_PI = tl.constexpr(2.0 * math.pi)


@triton.jit
def _bessel_table_kernel(
    x_ptr,
    raw_ptr,
    cnt_ptr,
    out_ptr,
    n_shells,
    n_start,
    U1: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    valid = offs < n_shells
    x = tl.load(x_ptr + offs, mask=valid, other=1.0)
    x = tl.maximum(x, 1e-30)
    inv_x = 1.0 / x

    j_high = tl.zeros([BLOCK_S], dtype=tl.float32)
    j_mid = tl.full([BLOCK_S], 1.0, dtype=tl.float32)
    n_res = tl.zeros([BLOCK_S], dtype=tl.int32)
    for k in range(0, n_start):
        n = n_start - k
        j_low = (2.0 * n + 1.0) * inv_x * j_mid - j_high
        j_high = j_mid
        j_mid = j_low
        over = tl.abs(j_mid) > RESCALE_THRESH
        j_mid = tl.where(over, j_mid * RESCALE_INV, j_mid)
        j_high = tl.where(over, j_high * RESCALE_INV, j_high)
        n_res += over.to(tl.int32)
        # Stored after the rescale with the count it was stored under: every later
        # rescale also applies to this row, which the fix-up below reproduces.
        u = n - 1
        row_ok = valid & (u < U1)
        tl.store(raw_ptr + offs * U1 + u, j_mid, mask=row_ok)
        tl.store(cnt_ptr + offs * U1 + u, n_res, mask=row_ok)

    # Normalise on j_0 or j_1, whichever is larger at this x. j_0 alone is
    # ill-conditioned near its zeros (x ~ k pi), where a ratio of two small numbers
    # rescales the whole row; in float32 that cost ~1e-3 of the largest j. The zeros
    # of j_0 and j_1 interlace, so the larger of the two is never near one.
    # libdevice, not tl.sin/cos: the fast approximation loses arguments far beyond pi.
    sx = libdevice.sin(x)
    cx = libdevice.cos(x)
    true_j0 = tl.where(x < 1e-30, 1.0, sx * inv_x)
    true_j1 = tl.where(x < 1e-30, 0.0, (sx * inv_x - cx) * inv_x)
    raw1 = tl.load(raw_ptr + offs * U1 + 1, mask=valid, other=1.0)
    cnt1 = tl.load(cnt_ptr + offs * U1 + 1, mask=valid, other=0)
    j1_final = raw1 * tl.exp2(-1.0 * RESCALE_EXP * (n_res - cnt1).to(tl.float32))
    use_j1 = tl.abs(true_j1) > tl.abs(true_j0)
    num = tl.where(use_j1, true_j1, true_j0)
    den = tl.where(use_j1, j1_final, j_mid)
    scale = num / tl.where(den == 0.0, 1.0, den)
    for u in range(0, U1):
        raw = tl.load(raw_ptr + offs * U1 + u, mask=valid, other=0.0)
        cnt = tl.load(cnt_ptr + offs * U1 + u, mask=valid, other=0)
        frame = tl.exp2(-1.0 * RESCALE_EXP * (n_res - cnt).to(tl.float32))
        tl.store(out_ptr + offs * U1 + u, raw * frame * scale, mask=valid)


@triton.jit
def _legendre_shell_kernel(
    cos_ptr,
    sin_ptr,
    phi_ptr,
    I_ptr,
    shell_ptr,
    a_ptr,
    b_ptr,
    sect_ptr,
    Tr_ptr,
    Ti_ptr,
    L,
    n_even,
    seed,
    BLOCK_R: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    s = tl.program_id(0)
    pb = tl.program_id(1)
    p0 = pb * BLOCK_P
    p = p0 + tl.arange(0, BLOCK_P)
    p_valid = p < L
    pf = p.to(tl.float32)
    sign = 1.0 - 2.0 * (p % 2).to(
        tl.float32
    )  # (-1)^p, the Condon-Shortley phase of conj(Y)

    r0 = tl.load(shell_ptr + s)
    r1 = tl.load(shell_ptr + s + 1)
    # Rows l < p0 are zero on every lane of this block, so the vertical recurrence
    # starts at l = p0 from zero rows; only the sectoral chain has to be walked up to
    # there, and that is one scalar per reflection.
    l_start = tl.maximum(p0, 1)

    for r in range(r0, r1, BLOCK_R):
        rows = r + tl.arange(0, BLOCK_R)
        rv = rows < r1
        cs = tl.load(cos_ptr + rows, mask=rv, other=0.0)
        sn = tl.load(sin_ptr + rows, mask=rv, other=0.0)
        ph = tl.load(phi_ptr + rows, mask=rv, other=0.0)
        inten = tl.load(I_ptr + rows, mask=rv, other=0.0)

        # p * phi reaches 64 pi; reduce to [-pi, pi] in turns first, because the fast
        # sin/cos is only accurate there.
        turns = ph[:, None] * pf[None, :] * INV_TWO_PI
        ang = (turns - tl.floor(turns + 0.5)) * TWO_PI
        amp = inten[:, None] * sign[None, :]
        Dr = amp * tl.cos(ang)
        Di = -amp * tl.sin(ang)

        sec = tl.full([BLOCK_R], seed, dtype=tl.float32)
        for l in range(1, l_start):
            sec = tl.load(sect_ptr + l) * sn * sec
        prev2 = tl.zeros([BLOCK_R, BLOCK_P], dtype=tl.float32)
        prev1 = tl.where((p[None, :] == 0) & rv[:, None], seed, 0.0) + prev2

        for l in range(l_start, L):
            a = tl.load(a_ptr + l * L + p, mask=p_valid, other=0.0)
            b = tl.load(b_ptr + l * L + p, mask=p_valid, other=0.0)
            cur = a[None, :] * cs[:, None] * prev1 - b[None, :] * prev2
            sec = tl.load(sect_ptr + l) * sn * sec
            cur = tl.where(p[None, :] == l, sec[:, None], cur)
            if l % 2 == 0:
                if l >= 2:
                    le = l // 2 - 1
                    off = (s * n_even + le) * L + p
                    tr = tl.sum(cur * Dr, axis=0)
                    ti = tl.sum(cur * Di, axis=0)
                    tl.store(
                        Tr_ptr + off,
                        tl.load(Tr_ptr + off, mask=p_valid) + tr,
                        mask=p_valid,
                    )
                    tl.store(
                        Ti_ptr + off,
                        tl.load(Ti_ptr + off, mask=p_valid) + ti,
                        mask=p_valid,
                    )
            prev2 = prev1
            prev1 = cur


@triton.jit
def _radial_kernel(
    j_ptr,
    x_ptr,
    Tr_ptr,
    Ti_ptr,
    exp_shell_ptr,
    cr_ptr,
    ci_ptr,
    L,
    n_even,
    n_rad,
    lmax_even,
    U1: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    e = tl.program_id(0)
    le = tl.program_id(1)
    pb = tl.program_id(2)
    l = 2 * le + 2

    n = tl.arange(0, BLOCK_N)
    # Phaser's per-l radial band (DataMR.cc:894): n_l = (lmax - l)/2 + 1 terms.
    n_valid = (n <= (lmax_even - l) // 2) & (n < n_rad)
    u = l + 2 * n + 1
    w = tl.sqrt((2.0 * u + 1.0).to(tl.float32))
    p = pb * BLOCK_P + tl.arange(0, BLOCK_P)
    p_valid = p < L

    s0 = tl.load(exp_shell_ptr + e)
    s1 = tl.load(exp_shell_ptr + e + 1)
    acc_r = tl.zeros([BLOCK_N, BLOCK_P], dtype=tl.float32)
    acc_i = tl.zeros([BLOCK_N, BLOCK_P], dtype=tl.float32)
    for s in range(s0, s1, BLOCK_S):
        ss = s + tl.arange(0, BLOCK_S)
        sv = ss < s1
        x = tl.load(x_ptr + ss, mask=sv, other=1.0)
        jv = tl.load(
            j_ptr + ss[:, None] * U1 + u[None, :],
            mask=sv[:, None] & n_valid[None, :],
            other=0.0,
        )
        rho = w[None, :] * jv / x[:, None]  # (S, N)
        toff = (ss[:, None] * n_even + le) * L + p[None, :]
        tmask = sv[:, None] & p_valid[None, :]
        tr = tl.load(Tr_ptr + toff, mask=tmask, other=0.0)  # (S, P)
        ti = tl.load(Ti_ptr + toff, mask=tmask, other=0.0)
        rho_t = tl.trans(rho)  # (N, S)
        acc_r = tl.dot(rho_t, tr, acc_r, input_precision="ieee")
        acc_i = tl.dot(rho_t, ti, acc_i, input_precision="ieee")

    coff = ((e * n_rad + n[:, None]) * n_even + le) * L + p[None, :]
    cmask = (n[:, None] < n_rad) & p_valid[None, :]
    tl.store(cr_ptr + coff, acc_r, mask=cmask)
    tl.store(ci_ptr + coff, acc_i, mask=cmask)


def launch_bessel_table(shell_x, *, u_max, n_extra):
    """``j_u(x)`` for ``u`` in ``[0, u_max]`` at every shell: ``(n_shells, u_max+1)``.

    ``shell_x`` is contiguous float32 on a CUDA device.
    """
    import torch

    f32 = torch.float32  # dtype-ok: the kernels' dtype contract
    dev = shell_x.device
    n_shells = int(shell_x.shape[0])
    U1 = u_max + 1
    n_start = max(u_max + n_extra, u_max + 2)
    raw = torch.empty((n_shells, U1), dtype=f32, device=dev)
    cnt = torch.empty(
        (n_shells, U1), dtype=torch.int32, device=dev
    )  # dtype-ok: kernel scratch
    jtab = torch.empty((n_shells, U1), dtype=f32, device=dev)
    BLOCK_S = 128
    _bessel_table_kernel[(triton.cdiv(n_shells, BLOCK_S),)](
        shell_x,
        raw,
        cnt,
        jtab,
        n_shells,
        n_start,
        U1=U1,
        BLOCK_S=BLOCK_S,
    )
    return jtab


def launch_expansion(
    cos_t,
    sin_t,
    phi,
    inten,
    shell_ptr,
    shell_x,
    jtab,
    exp_shell_ptr,
    a_coef,
    b_coef,
    sect,
    *,
    L,
    lmax_even,
    n_rad,
    seed,
):
    """Legendre/shell accumulation then radial contraction; ``(c_re, c_im)``.

    Both outputs are ``(E, n_rad, n_even, L)`` float32. All inputs must be contiguous
    float32 (int32 for the CSR pointers) on one CUDA device, with ``jtab`` from
    :func:`launch_bessel_table`. Reflections must be sorted by shell and shells by
    expansion -- the programs partition work by those ranges, and unsorted input would
    silently drop contributions rather than raise.
    """
    import torch

    dev = cos_t.device
    f32 = torch.float32  # dtype-ok: the kernels' dtype contract
    n_shells = int(shell_x.shape[0])
    n_exp = int(exp_shell_ptr.shape[0]) - 1
    n_even = (lmax_even - 2) // 2 + 1
    U1 = int(jtab.shape[1])

    Tr = torch.zeros((n_shells, n_even, L), dtype=f32, device=dev)
    Ti = torch.zeros((n_shells, n_even, L), dtype=f32, device=dev)
    BLOCK_P = 32
    _legendre_shell_kernel[(n_shells, triton.cdiv(L, BLOCK_P))](
        cos_t,
        sin_t,
        phi,
        inten,
        shell_ptr,
        a_coef,
        b_coef,
        sect,
        Tr,
        Ti,
        L,
        n_even,
        float(seed),
        BLOCK_R=32,
        BLOCK_P=BLOCK_P,
        num_warps=4,
    )

    c_re = torch.empty((n_exp, n_rad, n_even, L), dtype=f32, device=dev)
    c_im = torch.empty((n_exp, n_rad, n_even, L), dtype=f32, device=dev)
    BLOCK_N = max(16, triton.next_power_of_2(n_rad))
    _radial_kernel[(n_exp, n_even, triton.cdiv(L, BLOCK_P))](
        jtab,
        shell_x,
        Tr,
        Ti,
        exp_shell_ptr,
        c_re,
        c_im,
        L,
        n_even,
        n_rad,
        lmax_even,
        U1=U1,
        BLOCK_S=32,
        BLOCK_N=BLOCK_N,
        BLOCK_P=BLOCK_P,
        num_warps=4,
    )
    return c_re, c_im
