"""Triton kernel for batched fast-translation-function maps.

The Crowther-Blow map ``TF(t) = sum_h c(h) |sum_i G_i(h) e^{2 pi i (h R_i).t}|^2`` is,
term by term, a sum of pair coefficients ``c(h) conj(G_i(h)) G_j(h)`` placed at the
reciprocal-grid frequency ``(h R_j - h R_i) mod grid`` followed by one inverse FFT (see
:func:`~torchref.experimental.alignment.translation.fast_translation_function`). This
kernel does the placement for every candidate of every system in a single launch; the
FFT stays with cuFFT.

One program per (candidate, symmetry operation ``i``, block of reflections): it loads
``G_i`` once and walks ``j > i``. Different programs land on the same grid frequency,
so the accumulation uses float32 atomics -- the map is therefore not bit-reproducible
run to run (differences at float32 rounding), exactly like the ``index_add_`` of the
single-candidate torch path it replaces.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _tf_pair_scatter_kernel(
    gr_ptr,
    gi_ptr,
    hr_ptr,
    coeff_ptr,
    cand_S_ptr,
    cand_N_ptr,
    cand_g_off_ptr,
    cand_h_off_ptr,
    cand_c_off_ptr,
    cand_grid_ptr,
    cand_w_off_ptr,
    w_ptr,
    BLOCK_H: tl.constexpr,
):
    q = tl.program_id(0)
    i = tl.program_id(1)
    hb = tl.program_id(2)
    S = tl.load(cand_S_ptr + q)
    N = tl.load(cand_N_ptr + q)
    if (i < S - 1) & (hb * BLOCK_H < N):
        g_off = tl.load(cand_g_off_ptr + q).to(tl.int64)
        h_off = tl.load(cand_h_off_ptr + q).to(tl.int64)
        c_off = tl.load(cand_c_off_ptr + q).to(tl.int64)
        nx = tl.load(cand_grid_ptr + q * 3 + 0)
        ny = tl.load(cand_grid_ptr + q * 3 + 1)
        nz = tl.load(cand_grid_ptr + q * 3 + 2)
        w_base = tl.load(cand_w_off_ptr + q) * 2

        h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        hv = h < N
        c = tl.load(coeff_ptr + c_off + h, mask=hv, other=0.0)
        row_i = i * N + h
        gir = tl.load(gr_ptr + g_off + row_i, mask=hv, other=0.0)
        gii = tl.load(gi_ptr + g_off + row_i, mask=hv, other=0.0)
        hix = tl.load(hr_ptr + (h_off + row_i) * 3 + 0, mask=hv, other=0)
        hiy = tl.load(hr_ptr + (h_off + row_i) * 3 + 1, mask=hv, other=0)
        hiz = tl.load(hr_ptr + (h_off + row_i) * 3 + 2, mask=hv, other=0)
        # c * conj(G_i), formed once per program.
        ar = c * gir
        ai = -c * gii
        for j in range(i + 1, S):
            row_j = j * N + h
            gjr = tl.load(gr_ptr + g_off + row_j, mask=hv, other=0.0)
            gji = tl.load(gi_ptr + g_off + row_j, mask=hv, other=0.0)
            dx = tl.load(hr_ptr + (h_off + row_j) * 3 + 0, mask=hv, other=0) - hix
            dy = tl.load(hr_ptr + (h_off + row_j) * 3 + 1, mask=hv, other=0) - hiy
            dz = tl.load(hr_ptr + (h_off + row_j) * 3 + 2, mask=hv, other=0) - hiz
            # Python-style modulo: `%` on negative ints keeps the dividend's sign.
            gx = ((dx % nx) + nx) % nx
            gy = ((dy % ny) + ny) % ny
            gz = ((dz % nz) + nz) % nz
            g = ((gx * ny + gy) * nz + gz).to(tl.int64)
            vr = ar * gjr - ai * gji
            vi = ar * gji + ai * gjr
            tl.atomic_add(w_ptr + w_base + g * 2, vr, mask=hv)
            tl.atomic_add(w_ptr + w_base + g * 2 + 1, vi, mask=hv)


def launch_pair_scatter(
    gr,
    gi,
    hr,
    coeff,
    cand_S,
    cand_N,
    cand_g_off,
    cand_h_off,
    cand_c_off,
    cand_grid,
    cand_w_off,
    w_real,
    S_max,
    N_max,
):
    """Accumulate the pair coefficients of every candidate into ``w_real``.

    ``w_real`` is the flat float32 view of a zeroed complex64 buffer holding every
    candidate's grid back to back (``torch.view_as_real``); it is added to in place.
    ``gr``/``gi`` are the concatenated real/imaginary ``G`` rows, ``hr`` the
    concatenated ``(S*N, 3)`` int32 rotated indices, ``coeff`` the concatenated
    per-reflection weights. The ``cand_*`` arrays give each candidate's ``S``, ``N``,
    its offsets into those three, its grid ``(nx, ny, nz)`` and the complex offset of
    its grid in the buffer (int64).
    """
    Q = int(cand_S.shape[0])
    BLOCK_H = 128
    if S_max < 2 or Q == 0:
        return
    grid = (Q, S_max - 1, triton.cdiv(N_max, BLOCK_H))
    _tf_pair_scatter_kernel[grid](
        gr,
        gi,
        hr,
        coeff,
        cand_S,
        cand_N,
        cand_g_off,
        cand_h_off,
        cand_c_off,
        cand_grid,
        cand_w_off,
        w_real,
        BLOCK_H=BLOCK_H,
        num_warps=4,
    )
