"""Bessel-radial × spherical-harmonic expansion of obs and calc Pattersons.

Mirrors ``DataMR::dataMR_FRF`` (DataMR.cc) and the helper sums in
``Ensemble.cc``. Contains both the spherical-Bessel table (Miller
recurrence) and the chunked obs/calc Bessel×SH expansion fed into the
cross-correlation that ``SiteListAng::get_FRF`` consumes.

Citations:
  * Bessel-radial × SH expansion: DataMR.cc:993, 1107-1117
  * sqrt(2u+1) · j_u(h)/h radial weight: DataMR.cc:993
  * Even-l only (Patterson centrosymmetry): implicit in DataMR.cc
  * m-symmetry filter: DataMR.cc:863-870, 1117
"""
from __future__ import annotations

import math
import os
import time

import torch

_PROFILE = bool(os.environ.get("FRF_PROFILE"))

#: Byte budget for the per-chunk transients in :func:`bessel_sh_expand`.
#:
#: The chunk holds the Legendre recurrence's rolling rows, so this is really a
#: cache-residency knob, and it has an interior optimum. Measured on an EPYC
#: 9335 at four threads, seconds for the whole rotation function at cap 100 on
#: 3K7M: 24.9 at 2 MB, 12.1 at 8, **9.6 at 32**, 9.9 at 128, 11.0 at 256, 12.3
#: at 1024. Same shape at cap 64. Below the optimum the 100-iteration loop over
#: l is re-run for too many chunks and Python and dispatch overhead dominate;
#: above it the rolling rows stop fitting in cache and the recurrence becomes
#: memory-bound. The truth rank was identical at every setting.
CLUSTER_CHUNK_BYTES = 32_000_000


#: Grouping resolution for |s|, i.e. for the RADIAL factor. Reflections whose |s|
#: agrees to 1/this share one Bessel evaluation.
#:
#: This one has to be fine. The Bessel argument is `bessel_h_scale * |s|`, of
#: order 250 for a protein at L=64, and j_u oscillates on a scale of 2*pi in its
#: argument -- so an error in |s| is amplified by ~250 before it reaches j_u.
#: Against an ungrouped reference the expansion is bitwise exact at 1e9 and
#: 1.8e-8 to 7.5e-8 relative at 1e7, where this used to sit: a systematic error
#: at or above the engine's own run-to-run spread. Costs nothing where the
#: grouping pays most: the dense P1 calc box is exactly degenerate, so it groups
#: identically at 1e7, 1e9 and 1e11 alike.
_GROUP_SCALE_S = 10_000_000

#: Grouping resolution for cos(theta), i.e. for the ANGULAR factor.
#:
#: This one can be coarse, and that is where the speed is. The Legendre factor
#: varies smoothly in cos(theta) with no amplification, and Phaser -- the
#: reference this engine is validated against -- buckets cos(theta) at 1e-3
#: (`lib/sphericalY.h:43`, COSTHETA_LIMIT), evaluating the Legendre polynomials
#: once per bucket. Setting this finer than Phaser buys accuracy the rest of the
#: chain does not have; setting it coarser than Phaser would be a new
#: approximation and needs its own evidence.
_GROUP_SCALE_COS = 10_000_000

from ....config import get_complex_dtype, get_float_dtype
from ....utils.backends import run_or_degrade, select
from ..sh import legendre_recurrence_coefficients
from ._backends import LEGENDRE_BACKENDS

from .types import BesselSHCoefficients

__all__ = [
    "bessel_sh_expand",
    "cross_correlate_xi",
    "spherical_bessel_table",
]


#: Exponent of the running-magnitude rescale in :func:`spherical_bessel_table`.
#:
#: The unnormalised downward ladder is enormous before it is renormalised: at the
#: FRF's low-resolution end (``x = bessel_h_scale / d_max``, about 1.3) the
#: intermediates reach 1e157, and they overflow float32 for every ``x`` below
#: ~35 -- most of the resolution range. Rescaling by a fixed factor whenever the
#: running value crosses it keeps the ladder in range at ANY working precision.
#:
#: It has to be a power of two. Dividing by one only decrements the exponent, so
#: the mantissas of every stored value and of the closing renormalisation are
#: untouched and the rescale introduces no rounding at all -- the table comes out
#: bit-identical to the un-rescaled version. Measured event counts over the FRF's
#: range: 5 rescales at x=1.26, 4 at 1.9, 3 at 5, 2 at 10, 1 at 20, 0 at 64.
_BESSEL_RESCALE_EXP = 100


def spherical_bessel_table(
    x: torch.Tensor,
    u_max: int,
    n_extra: int = 25,
) -> torch.Tensor:
    """Tabulate spherical Bessel ``j_u(x)`` for ``u ∈ [0, u_max]``, batched over ``x``.

    Uses Miller's downward recurrence (the standard stable choice for
    ``j_n(x)`` with ``n > x``):

        j_{u-1}(x) = (2u + 1) / x · j_u(x)  −  j_{u+1}(x)

    Seed: ``n_start = u_max + n_extra``, ``j_{n_start+1} = 0``,
    ``j_{n_start} = 1`` (unnormalised), recur down to ``j_0``, then
    renormalise using the exact ``j_0(x) = sin(x) / x``.

    The ladder is rescaled on the fly by ``2**-_BESSEL_RESCALE_EXP`` whenever it
    grows past that magnitude -- see that constant for why the recurrence needs
    it and why it costs no accuracy. Every rescale divides ``j_mid``, ``j_high``
    **and every row already written**, so the whole table stays in one common
    frame and the closing renormalisation cancels it exactly.

    Note that the high-``u`` rows are genuinely negligible rather than merely
    small: at ``x = 1.9`` every ``u >= 33`` is below float32's smallest normal,
    which is 50% of the band, so those entries are already flushed to zero
    wherever the caller works in single precision.

    Returns
    -------
    j_table : torch.Tensor
        Shape ``(*x.shape, u_max + 1)``, dtype = ``x.dtype``.
    """
    real_dtype = x.dtype
    device = x.device
    x64 = x.to(torch.float64)
    safe_x = x64.clamp(min=1e-30)
    inv_x = 1.0 / safe_x

    n_start = max(u_max + n_extra, u_max + 2)
    j_high = torch.zeros_like(x64)
    j_mid = torch.ones_like(x64)
    j_table = torch.zeros(
        (u_max + 1, *x64.shape), dtype=torch.float64, device=device,
    )
    threshold = float(2 ** _BESSEL_RESCALE_EXP)
    inv_threshold = 1.0 / threshold
    # Rescales applied so far, per element. Every element's ladder sits in the
    # single frame 2**(-_BESSEL_RESCALE_EXP * n_rescales).
    n_rescales = torch.zeros_like(x64, dtype=torch.int32)

    for n in range(n_start, 0, -1):
        j_low = (2.0 * n + 1.0) * inv_x * j_mid - j_high
        if n - 1 <= u_max:
            j_table[n - 1] = j_low
        j_high = j_mid
        j_mid = j_low

        # The `.any()` costs a host sync per step (~90 per call, a handful of
        # calls per search) and buys skipping a pass over the written rows on
        # every step that does not need one. The rows are the expensive part.
        over = j_mid.abs() > threshold
        if bool(over.any()):
            factor = torch.where(over, inv_threshold, 1.0)
            j_mid = j_mid * factor
            j_high = j_high * factor
            if n - 1 <= u_max:
                j_table[n - 1:] = j_table[n - 1:] * factor
            n_rescales = n_rescales + over.to(torch.int32)

    true_j0 = torch.sin(x64) * inv_x
    true_j0 = torch.where(x64 < 1e-30, torch.ones_like(x64), true_j0)
    computed_j0 = j_table[0]
    # The degeneracy guard's 1e-30 is an absolute bound on the UNSCALED j_0, so
    # express it in the frame the ladder actually ended up in. `frame` is an
    # exact power of two; it underflows to 0 past ~10 rescales, at which point
    # the guard simply stops firing (it is unreachable for the FRF anyway, whose
    # x >= bessel_h_scale / d_max keeps sin(x)/x well away from zero).
    frame = torch.ldexp(
        torch.ones_like(x64), -_BESSEL_RESCALE_EXP * n_rescales,
    )
    safe_j0 = torch.where(
        computed_j0.abs() < 1e-30 * frame, frame, computed_j0,
    )
    scale = true_j0 / safe_j0
    j_table = j_table * scale.unsqueeze(0)

    perm = list(range(1, j_table.dim())) + [0]
    j_table = j_table.permute(*perm).contiguous()
    return j_table.to(real_dtype)


def bessel_sh_expand(
    s_vectors: torch.Tensor,
    intensity: torch.Tensor,
    *,
    L: int,
    bessel_h_scale: float,
    zsymm: int = 1,
    enforce_friedel: bool = True,
) -> BesselSHCoefficients:
    """Phaser-style ``c_nlm = Σ_h Y*_lm(ŝ) · I · sqrt(2u+1) · j_u(h)/h``.

    Memory-bounded and chunked, verified element-wise by
    ``tests/unit/frf_separate``. A direct implementation materialises the full
    ``(M, L, N_radial)`` Bessel table and
    ``(M, u_max+1)`` j-table for *all* reflections at once — at L≈100 with
    a symmetry-unrolled obs set (≳10⁶ reflections) that is tens of GB and
    OOMs. Here the j-table, Bessel weights and Y_lm are all computed
    *inside* the reflection-chunk loop, so peak memory is set by one chunk.

    Citations:
      * radial × SH expansion, sqrt(2u+1)·j_u(h)/h weight: DataMR.cc:993, 1107
      * even-l only (Patterson centrosymmetry) + m-filter: DataMR.cc:863-870, 1117

    Two precisions are in play and they are deliberately different.

    The **clustering keys** are computed at ``s_vectors``' own dtype, because
    ``_GROUP_SCALE_S`` keys ``|s|`` at 1e-7 and that is exactly where float32's
    resolution runs out: at ``|s| = 0.5`` a float32 rounding is ~0.3 of a key
    step, so reflections that are mathematically degenerate would sometimes land
    in adjacent keys and the degeneracy collapse the cost model depends on would
    fray. Callers therefore pass float64 ``s_vectors`` even when the rest of the
    chain is single precision.

    Everything else -- the Legendre/Y_lm precompute, the radial weights, the
    contraction and the returned coefficients -- runs at
    :func:`torchref.config.get_float_dtype`, which is this codebase's working
    precision and the dtype the fused CPU kernel is built for. The
    spherical-Bessel recurrence keeps its own float64 internals, where the
    downward ladder needs the dynamic range.
    """
    assert s_vectors.dim() == 2 and s_vectors.shape[-1] == 3
    assert intensity.dim() == 1 and intensity.shape[0] == s_vectors.shape[0]

    device = s_vectors.device
    # The working precision, and the dtype of everything returned. Not derived
    # from the input: the input is deliberately wider (see the docstring).
    comp_real = get_float_dtype()
    complex_dtype = get_complex_dtype()
    real_dtype = comp_real

    if enforce_friedel:
        s_vectors = torch.cat([s_vectors, -s_vectors], dim=0)
        intensity = torch.cat([intensity, intensity], dim=0)

    lmax = L - 1
    lmax_even = lmax if (lmax % 2 == 0) else (lmax - 1)
    if lmax_even < 2:
        raise ValueError(f"L={L} too small; need lmax_even >= 2 (so L >= 3).")
    N_radial = (lmax_even - 2) // 2 + 1
    u_max = lmax_even + 1

    # (l, n) -> u = l + 2n + 1 and the sqrt(2u+1) weight, precomputed once as
    # index tensors so the per-chunk Bessel fill is a single advanced-indexed
    # assignment instead of a Python loop over ~N_radial·(lmax/2) (l, n) pairs.
    even_ls = list(range(2, lmax_even + 1, 2))
    l_list, n_list, u_list, w_list = [], [], [], []
    for l in even_ls:
        # Phaser's per-l radial band: nmax = (lmax - l + 2)/2 (DataMR.cc:894),
        # narrowing from N_radial terms at l=2 to a single term at l=lmax, so
        # the high-l bands cannot carry more radial detail than the reflection
        # set supports. `N_radial` above is only the allocated width (Phaser's
        # widest band); the populated support is this per-l count.
        n_l = (lmax_even - l) // 2 + 1
        for n in range(n_l):
            u = l + 2 * n + 1
            l_list.append(l)
            n_list.append(n)
            u_list.append(u)
            w_list.append(math.sqrt(float(2 * u + 1)))
    l_idx = torch.tensor(l_list, dtype=torch.long, device=device)
    n_idx = torch.tensor(n_list, dtype=torch.long, device=device)
    u_idx = torch.tensor(u_list, dtype=torch.long, device=device)
    w_vec = torch.tensor(w_list, dtype=comp_real, device=device)
    # Only even degrees l ∈ [2, lmax_even] carry signal (odd-l and l=0 are zeroed
    # by Patterson centrosymmetry). Compute / contract Y_lm on these rows only —
    # the assembly + einsum are the bottleneck, so this ~halves them. The full
    # c_nlm keeps the (L, ...) shape with odd/zero rows left at zero.
    even_l_idx = torch.tensor(even_ls, dtype=torch.long, device=device)

    M = s_vectors.shape[0]
    einsum_dtype = complex_dtype

    prof = {"cluster": 0.0, "dbuild": 0.0, "bessel": 0.0, "legendre": 0.0,
            "scatter": 0.0, "contract": 0.0} if _PROFILE else None

    def _tick(t0):
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    if _PROFILE:
        t0 = time.perf_counter()

    # ---- Cluster reflections by (|s|, cosθ) ---------------------------------
    # Both the radial Bessel weight (a function of |s|) and the Legendre barP (a
    # function of cosθ) are constant within a cluster, so the per-reflection SH
    # expansion factorises — only the azimuthal phase e^{imφ} and the intensity
    # vary inside a cluster. This generalises Phaser's cosθ clustering
    # (DataMR.cc:918, HKL_clustered) by ALSO factoring the radial term, which
    # collapses the dominant contraction from O(M·L³) to O(n_clusters·L³). On the
    # dense P1 calc box (and on cubic/tetragonal obs lattices) reflections sharing
    # (H²+K²+L², L_z) land in one cluster → n_clusters ≪ M (~16× fewer). For a
    # non-degenerate reflection set it degrades gracefully to ≈ the per-reflection
    # cost (clusters are singletons), with no change in the result.
    s_mag_all = s_vectors.norm(dim=-1).clamp(min=1e-30)
    cos_all = (s_vectors[..., 2] / s_mag_all).clamp(min=-1.0, max=1.0)
    phi_all = torch.atan2(s_vectors[..., 1], s_vectors[..., 0])
    # Separate resolutions for the two factors: the radial term needs a fine
    # |s| key, the angular term does not. One shared key forces the finer of the
    # two on both, which costs merges the angular part never needed.
    k_s = (s_mag_all * _GROUP_SCALE_S).round().to(torch.int64)
    k_c = (cos_all * _GROUP_SCALE_COS).round().to(torch.int64) + _GROUP_SCALE_COS
    key = k_s * (2 * _GROUP_SCALE_COS + 1) + k_c
    uniq_key, inverse = torch.unique(key, return_inverse=True)
    n_clusters = int(uniq_key.shape[0])
    # Per-group geometry: the MEAN over the group's members, not an arbitrary
    # one of them. Members agree to the key's resolution but not exactly, so a
    # scatter-assignment leaves whichever member was written last -- an error up
    # to the full bin width, and biased. The mean costs one extra reduction and
    # centres it.
    def _group_mean(values, index, n_groups):
        tot = torch.zeros(n_groups, dtype=values.dtype, device=device)
        cnt = torch.zeros(n_groups, dtype=values.dtype, device=device)
        tot.index_add_(0, index, values)
        cnt.index_add_(0, index, torch.ones_like(values))
        return tot / cnt.clamp(min=1.0)

    rep_cos = _group_mean(cos_all.to(comp_real), inverse, n_clusters)
    rep_sin = torch.sqrt((1.0 - rep_cos * rep_cos).clamp(min=0.0))

    # Resolution shells: the distinct |s| values, and which shell each cluster
    # belongs to. The radial factor depends on |s| alone, so it is applied once
    # per shell rather than once per cluster -- and there are far fewer shells
    # than clusters, because many directions share a |s| on a lattice. Measured
    # over the benchmark: 2.7 to 39 clusters per shell.
    uniq_ks, inv_s = torch.unique(k_s, return_inverse=True)
    n_shells = int(uniq_ks.shape[0])
    shell_of_cluster = torch.zeros(n_clusters, dtype=torch.long, device=device)
    shell_of_cluster[inverse] = inv_s
    shell_smag = _group_mean(s_mag_all.to(comp_real), inv_s, n_shells)

    # Reorder the clusters so each shell's members are adjacent. The angular
    # accumulation below scatters every cluster into its shell's row of T; in
    # cluster order those writes land all over T, in shell order they sweep it
    # once. Same arithmetic, and the sort is one pass over n_clusters against a
    # scatter of n_clusters x n_even x L.
    order = torch.argsort(shell_of_cluster)
    shell_of_cluster = shell_of_cluster[order]
    rep_cos = rep_cos[order]
    rep_sin = rep_sin[order]
    if _PROFILE:
        prof["cluster"] += _tick(t0); t0 = time.perf_counter()

    # ---- S[c, p] = Σ_{h∈c} I_h e^{-i p φ_h}, for p = 0 .. L-1 ---------------
    # Only the non-negative half of m is built. The coefficients obey
    #     c[n, l, -p] = (-1)^p conj(c[n, l, +p])
    # exactly -- the intensity, the Bessel weight and the Legendre factor are all
    # real, so m enters only through the azimuthal phase, and P_{l,|m|} does not
    # distinguish +p from -p. Verified bit-exact against the full-range build.
    # That halves both this sum and the contraction below.
    #
    # The Y_lm convention (sh.evaluate_ylm) carries C(m, φ) = (-1)^m e^{imφ} for
    # m >= 0, so conj(C) contributes a (-1)^p factor. It is applied once per
    # (cluster, p) after the sum rather than once per (reflection, p) -- the same
    # number for a factor of M/n_clusters less work.
    p_idx = torch.arange(L, device=device)                          # (L,)
    # `inverse` maps a reflection to its cluster in the ORIGINAL cluster order;
    # the clusters were just permuted into shell order, so compose the two.
    rank_of_cluster = torch.empty_like(order)
    rank_of_cluster[order] = torch.arange(n_clusters, device=device)
    cluster_of_refl = rank_of_cluster[inverse]
    Sp = torch.zeros((n_clusters, L), dtype=einsum_dtype, device=device)
    dchunk = 262_144
    for start_i in range(0, M, dchunk):
        stop = min(start_i + dchunk, M)
        ph = phi_all[start_i:stop].to(comp_real)                   # (c,)
        i_c = intensity[start_i:stop].to(comp_real)                # (c,)
        # e^{-i p phi} = z^p with z = e^{-i phi}, so one transcendental per
        # reflection and a running product over p, rather than a transcendental
        # per (reflection, p). At L=101 over 2.6e6 reflections that is 2.6e8
        # sincos calls replaced by 2.6e6 of them plus a complex multiply each.
        # The product accumulates about L * eps of relative error, ~2e-14, six
        # orders below what the grouping already costs.
        z = torch.polar(torch.ones_like(ph), -ph)                  # (c,)
        ladder = z.unsqueeze(1).expand(-1, L).clone()
        ladder[:, 0] = 1.0                                          # p = 0
        e_neg = torch.cumprod(ladder, dim=1)                        # (c, L) = z^p
        e_neg = (e_neg * i_c.unsqueeze(1)).to(einsum_dtype)
        Sp.index_add_(0, cluster_of_refl[start_i:stop], e_neg)
    sign_p = ((-1.0) ** p_idx.to(comp_real)).to(einsum_dtype)      # (L,)
    Dp = Sp * sign_p.unsqueeze(0)                                   # (n_clusters, L)
    if _PROFILE:
        prof["dbuild"] += _tick(t0); t0 = time.perf_counter()

    # ---- contraction, in two steps ------------------------------------------
    # Direct:
    #     c[n,l,p] = Σ_c B[c,l,n] · P[c,l,p] · D[c,p]
    # but B depends only on |s| and P only on cos(theta), while the cluster index
    # carries both. Summing that way re-multiplies the radial factor once per
    # distinct direction at the same resolution. Grouping the clusters by shell i:
    #     T[i,l,p] = Σ_{c in shell i} P[c,l,p] · D[c,p]      (no radial axis)
    #     c[n,l,p] = Σ_i          B[i,l,n] · T[i,l,p]        (shells, not clusters)
    # which trades n_clusters·N_radial for n_clusters + n_shells·N_radial. On the
    # benchmark that is 2.5x to 18x fewer multiply-adds, and it shrinks the Bessel
    # table by the same clusters-per-shell factor. Exact, not an approximation.
    #
    # The Legendre recurrence is run here rather than called, so each row can be
    # accumulated into T the moment it exists and the (chunk, n_even, L) table is
    # never built. That table was what bounded the chunk width, and the loop over
    # l had to be repeated for every chunk -- 100 iterations of a handful of
    # small kernels, 71 times over, which cost more in launch overhead than the
    # arithmetic did. Without it the chunks are wide enough that the loop runs
    # once or twice in total.
    #
    # `a_coef` and `b_coef` are zero for m >= l, so the vertical recurrence runs
    # at full width; slicing to [:l] instead makes every iteration a differently
    # shaped, mostly tiny kernel.
    n_even = len(even_ls)
    le_idx = (l_idx - 2) // 2                    # l value -> even-l row index
    a_coef, b_coef, sect = legendre_recurrence_coefficients(L, comp_real, device)

    # The whole per-shell sum T and the whole radial table B used to be built at
    # full size, and both are large: at L=101 with 35k shells they are 2.8 GB and
    # 0.7 GB. They are also touched once each, so that is pure memory traffic --
    # and the scatter into a 2.8 GB target misses cache on essentially every
    # write.
    #
    # The clusters are sorted by shell, so a chunk of clusters spans a
    # *contiguous* range of shells. That lets both be per-chunk, and lets the
    # radial contraction be folded into the same loop: once a chunk's shells are
    # complete, contract them and drop them. The arithmetic is identical -- the
    # contraction still costs n_shells x n_even x N_radial x L in total -- but the
    # scatter target is now tens of MB rather than gigabytes, and the running
    # answer c_pos is a few MB, so both stay in cache.
    c_pos = torch.zeros((N_radial, n_even, L), dtype=einsum_dtype, device=device)

    rbytes = 4 if comp_real == torch.float32 else 8
    per_cluster = rbytes * 6 * L
    cstep = max(1, min(n_clusters, CLUSTER_CHUNK_BYTES // max(1, per_cluster)))
    for cs in range(0, n_clusters, cstep):
        ce = min(cs + cstep, n_clusters)
        if _PROFILE:
            t0 = time.perf_counter()
        sh_abs = shell_of_cluster[cs:ce]
        s0 = int(sh_abs[0])
        s1 = int(sh_abs[-1]) + 1                 # sorted, so this is the range
        nb = s1 - s0
        sh = sh_abs - s0                         # shell index within the chunk

        # Radial weights for this chunk's shells only.
        x_s = (bessel_h_scale * shell_smag[s0:s1]).clamp(min=1e-30)
        j_all = spherical_bessel_table(x_s, u_max)               # (nb, u_max+1)
        B = torch.zeros((nb, n_even, N_radial), dtype=comp_real, device=device)
        B[:, le_idx, n_idx] = w_vec.unsqueeze(0) * j_all[:, u_idx] / x_s.unsqueeze(-1)
        if _PROFILE:
            prof["bessel"] += _tick(t0); t0 = time.perf_counter()

        # (n_even, nb, L) each, so Tr[pos] is contiguous for index_add_. Two real
        # accumulators rather than one complex: the scatter stays a real
        # operation and no complex temporary is allocated per (l, chunk).
        Tr = torch.zeros((n_even, nb, L), dtype=comp_real, device=device)
        Ti = torch.zeros((n_even, nb, L), dtype=comp_real, device=device)

        # Legendre recurrence and per-shell accumulation. Both stages are
        # memory-bound in plain torch -- every row of the recurrence makes a
        # round trip -- so this dispatches to a fused kernel where the row stays
        # in cache, and falls back to the torch reference when none is built.
        # `shell_of_cluster` is sorted, which the fused kernel requires: it
        # partitions work by shell so that no two threads write the same
        # accumulator row.
        args = (Tr, Ti, rep_cos[cs:ce], rep_sin[cs:ce],
                Dp[cs:ce].real.contiguous(), Dp[cs:ce].imag.contiguous(),
                sh, a_coef, b_coef, sect)
        backend = select(LEGENDRE_BACKENDS, args[:6])
        run_or_degrade(LEGENDRE_BACKENDS, backend, False, *args)
        if _PROFILE:
            prof["legendre"] += _tick(t0); t0 = time.perf_counter()

        c_pos += torch.complex(
            torch.einsum("iln,lim->nlm", B, Tr),
            torch.einsum("iln,lim->nlm", B, Ti),
        ).to(einsum_dtype)
        if _PROFILE:
            prof["contract"] += _tick(t0)

    # Mirror onto m < 0: c[-p] = (-1)^p conj(c[+p]).
    c_e = torch.zeros((N_radial, n_even, 2 * L - 1), dtype=einsum_dtype,
                      device=device)
    c_e[:, :, (L - 1):] = c_pos
    if L > 1:
        mirror = c_pos[:, :, 1:].conj() * sign_p[1:].view(1, 1, -1)
        c_e[:, :, :(L - 1)] = torch.flip(mirror, dims=(-1,))
    c_nlm = torch.zeros((N_radial, L, 2 * L - 1), dtype=complex_dtype, device=device)
    c_nlm[:, even_l_idx, :] = c_e.to(complex_dtype)
    if _PROFILE:
        prof["contract"] += _tick(t0)

    if _PROFILE:
        tot = sum(prof.values()) + 1e-30
        print(f"[FRF_PROFILE] M={M} n_clusters={n_clusters} ({M/max(1,n_clusters):.1f}x) "
              f"n_shells={n_shells} ({n_clusters/max(1,n_shells):.1f} clu/shell) "
              f"L={L} dtype={comp_real} | "
              + " ".join(f"{k}={v*1000:.0f}ms({100*v/tot:.0f}%)" for k, v in prof.items()),
              flush=True)

    # m-symmetry filter (observed side only; caller passes zsymm=1 for calc).
    if zsymm > 1:
        m_vals = torch.arange(-(L - 1), L, device=device)
        c_nlm[:, :, (m_vals.abs() % zsymm) != 0] = 0.0

    return BesselSHCoefficients(
        coeffs=c_nlm,
        L=L,
        bessel_h_scale=float(bessel_h_scale),
    )


def cross_correlate_xi(
    c_obs: BesselSHCoefficients,
    c_calc: BesselSHCoefficients,
) -> torch.Tensor:
    """Contract obs/calc Bessel-SH coefficients on the radial-Bessel axis.

    Phaser source: the radial sum ``Σ_n c_obs[n,l,m] · conj(c_calc[n,l,m'])``
    happens inside ``DataMR::dataMR_FRF`` before being fed into
    ``SiteListAng::DoRfftStuff`` as the ``clmn`` tensor (FastRot.cc:39).

    Convention:
        xi[l, m, n] = Σ_r c_obs[r, l, n] · conj(c_calc[r, l, m])
    so that the peak Euler triple satisfies ``s_calc = R · s_obs``.

    **Accumulated one step wider than the coefficients.** The radial sum runs
    over oscillating ``j_u``, so the terms alternate in sign and cancel; the
    relative error on the result is then far worse than ``eps * sqrt(n_terms)``
    would suggest, and it compounds through the equally oscillatory Wigner
    contraction and the FFT downstream. Accumulating single-precision data in
    double is the ordinary remedy and it is cheap here -- ``xi`` is
    ``(L, 2L-1, 2L-1)``, 17 MB at L=65, against the 4.4M-element FFT it feeds.

    Running the whole tail in single instead was measured on 3K7M and 1DAW: the
    top peak and its z-score were unchanged to seven figures, but scores moved
    by 1e-4 to 1.4e-3 relative and only **1 of 500** candidate slots still held
    the same orientation, because the greedy SO(3) NMS is sequential and a
    reordering cascades through the suppression decisions. The candidate list is
    what the placement search consumes, so that is not a free trade.

    Returns
    -------
    xi : torch.Tensor (complex), shape (L, 2L-1, 2L-1)
    """
    if c_obs.L != c_calc.L:
        raise ValueError(f"L mismatch: obs={c_obs.L} calc={c_calc.L}")
    acc = (
        torch.complex128
        if c_obs.coeffs.dtype in (torch.complex64, torch.complex128)
        and torch.finfo(c_obs.coeffs.real.dtype).bits <= 32
        else c_obs.coeffs.dtype
    )
    return torch.einsum(
        "rln,rlm->lmn",
        c_obs.coeffs.to(acc),
        torch.conj(c_calc.coeffs).to(acc),
    )
