"""Batched fast rotation function: many reflection sets in one pass, maps out.

The single-system engine
(:class:`~torchref.experimental.alignment.frf.api.FastRotationFunction`) expands one
observed and one calculated set, correlates them, and samples the result on Phaser's
adaptive list. This module does the same arithmetic for a whole batch and returns
dense maps instead of samples:

* :class:`ExpansionPlan` -- ``E`` reflection sets to ``(E, N_rad, L, 2L-1)``
  coefficients, dispatched through :data:`EXPAND_BACKENDS` (fused Triton kernels on
  CUDA float32, a loop over
  :func:`~torchref.experimental.alignment.frf.data_mr.bessel_sh_expand` elsewhere).
  The geometry is prepared once and reused for every new set of intensities;
  :func:`batched_bessel_sh_expand` is the one-shot form.
* :func:`batched_cross_correlate_xi` -- ``xi`` for ``B`` (obs, calc) pairs.
* :func:`batched_rotation_maps` -- ``(B, n_beta, N, N)`` rotation-function maps, the
  per-beta Wigner contraction followed by one batched FFT.

Everything is forward-only (no autograd) and runs at the configured working precision.
The obs-side and calc-side preparation (Wilson normalisation, weights, the dense P1 box)
is the caller's, exactly as for the single-system engine.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch

from torchref.config import get_complex_dtype, get_float_dtype, get_int_dtype
from torchref.symmetry.symmetry import find_fft_friendly_size
from torchref.utils.backends import run_or_degrade, select

from ..sh import LEGENDRE_SEED, legendre_recurrence_coefficients
from ._backends import EXPAND_BACKENDS
from .wigner_d import _wigner_d_blocks

__all__ = [
    "ExpansionPlan",
    "batched_bessel_sh_expand",
    "batched_cross_correlate_xi",
    "batched_rotation_maps",
    "rotation_map_grid",
]

#: Grouping resolution for ``|s|`` (the radial factor); ``data_mr._GROUP_SCALE_S``.
_GROUP_SCALE_S = 10_000_000

#: Extra downward-recurrence steps above ``u_max`` for the Bessel ladder, as in
#: :func:`~torchref.experimental.alignment.frf.data_mr.spherical_bessel_table`.
_BESSEL_N_EXTRA = 25


def _band(L: int) -> Tuple[int, int, int]:
    """``(lmax_even, n_even, u_max)`` for bandwidth ``L``; ``n_even`` is also N_rad."""
    lmax = L - 1
    lmax_even = lmax if lmax % 2 == 0 else lmax - 1
    if lmax_even < 2:
        raise ValueError(f"L={L} too small; need lmax_even >= 2 (so L >= 3).")
    n_even = (lmax_even - 2) // 2 + 1
    return lmax_even, n_even, lmax_even + 1


def _as_list(value, n: int, name: str) -> list:
    if isinstance(value, (int, float)):
        return [value] * n
    value = list(value)
    if len(value) != n:
        raise ValueError(f"{name} has {len(value)} entries for {n} reflection sets")
    return value


def _assemble(c_pos: torch.Tensor, L: int, zsymm: Sequence[int]) -> torch.Tensor:
    """Mirror ``m >= 0`` onto ``m < 0``, place the even-l rows, apply each m-filter.

    ``c_pos`` is ``(E, N_rad, n_even, L)``; returns ``(E, N_rad, L, 2L-1)``.
    """
    E, n_rad, n_even, _ = c_pos.shape
    dev = c_pos.device
    p_idx = torch.arange(L, device=dev)
    sign_p = (1.0 - 2.0 * (p_idx % 2)).to(c_pos.dtype)
    c_e = torch.zeros((E, n_rad, n_even, 2 * L - 1), dtype=c_pos.dtype, device=dev)
    c_e[..., L - 1 :] = c_pos
    if L > 1:
        # c[-p] = (-1)^p conj(c[+p]): intensity, Bessel weight and Legendre factor are
        # real, so m enters only through the azimuthal phase.
        c_e[..., : L - 1] = torch.flip(c_pos[..., 1:].conj() * sign_p[1:], dims=(-1,))
    out = torch.zeros((E, n_rad, L, 2 * L - 1), dtype=c_pos.dtype, device=dev)
    even_l = torch.arange(2, 2 * n_even + 1, 2, device=dev)
    out[:, :, even_l, :] = c_e
    m_vals = torch.arange(-(L - 1), L, device=dev)
    for e, z in enumerate(zsymm):
        if int(z) > 1:
            out[e, :, :, (m_vals.abs() % int(z)) != 0] = 0.0
    return out


class _ShellGeometry:
    """The fused kernels' view of a batch of reflection sets: sorted and tabulated.

    Built once per geometry. Reflections are concatenated over sets, grouped into
    resolution shells per set, and sorted by (set, shell); ``order`` maps that sorted
    position back to the concatenated input, so intensities are gathered through it.
    """

    def __init__(self, s_list, L, scales):
        from .kernels.triton_expand import launch_bessel_table

        f32 = torch.float32  # dtype-ok: the kernels' dtype contract
        i32 = torch.int32  # dtype-ok: the kernels' CSR contract
        dev = s_list[0].device
        _, _, u_max = _band(L)
        E = len(s_list)
        counts = torch.tensor([int(s.shape[0]) for s in s_list], device=dev)
        exp_id = torch.repeat_interleave(torch.arange(E, device=dev), counts)
        s_all = torch.cat([s.detach() for s in s_list], dim=0)

        # Shell keys at the caller's precision: |s| is resolved to 1e-7 before the
        # Bessel argument multiplies it by ~250 (see data_mr._GROUP_SCALE_S). The packed
        # key needs int64; the set index sits above 2**32.
        smag = s_all.norm(dim=-1)
        k_s = (
            (smag * _GROUP_SCALE_S).round().to(torch.int64)
        )  # dtype-ok: packed shell key
        key = exp_id.to(torch.int64) * (1 << 32) + k_s  # dtype-ok: packed shell key
        uniq, shell_of = torch.unique(key, sorted=True, return_inverse=True)
        n_shells = int(uniq.shape[0])
        order = torch.argsort(shell_of, stable=True)
        shell_of = shell_of[order]
        s_all = s_all[order]
        smag = smag[order]

        counts_s = torch.bincount(shell_of, minlength=n_shells)
        self.shell_ptr = torch.zeros(n_shells + 1, dtype=i32, device=dev)
        self.shell_ptr[1:] = torch.cumsum(counts_s, 0).to(i32)
        smag_sum = torch.zeros(n_shells, dtype=smag.dtype, device=dev)
        smag_sum.index_add_(0, shell_of, smag)
        # The group MEAN, as in data_mr: members agree only to the key's resolution.
        shell_smag = (smag_sum / counts_s.to(smag.dtype)).to(f32)
        shell_exp = (uniq >> 32).to(get_int_dtype())
        scale_t = torch.tensor([float(b) for b in scales], dtype=f32, device=dev)
        self.shell_x = (scale_t[shell_exp] * shell_smag).clamp(min=1e-30).contiguous()
        self.exp_shell_ptr = torch.zeros(E + 1, dtype=i32, device=dev)
        self.exp_shell_ptr[1:] = torch.cumsum(
            torch.bincount(shell_exp, minlength=E), 0
        ).to(i32)

        smag_c = smag.clamp(min=1e-30)
        self.cos_t = (s_all[:, 2] / smag_c).clamp(-1.0, 1.0).to(f32).contiguous()
        self.sin_t = torch.sqrt(
            (1.0 - self.cos_t * self.cos_t).clamp(min=0.0)
        ).contiguous()
        self.phi = torch.atan2(s_all[:, 1], s_all[:, 0]).to(f32).contiguous()
        self.order = order
        a_coef, b_coef, sect = legendre_recurrence_coefficients(L, f32, dev)
        self.coefs = (a_coef.contiguous(), b_coef.contiguous(), sect.contiguous())
        self.jtab = launch_bessel_table(
            self.shell_x, u_max=u_max, n_extra=_BESSEL_N_EXTRA
        )


def _expand_triton(plan, I_list, zsymm):
    """Fused-kernel expansion; see :meth:`ExpansionPlan.expand`."""
    from .kernels.triton_expand import launch_expansion

    f32 = torch.float32  # dtype-ok: the kernels' dtype contract
    geo = plan._triton_geometry()
    lmax_even, n_even, _ = _band(plan.L)
    I_all = torch.cat([i.detach().to(f32).reshape(-1) for i in I_list])[
        geo.order
    ].contiguous()
    c_re, c_im = launch_expansion(
        geo.cos_t,
        geo.sin_t,
        geo.phi,
        I_all,
        geo.shell_ptr,
        geo.shell_x,
        geo.jtab,
        geo.exp_shell_ptr,
        *geo.coefs,
        L=plan.L,
        lmax_even=lmax_even,
        n_rad=n_even,
        seed=LEGENDRE_SEED,
    )
    return _assemble(torch.complex(c_re, c_im).to(get_complex_dtype()), plan.L, zsymm)


def _expand_portable(plan, I_list, zsymm):
    """Loop over the single-set expansion: the reference for the fused path."""
    from .data_mr import bessel_sh_expand

    return torch.stack(
        [
            bessel_sh_expand(
                s, i, L=plan.L, bessel_h_scale=float(b), zsymm=int(z)
            ).coeffs
            for s, i, b, z in zip(plan.s_list, I_list, plan.scales, zsymm)
        ]
    )


class ExpansionPlan:
    """A batch of reflection sets, prepared once for repeated Bessel-SH expansion.

    Everything the expansion needs from the geometry alone -- the shell grouping, the
    sort, the polar and azimuthal angles and the Bessel table -- depends only on the
    reflection vectors. A plan computes it on first use and keeps it, so expanding new
    intensities on the same reflections (a model that moves, the same observations
    against many models) costs only the two intensity-dependent kernels.

    Parameters
    ----------
    s_list : sequence of torch.Tensor
        ``E`` Cartesian reciprocal-space vector sets, each ``(M_e, 3)`` in Å⁻¹, all on
        one device. A double-precision set keeps its width for the shell keys only.
    L : int
        Angular bandwidth shared by every set; ``l`` in ``[0, L)``.
    bessel_h_scale : float or sequence of float
        Bessel argument scale ``h = bessel_h_scale * |s|`` in Å, per set or shared.

    Notes
    -----
    Holds references to ``s_list``; mutating those tensors in place afterwards leaves
    the plan describing the old geometry.
    """

    def __init__(self, s_list: Sequence[torch.Tensor], *, L: int, bessel_h_scale):
        if len(s_list) == 0:
            raise ValueError("need at least one reflection set")
        self.s_list = list(s_list)
        self.L = int(L)
        _band(self.L)
        self.scales = _as_list(bessel_h_scale, len(self.s_list), "bessel_h_scale")
        self.sizes = [int(s.shape[0]) for s in self.s_list]
        self._geometry = None

    def __len__(self) -> int:
        return len(self.s_list)

    def _triton_geometry(self) -> _ShellGeometry:
        if self._geometry is None:
            self._geometry = _ShellGeometry(self.s_list, self.L, self.scales)
        return self._geometry

    @torch.no_grad()
    def expand(
        self,
        I_list: Sequence[torch.Tensor],
        *,
        zsymm=1,
        force_portable: Optional[bool] = None,
    ) -> torch.Tensor:
        """Coefficients ``c[e, n, l, m]`` for one intensity vector per set.

        The same coefficients as
        :func:`~torchref.experimental.alignment.frf.data_mr.bessel_sh_expand` per set.
        On CUDA float32 the fused path groups reflections into resolution shells only,
        not into (shell, cos theta) clusters, so it evaluates the Legendre factor at
        each reflection's own polar angle and agrees with the single-set expansion to
        single precision, not bit for bit.

        Parameters
        ----------
        I_list : sequence of torch.Tensor
            ``E`` intensity vectors, ``(M_e,)`` each, in the order of ``s_list``.
        zsymm : int or sequence of int
            m-filter per set: zero every ``m`` that is not a multiple of it.
        force_portable : bool, optional
            Pin the portable path; ``None`` defers to the process-wide setting.

        Returns
        -------
        torch.Tensor
            ``(E, N_rad, L, 2L-1)`` complex, at the configured complex dtype.
        """
        if len(I_list) != len(self.s_list):
            raise ValueError(
                f"{len(I_list)} intensity vectors for {len(self.s_list)} sets"
            )
        for k, (i, n) in enumerate(zip(I_list, self.sizes)):
            if int(i.shape[0]) != n:
                raise ValueError(
                    f"set {k}: {int(i.shape[0])} intensities for {n} reflections"
                )
        zs = _as_list(zsymm, len(self.s_list), "zsymm")
        backend = select(EXPAND_BACKENDS, (I_list[0],), force_portable=force_portable)
        return run_or_degrade(EXPAND_BACKENDS, backend, False, self, list(I_list), zs)


def batched_bessel_sh_expand(
    s_list: Sequence[torch.Tensor],
    I_list: Sequence[torch.Tensor],
    *,
    L: int,
    bessel_h_scale,
    zsymm=1,
    force_portable: Optional[bool] = None,
) -> torch.Tensor:
    """One-shot :class:`ExpansionPlan` expansion: ``(E, N_rad, L, 2L-1)`` complex.

    See :class:`ExpansionPlan` for the parameters; build a plan instead when the same
    reflections will be expanded again.
    """
    plan = ExpansionPlan(s_list, L=L, bessel_h_scale=bessel_h_scale)
    return plan.expand(I_list, zsymm=zsymm, force_portable=force_portable)


@torch.no_grad()
def batched_cross_correlate_xi(
    c_obs: torch.Tensor,
    c_calc: torch.Tensor,
    obs_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``xi[b, l, m, n] = sum_r c_obs[b, r, l, n] conj(c_calc[b, r, l, m])``.

    Parameters
    ----------
    c_obs : torch.Tensor
        ``(E_obs, N_rad, L, 2L-1)`` observed-side coefficients.
    c_calc : torch.Tensor
        ``(B, N_rad, L, 2L-1)`` calculated-side coefficients, one per system.
    obs_index : torch.Tensor, optional
        ``(B,)`` integer index into ``c_obs`` per system, so several models can share
        one observed set. ``None`` pairs them one to one (``E_obs == B``).

    Returns
    -------
    torch.Tensor
        ``(B, L, 2L-1, 2L-1)`` complex, the convention of
        :func:`~torchref.experimental.alignment.frf.data_mr.cross_correlate_xi`.
    """
    obs = c_obs if obs_index is None else c_obs[obs_index]
    if obs.shape != c_calc.shape:
        raise ValueError(
            f"obs {tuple(obs.shape)} and calc {tuple(c_calc.shape)} disagree"
        )
    acc = get_complex_dtype()
    # `resolve_conj()` is load-bearing on MPS, where the batched complex matmul drops
    # a lazy conjugate bit; see `cross_correlate_xi`.
    return torch.einsum(
        "brln,brlm->blmn",
        obs.to(acc),
        torch.conj(c_calc).resolve_conj().to(acc),
    )


def rotation_map_grid(
    L: int,
    grid_sampling_deg: float = 3.0,
    fft_size: int = -1,
) -> Tuple[torch.Tensor, int]:
    """The beta grid and the (alpha, gamma) FFT size of :func:`batched_rotation_maps`.

    Returns ``(betas, N)``: ``betas`` is ``(n_beta,)`` in radians, ``b * Δ`` for
    ``b < ceil(180/Δ)`` -- the single-system engine's beta grid -- and ``N`` is the
    FFT size, ``alpha = 2 pi i / N`` and ``gamma = 2 pi k / N``.
    """
    bmax = math.ceil(180.0 / grid_sampling_deg)
    if fft_size < 0:
        fft_size = find_fft_friendly_size(2 * max(bmax, 2 * L - 1))
    if fft_size < 2 * L - 1:
        raise ValueError(f"fft_size={fft_size} must be >= 2L-1={2 * L - 1}")
    betas = torch.arange(bmax, dtype=torch.float64) * math.radians(
        grid_sampling_deg
    )  # dtype-ok: host-side grid, cast by the caller
    return betas, int(fft_size)


@torch.no_grad()
def batched_rotation_maps(
    xi: torch.Tensor,
    *,
    grid_sampling_deg: float = 3.0,
    fft_size: int = -1,
    max_batch_bytes: float = 2e9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dense rotation-function maps ``RF[b, beta, alpha, gamma]`` for a batch of ``xi``.

    Edmonds ZYZ, the single-system engine's convention: a peak ``R`` satisfies
    ``s_calc = R s_obs``. Exactly
    :func:`~torchref.experimental.alignment.frf.sitelist_ang.build_dense_map_per_beta`
    per system, with the Wigner contraction and the FFT batched; the adaptive
    resampling and the peak search are left to the caller.

    Parameters
    ----------
    xi : torch.Tensor
        ``(B, L, 2L-1, 2L-1)`` complex, from :func:`batched_cross_correlate_xi`. Only
        ``l = 0`` and even ``l`` are read: odd rows are taken to be zero, as they are
        for any Patterson.
    grid_sampling_deg : float
        beta spacing in degrees.
    fft_size : int, optional
        (alpha, gamma) grid size; default
        ``find_fft_friendly_size(2 max(n_beta, 2L-1))``.
    max_batch_bytes : float
        Bound on the complex FFT buffer; the batch is processed in chunks below it.

    Returns
    -------
    maps : torch.Tensor
        ``(B, n_beta, N, N)`` real, at ``(alpha, beta, gamma) = (2 pi i/N, betas[j],
        2 pi k/N)`` for ``maps[b, j, i, k]``.
    betas : torch.Tensor
        ``(n_beta,)`` beta values in radians, on ``xi``'s device, working float dtype.
    """
    B, L = int(xi.shape[0]), int(xi.shape[1])
    dev = xi.device
    real = (
        torch.float64 if xi.dtype == torch.complex128 else torch.float32
    )  # dtype-ok: follows the accumulator's width
    betas, N = rotation_map_grid(L, grid_sampling_deg, fft_size)
    n_beta = int(betas.shape[0])
    blocks = _wigner_d_blocks(L, betas, dev, real)
    D = 2 * L - 1
    c = L - 1
    idx = (torch.arange(-(L - 1), L, device=dev) % N).to(get_int_dtype())

    bytes_per = n_beta * N * N * xi.element_size()
    chunk = max(1, int(max_batch_bytes // max(1, bytes_per)))
    maps = torch.empty((B, n_beta, N, N), dtype=real, device=dev)
    for b0 in range(0, B, chunk):
        b1 = min(B, b0 + chunk)
        x = xi[b0:b1]
        S = torch.zeros((b1 - b0, n_beta, D, D), dtype=xi.dtype, device=dev)
        S[:, :, c, c] += x[:, 0, c, c].unsqueeze(1)
        # Odd l are zero for a Patterson (centrosymmetric), and every xi this module
        # builds has them zeroed; skipping them halves the contraction.
        for l in range(2, L, 2):
            lo, hi = c - l, c + l + 1
            S[:, :, lo:hi, lo:hi] += x[:, l, None, lo:hi, lo:hi] * blocks[l - 1][None]
        pad = torch.zeros((b1 - b0, n_beta, N, N), dtype=xi.dtype, device=dev)
        pad[:, :, idx.unsqueeze(1), idx.unsqueeze(0)] = S
        maps[b0:b1] = torch.fft.fft2(pad, dim=(-2, -1)).real
    return maps, betas.to(device=dev, dtype=get_float_dtype())
