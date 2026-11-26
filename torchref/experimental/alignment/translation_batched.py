"""Batched fast translation function: many systems and orientations at once.

:func:`~torchref.experimental.alignment.translation.prepare_candidate` and
:func:`~torchref.experimental.alignment.translation.fast_translation_function` handle
one oriented model in one crystal. This module does the same for a batch -- several
crystals, each with several candidate orientations:

* :func:`candidate_transforms` -- the normalised transforms ``G`` of one search model
  at ``R`` orientations, ``(R, S, N)``: the pipeline's FFT route per orientation, or
  one exact direct-summation call for all of them (rotating a model about its centre
  ``c`` only moves where its transform is sampled, ``F_R(s) = e^{2 pi i s.c}
  F_0(R^T s)``).
* :func:`batched_translation_maps` -- the Crowther-Blow maps from those ``G``, either
  ``(B, R, nx, ny, nz)`` on one shared grid or one ``(R, nx, ny, nz)`` tensor per
  system on its own grid. Dispatched through :data:`TF_MAP_BACKENDS`: a single Triton
  launch for the pair accumulation of the whole batch plus batched cuFFTs on CUDA
  float32, a loop over candidates in torch elsewhere.
* :func:`rotated_indices`, :func:`shared_grid_shape` -- the bookkeeping both need.
* :func:`translation_function_maps` -- the end-to-end driver from models and data.

The observed side (:class:`~torchref.experimental.alignment.translation.TranslationObs`)
is built exactly as for a single system. Forward-only.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch

from torchref.config import get_complex_dtype, get_float_dtype
from torchref.scaling import WilsonNormaliser
from torchref.utils.backends import Backend, BackendTable, run_or_degrade, select

from .translation import WILSON_N_COEFF, _grid_sizes

__all__ = [
    "TF_MAP_BACKENDS",
    "batched_translation_maps",
    "candidate_transforms",
    "rotated_indices",
    "shared_grid_shape",
    "translation_function_maps",
]

_THIS = "torchref.experimental.alignment.translation_batched"

GridShape = Tuple[int, int, int]


def rotated_indices(hkl: torch.Tensor, spacegroup) -> torch.Tensor:
    """``h R_i`` for every symmetry operation, as exact integers.

    Parameters
    ----------
    hkl : torch.Tensor
        ``(N, 3)`` Miller indices.
    spacegroup
        Supplies ``matrices`` ``(S, 3, 3)``; the row convention ``h R_i`` of
        :func:`~torchref.experimental.alignment.translation.prepare_candidate`.

    Returns
    -------
    torch.Tensor
        ``(S, N, 3)`` int32.
    """
    real = get_float_dtype()
    R = spacegroup.matrices.detach().to(device=hkl.device, dtype=real)
    h_R = torch.einsum("ne,ied->ind", hkl.to(real), R)
    return h_R.round().to(torch.int32)  # dtype-ok: the kernel reads int32 indices


def shared_grid_shape(cells: Sequence, grid_spacing_A: float) -> GridShape:
    """Per-axis maximum of each cell's FFT-friendly grid at ``grid_spacing_A``.

    Every map in a batch can then be sampled on this one fractional grid; a cell that
    needed fewer points is sampled more finely, which costs FFT size but no accuracy.
    """
    sizes = [_grid_sizes(c, grid_spacing_A) for c in cells]
    return tuple(max(s[d] for s in sizes) for d in range(3))


@torch.no_grad()
def candidate_transforms(
    model,
    obs,
    spacegroup,
    real_cell,
    rotations: torch.Tensor,
    *,
    center: Optional[torch.Tensor] = None,
    method: str = "fft",
) -> torch.Tensor:
    """Normalised transforms of one search model at ``R`` orientations.

    For each orientation, the ``G`` that
    :func:`~torchref.experimental.alignment.translation.prepare_candidate` returns for
    the model rotated by ``rotations[r]`` about ``center`` (coordinates and
    anisotropic ADPs, as :meth:`~torchref.model.Model.rotate` does) and placed in the
    crystal's P1 cell.

    Parameters
    ----------
    model : ModelFT
        The search model in its reference orientation. Not modified.
    obs : TranslationObs
        The observed side; supplies ``hkl``, ``s_mag`` and ``eps``.
    spacegroup, real_cell
        The crystal's.
    rotations : torch.Tensor
        ``(R, 3, 3)`` rotations applied to the Cartesian coordinates,
        ``x' = R (x - center) + center`` -- the ``R_recovered.T`` of
        :class:`~torchref.experimental.alignment.pipeline.MolecularReplacementPipeline`.
    center : torch.Tensor, optional
        ``(3,)`` rotation centre in Å; default the mean atom position.
    method : {"fft", "direct"}
        ``"fft"`` orients one P1 copy per rotation and calls ``prepare_candidate`` --
        the pipeline's own route, FFT structure factors at a grid of a third of the
        set's resolution. ``"direct"`` evaluates every orientation in one
        direct-summation call: exact (no ``TORCHREF_ALIASING_TOLERANCE`` blur), but
        its cost grows with atoms x orientations x reflections, so it pays only for
        small models.

    Returns
    -------
    torch.Tensor
        ``(R, S, N)`` complex at the configured complex dtype.
    """
    real = get_float_dtype()
    dev = obs.hkl.device
    rots = rotations.detach().to(device=dev, dtype=real).reshape(-1, 3, 3)
    if center is None:
        center = model.xyz().detach().to(device=dev, dtype=real).mean(dim=0)
    center = center.to(device=dev, dtype=real)
    if method == "fft":
        return _transforms_fft(model, obs, spacegroup, real_cell, rots, center)
    if method == "direct":
        return _transforms_direct(model, obs, spacegroup, real_cell, rots, center)
    raise ValueError(f"method={method!r}; expected 'fft' or 'direct'")


def _transforms_fft(model, obs, spacegroup, real_cell, rots, center):
    """One oriented P1 copy, re-oriented in place per rotation, as the pipeline does."""
    from torchref.base.targets.adp import rotate_U

    from .translation import prepare_candidate

    p1 = model.copy()
    # The grid of _prepare_translation_arrays: a third of the set's resolution / 1.5.
    p1.max_res = (1.0 / float(obs.s_mag.max())) / 1.5
    p1.cell = real_cell.clone().to(device=p1.device, dtype=p1.dtype_float)
    p1.spacegroup = "P 1"
    xyz0 = p1.xyz().detach().clone()
    u0 = p1.u().detach().clone() if bool(p1.aniso_flag.any()) else None
    c = center.to(device=xyz0.device, dtype=xyz0.dtype)
    out = []
    for R in rots:
        R = R.to(device=xyz0.device, dtype=xyz0.dtype)
        p1.xyz[:] = (xyz0 - c) @ R.T + c
        if u0 is not None:
            p1.u[:] = rotate_U(u0, R)
        out.append(prepare_candidate(p1, obs, spacegroup, real_cell).G)
    return torch.stack(out)


def _transforms_direct(model, obs, spacegroup, real_cell, rots, center):
    """Every orientation in one direct summation; see :func:`candidate_transforms`."""
    from torchref.base.direct_summation import ds_aniso, ds_iso

    real = get_float_dtype()
    cplx = get_complex_dtype()
    dev = obs.hkl.device
    xyz_i, adp_i, occ_i, A_i, B_i = (t.detach().to(dev) for t in model.get_iso())
    aniso = model.get_aniso()
    has_aniso = aniso[0] is not None and aniso[0].numel() > 0

    hkl = obs.hkl.to(device=dev, dtype=real)
    sym_R = spacegroup.matrices.detach().to(device=dev, dtype=real)
    sym_t = spacegroup.translations.detach().to(device=dev, dtype=real)
    rec = real_cell.reciprocal_basis_matrix.detach().to(device=dev, dtype=real)
    h_R = torch.einsum("ne,ied->ind", hkl, sym_R)  # (S, N, 3)
    s_cart = h_R @ rec  # (S, N, 3), Å⁻¹
    S, N = int(h_R.shape[0]), int(h_R.shape[1])
    R_n = int(rots.shape[0])

    # Row vectors: (R^T s)^T = s^T R. The direct-summation kernel forms 2 pi h.x, which
    # is the same number for a Cartesian s against Cartesian x.
    q = torch.einsum("ind,rde->rine", s_cart, rots).reshape(-1, 3).contiguous()
    smag = s_cart.norm(dim=-1).expand(R_n, S, N).reshape(-1).contiguous()
    F = ds_iso(q, smag, (xyz_i.to(real) - center).contiguous(), occ_i, adp_i, A_i, B_i)
    if has_aniso:
        xyz_a, u_a, occ_a, A_a, B_a = (t.detach().to(dev) for t in aniso)
        F = F + ds_aniso(
            q, q, (xyz_a.to(real) - center).contiguous(), occ_a, u_a, A_a, B_a
        )
    F = F.reshape(R_n, S, N).to(cplx)

    # e^{2 pi i s.c} restores the centre; e^{2 pi i h.t_i} is prepare_candidate's
    # symmetry-translation phase.
    phase = (2.0 * math.pi) * (s_cart @ center + torch.einsum("ne,ie->in", hkl, sym_t))
    G_raw = F * torch.polar(torch.ones_like(phase), phase).to(cplx).unsqueeze(0)

    s_mag = obs.s_mag.to(device=dev, dtype=real)
    eps = obs.eps.to(device=dev, dtype=real)
    I_P = (G_raw.abs() ** 2).mean(dim=1).to(real)  # (R, N)
    G = torch.empty_like(G_raw)
    for r in range(R_n):
        fit = WilsonNormaliser(
            I_P[r],
            s_mag,
            n_coeff=WILSON_N_COEFF,
            s_lo=float(s_mag.min()),
            s_hi=float(s_mag.max()),
        )
        norm = (eps * (S * fit.evaluate(s_mag).to(real))).clamp(min=1e-30).sqrt()
        G[r] = G_raw[r] / norm.to(cplx)
    return G


def _normalise_grids(grid_shape, B) -> Tuple[List[GridShape], bool]:
    """Per-system grids, and whether the caller asked for one shared grid."""
    if not isinstance(grid_shape[0], (tuple, list)):
        g = tuple(int(v) for v in grid_shape)
        return [g] * B, True
    grids = [tuple(int(v) for v in g) for g in grid_shape]
    if len(grids) != B:
        raise ValueError(f"{len(grids)} grid shapes for {B} systems")
    return grids, False


def _diag(G_list, coeff_list):
    """The ``i == j`` terms, which carry no ``t``: ``sum_h c(h) sum_i |G_i(h)|^2``."""
    return [
        (c.to(get_float_dtype()) * (G.abs() ** 2).sum(dim=1)).sum(dim=-1)  # (R_b,)
        for G, c in zip(G_list, coeff_list)
    ]


def _finish(W_list, diag_list, grids, shared):
    """``2 Re(ifftn(W)) * n + diag`` per system, batched over systems sharing a grid."""
    out = [None] * len(W_list)
    groups = {}
    for b, g in enumerate(grids):
        groups.setdefault(g, []).append(b)
    for g, members in groups.items():
        n = float(g[0] * g[1] * g[2])
        W = torch.cat([W_list[b].view(-1, *g) for b in members])
        maps = torch.fft.ifftn(W, dim=(-3, -2, -1)).real * (2.0 * n)
        k = 0
        for b in members:
            R_b = int(W_list[b].shape[0])
            out[b] = (maps[k : k + R_b] + diag_list[b].view(-1, 1, 1, 1)).to(
                get_float_dtype()
            )
            k += R_b
    if not shared:
        return out
    R_max = max(int(m.shape[0]) for m in out)
    padded = [
        torch.cat([m, m.new_zeros((R_max - m.shape[0], *m.shape[1:]))]) for m in out
    ]
    return torch.stack(padded)


def _maps_triton(G_list, hR_list, coeff_list, grids, shared):
    """One Triton launch for all candidates' pair accumulation, then batched cuFFTs."""
    from .kernels.triton_tf import launch_pair_scatter

    f32 = torch.float32  # dtype-ok: the kernel's dtype contract
    i32 = torch.int32  # dtype-ok: the kernel's index contract
    dev = G_list[0].device

    gr_parts, gi_parts, hr_parts, c_parts = [], [], [], []
    S_q, N_q, g_off, h_off, c_off, grid_q, w_off = [], [], [], [], [], [], []
    g_base = h_base = c_base = w_base = 0
    w_sizes = []
    for G, hR, c, g in zip(G_list, hR_list, coeff_list, grids):
        R_b, S_b, N_b = (int(v) for v in G.shape)
        Gc = G.detach().to(get_complex_dtype())
        gr_parts.append(Gc.real.to(f32).reshape(-1))
        gi_parts.append(Gc.imag.to(f32).reshape(-1))
        hr_parts.append(hR.to(i32).reshape(-1, 3))
        c_parts.append(c.detach().to(f32).reshape(-1))
        n_grid = g[0] * g[1] * g[2]
        for r in range(R_b):
            S_q.append(S_b)
            N_q.append(N_b)
            g_off.append(g_base + r * S_b * N_b)
            h_off.append(h_base)
            c_off.append(c_base)
            grid_q.extend(g)
            w_off.append(w_base + r * n_grid)
        w_sizes.append(R_b * n_grid)
        g_base += R_b * S_b * N_b
        h_base += S_b * N_b
        c_base += N_b
        w_base += R_b * n_grid
    if g_base >= 2**31 or h_base * 3 >= 2**31:
        raise ValueError("batch too large for int32 offsets; split it")

    def _i(v, dtype=i32):
        return torch.tensor(v, dtype=dtype, device=dev)

    W = torch.zeros(
        w_base, dtype=torch.complex64, device=dev
    )  # dtype-ok: the kernel accumulates float32 pairs
    launch_pair_scatter(
        torch.cat(gr_parts).contiguous(),
        torch.cat(gi_parts).contiguous(),
        torch.cat(hr_parts).contiguous(),
        torch.cat(c_parts).contiguous(),
        _i(S_q),
        _i(N_q),
        _i(g_off),
        _i(h_off),
        _i(c_off),
        _i(grid_q),
        _i(w_off, torch.int64),  # dtype-ok: grid offsets exceed int32 for large batches
        torch.view_as_real(W),
        S_max=max(int(h.shape[0]) for h in hR_list),
        N_max=max(int(h.shape[1]) for h in hR_list),
    )
    W_list = list(torch.split(W, w_sizes))
    W_list = [w.view(int(G.shape[0]), -1) for w, G in zip(W_list, G_list)]
    return _finish(W_list, _diag(G_list, coeff_list), grids, shared)


def _maps_portable(G_list, hR_list, coeff_list, grids, shared):
    """The single-candidate torch accumulation, looped; the reference for the kernel."""
    cplx = get_complex_dtype()
    W_list = []
    for G, hR, c, (nx, ny, nz) in zip(G_list, hR_list, coeff_list, grids):
        R_b = int(G.shape[0])
        W = torch.zeros((R_b, nx * ny * nz), dtype=cplx, device=G.device)
        h_R = hR.to(torch.int64)  # dtype-ok: flat grid index can exceed int32
        cc = c.to(cplx).view(1, -1)
        for r in range(R_b):
            Gr = G[r].to(cplx)
            for i in range(int(Gr.shape[0]) - 1):
                pair = Gr[i].conj().view(1, -1) * Gr[i + 1 :]
                dh = h_R[i + 1 :] - h_R[i : i + 1]
                flat = ((dh[..., 0] % nx) * ny + (dh[..., 1] % ny)) * nz + (
                    dh[..., 2] % nz
                )
                W[r].index_add_(0, flat.reshape(-1), (cc * pair).reshape(-1))
        W_list.append(W)
    return _finish(W_list, _diag(G_list, coeff_list), grids, shared)


def why_unavailable():
    """``None`` if the Triton translation-map kernel can run, else why it cannot."""
    from torchref.utils.backends import triton_available

    if not triton_available():
        return "triton is not importable"
    try:
        from .kernels import triton_tf  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - a failed import is a reason
        return (
            "the Triton translation kernel could not be imported "
            f"({type(exc).__name__}: {exc})"
        )
    return None


#: Dispatch for :func:`batched_translation_maps`. Both rows take
#: ``(G_list, hR_list, coeff_list, grids, shared)``. ``G`` is complex and so cannot be
#: probed against a real-dtype contract; the weights carry the working precision.
TF_MAP_BACKENDS = BackendTable(
    name="batched translation-function maps",
    backends=(
        Backend(
            name="triton",
            kernel=(_THIS, "_maps_triton", "_maps_triton"),
            device="cuda",
            dtypes=(torch.float32,),  # dtype-ok: backend capability declaration
            probes=(0,),
            probe=(_THIS, "why_unavailable"),
            expect_available="cuda",
            # Writes only a freshly zeroed buffer, so a failure leaves nothing behind.
            on_failure="degrade",
            second_order=False,
        ),
        Backend(
            name="portable",
            kernel=(_THIS, "_maps_portable", "_maps_portable"),
            expect_available="always",
            on_failure="raise",
            second_order=False,
        ),
    ),
)


@torch.no_grad()
def batched_translation_maps(
    G_list: Sequence[torch.Tensor],
    hR_list: Sequence[torch.Tensor],
    coeff_list: Sequence[torch.Tensor],
    grid_shape: Union[GridShape, Sequence[GridShape]],
    *,
    force_portable: Optional[bool] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Fast-translation-function maps for ``B`` systems x their orientations.

    For system ``b`` and candidate ``r`` the map is
    ``sum_h c_b(h) |sum_s G_{b,r,s}(h) exp(2 pi i (h R_s) . t)|^2`` at the fractional
    translations ``t = (i/nx, j/ny, k/nz)`` -- the map
    :func:`~torchref.experimental.alignment.translation.fast_translation_function`
    returns.

    Parameters
    ----------
    G_list : sequence of torch.Tensor
        Per system, ``(R_b, S_b, N_b)`` complex candidate transforms
        (:func:`candidate_transforms`).
    hR_list : sequence of torch.Tensor
        Per system, ``(S_b, N_b, 3)`` integer rotated indices (:func:`rotated_indices`).
    coeff_list : sequence of torch.Tensor
        Per system, ``(N_b,)`` weights (:attr:`TranslationObs.coeff`).
    grid_shape : (int, int, int) or sequence of them
        One ``(nx, ny, nz)`` for the whole batch (e.g. :func:`shared_grid_shape`), or
        one per system.
    force_portable : bool, optional
        Pin the torch reference path; ``None`` defers to the process-wide setting.

    Returns
    -------
    torch.Tensor or list of torch.Tensor
        With one shared grid, ``(B, R_max, nx, ny, nz)``, systems with fewer
        candidates padded with zero maps. With per-system grids, a list of
        ``(R_b, nx_b, ny_b, nz_b)``. Configured float dtype.

    Notes
    -----
    The CUDA path accumulates with float32 atomics, so repeated calls agree to float32
    rounding, not bit for bit -- as with the single-system ``index_add_``.
    """
    B = len(G_list)
    if B == 0 or len(hR_list) != B or len(coeff_list) != B:
        raise ValueError("G_list, hR_list and coeff_list must be non-empty and match")
    for G, hR, c in zip(G_list, hR_list, coeff_list):
        if (
            G.dim() != 3
            or tuple(G.shape[1:]) != tuple(hR.shape[:2])
            or c.shape[0] != G.shape[2]
        ):
            raise ValueError(
                f"inconsistent shapes: G {tuple(G.shape)}, hR {tuple(hR.shape)}, "
                f"coeff {tuple(c.shape)}"
            )
    grids, shared = _normalise_grids(grid_shape, B)
    backend = select(TF_MAP_BACKENDS, (coeff_list[0],), force_portable=force_portable)
    return run_or_degrade(
        TF_MAP_BACKENDS,
        backend,
        False,
        G_list,
        hR_list,
        coeff_list,
        grids,
        shared,
    )


def translation_function_maps(
    models: Sequence,
    datas: Sequence,
    rotations: Sequence[torch.Tensor],
    model_error_A,
    *,
    d_min: float = 4.0,
    d_max: float = 15.0,
    grid: str = "own",
    force_portable: Optional[bool] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Translation-function maps for many systems, each at several orientations.

    The per-system observed side is
    :meth:`MolecularReplacementPipeline._prepare_translation_arrays`'s: the data's
    valid reflections in ``[d_max, d_min]`` through
    :meth:`~torchref.experimental.alignment.translation.TranslationObs.build`. The
    transforms come from :func:`candidate_transforms` and the maps from
    :func:`batched_translation_maps`, on a grid a third of the set's resolution apart,
    as the pipeline samples it.

    Parameters
    ----------
    models, datas : sequence
        One search model and one data set per system.
    rotations : sequence of torch.Tensor
        Per system, ``(R_b, 3, 3)`` rotations applied to the model's Cartesian
        coordinates about its centroid.
    model_error_A : float or sequence of float
        Expected r.m.s. coordinate error per system (or shared), in Å.
    d_min, d_max : float
        Resolution window of the translation set, in Å.
    grid : {"own", "shared"}
        ``"own"``: each system on its own grid, returned as a list of
        ``(R_b, nx, ny, nz)``. ``"shared"``: one grid fine enough for every cell,
        returned as ``(B, R_max, nx, ny, nz)``.
    force_portable : bool, optional
        Pin the torch reference path for the maps.
    """
    from .rotation_search import _valid_mask
    from .translation import TranslationObs

    B = len(models)
    if B == 0 or len(datas) != B or len(rotations) != B:
        raise ValueError("models, datas and rotations must be non-empty and match")
    errs = (
        [float(model_error_A)] * B
        if isinstance(model_error_A, (int, float))
        else [float(e) for e in model_error_A]
    )
    real = get_float_dtype()
    G_list, hR_list, c_list, grids, cells, spacings = [], [], [], [], [], []
    for model, data, rots, err in zip(models, datas, rotations, errs):
        dev = data.hkl.device
        rec = data.cell.reciprocal_basis_matrix.to(device=dev, dtype=real)
        s_all = (data.hkl.to(real) @ rec).norm(dim=-1)
        keep = _valid_mask(data, dev) & (s_all <= 1.0 / d_min) & (s_all >= 1.0 / d_max)
        sig = getattr(data, "F_sigma", None)
        obs = TranslationObs.build(
            data.F[keep],
            data.hkl[keep],
            data.spacegroup,
            data.cell,
            sig_F=None if sig is None else sig[keep],
            delta_vrms_A=err,
            device=dev,
        )
        G_list.append(
            candidate_transforms(model, obs, data.spacegroup, data.cell, rots)
        )
        hR_list.append(rotated_indices(obs.hkl, data.spacegroup))
        c_list.append(obs.coeff)
        spacing = (1.0 / float(obs.s_mag.max())) / 3.0
        grids.append(_grid_sizes(data.cell, spacing))
        cells.append(data.cell)
        spacings.append(spacing)
    if grid == "shared":
        grid_shape = shared_grid_shape(cells, min(spacings))
    elif grid == "own":
        grid_shape = grids
    else:
        raise ValueError(f"grid={grid!r}; expected 'own' or 'shared'")
    return batched_translation_maps(
        G_list,
        hR_list,
        c_list,
        grid_shape,
        force_portable=force_portable,
    )
