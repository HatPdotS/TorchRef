"""
Merge a reflection dataset into another space group and report how well it merges.

:func:`merge_to_spacegroup` takes any
:class:`~torchref.io.datasets.reflection_data.ReflectionData`, generates every
source-symmetry equivalent of every usable observation (the route through P1),
maps each onto the target group's CCP4 asymmetric unit, and merges what lands
together. The accompanying :class:`MergeStats` answers whether the target
symmetry is real: when the target is higher than the source, reflections that
were independent measurements become symmetry mates, and their agreement
(Rmerge, Rmeas, CC_sym) is the evidence. When the target is the same or lower,
every merged reflection has a single observation and the R values are ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import gemmi
import torch

from torchref.base.reciprocal.hkl import get_d_spacing
from torchref.io.datasets.french_wilson import french_wilson_auto
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.symmetry import SpaceGroup, SpaceGroupLike

__all__ = ["merge_to_spacegroup", "MergeStats", "MergeShell"]


@dataclass
class MergeShell:
    """Merging statistics for one resolution shell (or overall).

    ``r_merge``, ``r_meas`` and ``cc_sym`` are ``None`` when no reflection in
    the shell has two or more observations, since agreement is then undefined.
    """

    d_max: float
    d_min: float
    n_unique: int
    n_obs: int
    r_merge: Optional[float]
    r_meas: Optional[float]
    cc_sym: Optional[float]

    @property
    def multiplicity(self) -> float:
        """Mean number of observations per merged reflection."""
        return self.n_obs / self.n_unique if self.n_unique else 0.0


@dataclass
class MergeStats:
    """Result of :func:`merge_to_spacegroup`; ``str()`` gives a table.

    Attributes
    ----------
    source, target : str
        Space-group symbols merged from and into.
    on : str
        ``"I"`` when the intensities were merged, ``"F^2"`` when only
        amplitudes were available. R values on ``F^2`` are not comparable with
        intensity R values from data processing.
    anomalous : bool
        Whether Bijvoet mates were kept apart.
    overall : MergeShell
        Statistics over all reflections.
    shells : list of MergeShell
        Per resolution shell, low to high resolution.
    n_absent_obs : int
        Source observations that land on reflections systematically absent in
        the target; they are dropped from the merge.
    absent_mean_i_over_sigma : float or None
        Their mean I/sigma(I). Clearly above zero is evidence against the
        target's screw axes or centring.
    """

    source: str
    target: str
    on: str
    anomalous: bool
    overall: MergeShell
    shells: List[MergeShell] = field(default_factory=list)
    n_absent_obs: int = 0
    absent_mean_i_over_sigma: Optional[float] = None

    def __str__(self) -> str:
        def fmt(v, spec):
            return format(v, spec) if v is not None else "-".rjust(len(format(0.0, spec)))

        head = (
            f"Merge {self.source} -> {self.target} on {self.on}"
            f"{' (anomalous)' if self.anomalous else ''}\n"
            f"{'d_max':>7} {'d_min':>6} {'n_uniq':>8} {'n_obs':>8} {'mult':>5} "
            f"{'Rmerge':>7} {'Rmeas':>7} {'CC_sym':>7}"
        )
        rows = []
        for s in self.shells + [self.overall]:
            rows.append(
                f"{s.d_max:7.2f} {s.d_min:6.2f} {s.n_unique:8d} {s.n_obs:8d} "
                f"{s.multiplicity:5.2f} {fmt(s.r_merge, '7.3f')} "
                f"{fmt(s.r_meas, '7.3f')} {fmt(s.cc_sym, '7.3f')}"
            )
        rows.insert(len(self.shells), "-" * 62)
        tail = ""
        if self.n_absent_obs:
            tail = (
                f"\n{self.n_absent_obs} observations on reflections absent in "
                f"{self.target}, mean I/sigma {self.absent_mean_i_over_sigma:.2f}"
            )
        return "\n".join([head, *rows]) + tail


def merge_to_spacegroup(
    data: ReflectionData,
    spacegroup: SpaceGroupLike,
    *,
    anomalous: Optional[bool] = None,
    n_bins: int = 10,
    seed: int = 0,
) -> Tuple[ReflectionData, MergeStats]:
    """Merge ``data`` into ``spacegroup`` and measure the agreement of symmetry mates.

    Every usable source observation is expanded under the source space group,
    each copy is mapped onto the target's CCP4 asymmetric unit, and each source
    observation contributes at most once to any target reflection -- expansion
    copies are never counted as independent measurements. Merged intensities
    are inverse-variance weighted means; amplitudes are then derived by
    French-Wilson, exactly as on load.

    Parameters
    ----------
    data : ReflectionData
        Source dataset. Only rows passing ``data.masks()`` with finite
        observations and finite, positive sigmas are used.
    spacegroup : SpaceGroupLike
        Target space group, in the same cell and setting as ``data``.
    anomalous : bool, optional
        Keep Bijvoet mates apart (acentric reflections only). Defaults to
        ``not data.friedel_merged``.
    n_bins : int, optional
        Number of resolution shells in the statistics, equal in unique
        reflections. Default 10.
    seed : int, optional
        Seed for the random half split behind ``cc_sym``, drawn from a private
        generator so the result is deterministic and the global RNG untouched.

    Returns
    -------
    merged : ReflectionData
        One row per merged reflection (two for a separated Bijvoet pair), with
        ``I``/``I_sigma`` and French-Wilson ``F``/``F_sigma``. Reflections
        French-Wilson rejects are masked. A merged reflection is free if any of
        its observations was free, and Bijvoet mates share one flag. Phases and
        figures of merit are not carried.
    stats : MergeStats
        Merging statistics; printed when ``data.verbose > 0``.

    Raises
    ------
    ValueError
        If the cell is incompatible with the target's lattice metric -- merging
        then produces R values that look like evidence against the symmetry --
        or if no observation is usable.

    Notes
    -----
    Without intensities the merge runs on ``F^2`` with ``sigma = 2 F sigma_F``,
    and the statistics say so. The computation runs on CPU; the result is
    moved to ``data.device``.
    """
    # Fresh CPU copies: .to() on the dataset's own objects would move them in place.
    target = SpaceGroup(spacegroup, device="cpu")
    source = SpaceGroup(data.spacegroup, device="cpu")
    cell = data.cell.data.detach().cpu()
    if not gemmi.UnitCell(*cell.tolist()).is_compatible_with_spacegroup(target.gemmi):
        raise ValueError(
            f"Cell {[round(c, 3) for c in cell.tolist()]} is incompatible with "
            f"{target.hm}; merging would measure the metric mismatch, not the "
            "symmetry."
        )
    if anomalous is None:
        anomalous = not data.friedel_merged

    if data.I is not None and data.I_sigma is not None:
        obs, sig, on = data.I, data.I_sigma, "I"
    else:
        obs, sig, on = data.F**2, 2.0 * data.F * data.F_sigma, "F^2"
    obs, sig = obs.detach().cpu(), sig.detach().cpu()
    use = data.masks().cpu() & torch.isfinite(obs) & torch.isfinite(sig) & (sig > 0)
    rows = torch.nonzero(use).squeeze(-1)
    if rows.numel() == 0:
        raise ValueError("No usable observations to merge.")
    obs, sig = obs[rows], sig[rows]

    # The signed index keeps a Bijvoet row on its own side of reciprocal space.
    hkl_src = data.hkl_anomalous if anomalous and data.hkl_anomalous is not None else data.hkl
    hkl_src = hkl_src.detach().cpu()[rows]
    n_src = len(rows)

    cand, cand_src, _, _ = source.equivalent_hkl(hkl_src, include_friedel=False)

    canon, _, friedel, sort_idx = target.canonicalize_hkl(cand, include_friedel=True)
    # canonicalize_hkl returns its outputs sorted; the source row must follow.
    cand_src = cand_src[sort_idx]

    absent = target.is_absent(canon)
    n_absent_obs, absent_isig = 0, None
    if bool(absent.any()):
        absent_src = torch.unique(cand_src[absent])
        n_absent_obs = int(absent_src.numel())
        absent_isig = float((obs[absent_src] / sig[absent_src]).mean())
        canon, friedel, cand_src = canon[~absent], friedel[~absent], cand_src[~absent]

    # Centric Bijvoet mates are the same reflection, so only acentric ones split.
    if anomalous:
        side = (friedel & ~target.is_centric(canon)).to(canon.dtype)
        key = torch.cat([canon, side.unsqueeze(-1)], dim=-1)
    else:
        side = torch.zeros(len(canon), dtype=canon.dtype)
        key = canon
    _, merge_id = torch.unique(key, dim=0, return_inverse=True)
    n_merge = int(merge_id.max()) + 1

    # One contribution per (merged reflection, source observation).
    pair = merge_id * n_src + cand_src
    order = torch.argsort(pair, stable=True)
    first = torch.ones(len(order), dtype=torch.bool)
    first[1:] = pair[order][1:] != pair[order][:-1]
    sel = order[first]
    gid, src = merge_id[sel], cand_src[sel]

    g_hkl = torch.empty((n_merge, 3), dtype=canon.dtype)
    g_hkl[gid] = canon[sel]
    g_side = torch.empty(n_merge, dtype=canon.dtype)
    g_side[gid] = side[sel]
    _, g_asu = torch.unique(g_hkl, dim=0, return_inverse=True)
    n_asu = int(g_asu.max()) + 1

    I_o, s_o = obs[src], sig[src]
    w = s_o.pow(-2)
    sum_w = torch.zeros(n_merge, dtype=w.dtype).index_add_(0, gid, w)
    I_m = torch.zeros_like(sum_w).index_add_(0, gid, w * I_o) / sum_w
    s_m = sum_w.rsqrt()
    n_g = torch.zeros_like(sum_w).index_add_(0, gid, torch.ones_like(w))

    d_g = get_d_spacing(g_hkl, cell.to(w.dtype)).to(w.dtype)
    stats = _merge_stats(
        I_o, gid, n_g, d_g, n_bins, torch.Generator().manual_seed(seed)
    )
    stats.source, stats.target, stats.on, stats.anomalous = (
        source.hm,
        target.hm,
        on,
        bool(anomalous),
    )
    stats.n_absent_obs, stats.absent_mean_i_over_sigma = n_absent_obs, absent_isig

    def _asu_any(flags: torch.Tensor) -> torch.Tensor:
        per_obs = flags.detach().cpu().to(torch.bool)[rows][src]
        hit = ReflectionData._group_any(per_obs, g_asu[gid], n_asu)
        return hit[g_asu]

    rfree = None
    if data.rfree_flags is not None:
        rfree = ~_asu_any(~data.rfree_flags.detach().cpu().to(torch.bool))
    validation = None
    if data.validation_flags is not None:
        validation = _asu_any(data.validation_flags)

    # The merged test set stays out of the French-Wilson prior.
    held_out = None
    if rfree is not None:
        held_out = ~rfree
    if validation is not None:
        held_out = validation if held_out is None else held_out | validation
    F_m, sF_m, keep = french_wilson_auto(
        I_m, s_m, g_hkl, d_g, target, exclude_from_fit=held_out
    )
    F_m = torch.where(keep, F_m, torch.full_like(F_m, float("nan")))

    # Signed indices, so canonicalization inside from_tensors rebuilds
    # friedel_flags / hkl_anomalous for a separated Bijvoet pair.
    hkl_out = torch.where(g_side.bool().unsqueeze(-1), -g_hkl, g_hkl)
    merged = ReflectionData.from_tensors(
        hkl_out.to(data.hkl.dtype),
        F_m,
        sF_m,
        data.cell.clone(),
        SpaceGroup(target, device=data.device),
        rfree_flags=rfree,
        device=data.device,
        verbose=data.verbose,
        friedel_merged=not anomalous,
        I=I_m,
        I_sigma=s_m,
        validation_flags=validation,
    )
    merged.source = data
    if data.verbose > 0:
        print(stats)
    return merged, stats


def _merge_stats(
    I_o: torch.Tensor,
    gid: torch.Tensor,
    n_g: torch.Tensor,
    d_g: torch.Tensor,
    n_bins: int,
    gen: torch.Generator,
) -> MergeStats:
    """Rmerge / Rmeas / CC_sym overall and per equal-count resolution shell.

    R values use the unweighted mean of each reflection's observations, as the
    conventional definitions do; only reflections with two or more
    observations contribute.
    """
    n_merge = len(n_g)
    mean_u = torch.zeros_like(n_g).index_add_(0, gid, I_o) / n_g
    absdev = (I_o - mean_u[gid]).abs()
    multi_o = n_g[gid] >= 2
    meas_w = torch.where(multi_o, (n_g / (n_g - 1).clamp(min=1)).sqrt()[gid], 0.0)

    # Random half split within each reflection: sort by (reflection, random key)
    # and send the first floor(n/2) members of each to half A.
    rnd = torch.rand(len(gid), generator=gen, dtype=I_o.dtype)
    order = torch.argsort(rnd)
    order = order[torch.argsort(gid[order], stable=True)]
    g_sorted = gid[order]
    start = torch.searchsorted(g_sorted, torch.arange(n_merge))
    pos = torch.arange(len(gid)) - start[g_sorted]
    in_a = torch.zeros(len(gid), dtype=torch.bool)
    in_a[order] = pos < (n_g[g_sorted] // 2)
    n_a = torch.zeros_like(n_g).index_add_(0, gid, in_a.to(I_o.dtype))
    half_a = torch.zeros_like(n_g).index_add_(0, gid, torch.where(in_a, I_o, 0.0))
    half_b = torch.zeros_like(n_g).index_add_(0, gid, torch.where(in_a, 0.0, I_o))
    half_a = half_a / n_a.clamp(min=1)
    half_b = half_b / (n_g - n_a).clamp(min=1)

    # Shells equal in unique reflections, low resolution first.
    rank = torch.empty(n_merge, dtype=gid.dtype)
    rank[torch.argsort(d_g, descending=True)] = torch.arange(n_merge)
    shell = (rank * n_bins) // max(n_merge, 1)

    def summarise(sel_g: torch.Tensor) -> MergeShell:
        sel_o = sel_g[gid]
        multi_g = sel_g & (n_g >= 2)
        m_o = sel_o & multi_o
        denom = float(I_o[m_o].sum())
        r_merge = float(absdev[m_o].sum()) / denom if bool(m_o.any()) and denom else None
        r_meas = (
            float((meas_w * absdev)[m_o].sum()) / denom
            if r_merge is not None
            else None
        )
        cc = None
        if int(multi_g.sum()) >= 3:
            a = half_a[multi_g] - half_a[multi_g].mean()
            b = half_b[multi_g] - half_b[multi_g].mean()
            norm = float(a.norm() * b.norm())
            cc = float((a * b).sum()) / norm if norm > 0 else None
        d_sel = d_g[sel_g]
        return MergeShell(
            d_max=float(d_sel.max()) if len(d_sel) else 0.0,
            d_min=float(d_sel.min()) if len(d_sel) else 0.0,
            n_unique=int(sel_g.sum()),
            n_obs=int(sel_o.sum()),
            r_merge=r_merge,
            r_meas=r_meas,
            cc_sym=cc,
        )

    shells = [summarise(shell == b) for b in range(n_bins) if bool((shell == b).any())]
    overall = summarise(torch.ones(n_merge, dtype=torch.bool))
    return MergeStats(
        source="", target="", on="", anomalous=False, overall=overall, shells=shells
    )
