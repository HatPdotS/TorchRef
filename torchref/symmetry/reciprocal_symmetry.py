"""Asymmetric-unit conventions for Miller indices.

The algorithms behind :class:`~torchref.symmetry.spacegroup.SpaceGroup`'s HKL verbs:
``equivalent_hkl`` (every symmetry copy, with its source row), ``expand_hkl``
(ASU -> P1, built on it) and ``canonicalize_hkl`` (CCP4 ASU representative). All
private -- call them through the space group, which is the only public entry point.

What makes these crystallographic rather than general symmetry is the choice of
asymmetric unit: the CCP4 convention, read off gemmi's ``ReciprocalAsu`` and keyed by
Laue class. That is why they hang off
:class:`~torchref.symmetry.spacegroup.SpaceGroup` and not
:class:`~torchref.symmetry.symmetry.Symmetry`.

Miller indices transform as ``h' = h @ R = R^T @ h`` with R the *real-space* rotation;
:attr:`~torchref.symmetry.symmetry.Symmetry.reciprocal` already holds the transpose.
Translations enter as phase shifts of ``-2 pi h.t``. That sign is load-bearing and a
wrong one is invisible in P21/P212121/C2 -- see ``_equivalent_hkl`` and
``tests/unit/symmetry/test_phase_convention.py``.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch

from torchref.config import get_float_dtype, get_int_dtype
from torchref.utils.utils import first_index_per_group


def _equivalent_hkl(
    sym,
    hkl: torch.Tensor,
    include_friedel: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implement ``SpaceGroup.equivalent_hkl``, which documents the contract."""
    if device is None:
        device = hkl.device
    hkl_float = hkl.to(dtype=get_float_dtype(), device=device)
    n = len(hkl_float)

    # h' = h @ R^T, one batched matmul for all operations.
    matrices = sym.reciprocal.matrices.to(device=device, dtype=hkl_float.dtype)
    rotated = torch.einsum("oij,nj->oni", matrices, hkl_float)
    copies = torch.round(rotated).to(get_int_dtype())
    # Phase shift from translation: -2π h·t, for h' = hR under the convention
    # F(h) = Σ_j f_j exp(+2πi h·x_j). Do NOT "simplify" the sign: the wrong sign
    # costs 4π h·t mod 2π, which is exactly zero for 2₁ screws and centring, so
    # P21/P212121/C2 cannot see it. tests/unit/symmetry/test_phase_convention.py.
    translations = sym.translations.to(device=device, dtype=hkl_float.dtype)
    phase = -2.0 * np.pi * (hkl_float @ translations.T).T

    copies = copies.reshape(-1, 3)
    phase = phase.reshape(-1)
    source = torch.arange(n, dtype=get_int_dtype(), device=device).repeat(sym.n_ops)
    is_friedel = torch.zeros(len(copies), dtype=torch.bool, device=device)
    if include_friedel:
        copies = torch.cat([copies, -copies])
        phase = torch.cat([phase, -phase])
        source = torch.cat([source, source])
        is_friedel = torch.cat([is_friedel, ~is_friedel])
    return copies, source, phase, is_friedel


def _expand_hkl(
    sym,
    hkl: torch.Tensor,
    include_friedel: bool = True,
    remove_absences: bool = True,
    device: Optional[torch.device] = None,
    return_friedel: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Implement ``SpaceGroup.expand_hkl``, which documents the contract.

    Each P1 index keeps its first copy in ``_equivalent_hkl`` order, so a rotation
    copy wins over a Friedel copy and signed Bijvoet rows expand without colliding;
    two input rows reaching one index by the same kind of copy raise.
    """
    if device is None:
        device = hkl.device
    copies, source, phase, is_friedel = _equivalent_hkl(
        sym, hkl, include_friedel=include_friedel, device=device
    )

    # On CPU: torch.unique(dim=0) is not reliably supported across accelerator
    # backends, and this runs once per expansion on integer data.
    copies_cpu = copies.cpu()
    source_cpu, friedel_cpu = source.cpu(), is_friedel.cpu()
    uniq, inverse = torch.unique(copies_cpu, dim=0, return_inverse=True)
    first = first_index_per_group(inverse)

    competing = friedel_cpu == friedel_cpu[first][inverse]
    clash = competing & (source_cpu != source_cpu[first][inverse])
    if bool(clash.any()):
        i = int(torch.nonzero(clash)[0])
        raise ValueError(
            f"Input rows {int(source_cpu[first][inverse][i])} and {int(source_cpu[i])} "
            f"both expand onto {copies_cpu[i].tolist()}; {int(clash.sum())} such "
            "collisions. The input holds symmetry-equivalent rows -- merge them "
            "first, or expand anomalous data from its signed indices."
        )

    keep = torch.sort(first).values.to(device)
    expanded_hkl = copies[keep]
    orig_idx_tensor = source[keep]
    phase_shifts = phase[keep]
    friedel = is_friedel[keep]

    if remove_absences and sym.number != 1:
        keep_mask = ~sym.is_absent(expanded_hkl)
        expanded_hkl = expanded_hkl[keep_mask]
        phase_shifts = phase_shifts[keep_mask]
        orig_idx_tensor = orig_idx_tensor[keep_mask]
        friedel = friedel[keep_mask]

    if return_friedel:
        return expanded_hkl, orig_idx_tensor, phase_shifts, friedel
    return expanded_hkl, orig_idx_tensor, phase_shifts


def _asu_condition_vectorized(h, k, l, condition_key):
    """Vectorized CCP4 ASU membership test over torch ``h``/``k``/``l`` tensors.

    ``condition_key`` is a condition string from
    ``gemmi.ReciprocalAsu.condition_str()``, which holds for indices in the
    reference setting; an unrecognised one raises ``ValueError``.
    """
    # Map the 10 distinct CCP4 ASU conditions (covers all 230 space groups).
    _conditions = {
        # Laue -1 (triclinic)
        "l>0 or (l=0 and (h>0 or (h=0 and k>=0)))": lambda h, k, l: (l > 0)
        | ((l == 0) & ((h > 0) | ((h == 0) & (k >= 0)))),
        # Laue 2/m (monoclinic)
        "k>=0 and (l>0 or (l=0 and h>=0))": lambda h, k, l: (k >= 0)
        & ((l > 0) | ((l == 0) & (h >= 0))),
        # Laue mmm (orthorhombic)
        "h>=0 and k>=0 and l>=0": lambda h, k, l: (h >= 0) & (k >= 0) & (l >= 0),
        # Laue 4/m, 6/m (tetragonal, hexagonal)
        "l>=0 and ((h>=0 and k>0) or (h=0 and k=0))": lambda h, k, l: (l >= 0)
        & (((h >= 0) & (k > 0)) | ((h == 0) & (k == 0))),
        # Laue 4/mmm, 6/mmm
        "h>=k and k>=0 and l>=0": lambda h, k, l: (h >= k) & (k >= 0) & (l >= 0),
        # Laue -3 (trigonal, no mirror)
        "(h>=0 and k>0) or (h=0 and k=0 and l>=0)": lambda h, k, l: ((h >= 0) & (k > 0))
        | ((h == 0) & (k == 0) & (l >= 0)),
        # Laue -3m, P312 variant
        "h>=k and k>=0 and (k>0 or l>=0)": lambda h, k, l: (h >= k)
        & (k >= 0)
        & ((k > 0) | (l >= 0)),
        # Laue -3m, P321 variant
        "h>=k and k>=0 and (h>k or l>=0)": lambda h, k, l: (h >= k)
        & (k >= 0)
        & ((h > k) | (l >= 0)),
        # Laue m-3 (cubic)
        "h>=0 and ((l>=h and k>h) or (l=h and k=h))": lambda h, k, l: (h >= 0)
        & (((l >= h) & (k > h)) | ((l == h) & (k == h))),
        # Laue m-3m (cubic, full symmetry)
        "k>=l and l>=h and h>=0": lambda h, k, l: (k >= l) & (l >= h) & (h >= 0),
    }

    fn = _conditions.get(condition_key)
    if fn is None:
        raise ValueError(f"Unknown ASU condition: {condition_key}")
    return fn(h, k, l)


def _canonicalize_hkl(
    sym,
    hkl: torch.Tensor,
    include_friedel: bool = True,
    device: Optional[torch.device] = None,
    sort: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Implement ``SpaceGroup.canonicalize_hkl``, which documents the contract.

    Runs on CPU with torch ops whatever device ``sym`` or ``hkl`` is on. The phase
    shift is ``-2 pi h.t`` (input ``h``, translation ``t`` of the mapping operation),
    with the sign flipped on Friedel-conjugated rows.
    """
    import gemmi

    if device is None:
        device = hkl.device
    hkl_dtype = hkl.dtype

    n_refl = len(hkl)
    if n_refl == 0:
        empty_hkl = torch.empty((0, 3), dtype=hkl_dtype, device=device)
        empty_f = torch.empty(0, dtype=get_float_dtype(), device=device)
        empty_b = torch.empty(0, dtype=torch.bool, device=device)
        # dtype-ok: the int64 permutation torch.argsort returns for non-empty input
        empty_i = torch.empty(0, dtype=torch.int64, device=device) if sort else None
        return empty_hkl, empty_f, empty_b, empty_i

    # The mapping runs on CPU whatever device ``sym`` or ``hkl`` live on; only the
    # returned tensors honour ``device``. Torch rather than numpy for the per-row
    # arithmetic: the work is a handful of elementwise passes over every
    # reflection, which torch spreads over threads.
    group = sym._gemmi
    condition_key = gemmi.ReciprocalAsu(group).condition_str()
    # The condition holds in the reference setting. As gemmi's ReciprocalAsu.is_in
    # does, an index of another setting is tested as hkl @ rot, with rot the
    # setting's change of basis for Miller indices, scaled by 24 -- which the
    # homogeneous conditions ignore.
    to_reference = None if group.is_reference_setting() else group.basisop.as_hkl().rot
    # Reciprocal-space rotation matrices are always integer-valued (0, ±1).
    recip_ops = torch.round(sym.reciprocal.matrices.detach().cpu()).to(get_int_dtype())
    translations = sym.translations.detach().cpu()  # (n_ops, 3)
    n_ops = len(recip_ops)
    hkl_cpu = hkl.detach().to(device="cpu", dtype=get_int_dtype())  # (N, 3)

    def in_asu(h, k, l):
        if to_reference is not None:
            h, k, l = [
                rotate([row[i] for row in to_reference], h, k, l) for i in range(3)
            ]
        return _asu_condition_vectorized(h, k, l, condition_key)

    def rotate(row, h, k, l):
        """``row . (h, k, l)`` for one row of an integer matrix; 0 and ±1 add no
        multiply."""
        out = None
        for coef, col in zip(row, (h, k, l)):
            if coef == 0:
                continue
            term = col if coef == 1 else -col if coef == -1 else col * coef
            out = term if out is None else out + term
        return torch.zeros_like(h) if out is None else out

    # One op (+ its Friedel mate) at a time, so high-symmetry groups exit early:
    # most reflections are resolved by the first few operators. ``todo`` holds the
    # still-unmapped rows in increasing order (``None`` while that is all of them).
    canonical = torch.empty_like(hkl_cpu)
    op_idx = torch.empty(n_refl, dtype=get_int_dtype())
    friedel = torch.zeros(n_refl, dtype=torch.bool)
    todo = None

    for i_op in range(n_ops):
        if todo is not None and todo.numel() == 0:
            break
        R = recip_ops[i_op].tolist()
        sub = hkl_cpu if todo is None else hkl_cpu.index_select(0, todo)
        h, k, l = sub.unbind(1)
        eh, ek, el = (rotate(R[i], h, k, l) for i in range(3))

        hit = in_asu(eh, ek, el)
        miss = ~hit
        rows = hit.nonzero().squeeze(1)
        left = miss.nonzero().squeeze(1)
        if todo is not None:
            rows, left = todo[rows], todo[left]
        if rows.numel():
            canonical.index_copy_(0, rows, torch.stack((eh[hit], ek[hit], el[hit]), 1))
            op_idx.index_fill_(0, rows, i_op)
        todo = left

        # The Friedel mate of R h is -(R h): reuse the rotated indices.
        if include_friedel and todo.numel():
            nh, nk, nl = -eh[miss], -ek[miss], -el[miss]
            hit_f = in_asu(nh, nk, nl)
            rows = todo[hit_f]
            if rows.numel():
                canonical.index_copy_(
                    0, rows, torch.stack((nh[hit_f], nk[hit_f], nl[hit_f]), 1)
                )
                op_idx.index_fill_(0, rows, i_op)
                friedel.index_fill_(0, rows, True)
            todo = todo[~hit_f]

    # ``canonical``/``op_idx`` are uninitialized ``torch.empty`` buffers, so an
    # unmapped row would propagate garbage indices and phases. Fail loudly instead.
    if todo is not None and todo.numel():
        example = hkl_cpu[todo[0]].tolist()
        raise ValueError(
            f"canonicalize_hkl could not map {todo.numel()} reflection(s) to the "
            f"reciprocal ASU of space group {sym} "
            f"(include_friedel={include_friedel}); e.g. hkl={example}. With "
            f"include_friedel=False the Friedel half of reciprocal space has no "
            f"pure-rotation representative in the Laue-based CCP4 ASU."
        )

    # Sign depends on whether the row was Friedel-flipped, because the consumer
    # already negated phi for those rows: -2π h·t normally, +2π h·t for Friedel.
    # A single uniform sign is wrong for one half and invisible in P21/P212121/C2,
    # where every shift is 0 or π. tests/unit/symmetry/test_phase_convention.py.
    # h·t is summed left to right so the value does not depend on a backend's
    # reduction order.
    if bool(translations.any()):
        t_sel = translations.index_select(0, op_idx)
        hf = hkl_cpu.to(get_float_dtype())
        h_dot_t = (
            hf[:, 0] * t_sel[:, 0] + hf[:, 1] * t_sel[:, 1] + hf[:, 2] * t_sel[:, 2]
        )
        friedel_sign = torch.where(friedel, 1.0, -1.0).to(get_float_dtype())
        phase = friedel_sign * 2.0 * math.pi * h_dot_t
    else:
        phase = torch.zeros(n_refl, dtype=get_float_dtype())

    canonical_hkl = canonical.to(dtype=hkl_dtype, device=device)
    phase_shifts = phase.to(dtype=get_float_dtype(), device=device)
    friedel_flags = friedel.to(device=device)
    if not sort:
        return canonical_hkl, phase_shifts, friedel_flags, None

    # Lexicographic sort by (h, k, l) via composite key
    h_max = int(canonical_hkl.abs().max().item()) + 1
    base = 2 * h_max + 1
    # dtype-ok: composite sort key h*base^2+k*base+l overflows int32 for large Miller indices
    hkl64 = canonical_hkl.to(torch.int64)
    sort_key = hkl64[:, 0] * base * base + hkl64[:, 1] * base + hkl64[:, 2]
    sort_indices = torch.argsort(sort_key)

    return (
        canonical_hkl[sort_indices],
        phase_shifts[sort_indices],
        friedel_flags[sort_indices],
        sort_indices,
    )
