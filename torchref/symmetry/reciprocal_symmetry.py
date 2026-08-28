"""Asymmetric-unit conventions for Miller indices.

The algorithms behind :class:`~torchref.symmetry.spacegroup.SpaceGroup`'s HKL verbs:
``expand_hkl`` (ASU -> P1), ``reduce_hkl`` (P1 -> ASU), ``complete_hkl`` (reflections
missing from a dataset, same space group) and ``canonicalize_hkl`` (CCP4 ASU
representative). All private -- call them through the space group, which is the only
public entry point.

What makes these crystallographic rather than general symmetry is the choice of
asymmetric unit: the CCP4 convention, read off gemmi's ``ReciprocalAsu`` and keyed by
Laue class. That is why they hang off
:class:`~torchref.symmetry.spacegroup.SpaceGroup` and not
:class:`~torchref.symmetry.symmetry.Symmetry`.

Miller indices transform as ``h' = h @ R = R^T @ h`` with R the *real-space* rotation;
:attr:`~torchref.symmetry.symmetry.Symmetry.reciprocal` already holds the transpose.
Translations enter as phase shifts of ``-2 pi h.t``. That sign is load-bearing and a
wrong one is invisible in P21/P212121/C2 -- see :func:`_expand_hkl` and
``tests/unit/symmetry/test_phase_convention.py``.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from torchref.config import get_float_dtype



def _expand_hkl(
    sym,
    hkl: torch.Tensor,
    include_friedel: bool = True,
    remove_absences: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand Miller indices under crystallographic symmetry (ASU -> P1).

    The low-level primitive: returns the expanded indices plus the index map and
    phase offsets needed to expand any associated per-reflection data.

    Parameters
    ----------
    sym : SpaceGroup
        The space group whose asymmetric unit convention applies.
    hkl : torch.Tensor, shape (N, 3)
        Input Miller indices (asymmetric unit).
    include_friedel : bool, default True
        Include Friedel mates (-h, -k, -l).
    remove_absences : bool, default True
        Remove systematically absent reflections.
    device : torch.device, optional
        Computation device. If None, uses hkl's device.

    Returns
    -------
    expanded_hkl : torch.Tensor, shape (M, 3), dtype=int32
        All unique expanded Miller indices.
    orig_indices : torch.Tensor, shape (M,), dtype=int64
        Index mapping expanded → original: ``F_expanded = F_orig[orig_indices]``.
    phase_shifts : torch.Tensor, shape (M,), dtype=float32
        Translation phase offsets in radians:
        ``phase_expanded = phase_orig[orig_indices] + phase_shifts``.
    """
    if device is None:
        device = hkl.device

    # Get symmetry operations
    n_ops = sym.n_ops
    recip_matrices = sym.reciprocal.matrices.to(device=device)
    translations = sym.translations.to(device=device)

    # Convert hkl to float for matrix operations
    hkl_float = hkl.to(dtype=get_float_dtype(), device=device)
    n_orig = len(hkl_float)

    # Apply all symmetry operations
    all_hkl = []
    all_phases = []

    for i in range(n_ops):
        # h' = h @ R^T
        hkl_transformed = torch.round(torch.matmul(hkl_float, recip_matrices[i].T)).to(
            torch.int32
        )
        # Phase shift from translation: -2π h·t, for h' = hR under the convention
        # F(h) = Σ_j f_j exp(+2πi h·x_j). Do NOT "simplify" the sign: the wrong sign
        # costs 4π h·t mod 2π, which is exactly zero for 2₁ screws and centring, so
        # P21/P212121/C2 cannot see it. tests/unit/symmetry/test_phase_convention.py.
        phase_shift = -2.0 * np.pi * torch.matmul(hkl_float, translations[i])

        all_hkl.append(hkl_transformed)
        all_phases.append(phase_shift)

    # Add Friedel mates if requested
    if include_friedel:
        for i in range(n_ops):
            all_hkl.append(-all_hkl[i])
            all_phases.append(-all_phases[i])

    # Stack all transformed hkl and phases
    hkl_expanded = torch.cat(all_hkl, dim=0)
    phases_expanded = torch.cat(all_phases, dim=0)

    # Remove duplicates - keep unique (h,k,l) tuples with index mapping
    hkl_np = hkl_expanded.cpu().numpy()
    phase_np = phases_expanded.cpu().numpy()

    # Build dictionary: key=(h,k,l), value=(first_occurrence_idx, phase)
    unique_dict = {}
    for idx, (h, phase) in enumerate(zip(hkl_np, phase_np)):
        key = tuple(h)
        if key not in unique_dict:
            unique_dict[key] = (idx, phase)

    # Extract unique data
    unique_indices = [v[0] for v in unique_dict.values()]
    unique_phases = [v[1] for v in unique_dict.values()]

    # Map back to original reflection index
    n_total_ops = n_ops * (2 if include_friedel else 1)
    orig_indices = [idx % n_orig for idx in unique_indices]

    # Build output tensors
    expanded_hkl = torch.tensor(
        [list(k) for k in unique_dict.keys()], dtype=torch.int32, device=device
    )
    phase_shifts = torch.tensor(unique_phases, dtype=get_float_dtype(), device=device)
    orig_idx_tensor = torch.tensor(orig_indices, dtype=torch.int64, device=device)

    if remove_absences and sym.number != 1:
        keep_mask = ~sym.is_absent(expanded_hkl)

        expanded_hkl = expanded_hkl[keep_mask]
        phase_shifts = phase_shifts[keep_mask]
        orig_idx_tensor = orig_idx_tensor[keep_mask]

    return expanded_hkl, orig_idx_tensor, phase_shifts


def _complete_hkl(
    sym,
    input_hkl: torch.Tensor,
    cell: torch.Tensor,
    d_min: float,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Complete a set of Miller indices by identifying missing reflections.

    Generates every reflection within ``d_min`` for ``sym`` (minus
    systematic absences), then maps the input onto that complete set. This does
    *not* expand symmetry -- the output stays in the input space group.

    Parameters
    ----------
    sym : SpaceGroup
        The space group whose asymmetric unit convention applies.
    input_hkl : torch.Tensor, shape (N, 3)
        Input Miller indices (may be incomplete).
    cell : torch.Tensor, shape (6,)
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    d_min : float
        High resolution limit in Angstroms.
    device : torch.device, optional
        Computation device. If None, uses input_hkl's device.

    Returns
    -------
    complete_hkl : torch.Tensor, shape (M, 3), dtype int32
        All possible Miller indices within resolution (minus systematic absences).
    input_indices : torch.Tensor, shape (M,), dtype int64
        Index mapping complete → input, or -1 where missing. Use as
        ``F_complete[~missing] = F_input[input_indices[~missing]]``.
    missing_mask : torch.Tensor, shape (M,), dtype bool
        True where reflection is missing from input.
    """
    from torchref.base.reciprocal import generate_possible_hkl

    if device is None:
        device = input_hkl.device

    # Generate all possible HKL within resolution
    all_hkl = generate_possible_hkl(cell, d_min, device=device)

    # Get symmetry operations for absence check
    if sym.number != 1:
        all_hkl = all_hkl[~sym.is_absent(all_hkl)]

    # Build lookup dictionary from input hkl to indices
    input_hkl_np = input_hkl.cpu().numpy()
    input_lookup = {}
    for idx, hkl in enumerate(input_hkl_np):
        key = tuple(hkl)
        input_lookup[key] = idx

    # Match complete set to input
    all_hkl_np = all_hkl.cpu().numpy()
    n_complete = len(all_hkl)

    input_indices = torch.full((n_complete,), -1, dtype=torch.int64, device=device)
    missing_mask = torch.ones(n_complete, dtype=torch.bool, device=device)

    for i, hkl in enumerate(all_hkl_np):
        key = tuple(hkl)
        if key in input_lookup:
            input_indices[i] = input_lookup[key]
            missing_mask[i] = False

    return all_hkl, input_indices, missing_mask


def _reduce_hkl(
    sym,
    hkl_p1: torch.Tensor,
    include_friedel: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce P1 Miller indices to the asymmetric unit of a target space group.

    The inverse of :meth:`~torchref.symmetry.spacegroup.SpaceGroup.expand_hkl`: symmetry-equivalent P1 reflections merge
    into one ASU reflection. The index map has *constant multiplicity* -- its
    second dimension is always ``n_equiv = n_ops * (2 if include_friedel else 1)``
    however many equivalents actually exist in ``hkl_p1`` -- so aggregation needs
    no variable-length ops.

    Parameters
    ----------
    sym : SpaceGroup
        The target space group.
    hkl_p1 : torch.Tensor, shape (N, 3)
        Input Miller indices in P1 (complete hemisphere).
    include_friedel : bool, default True
        If True, also consider Friedel mates when finding ASU representative.
    device : torch.device, optional
        Computation device. If None, uses hkl_p1's device.

    Returns
    -------
    hkl_asu : torch.Tensor, shape (M, 3), dtype int32
        Unique Miller indices in the asymmetric unit.
    reduction_indices : torch.Tensor, shape (M, n_equiv), dtype int64
        Indices into ``hkl_p1`` for each ASU reflection's equivalents, **-1 where
        no P1 reflection exists** -- mask or clamp before gathering, or a -1 will
        silently read the last row: ``F_asu = aggregate(F_p1[reduction_indices], dim=1)``.
    phase_shifts : torch.Tensor, shape (M, n_equiv), dtype float32
        Phase shifts to apply before aggregation; negated for Friedel mates.
    """
    if device is None:
        device = hkl_p1.device

    # Get symmetry operations
    n_ops = sym.n_ops
    recip_matrices = sym.reciprocal.matrices.to(device=device)
    translations = sym.translations.to(device=device)

    # Total number of equivalent positions per ASU reflection
    n_equiv = n_ops * (2 if include_friedel else 1)

    # Convert hkl to float for matrix operations
    hkl_float = hkl_p1.to(dtype=get_float_dtype(), device=device)
    n_p1 = len(hkl_float)

    # Build lookup from hkl tuple to index in P1 array
    hkl_p1_np = hkl_p1.cpu().numpy()
    p1_lookup = {tuple(h): idx for idx, h in enumerate(hkl_p1_np)}

    # For each P1 reflection, find its "canonical" ASU representative
    # The canonical form is the lexicographically smallest (h, k, l) among all equivalents
    def get_canonical_hkl(hkl_single):
        """Find canonical ASU representative for a single reflection."""
        equivalents = []

        for i in range(n_ops):
            # h' = h @ R^T
            hkl_trans = torch.round(torch.matmul(hkl_single, recip_matrices[i].T)).to(
                torch.int32
            )
            equivalents.append(hkl_trans)

            if include_friedel:
                equivalents.append(-hkl_trans)

        # Stack and find lexicographically smallest
        equiv_stack = torch.stack(equivalents, dim=0)

        # Sort by (h, k, l) lexicographically
        # Convert to tuple for comparison
        equiv_np = equiv_stack.cpu().numpy()
        equiv_tuples = [tuple(e) for e in equiv_np]
        canonical = min(equiv_tuples)

        return canonical

    # Map each P1 reflection to its canonical ASU representative
    p1_to_asu = {}  # maps P1 index to canonical ASU tuple
    asu_reflections = (
        {}
    )  # maps canonical ASU tuple to list of (P1_idx, phase_shift, equiv_idx)

    for p1_idx, hkl_single in enumerate(hkl_float):
        canonical = get_canonical_hkl(hkl_single)

        p1_to_asu[p1_idx] = canonical

        if canonical not in asu_reflections:
            asu_reflections[canonical] = []

        # Find which equivalent this P1 reflection corresponds to
        for equiv_idx in range(n_ops):
            R = recip_matrices[equiv_idx]
            t = translations[equiv_idx]

            hkl_trans = torch.round(torch.matmul(hkl_single, R.T)).to(torch.int32)
            # -2π h·t, same convention as expand_hkl (see the derivation there).
            phase_shift = -2.0 * np.pi * torch.matmul(hkl_single, t)

            if tuple(hkl_trans.cpu().numpy()) == canonical:
                asu_reflections[canonical].append(
                    (p1_idx, phase_shift.item(), equiv_idx)
                )
                break

            if include_friedel:
                if tuple((-hkl_trans).cpu().numpy()) == canonical:
                    # Friedel mate: phase is negated
                    asu_reflections[canonical].append(
                        (p1_idx, -phase_shift.item(), equiv_idx + n_ops)
                    )
                    break

    # Build output tensors
    asu_list = sorted(asu_reflections.keys())
    n_asu = len(asu_list)

    hkl_asu = torch.tensor(asu_list, dtype=torch.int32, device=device)
    reduction_indices = torch.full(
        (n_asu, n_equiv), -1, dtype=torch.int64, device=device
    )
    phase_shifts = torch.zeros((n_asu, n_equiv), dtype=get_float_dtype(), device=device)

    # Fill in the indices and phase shifts
    for asu_idx, asu_hkl in enumerate(asu_list):
        for p1_idx, phase, equiv_idx in asu_reflections[asu_hkl]:
            reduction_indices[asu_idx, equiv_idx] = p1_idx
            phase_shifts[asu_idx, equiv_idx] = phase

    return hkl_asu, reduction_indices, phase_shifts


def _asu_condition_vectorized(h, k, l, condition_key):
    """Vectorized CCP4 ASU membership test over numpy ``h``/``k``/``l`` arrays.

    ``condition_key`` is a condition string from
    ``gemmi.ReciprocalAsu.condition_str()``; an unrecognised one raises
    ``ValueError``, which callers catch to fall back on ``gemmi``'s own scalar check.
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map Miller indices to canonical CCP4 ASU representatives.

    Selects one representative per reflection under the standard CCP4 asymmetric
    unit convention. Runs on CPU/numpy regardless of ``device`` (the ASU lookup
    tables are numpy-backed), returning tensors on ``device``. Raises
    ``ValueError`` if any reflection has no ASU representative, which happens for
    the Friedel half of reciprocal space when ``include_friedel=False``.

    Parameters
    ----------
    sym : SpaceGroup
        The space group whose asymmetric unit convention applies.
    hkl : torch.Tensor, shape (N, 3), dtype int32
        Input Miller indices.
    include_friedel : bool, default True
        Whether Friedel mates are considered equivalent.
    device : torch.device, optional
        Computation device. If None, uses hkl's device.

    Returns
    -------
    canonical_hkl : torch.Tensor, shape (N, 3), dtype int32
        Remapped indices, sorted lexicographically by (h, k, l).
    phase_shifts : torch.Tensor, shape (N,), dtype float32
        Additive phase correction in radians.
    friedel_flags : torch.Tensor, shape (N,), dtype bool
        True where Friedel conjugation was applied.
    sort_indices : torch.Tensor, shape (N,), dtype int64
        Permutation from original to sorted order.

    Notes
    -----
    ``phase_shifts`` assumes the caller conjugates first — the contract is
    ``phi_new = torch.where(friedel_flags, -phi_old, phi_old) + phase_shifts``.
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
        empty_i = torch.empty(0, dtype=torch.int64, device=device)
        return empty_hkl, empty_f, empty_b, empty_i

    # The ASU lookup tables are numpy-backed, so the operations come across to CPU
    # regardless of where ``sym`` lives; only the returned tensors honour ``device``.
    asu = gemmi.ReciprocalAsu(sym._gemmi)
    condition_key = asu.condition_str()
    recip_mats = sym.reciprocal.matrices.detach().cpu().numpy()  # (n_ops, 3, 3)
    translations_np = sym.translations.detach().cpu().numpy()  # (n_ops, 3)
    n_ops = len(recip_mats)

    hkl_np = hkl.cpu().numpy().astype(np.int32)  # (N, 3)
    # Reciprocal-space rotation matrices are always integer-valued (0, ±1).
    recip_mats_i = np.round(recip_mats).astype(np.int32)

    # One op (+ its Friedel mate) at a time, so high-symmetry groups exit early:
    # most reflections are resolved by the first few operators.
    canonical_np = np.empty_like(hkl_np)
    op_idx = np.empty(n_refl, dtype=np.int32)
    friedel_np = np.zeros(n_refl, dtype=bool)
    remaining = np.ones(n_refl, dtype=bool)

    for i_op in range(n_ops):
        if not remaining.any():
            break
        idx = np.where(remaining)[0]
        hkl_sub = hkl_np[idx]  # (M, 3)
        R = recip_mats_i[i_op]  # (3, 3)
        equiv_sub = hkl_sub @ R.T  # (M, 3), int32 matmul — no rounding needed

        # Check non-Friedel
        h, k, l = equiv_sub[:, 0], equiv_sub[:, 1], equiv_sub[:, 2]
        try:
            in_asu_pos = _asu_condition_vectorized(h, k, l, condition_key)
        except ValueError:
            in_asu_pos = np.array(
                [asu.is_in(row.tolist()) for row in equiv_sub], dtype=bool
            )

        hit_pos = np.where(in_asu_pos)[0]
        if len(hit_pos) > 0:
            global_idx = idx[hit_pos]
            canonical_np[global_idx] = equiv_sub[hit_pos]
            op_idx[global_idx] = i_op
            remaining[global_idx] = False

        # Check Friedel mate
        if include_friedel and remaining.any():
            # Recompute idx for remaining after non-Friedel hits
            idx_f = np.where(remaining)[0]
            hkl_sub_f = hkl_np[idx_f]
            equiv_neg = -(hkl_sub_f @ R.T)

            h_n, k_n, l_n = equiv_neg[:, 0], equiv_neg[:, 1], equiv_neg[:, 2]
            try:
                in_asu_neg = _asu_condition_vectorized(h_n, k_n, l_n, condition_key)
            except ValueError:
                in_asu_neg = np.array(
                    [asu.is_in(row.tolist()) for row in equiv_neg], dtype=bool
                )

            hit_neg = np.where(in_asu_neg)[0]
            if len(hit_neg) > 0:
                global_idx_f = idx_f[hit_neg]
                canonical_np[global_idx_f] = equiv_neg[hit_neg]
                op_idx[global_idx_f] = i_op
                friedel_np[global_idx_f] = True
                remaining[global_idx_f] = False

    # ``canonical_np``/``op_idx`` are uninitialized ``np.empty`` buffers, so an
    # unmapped row would propagate garbage indices and phases. Fail loudly instead.
    if remaining.any():
        n_unmapped = int(remaining.sum())
        example = hkl_np[np.where(remaining)[0][0]].tolist()
        raise ValueError(
            f"canonicalize_hkl could not map {n_unmapped} reflection(s) to the "
            f"reciprocal ASU of space group {sym} "
            f"(include_friedel={include_friedel}); e.g. hkl={example}. With "
            f"include_friedel=False the Friedel half of reciprocal space has no "
            f"pure-rotation representative in the Laue-based CCP4 ASU."
        )

    # Sign depends on whether the row was Friedel-flipped, because the consumer
    # already negated phi for those rows: -2π h·t normally, +2π h·t for Friedel.
    # A single uniform sign is wrong for one half and invisible in P21/P212121/C2,
    # where every shift is 0 or π. tests/unit/symmetry/test_phase_convention.py.
    t_selected = translations_np[op_idx]  # (N, 3)
    friedel_sign = np.where(friedel_np, 1.0, -1.0).astype(np.float32)
    phase_shifts_np = (
        friedel_sign
        * 2.0
        * np.pi
        * np.sum(hkl_np.astype(np.float32) * t_selected, axis=1)
    ).astype(np.float32)

    # --- Convert to tensors and sort ---
    canonical_hkl = torch.tensor(canonical_np, dtype=hkl_dtype, device=device)
    phase_shifts = torch.tensor(phase_shifts_np, dtype=get_float_dtype(), device=device)
    friedel_flags = torch.tensor(friedel_np, dtype=torch.bool, device=device)

    # Lexicographic sort by (h, k, l) via composite key
    h_max = int(canonical_hkl.abs().max().item()) + 1
    base = 2 * h_max + 1
    sort_key = (
        canonical_hkl[:, 0].to(torch.int64) * base * base
        + canonical_hkl[:, 1].to(torch.int64) * base
        + canonical_hkl[:, 2].to(torch.int64)
    )
    sort_indices = torch.argsort(sort_key)

    return (
        canonical_hkl[sort_indices],
        phase_shifts[sort_indices],
        friedel_flags[sort_indices],
        sort_indices,
    )
