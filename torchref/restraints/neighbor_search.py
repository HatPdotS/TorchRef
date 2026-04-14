"""
GPU-native periodic neighbor search for VDW restraints.

Works in fractional space with periodic boundary conditions.
Avoids explicit symmetry expansion by assigning (atom, symop+offset)
entries to grid cells and using padded batched ``torch.cdist``
for distance computation.

All operations run under ``torch.no_grad()`` on whatever device
the input coordinates live on (CPU or GPU).
"""

from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

import numpy as np
import torch

from torchref.config import dtypes

if TYPE_CHECKING:
    from torchref.symmetry.cell import Cell
    from torchref.symmetry.spacegroup import SpaceGroup


# ------------------------------------------------------------------ #
# Step 1 – centroid pre-filter
# ------------------------------------------------------------------ #

def prefilter_symop_offsets(
    cell: "Cell",
    sg: "SpaceGroup",
    xyz_frac: torch.Tensor,
    cutoff: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select (symop, cell_offset) combos that could produce contacts.

    Uses the ASU centroid and molecule radius to eliminate obviously
    distant combinations.  Always includes identity (op=0, offset=0).

    Parameters
    ----------
    cell, sg : Cell, SpaceGroup
    xyz_frac : (N, 3) fractional ASU coordinates
    cutoff : float  Cartesian cutoff in Angstrom

    Returns
    -------
    op_indices : (M,) long – symop indices for each valid combo
    cell_offsets : (M, 3) long – integer cell translations
    """
    device = xyz_frac.device
    fdtype = dtypes.float

    centroid_frac = xyz_frac.mean(dim=0)
    centroid_cart = cell.fractional_to_cartesian(xyz_frac).mean(dim=0)
    xyz_cart = cell.fractional_to_cartesian(xyz_frac)
    molecule_radius = (xyz_cart - centroid_cart).norm(dim=1).max().item()
    threshold = 2.0 * molecule_radius + cutoff

    B = cell.fractional_matrix.to(device=device, dtype=fdtype)
    I_mat = torch.eye(3, dtype=fdtype, device=device)

    matrices = sg.matrices.to(device=device, dtype=fdtype)
    translations = sg.translations.to(device=device, dtype=fdtype)

    valid_ops = []
    valid_offsets = []

    for op_idx in range(sg.n_ops):
        R = matrices[op_idx]
        t = translations[op_idx]
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    offset = torch.tensor([dx, dy, dz], dtype=fdtype,
                                          device=device)
                    d_frac = (R - I_mat) @ centroid_frac + t + offset
                    d_cart = B @ d_frac
                    if d_cart.norm().item() <= threshold:
                        valid_ops.append(op_idx)
                        valid_offsets.append([dx, dy, dz])

    op_indices = torch.tensor(valid_ops, dtype=torch.long, device=device)
    cell_offsets = torch.tensor(valid_offsets, dtype=torch.long, device=device)
    return op_indices, cell_offsets


# ------------------------------------------------------------------ #
# Step 2 – vectorised image positions + grid assignment
# ------------------------------------------------------------------ #

def assign_to_grid(
    xyz_frac: torch.Tensor,
    cell: "Cell",
    sg: "SpaceGroup",
    op_indices: torch.Tensor,
    cell_offsets: torch.Tensor,
    grid_dims: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Cartesian image positions and assign to grid cells.

    Parameters
    ----------
    xyz_frac : (N, 3)
    cell : Cell
    sg : SpaceGroup
    op_indices : (M,) long
    cell_offsets : (M, 3) long
    grid_dims : (3,) long – number of grid cells per axis

    Returns
    -------
    flat_cell : (N*M,) long – flat grid cell index per entry
    atom_idx : (N*M,) long – ASU atom index
    combo_idx : (N*M,) long – index into op_indices / cell_offsets
    cart_pos : (N*M, 3) float – Cartesian positions (reused in step 4)
    """
    device = xyz_frac.device
    fdtype = dtypes.float
    N = xyz_frac.shape[0]
    M = op_indices.shape[0]

    R_sel = sg.matrices[op_indices].to(dtype=fdtype)        # (M, 3, 3)
    t_sel = sg.translations[op_indices].to(dtype=fdtype)    # (M, 3)
    offs = cell_offsets.to(dtype=fdtype)                     # (M, 3)

    # (N, M, 3) = einsum over symops applied to each atom
    frac_images = (
        torch.einsum("mij,nj->nmi", R_sel, xyz_frac.to(fdtype))
        + t_sel[None, :, :]
        + offs[None, :, :]
    )

    # Cartesian positions (stored for reuse)
    cart_pos = cell.fractional_to_cartesian(
        frac_images.reshape(-1, 3)
    )  # (N*M, 3)

    # Wrap to [0, 1) for grid assignment
    frac_wrapped = frac_images % 1.0
    gd = grid_dims.to(device=device, dtype=fdtype)
    cell_ijk = (frac_wrapped * gd[None, None, :]).long()
    cell_ijk = cell_ijk.clamp(
        min=torch.zeros(3, dtype=torch.long, device=device),
        max=(grid_dims - 1).to(device),
    )

    gy, gz = grid_dims[1].item(), grid_dims[2].item()
    flat_cell = (
        cell_ijk[..., 0] * (gy * gz)
        + cell_ijk[..., 1] * gz
        + cell_ijk[..., 2]
    ).reshape(-1)  # (N*M,)

    atom_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, M).reshape(-1)
    combo_idx = torch.arange(M, device=device).unsqueeze(0).expand(N, M).reshape(-1)

    return flat_cell, atom_idx, combo_idx, cart_pos


# ------------------------------------------------------------------ #
# Step 3 – sort into cell list (CSR)
# ------------------------------------------------------------------ #

def build_cell_list(
    flat_cell: torch.Tensor,
    n_grid_total: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort entries by grid cell and build CSR boundary arrays.

    Returns
    -------
    sort_order : (E,) long
    unique_cells : (C,) long – occupied cell indices
    starts : (C+1,) long – CSR boundaries into sorted arrays
    cell_lookup : (n_grid_total,) long – maps flat cell → index in
        unique_cells, or -1 if empty.
    """
    device = flat_cell.device
    sort_order = flat_cell.argsort()
    sorted_cells = flat_cell[sort_order]

    unique_cells, counts = torch.unique_consecutive(
        sorted_cells, return_counts=True
    )
    starts = torch.zeros(len(unique_cells) + 1, dtype=torch.long, device=device)
    starts[1:] = counts.cumsum(0)

    cell_lookup = torch.full(
        (n_grid_total,), -1, dtype=torch.long, device=device
    )
    cell_lookup[unique_cells] = torch.arange(
        len(unique_cells), dtype=torch.long, device=device
    )

    return sort_order, unique_cells, starts, cell_lookup


# ------------------------------------------------------------------ #
# Step 4 – padded batched cdist over 27 neighbor offsets
# ------------------------------------------------------------------ #

_NEIGHBOR_OFFSETS_27 = None
_NEIGHBOR_OFFSETS_14 = None


def _get_neighbor_offsets(device: torch.device) -> torch.Tensor:
    """Return (27, 3) tensor of all neighbor offsets including self."""
    global _NEIGHBOR_OFFSETS_27
    if _NEIGHBOR_OFFSETS_27 is None or _NEIGHBOR_OFFSETS_27.device != device:
        offsets = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    offsets.append([dx, dy, dz])
        _NEIGHBOR_OFFSETS_27 = torch.tensor(offsets, dtype=torch.long, device=device)
    return _NEIGHBOR_OFFSETS_27


def _get_canonical_offsets_14(device: torch.device) -> torch.Tensor:
    """Return (14, 3) lex-positive half of the 27-offset cube, including self.

    For each mirror pair ``(d, -d)`` in the 26 non-zero offsets, keep the
    direction whose first non-zero component is positive. Plus the (0,0,0)
    self-offset. Processing only this half covers every unique cell pair
    exactly once, with the symmetric partner reached by swapping source
    and target indices. Net cdist work is ~half of the 27-offset path.
    """
    global _NEIGHBOR_OFFSETS_14
    if _NEIGHBOR_OFFSETS_14 is None or _NEIGHBOR_OFFSETS_14.device != device:
        offsets = [[0, 0, 0]]
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    if (dx, dy, dz) == (0, 0, 0):
                        continue
                    # Lex-positive: first non-zero component is positive.
                    if dx > 0:
                        offsets.append([dx, dy, dz])
                    elif dx == 0 and dy > 0:
                        offsets.append([dx, dy, dz])
                    elif dx == 0 and dy == 0 and dz > 0:
                        offsets.append([dx, dy, dz])
        assert len(offsets) == 14, f"expected 14 canonical offsets, got {len(offsets)}"
        _NEIGHBOR_OFFSETS_14 = torch.tensor(
            offsets, dtype=torch.long, device=device
        )
    return _NEIGHBOR_OFFSETS_14


def _build_padded_cells(
    cart_sorted: torch.Tensor,
    starts: torch.Tensor,
    atom_idx_sorted: torch.Tensor,
    combo_idx_sorted: torch.Tensor,
    identity_combo: int,
    max_per_cell: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one padded ``(C, max_per_cell, 3)`` tensor + per-slot masks.

    Precomputes everything we need for the cdist loop: positions padded
    to a fixed row length, a validity mask (``True`` for real atoms,
    ``False`` for padding), and an ASU-membership mask (``True`` for
    entries whose combo is the identity).

    Returns
    -------
    padded_xyz : (C, max_per_cell, 3) float, padding filled with ``inf``
    valid_mask : (C, max_per_cell) bool, True for real entries
    asu_mask   : (C, max_per_cell) bool, True for identity-combo entries
    """
    device = cart_sorted.device
    C = starts.shape[0] - 1
    cell_starts = starts[:-1]             # (C,)
    counts = starts[1:] - cell_starts     # (C,)

    # (max_per_cell,) running index inside each row
    col = torch.arange(max_per_cell, device=device)

    # (C, max_per_cell) boolean mask of valid entries.
    valid_mask = col.unsqueeze(0) < counts.unsqueeze(1)

    # (C, max_per_cell) global index into the sorted CSR arrays, safe for
    # padding slots (clamped to last valid entry; those slots are masked out).
    gidx = cell_starts.unsqueeze(1) + col.unsqueeze(0)
    gidx = gidx.clamp(max=cart_sorted.shape[0] - 1)

    padded_xyz = torch.full(
        (C, max_per_cell, 3),
        float("inf"),
        dtype=cart_sorted.dtype,
        device=device,
    )
    padded_xyz[valid_mask] = cart_sorted[gidx[valid_mask]]

    # ASU mask: True iff entry is a real one AND its combo is identity.
    is_asu_entry = combo_idx_sorted == identity_combo
    asu_gather = torch.zeros_like(valid_mask)
    asu_gather[valid_mask] = is_asu_entry[gidx[valid_mask]]

    return padded_xyz, valid_mask, asu_gather


def _gather_padded(
    cart_sorted: torch.Tensor,
    starts: torch.Tensor,
    cell_indices: torch.Tensor,
    max_per_cell: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather Cartesian positions into a padded (batch, max_per_cell, 3) tensor.

    Parameters
    ----------
    cart_sorted : (E, 3) Cartesian positions sorted by cell
    starts : (C+1,) CSR boundaries
    cell_indices : (B,) which occupied-cell indices to gather
    max_per_cell : padding size

    Returns
    -------
    padded : (B, max_per_cell, 3) – padded with inf
    counts : (B,) – actual atom count per cell
    """
    device = cart_sorted.device
    B = cell_indices.shape[0]
    padded = torch.full(
        (B, max_per_cell, 3), float("inf"), dtype=cart_sorted.dtype, device=device
    )
    cell_starts = starts[cell_indices]
    cell_ends = starts[cell_indices + 1]
    counts = cell_ends - cell_starts

    # Vectorised fill: build flat indices for all entries across all cells
    # within_cell_pos[k] = position within its cell (0, 1, 2, ...)
    max_count = counts.max().item()
    arange = torch.arange(max_count, device=device)
    # (B, max_count) mask of valid positions
    valid = arange.unsqueeze(0) < counts.unsqueeze(1)
    # Global source indices into cart_sorted
    src_idx = cell_starts.unsqueeze(1) + arange.unsqueeze(0)  # (B, max_count)
    src_idx = src_idx.clamp(max=len(cart_sorted) - 1)

    # Scatter into padded tensor
    padded[:, :max_count, :][valid] = cart_sorted[src_idx[valid]]

    return padded, counts


def find_pairs_periodic_grid(
    cart_sorted: torch.Tensor,
    atom_idx_sorted: torch.Tensor,
    combo_idx_sorted: torch.Tensor,
    unique_cells: torch.Tensor,
    starts: torch.Tensor,
    cell_lookup: torch.Tensor,
    grid_dims: torch.Tensor,
    cutoff: float,
    identity_combo: int,
    chunk_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find all ASU-vs-everything pairs within cutoff using periodic grid.

    Fully vectorized: 27 neighbor offsets × chunked batched cdist.
    No Python loops over individual pairs.

    Parameters
    ----------
    cart_sorted : (E, 3) sorted Cartesian positions
    atom_idx_sorted : (E,) ASU atom index per entry
    combo_idx_sorted : (E,) combo index per entry
    unique_cells : (C,) occupied cell flat indices
    starts : (C+1,) CSR boundaries
    cell_lookup : (G,) flat cell → occupied index or -1
    grid_dims : (3,) grid dimensions
    cutoff : float
    identity_combo : int – combo index for identity (op=0, offset=0)
    chunk_size : int – cell pairs per cdist batch

    Returns
    -------
    pair_atom_i : (P,) ASU atom index (always from identity/ASU)
    pair_atom_j : (P,) ASU atom index (mate source)
    pair_combo_j : (P,) combo index for atom j
    """
    device = cart_sorted.device
    cutoff_sq = cutoff * cutoff
    neighbor_offsets = _get_neighbor_offsets(device)

    gy = grid_dims[1].item()
    gz = grid_dims[2].item()

    n_occupied = len(unique_cells)
    counts = starts[1:] - starts[:-1]
    max_per_cell = counts.max().item()

    # Decode unique_cells back to (i, j, k)
    cell_ijk = torch.stack([
        unique_cells // (gy * gz),
        (unique_cells % (gy * gz)) // gz,
        unique_cells % gz,
    ], dim=1)  # (C, 3)

    # Precompute: which cells have ASU entries?
    is_asu = combo_idx_sorted == identity_combo
    # Map each sorted entry to its occupied cell index (vectorised)
    cell_sizes = starts[1:] - starts[:-1]
    entry_cell_idx = torch.repeat_interleave(
        torch.arange(n_occupied, device=device), cell_sizes
    )
    # Scatter-or: if any entry in a cell is ASU, mark the cell
    has_asu_per_cell = torch.zeros(n_occupied, dtype=torch.long, device=device)
    has_asu_per_cell.scatter_add_(0, entry_cell_idx, is_asu.long())
    has_asu_per_cell = has_asu_per_cell > 0

    all_pair_atom_i = []
    all_pair_atom_j = []
    all_pair_combo_j = []

    for offset_idx in range(27):
        d = neighbor_offsets[offset_idx]  # (3,)
        is_self_offset = (d == 0).all().item()

        # Neighbor cell indices with periodic wrapping
        nb_ijk = (cell_ijk + d[None, :]) % grid_dims[None, :]
        nb_flat = nb_ijk[:, 0] * (gy * gz) + nb_ijk[:, 1] * gz + nb_ijk[:, 2]
        nb_occ_idx = cell_lookup[nb_flat]  # (C,) -1 if empty

        active = (nb_occ_idx >= 0) & has_asu_per_cell
        active_idx = active.nonzero(as_tuple=True)[0]
        active_nb = nb_occ_idx[active_idx]

        if len(active_idx) == 0:
            continue

        # Process in chunks
        for cs in range(0, len(active_idx), chunk_size):
            ce = min(cs + chunk_size, len(active_idx))
            c_batch = active_idx[cs:ce]
            nb_batch = active_nb[cs:ce]
            B = len(c_batch)

            # Gather positions into padded tensors
            src_padded, src_counts = _gather_padded(
                cart_sorted, starts, c_batch, max_per_cell
            )
            nb_padded, nb_counts = _gather_padded(
                cart_sorted, starts, nb_batch, max_per_cell
            )

            # Batched cdist: (B, max_per_cell, max_per_cell)
            dists = torch.cdist(src_padded, nb_padded)
            within = dists < cutoff  # inf padding is never < cutoff

            # Get all hit indices: (batch, local_i, local_j)
            b_idx, local_i, local_j = within.nonzero(as_tuple=True)

            if len(b_idx) == 0:
                continue

            # Map local indices to global sorted indices (vectorised)
            c_occ = c_batch[b_idx]              # occupied cell idx for source
            nb_occ = nb_batch[b_idx]             # occupied cell idx for neighbor
            global_i = starts[c_occ] + local_i   # global sorted index
            global_j = starts[nb_occ] + local_j

            # Bounds check (padding entries)
            valid = (global_i < starts[c_occ + 1]) & (global_j < starts[nb_occ + 1])

            # Source must be ASU
            valid = valid & is_asu[global_i]

            # Dedup: for intra-ASU pairs (both identity combo), only keep
            # atom_i < atom_j to avoid counting (A,B) and (B,A).
            # For symmetry pairs (j is not identity), no dedup needed
            # since only one direction has an ASU source.
            ai_temp = atom_idx_sorted[global_i]
            aj_temp = atom_idx_sorted[global_j]
            cj_temp = combo_idx_sorted[global_j]
            both_asu = is_asu[global_j]  # global_i is always ASU
            valid = valid & (~both_asu | (ai_temp < aj_temp))

            # Apply validity mask
            global_i = global_i[valid]
            global_j = global_j[valid]

            ai = atom_idx_sorted[global_i]
            aj = atom_idx_sorted[global_j]
            cj = combo_idx_sorted[global_j]

            # Remove true self-pairs (same atom, identity combo)
            not_self = ~((ai == aj) & (cj == identity_combo))
            ai = ai[not_self]
            aj = aj[not_self]
            cj = cj[not_self]

            if len(ai) > 0:
                all_pair_atom_i.append(ai)
                all_pair_atom_j.append(aj)
                all_pair_combo_j.append(cj)

    if not all_pair_atom_i:
        empty = torch.tensor([], dtype=torch.long, device=device)
        return empty, empty, empty

    return (
        torch.cat(all_pair_atom_i),
        torch.cat(all_pair_atom_j),
        torch.cat(all_pair_combo_j),
    )


def find_pairs_periodic_grid_v2(
    cart_sorted: torch.Tensor,
    atom_idx_sorted: torch.Tensor,
    combo_idx_sorted: torch.Tensor,
    unique_cells: torch.Tensor,
    starts: torch.Tensor,
    cell_lookup: torch.Tensor,
    grid_dims: torch.Tensor,
    cutoff: float,
    identity_combo: int,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimised replacement for :func:`find_pairs_periodic_grid`.

    Three things changed vs the original:

    1. **14 canonical offsets** (self + 13 lex-positive). For cell pairs
       where both cells contain ASU atoms, the original algorithm finds
       each pair twice (once via ``+d`` and once via ``-d``) and dedups at
       the end — pure waste. Processing only half the offsets eliminates
       the redundant cdist work. We still cover every ASU-containing pair
       by allowing the "target is ASU" case via an index swap on emission.

    2. **Precomputed padded tensor.** ``padded_xyz``, ``valid_mask`` and
       ``asu_mask`` are built once (not per-chunk per-offset as in the
       original). Per-cell metadata is also precomputed: ``cell_has_asu``,
       and the flat index offsets for mapping local ``(row, col)`` hits
       back to global CSR indices.

    3. **Use torch.cdist** (matmul-based internally) instead of broadcast
       subtract so the intermediate tensor stays cache-friendly. Chunking
       is kept but with a larger default (``512``) so we pay less Python
       overhead without blowing the cache.

    Correctness invariant: within a hit ``(src_atom, tgt_atom)`` we emit
    a pair iff ``is_asu[src] | is_asu[tgt]``, and we canonicalise so the
    ASU atom is always on the ``i`` side. For intra-ASU pairs we further
    enforce ``atom_i < atom_j`` after the swap so both-ASU pairs emit in
    a single canonical order (matching the current v1 invariant used by
    the downstream dedup).

    Parameters and return signature match
    :func:`find_pairs_periodic_grid`.
    """
    device = cart_sorted.device
    offsets14 = _get_canonical_offsets_14(device)

    # Device-adaptive chunk size. GPU: single-cdist (kernel launches
    # dominate at small chunks). CPU: cache-sized (~9 MB per cdist tile
    # for max_per_cell=135). Override via ``chunk_size`` for tuning.
    if chunk_size is None:
        chunk_size = 1_000_000 if device.type == "cuda" else 128

    gy = grid_dims[1].item()
    gz = grid_dims[2].item()

    # ---- Precompute ----
    counts = starts[1:] - starts[:-1]
    max_per_cell = counts.max().item()

    padded_xyz, valid_mask, asu_mask = _build_padded_cells(
        cart_sorted=cart_sorted,
        starts=starts,
        atom_idx_sorted=atom_idx_sorted,
        combo_idx_sorted=combo_idx_sorted,
        identity_combo=identity_combo,
        max_per_cell=max_per_cell,
    )
    cell_has_asu = asu_mask.any(dim=1)                  # (C,) bool

    cell_ijk = torch.stack([
        unique_cells // (gy * gz),
        (unique_cells % (gy * gz)) // gz,
        unique_cells % gz,
    ], dim=1)                                           # (C, 3)

    cell_base = starts[:-1]                             # (C,)

    all_pair_atom_i: List[torch.Tensor] = []
    all_pair_atom_j: List[torch.Tensor] = []
    all_pair_combo_j: List[torch.Tensor] = []

    # ---- Offset sweep (14 canonical) ----
    for offset_idx in range(offsets14.shape[0]):
        d = offsets14[offset_idx]
        is_self_offset = bool((d == 0).all().item())

        nb_ijk = (cell_ijk + d[None, :]) % grid_dims[None, :]
        nb_flat = (
            nb_ijk[:, 0] * (gy * gz)
            + nb_ijk[:, 1] * gz
            + nb_ijk[:, 2]
        )
        nb_occ_idx = cell_lookup[nb_flat]               # (C,) long, -1 empty

        has_nb = nb_occ_idx >= 0
        nb_occ_safe = nb_occ_idx.clamp(min=0)
        nb_has_asu = cell_has_asu[nb_occ_safe] & has_nb
        active = has_nb & (cell_has_asu | nb_has_asu)
        if not active.any().item():
            continue

        active_src_idx = active.nonzero(as_tuple=True)[0]  # (B_total,)
        active_nb_idx = nb_occ_idx[active_src_idx]         # (B_total,)
        B_total = active_src_idx.shape[0]

        # Chunk to keep cdist tiles cache-friendly on CPU without adding
        # more than a handful of Python iterations per offset.
        for cs in range(0, B_total, chunk_size):
            ce = min(cs + chunk_size, B_total)
            src_cells_chunk = active_src_idx[cs:ce]       # (B,)
            nb_cells_chunk = active_nb_idx[cs:ce]

            src_padded = padded_xyz[src_cells_chunk]      # (B, m, 3)
            nb_padded = padded_xyz[nb_cells_chunk]        # (B, m, 3)
            src_valid = valid_mask[src_cells_chunk]       # (B, m)
            nb_valid = valid_mask[nb_cells_chunk]         # (B, m)
            src_asu = asu_mask[src_cells_chunk]           # (B, m)
            nb_asu = asu_mask[nb_cells_chunk]             # (B, m)

            # Distance matrix via matmul-based cdist. Padding is ``inf``
            # so padded slots never compare within cutoff.
            dists = torch.cdist(src_padded, nb_padded)    # (B, m, m)
            hits = (
                (dists < cutoff)
                & src_valid.unsqueeze(2)
                & nb_valid.unsqueeze(1)
            )

            # Either side must be ASU.
            atom_is_asu_either = src_asu.unsqueeze(2) | nb_asu.unsqueeze(1)
            hits = hits & atom_is_asu_either

            if is_self_offset:
                # Kill diagonal; atom-order canonicalisation is done
                # globally after emission (not via local r<c here, because
                # local entry order within a cell need not match atom id
                # order).
                m = hits.shape[1]
                row = torch.arange(m, device=device)
                not_diag = row.view(1, m, 1) != row.view(1, 1, m)
                hits = hits & not_diag

            if not hits.any().item():
                continue

            b_idx, li, lj = hits.nonzero(as_tuple=True)
            if b_idx.numel() == 0:
                continue

            src_cells_hit = src_cells_chunk[b_idx]        # (P,)
            nb_cells_hit = nb_cells_chunk[b_idx]          # (P,)
            g_src = cell_base[src_cells_hit] + li         # (P,)
            g_nb = cell_base[nb_cells_hit] + lj           # (P,)

            src_is_asu_pair = asu_mask[src_cells_hit, li]  # (P,)
            nb_is_asu_pair = asu_mask[nb_cells_hit, lj]    # (P,)

            # Canonicalise: ASU on side i. If only target is ASU swap;
            # if both are ASU, still make sure ASU-sided sort is
            # well-defined by the later atom_i < atom_j rule below.
            swap_target_asu_only = nb_is_asu_pair & ~src_is_asu_pair
            g_i = torch.where(swap_target_asu_only, g_nb, g_src)
            g_j = torch.where(swap_target_asu_only, g_src, g_nb)

            ai = atom_idx_sorted[g_i]
            aj = atom_idx_sorted[g_j]
            ci = combo_idx_sorted[g_i]
            cj = combo_idx_sorted[g_j]

            # Intra-ASU pairs (both identity combo): enforce ai < aj so
            # (a,b) and (b,a) canonicalise to the same output, matching
            # the v1 upper-triangle convention.
            both_asu = (ci == identity_combo) & (cj == identity_combo)
            swap_intra = both_asu & (ai > aj)
            tmp_a = torch.where(swap_intra, aj, ai)
            aj = torch.where(swap_intra, ai, aj)
            ai = tmp_a
            # cj for intra-ASU pair is always identity, unchanged by swap.

            # Drop true self-pairs (same atom, identity combo on the j side).
            not_self = ~((ai == aj) & (cj == identity_combo))
            if not bool(not_self.all().item()):
                ai = ai[not_self]
                aj = aj[not_self]
                cj = cj[not_self]

            if ai.numel() > 0:
                all_pair_atom_i.append(ai)
                all_pair_atom_j.append(aj)
                all_pair_combo_j.append(cj)

    if not all_pair_atom_i:
        empty = torch.tensor([], dtype=torch.long, device=device)
        return empty, empty, empty

    return (
        torch.cat(all_pair_atom_i),
        torch.cat(all_pair_atom_j),
        torch.cat(all_pair_combo_j),
    )


# ------------------------------------------------------------------ #
# Step 5 – filtering
# ------------------------------------------------------------------ #

def exclusion_set_to_hash(
    exclusion_set: Set[Tuple[int, int]],
    max_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert Python exclusion set to sorted hash tensor.

    Hash: min(i,j) * max_idx + max(i,j), sorted for searchsorted.
    """
    if not exclusion_set:
        return torch.tensor([], dtype=torch.long, device=device)
    arr = np.array(list(exclusion_set), dtype=np.int64)
    hashes = arr[:, 0] * max_idx + arr[:, 1]  # already (min, max)
    hashes.sort()
    return torch.tensor(hashes, dtype=torch.long, device=device)


def filter_pairs(
    pair_atom_i: torch.Tensor,
    pair_atom_j: torch.Tensor,
    pair_combo_j: torch.Tensor,
    identity_combo: int,
    excl_hash: torch.Tensor,
    max_idx: int,
    pdb,
    inter_residue_only: bool = True,
) -> torch.Tensor:
    """Apply exclusion, residue, and altloc filters. Returns keep mask."""
    device = pair_atom_i.device
    N = len(pair_atom_i)
    keep = torch.ones(N, dtype=torch.bool, device=device)

    is_intra_asu = pair_combo_j == identity_combo

    # Bonded exclusions (1-2, 1-3, 1-4) – intra-ASU only
    if len(excl_hash) > 0 and is_intra_asu.any():
        norm_i = torch.minimum(pair_atom_i, pair_atom_j)
        norm_j = torch.maximum(pair_atom_i, pair_atom_j)
        pair_hash = norm_i * max_idx + norm_j
        # searchsorted: check if hash exists in sorted excl_hash
        ins = torch.searchsorted(excl_hash, pair_hash)
        ins = ins.clamp(max=len(excl_hash) - 1)
        is_excluded = excl_hash[ins] == pair_hash
        keep &= ~(is_excluded & is_intra_asu)

    # Same-residue filter – intra-ASU only
    if inter_residue_only:
        chainid = pdb["chainid"].values
        resseq = pdb["resseq"].values
        ai_np = pair_atom_i.cpu().numpy()
        aj_np = pair_atom_j.cpu().numpy()
        same_res = (
            (chainid[ai_np] == chainid[aj_np])
            & (resseq[ai_np] == resseq[aj_np])
        )
        same_res_t = torch.tensor(same_res, dtype=torch.bool, device=device)
        keep &= ~(same_res_t & is_intra_asu)

    # Altloc compatibility – intra-ASU only
    if "altloc" in pdb.columns:
        altloc = pdb["altloc"].values.astype(str)
        altloc = np.where(np.isin(altloc, ["", " "]), " ", altloc)
        ai_np = pair_atom_i.cpu().numpy()
        aj_np = pair_atom_j.cpu().numpy()
        alt_i = altloc[ai_np]
        alt_j = altloc[aj_np]
        incompat = (alt_i != " ") & (alt_j != " ") & (alt_i != alt_j)
        incompat_t = torch.tensor(incompat, dtype=torch.bool, device=device)
        keep &= ~(incompat_t & is_intra_asu)

    return keep


# ------------------------------------------------------------------ #
# Orchestrator
# ------------------------------------------------------------------ #

@torch.no_grad()
def build_vdw_restraints_gpu(
    xyz_fn,
    vdw_radii_fn,
    cell: "Cell",
    sg: "SpaceGroup",
    pdb,
    exclusion_set: Set[Tuple[int, int]],
    cutoff: float = 5.0,
    sigma: float = 0.2,
    inter_residue_only: bool = True,
    verbose: int = 0,
) -> Dict[str, torch.Tensor]:
    """Build VDW restraints using GPU-native periodic grid search.

    Parameters
    ----------
    xyz_fn : callable  returns (N, 3) Cartesian coordinates
    vdw_radii_fn : callable  returns (N,) VDW radii
    cell : Cell
    sg : SpaceGroup
    pdb : DataFrame
    exclusion_set : set of (int, int) bonded exclusion pairs
    cutoff, sigma : float
    inter_residue_only : bool
    verbose : int

    Returns
    -------
    dict with keys: indices, min_distances, sigmas, symop_indices, cell_offsets
    """
    from torchref.symmetry.spacegroup import SpaceGroup as SG

    xyz = xyz_fn()
    device = xyz.device
    fdtype = dtypes.float
    n_asu = xyz.shape[0]

    if not isinstance(sg, SG):
        sg = SG(sg)

    empty_result = {
        "indices": torch.zeros(0, 2, dtype=torch.long, device=device),
        "min_distances": torch.zeros(0, dtype=torch.float32, device=device),
        "sigmas": torch.zeros(0, dtype=torch.float32, device=device),
        "symop_indices": torch.zeros(0, dtype=torch.long, device=device),
        "cell_offsets": torch.zeros(0, 3, dtype=torch.long, device=device),
    }

    # Step 1: prefilter symop combos
    xyz_frac = cell.cartesian_to_fractional(xyz.detach().to(fdtype))
    op_indices, cell_offsets_valid = prefilter_symop_offsets(
        cell, sg, xyz_frac, cutoff
    )
    M = len(op_indices)

    if verbose > 0:
        print(f"  Symmetry expansion: {M} valid (symop, offset) combos")

    # Find the identity combo index
    is_identity = (
        (op_indices == 0)
        & (cell_offsets_valid == 0).all(dim=1)
    )
    identity_indices = is_identity.nonzero(as_tuple=True)[0]
    if len(identity_indices) == 0:
        # Identity not in valid combos — should not happen, but add it
        op_indices = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device), op_indices
        ])
        cell_offsets_valid = torch.cat([
            torch.zeros(1, 3, dtype=torch.long, device=device), cell_offsets_valid
        ])
        identity_combo = 0
        M = len(op_indices)
    else:
        identity_combo = identity_indices[0].item()

    # Step 2: compute image positions + assign to grid
    # Grid dims: at least 1 cell per cutoff along each axis
    cell_lengths = torch.tensor([
        cell.a.item(), cell.b.item(), cell.c.item()
    ], dtype=fdtype, device=device)
    grid_dims = torch.clamp(
        (cell_lengths / cutoff).long(), min=1
    )  # (3,)

    flat_cell, atom_idx, combo_idx, cart_pos = assign_to_grid(
        xyz_frac, cell, sg, op_indices, cell_offsets_valid, grid_dims
    )

    n_grid_total = grid_dims[0].item() * grid_dims[1].item() * grid_dims[2].item()

    # Step 3: sort into cell list
    sort_order, unique_cells, starts, cell_lookup = build_cell_list(
        flat_cell, n_grid_total
    )
    cart_sorted = cart_pos[sort_order]
    atom_idx_sorted = atom_idx[sort_order]
    combo_idx_sorted = combo_idx[sort_order]

    if verbose > 0:
        n_occupied = len(unique_cells)
        counts = starts[1:] - starts[:-1]
        print(f"  Grid: {grid_dims.tolist()}, "
              f"{n_occupied}/{n_grid_total} cells occupied, "
              f"max {counts.max().item()} entries/cell")

    # Step 4: find pairs via periodic grid + batched cdist.
    # Uses the canonical-14-offset path; this covers each unique cell
    # pair exactly once (see find_pairs_periodic_grid_v2), so the raw
    # output is already (nearly) dedup-free — the downstream hash dedup
    # still runs as a safety net for the small number of intra-cell
    # swap-canonicalisation collisions.
    pair_atom_i, pair_atom_j, pair_combo_j = find_pairs_periodic_grid_v2(
        cart_sorted, atom_idx_sorted, combo_idx_sorted,
        unique_cells, starts, cell_lookup, grid_dims,
        cutoff, identity_combo,
    )

    if len(pair_atom_i) == 0:
        if verbose > 0:
            print("  Built 0 VDW restraints")
        return empty_result

    # Step 5: filter
    max_idx = max(n_asu, int(pair_atom_i.max().item()) + 1,
                  int(pair_atom_j.max().item()) + 1)
    excl_hash = exclusion_set_to_hash(exclusion_set, max_idx, device)

    keep = filter_pairs(
        pair_atom_i, pair_atom_j, pair_combo_j,
        identity_combo, excl_hash, max_idx,
        pdb, inter_residue_only,
    )

    pair_atom_i = pair_atom_i[keep]
    pair_atom_j = pair_atom_j[keep]
    pair_combo_j = pair_combo_j[keep]

    if len(pair_atom_i) == 0:
        if verbose > 0:
            print("  Built 0 VDW restraints (all filtered)")
        return empty_result

    # Deduplicate: keep first occurrence of each (atom_i, atom_j, combo_j)
    dedup_hash = pair_atom_i * (n_asu * M) + pair_atom_j * M + pair_combo_j
    _, inverse, counts = torch.unique(
        dedup_hash, return_inverse=True, return_counts=True
    )
    # First occurrence: for each unique hash, the minimum index
    perm = torch.arange(len(inverse), device=device)
    first_occ = torch.empty_like(counts).fill_(len(inverse))
    first_occ.scatter_reduce_(0, inverse, perm, reduce="amin")
    first_mask = torch.zeros(len(pair_atom_i), dtype=torch.bool, device=device)
    first_mask[first_occ] = True

    pair_atom_i = pair_atom_i[first_mask]
    pair_atom_j = pair_atom_j[first_mask]
    pair_combo_j = pair_combo_j[first_mask]

    # Map combo_j back to symop index and cell offset
    symop_indices = op_indices[pair_combo_j]
    pair_cell_offsets = cell_offsets_valid[pair_combo_j]

    # VDW radii
    vdw_radii = vdw_radii_fn()
    min_distances = vdw_radii[pair_atom_i] + vdw_radii[pair_atom_j]

    # Build output
    indices = torch.stack([pair_atom_i, pair_atom_j], dim=1)

    result = {
        "indices": indices,
        "min_distances": min_distances.to(torch.float32),
        "sigmas": torch.full(
            (len(indices),), sigma, dtype=torch.float32, device=device
        ),
        "symop_indices": symop_indices,
        "cell_offsets": pair_cell_offsets,
        # Cached data for forward-time H-VDW pair search
        "valid_op_indices": op_indices,
        "valid_cell_offsets": cell_offsets_valid,
        "grid_dims": grid_dims,
        "identity_combo": torch.tensor(identity_combo, dtype=torch.long, device=device),
    }

    if verbose > 0:
        n_sym = (
            (symop_indices != 0) | (pair_cell_offsets != 0).any(dim=1)
        ).sum().item()
        print(f"  Built {len(indices)} VDW restraints, {n_sym} symmetry contacts")

    return result


# ------------------------------------------------------------------ #
# H-involving pair search (forward-time, called every evaluation)
# ------------------------------------------------------------------ #

@torch.no_grad()
def find_h_vdw_pairs_gpu(
    xyz_heavy: torch.Tensor,
    xyz_h: torch.Tensor,
    cell: "Cell",
    sg: "SpaceGroup",
    op_indices: torch.Tensor,
    cell_offsets_valid: torch.Tensor,
    grid_dims: torch.Tensor,
    identity_combo: int,
    n_heavy: int,
    cutoff: float = 3.5,
    h_excl_hash: Optional[torch.Tensor] = None,
    pdb=None,
    h_chainid_enc: Optional[torch.Tensor] = None,
    h_resseq: Optional[torch.Tensor] = None,
    inter_residue_only: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find VDW pairs involving at least one hydrogen atom.

    Uses the same periodic-grid spatial hash as the heavy-atom search but
    operates on a combined (heavy + H) coordinate set and only returns
    pairs where at least one participant is a hydrogen (index >= n_heavy).

    Called at every forward evaluation — designed for low latency.

    Parameters
    ----------
    xyz_heavy : (N_heavy, 3) Cartesian ASU heavy-atom positions
    xyz_h : (N_h, 3) Cartesian ASU hydrogen positions
    cell, sg : Cell, SpaceGroup
    op_indices : (M,) cached valid symop indices
    cell_offsets_valid : (M, 3) cached valid cell translations
    grid_dims : (3,) cached grid dimensions
    identity_combo : int
    n_heavy : int
        Number of heavy atoms (indices 0..n_heavy-1 are heavy)
    cutoff : float
        Cartesian cutoff (Å), tighter than heavy-atom search
    h_excl_hash : (E,) sorted long
        Hash tensor for H-specific 1-2/1-3 exclusions
    pdb : DataFrame
        Heavy-atom pdb for same-residue filtering
    h_chainid_enc : (N_h,) long
        Chain ID encoding for H atoms
    h_resseq : (N_h,) long
        Residue sequence number for H atoms
    inter_residue_only : bool

    Returns
    -------
    pair_atom_i, pair_atom_j, pair_combo_j : each (P,) long
        Indices into the combined (heavy + H) array.  atom_i is always
        from the ASU (identity combo).
    """
    from torchref.symmetry.spacegroup import SpaceGroup as SG

    device = xyz_heavy.device
    fdtype = dtypes.float

    if not isinstance(sg, SG):
        sg = SG(sg)

    # Combine heavy + H into a single coordinate set
    xyz_all = torch.cat([xyz_heavy, xyz_h], dim=0)  # (N_all, 3)
    n_all = xyz_all.shape[0]

    empty = torch.tensor([], dtype=torch.long, device=device)
    if n_all == 0:
        return empty, empty, empty

    # Convert to fractional
    xyz_frac = cell.cartesian_to_fractional(xyz_all.detach().to(fdtype))

    M = op_indices.shape[0]

    # Step 2: assign to grid (reusing cached grid_dims and symop combos)
    flat_cell, atom_idx, combo_idx, cart_pos = assign_to_grid(
        xyz_frac, cell, sg, op_indices, cell_offsets_valid, grid_dims
    )

    n_grid_total = grid_dims[0].item() * grid_dims[1].item() * grid_dims[2].item()

    # Step 3: sort into cell list
    sort_order, unique_cells, starts, cell_lookup = build_cell_list(
        flat_cell, n_grid_total
    )
    cart_sorted = cart_pos[sort_order]
    atom_idx_sorted = atom_idx[sort_order]
    combo_idx_sorted = combo_idx[sort_order]

    # Step 4: find pairs (use the canonical-14-offset fast path)
    pair_atom_i, pair_atom_j, pair_combo_j = find_pairs_periodic_grid_v2(
        cart_sorted, atom_idx_sorted, combo_idx_sorted,
        unique_cells, starts, cell_lookup, grid_dims,
        cutoff, identity_combo,
    )

    if len(pair_atom_i) == 0:
        return empty, empty, empty

    # Filter to keep only pairs involving at least one H
    has_h = (pair_atom_i >= n_heavy) | (pair_atom_j >= n_heavy)
    pair_atom_i = pair_atom_i[has_h]
    pair_atom_j = pair_atom_j[has_h]
    pair_combo_j = pair_combo_j[has_h]

    if len(pair_atom_i) == 0:
        return empty, empty, empty

    # Step 5: filtering

    is_intra_asu = pair_combo_j == identity_combo

    # Bonded exclusions (1-2 H-parent, 1-3 H-parent_neighbor) — intra-ASU
    if h_excl_hash is not None and len(h_excl_hash) > 0 and is_intra_asu.any():
        max_idx = max(n_all, int(pair_atom_i.max().item()) + 1,
                      int(pair_atom_j.max().item()) + 1)
        norm_i = torch.minimum(pair_atom_i, pair_atom_j)
        norm_j = torch.maximum(pair_atom_i, pair_atom_j)
        pair_hash = norm_i * max_idx + norm_j
        ins = torch.searchsorted(h_excl_hash, pair_hash)
        ins = ins.clamp(max=len(h_excl_hash) - 1)
        is_excluded = h_excl_hash[ins] == pair_hash
        keep = ~(is_excluded & is_intra_asu)
        pair_atom_i = pair_atom_i[keep]
        pair_atom_j = pair_atom_j[keep]
        pair_combo_j = pair_combo_j[keep]
        is_intra_asu = is_intra_asu[keep]

    if len(pair_atom_i) == 0:
        return empty, empty, empty

    # Same-residue filter — intra-ASU only
    if inter_residue_only and pdb is not None:
        pdb_chainid = pdb["chainid"].values
        pdb_resseq = pdb["resseq"].values.astype(np.int64)

        # Build combined chain/resseq arrays (heavy from pdb, H from topology)
        if h_chainid_enc is not None and h_resseq is not None:
            # For heavy atoms, encode chain IDs consistently
            chain_vals = pdb_chainid.astype(str)
            unique_chains = np.unique(chain_vals)
            chain_to_int = {c: i for i, c in enumerate(unique_chains)}
            heavy_chain_enc = np.array([chain_to_int.get(c, -1) for c in chain_vals],
                                       dtype=np.int64)
            heavy_resseq = pdb_resseq

            all_chain_enc = np.concatenate([
                heavy_chain_enc,
                h_chainid_enc.cpu().numpy(),
            ])
            all_resseq = np.concatenate([
                heavy_resseq,
                h_resseq.cpu().numpy(),
            ])
        else:
            all_chain_enc = None

        if all_chain_enc is not None:
            ai_np = pair_atom_i.cpu().numpy()
            aj_np = pair_atom_j.cpu().numpy()
            same_res = (
                (all_chain_enc[ai_np] == all_chain_enc[aj_np])
                & (all_resseq[ai_np] == all_resseq[aj_np])
            )
            same_res_t = torch.tensor(same_res, dtype=torch.bool, device=device)
            keep = ~(same_res_t & is_intra_asu)
            pair_atom_i = pair_atom_i[keep]
            pair_atom_j = pair_atom_j[keep]
            pair_combo_j = pair_combo_j[keep]

    if len(pair_atom_i) == 0:
        return empty, empty, empty

    # Altloc compatibility — intra-ASU, only relevant for heavy atoms
    if pdb is not None and "altloc" in pdb.columns:
        is_intra_asu = pair_combo_j == identity_combo
        # Only check altloc for pairs where both are heavy atoms
        both_heavy = (pair_atom_i < n_heavy) & (pair_atom_j < n_heavy) & is_intra_asu
        if both_heavy.any():
            altloc = pdb["altloc"].values.astype(str)
            altloc = np.where(np.isin(altloc, ["", " "]), " ", altloc)
            ai_np = pair_atom_i[both_heavy].cpu().numpy()
            aj_np = pair_atom_j[both_heavy].cpu().numpy()
            incompat = (altloc[ai_np] != " ") & (altloc[aj_np] != " ") & (altloc[ai_np] != altloc[aj_np])
            reject = torch.zeros(len(pair_atom_i), dtype=torch.bool, device=device)
            reject[both_heavy] = torch.tensor(incompat, dtype=torch.bool, device=device)
            keep = ~reject
            pair_atom_i = pair_atom_i[keep]
            pair_atom_j = pair_atom_j[keep]
            pair_combo_j = pair_combo_j[keep]

    if len(pair_atom_i) == 0:
        return empty, empty, empty

    # Deduplicate
    dedup_hash = pair_atom_i * (n_all * M) + pair_atom_j * M + pair_combo_j
    _, inverse, counts = torch.unique(
        dedup_hash, return_inverse=True, return_counts=True
    )
    perm = torch.arange(len(inverse), device=device)
    first_occ = torch.empty_like(counts).fill_(len(inverse))
    first_occ.scatter_reduce_(0, inverse, perm, reduce="amin")
    first_mask = torch.zeros(len(pair_atom_i), dtype=torch.bool, device=device)
    first_mask[first_occ] = True

    return pair_atom_i[first_mask], pair_atom_j[first_mask], pair_combo_j[first_mask]
