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

    # Step 4: find pairs via periodic grid + batched cdist
    pair_atom_i, pair_atom_j, pair_combo_j = find_pairs_periodic_grid(
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
    }

    if verbose > 0:
        n_sym = (
            (symop_indices != 0) | (pair_cell_offsets != 0).any(dim=1)
        ).sum().item()
        print(f"  Built {len(indices)} VDW restraints, {n_sym} symmetry contacts")

    return result
