"""
GPU-native periodic neighbor search for VDW restraints.

Works in fractional space with periodic boundary conditions.
Avoids explicit symmetry expansion by assigning (atom, symop+offset)
entries to grid cells and using padded batched ``torch.cdist``
for distance computation. On CPU the pair search itself is a
k-d tree instead (:func:`find_pairs_kdtree`), with the same output.

All operations run under ``torch.no_grad()`` on whatever device
the input coordinates live on (CPU or GPU). :func:`vdw_radii_for_elements`
gives the per-atom radii the contact distances are summed from.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from torchref.config import dtypes, get_float_dtype, get_int_dtype

if TYPE_CHECKING:
    from torchref.symmetry.cell import Cell
    from torchref.symmetry.spacegroup import SpaceGroup

#: Radius in Å for an element missing from ``atomic_vdw_radii.csv``.
_DEFAULT_VDW_RADIUS = 1.9


def vdw_radii_for_elements(elements) -> np.ndarray:
    """Van der Waals radius of each atom, looked up by element.

    Parameters
    ----------
    elements : array-like of str
        Element symbols, one per atom; case and surrounding whitespace are ignored.

    Returns
    -------
    numpy.ndarray
        Radii in Å, shape ``(n_atoms,)``, float64. Elements the table does not list
        get 1.9 Å.
    """
    import os

    import pandas as pd

    from torchref import PATH_TORCHREF_DATA

    table = pd.read_csv(
        os.path.join(PATH_TORCHREF_DATA, "atomic_vdw_radii.csv"), comment="#"
    )
    radius = dict(
        zip(
            table["element"].str.strip().str.capitalize(),
            table["vdW_Radius_Angstrom"],
        )
    )
    symbols = np.char.capitalize(np.char.strip(np.asarray(elements).astype(str)))
    return np.array(
        [radius.get(e, _DEFAULT_VDW_RADIUS) for e in symbols], dtype=np.float64
    )


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
    cell : Cell
        Crystallographic unit cell.
    sg : SpaceGroup
        Space group providing the symmetry operators.
    xyz_frac : torch.Tensor
        ``(N, 3)`` fractional ASU coordinates.
    cutoff : float
        Cartesian cutoff in Angstrom.

    Returns
    -------
    op_indices : (M,) int – symop indices for each valid combo
    cell_offsets : (M, 3) int – integer cell translations
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

    op_indices = torch.tensor(valid_ops, dtype=get_int_dtype(), device=device)
    cell_offsets = torch.tensor(valid_offsets, dtype=get_int_dtype(), device=device)
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
    op_indices : (M,) int
    cell_offsets : (M, 3) int
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
        min=torch.zeros(3, dtype=get_int_dtype(), device=device),
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
    starts : (C+1,) int – CSR boundaries into sorted arrays
    cell_lookup : (n_grid_total,) int – maps flat cell → index in
        unique_cells, or -1 if empty.
    """
    device = flat_cell.device
    sort_order = flat_cell.argsort()
    sorted_cells = flat_cell[sort_order]

    unique_cells, counts = torch.unique_consecutive(
        sorted_cells, return_counts=True
    )
    starts = torch.zeros(len(unique_cells) + 1, dtype=get_int_dtype(), device=device)
    starts[1:] = counts.cumsum(0)

    cell_lookup = torch.full((n_grid_total,), -1, dtype=get_int_dtype(), device=device)
    cell_lookup[unique_cells] = torch.arange(
        len(unique_cells), dtype=get_int_dtype(), device=device
    )

    return sort_order, unique_cells, starts, cell_lookup


# ------------------------------------------------------------------ #
# Step 4 – padded batched cdist over 27 neighbor offsets
# ------------------------------------------------------------------ #

_NEIGHBOR_OFFSETS_27 = None
_NEIGHBOR_OFFSETS_14 = None


def _get_canonical_offsets_14(device: torch.device) -> torch.Tensor:
    """Return (14, 3): the self-offset plus the lex-positive half of the 26 others.

    Keeping one of each mirror pair ``(d, -d)`` visits every unique cell pair once;
    the symmetric partner must be recovered by swapping source and target indices.
    Cached per device in a module global.
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
            offsets, dtype=get_int_dtype(), device=device
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
    """Pad the CSR cell list to fixed rows for the cdist loop.

    Returns
    -------
    padded_xyz : (C, max_per_cell, 3) float, padding filled with ``inf`` so
        padded slots can never fall within any cutoff
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
    """Find candidate VDW pairs on a periodic grid with batched cdist.

    Sweeps the 14 canonical neighbour offsets so each unique cell pair is visited
    once, recovering the symmetric partner by swapping source and target indices
    on emission.

    Parameters
    ----------
    cart_sorted : (E, 3) float
        Cartesian image positions, sorted into CSR cell order.
    atom_idx_sorted, combo_idx_sorted : (E,) long
        ASU atom index and (symop, offset) combo index per entry.
    unique_cells : (C,) long
        Occupied flat grid-cell indices.
    starts : (C+1,) int
        CSR boundaries into the sorted arrays.
    cell_lookup : (n_grid_total,) int
        Maps a flat cell index to its position in ``unique_cells`` (-1 empty).
    grid_dims : (3,) long
        Number of grid cells per axis.
    cutoff : float
        Cartesian distance cutoff in Angstrom.
    identity_combo : int
        Combo index corresponding to the identity (op=0, offset=0).
    chunk_size : int, optional
        Number of source cells per cdist tile. Defaults to a device-adaptive
        value (CUDA: cap the tile at ~256 MB; CPU: 128).

    Returns
    -------
    pair_atom_i, pair_atom_j, pair_combo_j : each (P,) long
        ASU atom index, partner atom index, and partner combo index.

    Notes
    -----
    A pair is emitted iff at least one side is an ASU atom; it is
    canonicalised so the ASU atom is on the ``i`` side. For intra-ASU pairs
    ``atom_i < atom_j`` is enforced so ``(a, b)`` and ``(b, a)`` collapse to
    one entry, matching the invariant expected by the downstream dedup.
    """
    device = cart_sorted.device
    offsets14 = _get_canonical_offsets_14(device)

    gy = grid_dims[1].item()
    gz = grid_dims[2].item()

    # ---- Precompute ----
    counts = starts[1:] - starts[:-1]
    max_per_cell = counts.max().item()

    # The cdist tile is chunk_size * max_per_cell**2 * 4 bytes, and max_per_cell
    # varies several-fold with packing density, so cap the tile rather than the
    # chunk count -- a fixed chunk_size reaches multi-GB on dense cells.
    if chunk_size is None:
        if device.type == "cuda":
            max_tile_bytes = 256 * 1024 * 1024
            per_chunk_bytes = max(1, max_per_cell * max_per_cell * 4)
            chunk_size = max(64, max_tile_bytes // per_chunk_bytes)
        else:
            chunk_size = 128

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
        nb_occ_idx = cell_lookup[nb_flat]               # (C,) int, -1 empty

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
        empty = torch.tensor([], dtype=get_int_dtype(), device=device)
        return empty, empty, empty

    return (
        torch.cat(all_pair_atom_i),
        torch.cat(all_pair_atom_j),
        torch.cat(all_pair_combo_j),
    )


def find_pairs_kdtree(
    cart_pos: torch.Tensor,
    atom_idx: torch.Tensor,
    combo_idx: torch.Tensor,
    cutoff: float,
    identity_combo: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU counterpart of :func:`find_pairs_periodic_grid_v2`, on a k-d tree.

    Returns the same pairs under the same convention: every ASU atom ``i`` and image
    ``(j, combo_j)`` closer than ``cutoff``, intra-ASU pairs once with ``i < j``, no
    self-pairs. Sorted by ``(i, j, combo_j)``.

    The grid search pads every cell to the fullest one and takes dense distance tiles
    between neighbouring cells, so its cost goes with the square of the peak cell
    occupancy -- about 4x for the same volume once hydrogens are added. Here the cost
    goes with the pairs actually within the cutoff. A tree also has no cell width to
    fall short of the cutoff, which a fractional grid cell can do in an oblique cell.

    Parameters
    ----------
    cart_pos : (E, 3) float
        Cartesian image positions from :func:`assign_to_grid`, unsorted.
    atom_idx, combo_idx : (E,) long
        ASU atom index and (symop, offset) combo index per entry.
    cutoff : float
        Cartesian distance cutoff in Angstrom.
    identity_combo : int
        Combo index corresponding to the identity (op=0, offset=0).
    """
    from scipy.spatial import cKDTree

    device = cart_pos.device
    pos = cart_pos.detach().cpu().numpy()
    atoms = atom_idx.cpu().numpy()
    combos = combo_idx.cpu().numpy()

    asu = np.nonzero(combos == identity_combo)[0]
    found = cKDTree(pos[asu]).sparse_distance_matrix(
        cKDTree(pos), cutoff, output_type="ndarray"
    )
    # sparse_distance_matrix keeps d <= cutoff; the grid search keeps d < cutoff.
    found = found[found["v"] < cutoff]
    ai = atoms[asu[found["i"]]]
    aj = atoms[found["j"]]
    cj = combos[found["j"]]
    # Intra-ASU contacts come back from both ends; keep i < j, which also drops self.
    keep = (cj != identity_combo) | (ai < aj)
    ai, aj, cj = ai[keep], aj[keep], cj[keep]

    order = np.lexsort((cj, aj, ai))
    return tuple(
        torch.from_numpy(np.ascontiguousarray(a[order], dtype=np.int64)).to(device)
        for a in (ai, aj, cj)
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
        # dtype-ok: packed pair key min*max_idx+max overflows int32 beyond ~46k atoms; searchsorted needs both sides int64
        return torch.tensor([], dtype=torch.long, device=device)
    arr = np.array(list(exclusion_set), dtype=np.int64)
    hashes = arr[:, 0] * max_idx + arr[:, 1]  # already (min, max)
    hashes.sort()
    # dtype-ok: packed pair key min*max_idx+max overflows int32 beyond ~46k atoms; searchsorted needs both sides int64
    return torch.tensor(hashes, dtype=torch.long, device=device)


def filter_pairs(
    pair_atom_i: torch.Tensor,
    pair_atom_j: torch.Tensor,
    pair_combo_j: torch.Tensor,
    identity_combo: int,
    excl_hash: torch.Tensor,
    max_idx: int,
    topology,
    inter_residue_only: bool = True,
) -> torch.Tensor:
    """Apply exclusion, residue, and altloc filters. Returns keep mask.

    Residues are the topology's ``(chain, resseq, icode)`` nodes, so atoms of residues
    100 and 100A are in different residues.
    """
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
    ai_np = pair_atom_i.cpu().numpy()
    aj_np = pair_atom_j.cpu().numpy()
    if inter_residue_only:
        residue_of = topology.atoms.residue_of.cpu().numpy()
        same_res = residue_of[ai_np] == residue_of[aj_np]
        same_res_t = torch.tensor(same_res, dtype=torch.bool, device=device)
        keep &= ~(same_res_t & is_intra_asu)

    # Altloc compatibility – intra-ASU only
    altloc = topology.atoms.altloc
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
    xyz: torch.Tensor,
    vdw_radii: torch.Tensor,
    cell: "Cell",
    sg: "SpaceGroup",
    topology,
    exclusion_set: Set[Tuple[int, int]],
    cutoff: float = 5.0,
    sigma: float = 0.2,
    inter_residue_only: bool = True,
    verbose: int = 0,
) -> Dict[str, torch.Tensor]:
    """Build VDW restraints using GPU-native periodic grid search.

    Parameters
    ----------
    xyz : torch.Tensor
        ``(N, 3)`` Cartesian ASU coordinates in Å.
    vdw_radii : torch.Tensor
        ``(N,)`` van der Waals radii in Å.
    cell : Cell
    sg : SpaceGroup
    topology : Topology
        Residue membership and altlocs for the same-residue and altloc filters.
    exclusion_set : set of (int, int) bonded exclusion pairs
    cutoff : float
        Contact distance cutoff in Angstrom.
    sigma : float
        Standard deviation assigned to each VDW restraint.
    inter_residue_only : bool
    verbose : int

    Returns
    -------
    dict with keys: indices, min_distances, sigmas, symop_indices, cell_offsets
    """
    from torchref.symmetry.spacegroup import SpaceGroup as SG

    device = xyz.device
    fdtype = dtypes.float
    n_asu = xyz.shape[0]

    if not isinstance(sg, SG):
        sg = SG(sg)

    empty_result = {
        "indices": torch.zeros(0, 2, dtype=get_int_dtype(), device=device),
        "min_distances": torch.zeros(0, dtype=get_float_dtype(), device=device),
        "sigmas": torch.zeros(0, dtype=get_float_dtype(), device=device),
        "symop_indices": torch.zeros(0, dtype=get_int_dtype(), device=device),
        "cell_offsets": torch.zeros(0, 3, dtype=get_int_dtype(), device=device),
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
        op_indices = torch.cat(
            [torch.zeros(1, dtype=get_int_dtype(), device=device), op_indices]
        )
        cell_offsets_valid = torch.cat(
            [
                torch.zeros(1, 3, dtype=get_int_dtype(), device=device),
                cell_offsets_valid,
            ]
        )
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

    if device.type == "cpu":
        # Steps 3-4 on CPU: a k-d tree, whose cost follows the pairs found rather
        # than the padded cell tiles of the grid search below.
        if verbose > 0:
            print(f"  Pair search: k-d tree over {cart_pos.shape[0]} images")
        pair_atom_i, pair_atom_j, pair_combo_j = find_pairs_kdtree(
            cart_pos, atom_idx, combo_idx, cutoff, identity_combo
        )
    else:
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

        # Step 4: find pairs via periodic grid + batched cdist. Nearly dedup-free by
        # construction, but the hash dedup below still catches intra-cell
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
        topology, inter_residue_only,
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
    # First occurrence: for each unique hash, the minimum index.
    # Use the configured int dtype (int32 by default) — MPS does not support
    # int64 scatter_reduce and N_pairs fits comfortably in int32.
    _int_dtype = dtypes.int
    inverse_i = inverse.to(_int_dtype)
    perm = torch.arange(len(inverse), device=device, dtype=_int_dtype)
    first_occ = torch.full(
        (counts.shape[0],), len(inverse), device=device, dtype=_int_dtype
    )
    first_occ.scatter_reduce_(0, inverse_i, perm, reduce="amin")
    first_mask = torch.zeros(len(pair_atom_i), dtype=torch.bool, device=device)
    first_mask[first_occ.long()] = True

    pair_atom_i = pair_atom_i[first_mask]
    pair_atom_j = pair_atom_j[first_mask]
    pair_combo_j = pair_combo_j[first_mask]

    # Map combo_j back to symop index and cell offset
    symop_indices = op_indices[pair_combo_j]
    pair_cell_offsets = cell_offsets_valid[pair_combo_j]

    min_distances = vdw_radii[pair_atom_i] + vdw_radii[pair_atom_j]

    # Build output
    indices = torch.stack([pair_atom_i, pair_atom_j], dim=1)

    result = {
        "indices": indices,
        "min_distances": min_distances.to(get_float_dtype()),
        "sigmas": torch.full(
            (len(indices),), sigma, dtype=get_float_dtype(), device=device
        ),
        "symop_indices": symop_indices,
        "cell_offsets": pair_cell_offsets,
        # Cached data for forward-time H-VDW pair search
        "valid_op_indices": op_indices,
        "valid_cell_offsets": cell_offsets_valid,
        "grid_dims": grid_dims,
        "identity_combo": torch.tensor(
            identity_combo, dtype=get_int_dtype(), device=device
        ),
    }

    if verbose > 0:
        n_sym = (
            (symop_indices != 0) | (pair_cell_offsets != 0).any(dim=1)
        ).sum().item()
        print(f"  Built {len(indices)} VDW restraints, {n_sym} symmetry contacts")

    return result
