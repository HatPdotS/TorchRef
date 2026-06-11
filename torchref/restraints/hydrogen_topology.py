"""
Riding hydrogen topology and vectorized placement for VDW restraints.

Builds a static topology map at restraints-construction time that describes
how to generate transient hydrogen atom positions from heavy-atom coordinates.
At each VDW evaluation the ``place_riding_hydrogens`` function produces H
positions in a single vectorized pass (no Python loops over atoms).

Hydrogen positions are fully determined by the parent heavy atom and its
bonded heavy-atom neighbours, so gradients flow from the VDW loss through
the H positions back to the heavy-atom coordinates via standard autograd.
"""

from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from torchref.config import dtypes, get_default_device
from torchref.utils.device_mixin import DeviceMixin

# ---------------------------------------------------------------------------
# Placement-type constants
# ---------------------------------------------------------------------------
ANTI_SUM = 0  # 1 H, >=2 heavy neighbours → opposite to sum of vectors
CH2_A = 1  # 2 H on 2-neighbour parent, slot 0 → +perp component
CH2_B = 2  # 2 H on 2-neighbour parent, slot 1 → -perp component
METHYL = 3  # 3 H on 1-neighbour parent → 120° staggered around axis
OPPOSITE_1NB = 4  # 1 H, 1 heavy neighbour → directly opposite
NH2_A = 5  # 2 H on 1-neighbour parent, slot 0
NH2_B = 6  # 2 H on 1-neighbour parent, slot 1

# Pre-computed tetrahedral geometry constants
_COS_TET = -1.0 / 3.0  # cos(180 - 109.47) from axis
_SIN_TET = np.sqrt(8.0 / 9.0)  # sin(180 - 109.47)
_COS_120 = np.cos(2.0 * np.pi / 3.0)  # -0.5
_SIN_120 = np.sin(2.0 * np.pi / 3.0)  # √3/2
_COS_240 = np.cos(4.0 * np.pi / 3.0)  # -0.5
_SIN_240 = np.sin(4.0 * np.pi / 3.0)  # -√3/2

MAX_HEAVY_NB = 4  # Maximum heavy-atom neighbours to store per parent


# ---------------------------------------------------------------------------
# HydrogenTopology
# ---------------------------------------------------------------------------


class HydrogenTopology(DeviceMixin, nn.Module):
    """Static topology describing riding hydrogens for VDW evaluation.

    All data are stored as registered buffers so they move automatically
    with ``.to(device)`` and appear in ``state_dict``.

    Attributes
    ----------
    h_parent_idx : (N_h,) long
        Index into heavy-atom array for each riding H.
    h_bond_length : (N_h,) float
        Ideal H–parent bond length (Å).
    h_vdw_radius : (N_h,) float
        Van der Waals radius for each H (1.20 Å).
    h_placement_type : (N_h,) long
        Placement-geometry enum (see module-level constants).
    h_slot_in_parent : (N_h,) long
        Ordinal within sibling H atoms on the same parent (0, 1, 2).
    parent_neighbor_idx : (N_h, MAX_HEAVY_NB) long
        Heavy-atom neighbour indices of the parent (-1 = padding).
    parent_neighbor_count : (N_h,) long
        Actual number of heavy-atom neighbours for the parent.
    h_chainid_enc : (N_h,) long
        Encoded chain ID (for same-residue filtering).
    h_resseq : (N_h,) long
        Residue sequence number (for same-residue filtering).
    """

    def __init__(self):
        super().__init__()
        # Buffers are registered by build_hydrogen_topology()

    @property
    def n_hydrogens(self) -> int:
        if hasattr(self, "h_parent_idx"):
            return self.h_parent_idx.shape[0]
        return 0

    @property
    def has_candidates(self) -> bool:
        """Whether precomputed H candidate pairs are available."""
        return hasattr(self, "cand_idx_i") and self.cand_idx_i.shape[0] > 0


# ---------------------------------------------------------------------------
# Build-time topology construction
# ---------------------------------------------------------------------------


def _load_cif_hydrogen_info(pdb, verbose: int = 0) -> Dict:
    """Load per-residue-type H topology from the monomer library.

    Re-uses the same CIF cache as ``Model.hydrogenate()``.

    Returns
    -------
    cache : dict
        ``{resname: {ids, elems, coords, is_h, id_to_idx,
                     parent_map, ideal_bl, heavy_neighbor_map, ...} | None}``
    """
    from torchref.model.model import Model
    from torchref.restraints.library import MonomerLibraryManager

    lib = MonomerLibraryManager(verbose=0)
    cache = Model._hydrogenate_cif_cache

    for rn in pdb["resname"].unique():
        rn_str = str(rn).strip()
        if not rn_str:
            continue
        if rn_str in cache:
            if cache[rn_str] is None or "heavy_neighbor_map" in cache[rn_str]:
                continue
            del cache[rn_str]

        cif_path = lib.get_cif_file(rn_str)
        if cif_path is None:
            cache[rn_str] = None
            continue

        try:
            import pandas as pd

            from torchref.io.cif_readers import RestraintCIFReader

            reader = RestraintCIFReader(str(cif_path))
            all_data = reader.get_all_restraints()
            comp_data = all_data.get(rn_str) or all_data.get(rn_str.upper())
            if comp_data is None:
                cache[rn_str] = None
                continue
            atom_df = comp_data.get("atoms", comp_data.get("atom"))
            bond_df = comp_data.get("bonds", comp_data.get("bond"))
            if atom_df is None or atom_df.empty or "x" not in atom_df.columns:
                cache[rn_str] = None
                continue
        except Exception:
            cache[rn_str] = None
            continue

        ids = atom_df["atom_id"].astype(str).str.strip().values
        elems = atom_df["type_symbol"].astype(str).str.strip().values
        coords = atom_df[["x", "y", "z"]].values.astype(np.float64)
        is_h = np.array([e.upper() == "H" for e in elems])
        id_to_idx = {n: i for i, n in enumerate(ids)}

        parent_map = {}
        ideal_bl = {}
        heavy_neighbor_map = {}
        if bond_df is not None and not bond_df.empty:
            a1s = bond_df["atom1"].astype(str).str.strip().values
            a2s = bond_df["atom2"].astype(str).str.strip().values
            vals = pd.to_numeric(bond_df["value"], errors="coerce").values
            h_set = set(ids[is_h])
            for i in range(len(a1s)):
                b1, b2 = a1s[i], a2s[i]
                if b1 in h_set and b2 in id_to_idx and not is_h[id_to_idx[b2]]:
                    parent_map[b1] = b2
                    if np.isfinite(vals[i]):
                        ideal_bl[b1] = float(vals[i])
                elif b2 in h_set and b1 in id_to_idx and not is_h[id_to_idx[b1]]:
                    parent_map[b2] = b1
                    if np.isfinite(vals[i]):
                        ideal_bl[b2] = float(vals[i])
                i1, i2 = id_to_idx.get(b1), id_to_idx.get(b2)
                if i1 is not None and i2 is not None and not is_h[i1] and not is_h[i2]:
                    heavy_neighbor_map.setdefault(b1, []).append(b2)
                    heavy_neighbor_map.setdefault(b2, []).append(b1)

        cache[rn_str] = {
            "ids": ids,
            "elems": elems,
            "coords": coords,
            "is_h": is_h,
            "id_to_idx": id_to_idx,
            "heavy_names": ids[~is_h],
            "heavy_coords": coords[~is_h],
            "h_names": ids[is_h],
            "h_coords": coords[is_h],
            "parent_map": parent_map,
            "ideal_bl": ideal_bl,
            "heavy_neighbor_map": heavy_neighbor_map,
        }

    return cache


def _classify_placement(n_h_on_parent: int, n_heavy_nb: int, slot: int) -> int:
    """Return placement-type code for an H atom, or -1 to skip.

    Returns -1 when the parent has no heavy neighbours so geometry
    cannot be determined.
    """
    if n_heavy_nb == 0:
        return -1  # cannot determine geometry — skip this H
    if n_h_on_parent == 1:
        if n_heavy_nb >= 2:
            return ANTI_SUM
        else:
            return OPPOSITE_1NB
    elif n_h_on_parent == 2:
        if n_heavy_nb >= 2:
            return CH2_A if slot == 0 else CH2_B
        else:
            return NH2_A if slot == 0 else NH2_B
    elif n_h_on_parent == 3:
        return METHYL
    # Fallback for >3 H (rare)
    return ANTI_SUM


def build_hydrogen_topology(
    pdb,
    device: torch.device = None,
    verbose: int = 0,
) -> HydrogenTopology:
    """Build riding-hydrogen topology from the model's heavy-atom DataFrame.

    Parameters
    ----------
    pdb : pd.DataFrame
        Heavy-atom DataFrame (``strip_H=True``).
    device : torch.device
        Target device for tensors.
    verbose : int
        Verbosity level.

    Returns
    -------
    HydrogenTopology
        Module with registered buffer tensors.
    """
    if device is None:
        device = get_default_device()
    cache = _load_cif_hydrogen_info(pdb, verbose)

    model_names = pdb["name"].astype(str).str.strip().values
    model_xyz = pdb[["x", "y", "z"]].values.astype(np.float64)
    model_elem = pdb["element"].astype(str).str.strip().values
    model_heavy_mask = np.array([e.upper() != "H" for e in model_elem])

    # Encode chain IDs as integers for fast same-residue comparison
    chain_vals = pdb["chainid"].values.astype(str)
    unique_chains = np.unique(chain_vals)
    chain_to_int = {c: i for i, c in enumerate(unique_chains)}
    model_chainid_enc = np.array([chain_to_int[c] for c in chain_vals], dtype=np.int64)
    model_resseq = pdb["resseq"].values.astype(np.int64)

    # Group residues
    group_cols = ["chainid", "resseq", "icode", "resname"]
    group_keys = pdb[group_cols].values
    changes = np.zeros(len(group_keys), dtype=bool)
    changes[0] = True
    for c in range(4):
        changes[1:] |= group_keys[1:, c] != group_keys[:-1, c]
    group_starts = np.nonzero(changes)[0]
    group_ends = np.append(group_starts[1:], len(group_keys))

    # Standard valence for expected-H-count capping
    _std_val = {"C": 4, "N": 3, "O": 2, "S": 2}

    # Accumulators
    acc_parent_idx = []
    acc_bond_length = []
    acc_placement_type = []
    acc_slot = []
    acc_nb_idx = []  # list of (MAX_HEAVY_NB,) arrays
    acc_nb_count = []
    acc_chainid_enc = []
    acc_resseq = []

    for gi in range(len(group_starts)):
        s, e = group_starts[gi], group_ends[gi]
        rn = str(group_keys[s, 3]).strip()
        info = cache.get(rn)
        if info is None:
            continue

        chainid_enc = model_chainid_enc[s]
        resseq = model_resseq[s]

        names_in_model = set(model_names[s:e])
        h_to_add_mask = np.array(
            [n not in names_in_model for n in info["h_names"]], dtype=bool
        )
        if not h_to_add_mask.any():
            continue
        h_names_add = info["h_names"][h_to_add_mask]

        # Build name→global-index map for this residue
        name_to_global = {}
        for j in range(s, e):
            nm = model_names[j]
            if nm not in name_to_global:
                name_to_global[nm] = j

        # Group H atoms by parent
        parent_to_h = {}
        for h_name in h_names_add:
            pn = info["parent_map"].get(h_name)
            if pn is not None and pn in name_to_global:
                parent_to_h.setdefault(pn, []).append(h_name)

        hnm = info.get("heavy_neighbor_map", {})
        id2i = info["id_to_idx"]

        for par_name, h_list in parent_to_h.items():
            pidx = name_to_global[par_name]

            # Find heavy-atom neighbours of parent via distance
            dvec = model_xyz - model_xyz[pidx]
            dists_sq = (dvec**2).sum(1)
            bonded = np.where((dists_sq > 0.09) & (dists_sq < 3.61) & model_heavy_mask)[
                0
            ]
            bonded = bonded[bonded != pidx]
            n_model_heavy = len(bonded)

            # Cap H count by expected valence
            par_elem = info["elems"][id2i[par_name]].upper()
            expected_h = max(0, _std_val.get(par_elem, 4) - n_model_heavy)
            h_list_capped = sorted(h_list)[:expected_h]
            if not h_list_capped:
                continue

            n_h = len(h_list_capped)

            # Neighbour index array (padded)
            nb_arr = np.full(MAX_HEAVY_NB, -1, dtype=np.int64)
            nb_count = min(n_model_heavy, MAX_HEAVY_NB)
            nb_arr[:nb_count] = bonded[:nb_count]

            for slot, h_name in enumerate(h_list_capped):
                bl = info["ideal_bl"].get(h_name, 0.97)
                ptype = _classify_placement(n_h, n_model_heavy, slot)

                if ptype < 0:
                    continue  # skip — cannot determine geometry

                acc_parent_idx.append(pidx)
                acc_bond_length.append(bl)
                acc_placement_type.append(ptype)
                acc_slot.append(slot)
                acc_nb_idx.append(nb_arr.copy())
                acc_nb_count.append(nb_count)
                acc_chainid_enc.append(chainid_enc)
                acc_resseq.append(resseq)

    topo = HydrogenTopology()
    n_h_total = len(acc_parent_idx)
    fdtype = dtypes.float

    if n_h_total == 0:
        topo.register_buffer(
            "h_parent_idx", torch.zeros(0, dtype=torch.long, device=device)
        )
        topo.register_buffer(
            "h_bond_length", torch.zeros(0, dtype=fdtype, device=device)
        )
        topo.register_buffer(
            "h_vdw_radius", torch.zeros(0, dtype=fdtype, device=device)
        )
        topo.register_buffer(
            "h_placement_type", torch.zeros(0, dtype=torch.long, device=device)
        )
        topo.register_buffer(
            "h_slot_in_parent", torch.zeros(0, dtype=torch.long, device=device)
        )
        topo.register_buffer(
            "parent_neighbor_idx",
            torch.zeros(0, MAX_HEAVY_NB, dtype=torch.long, device=device),
        )
        topo.register_buffer(
            "parent_neighbor_count", torch.zeros(0, dtype=torch.long, device=device)
        )
        topo.register_buffer(
            "h_chainid_enc", torch.zeros(0, dtype=torch.long, device=device)
        )
        topo.register_buffer(
            "h_resseq", torch.zeros(0, dtype=torch.long, device=device)
        )
        return topo

    # Sort all topology arrays by placement type for contiguous slicing
    ptype_arr = np.array(acc_placement_type, dtype=np.int64)
    sort_order = np.argsort(ptype_arr, kind="stable")

    acc_parent_idx = [acc_parent_idx[i] for i in sort_order]
    acc_bond_length = [acc_bond_length[i] for i in sort_order]
    acc_placement_type = [acc_placement_type[i] for i in sort_order]
    acc_slot = [acc_slot[i] for i in sort_order]
    acc_nb_idx = [acc_nb_idx[i] for i in sort_order]
    acc_nb_count = [acc_nb_count[i] for i in sort_order]
    acc_chainid_enc = [acc_chainid_enc[i] for i in sort_order]
    acc_resseq = [acc_resseq[i] for i in sort_order]

    # Compute type boundaries: type_bounds[t] = (start, end) slice
    sorted_types = np.array(acc_placement_type, dtype=np.int64)
    type_bounds = {}
    for t in range(7):
        mask = sorted_types == t
        if mask.any():
            idxs = np.where(mask)[0]
            type_bounds[t] = (int(idxs[0]), int(idxs[-1]) + 1)

    topo.register_buffer(
        "h_parent_idx",
        torch.tensor(acc_parent_idx, dtype=torch.long, device=device),
    )
    topo.register_buffer(
        "h_bond_length",
        torch.tensor(acc_bond_length, dtype=fdtype, device=device),
    )
    topo.register_buffer(
        "h_vdw_radius",
        torch.full((n_h_total,), 1.20, dtype=fdtype, device=device),
    )
    topo.register_buffer(
        "h_placement_type",
        torch.tensor(acc_placement_type, dtype=torch.long, device=device),
    )
    topo.register_buffer(
        "h_slot_in_parent",
        torch.tensor(acc_slot, dtype=torch.long, device=device),
    )
    topo.register_buffer(
        "parent_neighbor_idx",
        torch.tensor(np.stack(acc_nb_idx), dtype=torch.long, device=device),
    )
    topo.register_buffer(
        "parent_neighbor_count",
        torch.tensor(acc_nb_count, dtype=torch.long, device=device),
    )
    topo.type_bounds = type_bounds  # dict: type_code -> (start, end)
    topo.register_buffer(
        "h_chainid_enc",
        torch.tensor(acc_chainid_enc, dtype=torch.long, device=device),
    )
    topo.register_buffer(
        "h_resseq",
        torch.tensor(acc_resseq, dtype=torch.long, device=device),
    )

    if verbose > 0:
        print(f"  Hydrogen topology: {n_h_total} riding H atoms")

    return topo


# ---------------------------------------------------------------------------
# Vectorized H placement (forward-time, differentiable)
# ---------------------------------------------------------------------------


def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize vectors along last dimension with epsilon for stability."""
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def _orthonormal_basis(axis: torch.Tensor) -> tuple:
    """Build two perpendicular unit vectors for each axis vector.

    Parameters
    ----------
    axis : (B, 3) unit vectors

    Returns
    -------
    perp1, perp2 : each (B, 3) unit vectors forming a right-handed frame
    """
    B = axis.shape[0]
    # Choose the cardinal axis least aligned with the input axis
    abs_ax = axis.abs()
    # Least component → use that cardinal direction
    min_idx = abs_ax.argmin(dim=-1)  # (B,)
    cardinal = torch.zeros_like(axis)
    cardinal[torch.arange(B, device=axis.device), min_idx] = 1.0

    perp1 = torch.cross(axis, cardinal, dim=-1)
    perp1 = _safe_normalize(perp1)
    perp2 = torch.cross(axis, perp1, dim=-1)
    return perp1, perp2


def _precompute_direction_coefficients(topo: HydrogenTopology) -> torch.Tensor:
    """Precompute (c_base, c_perp1, c_perp2) per H atom.

    Every riding H direction is  ``c0*base + c1*perp1 + c2*perp2``
    where (base, perp1, perp2) is a frame built from neighbour vectors.
    Coefficients depend only on placement type — constant across steps.
    """
    device = topo.h_placement_type.device
    fdtype = topo.h_bond_length.dtype
    N_h = topo.h_parent_idx.shape[0]
    coeffs = torch.zeros(N_h, 3, dtype=fdtype, device=device)

    a_ch2 = 1.0 / np.sqrt(3.0)
    b_ch2 = np.sqrt(2.0 / 3.0)

    tb = getattr(topo, "type_bounds", None)
    if tb is None:
        return coeffs

    slot = topo.h_slot_in_parent
    for code in range(7):
        if code not in tb:
            continue
        s, e = tb[code]
        if code == ANTI_SUM or code == OPPOSITE_1NB:
            coeffs[s:e, 0] = 1.0
        elif code == CH2_A:
            coeffs[s:e, 0] = a_ch2
            coeffs[s:e, 1] = b_ch2
        elif code == CH2_B:
            coeffs[s:e, 0] = a_ch2
            coeffs[s:e, 1] = -b_ch2
        elif code == METHYL:
            angle = slot[s:e].to(fdtype) * (2.0 * np.pi / 3.0)
            coeffs[s:e, 0] = _COS_TET
            coeffs[s:e, 1] = _SIN_TET * torch.cos(angle)
            coeffs[s:e, 2] = _SIN_TET * torch.sin(angle)
        elif code == NH2_A:
            coeffs[s:e, 0] = 0.5
            coeffs[s:e, 1] = np.sqrt(3.0) / 2.0
        elif code == NH2_B:
            coeffs[s:e, 0] = 0.5
            coeffs[s:e, 1] = -np.sqrt(3.0) / 2.0
    return coeffs


@torch.jit.script
def _place_h_jit(
    xyz_heavy: torch.Tensor,
    h_parent_idx: torch.Tensor,
    nb_idx_clamped: torch.Tensor,
    nb_valid: torch.Tensor,
    coeffs: torch.Tensor,
    bond_length: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled H placement kernel — fuses ~230 ops into few GPU kernels.

    Parameters
    ----------
    xyz_heavy : (N_heavy, 3)
    h_parent_idx : (N_h,) long
    nb_idx_clamped : (N_h, 4) long — neighbour indices (clamped, -1 → 0)
    nb_valid : (N_h, 4, 1) float — 1.0 where neighbour is valid, 0.0 where pad
    coeffs : (N_h, 3) float — [c_base, c_perp1, c_perp2]
    bond_length : (N_h, 1) float

    Returns
    -------
    (N_h, 3) H positions
    """
    eps = 1e-8
    N_h = h_parent_idx.shape[0]

    # Gather
    parent_pos = xyz_heavy[h_parent_idx]  # (N_h, 3)
    nb_pos = xyz_heavy[nb_idx_clamped]  # (N_h, 4, 3)

    # Neighbour vectors (masked)
    vecs = (nb_pos - parent_pos.unsqueeze(1)) * nb_valid

    # Base direction: -normalized sum of neighbour vectors
    neg_sum = -(vecs.sum(dim=1))
    base = neg_sum / (torch.norm(neg_sum, 2, -1, True) + eps)

    # Perpendicular frame from cross(v1, v2)
    v1 = vecs[:, 0, :]
    v2 = vecs[:, 1, :]
    cross12 = torch.linalg.cross(v1, v2)
    cross_norm = torch.norm(cross12, 2, -1, True)
    has_cross = cross_norm > 1e-6

    perp1_cross = cross12 / (cross_norm + eps)

    # Fallback for 1-neighbour atoms: pick least-aligned cardinal axis
    abs_base = base.abs()
    min_idx = abs_base.argmin(dim=-1)
    cardinal = torch.zeros_like(base)
    cardinal.scatter_(1, min_idx.unsqueeze(1), 1.0)
    perp1_ortho_raw = torch.linalg.cross(base, cardinal)
    perp1_ortho = perp1_ortho_raw / (torch.norm(perp1_ortho_raw, 2, -1, True) + eps)

    perp1 = torch.where(has_cross, perp1_cross, perp1_ortho)
    perp2 = torch.linalg.cross(base, perp1)

    # Direction = c0*base + c1*perp1 + c2*perp2
    direction = coeffs[:, 0:1] * base + coeffs[:, 1:2] * perp1 + coeffs[:, 2:3] * perp2

    return parent_pos + bond_length * direction


def place_riding_hydrogens(
    xyz_heavy: torch.Tensor,
    topo: HydrogenTopology,
) -> torch.Tensor:
    """Generate riding hydrogen positions from heavy-atom coordinates.

    Delegates to a JIT-compiled kernel that fuses element-wise ops,
    reducing GPU kernel launches from ~230 to ~30.

    Parameters
    ----------
    xyz_heavy : (N_heavy, 3) float tensor (requires_grad typically True)
    topo : HydrogenTopology

    Returns
    -------
    xyz_h : (N_h, 3) float tensor, differentiable w.r.t. xyz_heavy
    """
    N_h = topo.h_parent_idx.shape[0]
    if N_h == 0:
        return torch.zeros(0, 3, dtype=xyz_heavy.dtype, device=xyz_heavy.device)

    # Precompute direction coefficients on first call
    if not hasattr(topo, "_dir_coeffs") or topo._dir_coeffs is None:
        topo._dir_coeffs = _precompute_direction_coefficients(topo)

    # Precompute static tensors on first call (avoid recomputing every step)
    if not hasattr(topo, "_nb_idx_clamped"):
        topo._nb_idx_clamped = topo.parent_neighbor_idx.clamp(min=0)
        topo._nb_valid = (
            (topo.parent_neighbor_idx >= 0).unsqueeze(-1).to(topo.h_bond_length.dtype)
        )
        topo._bond_len_col = topo.h_bond_length.unsqueeze(-1)

    # On CUDA fp32 use the fused Triton forward + analytic Triton
    # backward (saves ~1.5–1.9 ms on the H-VDW backward — that path is
    # where ~40 % of the non-bonded fwd+bw cost lives). Falls back to
    # the JIT-scripted eager helper otherwise.
    if xyz_heavy.is_cuda and xyz_heavy.dtype == torch.float32:
        try:
            from torchref.base.targets.triton.place_hydrogens import (
                place_riding_hydrogens_triton,
            )

            return place_riding_hydrogens_triton(
                xyz_heavy,
                topo.h_parent_idx,
                topo._nb_idx_clamped,
                topo._nb_valid,
                topo._dir_coeffs,
                topo._bond_len_col,
            )
        except ImportError:
            pass

    return _place_h_jit(
        xyz_heavy,
        topo.h_parent_idx,
        topo._nb_idx_clamped,
        topo._nb_valid,
        topo._dir_coeffs,
        topo._bond_len_col,
    )


# ---------------------------------------------------------------------------
# Build-time H candidate pair precomputation
# ---------------------------------------------------------------------------


def build_h_candidate_pairs(
    h_topo: HydrogenTopology,
    vdw_data: dict,
    pdb,
    h_excl_hash: torch.Tensor,
    device: torch.device = None,
    verbose: int = 0,
) -> None:
    """Precompute candidate H-involving VDW pairs from heavy-atom pair list.

    For each heavy-heavy VDW pair (A, B, symop, offset), derives candidate
    H-heavy pairs where H rides on A and could interact with B (or vice
    versa).  Applies exclusion and same-residue filters at build time so
    that the forward pass only needs to compute distances and energy.

    Results are stored as registered buffers on ``h_topo``:

    * ``cand_idx_i``      (C,) long — first atom (combined index)
    * ``cand_idx_j``      (C,) long — second atom (combined index)
    * ``cand_symop_idx``  (C,) long — symop index for the heavy atom
    * ``cand_cell_offset`` (C, 3) long — cell translation for the heavy atom
    * ``cand_min_dist``   (C,) float — VDW radius sum (H + heavy)

    Parameters
    ----------
    h_topo : HydrogenTopology
    vdw_data : dict
        Output of ``build_vdw_restraints_gpu`` (keys: indices, symop_indices,
        cell_offsets, etc.).
    pdb : DataFrame
        Heavy-atom DataFrame.
    h_excl_hash : (E,) long
        Sorted exclusion hash tensor for H-specific 1-2/1-3 pairs.
    device : torch.device
    verbose : int
    """
    if device is None:
        device = get_default_device()
    n_h = h_topo.n_hydrogens
    n_heavy = len(pdb)

    if n_h == 0:
        for name in ("cand_idx_i", "cand_idx_j", "cand_symop_idx"):
            h_topo.register_buffer(
                name, torch.zeros(0, dtype=torch.long, device=device)
            )
        h_topo.register_buffer(
            "cand_cell_offset", torch.zeros(0, 3, dtype=torch.long, device=device)
        )
        h_topo.register_buffer(
            "cand_min_dist", torch.zeros(0, dtype=dtypes.float, device=device)
        )
        return

    heavy_indices = vdw_data["indices"]  # (P, 2)
    heavy_symop = vdw_data["symop_indices"]  # (P,)
    heavy_offsets = vdw_data["cell_offsets"]  # (P, 3)

    parent_idx_np = h_topo.h_parent_idx.cpu().numpy()  # (N_h,)
    h_vdw_np = h_topo.h_vdw_radius.cpu().numpy()  # (N_h,)
    h_chain_np = h_topo.h_chainid_enc.cpu().numpy()  # (N_h,)
    h_resseq_np = h_topo.h_resseq.cpu().numpy()  # (N_h,)

    # Build parent → H index mapping
    parent_to_h = {}
    for hi in range(n_h):
        p = int(parent_idx_np[hi])
        parent_to_h.setdefault(p, []).append(hi)

    # Heavy atom chain/resseq for same-residue filter
    chain_vals = pdb["chainid"].values.astype(str)
    unique_chains = np.unique(chain_vals)
    chain_to_int = {c: i for i, c in enumerate(unique_chains)}
    heavy_chain_np = np.array(
        [chain_to_int.get(c, -1) for c in chain_vals], dtype=np.int64
    )
    heavy_resseq_np = pdb["resseq"].values.astype(np.int64)

    idx_A = heavy_indices[:, 0].cpu().numpy()
    idx_B = heavy_indices[:, 1].cpu().numpy()
    symop_np = heavy_symop.cpu().numpy()
    offsets_np = heavy_offsets.cpu().numpy()

    # Read VDW radii from the min_distances and indices of existing pairs
    # Instead, read from data file — we need per-atom VDW radii
    # Use h_vdw_radius for H, and derive heavy VDW radii from vdw_data
    # Since we have min_distances = radii[A] + radii[B], we can't easily
    # decompose.  Instead, just load the model's VDW radii.
    # The caller should pass these.  For now, compute from min_distances
    # and use a simple approach: store radii per heavy atom.

    # Actually, just use a lookup: min_dist_h_heavy = H_vdw + heavy_vdw
    # We need heavy_vdw per atom.  Derive from existing pair data:
    # For any pair (A, B): min_dist = radii[A] + radii[B]
    # We can solve if we have a self-pair, but we don't.  Instead, pass
    # heavy_vdw_radii directly from the restraints object.
    # For now, use a default approach: 1.7 Å for heavy atoms and refine later.
    # This will be overridden by the caller.

    # Candidate pairs stored as indices into the combined array:
    #   [0 .. n_heavy-1] = heavy atoms,  [n_heavy .. n_heavy+n_h-1] = H atoms
    # This way both H-heavy AND H-H pairs use the same format.
    acc_idx_i = []  # ASU atom (combined index)
    acc_idx_j = []  # partner atom (combined index, may need symop)
    acc_symop = []
    acc_offset = []

    def _same_res(chain_a, resseq_a, chain_b, resseq_b):
        return chain_a == chain_b and resseq_a == resseq_b

    for p_idx in range(len(idx_A)):
        A, B = int(idx_A[p_idx]), int(idx_B[p_idx])
        sym = int(symop_np[p_idx])
        off = offsets_np[p_idx]
        is_intra_asu = (sym == 0) and (off == 0).all()

        h_on_A = parent_to_h.get(A, [])
        h_on_B = parent_to_h.get(B, [])

        # --- H on A ↔ heavy B ---
        for hi in h_on_A:
            if is_intra_asu and _same_res(
                h_chain_np[hi], h_resseq_np[hi], heavy_chain_np[B], heavy_resseq_np[B]
            ):
                continue
            acc_idx_i.append(n_heavy + hi)
            acc_idx_j.append(B)
            acc_symop.append(sym)
            acc_offset.append(off)

        # --- H on B ↔ heavy A ---
        for hi in h_on_B:
            if is_intra_asu and _same_res(
                h_chain_np[hi], h_resseq_np[hi], heavy_chain_np[A], heavy_resseq_np[A]
            ):
                continue
            acc_idx_i.append(n_heavy + hi)
            acc_idx_j.append(A)
            acc_symop.append(0)
            acc_offset.append(np.zeros(3, dtype=np.int64))

        # --- H on A ↔ H on B  (H-H contacts) ---
        for hi_a in h_on_A:
            for hi_b in h_on_B:
                if is_intra_asu and _same_res(
                    h_chain_np[hi_a],
                    h_resseq_np[hi_a],
                    h_chain_np[hi_b],
                    h_resseq_np[hi_b],
                ):
                    continue
                # For intra-ASU, only keep hi_a < hi_b to avoid double-counting
                if is_intra_asu and hi_a >= hi_b:
                    continue
                acc_idx_i.append(n_heavy + hi_a)
                acc_idx_j.append(n_heavy + hi_b)
                acc_symop.append(sym)
                acc_offset.append(off)

    if not acc_idx_i:
        for name in ("cand_idx_i", "cand_idx_j", "cand_symop_idx"):
            h_topo.register_buffer(
                name, torch.zeros(0, dtype=torch.long, device=device)
            )
        h_topo.register_buffer(
            "cand_cell_offset", torch.zeros(0, 3, dtype=torch.long, device=device)
        )
        h_topo.register_buffer(
            "cand_min_dist", torch.zeros(0, dtype=dtypes.float, device=device)
        )
        return

    cand_i = torch.tensor(acc_idx_i, dtype=torch.long, device=device)
    cand_j = torch.tensor(acc_idx_j, dtype=torch.long, device=device)
    cand_sym = torch.tensor(acc_symop, dtype=torch.long, device=device)
    cand_off = torch.tensor(np.stack(acc_offset), dtype=torch.long, device=device)

    # Apply 1-2 / 1-3 exclusions for intra-ASU candidates
    if h_excl_hash is not None and len(h_excl_hash) > 0:
        is_intra = (cand_sym == 0) & (cand_off == 0).all(dim=1)
        if is_intra.any():
            max_idx = n_heavy + n_h
            norm_i = torch.minimum(cand_i, cand_j)
            norm_j = torch.maximum(cand_i, cand_j)
            pair_hash = norm_i * max_idx + norm_j
            ins = torch.searchsorted(h_excl_hash, pair_hash).clamp(
                max=len(h_excl_hash) - 1
            )
            is_excluded = (h_excl_hash[ins] == pair_hash) & is_intra
            keep = ~is_excluded
            cand_i = cand_i[keep]
            cand_j = cand_j[keep]
            cand_sym = cand_sym[keep]
            cand_off = cand_off[keep]

    # Deduplicate
    if len(cand_i) > 0:
        n_all = n_heavy + n_h
        dedup_key = (
            cand_i.long() * (n_all * 1000)
            + cand_j.long() * 1000
            + cand_sym.long() * 27
            + (cand_off[:, 0] + 1) * 9
            + (cand_off[:, 1] + 1) * 3
            + (cand_off[:, 2] + 1)
        )
        _, first_idx = torch.unique(dedup_key, return_inverse=True)
        # MPS does not support int64 scatter_reduce; use configured int dtype.
        _int_dtype = dtypes.int
        first_idx_i = first_idx.to(_int_dtype)
        perm = torch.arange(len(cand_i), device=device, dtype=_int_dtype)
        n_unique = first_idx.max().item() + 1
        first_occ = torch.full(
            (n_unique,), len(cand_i), dtype=_int_dtype, device=device
        )
        first_occ.scatter_reduce_(0, first_idx_i, perm, reduce="amin")
        mask = torch.zeros(len(cand_i), dtype=torch.bool, device=device)
        mask[first_occ.long()] = True
        cand_i = cand_i[mask]
        cand_j = cand_j[mask]
        cand_sym = cand_sym[mask]
        cand_off = cand_off[mask]

    # Sort: ASU candidates first, symmetry last
    is_asu = (cand_sym == 0) & (cand_off == 0).all(dim=1)
    sort_order = (~is_asu).long().argsort(stable=True)
    cand_i = cand_i[sort_order]
    cand_j = cand_j[sort_order]
    cand_sym = cand_sym[sort_order]
    cand_off = cand_off[sort_order]
    n_asu_cand = is_asu.sum().item()

    h_topo.register_buffer("cand_idx_i", cand_i)
    h_topo.register_buffer("cand_idx_j", cand_j)
    h_topo.register_buffer("cand_symop_idx", cand_sym)
    h_topo.register_buffer("cand_cell_offset", cand_off)
    h_topo.n_asu_candidates = n_asu_cand

    h_topo.register_buffer(
        "cand_min_dist",
        torch.zeros(len(cand_i), dtype=dtypes.float, device=device),
    )

    if verbose > 0:
        n_hh = ((cand_i >= n_heavy) & (cand_j >= n_heavy)).sum().item()
        n_sym = ((cand_sym != 0) | (cand_off != 0).any(dim=1)).sum().item()
        print(
            f"  H candidate pairs: {len(cand_i)} "
            f"({n_hh} H-H, {len(cand_i)-n_hh} H-heavy, {n_sym} symmetry)"
        )
