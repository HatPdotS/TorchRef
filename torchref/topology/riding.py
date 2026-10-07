"""Riding hydrogens: the sterics of hydrogens a model does not carry.

For a model whose atoms are heavy only (loaded with ``hydrogens="strip"``, or from a
file without hydrogens). A static map
built once at restraint-construction time says how to reconstruct each absent hydrogen
from its parent and the parent's bonded neighbours -- for a parent with a single heavy
neighbour, that neighbour and an atom bonded to it, so the template's torsion holds --
and ``place_riding_hydrogens`` then
produces those positions in one vectorized pass at every non-bonded evaluation and
throws them away again. The positions are a function of the heavy atoms, so gradients
reach the heavy coordinates through them by ordinary autograd.

Contrast :mod:`torchref.topology.hydrogens`, which *adds* hydrogens to the model as
real atoms with their own parameters (``hydrogens="add"``), the better answer where
both apply: the hydrogen has a refinable position instead of one reconstructed each
step, and it contributes to the structure factors. Riding hydrogens serve a model
without any -- the default ``hydrogens="keep"`` on a heavy-only file, or ``"strip"`` --
and the two must not run together: riding placement alongside real hydrogens puts
phantom atoms in the structure that push the real ones around.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

from torchref.base.coordinates.local_frame import frame_is_degenerate
from torchref.base.coordinates.symmetry_images import is_symmetry_image
from torchref.config import dtypes, get_int_dtype, normalize_device
from torchref.topology.nonbonded import IMAGE_PAIR_WEIGHT
from torchref.topology.residue_graph import build_residue_nodes
from torchref.utils.device_mixin import DeviceMixin
from torchref.utils.device_resolution import resolve_device

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

MAX_HEAVY_NB = 4  # Maximum heavy-atom neighbours to store per parent


# ---------------------------------------------------------------------------
# HydrogenTopology
# ---------------------------------------------------------------------------


@dataclass(eq=False, repr=False)
class HydrogenTopology(DeviceMixin):
    """Static topology describing riding hydrogens for VDW evaluation.

    Every tensor field starts as ``None``; :func:`build_hydrogen_topology` and
    :func:`build_h_candidate_pairs` fill them, independently -- a topology can carry
    hydrogens and no candidate pairs. Test :attr:`n_hydrogens` and
    :attr:`has_candidates` rather than the fields.

    Holds no refinable parameters, so this is a dataclass rather than an
    ``nn.Module``; ``DeviceMixin`` still moves every tensor with ``.to(device)``.

    Parameters
    ----------
    device : torch.device, optional
        Where the builders should allocate. Tracked from construction so a
        ``resolve_device(h_topo, ...)`` called before anything is attached still
        answers truthfully.

    Attributes
    ----------
    h_parent_idx : torch.Tensor
        Heavy-atom index of each riding H's parent, ``(N_h,)`` int.
    h_bond_length : torch.Tensor
        Ideal H-parent bond length in Angstroms, ``(N_h,)``.
    h_vdw_radius : torch.Tensor
        Van der Waals radius per H (1.20 A), ``(N_h,)``.
    h_placement_type : torch.Tensor
        Placement-geometry enum, ``(N_h,)`` int; see the module-level constants.
    h_slot_in_parent : torch.Tensor
        Ordinal among sibling H atoms on the same parent (0, 1, 2), ``(N_h,)`` int.
    parent_neighbor_idx : torch.Tensor
        Heavy-atom neighbours of the parent, ``(N_h, MAX_HEAVY_NB)`` int, ``-1``
        padded.
    parent_neighbor_count : torch.Tensor
        Heavy-atom neighbour count per parent, ``(N_h,)`` int.
    h_frame_atom : torch.Tensor
        For a hydrogen whose parent has one heavy neighbour, an atom bonded to that
        neighbour, completing the ``(parent, neighbour, frame atom)`` frame the
        hydrogen rides in; ``-1`` otherwise, or where no such atom is found, which
        leaves the rotation about the parent-neighbour bond arbitrary. ``(N_h,)`` int.
    h_frame_direction : torch.Tensor
        Unit parent-to-hydrogen direction in that frame's
        :func:`~torchref.base.coordinates.local_frame_axes`, read off the template,
        so the hydrogen keeps the template's bond angle and torsion (turned into the
        plane where a plane restraint holds it). ``(N_h, 3)``.
    type_bounds : dict
        ``{placement_type: (start, end)}`` bounds into the type-sorted arrays.
    cand_idx_i, cand_idx_j, cand_symop_idx, cand_cell_offset : torch.Tensor
        Precomputed H candidate pairs.
    cand_min_dist : torch.Tensor
        Per-pair minimum-distance scratch buffer, ``(P,)``.
    cand_weight : torch.Tensor
        Loss weight per candidate pair, ``(P,)``:
        :data:`~torchref.topology.nonbonded.IMAGE_PAIR_WEIGHT` for an H-H crystal
        contact, which the list holds from both of its ends, 1 otherwise.
    """

    device: Optional[torch.device] = None

    h_parent_idx: Optional[torch.Tensor] = None
    h_bond_length: Optional[torch.Tensor] = None
    h_vdw_radius: Optional[torch.Tensor] = None
    h_placement_type: Optional[torch.Tensor] = None
    h_slot_in_parent: Optional[torch.Tensor] = None
    parent_neighbor_idx: Optional[torch.Tensor] = None
    parent_neighbor_count: Optional[torch.Tensor] = None
    h_frame_atom: Optional[torch.Tensor] = None
    h_frame_direction: Optional[torch.Tensor] = None
    type_bounds: Dict[int, tuple] = field(default_factory=dict)

    cand_idx_i: Optional[torch.Tensor] = None
    cand_idx_j: Optional[torch.Tensor] = None
    cand_symop_idx: Optional[torch.Tensor] = None
    cand_cell_offset: Optional[torch.Tensor] = None
    cand_min_dist: Optional[torch.Tensor] = None
    cand_weight: Optional[torch.Tensor] = None

    # Derived at first placement and reused across steps; see reset_cache.
    _dir_coeffs: Optional[torch.Tensor] = field(default=None, repr=False)
    _nb_idx_clamped: Optional[torch.Tensor] = field(default=None, repr=False)
    _nb_valid: Optional[torch.Tensor] = field(default=None, repr=False)
    _bond_len_col: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Resolve the device tracker the builders allocate against."""
        self.device = normalize_device(self.device)

    @property
    def n_hydrogens(self) -> int:
        """Number of riding hydrogens, or 0 before the builders have run."""
        if self.h_parent_idx is None:
            return 0
        return int(self.h_parent_idx.shape[0])

    @property
    def has_candidates(self) -> bool:
        """Whether precomputed H candidate pairs are available."""
        return self.cand_idx_i is not None and self.cand_idx_i.shape[0] > 0

    def reset_cache(self) -> None:
        """Drop the derived placement tensors; rebuilt on the next placement call.

        Called by ``DeviceMixin`` on every ``.to()``, which is what keeps the clamped
        neighbour indices and bond-length column from surviving a device move.
        """
        self._dir_coeffs = None
        self._nb_idx_clamped = None
        self._nb_valid = None
        self._bond_len_col = None

    def __repr__(self) -> str:
        return (
            f"HydrogenTopology(n_hydrogens={self.n_hydrogens}, "
            f"has_candidates={self.has_candidates})"
        )


# ---------------------------------------------------------------------------
# Build-time topology construction
# ---------------------------------------------------------------------------


def _template_frame(
    info: Dict,
    parent_name: str,
    neighbour: int,
    h_names,
    model_names: np.ndarray,
    name_to_global: Dict[str, int],
    model_xyz: np.ndarray,
) -> Optional[tuple]:
    """Frame atom and template directions for hydrogens on a single-neighbour parent.

    The frame atom is the neighbour's first template neighbour, other than the parent,
    that the residue holds; the directions are the template's, in the
    ``(parent, neighbour, frame atom)`` local frame
    (:func:`torchref.topology.hydrogens._frame_directions`, which also turns a planar
    group into its plane).

    Returns
    -------
    tuple or None
        ``(frame atom row, directions of shape (len(h_names), 3))``; None when the
        template or the model lacks such an atom, or either frame is degenerate.
    """
    from torchref.topology.hydrogens import _frame_directions

    neighbour_name = model_names[neighbour]
    if name_to_global.get(neighbour_name) != neighbour:
        return None
    rows = sorted(
        name_to_global[name]
        for name in info["heavy_adjacency"].get(neighbour_name, [])
        if name != parent_name and name in name_to_global
    )
    if not rows:
        return None
    frame_names = (parent_name, neighbour_name, model_names[rows[0]])
    index, coords = info["id_to_index"], info["coords"]
    if any(name not in index for name in (*frame_names, *h_names)):
        return None
    directions = _frame_directions(
        np.stack([coords[index[name]] for name in frame_names]),
        np.stack([coords[index[name]] for name in h_names]),
        bool(info["planar_h"] & set(h_names)),
    )
    if directions is None:
        return None
    model = model_xyz[[name_to_global[parent_name], neighbour, rows[0]]]
    model = torch.as_tensor(model - model[0], dtype=directions.dtype)
    if frame_is_degenerate(*model[:, None].unbind(0)).any():
        return None
    return rows[0], directions.numpy()


def _classify_placement(n_h_on_parent: int, n_heavy_nb: int, slot: int) -> int:
    """Placement-type code for an H atom, or -1 if the parent has no heavy
    neighbour and the geometry is therefore undetermined."""
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
    cif_dict: Optional[Dict] = None,
) -> HydrogenTopology:
    """Build riding-hydrogen topology from the model's heavy-atom DataFrame.

    Parameters
    ----------
    pdb : pd.DataFrame
        Heavy-atom DataFrame (no hydrogen rows).
    device : torch.device
        Target device for tensors.
    verbose : int
        Verbosity level.
    cif_dict : dict, optional
        Restraint dictionary keyed by residue name, such as ``Restraints.cif_dict``;
        each residue's hydrogens are read from its template there. None looks every
        residue up in the monomer library instead, whose last resort is a download.

    Returns
    -------
    HydrogenTopology
        Placement fields set; :func:`build_h_candidate_pairs` adds the candidate pairs.
    """
    from torchref.topology.hydrogens import _template

    device = normalize_device(device)
    resnames = pdb["resname"].astype(str).str.strip().unique()
    if cif_dict is None:
        from torchref.topology.monomer.cif import find_cif_file_in_library, read_cif

        cif_dict = {}
        for resname in resnames:
            path = find_cif_file_in_library(resname) if resname else None
            if path is None:
                continue
            try:
                cif_dict.update(read_cif(str(path)))
            except Exception:
                # An unreadable entry leaves its residue without riding hydrogens,
                # as a missing one does.
                continue
    templates = {resname: _template(cif_dict, resname) for resname in resnames}

    model_names = pdb["name"].astype(str).str.strip().values
    model_xyz = pdb[["x", "y", "z"]].values.astype(np.float64)
    model_elem = pdb["element"].astype(str).str.strip().values
    model_heavy_mask = np.array([e.upper() != "H" for e in model_elem])

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
    acc_frame_atom = []
    acc_frame_direction = []

    for gi in range(len(group_starts)):
        s, e = group_starts[gi], group_ends[gi]
        rn = str(group_keys[s, 3]).strip()
        info = templates.get(rn)
        if info is None:
            continue

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
            pn = info["parent_of"].get(h_name)
            if pn is not None and pn in name_to_global:
                parent_to_h.setdefault(pn, []).append(h_name)

        id2i = info["id_to_index"]

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
            par_elem = info["elements"][id2i[par_name]].upper()
            expected_h = max(0, _std_val.get(par_elem, 4) - n_model_heavy)
            h_list_capped = sorted(h_list)[:expected_h]
            if not h_list_capped:
                continue

            n_h = len(h_list_capped)
            frame_atom, directions = -1, np.zeros((n_h, 3))
            if n_model_heavy == 1:
                frame = _template_frame(
                    info,
                    par_name,
                    int(bonded[0]),
                    h_list_capped,
                    model_names,
                    name_to_global,
                    model_xyz,
                )
                if frame is not None:
                    frame_atom, directions = frame

            # Neighbour index array (padded)
            nb_arr = np.full(MAX_HEAVY_NB, -1, dtype=np.int64)
            nb_count = min(n_model_heavy, MAX_HEAVY_NB)
            nb_arr[:nb_count] = bonded[:nb_count]

            for slot, h_name in enumerate(h_list_capped):
                bl = info["ideal_length"].get(h_name, 0.97)
                ptype = _classify_placement(n_h, n_model_heavy, slot)

                if ptype < 0:
                    continue  # skip — cannot determine geometry

                acc_parent_idx.append(pidx)
                acc_bond_length.append(bl)
                acc_placement_type.append(ptype)
                acc_slot.append(slot)
                acc_nb_idx.append(nb_arr.copy())
                acc_nb_count.append(nb_count)
                acc_frame_atom.append(frame_atom)
                acc_frame_direction.append(directions[slot])

    # Seed the tracker with the device its buffers are about to be built on,
    # so a later ``resolve_device(h_topo, ...)`` sees the truth.
    topo = HydrogenTopology(device=device)
    n_h_total = len(acc_parent_idx)
    fdtype = dtypes.float

    if n_h_total == 0:
        topo.h_parent_idx = torch.zeros(0, dtype=get_int_dtype(), device=device)
        topo.h_bond_length = torch.zeros(0, dtype=fdtype, device=device)
        topo.h_vdw_radius = torch.zeros(0, dtype=fdtype, device=device)
        topo.h_placement_type = torch.zeros(0, dtype=get_int_dtype(), device=device)
        topo.h_slot_in_parent = torch.zeros(0, dtype=get_int_dtype(), device=device)
        topo.parent_neighbor_idx = torch.zeros(
            0, MAX_HEAVY_NB, dtype=get_int_dtype(), device=device
        )
        topo.parent_neighbor_count = torch.zeros(
            0, dtype=get_int_dtype(), device=device
        )
        topo.h_frame_atom = torch.zeros(0, dtype=get_int_dtype(), device=device)
        topo.h_frame_direction = torch.zeros(0, 3, dtype=fdtype, device=device)
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
    acc_frame_atom = [acc_frame_atom[i] for i in sort_order]
    acc_frame_direction = [acc_frame_direction[i] for i in sort_order]

    # Compute type boundaries: type_bounds[t] = (start, end) slice
    sorted_types = np.array(acc_placement_type, dtype=np.int64)
    type_bounds = {}
    for t in range(7):
        mask = sorted_types == t
        if mask.any():
            idxs = np.where(mask)[0]
            type_bounds[t] = (int(idxs[0]), int(idxs[-1]) + 1)

    topo.h_parent_idx = torch.tensor(
        acc_parent_idx, dtype=get_int_dtype(), device=device
    )
    topo.h_bond_length = torch.tensor(acc_bond_length, dtype=fdtype, device=device)
    topo.h_vdw_radius = torch.full((n_h_total,), 1.20, dtype=fdtype, device=device)
    topo.h_placement_type = torch.tensor(
        acc_placement_type, dtype=get_int_dtype(), device=device
    )
    topo.h_slot_in_parent = torch.tensor(acc_slot, dtype=get_int_dtype(), device=device)
    topo.parent_neighbor_idx = torch.tensor(
        np.stack(acc_nb_idx), dtype=get_int_dtype(), device=device
    )
    topo.parent_neighbor_count = torch.tensor(
        acc_nb_count, dtype=get_int_dtype(), device=device
    )
    topo.h_frame_atom = torch.tensor(
        acc_frame_atom, dtype=get_int_dtype(), device=device
    )
    topo.h_frame_direction = torch.tensor(
        np.stack(acc_frame_direction), dtype=fdtype, device=device
    )
    topo.type_bounds = type_bounds  # dict: type_code -> (start, end)

    if verbose > 0:
        print(f"  Hydrogen topology: {n_h_total} riding H atoms")

    return topo


# ---------------------------------------------------------------------------
# Vectorized H placement (forward-time, differentiable)
# ---------------------------------------------------------------------------


def _precompute_direction_coefficients(topo: HydrogenTopology) -> torch.Tensor:
    """(N_h, 3) of ``(c_base, c_perp1, c_perp2)``, zeros if ``type_bounds`` is unset.

    Every riding-H direction is ``c0·base + c1·perp1 + c2·perp2`` in a frame built
    from neighbour vectors; the coefficients depend only on placement type, or for a
    framed hydrogen on its template direction, so they are constant across refinement
    steps.
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
    # A framed hydrogen's kernel frame is (base, perp1, perp2) = (-e1, e3, e2) of the
    # local frame its template direction is expressed in; see _kernel_neighbours.
    framed = topo.h_frame_atom >= 0
    direction = topo.h_frame_direction[framed]
    coeffs[framed] = torch.stack(
        [-direction[:, 0], direction[:, 2], direction[:, 1]], dim=1
    )
    return coeffs


def _kernel_neighbours(topo: HydrogenTopology) -> tuple:
    """Neighbour slots and slot weights the placement kernels read, ``(N_h, 4)`` each.

    The kernels take ``base`` from the weighted sum of the slot vectors and ``perp1``
    from the cross product of the first two. A framed hydrogen lists its frame atom in
    slots 1 and 2 at weights +1 and -1: it orients ``perp1`` but cancels out of the
    sum, so ``base`` still points away from the one real neighbour. Padding has
    weight 0.
    """
    index = topo.parent_neighbor_idx.clone()
    weight = (index >= 0).to(topo.h_bond_length.dtype)
    framed = topo.h_frame_atom >= 0
    index[framed, 1] = topo.h_frame_atom[framed]
    index[framed, 2] = topo.h_frame_atom[framed]
    weight[framed, 1] = 1.0
    weight[framed, 2] = -1.0
    return index, weight


@torch.jit.script
def _place_h_jit(
    xyz_heavy: torch.Tensor,
    h_parent_idx: torch.Tensor,
    nb_idx_clamped: torch.Tensor,
    nb_valid: torch.Tensor,
    coeffs: torch.Tensor,
    bond_length: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled H placement kernel; returns (N_h, 3) H positions.

    ``nb_idx_clamped`` (N_h, 4) must already have -1 padding clamped to 0, with
    ``nb_valid`` (N_h, 4, 1) weighting each slot (:func:`_kernel_neighbours`): 0.0
    masks the padding back out -- passing raw -1 indices reads the wrong atoms
    instead of failing. ``coeffs`` is (N_h, 3) from
    :func:`_precompute_direction_coefficients`.
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
    substantially reducing the number of GPU kernel launches.

    Parameters
    ----------
    xyz_heavy : (N_heavy, 3) float tensor (requires_grad typically True)
    topo : HydrogenTopology

    Returns
    -------
    xyz_h : (N_h, 3) float tensor, differentiable w.r.t. xyz_heavy
    """
    # Function-local, matching every other target dispatch site: importing the gate at
    # module scope would pull ``torchref.base.targets`` into ``torchref.topology``.
    from torchref.base.targets._dispatch import use_triton

    N_h = topo.h_parent_idx.shape[0]
    if N_h == 0:
        return torch.zeros(0, 3, dtype=xyz_heavy.dtype, device=xyz_heavy.device)

    # Precompute direction coefficients on first call
    if topo._dir_coeffs is None:
        topo._dir_coeffs = _precompute_direction_coefficients(topo)

    # Precompute static tensors on first call (avoid recomputing every step)
    if topo._nb_idx_clamped is None:
        index, weight = _kernel_neighbours(topo)
        topo._nb_idx_clamped = index.clamp(min=0)
        topo._nb_valid = weight.unsqueeze(-1)
        topo._bond_len_col = topo.h_bond_length.unsqueeze(-1)

    # On CUDA fp32 use the fused Triton forward + analytic Triton
    # backward (the H-VDW backward is a significant fraction of the
    # non-bonded fwd+bw cost). Falls back to the JIT-scripted eager
    # helper otherwise.
    #
    # Gated by the shared ``use_triton`` rather than a hand-rolled
    # ``is_cuda and dtype == float32``. The inline check ignored the shared gate entirely, so
    # a block that pinned the portable path still ran the Triton kernel here -- and since
    # EAGER is the documented double-differentiable route, a Hessian taken through hydrogen
    # placement was silently going through a first-order-only kernel.
    if use_triton(xyz_heavy):
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
    """Precompute candidate H-involving VDW pairs from the heavy-atom pair list.

    From each heavy-heavy pair (A, B, symop, offset), derives the H-heavy pairs
    where an H riding on A could reach B, and for intra-ASU pairs vice versa,
    applying the exclusion and same-residue filters now so the forward pass only
    computes distances. An image pair's other direction comes from its reverse
    entry, B against A under the inverse operation, so the heavy list must hold both
    directions of every image contact, as ``build_vdw_restraints_gpu`` emits them.
    Mutates ``h_topo`` in place, setting ``cand_idx_i``/``cand_idx_j`` (combined-array
    atom indices), ``cand_symop_idx`` and ``cand_cell_offset`` (the image of the
    ``cand_idx_j`` end), ``cand_weight``, and ``cand_min_dist`` as zeros for the caller
    to fill with the contact distances (:func:`candidate_contact_distances`).

    Parameters
    ----------
    h_topo : HydrogenTopology
    vdw_data : dict
        Output of ``build_vdw_restraints_gpu`` (keys: indices, symop_indices,
        cell_offsets, etc.).
    pdb : DataFrame
        Heavy-atom table in the order of the pair list's atom indices; its
        ``chainid``, ``resseq``, ``icode`` and ``resname`` columns give the residues.
    h_excl_hash : (E,) long
        Sorted exclusion hash tensor for H-specific 1-2/1-3 pairs.
    device : torch.device
    verbose : int
    """
    # The candidate tensors are set on ``h_topo`` below, so they follow its device
    # rather than the global default.
    device = resolve_device(h_topo, device=device)
    n_h = h_topo.n_hydrogens
    n_heavy = len(pdb)

    if n_h == 0:
        for name in ("cand_idx_i", "cand_idx_j", "cand_symop_idx"):
            setattr(h_topo, name, torch.zeros(0, dtype=get_int_dtype(), device=device))
        h_topo.cand_cell_offset = torch.zeros(
            0, 3, dtype=get_int_dtype(), device=device
        )
        h_topo.cand_min_dist = torch.zeros(0, dtype=dtypes.float, device=device)
        h_topo.cand_weight = torch.zeros(0, dtype=dtypes.float, device=device)
        return

    heavy_indices = vdw_data["indices"]  # (P, 2)
    heavy_symop = vdw_data["symop_indices"]  # (P,)
    heavy_offsets = vdw_data["cell_offsets"]  # (P, 3)

    parent_idx_np = h_topo.h_parent_idx.cpu().numpy()  # (N_h,)

    # Build parent → H index mapping
    parent_to_h = {}
    for hi in range(n_h):
        p = int(parent_idx_np[hi])
        parent_to_h.setdefault(p, []).append(hi)

    # The topology's residues, keyed (chain, resseq, icode): 100 and 100A are two.
    nodes = build_residue_nodes(
        *(pdb[column].values for column in ("chainid", "resseq", "icode", "resname"))
    )
    residue_of = np.repeat(
        np.arange(len(nodes["chain"])), nodes["atom_end"] - nodes["atom_start"]
    )

    idx_A = heavy_indices[:, 0].cpu().numpy()
    idx_B = heavy_indices[:, 1].cpu().numpy()
    symop_np = heavy_symop.cpu().numpy()
    offsets_np = heavy_offsets.cpu().numpy()
    is_image_np = is_symmetry_image(heavy_symop, heavy_offsets).cpu().numpy()

    # Contact distances are not computed here: cand_min_dist is allocated as zeros
    # below for the caller, which has the model's radii and hydrogen-bond roles.

    # Candidate pairs stored as indices into the combined array:
    #   [0 .. n_heavy-1] = heavy atoms,  [n_heavy .. n_heavy+n_h-1] = H atoms
    # This way both H-heavy AND H-H pairs use the same format.
    acc_idx_i = []  # ASU atom (combined index)
    acc_idx_j = []  # partner atom (combined index, may need symop)
    acc_symop = []
    acc_offset = []

    for p_idx in range(len(idx_A)):
        A, B = int(idx_A[p_idx]), int(idx_B[p_idx])
        sym = int(symop_np[p_idx])
        off = offsets_np[p_idx]
        is_intra_asu = not is_image_np[p_idx]
        # A hydrogen is in its parent's residue, so every candidate an intra-ASU
        # same-residue pair would give is a same-residue contact.
        if is_intra_asu and residue_of[A] == residue_of[B]:
            continue

        h_on_A = parent_to_h.get(A, [])
        h_on_B = parent_to_h.get(B, [])

        # --- H on A ↔ heavy B ---
        for hi in h_on_A:
            acc_idx_i.append(n_heavy + hi)
            acc_idx_j.append(B)
            acc_symop.append(sym)
            acc_offset.append(off)

        # --- H on B ↔ heavy A ---
        # Intra-ASU pairs only. An image pair -- A against B under (symop, offset)
        # -- is listed together with B against A under the inverse operation, whose
        # "H on A" branch above emits this contact with the image on the right atom.
        # From here it could only carry this pair's operation, which images B, not A.
        if is_intra_asu:
            for hi in h_on_B:
                acc_idx_i.append(n_heavy + hi)
                acc_idx_j.append(A)
                acc_symop.append(0)
                acc_offset.append(np.zeros(3, dtype=np.int64))

        # --- H on A ↔ H on B  (H-H contacts) ---
        # An intra-ASU heavy pair is listed once, so each of its H-H contacts is too.
        for hi_a in h_on_A:
            for hi_b in h_on_B:
                acc_idx_i.append(n_heavy + hi_a)
                acc_idx_j.append(n_heavy + hi_b)
                acc_symop.append(sym)
                acc_offset.append(off)

    if not acc_idx_i:
        for name in ("cand_idx_i", "cand_idx_j", "cand_symop_idx"):
            setattr(h_topo, name, torch.zeros(0, dtype=get_int_dtype(), device=device))
        h_topo.cand_cell_offset = torch.zeros(
            0, 3, dtype=get_int_dtype(), device=device
        )
        h_topo.cand_min_dist = torch.zeros(0, dtype=dtypes.float, device=device)
        h_topo.cand_weight = torch.zeros(0, dtype=dtypes.float, device=device)
        return

    cand_i = torch.tensor(acc_idx_i, dtype=get_int_dtype(), device=device)
    cand_j = torch.tensor(acc_idx_j, dtype=get_int_dtype(), device=device)
    cand_sym = torch.tensor(acc_symop, dtype=get_int_dtype(), device=device)
    cand_off = torch.tensor(np.stack(acc_offset), dtype=get_int_dtype(), device=device)

    # Apply 1-2 / 1-3 exclusions for intra-ASU candidates
    if h_excl_hash is not None and len(h_excl_hash) > 0:
        is_intra = ~is_symmetry_image(cand_sym, cand_off)
        if is_intra.any():
            max_idx = n_heavy + n_h
            norm_i = torch.minimum(cand_i, cand_j)
            norm_j = torch.maximum(cand_i, cand_j)
            # dtype-ok: packed pair key overflows int32; searchsorted needs int64 like the table
            pair_hash = norm_i.to(torch.int64) * max_idx + norm_j.to(torch.int64)
            ins = torch.searchsorted(h_excl_hash, pair_hash).clamp(
                max=len(h_excl_hash) - 1
            )
            is_excluded = (h_excl_hash[ins] == pair_hash) & is_intra
            keep = ~is_excluded
            cand_i = cand_i[keep]
            cand_j = cand_j[keep]
            cand_sym = cand_sym[keep]
            cand_off = cand_off[keep]

    # Deduplicate on whole (i, j, symop, offset) rows. No fixed-stride packed key is
    # safe: offsets are not confined to -1..1, nor operations to a small count.
    if len(cand_i) > 0:
        rows = torch.cat(
            [torch.stack([cand_i, cand_j, cand_sym], dim=1), cand_off], dim=1
        )
        _, first_idx = torch.unique(rows, dim=0, return_inverse=True)
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

    h_topo.cand_idx_i = cand_i
    h_topo.cand_idx_j = cand_j
    h_topo.cand_symop_idx = cand_sym
    h_topo.cand_cell_offset = cand_off

    h_topo.cand_min_dist = torch.zeros(len(cand_i), dtype=dtypes.float, device=device)

    # An H-H crystal contact comes from an image heavy pair and from its reverse, so
    # it is listed from both ends; an H-heavy one only from its hydrogen's end, since
    # the reverse heavy pair gives the other atom's hydrogens instead.
    image = is_symmetry_image(cand_sym, cand_off)
    both_h = (cand_i >= n_heavy) & (cand_j >= n_heavy)
    h_topo.cand_weight = torch.where(image & both_h, IMAGE_PAIR_WEIGHT, 1.0).to(
        dtypes.float
    )

    if verbose > 0:
        n_hh = both_h.sum().item()
        n_sym = image.sum().item()
        print(
            f"  H candidate pairs: {len(cand_i)} "
            f"({n_hh} H-H, {len(cand_i)-n_hh} H-heavy, {n_sym} symmetry)"
        )


def candidate_contact_distances(
    h_topo: HydrogenTopology,
    heavy_radii: torch.Tensor,
    heavy_roles: Optional[torch.Tensor],
) -> torch.Tensor:
    """Minimum contact distance of every candidate pair, for ``cand_min_dist``.

    :func:`~torchref.topology.nonbonded.contact_distances` over the combined
    ``[heavy | riding H]`` atoms the candidate indices address: each riding hydrogen
    takes its ``h_vdw_radius``, and the polar-hydrogen role when its parent is a
    hydrogen-bond donor, so an N-H...O hydrogen bond is not scored as a clash.

    Parameters
    ----------
    h_topo : HydrogenTopology
        Riding topology with candidate pairs built.
    heavy_radii : torch.Tensor
        Contact radius per heavy atom in Å, shape ``(N_heavy,)``.
    heavy_roles : torch.Tensor or None
        ``AtomGraph.hb_type`` of the heavy atoms, shape ``(N_heavy,)``, on any device;
        None scores every pair by its radius sum.

    Returns
    -------
    torch.Tensor
        Distances in Å, shape ``(P,)``, in ``heavy_radii``' dtype.
    """
    from torchref.topology.nonbonded import contact_distances, hydrogen_roles

    radii = torch.cat([heavy_radii, h_topo.h_vdw_radius.to(heavy_radii)])
    roles = None
    if heavy_roles is not None:
        heavy_roles = heavy_roles.to(heavy_radii.device)
        riding = hydrogen_roles(heavy_roles[h_topo.h_parent_idx])
        roles = torch.cat([heavy_roles, riding])
    pairs = torch.stack([h_topo.cand_idx_i, h_topo.cand_idx_j], dim=1)
    return contact_distances(radii, roles, pairs)
