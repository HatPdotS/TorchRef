"""
Closed segmented internal coordinate parametrization for atomic structures.

This module provides the ClosedSegmentedInternalCoordinateTensor class which
extends the segmented approach with analytical chain closure. Between larger
segments (~18 residues each), 3-residue "junctions" are placed whose backbone
torsions (phi, psi) are slave DOFs solved by Newton's method. Backward
gradients flow via the Implicit Function Theorem, giving exact gradients
without differentiating through the solver.

Key differences from SegmentedInternalCoordinateTensor:
- Larger segments (default 18 residues vs 3)
- Junction residues between segments maintain chain continuity
- Junction phi/psi are solved, not free parameters
- IFT provides exact gradients through the closure constraint
- Junctions preferentially placed in loop regions
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torchref.base.alignment.rotation import rotation_matrix_euler_zyz
from torchref.base.chain_closure import (
    identify_backbone_atoms,
    get_chain_residues,
    compute_backbone_torsions,
    estimate_secondary_structure,
    plan_junction_placement,
    get_junction_backbone_indices,
    backbone_fk_junction,
    closure_residual,
    JunctionSolver,
    AA_NAMES,
)
from torchref.utils.caching import CachedForwardMixin

logger = logging.getLogger(__name__)


class ClosedSegmentedInternalCoordinateTensor(CachedForwardMixin, nn.Module):
    """
    Parameter wrapper using segmented internal coordinates with chain closure.

    Stores: per-segment bond_lengths, angles, torsions, segment_positions,
    segment_orientations. Junction backbone torsions are slave DOFs solved
    by Newton's method with IFT gradients.

    Parameters
    ----------
    initial_xyz : torch.Tensor
        Initial Cartesian coordinates of shape (N, 3).
    pdb : pd.DataFrame
        PDB DataFrame with columns 'chainid', 'resseq', 'name', 'index', 'resname'.
    n_aa_per_segment : int, optional
        Number of amino acids per segment. Default is 18.
    junction_size : int, optional
        Number of residues per junction. Default is 3.
    bond_cutoff : float, optional
        Distance cutoff for bond detection in Angstroms. Default is 2.0.
    cif_dict : dict, optional
        CIF dictionary containing bond definitions per residue type.
    prefer_loops : bool, optional
        If True, slide junctions to prefer loop regions. Default is True.
    requires_grad : bool, optional
        Whether parameters should have gradients. Default is True.
    dtype : torch.dtype, optional
        Data type for tensors.
    device : torch.device, optional
        Device for tensors.
    """

    def __init__(
        self,
        initial_xyz: torch.Tensor,
        pdb: pd.DataFrame,
        n_aa_per_segment: int = 18,
        junction_size: int = 3,
        bond_cutoff: float = 2.0,
        cif_dict: Optional[dict] = None,
        prefer_loops: bool = True,
        requires_grad: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if dtype is None:
            dtype = initial_xyz.dtype
        if device is None:
            device = initial_xyz.device

        self._dtype = dtype
        self._device = device
        self.n_atoms = initial_xyz.shape[0]
        self.bond_cutoff = bond_cutoff
        self.n_aa_per_segment = n_aa_per_segment
        self.junction_size = junction_size
        self.prefer_loops = prefer_loops
        self.cif_dict = cif_dict

        initial_xyz = initial_xyz.to(dtype=dtype, device=device)
        self.pdb = pdb

        # Step 1: Identify backbone atoms and plan segmentation
        self._identify_backbone_and_plan(pdb, initial_xyz)

        # Step 2: Build segments (assigns atoms to segments, finds roots)
        self._build_segments_fast(pdb)

        # Step 3: Build BFS spanning trees for all segments
        self._build_segment_trees_parallel(initial_xyz)

        # Step 4: Extract backbone geometry for junctions
        self._extract_junction_geometry(initial_xyz)

        # Step 5: Extract internal coordinates
        self._extract_internal_coords(initial_xyz, requires_grad)

        # Step 6: Initialize junction solver
        self._init_junction_solver(initial_xyz)

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    # =========================================================================
    # Initialization methods
    # =========================================================================

    def _identify_backbone_and_plan(
        self, pdb: pd.DataFrame, xyz: torch.Tensor
    ) -> None:
        """Identify backbone atoms, compute torsions, plan junctions."""
        self._backbone_map = identify_backbone_atoms(pdb)
        self._chain_residues = get_chain_residues(pdb)

        # Compute backbone torsions for secondary structure estimation
        torsions = compute_backbone_torsions(
            xyz.detach(), self._backbone_map, self._chain_residues
        )
        ss = estimate_secondary_structure(torsions) if self.prefer_loops else None

        # Plan junction placement
        segments, junctions = plan_junction_placement(
            self._chain_residues,
            self._backbone_map,
            n_aa_per_segment=self.n_aa_per_segment,
            junction_size=self.junction_size,
            ss=ss,
            prefer_loops=self.prefer_loops,
        )

        self._planned_segments = segments
        self._planned_junctions = junctions
        self.n_junctions = len(junctions)

        # Build sets of junction atom indices for quick lookup
        self._junction_atom_set = set()
        self._junction_backbone_set = set()
        self._junction_residue_keys = set()
        self._junction_data = []  # List of dicts with junction info

        for junc_idx, junc_residues in enumerate(junctions):
            junc_backbone = get_junction_backbone_indices(
                junc_residues, self._backbone_map
            )
            junc_info = {
                "residues": junc_residues,
                "backbone": junc_backbone,  # List of {N: idx, CA: idx, C: idx}
            }
            self._junction_data.append(junc_info)

            for res_key in junc_residues:
                self._junction_residue_keys.add(res_key)
                bb = self._backbone_map[res_key]
                for atom_name in ("N", "CA", "C"):
                    self._junction_backbone_set.add(bb[atom_name])

            # All atoms in junction residues (including sidechains)
            for res_key in junc_residues:
                chainid, resseq = res_key
                mask = (pdb["chainid"] == chainid) & (pdb["resseq"] == resseq)
                for idx in pdb.loc[mask, "index"].values:
                    self._junction_atom_set.add(int(idx))

        # Store junction atom indices as tensor
        if self._junction_atom_set:
            self.register_buffer(
                "junction_atom_indices",
                torch.tensor(
                    sorted(self._junction_atom_set),
                    dtype=torch.long, device=self._device,
                ),
            )
            self.register_buffer(
                "junction_backbone_indices",
                torch.tensor(
                    sorted(self._junction_backbone_set),
                    dtype=torch.long, device=self._device,
                ),
            )
        else:
            self.register_buffer(
                "junction_atom_indices",
                torch.tensor([], dtype=torch.long, device=self._device),
            )
            self.register_buffer(
                "junction_backbone_indices",
                torch.tensor([], dtype=torch.long, device=self._device),
            )

    def _build_segments_fast(self, pdb: pd.DataFrame) -> None:
        """
        Group atoms into segments based on planned segmentation.

        Uses the pre-planned segments and junctions. Each junction is its own
        segment for BFS purposes but its backbone torsions are excluded from
        the free DOF set.
        """
        device = self._device

        # Build lookup for atom names
        n_mask = pdb["name"].str.strip() == "N"
        n_lookup = {
            (c, r): i for c, r, i in zip(
                pdb.loc[n_mask, "chainid"].values,
                pdb.loc[n_mask, "resseq"].values,
                pdb.loc[n_mask, "index"].values,
            )
        }
        ca_mask = pdb["name"].str.strip() == "CA"
        ca_lookup = {
            (c, r): i for c, r, i in zip(
                pdb.loc[ca_mask, "chainid"].values,
                pdb.loc[ca_mask, "resseq"].values,
                pdb.loc[ca_mask, "index"].values,
            )
        }

        # Build atom lookup by residue key
        residue_atoms = {}
        for (chainid, resseq), group in pdb.groupby(["chainid", "resseq"]):
            residue_atoms[(chainid, resseq)] = group["index"].values.tolist()

        # Build segments: planned_segments + junction segments + non-protein
        segments = []
        segment_root_atoms = []
        segment_types = []  # 'segment', 'junction', 'non_protein'

        # Map junction index to segment index
        self._junction_to_segment = {}

        # Add planned segments (free DOF segments)
        for seg_residues in self._planned_segments:
            seg_atoms = []
            for res_key in seg_residues:
                if res_key in residue_atoms:
                    seg_atoms.extend(residue_atoms[res_key])

            if not seg_atoms:
                continue

            first_res = seg_residues[0]
            if first_res in n_lookup:
                root = n_lookup[first_res]
            elif first_res in ca_lookup:
                root = ca_lookup[first_res]
            else:
                root = seg_atoms[0]

            segments.append(seg_atoms)
            segment_root_atoms.append(root)
            segment_types.append("segment")

        # Add junction segments
        for junc_idx, junc_residues in enumerate(self._planned_junctions):
            junc_atoms = []
            for res_key in junc_residues:
                if res_key in residue_atoms:
                    junc_atoms.extend(residue_atoms[res_key])

            if not junc_atoms:
                continue

            first_res = junc_residues[0]
            if first_res in n_lookup:
                root = n_lookup[first_res]
            elif first_res in ca_lookup:
                root = ca_lookup[first_res]
            else:
                root = junc_atoms[0]

            self._junction_to_segment[junc_idx] = len(segments)
            segments.append(junc_atoms)
            segment_root_atoms.append(root)
            segment_types.append("junction")

        # Add non-protein residues
        protein_atoms = set()
        for seg in self._planned_segments:
            for res_key in seg:
                if res_key in residue_atoms:
                    protein_atoms.update(residue_atoms[res_key])
        for junc in self._planned_junctions:
            for res_key in junc:
                if res_key in residue_atoms:
                    protein_atoms.update(residue_atoms[res_key])

        is_protein = pdb["resname"].str.upper().isin(AA_NAMES)
        non_protein_residues = []
        for (chainid, resseq), group in pdb.groupby(["chainid", "resseq"]):
            resname = group["resname"].iloc[0]
            if resname.upper() not in AA_NAMES:
                atoms = group["index"].values.tolist()
                non_protein_residues.append(atoms)
            else:
                # Check for protein residues not in any planned segment or junction
                atoms = group["index"].values.tolist()
                if not any(a in protein_atoms for a in atoms):
                    non_protein_residues.append(atoms)

        for atoms in non_protein_residues:
            segments.append(atoms)
            segment_root_atoms.append(atoms[0])
            segment_types.append("non_protein")

        self.n_segments = len(segments)
        self._segment_types = segment_types

        # Create mapping tensors
        atom_to_segment = torch.full(
            (self.n_atoms,), -1, dtype=torch.long, device=device
        )
        max_segment_size = max(len(s) for s in segments) if segments else 0
        segment_atom_indices = torch.full(
            (self.n_segments, max_segment_size), -1, dtype=torch.long, device=device
        )
        segment_sizes = torch.zeros(self.n_segments, dtype=torch.long, device=device)

        for seg_idx, atom_indices in enumerate(segments):
            n_atoms_seg = len(atom_indices)
            segment_sizes[seg_idx] = n_atoms_seg
            segment_atom_indices[seg_idx, :n_atoms_seg] = torch.tensor(
                atom_indices, dtype=torch.long, device=device
            )
            atom_to_segment[atom_indices] = seg_idx

        self.register_buffer("atom_to_segment", atom_to_segment)
        self.register_buffer("segment_atom_indices", segment_atom_indices)
        self.register_buffer("segment_sizes", segment_sizes)
        self.register_buffer(
            "segment_roots",
            torch.tensor(segment_root_atoms, dtype=torch.long, device=device),
        )

        self._segments = segments

    def _build_segment_trees_parallel(self, xyz: torch.Tensor) -> None:
        """
        Build spanning trees for all segments using fully parallel multi-source BFS.

        Identical to SegmentedInternalCoordinateTensor's implementation.
        """
        device = self._device
        n_atoms = self.n_atoms

        parent_indices = torch.full(
            (n_atoms,), -1, dtype=torch.long, device=device
        )
        grandparent_indices = torch.full(
            (n_atoms,), -1, dtype=torch.long, device=device
        )
        great_grandparent_indices = torch.full(
            (n_atoms,), -1, dtype=torch.long, device=device
        )
        atom_depth = torch.full((n_atoms,), -1, dtype=torch.long, device=device)

        # Build molecular graph
        if self.cif_dict is not None:
            adjacency = self._build_adjacency_from_cif(xyz)
        else:
            distances = torch.cdist(xyz, xyz)
            adjacency = (distances < self.bond_cutoff) & (distances > 0.1)

        # Build compact neighbor list
        neighbor_counts = adjacency.sum(dim=1).to(torch.long)
        max_neighbors = neighbor_counts.max().item()

        edge_src, edge_tgt = torch.where(adjacency)
        sorted_indices = torch.argsort(edge_src, stable=True)
        edge_src_sorted = edge_src[sorted_indices]
        edge_tgt_sorted = edge_tgt[sorted_indices]

        neighbors_padded = torch.full(
            (n_atoms, max_neighbors), -1, dtype=torch.long, device=device
        )

        if edge_src_sorted.numel() > 0:
            offsets = torch.zeros(n_atoms + 1, dtype=torch.long, device=device)
            offsets[1:] = torch.cumsum(neighbor_counts, dim=0)
            edge_global_idx = torch.arange(len(edge_src_sorted), device=device)
            pos_in_group = edge_global_idx - offsets[edge_src_sorted]
            neighbors_padded[edge_src_sorted, pos_in_group] = edge_tgt_sorted

        # Parallel BFS from all roots
        atom_depth[self.segment_roots] = 0
        current_frontier = self.segment_roots.clone()
        current_depth = 0
        max_depth = 0

        while current_frontier.numel() > 0:
            frontier_neighbors = neighbors_padded[current_frontier]
            candidates = frontier_neighbors.reshape(-1)
            frontier_expanded = current_frontier.unsqueeze(1).expand_as(
                frontier_neighbors
            )
            parents = frontier_expanded.reshape(-1)

            valid_mask = candidates >= 0
            candidates = candidates[valid_mask]
            parents = parents[valid_mask]
            if candidates.numel() == 0:
                break

            unvisited_mask = atom_depth[candidates] == -1
            candidates = candidates[unvisited_mask]
            parents = parents[unvisited_mask]
            if candidates.numel() == 0:
                break

            candidate_segments = self.atom_to_segment[candidates]
            parent_segments = self.atom_to_segment[parents]
            same_segment_mask = candidate_segments == parent_segments
            candidates = candidates[same_segment_mask]
            parents = parents[same_segment_mask]
            if candidates.numel() == 0:
                break

            unique_candidates, inverse_indices = torch.unique(
                candidates, return_inverse=True
            )
            first_occurrence = torch.zeros(
                unique_candidates.numel(), dtype=torch.long, device=device
            )
            first_occurrence.scatter_(0, inverse_indices, parents)

            new_depth = current_depth + 1
            atom_depth[unique_candidates] = new_depth
            parent_indices[unique_candidates] = first_occurrence

            parent_of_new = first_occurrence
            gp_of_new = parent_indices[parent_of_new]
            has_gp = gp_of_new >= 0
            grandparent_indices[unique_candidates[has_gp]] = gp_of_new[has_gp]

            ggp_of_new = grandparent_indices[parent_of_new]
            has_ggp = ggp_of_new >= 0
            great_grandparent_indices[unique_candidates[has_ggp]] = ggp_of_new[
                has_ggp
            ]

            max_depth = new_depth
            current_frontier = unique_candidates
            current_depth = new_depth

        # Handle disconnected components
        max_component_iterations = 100
        for _ in range(max_component_iterations):
            orphan_mask = (atom_depth == -1) & (self.atom_to_segment >= 0)
            if not orphan_mask.any():
                break

            orphan_indices = torch.where(orphan_mask)[0]
            orphan_segments = self.atom_to_segment[orphan_indices]
            unique_orphan_segments = torch.unique(orphan_segments)

            new_roots = []
            for seg in unique_orphan_segments:
                seg_orphans = orphan_indices[orphan_segments == seg]
                if seg_orphans.numel() > 0:
                    new_roots.append(seg_orphans[0].item())

            if not new_roots:
                break

            new_roots = torch.tensor(new_roots, dtype=torch.long, device=device)
            atom_depth[new_roots] = 0
            parent_indices[new_roots] = -1

            current_frontier = new_roots
            while current_frontier.numel() > 0:
                frontier_neighbors = neighbors_padded[current_frontier]
                candidates = frontier_neighbors.reshape(-1)
                frontier_expanded = current_frontier.unsqueeze(1).expand_as(
                    frontier_neighbors
                )
                parents = frontier_expanded.reshape(-1)

                valid_mask = candidates >= 0
                candidates = candidates[valid_mask]
                parents = parents[valid_mask]
                if candidates.numel() == 0:
                    break

                unvisited_mask = atom_depth[candidates] == -1
                candidates = candidates[unvisited_mask]
                parents = parents[unvisited_mask]
                if candidates.numel() == 0:
                    break

                candidate_segments = self.atom_to_segment[candidates]
                parent_segments = self.atom_to_segment[parents]
                same_segment_mask = candidate_segments == parent_segments
                candidates = candidates[same_segment_mask]
                parents = parents[same_segment_mask]
                if candidates.numel() == 0:
                    break

                unique_candidates, inverse_indices = torch.unique(
                    candidates, return_inverse=True
                )
                first_occurrence = torch.zeros(
                    unique_candidates.numel(), dtype=torch.long, device=device
                )
                first_occurrence.scatter_(0, inverse_indices, parents)

                parent_depths = atom_depth[first_occurrence]
                new_depths = parent_depths + 1

                atom_depth[unique_candidates] = new_depths
                parent_indices[unique_candidates] = first_occurrence

                parent_of_new = first_occurrence
                gp_of_new = parent_indices[parent_of_new]
                has_gp = gp_of_new >= 0
                grandparent_indices[unique_candidates[has_gp]] = gp_of_new[has_gp]

                ggp_of_new = grandparent_indices[parent_of_new]
                has_ggp = ggp_of_new >= 0
                great_grandparent_indices[unique_candidates[has_ggp]] = ggp_of_new[
                    has_ggp
                ]

                max_depth = max(max_depth, new_depths.max().item())
                current_frontier = unique_candidates

        # Identify secondary roots
        depth0_mask = atom_depth == 0
        is_primary_root = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        is_primary_root[self.segment_roots] = True
        secondary_root_mask = depth0_mask & ~is_primary_root
        secondary_root_indices = torch.where(secondary_root_mask)[0]
        self.register_buffer("secondary_root_indices", secondary_root_indices)

        self.max_depth = max_depth
        self.register_buffer("parent_indices", parent_indices)
        self.register_buffer("grandparent_indices", grandparent_indices)
        self.register_buffer("great_grandparent_indices", great_grandparent_indices)
        self.register_buffer("atom_depth", atom_depth)

        self._build_param_indices_vectorized()
        self._build_depth_indices()
        self._detect_rings_vectorized(adjacency)

    def _build_adjacency_from_cif(self, xyz: torch.Tensor) -> torch.Tensor:
        """Build adjacency matrix from CIF dictionary (same as parent class)."""
        device = self._device
        n_atoms = self.n_atoms
        pdb = self.pdb

        bond_idx1_list = []
        bond_idx2_list = []

        pdb["name_stripped"] = pdb["name"].str.strip()
        residue_groups = pdb.groupby(["chainid", "resseq"])
        atoms_with_cif_bonds = set()

        for (chainid, resseq), group in residue_groups:
            resname = group["resname"].iloc[0]
            if resname not in self.cif_dict:
                continue
            cif_residue = self.cif_dict[resname]
            if "bonds" not in cif_residue:
                continue
            cif_bonds = cif_residue["bonds"]
            name_to_idx = dict(zip(group["name_stripped"], group["index"]))
            atom1_names = cif_bonds["atom1"].str.strip().values
            atom2_names = cif_bonds["atom2"].str.strip().values
            for a1, a2 in zip(atom1_names, atom2_names):
                if a1 in name_to_idx and a2 in name_to_idx:
                    idx1, idx2 = name_to_idx[a1], name_to_idx[a2]
                    bond_idx1_list.append(idx1)
                    bond_idx2_list.append(idx2)
                    atoms_with_cif_bonds.add(idx1)
                    atoms_with_cif_bonds.add(idx2)

        # Peptide bonds
        c_atoms = pdb[pdb["name_stripped"] == "C"].set_index(
            ["chainid", "resseq"]
        )["index"]
        n_atoms_df = pdb[pdb["name_stripped"] == "N"].set_index(
            ["chainid", "resseq"]
        )["index"]

        for chainid in pdb["chainid"].unique():
            chain = pdb[pdb["chainid"] == chainid]
            chain_aa = chain[chain["resname"].isin(AA_NAMES)]
            resseqs = sorted(chain_aa["resseq"].unique())
            for i in range(len(resseqs) - 1):
                r1, r2 = resseqs[i], resseqs[i + 1]
                if r2 - r1 != 1:
                    continue
                try:
                    idx_c = c_atoms.loc[(chainid, r1)]
                    idx_n = n_atoms_df.loc[(chainid, r2)]
                    bond_idx1_list.append(idx_c)
                    bond_idx2_list.append(idx_n)
                    atoms_with_cif_bonds.add(idx_c)
                    atoms_with_cif_bonds.add(idx_n)
                except KeyError:
                    continue

        adjacency = torch.zeros(n_atoms, n_atoms, dtype=torch.bool, device=device)
        if bond_idx1_list:
            idx1 = torch.tensor(bond_idx1_list, dtype=torch.long, device=device)
            idx2 = torch.tensor(bond_idx2_list, dtype=torch.long, device=device)
            adjacency[idx1, idx2] = True
            adjacency[idx2, idx1] = True

        # Fallback for atoms without CIF bonds
        atoms_without_cif = set(range(n_atoms)) - atoms_with_cif_bonds
        if atoms_without_cif:
            atoms_list = sorted(atoms_without_cif)
            atoms_tensor = torch.tensor(atoms_list, dtype=torch.long, device=device)
            xyz_subset = xyz[atoms_tensor]
            distances_subset = torch.cdist(xyz_subset, xyz)
            bond_mask = (distances_subset < self.bond_cutoff) & (
                distances_subset > 0.1
            )
            rows, cols = torch.where(bond_mask)
            adjacency[atoms_tensor[rows], cols] = True
            adjacency[cols, atoms_tensor[rows]] = True

        pdb.drop(columns=["name_stripped"], inplace=True, errors="ignore")
        return adjacency

    def _build_param_indices_vectorized(self) -> None:
        """Build mappings from atoms to parameter indices."""
        device = self._device

        has_bond = self.atom_depth >= 1
        has_angle = self.atom_depth >= 2
        has_torsion = self.atom_depth >= 3

        # Build junction masks
        is_junction_backbone = torch.zeros(
            self.n_atoms, dtype=torch.bool, device=device
        )
        if self.junction_backbone_indices.numel() > 0:
            is_junction_backbone[self.junction_backbone_indices] = True

        is_junction_atom = torch.zeros(
            self.n_atoms, dtype=torch.bool, device=device
        )
        if self.junction_atom_indices.numel() > 0:
            is_junction_atom[self.junction_atom_indices] = True

        # For torsions: exclude junction backbone atoms only (solved by Newton).
        # Junction sidechain torsions (chi angles) remain free DOFs.
        has_free_torsion = has_torsion & ~is_junction_backbone

        self.n_bonds = has_bond.sum().item()
        self.n_angles = has_angle.sum().item()
        self.n_torsions = has_free_torsion.sum().item()
        self.n_all_torsions = has_torsion.sum().item()

        bond_param_indices = torch.full(
            (self.n_atoms,), -1, dtype=torch.long, device=device
        )
        angle_param_indices = torch.full(
            (self.n_atoms,), -1, dtype=torch.long, device=device
        )
        torsion_param_indices = torch.full(
            (self.n_atoms,), -1, dtype=torch.long, device=device
        )

        if has_bond.any():
            bond_cumsum = torch.cumsum(has_bond.long(), dim=0) - 1
            bond_param_indices[has_bond] = bond_cumsum[has_bond]

        if has_angle.any():
            angle_cumsum = torch.cumsum(has_angle.long(), dim=0) - 1
            angle_param_indices[has_angle] = angle_cumsum[has_angle]

        if has_free_torsion.any():
            torsion_cumsum = torch.cumsum(has_free_torsion.long(), dim=0) - 1
            torsion_param_indices[has_free_torsion] = torsion_cumsum[has_free_torsion]

        self.register_buffer("bond_param_indices", bond_param_indices)
        self.register_buffer("angle_param_indices", angle_param_indices)
        self.register_buffer("torsion_param_indices", torsion_param_indices)
        self.register_buffer("is_junction_backbone", is_junction_backbone)
        self.register_buffer("is_junction_atom", is_junction_atom)

    def _build_depth_indices(self) -> None:
        """Pre-compute atom indices for each depth level.

        Builds two sets of indices:
        1. Regular depth indices — excludes ALL junction atoms
        2. Junction sidechain depth indices — only junction non-backbone atoms,
           placed after the junction solver provides backbone positions
        """
        device = self._device
        empty = torch.tensor([], dtype=torch.long, device=device)

        # --- Regular (non-junction) depth-3+ indices ---
        depth_atom_indices = []
        depth_parent_indices = []
        depth_gp_indices = []
        depth_ggp_indices = []
        depth_bond_param_indices = []
        depth_angle_param_indices = []
        depth_torsion_param_indices = []

        for d in range(3, self.max_depth + 1):
            # Exclude ALL junction atoms from regular placement
            mask = (self.atom_depth == d) & ~self.is_junction_atom
            if mask.any():
                atom_idx = torch.where(mask)[0]
                depth_atom_indices.append(atom_idx)
                depth_parent_indices.append(self.parent_indices[atom_idx])
                depth_gp_indices.append(self.grandparent_indices[atom_idx])
                depth_ggp_indices.append(
                    self.great_grandparent_indices[atom_idx]
                )
                depth_bond_param_indices.append(self.bond_param_indices[atom_idx])
                depth_angle_param_indices.append(
                    self.angle_param_indices[atom_idx]
                )
                depth_torsion_param_indices.append(
                    self.torsion_param_indices[atom_idx]
                )
            else:
                depth_atom_indices.append(empty.clone())
                depth_parent_indices.append(empty.clone())
                depth_gp_indices.append(empty.clone())
                depth_ggp_indices.append(empty.clone())
                depth_bond_param_indices.append(empty.clone())
                depth_angle_param_indices.append(empty.clone())
                depth_torsion_param_indices.append(empty.clone())

        self._depth_atom_indices = depth_atom_indices
        self._depth_parent_indices = depth_parent_indices
        self._depth_gp_indices = depth_gp_indices
        self._depth_ggp_indices = depth_ggp_indices
        self._depth_bond_param_indices = depth_bond_param_indices
        self._depth_angle_param_indices = depth_angle_param_indices
        self._depth_torsion_param_indices = depth_torsion_param_indices

        # --- Junction sidechain depth indices ---
        # These atoms are placed AFTER the junction solver provides backbone positions.
        # We need depth-ordered placement for junction non-backbone atoms.
        jsc_depth_atom_indices = []
        jsc_depth_parent_indices = []
        jsc_depth_gp_indices = []
        jsc_depth_ggp_indices = []
        jsc_depth_bond_param_indices = []
        jsc_depth_angle_param_indices = []

        # Junction sidechain atoms: in junction but not backbone
        is_jsc = self.is_junction_atom & ~self.is_junction_backbone

        for d in range(1, self.max_depth + 1):
            mask = (self.atom_depth == d) & is_jsc
            if mask.any():
                atom_idx = torch.where(mask)[0]
                jsc_depth_atom_indices.append(atom_idx)
                jsc_depth_parent_indices.append(self.parent_indices[atom_idx])
                jsc_depth_gp_indices.append(self.grandparent_indices[atom_idx])
                jsc_depth_ggp_indices.append(
                    self.great_grandparent_indices[atom_idx]
                )
                jsc_depth_bond_param_indices.append(self.bond_param_indices[atom_idx])
                jsc_depth_angle_param_indices.append(self.angle_param_indices[atom_idx])
            else:
                jsc_depth_atom_indices.append(empty.clone())
                jsc_depth_parent_indices.append(empty.clone())
                jsc_depth_gp_indices.append(empty.clone())
                jsc_depth_ggp_indices.append(empty.clone())
                jsc_depth_bond_param_indices.append(empty.clone())
                jsc_depth_angle_param_indices.append(empty.clone())

        self._jsc_depth_atom_indices = jsc_depth_atom_indices
        self._jsc_depth_parent_indices = jsc_depth_parent_indices
        self._jsc_depth_gp_indices = jsc_depth_gp_indices
        self._jsc_depth_ggp_indices = jsc_depth_ggp_indices
        self._jsc_depth_bond_param_indices = jsc_depth_bond_param_indices
        self._jsc_depth_angle_param_indices = jsc_depth_angle_param_indices
        self._jsc_min_depth = 1  # starting depth for junction sidechain placement

    def _detect_rings_vectorized(self, adjacency: torch.Tensor) -> None:
        """Detect rings in the molecular graph (same as parent class)."""
        device = self._device
        n_atoms = self.n_atoms

        ring_member_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        ring_group_id = torch.full(
            (n_atoms,), -1, dtype=torch.long, device=device
        )

        edge_i, edge_j = torch.where(adjacency.triu(diagonal=1))
        is_tree_edge = (self.parent_indices[edge_i] == edge_j) | (
            self.parent_indices[edge_j] == edge_i
        )
        same_segment = (
            self.atom_to_segment[edge_i] == self.atom_to_segment[edge_j]
        )
        back_edge_mask = ~is_tree_edge & same_segment
        back_edge_i = edge_i[back_edge_mask]
        back_edge_j = edge_j[back_edge_mask]

        individual_rings = []
        for idx in range(len(back_edge_i)):
            i = back_edge_i[idx].item()
            j = back_edge_j[idx].item()

            ancestors_i = set()
            current = i
            while current >= 0:
                ancestors_i.add(current)
                current = self.parent_indices[current].item()

            ring_atoms = []
            current = j
            common_ancestor = None
            while current >= 0:
                ring_atoms.append(current)
                if current in ancestors_i:
                    common_ancestor = current
                    break
                current = self.parent_indices[current].item()

            if common_ancestor is None:
                continue

            current = i
            while current != common_ancestor:
                ring_atoms.append(current)
                current = self.parent_indices[current].item()

            individual_rings.append(set(ring_atoms))

        ring_systems = self._merge_overlapping_rings(individual_rings)

        ring_anchors = []
        ring_members_list = []

        for ring_idx, ring_atoms_set in enumerate(ring_systems):
            ring_atoms = list(ring_atoms_set)
            for atom in ring_atoms:
                ring_member_mask[atom] = True
                ring_group_id[atom] = ring_idx
            depths = self.atom_depth[ring_atoms]
            anchor_idx = ring_atoms[depths.argmin().item()]
            ring_anchors.append(anchor_idx)
            ring_members_list.append(ring_atoms)

        self.register_buffer("ring_member_mask", ring_member_mask)
        self.register_buffer("ring_group_id", ring_group_id)
        self.n_rings = len(ring_anchors)

        if self.n_rings > 0:
            self.register_buffer(
                "ring_anchor_atoms",
                torch.tensor(ring_anchors, dtype=torch.long, device=device),
            )
            max_ring_size = max(len(r) for r in ring_members_list)
            ring_members_tensor = torch.full(
                (self.n_rings, max_ring_size), -1, dtype=torch.long, device=device
            )
            ring_sizes = torch.zeros(
                self.n_rings, dtype=torch.long, device=device
            )
            for i, members in enumerate(ring_members_list):
                ring_sizes[i] = len(members)
                ring_members_tensor[i, : len(members)] = torch.tensor(
                    members, dtype=torch.long, device=device
                )
            self.register_buffer("ring_members", ring_members_tensor)
            self.register_buffer("ring_sizes", ring_sizes)
        else:
            self.register_buffer(
                "ring_anchor_atoms",
                torch.tensor([], dtype=torch.long, device=device),
            )
            self.register_buffer(
                "ring_members",
                torch.tensor([], dtype=torch.long, device=device).reshape(0, 0),
            )
            self.register_buffer(
                "ring_sizes",
                torch.tensor([], dtype=torch.long, device=device),
            )

    def _merge_overlapping_rings(self, rings: list) -> list:
        """Merge overlapping rings using union-find."""
        if not rings:
            return []

        n_rings = len(rings)
        parent = list(range(n_rings))
        rank = [0] * n_rings

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        for i in range(n_rings):
            for j in range(i + 1, n_rings):
                if rings[i] & rings[j]:
                    union(i, j)

        groups = defaultdict(list)
        for i in range(n_rings):
            groups[find(i)].append(i)

        ring_systems = []
        for ring_indices in groups.values():
            merged = set()
            for ri in ring_indices:
                merged |= rings[ri]
            ring_systems.append(merged)

        return ring_systems

    def _extract_junction_geometry(self, xyz: torch.Tensor) -> None:
        """
        Extract fixed backbone geometry for junctions.

        Stores bond lengths, bond angles (in NeRF placement order), omega angles,
        psi_prev, and target C positions. All stored as buffers (fixed, not parameters).

        NeRF placement order per junction residue:
        - Bond lengths: [C_prev-N, N-CA, CA-C]
        - Bond angles: [angle_at_C_prev_for_N, angle_at_N_for_CA, angle_at_CA_for_C]
        """
        device = self._device
        dtype = self._dtype

        if self.n_junctions == 0:
            self.register_buffer(
                "junction_bond_lengths",
                torch.zeros(0, 3 * self.junction_size, dtype=dtype, device=device),
            )
            self.register_buffer(
                "junction_nerf_angles",
                torch.zeros(0, 3 * self.junction_size, dtype=dtype, device=device),
            )
            self.register_buffer(
                "junction_omega",
                torch.zeros(0, self.junction_size, dtype=dtype, device=device),
            )
            self.register_buffer(
                "junction_psi_prev",
                torch.zeros(0, dtype=dtype, device=device),
            )
            self.register_buffer(
                "junction_post_bond_length",
                torch.zeros(0, dtype=dtype, device=device),
            )
            self.register_buffer(
                "junction_post_bond_angle",
                torch.zeros(0, dtype=dtype, device=device),
            )
            self._junction_post_n_indices = []
            return

        all_bond_lengths = []
        all_nerf_angles = []
        all_omega = []
        all_psi_prev = []
        all_post_bl = []
        all_post_ba = []
        post_n_indices = []  # Global index of post-junction N atom per junction

        for junc_idx, junc_info in enumerate(self._junction_data):
            junc_residues = junc_info["residues"]
            junc_backbone = junc_info["backbone"]

            bl_list = []
            nerf_ang_list = []
            omega_list = []

            pre_seg_idx = self._find_pre_junction_segment(junc_idx)

            for res_i, res_key in enumerate(junc_residues):
                bb = junc_backbone[res_i]
                n_pos = xyz[bb["N"]]
                ca_pos = xyz[bb["CA"]]
                c_pos = xyz[bb["C"]]

                # Previous residue atoms
                if res_i == 0:
                    if pre_seg_idx is not None:
                        prev_res = self._planned_segments[pre_seg_idx][-1]
                        if prev_res in self._backbone_map:
                            prev_bb = self._backbone_map[prev_res]
                            prev_n = xyz[prev_bb["N"]]
                            prev_ca = xyz[prev_bb["CA"]]
                            prev_c = xyz[prev_bb["C"]]
                        else:
                            prev_n = n_pos - torch.tensor([2.5, 0, 0], dtype=dtype, device=device)
                            prev_ca = n_pos - torch.tensor([1.3, 0, 0], dtype=dtype, device=device)
                            prev_c = n_pos - torch.tensor([0.5, 0, 0], dtype=dtype, device=device)
                    else:
                        prev_n = n_pos - torch.tensor([2.5, 0, 0], dtype=dtype, device=device)
                        prev_ca = n_pos - torch.tensor([1.3, 0, 0], dtype=dtype, device=device)
                        prev_c = n_pos - torch.tensor([0.5, 0, 0], dtype=dtype, device=device)
                else:
                    prev_bb_idx = junc_backbone[res_i - 1]
                    prev_n = xyz[prev_bb_idx["N"]]
                    prev_ca = xyz[prev_bb_idx["CA"]]
                    prev_c = xyz[prev_bb_idx["C"]]

                # Bond lengths: C_prev-N, N-CA, CA-C
                bl_list.append(torch.linalg.norm(n_pos - prev_c))
                bl_list.append(torch.linalg.norm(ca_pos - n_pos))
                bl_list.append(torch.linalg.norm(c_pos - ca_pos))

                # NeRF angles at pivot atoms:
                # 1. Angle at C_prev for placing N: angle(CA_prev, C_prev, N)
                nerf_ang_list.append(_compute_angle(prev_ca, prev_c, n_pos))
                # 2. Angle at N for placing CA: angle(C_prev, N, CA)
                nerf_ang_list.append(_compute_angle(prev_c, n_pos, ca_pos))
                # 3. Angle at CA for placing C: angle(N, CA, C)
                nerf_ang_list.append(_compute_angle(n_pos, ca_pos, c_pos))

                # Omega: dihedral(CA_prev, C_prev, N, CA)
                omega_list.append(_compute_torsion(prev_ca, prev_c, n_pos, ca_pos))

            # psi_prev: dihedral(N_prev, CA_prev, C_prev, N_first_junction)
            first_bb = junc_backbone[0]
            first_n = xyz[first_bb["N"]]
            if pre_seg_idx is not None:
                prev_res = self._planned_segments[pre_seg_idx][-1]
                if prev_res in self._backbone_map:
                    prev_bb = self._backbone_map[prev_res]
                    psi_prev = _compute_torsion(
                        xyz[prev_bb["N"]], xyz[prev_bb["CA"]], xyz[prev_bb["C"]], first_n
                    )
                else:
                    psi_prev = torch.tensor(0.0, dtype=dtype, device=device)
            else:
                psi_prev = torch.tensor(0.0, dtype=dtype, device=device)

            # Post-junction geometry: C(last_junc) -> N(first_post_segment)
            # for extending the FK to target the post-junction N dynamically.
            last_bb = junc_backbone[-1]
            last_c = xyz[last_bb["C"]]
            last_ca = xyz[last_bb["CA"]]
            last_n = xyz[last_bb["N"]]

            post_seg_idx = self._find_post_junction_segment(junc_idx)
            if post_seg_idx is not None and post_seg_idx < len(self._planned_segments):
                post_res = self._planned_segments[post_seg_idx][0]
                if post_res in self._backbone_map:
                    post_n_idx = self._backbone_map[post_res]["N"]
                    post_n_pos = xyz[post_n_idx]
                    post_bl = torch.linalg.norm(post_n_pos - last_c)
                    post_ba = _compute_angle(last_ca, last_c, post_n_pos)
                    post_n_indices.append(post_n_idx)
                else:
                    # Fallback: use typical peptide bond
                    post_bl = torch.tensor(1.329, dtype=dtype, device=device)
                    post_ba = torch.tensor(2.028, dtype=dtype, device=device)  # ~116 deg
                    post_n_indices.append(-1)
            else:
                # Last junction with no post-segment: use typical values
                post_bl = torch.tensor(1.329, dtype=dtype, device=device)
                post_ba = torch.tensor(2.028, dtype=dtype, device=device)
                post_n_indices.append(-1)

            all_bond_lengths.append(torch.stack(bl_list).to(dtype=dtype, device=device))
            all_nerf_angles.append(torch.stack(nerf_ang_list).to(dtype=dtype, device=device))
            all_omega.append(torch.stack(omega_list).to(dtype=dtype, device=device))
            all_psi_prev.append(psi_prev.to(dtype=dtype, device=device))
            all_post_bl.append(post_bl.to(dtype=dtype, device=device))
            all_post_ba.append(post_ba.to(dtype=dtype, device=device))

        self.register_buffer("junction_bond_lengths", torch.stack(all_bond_lengths))
        self.register_buffer("junction_nerf_angles", torch.stack(all_nerf_angles))
        self.register_buffer("junction_omega", torch.stack(all_omega))
        self.register_buffer("junction_psi_prev", torch.stack(all_psi_prev))
        self.register_buffer("junction_post_bond_length", torch.stack(all_post_bl))
        self.register_buffer("junction_post_bond_angle", torch.stack(all_post_ba))
        self._junction_post_n_indices = post_n_indices

    def _find_pre_junction_segment(self, junc_idx: int) -> Optional[int]:
        """Find the index (into _planned_segments) of the segment before this junction."""
        # Junction junc_idx connects segment junc_idx to segment junc_idx+1
        if junc_idx < len(self._planned_segments):
            return junc_idx
        return None

    def _find_post_junction_segment(self, junc_idx: int) -> Optional[int]:
        """Find the index (into _planned_segments) of the segment after this junction."""
        if junc_idx + 1 < len(self._planned_segments):
            return junc_idx + 1
        return None

    def _extract_internal_coords(
        self, xyz: torch.Tensor, requires_grad: bool = True
    ) -> None:
        """
        Extract internal coordinates from Cartesian coordinates.

        Junction backbone torsions are excluded from the free torsion parameters.
        """
        device = self._device
        dtype = self._dtype

        # Bond lengths (all atoms with depth >= 1)
        bond_mask = self.atom_depth >= 1
        if bond_mask.any():
            child_xyz = xyz[bond_mask]
            parent_xyz = xyz[self.parent_indices[bond_mask]]
            bond_lengths = torch.linalg.norm(child_xyz - parent_xyz, dim=-1)
        else:
            bond_lengths = torch.tensor([], dtype=dtype, device=device)

        # Angles (all atoms with depth >= 2)
        angle_mask = self.atom_depth >= 2
        if angle_mask.any():
            child_xyz = xyz[angle_mask]
            parent_xyz = xyz[self.parent_indices[angle_mask]]
            grandparent_xyz = xyz[self.grandparent_indices[angle_mask]]
            v1 = child_xyz - parent_xyz
            v2 = grandparent_xyz - parent_xyz
            cos_angles = torch.sum(v1 * v2, dim=-1) / (
                torch.linalg.norm(v1, dim=-1)
                * torch.linalg.norm(v2, dim=-1)
                + 1e-10
            )
            cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
            angles = torch.acos(cos_angles)
        else:
            angles = torch.tensor([], dtype=dtype, device=device)

        # Torsions (depth >= 3, EXCLUDING junction backbone atoms only)
        torsion_mask = (self.atom_depth >= 3) & ~self.is_junction_backbone
        if torsion_mask.any():
            child_xyz = xyz[torsion_mask]
            parent_xyz = xyz[self.parent_indices[torsion_mask]]
            grandparent_xyz = xyz[self.grandparent_indices[torsion_mask]]
            great_grandparent_xyz = xyz[
                self.great_grandparent_indices[torsion_mask]
            ]
            torsions = self._compute_torsion_angles(
                great_grandparent_xyz, grandparent_xyz, parent_xyz, child_xyz
            )
        else:
            torsions = torch.tensor([], dtype=dtype, device=device)

        # Segment positions and orientations
        segment_positions = xyz[self.segment_roots].clone()

        if self.secondary_root_indices.numel() > 0:
            secondary_root_positions = xyz[self.secondary_root_indices].clone()
        else:
            secondary_root_positions = torch.zeros(0, 3, dtype=dtype, device=device)

        segment_orientations = torch.zeros(
            self.n_segments, 3, dtype=dtype, device=device
        )

        # Ring local coordinates
        if self.n_rings > 0:
            ring_local_coords = self._extract_ring_local_coords_vectorized(xyz)
            self.register_buffer("ring_local_coords", ring_local_coords)
        else:
            self.register_buffer(
                "ring_local_coords",
                torch.tensor([], dtype=dtype, device=device).reshape(0, 0, 3),
            )

        # Shallow atom references
        self._setup_shallow_atom_references_vectorized(xyz)

        # Register parameters (free DOFs)
        self.bond_lengths = nn.Parameter(
            bond_lengths.clone(), requires_grad=requires_grad
        )
        self.angles = nn.Parameter(angles.clone(), requires_grad=requires_grad)
        self.torsions = nn.Parameter(
            torsions.clone(), requires_grad=requires_grad
        )
        self.segment_positions = nn.Parameter(
            segment_positions.clone(), requires_grad=requires_grad
        )
        self.segment_orientations = nn.Parameter(
            segment_orientations.clone(), requires_grad=requires_grad
        )
        self.secondary_root_positions = nn.Parameter(
            secondary_root_positions.clone(), requires_grad=requires_grad
        )

        # Initialize refinable mask
        self.register_buffer(
            "refinable_mask",
            torch.ones(self.n_atoms, dtype=torch.bool, device=device),
        )
        self.register_buffer("fixed_xyz", xyz.clone())

    def _init_junction_solver(self, xyz: torch.Tensor) -> None:
        """Initialize the junction solver with warm-start from initial coordinates."""
        if self.n_junctions == 0:
            self.junction_solver = JunctionSolver(
                n_junctions=0,
                junction_size=self.junction_size,
                initial_phi_psi=torch.zeros(
                    0, 2 * self.junction_size,
                    dtype=self._dtype, device=self._device,
                ),
                dtype=self._dtype,
                device=self._device,
            )
            return

        # Extract initial phi/psi for each junction from coordinates
        torsions = compute_backbone_torsions(
            xyz.detach(), self._backbone_map, self._chain_residues
        )

        initial_phi_psi = []
        for junc_info in self._junction_data:
            junc_residues = junc_info["residues"]
            pp = []
            for res_key in junc_residues:
                if res_key in torsions:
                    phi = torsions[res_key]["phi"]
                    psi = torsions[res_key]["psi"]
                    pp.append(phi if not np.isnan(phi) else 0.0)
                    pp.append(psi if not np.isnan(psi) else 0.0)
                else:
                    pp.extend([0.0, 0.0])
            initial_phi_psi.append(
                torch.tensor(pp, dtype=self._dtype, device=self._device)
            )

        initial_phi_psi = torch.stack(initial_phi_psi)  # (J, 2*junction_size)

        self.junction_solver = JunctionSolver(
            n_junctions=self.n_junctions,
            junction_size=self.junction_size,
            initial_phi_psi=initial_phi_psi,
            max_iter=50,
            tol=1e-4,
            tikhonov_eps=1e-6,
            dtype=self._dtype,
            device=self._device,
        )

        # Build mapping from junction backbone atom indices to positions in
        # the junction_solver output
        self._build_junction_atom_mapping()

    def _build_junction_atom_mapping(self) -> None:
        """Build mapping from junction backbone atoms to solver output positions."""
        device = self._device

        # For each junction, the solver outputs 3*junction_size backbone atoms
        # in order: [N0, CA0, C0, N1, CA1, C1, ...]
        # We need to map these to global atom indices

        junc_global_indices = []  # Global atom index
        junc_output_indices = []  # (junction_idx, position_in_output)

        for junc_idx, junc_info in enumerate(self._junction_data):
            for res_i, res_key in enumerate(junc_info["residues"]):
                bb = junc_info["backbone"][res_i]
                for atom_i, atom_name in enumerate(["N", "CA", "C"]):
                    global_idx = bb[atom_name]
                    output_pos = res_i * 3 + atom_i
                    junc_global_indices.append(global_idx)
                    junc_output_indices.append((junc_idx, output_pos))

        if junc_global_indices:
            self.register_buffer(
                "_junc_global_indices",
                torch.tensor(junc_global_indices, dtype=torch.long, device=device),
            )
            self._junc_output_indices = junc_output_indices
        else:
            self.register_buffer(
                "_junc_global_indices",
                torch.tensor([], dtype=torch.long, device=device),
            )
            self._junc_output_indices = []

        # Also build mapping for junction sidechain atoms
        # These are placed by NeRF from the junction backbone positions
        # They're already in the depth indices as regular NeRF atoms
        # (unless they're junction backbone atoms, which are excluded)

    def _setup_shallow_atom_references_vectorized(
        self, xyz: torch.Tensor
    ) -> None:
        """Store reference directions for depth-1 and depth-2 atoms."""
        device = self._device
        dtype = self._dtype

        # Depth-1 atoms
        depth1_mask = self.atom_depth == 1
        n_depth1 = depth1_mask.sum().item()

        if n_depth1 > 0:
            depth1_indices = torch.where(depth1_mask)[0]
            parent_idx = self.parent_indices[depth1_indices]
            directions = xyz[depth1_indices] - xyz[parent_idx]
            norms = torch.linalg.norm(
                directions, dim=-1, keepdim=True
            ).clamp(min=1e-10)
            depth1_dirs = directions / norms

            depth1_atom_to_dir_idx = torch.full(
                (self.n_atoms,), -1, dtype=torch.long, device=device
            )
            depth1_atom_to_dir_idx[depth1_indices] = torch.arange(
                n_depth1, dtype=torch.long, device=device
            )
            self.register_buffer("depth1_dirs", depth1_dirs)
            self.register_buffer(
                "depth1_atom_to_dir_idx", depth1_atom_to_dir_idx
            )
        else:
            self.register_buffer(
                "depth1_dirs", torch.zeros(0, 3, dtype=dtype, device=device)
            )
            self.register_buffer(
                "depth1_atom_to_dir_idx",
                torch.full(
                    (self.n_atoms,), -1, dtype=torch.long, device=device
                ),
            )

        # Depth-2 atoms
        depth2_mask = self.atom_depth == 2
        n_depth2 = depth2_mask.sum().item()

        if n_depth2 > 0:
            depth2_indices = torch.where(depth2_mask)[0]
            parent_idx = self.parent_indices[depth2_indices]
            grandparent_idx = self.grandparent_indices[depth2_indices]

            v_bond = xyz[parent_idx] - xyz[grandparent_idx]
            v_bond_norm = torch.linalg.norm(
                v_bond, dim=-1, keepdim=True
            ).clamp(min=1e-10)
            v_bond = v_bond / v_bond_norm

            v_child = xyz[depth2_indices] - xyz[parent_idx]
            proj = torch.sum(v_child * v_bond, dim=-1, keepdim=True)
            v_perp = v_child - proj * v_bond
            v_perp_norm = torch.linalg.norm(v_perp, dim=-1, keepdim=True)

            collinear = v_perp_norm.squeeze(-1) < 1e-10
            if collinear.any():
                default_perp = torch.zeros_like(v_perp)
                nearly_x = torch.abs(v_bond[:, 0]) >= 0.9
                default_perp[~nearly_x, 0] = 1.0
                default_perp[nearly_x, 1] = 1.0
                proj_default = torch.sum(
                    default_perp * v_bond, dim=-1, keepdim=True
                )
                default_perp = default_perp - proj_default * v_bond
                default_perp = default_perp / torch.linalg.norm(
                    default_perp, dim=-1, keepdim=True
                ).clamp(min=1e-10)
                v_perp[collinear] = default_perp[collinear]
                v_perp_norm[collinear] = 1.0

            depth2_perps = v_perp / v_perp_norm.clamp(min=1e-10)

            depth2_atom_to_perp_idx = torch.full(
                (self.n_atoms,), -1, dtype=torch.long, device=device
            )
            depth2_atom_to_perp_idx[depth2_indices] = torch.arange(
                n_depth2, dtype=torch.long, device=device
            )
            self.register_buffer("depth2_perps", depth2_perps)
            self.register_buffer(
                "depth2_atom_to_perp_idx", depth2_atom_to_perp_idx
            )
        else:
            self.register_buffer(
                "depth2_perps", torch.zeros(0, 3, dtype=dtype, device=device)
            )
            self.register_buffer(
                "depth2_atom_to_perp_idx",
                torch.full(
                    (self.n_atoms,), -1, dtype=torch.long, device=device
                ),
            )

    def _extract_ring_local_coords_vectorized(
        self, xyz: torch.Tensor
    ) -> torch.Tensor:
        """Extract local coordinates for ring atoms."""
        device = self._device
        dtype = self._dtype
        max_ring_size = (
            self.ring_members.shape[1] if self.n_rings > 0 else 0
        )

        ring_local_coords = torch.zeros(
            self.n_rings, max_ring_size, 3, dtype=dtype, device=device
        )
        if self.n_rings == 0:
            return ring_local_coords

        anchor_pos = xyz[self.ring_anchor_atoms]
        parent_idx = self.parent_indices[self.ring_anchor_atoms]
        valid_parent = parent_idx >= 0
        parent_pos = torch.zeros_like(anchor_pos)
        parent_pos[valid_parent] = xyz[parent_idx[valid_parent]]

        z_axis = anchor_pos - parent_pos
        z_norm = torch.linalg.norm(z_axis, dim=-1, keepdim=True)
        z_axis = torch.where(
            z_norm > 1e-10,
            z_axis / z_norm.clamp(min=1e-10),
            torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device),
        )
        z_axis[~valid_parent] = torch.tensor(
            [0.0, 0.0, 1.0], dtype=dtype, device=device
        )

        x_base = torch.zeros_like(z_axis)
        x_base[:, 0] = 1.0
        nearly_x = torch.abs(z_axis[:, 0]) >= 0.9
        x_base[nearly_x, 0] = 0.0
        x_base[nearly_x, 1] = 1.0

        x_axis = (
            x_base - (x_base * z_axis).sum(dim=-1, keepdim=True) * z_axis
        )
        x_axis = x_axis / torch.linalg.norm(
            x_axis, dim=-1, keepdim=True
        ).clamp(min=1e-10)
        y_axis = torch.linalg.cross(z_axis, x_axis)

        R = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        valid_mask = self.ring_members >= 0
        safe_members = self.ring_members.clamp(min=0)
        member_pos = xyz[safe_members]
        offsets = member_pos - anchor_pos.unsqueeze(1)
        ring_local_coords = torch.bmm(offsets, R)
        ring_local_coords = ring_local_coords * valid_mask.unsqueeze(-1)

        return ring_local_coords

    @staticmethod
    def _compute_torsion_angles(
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor,
    ) -> torch.Tensor:
        """Compute torsion angles for batched positions."""
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        n1 = torch.linalg.cross(b1, b2)
        n2 = torch.linalg.cross(b2, b3)
        n1 = n1 / torch.linalg.norm(n1, dim=-1, keepdim=True).clamp(min=1e-10)
        n2 = n2 / torch.linalg.norm(n2, dim=-1, keepdim=True).clamp(min=1e-10)
        b2_norm = b2 / torch.linalg.norm(b2, dim=-1, keepdim=True).clamp(
            min=1e-10
        )
        m1 = torch.linalg.cross(n1, b2_norm)
        x = torch.sum(n1 * n2, dim=-1)
        y = torch.sum(m1 * n2, dim=-1)
        return torch.atan2(y, x)

    # =========================================================================
    # Forward pass
    # =========================================================================

    def forward(self) -> torch.Tensor:
        """
        Reconstruct Cartesian xyz from internal coordinates with chain closure.

        Steps:
        1. Place segment atoms (existing NeRF pipeline)
        2. Solve junction closures (Newton + IFT)
        3. Place junction backbone atoms
        4. Place junction sidechain atoms
        5. Apply frozen overlay

        Returns
        -------
        torch.Tensor
            Reconstructed Cartesian coordinates of shape (N, 3).
        """
        xyz = torch.zeros(
            self.n_atoms, 3, dtype=self._dtype, device=self._device
        )

        # Get rotation matrices for all segments (batched)
        R_matrices = rotation_matrix_euler_zyz(self.segment_orientations)

        # Place segment root atoms
        xyz[self.segment_roots] = self.segment_positions

        # Place secondary roots
        if self.secondary_root_indices.numel() > 0:
            xyz[self.secondary_root_indices] = self.secondary_root_positions

        # Place depth-1 atoms
        xyz = self._place_depth1_atoms(xyz, R_matrices)

        # Place depth-2 atoms
        xyz = self._place_depth2_atoms(xyz, R_matrices)

        # Place depth-3+ atoms (excluding ALL junction atoms)
        for depth_idx in range(len(self._depth_atom_indices)):
            xyz = self._place_atoms_at_depth_fast(xyz, depth_idx)

        # Place rigid ring atoms (non-junction only; junction rings placed below)
        xyz = self._place_rigid_rings(xyz)

        # Solve junction closures, place junction backbone, then sidechain atoms
        if self.n_junctions > 0:
            xyz = self._solve_and_place_junctions(xyz)
            xyz = self._place_junction_sidechain_atoms(xyz)

        # Apply frozen coordinates
        if not self.refinable_mask.all():
            frozen_mask = ~self.refinable_mask
            xyz[frozen_mask] = self.fixed_xyz[frozen_mask]

        return xyz

    def _place_depth1_atoms(
        self, xyz: torch.Tensor, R_matrices: torch.Tensor
    ) -> torch.Tensor:
        """Place atoms at depth 1 (excluding junction atoms)."""
        mask = (self.atom_depth == 1) & ~self.is_junction_atom
        if not mask.any():
            return xyz

        xyz = xyz.clone()
        atom_indices = torch.where(mask)[0]
        seg_ids = self.atom_to_segment[atom_indices]
        parent_idx = self.parent_indices[atom_indices]
        parent_positions = xyz[parent_idx]
        R = R_matrices[seg_ids]
        dir_indices = self.depth1_atom_to_dir_idx[atom_indices]
        base_dirs = self.depth1_dirs[dir_indices]
        rotated_dirs = torch.bmm(R, base_dirs.unsqueeze(-1)).squeeze(-1)
        bond_idx = self.bond_param_indices[mask]
        bond_lengths = self.bond_lengths[bond_idx]
        new_positions = parent_positions + bond_lengths.unsqueeze(-1) * rotated_dirs
        xyz[mask] = new_positions
        return xyz

    def _place_depth2_atoms(
        self, xyz: torch.Tensor, R_matrices: torch.Tensor
    ) -> torch.Tensor:
        """Place atoms at depth 2 (excluding junction atoms)."""
        mask = (self.atom_depth == 2) & ~self.is_junction_atom
        if not mask.any():
            return xyz

        xyz = xyz.clone()
        parent_idx = self.parent_indices[mask]
        grandparent_idx = self.grandparent_indices[mask]
        parent_pos = xyz[parent_idx]
        grandparent_pos = xyz[grandparent_idx]

        bond_idx = self.bond_param_indices[mask]
        angle_idx = self.angle_param_indices[mask]
        bond_lengths = self.bond_lengths[bond_idx]
        angles = self.angles[angle_idx]

        bc = parent_pos - grandparent_pos
        bc_norm = torch.linalg.norm(bc, dim=-1, keepdim=True).clamp(min=1e-10)
        bc_unit = bc / bc_norm

        atom_indices = torch.where(mask)[0]
        perp_idx = self.depth2_atom_to_perp_idx[atom_indices]
        ref_perp = self.depth2_perps[perp_idx]

        seg_ids = self.atom_to_segment[atom_indices]
        R = R_matrices[seg_ids]
        rotated_ref = torch.bmm(R, ref_perp.unsqueeze(-1)).squeeze(-1)

        proj = torch.sum(rotated_ref * bc_unit, dim=-1, keepdim=True)
        rotated_ref = rotated_ref - proj * bc_unit
        ref_norm = torch.linalg.norm(
            rotated_ref, dim=-1, keepdim=True
        ).clamp(min=1e-10)
        rotated_ref = rotated_ref / ref_norm

        theta_internal = torch.pi - angles
        dx = bond_lengths * torch.cos(theta_internal)
        dy = bond_lengths * torch.sin(theta_internal)

        new_positions = (
            parent_pos
            + dx.unsqueeze(-1) * bc_unit
            + dy.unsqueeze(-1) * rotated_ref
        )
        xyz[mask] = new_positions
        return xyz

    def _place_atoms_at_depth_fast(
        self, xyz: torch.Tensor, depth_idx: int
    ) -> torch.Tensor:
        """Place atoms at a depth level using pre-computed indices."""
        atom_idx = self._depth_atom_indices[depth_idx]
        if atom_idx.numel() == 0:
            return xyz

        xyz = xyz.clone()
        parent_idx = self._depth_parent_indices[depth_idx]
        gp_idx = self._depth_gp_indices[depth_idx]
        ggp_idx = self._depth_ggp_indices[depth_idx]

        p1 = xyz[ggp_idx]
        p2 = xyz[gp_idx]
        p3 = xyz[parent_idx]

        bond_idx = self._depth_bond_param_indices[depth_idx]
        angle_idx = self._depth_angle_param_indices[depth_idx]
        torsion_idx = self._depth_torsion_param_indices[depth_idx]

        d = self.bond_lengths[bond_idx]
        theta = self.angles[angle_idx]
        phi = self.torsions[torsion_idx]

        bc = p3 - p2
        bc = bc / torch.linalg.norm(bc, dim=-1, keepdim=True).clamp(min=1e-10)

        ab = p2 - p1
        n = torch.linalg.cross(ab, bc)
        n = n / torch.linalg.norm(n, dim=-1, keepdim=True).clamp(min=1e-10)

        m = torch.linalg.cross(n, bc)

        theta_internal = torch.pi - theta
        sin_theta = torch.sin(theta_internal)
        cos_theta = torch.cos(theta_internal)

        dx = d * cos_theta
        dy = d * sin_theta * torch.cos(phi)
        dz = d * sin_theta * torch.sin(phi)

        new_pos = (
            p3
            + dx.unsqueeze(-1) * bc
            + dy.unsqueeze(-1) * m
            - dz.unsqueeze(-1) * n
        )
        xyz[atom_idx] = new_pos
        return xyz

    def _place_rigid_rings(self, xyz: torch.Tensor) -> torch.Tensor:
        """Place ring atoms as rigid groups."""
        if self.n_rings == 0:
            return xyz

        xyz = xyz.clone()
        anchor_pos = xyz[self.ring_anchor_atoms]
        parent_idx = self.parent_indices[self.ring_anchor_atoms]
        valid_parent_mask = parent_idx >= 0
        parent_pos = torch.zeros_like(anchor_pos)
        parent_pos[valid_parent_mask] = xyz[parent_idx[valid_parent_mask]]

        z_axis = anchor_pos - parent_pos
        z_norm = torch.linalg.norm(z_axis, dim=-1, keepdim=True)
        z_axis = torch.where(
            z_norm > 1e-10,
            z_axis / z_norm.clamp(min=1e-10),
            torch.tensor(
                [0.0, 0.0, 1.0], dtype=self._dtype, device=self._device
            ),
        )
        z_axis[~valid_parent_mask] = torch.tensor(
            [0.0, 0.0, 1.0], dtype=self._dtype, device=self._device
        )

        x_base = torch.zeros_like(z_axis)
        x_base[:, 0] = 1.0
        nearly_x = torch.abs(z_axis[:, 0]) >= 0.9
        x_base[nearly_x, 0] = 0.0
        x_base[nearly_x, 1] = 1.0

        x_axis = (
            x_base - (x_base * z_axis).sum(dim=-1, keepdim=True) * z_axis
        )
        x_axis = x_axis / torch.linalg.norm(
            x_axis, dim=-1, keepdim=True
        ).clamp(min=1e-10)
        y_axis = torch.linalg.cross(z_axis, x_axis)

        R = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        global_offsets = torch.bmm(
            self.ring_local_coords, R.transpose(-1, -2)
        )
        global_pos = global_offsets + anchor_pos.unsqueeze(1)

        valid_mask = self.ring_members >= 0
        anchor_expanded = self.ring_anchor_atoms.unsqueeze(1)
        valid_mask = valid_mask & (self.ring_members != anchor_expanded)

        ring_indices, local_indices = torch.where(valid_mask)
        atom_indices = self.ring_members[ring_indices, local_indices]
        positions = global_pos[ring_indices, local_indices]
        xyz[atom_indices] = positions
        return xyz

    def _solve_and_place_junctions(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Solve junction closures and place junction backbone atoms.

        Extracts (N, CA, C) of the last pre-junction residue as starting
        positions for the NeRF-based FK, uses stored target C positions,
        runs the Newton solver, and scatters backbone atoms into xyz.
        """
        xyz = xyz.clone()

        # Build start positions for each junction: N, CA, C of last pre-junction residue
        p1_list = []  # N positions
        p2_list = []  # CA positions
        p3_list = []  # C positions

        for junc_idx in range(self.n_junctions):
            pre_seg_idx = self._find_pre_junction_segment(junc_idx)
            if pre_seg_idx is not None and pre_seg_idx < len(self._planned_segments):
                last_res = self._planned_segments[pre_seg_idx][-1]
                if last_res in self._backbone_map:
                    bb = self._backbone_map[last_res]
                    p1_list.append(xyz[bb["N"]])
                    p2_list.append(xyz[bb["CA"]])
                    p3_list.append(xyz[bb["C"]])
                else:
                    # Fallback: use identity-like positions
                    p1_list.append(torch.zeros(3, dtype=self._dtype, device=self._device))
                    p2_list.append(torch.tensor([1.0, 0.0, 0.0], dtype=self._dtype, device=self._device))
                    p3_list.append(torch.tensor([2.0, 0.0, 0.0], dtype=self._dtype, device=self._device))
            else:
                p1_list.append(torch.zeros(3, dtype=self._dtype, device=self._device))
                p2_list.append(torch.tensor([1.0, 0.0, 0.0], dtype=self._dtype, device=self._device))
                p3_list.append(torch.tensor([2.0, 0.0, 0.0], dtype=self._dtype, device=self._device))

        p1_start = torch.stack(p1_list)  # (J, 3)
        p2_start = torch.stack(p2_list)  # (J, 3)
        p3_start = torch.stack(p3_list)  # (J, 3)

        # Build dynamic targets: N of first post-junction residue
        target_list = []
        for junc_idx in range(self.n_junctions):
            n_idx = self._junction_post_n_indices[junc_idx]
            if n_idx >= 0:
                target_list.append(xyz[n_idx])
            else:
                # No post-junction segment — use a dummy target
                target_list.append(torch.zeros(3, dtype=self._dtype, device=self._device))
        target_n = torch.stack(target_list)  # (J, 3)

        # Solve junction closures
        phi_psi, backbone_xyz = self.junction_solver(
            p1_start, p2_start, p3_start,
            target_n,
            self.junction_bond_lengths,
            self.junction_nerf_angles,
            self.junction_omega,
            self.junction_psi_prev,
            self.junction_post_bond_length,
            self.junction_post_bond_angle,
        )

        # Scatter junction backbone atoms into xyz
        for i, (junc_idx, output_pos) in enumerate(self._junc_output_indices):
            global_idx = self._junc_global_indices[i]
            xyz[global_idx] = backbone_xyz[junc_idx, output_pos]

        return xyz

    def _place_junction_sidechain_atoms(
        self, xyz: torch.Tensor
    ) -> torch.Tensor:
        """
        Place junction sidechain atoms using NeRF from solved backbone positions.

        Called AFTER the junction solver has placed backbone atoms. Uses the
        pre-built junction sidechain depth indices to place atoms in depth order,
        using the same bond length/angle parameters and stored reference geometry.
        """
        xyz = xyz.clone()

        for depth_list_idx in range(len(self._jsc_depth_atom_indices)):
            atom_idx = self._jsc_depth_atom_indices[depth_list_idx]
            if atom_idx.numel() == 0:
                continue

            actual_depth = depth_list_idx + self._jsc_min_depth

            if actual_depth == 1:
                # Depth-1: use reference directions (like _place_depth1_atoms)
                parent_idx = self._jsc_depth_parent_indices[depth_list_idx]
                parent_pos = xyz[parent_idx]
                bond_idx = self._jsc_depth_bond_param_indices[depth_list_idx]
                bond_lengths = self.bond_lengths[bond_idx]

                seg_ids = self.atom_to_segment[atom_idx]
                R = rotation_matrix_euler_zyz(self.segment_orientations)[seg_ids]
                dir_indices = self.depth1_atom_to_dir_idx[atom_idx]
                base_dirs = self.depth1_dirs[dir_indices]
                rotated_dirs = torch.bmm(R, base_dirs.unsqueeze(-1)).squeeze(-1)

                new_pos = parent_pos + bond_lengths.unsqueeze(-1) * rotated_dirs
                xyz[atom_idx] = new_pos

            elif actual_depth == 2:
                # Depth-2: use reference perpendiculars (like _place_depth2_atoms)
                parent_idx = self._jsc_depth_parent_indices[depth_list_idx]
                gp_idx = self._jsc_depth_gp_indices[depth_list_idx]
                parent_pos = xyz[parent_idx]
                gp_pos = xyz[gp_idx]

                bond_idx = self._jsc_depth_bond_param_indices[depth_list_idx]
                angle_idx = self._jsc_depth_angle_param_indices[depth_list_idx]
                bond_lengths = self.bond_lengths[bond_idx]
                angles = self.angles[angle_idx]

                bc = parent_pos - gp_pos
                bc_norm = torch.linalg.norm(bc, dim=-1, keepdim=True).clamp(min=1e-10)
                bc_unit = bc / bc_norm

                perp_idx = self.depth2_atom_to_perp_idx[atom_idx]
                ref_perp = self.depth2_perps[perp_idx]

                seg_ids = self.atom_to_segment[atom_idx]
                R = rotation_matrix_euler_zyz(self.segment_orientations)[seg_ids]
                rotated_ref = torch.bmm(R, ref_perp.unsqueeze(-1)).squeeze(-1)

                proj = torch.sum(rotated_ref * bc_unit, dim=-1, keepdim=True)
                rotated_ref = rotated_ref - proj * bc_unit
                ref_norm = torch.linalg.norm(
                    rotated_ref, dim=-1, keepdim=True
                ).clamp(min=1e-10)
                rotated_ref = rotated_ref / ref_norm

                theta_internal = torch.pi - angles
                dx = bond_lengths * torch.cos(theta_internal)
                dy = bond_lengths * torch.sin(theta_internal)

                new_pos = (
                    parent_pos
                    + dx.unsqueeze(-1) * bc_unit
                    + dy.unsqueeze(-1) * rotated_ref
                )
                xyz[atom_idx] = new_pos

            else:
                # Depth-3+: standard NeRF formula
                parent_idx = self._jsc_depth_parent_indices[depth_list_idx]
                gp_idx = self._jsc_depth_gp_indices[depth_list_idx]
                ggp_idx = self._jsc_depth_ggp_indices[depth_list_idx]

                p1 = xyz[ggp_idx]
                p2 = xyz[gp_idx]
                p3 = xyz[parent_idx]

                bond_idx = self._jsc_depth_bond_param_indices[depth_list_idx]
                angle_idx = self._jsc_depth_angle_param_indices[depth_list_idx]
                d = self.bond_lengths[bond_idx]
                theta = self.angles[angle_idx]

                # Junction sidechain torsions are in self.torsions (free DOFs)
                torsion_idx = self.torsion_param_indices[atom_idx]
                phi = self.torsions[torsion_idx]

                bc = p3 - p2
                bc = bc / torch.linalg.norm(bc, dim=-1, keepdim=True).clamp(min=1e-10)
                ab = p2 - p1
                n = torch.linalg.cross(ab, bc)
                n = n / torch.linalg.norm(n, dim=-1, keepdim=True).clamp(min=1e-10)
                m = torch.linalg.cross(n, bc)

                theta_internal = torch.pi - theta
                sin_theta = torch.sin(theta_internal)
                cos_theta = torch.cos(theta_internal)
                dx = d * cos_theta
                dy = d * sin_theta * torch.cos(phi)
                dz = d * sin_theta * torch.sin(phi)

                new_pos = (
                    p3
                    + dx.unsqueeze(-1) * bc
                    + dy.unsqueeze(-1) * m
                    - dz.unsqueeze(-1) * n
                )
                xyz[atom_idx] = new_pos

        return xyz

    # =========================================================================
    # Interface methods (same API as SegmentedInternalCoordinateTensor)
    # =========================================================================

    def shake(self, magnitude: float = 0.1) -> torch.Tensor:
        """Add Gaussian noise to internal parameters.

        Perturbs torsions, bond lengths, and bond angles. Segment positions
        and orientations are NOT perturbed because random independent
        translation of segments creates gaps that exceed the junction chain's
        reach. During optimization, the optimizer adjusts these rigid-body
        DOFs smoothly.
        """
        with torch.no_grad():
            if self.bond_lengths.numel() > 0:
                self.bond_lengths.data += (
                    torch.randn_like(self.bond_lengths) * magnitude
                )
                self.bond_lengths.data = torch.clamp(
                    self.bond_lengths.data, min=0.5
                )

            if self.angles.numel() > 0:
                self.angles.data += torch.randn_like(self.angles) * magnitude
                self.angles.data = torch.clamp(
                    self.angles.data, min=0.1, max=torch.pi - 0.1
                )

            if self.torsions.numel() > 0:
                self.torsions.data += (
                    torch.randn_like(self.torsions) * magnitude
                )
                self.torsions.data = torch.atan2(
                    torch.sin(self.torsions.data),
                    torch.cos(self.torsions.data),
                )

        return self.forward()

    def fix(
        self,
        selection: Union[torch.Tensor, slice, None] = None,
        freeze_at_current: bool = True,
    ) -> None:
        """Fix (freeze) atoms."""
        if selection is None:
            selection = slice(None)

        mask = self._selection_to_mask(selection)

        if freeze_at_current:
            with torch.no_grad():
                current_xyz = self.forward()
                self.fixed_xyz[mask] = current_xyz[mask]

        self.refinable_mask[mask] = False

    def freeze(
        self,
        selection: Union[torch.Tensor, slice, None] = None,
        freeze_at_current: bool = True,
    ) -> None:
        """Alias for fix()."""
        self.fix(selection, freeze_at_current)

    def refine(
        self,
        selection: Union[torch.Tensor, slice, None] = None,
        rebuild: bool = True,
    ) -> None:
        """Make atoms refinable."""
        if selection is None:
            selection = slice(None)

        mask = self._selection_to_mask(selection)

        if rebuild:
            self._update_internal_coords_from_xyz(self.fixed_xyz, mask)

        self.refinable_mask[mask] = True

    def unfreeze(
        self,
        selection: Union[torch.Tensor, slice, None] = None,
        rebuild: bool = True,
    ) -> None:
        """Alias for refine()."""
        self.refine(selection, rebuild)

    def fix_all(self, freeze_at_current: bool = True) -> None:
        """Fix all atoms."""
        self.fix(None, freeze_at_current)

    def freeze_all(self, freeze_at_current: bool = True) -> None:
        """Alias for fix_all()."""
        self.fix_all(freeze_at_current)

    def refine_all(self, rebuild: bool = True) -> None:
        """Make all atoms refinable."""
        self.refine(None, rebuild)

    def unfreeze_all(self, rebuild: bool = True) -> None:
        """Alias for refine_all()."""
        self.refine_all(rebuild)

    def _selection_to_mask(
        self, selection: Union[torch.Tensor, slice, None]
    ) -> torch.Tensor:
        """Convert selection to boolean mask."""
        if isinstance(selection, torch.Tensor) and selection.dtype == torch.bool:
            return selection
        elif isinstance(selection, torch.Tensor):
            mask = torch.zeros(
                self.n_atoms, dtype=torch.bool, device=self._device
            )
            mask[selection] = True
            return mask
        elif isinstance(selection, slice):
            mask = torch.zeros(
                self.n_atoms, dtype=torch.bool, device=self._device
            )
            mask[selection] = True
            return mask
        else:
            raise TypeError(
                f"selection must be Tensor, slice, or None, got {type(selection)}"
            )

    def _update_internal_coords_from_xyz(
        self, xyz: torch.Tensor, mask: torch.Tensor
    ) -> None:
        """Update internal coordinate parameters from xyz."""
        # Bonds
        bond_update_mask = mask & (self.atom_depth >= 1)
        if bond_update_mask.any():
            child_xyz = xyz[bond_update_mask]
            parent_xyz = xyz[self.parent_indices[bond_update_mask]]
            new_bonds = torch.linalg.norm(child_xyz - parent_xyz, dim=-1)
            bond_idx = self.bond_param_indices[bond_update_mask]
            with torch.no_grad():
                self.bond_lengths.data[bond_idx] = new_bonds

        # Angles
        angle_update_mask = mask & (self.atom_depth >= 2)
        if angle_update_mask.any():
            child_xyz = xyz[angle_update_mask]
            parent_xyz = xyz[self.parent_indices[angle_update_mask]]
            grandparent_xyz = xyz[self.grandparent_indices[angle_update_mask]]
            v1 = child_xyz - parent_xyz
            v2 = grandparent_xyz - parent_xyz
            cos_angles = torch.sum(v1 * v2, dim=-1) / (
                torch.linalg.norm(v1, dim=-1)
                * torch.linalg.norm(v2, dim=-1)
                + 1e-10
            )
            cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
            new_angles = torch.acos(cos_angles)
            angle_idx = self.angle_param_indices[angle_update_mask]
            with torch.no_grad():
                self.angles.data[angle_idx] = new_angles

        # Torsions (only free torsions — excludes junction backbone)
        torsion_update_mask = (
            mask & (self.atom_depth >= 3) & ~self.is_junction_backbone
        )
        if torsion_update_mask.any():
            child_xyz = xyz[torsion_update_mask]
            parent_xyz = xyz[self.parent_indices[torsion_update_mask]]
            grandparent_xyz = xyz[
                self.grandparent_indices[torsion_update_mask]
            ]
            great_grandparent_xyz = xyz[
                self.great_grandparent_indices[torsion_update_mask]
            ]
            new_torsions = self._compute_torsion_angles(
                great_grandparent_xyz, grandparent_xyz, parent_xyz, child_xyz
            )
            torsion_idx = self.torsion_param_indices[torsion_update_mask]
            with torch.no_grad():
                self.torsions.data[torsion_idx] = new_torsions

        # Segment positions
        root_in_mask = mask[self.segment_roots]
        if root_in_mask.any():
            with torch.no_grad():
                self.segment_positions.data[root_in_mask] = xyz[
                    self.segment_roots[root_in_mask]
                ]

        # Secondary root positions
        if self.secondary_root_indices.numel() > 0:
            sec_root_in_mask = mask[self.secondary_root_indices]
            if sec_root_in_mask.any():
                with torch.no_grad():
                    self.secondary_root_positions.data[sec_root_in_mask] = xyz[
                        self.secondary_root_indices[sec_root_in_mask]
                    ]

    @property
    def n_refinable(self) -> int:
        return self.refinable_mask.sum().item()

    @property
    def n_fixed(self) -> int:
        return (~self.refinable_mask).sum().item()

    @property
    def closure_residuals(self) -> Optional[torch.Tensor]:
        """Get last closure residuals from the junction solver."""
        return self.junction_solver.closure_residuals

    @property
    def max_closure_gap(self) -> float:
        """Maximum closure gap in Angstroms across all junctions."""
        if self.n_junctions == 0:
            return 0.0
        with torch.no_grad():
            xyz = self.forward()
            max_gap = 0.0
            for junc_idx in range(self.n_junctions):
                junc_info = self._junction_data[junc_idx]
                # Check gap between last junction atom and first post-junction atom
                last_res = junc_info["residues"][-1]
                last_bb = junc_info["backbone"][-1]
                c_pos = xyz[last_bb["C"]]

                post_seg_idx = self._find_post_junction_segment(junc_idx)
                if post_seg_idx is not None and post_seg_idx < len(self._planned_segments):
                    next_res = self._planned_segments[post_seg_idx][0]
                    if next_res in self._backbone_map:
                        n_pos = xyz[self._backbone_map[next_res]["N"]]
                        gap = torch.linalg.norm(c_pos - n_pos).item()
                        max_gap = max(max_gap, gap)
            return max_gap

    def to(self, device=None, dtype=None):
        """Move tensor to specified device and/or dtype."""
        if device is not None:
            self._device = torch.device(device)
        if dtype is not None:
            self._dtype = dtype
        return super().to(device=device, dtype=dtype)

    def __repr__(self) -> str:
        n_secondary = self.secondary_root_indices.numel()
        return (
            f"ClosedSegmentedInternalCoordinateTensor("
            f"n_atoms={self.n_atoms}, "
            f"n_segments={self.n_segments}, "
            f"n_junctions={self.n_junctions}, "
            f"n_aa_per_segment={self.n_aa_per_segment}, "
            f"junction_size={self.junction_size}, "
            f"n_secondary_roots={n_secondary}, "
            f"n_refinable={self.n_refinable}, n_fixed={self.n_fixed}, "
            f"n_bonds={self.n_bonds}, n_angles={self.n_angles}, "
            f"n_torsions={self.n_torsions}, n_rings={self.n_rings}, "
            f"max_depth={self.max_depth}, device={self._device})"
        )


# =========================================================================
# Helper functions (module-level)
# =========================================================================


def _compute_angle(
    p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor,
) -> torch.Tensor:
    """Compute angle at p2 between p1-p2-p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_a = torch.dot(v1, v2) / (
        torch.linalg.norm(v1) * torch.linalg.norm(v2) + 1e-10
    )
    cos_a = torch.clamp(cos_a, -1.0, 1.0)
    return torch.acos(cos_a)


def _compute_torsion(
    p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor,
) -> torch.Tensor:
    """Compute torsion angle for four points."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)
    n1 = n1 / torch.linalg.norm(n1).clamp(min=1e-10)
    n2 = n2 / torch.linalg.norm(n2).clamp(min=1e-10)
    b2_norm = b2 / torch.linalg.norm(b2).clamp(min=1e-10)
    m1 = torch.linalg.cross(n1, b2_norm)
    x = torch.dot(n1, n2)
    y = torch.dot(m1, n2)
    return torch.atan2(y, x)
