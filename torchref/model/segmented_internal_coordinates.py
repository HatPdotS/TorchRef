"""
Segmented internal coordinate parametrization for atomic structures.

This module provides the SegmentedInternalCoordinateTensor class which addresses
the "lever arm problem" in internal coordinate parametrization by breaking the
molecular chain into independent segments, each with its own rigid body parameters.

Key features:
- Segments the molecule into groups of N amino acids (default: 3 per segment)
- Each segment has independent internal coordinates (bonds, angles, torsions)
- Each segment has rigid body parameters (position + orientation)
- Shallow spanning trees within segments (depth ~15-30 instead of ~1000)
- Changes in one segment don't propagate to distant segments
- Fully differentiable reconstruction from internal coordinates
- Parallelized construction for fast initialization
- Fused ring systems (indole in TRP, purines, etc.) treated as single rigid groups

This approach solves the lever arm problem where small torsion changes near the
root of a deep tree cause large displacements at distant atoms.
"""

from collections import defaultdict
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torchref.base.alignment.rotation import rotation_matrix_euler_zyz
from torchref.utils.caching import CachedForwardMixin


class SegmentedInternalCoordinateTensor(CachedForwardMixin, nn.Module):
    """
    Parameter wrapper using segmented internal coordinates.

    Stores: per-segment bond_lengths, angles, torsions, segment_positions, segment_orientations
    Reconstructs: Cartesian xyz on forward()

    This provides a physically meaningful parametrization that avoids the lever arm
    problem by breaking the molecule into independent segments, each with shallow
    spanning trees and rigid body parameters.

    Parameters
    ----------
    initial_xyz : torch.Tensor
        Initial Cartesian coordinates of shape (N, 3).
    pdb : pd.DataFrame
        PDB DataFrame with columns 'chainid', 'resseq', 'name', 'index'.
    n_aa_per_segment : int, optional
        Number of amino acids per segment. Default is 3.
    bond_cutoff : float, optional
        Distance cutoff for bond detection in Angstroms. Default is 2.0.
    requires_grad : bool, optional
        Whether parameters should have gradients. Default is True.
    dtype : torch.dtype, optional
        Data type for tensors. Default is same as initial_xyz.
    device : torch.device, optional
        Device for tensors. Default is same as initial_xyz.

    Attributes
    ----------
    n_atoms : int
        Number of atoms.
    n_segments : int
        Number of segments.
    max_depth : int
        Maximum depth in any segment's spanning tree.
    bond_lengths : nn.Parameter
        Bond length parameters in Angstroms.
    angles : nn.Parameter
        Angle parameters in radians.
    torsions : nn.Parameter
        Torsion angle parameters in radians.
    segment_positions : nn.Parameter
        Absolute positions of segment root atoms.
    segment_orientations : nn.Parameter
        ZYZ Euler angle orientations for each segment.
    """

    # Standard amino acid residue names
    AA_NAMES = frozenset({
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
        "TYR", "VAL", "MSE", "SEC",
    })

    def __init__(
        self,
        initial_xyz: torch.Tensor,
        pdb: pd.DataFrame,
        n_aa_per_segment: int = 3,
        bond_cutoff: float = 2.0,
        cif_dict: Optional[dict] = None,
        requires_grad: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize SegmentedInternalCoordinateTensor.

        Parameters
        ----------
        initial_xyz : torch.Tensor
            Initial Cartesian coordinates of shape (N, 3).
        pdb : pd.DataFrame
            PDB DataFrame with columns 'chainid', 'resseq', 'name', 'index', 'resname'.
        n_aa_per_segment : int, optional
            Number of amino acids per segment. Default is 3.
        bond_cutoff : float, optional
            Distance cutoff for bond detection in Angstroms (used as fallback).
            Default is 2.0.
        cif_dict : dict, optional
            CIF dictionary containing bond definitions per residue type.
            If provided, bonds are determined from chemical definitions rather
            than distances, which is more robust for structures with poor geometry.
            Expected format: cif_dict[resname]['bonds'] is a DataFrame with
            'atom1' and 'atom2' columns.
        requires_grad : bool, optional
            Whether parameters should have gradients. Default is True.
        dtype : torch.dtype, optional
            Data type for tensors. Default is same as initial_xyz.
        device : torch.device, optional
            Device for tensors. Default is same as initial_xyz.
        """
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
        self.cif_dict = cif_dict

        # Move initial coordinates to specified device/dtype
        initial_xyz = initial_xyz.to(dtype=dtype, device=device)

        # Store a copy of the PDB DataFrame for reference
        self.pdb = pdb

        # Build segments from residue information (vectorized)
        self._build_segments_fast(pdb)

        # Build internal coordinate trees for each segment (parallelized)
        self._build_segment_trees_parallel(initial_xyz)

        # Extract internal coordinates from initial xyz (vectorized)
        self._extract_internal_coords(initial_xyz, requires_grad)

    @property
    def dtype(self):
        """Return the dtype of tensors."""
        return self._dtype

    @property
    def device(self):
        """Return the device of tensors."""
        return self._device

    def _build_segments_fast(self, pdb: pd.DataFrame) -> None:
        """
        Group atoms into segments based on residue information (vectorized).

        Uses pandas groupby operations for efficient residue grouping.

        Parameters
        ----------
        pdb : pd.DataFrame
            PDB DataFrame with 'chainid', 'resseq', 'name', 'index', 'resname' columns.
        """
        device = self._device

        # Vectorized residue classification
        is_protein = pdb["resname"].str.upper().isin(self.AA_NAMES)

        # Group by (chainid, resseq) - this is O(N log N)
        grouped = pdb.groupby(["chainid", "resseq"], sort=False)

        # Build residue info lists efficiently
        residue_data = []
        for (chain_id, resseq), group in grouped:
            atom_indices = group["index"].values.tolist()
            res_name = group["resname"].iloc[0]
            is_aa = res_name.upper() in self.AA_NAMES
            residue_data.append({
                "chain_id": chain_id,
                "resseq": resseq,
                "atom_indices": atom_indices,
                "is_protein": is_aa,
                "resname": res_name,
            })

        # Separate protein and non-protein residues
        protein_residues = [r for r in residue_data if r["is_protein"]]
        non_protein_residues = [r for r in residue_data if not r["is_protein"]]

        # Group protein residues by chain and sort
        chain_residues = {}
        for r in protein_residues:
            chain_id = r["chain_id"]
            if chain_id not in chain_residues:
                chain_residues[chain_id] = []
            chain_residues[chain_id].append(r)

        for chain_id in chain_residues:
            chain_residues[chain_id].sort(key=lambda x: x["resseq"])

        # Build segments
        segments = []
        segment_root_atoms = []

        # Create name-to-index lookups for backbone atoms (vectorized)
        # Use N atom of first residue as root - this ensures connectivity
        # since peptide chain goes N->CA->C->N->...
        n_mask = pdb["name"].str.strip() == "N"
        n_indices = pdb.loc[n_mask, "index"].values
        n_resseqs = pdb.loc[n_mask, "resseq"].values
        n_chains = pdb.loc[n_mask, "chainid"].values
        n_lookup = {(c, r): i for c, r, i in zip(n_chains, n_resseqs, n_indices)}

        # Fallback to CA if N not found
        ca_mask = pdb["name"].str.strip() == "CA"
        ca_indices = pdb.loc[ca_mask, "index"].values
        ca_resseqs = pdb.loc[ca_mask, "resseq"].values
        ca_chains = pdb.loc[ca_mask, "chainid"].values
        ca_lookup = {(c, r): i for c, r, i in zip(ca_chains, ca_resseqs, ca_indices)}

        # Process each chain
        for chain_id, residues in chain_residues.items():
            n_residues = len(residues)

            for i in range(0, n_residues, self.n_aa_per_segment):
                segment_res = residues[i:i + self.n_aa_per_segment]

                # Collect all atoms in segment
                segment_atoms = []
                for r in segment_res:
                    segment_atoms.extend(r["atom_indices"])

                # Use N atom of first residue as root for better connectivity
                # This ensures BFS can traverse forward through all atoms
                first_res = segment_res[0]
                first_key = (first_res["chain_id"], first_res["resseq"])

                if first_key in n_lookup:
                    root_atom = n_lookup[first_key]
                elif first_key in ca_lookup:
                    root_atom = ca_lookup[first_key]
                else:
                    root_atom = first_res["atom_indices"][0]

                segments.append(segment_atoms)
                segment_root_atoms.append(root_atom)

        # Add non-protein residues as individual segments
        for r in non_protein_residues:
            segments.append(r["atom_indices"])
            segment_root_atoms.append(r["atom_indices"][0])

        self.n_segments = len(segments)

        # Create mapping tensors (vectorized)
        atom_to_segment = torch.full((self.n_atoms,), -1, dtype=torch.long, device=device)
        max_segment_size = max(len(s) for s in segments) if segments else 0
        segment_atom_indices = torch.full(
            (self.n_segments, max_segment_size), -1, dtype=torch.long, device=device
        )
        segment_sizes = torch.zeros(self.n_segments, dtype=torch.long, device=device)

        # Vectorized assignment using numpy then convert
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
            torch.tensor(segment_root_atoms, dtype=torch.long, device=device)
        )

        self._segments = segments

    def _build_adjacency_from_cif(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Build adjacency matrix from CIF dictionary bond definitions.

        Uses chemical bond definitions from the CIF dictionary for intra-residue
        bonds and adds peptide bonds (C-N) between consecutive residues.
        Falls back to distance-based detection for atoms not in the dictionary.

        Parameters
        ----------
        xyz : torch.Tensor
            Atomic coordinates of shape (N, 3).

        Returns
        -------
        torch.Tensor
            Boolean adjacency matrix of shape (N, N).
        """
        device = self._device
        n_atoms = self.n_atoms
        pdb = self.pdb

        # Collect bond pairs as lists for batch processing
        bond_idx1_list = []
        bond_idx2_list = []

        # Build fast lookup: create multi-index for vectorized matching
        # (chainid, resseq, atom_name) -> atom_index
        pdb_lookup = pdb.set_index(["chainid", "resseq"])
        pdb["name_stripped"] = pdb["name"].str.strip()

        # Group atoms by residue for fast lookup
        residue_groups = pdb.groupby(["chainid", "resseq"])

        # 1. Build intra-residue bonds from CIF dictionary (vectorized per residue)
        atoms_with_cif_bonds = set()

        for (chainid, resseq), group in residue_groups:
            resname = group["resname"].iloc[0]

            if resname not in self.cif_dict:
                continue

            cif_residue = self.cif_dict[resname]
            if "bonds" not in cif_residue:
                continue

            cif_bonds = cif_residue["bonds"]

            # Create atom name to index lookup for this residue
            name_to_idx = dict(zip(group["name_stripped"], group["index"]))

            # Vectorized bond matching using pandas merge
            atom1_names = cif_bonds["atom1"].str.strip().values
            atom2_names = cif_bonds["atom2"].str.strip().values

            for a1, a2 in zip(atom1_names, atom2_names):
                if a1 in name_to_idx and a2 in name_to_idx:
                    idx1 = name_to_idx[a1]
                    idx2 = name_to_idx[a2]
                    bond_idx1_list.append(idx1)
                    bond_idx2_list.append(idx2)
                    atoms_with_cif_bonds.add(idx1)
                    atoms_with_cif_bonds.add(idx2)

        # 2. Build peptide bonds (C-N between consecutive residues)
        # Pre-compute C and N atom indices per residue
        c_atoms = pdb[pdb["name_stripped"] == "C"].set_index(["chainid", "resseq"])["index"]
        n_atoms_df = pdb[pdb["name_stripped"] == "N"].set_index(["chainid", "resseq"])["index"]

        for chainid in pdb["chainid"].unique():
            chain = pdb[pdb["chainid"] == chainid]
            chain_aa = chain[chain["resname"].isin(self.AA_NAMES)]
            resseqs = sorted(chain_aa["resseq"].unique())

            for i in range(len(resseqs) - 1):
                resseq1 = resseqs[i]
                resseq2 = resseqs[i + 1]

                # Only add peptide bond if residues are consecutive
                if resseq2 - resseq1 != 1:
                    continue

                try:
                    idx_c = c_atoms.loc[(chainid, resseq1)]
                    idx_n = n_atoms_df.loc[(chainid, resseq2)]
                    bond_idx1_list.append(idx_c)
                    bond_idx2_list.append(idx_n)
                    atoms_with_cif_bonds.add(idx_c)
                    atoms_with_cif_bonds.add(idx_n)
                except KeyError:
                    continue

        # Convert to tensors and build adjacency
        adjacency = torch.zeros(n_atoms, n_atoms, dtype=torch.bool, device=device)

        if bond_idx1_list:
            idx1 = torch.tensor(bond_idx1_list, dtype=torch.long, device=device)
            idx2 = torch.tensor(bond_idx2_list, dtype=torch.long, device=device)
            adjacency[idx1, idx2] = True
            adjacency[idx2, idx1] = True

        # 3. Fallback: use distance-based detection for atoms without CIF bonds
        atoms_without_cif = set(range(n_atoms)) - atoms_with_cif_bonds

        if atoms_without_cif:
            atoms_list = sorted(atoms_without_cif)
            atoms_tensor = torch.tensor(atoms_list, dtype=torch.long, device=device)

            # Compute distances only for atoms without CIF bonds
            xyz_subset = xyz[atoms_tensor]
            distances_subset = torch.cdist(xyz_subset, xyz)

            # Find bonds based on distance
            bond_mask = (distances_subset < self.bond_cutoff) & (distances_subset > 0.1)

            # Vectorized addition to adjacency
            rows, cols = torch.where(bond_mask)
            adjacency[atoms_tensor[rows], cols] = True
            adjacency[cols, atoms_tensor[rows]] = True

        # Clean up temporary column
        pdb.drop(columns=["name_stripped"], inplace=True, errors="ignore")

        return adjacency

    def _build_segment_trees_parallel(self, xyz: torch.Tensor) -> None:
        """
        Build spanning trees for all segments using fully parallel multi-source BFS.

        Uses vectorized tensor operations to process all segments simultaneously,
        achieving significant speedup compared to sequential per-segment BFS.

        Parameters
        ----------
        xyz : torch.Tensor
            Atomic coordinates of shape (N, 3).
        """
        device = self._device
        n_atoms = self.n_atoms

        # Initialize per-atom arrays
        parent_indices = torch.full((n_atoms,), -1, dtype=torch.long, device=device)
        grandparent_indices = torch.full((n_atoms,), -1, dtype=torch.long, device=device)
        great_grandparent_indices = torch.full((n_atoms,), -1, dtype=torch.long, device=device)
        atom_depth = torch.full((n_atoms,), -1, dtype=torch.long, device=device)

        # Build molecular graph
        if self.cif_dict is not None:
            # Use CIF dictionary for bond definitions (more robust for bad geometry)
            adjacency = self._build_adjacency_from_cif(xyz)
        else:
            # Fallback to distance-based bond detection
            distances = torch.cdist(xyz, xyz)
            adjacency = (distances < self.bond_cutoff) & (distances > 0.1)

        # Build compact neighbor list representation for fast lookups (fully vectorized)
        neighbor_counts = adjacency.sum(dim=1).to(torch.long)
        max_neighbors = neighbor_counts.max().item()

        # Create padded neighbor lists using vectorized scatter
        # Edge list: (source, target) pairs
        edge_src, edge_tgt = torch.where(adjacency)

        # Compute position within each source's neighbor list using segment cumsum
        # Sort edges by source to group them
        sorted_indices = torch.argsort(edge_src, stable=True)
        edge_src_sorted = edge_src[sorted_indices]
        edge_tgt_sorted = edge_tgt[sorted_indices]

        # Position within source group using vectorized segment cumsum
        neighbors_padded = torch.full(
            (n_atoms, max_neighbors), -1, dtype=torch.long, device=device
        )

        if edge_src_sorted.numel() > 0:
            # Compute cumsum offset for each source atom
            offsets = torch.zeros(n_atoms + 1, dtype=torch.long, device=device)
            offsets[1:] = torch.cumsum(neighbor_counts, dim=0)

            # For each edge, position = edge_index - offset[source]
            edge_global_idx = torch.arange(len(edge_src_sorted), device=device)
            pos_in_group = edge_global_idx - offsets[edge_src_sorted]

            # Scatter targets into padded array
            neighbors_padded[edge_src_sorted, pos_in_group] = edge_tgt_sorted

        # ===== PARALLEL MULTI-SOURCE BFS =====
        # Key insight: Run BFS from all segment roots simultaneously
        # using vectorized tensor operations

        # Initialize root atoms (depth 0)
        atom_depth[self.segment_roots] = 0

        # Track visited atoms using depth >= 0
        # Initialize current frontier with all roots
        current_frontier = self.segment_roots.clone()

        current_depth = 0
        max_depth = 0

        while current_frontier.numel() > 0:
            # Get all neighbors of current frontier atoms (vectorized)
            frontier_neighbors = neighbors_padded[current_frontier]  # (F, max_neighbors)

            # Flatten to get candidate neighbors
            candidates = frontier_neighbors.reshape(-1)  # (F * max_neighbors,)

            # Create parent mapping: which frontier atom is each candidate from
            frontier_expanded = current_frontier.unsqueeze(1).expand_as(frontier_neighbors)
            parents = frontier_expanded.reshape(-1)  # (F * max_neighbors,)

            # Filter valid candidates (not -1 padding)
            valid_mask = candidates >= 0
            candidates = candidates[valid_mask]
            parents = parents[valid_mask]

            if candidates.numel() == 0:
                break

            # Filter candidates: must be unvisited (depth == -1)
            unvisited_mask = atom_depth[candidates] == -1
            candidates = candidates[unvisited_mask]
            parents = parents[unvisited_mask]

            if candidates.numel() == 0:
                break

            # Filter candidates: must be in same segment as parent
            candidate_segments = self.atom_to_segment[candidates]
            parent_segments = self.atom_to_segment[parents]
            same_segment_mask = candidate_segments == parent_segments
            candidates = candidates[same_segment_mask]
            parents = parents[same_segment_mask]

            if candidates.numel() == 0:
                break

            # Handle duplicates: if same atom is reachable from multiple parents,
            # keep only the first one (using unique with return_inverse)
            unique_candidates, inverse_indices = torch.unique(
                candidates, return_inverse=True
            )

            # For each unique candidate, get the first parent
            first_occurrence = torch.zeros(
                unique_candidates.numel(), dtype=torch.long, device=device
            )
            # Scatter the parent indices - last write wins, but all parents are valid
            # for BFS so this is fine
            first_occurrence.scatter_(0, inverse_indices, parents)

            # Update depth and parent for new frontier
            new_depth = current_depth + 1
            atom_depth[unique_candidates] = new_depth
            parent_indices[unique_candidates] = first_occurrence

            # Update grandparent: grandparent = parent of parent
            parent_of_new = first_occurrence
            gp_of_new = parent_indices[parent_of_new]
            has_grandparent = gp_of_new >= 0
            grandparent_indices[unique_candidates[has_grandparent]] = gp_of_new[
                has_grandparent
            ]

            # Update great-grandparent: ggp = grandparent of parent
            ggp_of_new = grandparent_indices[parent_of_new]
            has_ggp = ggp_of_new >= 0
            great_grandparent_indices[unique_candidates[has_ggp]] = ggp_of_new[has_ggp]

            # Update max depth
            max_depth = new_depth

            # Next frontier
            current_frontier = unique_candidates
            current_depth = new_depth

        # ===== HANDLE DISCONNECTED COMPONENTS =====
        # Chain breaks (missing residues) can create disconnected components
        # within segments. For each disconnected component, pick a new root
        # and run BFS from it.

        max_component_iterations = 100  # Safety limit
        for _ in range(max_component_iterations):
            # Find atoms that are still unreached
            orphan_mask = (atom_depth == -1) & (self.atom_to_segment >= 0)
            if not orphan_mask.any():
                break

            orphan_indices = torch.where(orphan_mask)[0]

            # Pick one orphan from each segment as a new root
            orphan_segments = self.atom_to_segment[orphan_indices]
            unique_orphan_segments = torch.unique(orphan_segments)

            # For each segment with orphans, pick the first orphan as new root
            new_roots = []
            for seg in unique_orphan_segments:
                seg_orphans = orphan_indices[orphan_segments == seg]
                if seg_orphans.numel() > 0:
                    new_roots.append(seg_orphans[0].item())

            if not new_roots:
                break

            new_roots = torch.tensor(new_roots, dtype=torch.long, device=device)

            # Initialize new roots at depth 0 (they become local roots)
            atom_depth[new_roots] = 0
            parent_indices[new_roots] = -1

            # Run BFS from new roots
            current_frontier = new_roots

            while current_frontier.numel() > 0:
                frontier_neighbors = neighbors_padded[current_frontier]
                candidates = frontier_neighbors.reshape(-1)
                frontier_expanded = current_frontier.unsqueeze(1).expand_as(frontier_neighbors)
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

                # Get depth of parents
                parent_depths = atom_depth[first_occurrence]
                new_depths = parent_depths + 1

                atom_depth[unique_candidates] = new_depths
                parent_indices[unique_candidates] = first_occurrence

                parent_of_new = first_occurrence
                gp_of_new = parent_indices[parent_of_new]
                has_grandparent = gp_of_new >= 0
                grandparent_indices[unique_candidates[has_grandparent]] = gp_of_new[
                    has_grandparent
                ]

                ggp_of_new = grandparent_indices[parent_of_new]
                has_ggp = ggp_of_new >= 0
                great_grandparent_indices[unique_candidates[has_ggp]] = ggp_of_new[has_ggp]

                max_depth = max(max_depth, new_depths.max().item())
                current_frontier = unique_candidates

        # Identify secondary roots (depth-0 atoms that are not segment roots)
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

        # Build parameter index mappings (vectorized)
        self._build_param_indices_vectorized()

        # Pre-compute depth indices for fast forward pass
        self._build_depth_indices()

        # Detect rings (vectorized)
        self._detect_rings_vectorized(adjacency)

    def _build_param_indices_vectorized(self) -> None:
        """
        Build mappings from atoms to parameter indices using vectorized operations.
        """
        device = self._device

        # Vectorized masks
        has_bond = self.atom_depth >= 1
        has_angle = self.atom_depth >= 2
        has_torsion = self.atom_depth >= 3

        self.n_bonds = has_bond.sum().item()
        self.n_angles = has_angle.sum().item()
        self.n_torsions = has_torsion.sum().item()

        # Vectorized cumsum for index assignment
        bond_param_indices = torch.full((self.n_atoms,), -1, dtype=torch.long, device=device)
        angle_param_indices = torch.full((self.n_atoms,), -1, dtype=torch.long, device=device)
        torsion_param_indices = torch.full((self.n_atoms,), -1, dtype=torch.long, device=device)

        # Use cumsum for efficient index computation
        if has_bond.any():
            bond_cumsum = torch.cumsum(has_bond.long(), dim=0) - 1
            bond_param_indices[has_bond] = bond_cumsum[has_bond]

        if has_angle.any():
            angle_cumsum = torch.cumsum(has_angle.long(), dim=0) - 1
            angle_param_indices[has_angle] = angle_cumsum[has_angle]

        if has_torsion.any():
            torsion_cumsum = torch.cumsum(has_torsion.long(), dim=0) - 1
            torsion_param_indices[has_torsion] = torsion_cumsum[has_torsion]

        self.register_buffer("bond_param_indices", bond_param_indices)
        self.register_buffer("angle_param_indices", angle_param_indices)
        self.register_buffer("torsion_param_indices", torsion_param_indices)

    def _build_depth_indices(self) -> None:
        """
        Pre-compute atom indices for each depth level.
        """
        depth_atom_indices = []
        depth_parent_indices = []
        depth_gp_indices = []
        depth_ggp_indices = []
        depth_bond_param_indices = []
        depth_angle_param_indices = []
        depth_torsion_param_indices = []

        for d in range(3, self.max_depth + 1):
            mask = self.atom_depth == d
            if mask.any():
                atom_idx = torch.where(mask)[0]
                depth_atom_indices.append(atom_idx)
                depth_parent_indices.append(self.parent_indices[atom_idx])
                depth_gp_indices.append(self.grandparent_indices[atom_idx])
                depth_ggp_indices.append(self.great_grandparent_indices[atom_idx])
                depth_bond_param_indices.append(self.bond_param_indices[atom_idx])
                depth_angle_param_indices.append(self.angle_param_indices[atom_idx])
                depth_torsion_param_indices.append(self.torsion_param_indices[atom_idx])
            else:
                empty = torch.tensor([], dtype=torch.long, device=self._device)
                depth_atom_indices.append(empty)
                depth_parent_indices.append(empty)
                depth_gp_indices.append(empty)
                depth_ggp_indices.append(empty)
                depth_bond_param_indices.append(empty)
                depth_angle_param_indices.append(empty)
                depth_torsion_param_indices.append(empty)

        self._depth_atom_indices = depth_atom_indices
        self._depth_parent_indices = depth_parent_indices
        self._depth_gp_indices = depth_gp_indices
        self._depth_ggp_indices = depth_ggp_indices
        self._depth_bond_param_indices = depth_bond_param_indices
        self._depth_angle_param_indices = depth_angle_param_indices
        self._depth_torsion_param_indices = depth_torsion_param_indices

    def _detect_rings_vectorized(self, adjacency: torch.Tensor) -> None:
        """
        Detect rings in the molecular graph using vectorized operations.

        Fused ring systems (like indole in Tryptophan) are automatically merged
        into single rigid groups. Any rings that share atoms are combined.

        Parameters
        ----------
        adjacency : torch.Tensor
            Boolean adjacency matrix of shape (N, N).
        """
        device = self._device
        n_atoms = self.n_atoms

        ring_member_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        ring_group_id = torch.full((n_atoms,), -1, dtype=torch.long, device=device)

        # Find back edges using vectorized operations
        # An edge (i, j) is a back edge if neither i is parent of j nor j is parent of i
        edge_i, edge_j = torch.where(adjacency.triu(diagonal=1))

        # Check if edges are in spanning tree
        is_tree_edge = (
            (self.parent_indices[edge_i] == edge_j) |
            (self.parent_indices[edge_j] == edge_i)
        )

        # Check if in same segment
        same_segment = self.atom_to_segment[edge_i] == self.atom_to_segment[edge_j]

        # Back edges: not tree edges, same segment
        back_edge_mask = ~is_tree_edge & same_segment
        back_edge_i = edge_i[back_edge_mask]
        back_edge_j = edge_j[back_edge_mask]

        # First pass: detect individual rings from back edges
        individual_rings = []

        for idx in range(len(back_edge_i)):
            i = back_edge_i[idx].item()
            j = back_edge_j[idx].item()

            # Find path from i to j through tree
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

            # Add path from i to common ancestor
            current = i
            while current != common_ancestor:
                ring_atoms.append(current)
                current = self.parent_indices[current].item()

            individual_rings.append(set(ring_atoms))

        # Second pass: merge overlapping rings into ring systems
        # This handles fused rings like indole (TRP), naphthalene, etc.
        ring_systems = self._merge_overlapping_rings(individual_rings)

        # Build final ring data structures
        ring_anchors = []
        ring_members_list = []

        for ring_idx, ring_atoms_set in enumerate(ring_systems):
            ring_atoms = list(ring_atoms_set)

            # Mark ring atoms
            for atom in ring_atoms:
                ring_member_mask[atom] = True
                ring_group_id[atom] = ring_idx

            # Anchor is atom closest to root (smallest depth)
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
                torch.tensor(ring_anchors, dtype=torch.long, device=device)
            )
            max_ring_size = max(len(r) for r in ring_members_list)
            ring_members_tensor = torch.full(
                (self.n_rings, max_ring_size), -1, dtype=torch.long, device=device
            )
            ring_sizes = torch.zeros(self.n_rings, dtype=torch.long, device=device)
            for i, members in enumerate(ring_members_list):
                ring_sizes[i] = len(members)
                ring_members_tensor[i, :len(members)] = torch.tensor(
                    members, dtype=torch.long, device=device
                )
            self.register_buffer("ring_members", ring_members_tensor)
            self.register_buffer("ring_sizes", ring_sizes)
        else:
            self.register_buffer(
                "ring_anchor_atoms",
                torch.tensor([], dtype=torch.long, device=device)
            )
            self.register_buffer(
                "ring_members",
                torch.tensor([], dtype=torch.long, device=device).reshape(0, 0)
            )
            self.register_buffer(
                "ring_sizes",
                torch.tensor([], dtype=torch.long, device=device)
            )

    def _merge_overlapping_rings(self, rings: list) -> list:
        """
        Merge overlapping rings into unified ring systems using union-find.

        This handles fused ring systems like:
        - Indole in Tryptophan (5-membered + 6-membered fused rings)
        - Purine bases (adenine, guanine)
        - Naphthalene-like systems

        Parameters
        ----------
        rings : list of set
            List of sets, each containing atom indices for a detected ring.

        Returns
        -------
        list of set
            List of merged ring systems, where overlapping rings are combined.
        """
        if len(rings) == 0:
            return []

        n_rings = len(rings)

        # Union-find data structure
        parent = list(range(n_rings))
        rank = [0] * n_rings

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            # Union by rank
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Merge rings that share any atoms
        for i in range(n_rings):
            for j in range(i + 1, n_rings):
                # If rings share any atoms, merge them
                if rings[i] & rings[j]:  # Set intersection
                    union(i, j)

        # Group rings by their root in union-find
        groups = defaultdict(list)
        for i in range(n_rings):
            groups[find(i)].append(i)

        # Create merged ring systems
        ring_systems = []
        for ring_indices in groups.values():
            # Merge all atoms from rings in this group
            merged_atoms = set()
            for ring_idx in ring_indices:
                merged_atoms |= rings[ring_idx]
            ring_systems.append(merged_atoms)

        return ring_systems

    def _extract_internal_coords(
        self, xyz: torch.Tensor, requires_grad: bool = True
    ) -> None:
        """
        Extract internal coordinates from Cartesian coordinates (vectorized).

        Parameters
        ----------
        xyz : torch.Tensor
            Atomic coordinates of shape (N, 3).
        requires_grad : bool
            Whether parameters should have gradients.
        """
        device = self._device
        dtype = self._dtype

        # Extract bond lengths (vectorized)
        bond_mask = self.atom_depth >= 1
        if bond_mask.any():
            child_xyz = xyz[bond_mask]
            parent_xyz = xyz[self.parent_indices[bond_mask]]
            bond_lengths = torch.linalg.norm(child_xyz - parent_xyz, dim=-1)
        else:
            bond_lengths = torch.tensor([], dtype=dtype, device=device)

        # Extract angles (vectorized)
        angle_mask = self.atom_depth >= 2
        if angle_mask.any():
            child_xyz = xyz[angle_mask]
            parent_xyz = xyz[self.parent_indices[angle_mask]]
            grandparent_xyz = xyz[self.grandparent_indices[angle_mask]]

            v1 = child_xyz - parent_xyz
            v2 = grandparent_xyz - parent_xyz
            cos_angles = torch.sum(v1 * v2, dim=-1) / (
                torch.linalg.norm(v1, dim=-1) * torch.linalg.norm(v2, dim=-1) + 1e-10
            )
            cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
            angles = torch.acos(cos_angles)
        else:
            angles = torch.tensor([], dtype=dtype, device=device)

        # Extract torsions (vectorized)
        torsion_mask = self.atom_depth >= 3
        if torsion_mask.any():
            child_xyz = xyz[torsion_mask]
            parent_xyz = xyz[self.parent_indices[torsion_mask]]
            grandparent_xyz = xyz[self.grandparent_indices[torsion_mask]]
            great_grandparent_xyz = xyz[self.great_grandparent_indices[torsion_mask]]

            torsions = self._compute_torsion_angles(
                great_grandparent_xyz, grandparent_xyz, parent_xyz, child_xyz
            )
        else:
            torsions = torch.tensor([], dtype=dtype, device=device)

        # Extract segment positions
        segment_positions = xyz[self.segment_roots].clone()

        # Extract secondary root positions (disconnected component roots)
        if self.secondary_root_indices.numel() > 0:
            secondary_root_positions = xyz[self.secondary_root_indices].clone()
        else:
            secondary_root_positions = torch.zeros(0, 3, dtype=dtype, device=device)

        # Initialize segment orientations to zero (identity rotation)
        segment_orientations = torch.zeros(
            self.n_segments, 3, dtype=dtype, device=device
        )

        # Extract ring local coordinates (vectorized where possible)
        if self.n_rings > 0:
            ring_local_coords = self._extract_ring_local_coords_vectorized(xyz)
            self.register_buffer("ring_local_coords", ring_local_coords)
        else:
            self.register_buffer(
                "ring_local_coords",
                torch.tensor([], dtype=dtype, device=device).reshape(0, 0, 3)
            )

        # Setup shallow atom references (vectorized)
        self._setup_shallow_atom_references_vectorized(xyz)

        # Register parameters
        self.bond_lengths = nn.Parameter(
            bond_lengths.clone(), requires_grad=requires_grad
        )
        self.angles = nn.Parameter(angles.clone(), requires_grad=requires_grad)
        self.torsions = nn.Parameter(torsions.clone(), requires_grad=requires_grad)
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
            torch.ones(self.n_atoms, dtype=torch.bool, device=device)
        )
        self.register_buffer("fixed_xyz", xyz.clone())

    def _setup_shallow_atom_references_vectorized(self, xyz: torch.Tensor) -> None:
        """
        Store reference directions for depth-1 and depth-2 atoms (vectorized).

        Parameters
        ----------
        xyz : torch.Tensor
            Atomic coordinates.
        """
        device = self._device
        dtype = self._dtype

        # Depth-1 atoms (vectorized)
        depth1_mask = self.atom_depth == 1
        n_depth1 = depth1_mask.sum().item()

        if n_depth1 > 0:
            depth1_indices = torch.where(depth1_mask)[0]

            # Vectorized direction computation
            parent_idx = self.parent_indices[depth1_indices]
            directions = xyz[depth1_indices] - xyz[parent_idx]
            norms = torch.linalg.norm(directions, dim=-1, keepdim=True).clamp(min=1e-10)
            depth1_dirs = directions / norms

            # Create index mapping
            depth1_atom_to_dir_idx = torch.full(
                (self.n_atoms,), -1, dtype=torch.long, device=device
            )
            depth1_atom_to_dir_idx[depth1_indices] = torch.arange(
                n_depth1, dtype=torch.long, device=device
            )

            self.register_buffer("depth1_dirs", depth1_dirs)
            self.register_buffer("depth1_atom_to_dir_idx", depth1_atom_to_dir_idx)
        else:
            self.register_buffer(
                "depth1_dirs", torch.zeros(0, 3, dtype=dtype, device=device)
            )
            self.register_buffer(
                "depth1_atom_to_dir_idx",
                torch.full((self.n_atoms,), -1, dtype=torch.long, device=device)
            )

        # Depth-2 atoms (vectorized)
        depth2_mask = self.atom_depth == 2
        n_depth2 = depth2_mask.sum().item()

        if n_depth2 > 0:
            depth2_indices = torch.where(depth2_mask)[0]

            parent_idx = self.parent_indices[depth2_indices]
            grandparent_idx = self.grandparent_indices[depth2_indices]

            # Bond direction (vectorized)
            v_bond = xyz[parent_idx] - xyz[grandparent_idx]
            v_bond_norm = torch.linalg.norm(v_bond, dim=-1, keepdim=True).clamp(min=1e-10)
            v_bond = v_bond / v_bond_norm

            # Child direction
            v_child = xyz[depth2_indices] - xyz[parent_idx]

            # Perpendicular component (vectorized)
            proj = torch.sum(v_child * v_bond, dim=-1, keepdim=True)
            v_perp = v_child - proj * v_bond
            v_perp_norm = torch.linalg.norm(v_perp, dim=-1, keepdim=True)

            # Handle collinear cases
            collinear = v_perp_norm.squeeze(-1) < 1e-10

            if collinear.any():
                # For collinear atoms, use arbitrary perpendicular
                default_perp = torch.zeros_like(v_perp)
                nearly_x = torch.abs(v_bond[:, 0]) >= 0.9
                default_perp[~nearly_x, 0] = 1.0
                default_perp[nearly_x, 1] = 1.0

                # Make perpendicular
                proj_default = torch.sum(default_perp * v_bond, dim=-1, keepdim=True)
                default_perp = default_perp - proj_default * v_bond
                default_perp = default_perp / torch.linalg.norm(
                    default_perp, dim=-1, keepdim=True
                ).clamp(min=1e-10)

                v_perp[collinear] = default_perp[collinear]
                v_perp_norm[collinear] = 1.0

            depth2_perps = v_perp / v_perp_norm.clamp(min=1e-10)

            # Create index mapping
            depth2_atom_to_perp_idx = torch.full(
                (self.n_atoms,), -1, dtype=torch.long, device=device
            )
            depth2_atom_to_perp_idx[depth2_indices] = torch.arange(
                n_depth2, dtype=torch.long, device=device
            )

            self.register_buffer("depth2_perps", depth2_perps)
            self.register_buffer("depth2_atom_to_perp_idx", depth2_atom_to_perp_idx)
        else:
            self.register_buffer(
                "depth2_perps", torch.zeros(0, 3, dtype=dtype, device=device)
            )
            self.register_buffer(
                "depth2_atom_to_perp_idx",
                torch.full((self.n_atoms,), -1, dtype=torch.long, device=device)
            )

    def _extract_ring_local_coords_vectorized(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Extract local coordinates for ring atoms (fully vectorized).

        Parameters
        ----------
        xyz : torch.Tensor
            Atomic coordinates.

        Returns
        -------
        torch.Tensor
            Ring local coordinates of shape (n_rings, max_ring_size, 3).
        """
        device = self._device
        dtype = self._dtype
        max_ring_size = self.ring_members.shape[1] if self.n_rings > 0 else 0

        ring_local_coords = torch.zeros(
            self.n_rings, max_ring_size, 3, dtype=dtype, device=device
        )

        if self.n_rings == 0:
            return ring_local_coords

        # Get anchor positions
        anchor_pos = xyz[self.ring_anchor_atoms]  # (n_rings, 3)
        parent_idx = self.parent_indices[self.ring_anchor_atoms]

        # Compute z-axis for all rings (vectorized)
        valid_parent = parent_idx >= 0
        parent_pos = torch.zeros_like(anchor_pos)
        parent_pos[valid_parent] = xyz[parent_idx[valid_parent]]

        z_axis = anchor_pos - parent_pos
        z_norm = torch.linalg.norm(z_axis, dim=-1, keepdim=True)
        z_axis = torch.where(
            z_norm > 1e-10,
            z_axis / z_norm.clamp(min=1e-10),
            torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        )
        z_axis[~valid_parent] = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

        # Compute perpendicular axes (vectorized)
        x_base = torch.zeros_like(z_axis)
        x_base[:, 0] = 1.0
        nearly_x = torch.abs(z_axis[:, 0]) >= 0.9
        x_base[nearly_x, 0] = 0.0
        x_base[nearly_x, 1] = 1.0

        x_axis = x_base - (x_base * z_axis).sum(dim=-1, keepdim=True) * z_axis
        x_axis = x_axis / torch.linalg.norm(x_axis, dim=-1, keepdim=True).clamp(min=1e-10)
        y_axis = torch.linalg.cross(z_axis, x_axis)

        # Build rotation matrices: R = [x, y, z] as columns
        R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (n_rings, 3, 3)

        # Fully vectorized ring member processing
        # Expand ring_members to get all atom positions at once
        # Use -1 padding to create valid mask
        valid_mask = self.ring_members >= 0  # (n_rings, max_ring_size)

        # Replace -1 with 0 for indexing (will be masked out)
        safe_members = self.ring_members.clamp(min=0)

        # Get all member positions: (n_rings, max_ring_size, 3)
        member_pos = xyz[safe_members]

        # Compute offsets from anchor: (n_rings, max_ring_size, 3)
        offsets = member_pos - anchor_pos.unsqueeze(1)

        # Transform to local coords using batched matmul
        # offsets: (n_rings, max_ring_size, 3)
        # R: (n_rings, 3, 3)
        # Result: (n_rings, max_ring_size, 3)
        ring_local_coords = torch.bmm(
            offsets,
            R  # (n_rings, 3, 3)
        )

        # Zero out invalid entries
        ring_local_coords = ring_local_coords * valid_mask.unsqueeze(-1)

        return ring_local_coords

    @staticmethod
    def _compute_torsion_angles(
        p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute torsion angles for batched positions (vectorized).

        Parameters
        ----------
        p1, p2, p3, p4 : torch.Tensor
            Atom positions of shape (N, 3).

        Returns
        -------
        torch.Tensor
            Torsion angles in radians, shape (N,).
        """
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3

        n1 = torch.linalg.cross(b1, b2)
        n2 = torch.linalg.cross(b2, b3)

        n1 = n1 / torch.linalg.norm(n1, dim=-1, keepdim=True).clamp(min=1e-10)
        n2 = n2 / torch.linalg.norm(n2, dim=-1, keepdim=True).clamp(min=1e-10)

        b2_norm = b2 / torch.linalg.norm(b2, dim=-1, keepdim=True).clamp(min=1e-10)
        m1 = torch.linalg.cross(n1, b2_norm)

        x = torch.sum(n1 * n2, dim=-1)
        y = torch.sum(m1 * n2, dim=-1)

        return torch.atan2(y, x)

    def forward(self) -> torch.Tensor:
        """
        Reconstruct Cartesian xyz from internal coordinates.

        Uses fully vectorized operations for maximum performance.

        Returns
        -------
        torch.Tensor
            Reconstructed Cartesian coordinates of shape (N, 3).
        """
        xyz = torch.zeros(
            self.n_atoms, 3, dtype=self._dtype, device=self._device
        )

        # Get rotation matrices for all segments (batched)
        R_matrices = rotation_matrix_euler_zyz(
            self.segment_orientations
        )  # (n_segments, 3, 3)

        # Place segment root atoms (depth=0)
        xyz[self.segment_roots] = self.segment_positions

        # Place secondary roots (disconnected component roots)
        if self.secondary_root_indices.numel() > 0:
            xyz[self.secondary_root_indices] = self.secondary_root_positions

        # Place depth-1 atoms (vectorized)
        xyz = self._place_depth1_atoms(xyz, R_matrices)

        # Place depth-2 atoms (vectorized)
        xyz = self._place_depth2_atoms(xyz, R_matrices)

        # Place depth-3+ atoms using NeRF (vectorized per depth)
        for depth_idx in range(len(self._depth_atom_indices)):
            xyz = self._place_atoms_at_depth_fast(xyz, depth_idx)

        # Place rigid ring atoms (vectorized)
        xyz = self._place_rigid_rings(xyz)

        # Apply frozen coordinates
        if not self.refinable_mask.all():
            frozen_mask = ~self.refinable_mask
            xyz[frozen_mask] = self.fixed_xyz[frozen_mask]

        return xyz

    def _place_depth1_atoms(
        self, xyz: torch.Tensor, R_matrices: torch.Tensor
    ) -> torch.Tensor:
        """
        Place atoms at depth 1 (vectorized).

        Parameters
        ----------
        xyz : torch.Tensor
            Current coordinates.
        R_matrices : torch.Tensor
            Segment rotation matrices of shape (n_segments, 3, 3).

        Returns
        -------
        torch.Tensor
            Updated coordinates.
        """
        mask = self.atom_depth == 1
        if not mask.any():
            return xyz

        xyz = xyz.clone()

        atom_indices = torch.where(mask)[0]
        seg_ids = self.atom_to_segment[atom_indices]

        # Get actual parent positions from xyz (handles both primary and secondary roots)
        parent_idx = self.parent_indices[atom_indices]
        parent_positions = xyz[parent_idx]

        # Get rotation matrices based on segment
        R = R_matrices[seg_ids]

        # Get stored directions
        dir_indices = self.depth1_atom_to_dir_idx[atom_indices]
        base_dirs = self.depth1_dirs[dir_indices]

        # Rotate directions (batched matmul)
        rotated_dirs = torch.bmm(R, base_dirs.unsqueeze(-1)).squeeze(-1)

        # Get bond lengths
        bond_idx = self.bond_param_indices[mask]
        bond_lengths = self.bond_lengths[bond_idx]

        # Compute positions relative to actual parent (vectorized)
        new_positions = parent_positions + bond_lengths.unsqueeze(-1) * rotated_dirs
        xyz[mask] = new_positions

        return xyz

    def _place_depth2_atoms(
        self, xyz: torch.Tensor, R_matrices: torch.Tensor
    ) -> torch.Tensor:
        """
        Place atoms at depth 2 (vectorized).

        Parameters
        ----------
        xyz : torch.Tensor
            Current coordinates.
        R_matrices : torch.Tensor
            Segment rotation matrices.

        Returns
        -------
        torch.Tensor
            Updated coordinates.
        """
        mask = self.atom_depth == 2
        if not mask.any():
            return xyz

        xyz = xyz.clone()

        # Get positions (vectorized gather)
        parent_idx = self.parent_indices[mask]
        grandparent_idx = self.grandparent_indices[mask]

        parent_pos = xyz[parent_idx]
        grandparent_pos = xyz[grandparent_idx]

        # Get parameters
        bond_idx = self.bond_param_indices[mask]
        angle_idx = self.angle_param_indices[mask]

        bond_lengths = self.bond_lengths[bond_idx]
        angles = self.angles[angle_idx]

        # Build local frame (vectorized)
        bc = parent_pos - grandparent_pos
        bc_norm = torch.linalg.norm(bc, dim=-1, keepdim=True).clamp(min=1e-10)
        bc_unit = bc / bc_norm

        # Get perpendicular references
        atom_indices = torch.where(mask)[0]
        perp_idx = self.depth2_atom_to_perp_idx[atom_indices]
        ref_perp = self.depth2_perps[perp_idx]

        # Apply segment rotation (batched)
        seg_ids = self.atom_to_segment[atom_indices]
        R = R_matrices[seg_ids]
        rotated_ref = torch.bmm(R, ref_perp.unsqueeze(-1)).squeeze(-1)

        # Make perpendicular to bc_unit (vectorized)
        proj = torch.sum(rotated_ref * bc_unit, dim=-1, keepdim=True)
        rotated_ref = rotated_ref - proj * bc_unit
        ref_norm = torch.linalg.norm(rotated_ref, dim=-1, keepdim=True).clamp(min=1e-10)
        rotated_ref = rotated_ref / ref_norm

        # Compute position (vectorized)
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
        """
        Place atoms at a depth level using pre-computed indices (vectorized).

        Parameters
        ----------
        xyz : torch.Tensor
            Current coordinates.
        depth_idx : int
            Index into pre-computed depth arrays.

        Returns
        -------
        torch.Tensor
            Updated coordinates.
        """
        atom_idx = self._depth_atom_indices[depth_idx]

        if atom_idx.numel() == 0:
            return xyz

        xyz = xyz.clone()

        # Use pre-computed indices (vectorized gather)
        parent_idx = self._depth_parent_indices[depth_idx]
        gp_idx = self._depth_gp_indices[depth_idx]
        ggp_idx = self._depth_ggp_indices[depth_idx]

        p1 = xyz[ggp_idx]
        p2 = xyz[gp_idx]
        p3 = xyz[parent_idx]

        # Get parameters
        bond_idx = self._depth_bond_param_indices[depth_idx]
        angle_idx = self._depth_angle_param_indices[depth_idx]
        torsion_idx = self._depth_torsion_param_indices[depth_idx]

        d = self.bond_lengths[bond_idx]
        theta = self.angles[angle_idx]
        phi = self.torsions[torsion_idx]

        # Build local coordinate frame (vectorized)
        bc = p3 - p2
        bc = bc / torch.linalg.norm(bc, dim=-1, keepdim=True).clamp(min=1e-10)

        ab = p2 - p1
        n = torch.linalg.cross(ab, bc)
        n = n / torch.linalg.norm(n, dim=-1, keepdim=True).clamp(min=1e-10)

        m = torch.linalg.cross(n, bc)

        # Compute positions (vectorized)
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
        """
        Place ring atoms as rigid groups (vectorized).

        Parameters
        ----------
        xyz : torch.Tensor
            Current coordinates.

        Returns
        -------
        torch.Tensor
            Updated coordinates with ring atoms placed.
        """
        if self.n_rings == 0:
            return xyz

        # Get anchor positions (vectorized)
        anchor_pos = xyz[self.ring_anchor_atoms]
        parent_idx = self.parent_indices[self.ring_anchor_atoms]

        valid_parent_mask = parent_idx >= 0
        parent_pos = torch.zeros_like(anchor_pos)
        parent_pos[valid_parent_mask] = xyz[parent_idx[valid_parent_mask]]

        # Compute z-axis (vectorized)
        z_axis = anchor_pos - parent_pos
        z_norm = torch.linalg.norm(z_axis, dim=-1, keepdim=True)
        z_axis = torch.where(
            z_norm > 1e-10,
            z_axis / z_norm.clamp(min=1e-10),
            torch.tensor([0.0, 0.0, 1.0], dtype=self._dtype, device=self._device)
        )
        z_axis[~valid_parent_mask] = torch.tensor(
            [0.0, 0.0, 1.0], dtype=self._dtype, device=self._device
        )

        # Compute perpendicular axes (vectorized)
        x_base = torch.zeros_like(z_axis)
        x_base[:, 0] = 1.0
        nearly_x = torch.abs(z_axis[:, 0]) >= 0.9
        x_base[nearly_x, 0] = 0.0
        x_base[nearly_x, 1] = 1.0

        x_axis = x_base - (x_base * z_axis).sum(dim=-1, keepdim=True) * z_axis
        x_axis = x_axis / torch.linalg.norm(x_axis, dim=-1, keepdim=True).clamp(min=1e-10)
        y_axis = torch.linalg.cross(z_axis, x_axis)

        R = torch.stack([x_axis, y_axis, z_axis], dim=-1)

        # Transform local coords to global (batched matmul)
        global_offsets = torch.bmm(
            self.ring_local_coords,
            R.transpose(-1, -2)
        )
        global_pos = global_offsets + anchor_pos.unsqueeze(1)

        # Scatter results (vectorized)
        valid_mask = self.ring_members >= 0
        anchor_expanded = self.ring_anchor_atoms.unsqueeze(1)
        valid_mask = valid_mask & (self.ring_members != anchor_expanded)

        ring_indices, local_indices = torch.where(valid_mask)
        atom_indices = self.ring_members[ring_indices, local_indices]
        positions = global_pos[ring_indices, local_indices]

        xyz[atom_indices] = positions
        return xyz

    def shake(self, magnitude: float = 0.1) -> torch.Tensor:
        """
        Add Gaussian noise to internal parameters.

        Parameters
        ----------
        magnitude : float, optional
            Standard deviation of Gaussian noise. Default is 0.1.

        Returns
        -------
        torch.Tensor
            New Cartesian coordinates after perturbation.
        """
        with torch.no_grad():
            if self.bond_lengths.numel() > 0:
                self.bond_lengths.data += torch.randn_like(self.bond_lengths) * magnitude
                self.bond_lengths.data = torch.clamp(self.bond_lengths.data, min=0.5)

            if self.angles.numel() > 0:
                self.angles.data += torch.randn_like(self.angles) * magnitude
                self.angles.data = torch.clamp(
                    self.angles.data, min=0.1, max=torch.pi - 0.1
                )

            if self.torsions.numel() > 0:
                self.torsions.data += torch.randn_like(self.torsions) * magnitude
                self.torsions.data = torch.atan2(
                    torch.sin(self.torsions.data), torch.cos(self.torsions.data)
                )

            if self.segment_positions.numel() > 0:
                self.segment_positions.data += (
                    torch.randn_like(self.segment_positions) * magnitude
                )

            if self.secondary_root_positions.numel() > 0:
                self.secondary_root_positions.data += (
                    torch.randn_like(self.secondary_root_positions) * magnitude
                )

        return self.forward()

    # ===== Freeze/Unfreeze Methods =====

    def fix(
        self,
        selection: Union[torch.Tensor, slice, None] = None,
        freeze_at_current: bool = True
    ) -> None:
        """
        Fix (freeze) atoms to use fixed xyz coordinates.

        Parameters
        ----------
        selection : torch.Tensor, slice, or None
            Boolean mask or indices of atoms to fix.
        freeze_at_current : bool, optional
            If True, store current coordinates for selected atoms.
        """
        if selection is None:
            selection = slice(None)

        if isinstance(selection, torch.Tensor) and selection.dtype == torch.bool:
            mask = selection
        elif isinstance(selection, torch.Tensor):
            mask = torch.zeros(self.n_atoms, dtype=torch.bool, device=self._device)
            mask[selection] = True
        elif isinstance(selection, slice):
            mask = torch.zeros(self.n_atoms, dtype=torch.bool, device=self._device)
            mask[selection] = True
        else:
            raise TypeError(
                f"selection must be Tensor, slice, or None, got {type(selection)}"
            )

        if freeze_at_current:
            with torch.no_grad():
                current_xyz = self.forward()
                self.fixed_xyz[mask] = current_xyz[mask]

        self.refinable_mask[mask] = False

    def freeze(
        self,
        selection: Union[torch.Tensor, slice, None] = None,
        freeze_at_current: bool = True
    ) -> None:
        """Alias for fix()."""
        self.fix(selection, freeze_at_current)

    def refine(
        self,
        selection: Union[torch.Tensor, slice, None] = None,
        rebuild: bool = True
    ) -> None:
        """
        Make atoms refinable.

        Parameters
        ----------
        selection : torch.Tensor, slice, or None
            Boolean mask or indices of atoms to make refinable.
        rebuild : bool, optional
            If True, rebuild internal coordinates from fixed_xyz.
        """
        if selection is None:
            selection = slice(None)

        if isinstance(selection, torch.Tensor) and selection.dtype == torch.bool:
            mask = selection
        elif isinstance(selection, torch.Tensor):
            mask = torch.zeros(self.n_atoms, dtype=torch.bool, device=self._device)
            mask[selection] = True
        elif isinstance(selection, slice):
            mask = torch.zeros(self.n_atoms, dtype=torch.bool, device=self._device)
            mask[selection] = True
        else:
            raise TypeError(
                f"selection must be Tensor, slice, or None, got {type(selection)}"
            )

        if rebuild:
            self._update_internal_coords_from_xyz(self.fixed_xyz, mask)

        self.refinable_mask[mask] = True

    def unfreeze(
        self,
        selection: Union[torch.Tensor, slice, None] = None,
        rebuild: bool = True
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

    def _update_internal_coords_from_xyz(
        self, xyz: torch.Tensor, mask: torch.Tensor
    ) -> None:
        """
        Update internal coordinate parameters from xyz (vectorized).

        Parameters
        ----------
        xyz : torch.Tensor
            Atomic coordinates.
        mask : torch.Tensor
            Boolean mask indicating which atoms to update.
        """
        # Update bond lengths (vectorized)
        bond_update_mask = mask & (self.atom_depth >= 1)
        if bond_update_mask.any():
            child_xyz = xyz[bond_update_mask]
            parent_xyz = xyz[self.parent_indices[bond_update_mask]]
            new_bonds = torch.linalg.norm(child_xyz - parent_xyz, dim=-1)
            bond_idx = self.bond_param_indices[bond_update_mask]
            with torch.no_grad():
                self.bond_lengths.data[bond_idx] = new_bonds

        # Update angles (vectorized)
        angle_update_mask = mask & (self.atom_depth >= 2)
        if angle_update_mask.any():
            child_xyz = xyz[angle_update_mask]
            parent_xyz = xyz[self.parent_indices[angle_update_mask]]
            grandparent_xyz = xyz[self.grandparent_indices[angle_update_mask]]

            v1 = child_xyz - parent_xyz
            v2 = grandparent_xyz - parent_xyz
            cos_angles = torch.sum(v1 * v2, dim=-1) / (
                torch.linalg.norm(v1, dim=-1) * torch.linalg.norm(v2, dim=-1) + 1e-10
            )
            cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
            new_angles = torch.acos(cos_angles)
            angle_idx = self.angle_param_indices[angle_update_mask]
            with torch.no_grad():
                self.angles.data[angle_idx] = new_angles

        # Update torsions (vectorized)
        torsion_update_mask = mask & (self.atom_depth >= 3)
        if torsion_update_mask.any():
            child_xyz = xyz[torsion_update_mask]
            parent_xyz = xyz[self.parent_indices[torsion_update_mask]]
            grandparent_xyz = xyz[self.grandparent_indices[torsion_update_mask]]
            great_grandparent_xyz = xyz[
                self.great_grandparent_indices[torsion_update_mask]
            ]

            new_torsions = self._compute_torsion_angles(
                great_grandparent_xyz, grandparent_xyz, parent_xyz, child_xyz
            )
            torsion_idx = self.torsion_param_indices[torsion_update_mask]
            with torch.no_grad():
                self.torsions.data[torsion_idx] = new_torsions

        # Update segment positions (vectorized where possible)
        root_in_mask = mask[self.segment_roots]
        if root_in_mask.any():
            with torch.no_grad():
                self.segment_positions.data[root_in_mask] = xyz[
                    self.segment_roots[root_in_mask]
                ]

        # Update secondary root positions
        if self.secondary_root_indices.numel() > 0:
            sec_root_in_mask = mask[self.secondary_root_indices]
            if sec_root_in_mask.any():
                with torch.no_grad():
                    self.secondary_root_positions.data[sec_root_in_mask] = xyz[
                        self.secondary_root_indices[sec_root_in_mask]
                    ]

    @property
    def n_refinable(self) -> int:
        """Return the number of refinable atoms."""
        return self.refinable_mask.sum().item()

    @property
    def n_fixed(self) -> int:
        """Return the number of fixed atoms."""
        return (~self.refinable_mask).sum().item()

    def to(self, device=None, dtype=None):
        """
        Move tensor to specified device and/or dtype.

        Parameters
        ----------
        device : torch.device, optional
            Target device.
        dtype : torch.dtype, optional
            Target dtype.

        Returns
        -------
        SegmentedInternalCoordinateTensor
            Self for method chaining.
        """
        if device is not None:
            self._device = torch.device(device)
        if dtype is not None:
            self._dtype = dtype
        return super().to(device=device, dtype=dtype)

    def __repr__(self) -> str:
        n_secondary = self.secondary_root_indices.numel()
        return (
            f"SegmentedInternalCoordinateTensor(n_atoms={self.n_atoms}, "
            f"n_segments={self.n_segments}, n_aa_per_segment={self.n_aa_per_segment}, "
            f"n_secondary_roots={n_secondary}, "
            f"n_refinable={self.n_refinable}, n_fixed={self.n_fixed}, "
            f"n_bonds={self.n_bonds}, n_angles={self.n_angles}, "
            f"n_torsions={self.n_torsions}, n_rings={self.n_rings}, "
            f"max_depth={self.max_depth}, device={self._device})"
        )
