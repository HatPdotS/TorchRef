"""
Internal coordinate parametrization for atomic structures.

This module provides the InternalCoordinateTensor class which parametrizes
atomic XYZ coordinates using internal coordinates (bond lengths, angles,
torsions) instead of Cartesian coordinates. This enables physically meaningful
perturbations and differentiable reconstruction.

Key features:
- Bond detection: Atoms within 2Å are considered bonded (using torch.cdist)
- Internal coordinate parametrization: N atoms → N-1 bonds, N-2 angles, N-3 torsions per chain
- Chain handling: Unconnected chains treated as rigid groups with position/orientation
- Fully differentiable: Complete gradient flow from internal params → Cartesian coords
- Fully vectorized: No Python loops over atoms - all operations via tensor ops
- Ring handling: Rings are treated as rigid entities with only the anchor movable
"""

from typing import Optional, Union

import torch
import torch.nn as nn

from torchref.base.alignment.rotation import axis_angle_to_rotation_matrix
from torchref.utils.device_mixin import DeviceMixin


class InternalCoordinateTensor(DeviceMixin, nn.Module):
    """
    Parameter wrapper using internal coordinates (Z-matrix style).

    Stores: bond_lengths, angles, torsions, chain_positions, chain_orientations
    Reconstructs: Cartesian xyz on forward()

    This provides a physically meaningful parametrization of atomic coordinates
    where perturbations correspond to changes in bond lengths, angles, and
    torsion angles rather than arbitrary Cartesian displacements.

    Parameters
    ----------
    initial_xyz : torch.Tensor
        Initial Cartesian coordinates of shape (N, 3).
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
    n_chains : int
        Number of disconnected chains.
    max_depth : int
        Maximum depth in the spanning tree.
    bond_lengths : nn.Parameter
        Bond length parameters in Angstroms.
    angles : nn.Parameter
        Angle parameters in radians.
    torsions : nn.Parameter
        Torsion angle parameters in radians.
    chain_positions : nn.Parameter
        Absolute positions of chain root atoms.
    chain_orientations : nn.Parameter
        Axis-angle orientations for each chain.
    """

    def __init__(
        self,
        initial_xyz: torch.Tensor,
        bond_cutoff: float = 2.0,
        requires_grad: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize InternalCoordinateTensor from Cartesian coordinates.

        Parameters
        ----------
        initial_xyz : torch.Tensor
            Initial Cartesian coordinates of shape (N, 3).
        bond_cutoff : float, optional
            Distance cutoff for bond detection in Angstroms. Default is 2.0.
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

        # Move initial coordinates to specified device/dtype
        initial_xyz = initial_xyz.to(dtype=dtype, device=device)

        # Build molecular graph from distances
        adjacency = self._build_molecular_graph(initial_xyz, bond_cutoff)

        # Build spanning trees for each connected component
        self._build_spanning_trees(adjacency)

        # Detect rings and setup rigid ring handling
        self._detect_rings(adjacency)

        # Extract internal coordinates from initial xyz
        self._extract_internal_coords(initial_xyz, requires_grad)

    @property
    def dtype(self):
        """Return the dtype of tensors."""
        return self._dtype

    @property
    def device(self):
        """Return the device of tensors."""
        return self._device

    @staticmethod
    def _build_molecular_graph(
        xyz: torch.Tensor, cutoff: float = 2.0
    ) -> torch.Tensor:
        """
        Build adjacency matrix from atomic distances.

        Parameters
        ----------
        xyz : torch.Tensor
            Atomic coordinates of shape (N, 3).
        cutoff : float
            Distance cutoff for bonds in Angstroms.

        Returns
        -------
        torch.Tensor
            Boolean adjacency matrix of shape (N, N).
        """
        # Compute pairwise distances
        distances = torch.cdist(xyz, xyz)

        # Atoms within cutoff are bonded (exclude self-connections)
        adjacency = (distances < cutoff) & (distances > 0.1)

        return adjacency

    def _build_spanning_trees(self, adjacency: torch.Tensor) -> None:
        """
        Build spanning tree(s) via BFS and populate index buffers.

        For each connected component, builds a spanning tree and records
        parent/grandparent/great-grandparent indices for each atom.

        Parameters
        ----------
        adjacency : torch.Tensor
            Boolean adjacency matrix of shape (N, N).
        """
        n_atoms = self.n_atoms
        device = self._device

        # Initialize arrays
        parent_indices = torch.full((n_atoms,), -1, dtype=torch.long, device=device)
        grandparent_indices = torch.full(
            (n_atoms,), -1, dtype=torch.long, device=device
        )
        great_grandparent_indices = torch.full(
            (n_atoms,), -1, dtype=torch.long, device=device
        )
        atom_depth = torch.full((n_atoms,), -1, dtype=torch.long, device=device)
        chain_ids = torch.full((n_atoms,), -1, dtype=torch.long, device=device)

        chain_roots = []
        visited = torch.zeros(n_atoms, dtype=torch.bool, device=device)

        chain_id = 0

        # Process each connected component
        for start_atom in range(n_atoms):
            if visited[start_atom]:
                continue

            # BFS from this atom
            chain_roots.append(start_atom)
            queue = [start_atom]
            visited[start_atom] = True
            parent_indices[start_atom] = -1
            atom_depth[start_atom] = 0
            chain_ids[start_atom] = chain_id

            queue_idx = 0
            while queue_idx < len(queue):
                current = queue[queue_idx]
                queue_idx += 1

                # Find neighbors
                neighbors = torch.where(adjacency[current])[0]

                for neighbor in neighbors:
                    neighbor = neighbor.item()
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        parent_indices[neighbor] = current
                        atom_depth[neighbor] = atom_depth[current] + 1
                        chain_ids[neighbor] = chain_id

                        # Set grandparent
                        if parent_indices[current] >= 0:
                            grandparent_indices[neighbor] = parent_indices[current]

                        # Set great-grandparent
                        if grandparent_indices[current] >= 0:
                            great_grandparent_indices[neighbor] = grandparent_indices[
                                current
                            ]

                        queue.append(neighbor)

            chain_id += 1

        # Register buffers
        self.register_buffer("parent_indices", parent_indices)
        self.register_buffer("grandparent_indices", grandparent_indices)
        self.register_buffer("great_grandparent_indices", great_grandparent_indices)
        self.register_buffer("atom_depth", atom_depth)
        self.register_buffer("chain_ids", chain_ids)
        self.register_buffer(
            "chain_roots", torch.tensor(chain_roots, dtype=torch.long, device=device)
        )

        self.n_chains = len(chain_roots)
        self.max_depth = atom_depth.max().item()

        # Build parameter index mappings
        self._build_param_indices()

        # Pre-compute depth indices for fast forward pass
        self._build_depth_indices()

    def _build_param_indices(self) -> None:
        """
        Build mappings from atoms to parameter indices.

        Creates bond_param_indices, angle_param_indices, and torsion_param_indices
        which map each atom to its corresponding internal coordinate parameter.
        """
        device = self._device
        n_atoms = self.n_atoms

        # Atoms with depth >= 1 have bonds
        # Atoms with depth >= 2 have angles
        # Atoms with depth >= 3 have torsions

        # Count parameters
        has_bond = self.atom_depth >= 1
        has_angle = self.atom_depth >= 2
        has_torsion = self.atom_depth >= 3

        n_bonds = has_bond.sum().item()
        n_angles = has_angle.sum().item()
        n_torsions = has_torsion.sum().item()

        # Create index mappings
        bond_param_indices = torch.full((n_atoms,), -1, dtype=torch.long, device=device)
        angle_param_indices = torch.full(
            (n_atoms,), -1, dtype=torch.long, device=device
        )
        torsion_param_indices = torch.full(
            (n_atoms,), -1, dtype=torch.long, device=device
        )

        # Assign indices in order of atom index
        bond_idx = 0
        angle_idx = 0
        torsion_idx = 0

        for i in range(n_atoms):
            if has_bond[i]:
                bond_param_indices[i] = bond_idx
                bond_idx += 1
            if has_angle[i]:
                angle_param_indices[i] = angle_idx
                angle_idx += 1
            if has_torsion[i]:
                torsion_param_indices[i] = torsion_idx
                torsion_idx += 1

        self.register_buffer("bond_param_indices", bond_param_indices)
        self.register_buffer("angle_param_indices", angle_param_indices)
        self.register_buffer("torsion_param_indices", torsion_param_indices)

        self.n_bonds = n_bonds
        self.n_angles = n_angles
        self.n_torsions = n_torsions

    def _build_depth_indices(self) -> None:
        """
        Pre-compute atom indices for each depth level.

        This avoids recomputing masks during the forward pass, which is
        the main performance bottleneck for large structures.
        """
        # Store indices for each depth level (depths 3+ use full NeRF)
        # We store as a list of tensors since depths have varying atom counts
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
                # Empty tensor for this depth
                empty = torch.tensor([], dtype=torch.long, device=self._device)
                depth_atom_indices.append(empty)
                depth_parent_indices.append(empty)
                depth_gp_indices.append(empty)
                depth_ggp_indices.append(empty)
                depth_bond_param_indices.append(empty)
                depth_angle_param_indices.append(empty)
                depth_torsion_param_indices.append(empty)

        # Store as instance attributes (not buffers since they're lists)
        self._depth_atom_indices = depth_atom_indices
        self._depth_parent_indices = depth_parent_indices
        self._depth_gp_indices = depth_gp_indices
        self._depth_ggp_indices = depth_ggp_indices
        self._depth_bond_param_indices = depth_bond_param_indices
        self._depth_angle_param_indices = depth_angle_param_indices
        self._depth_torsion_param_indices = depth_torsion_param_indices

    def _detect_rings(self, adjacency: torch.Tensor) -> None:
        """
        Detect rings in the molecular graph and set up rigid ring handling.

        Rings are identified by finding back-edges in the spanning tree.
        Ring atoms are marked for special handling during reconstruction.

        Parameters
        ----------
        adjacency : torch.Tensor
            Boolean adjacency matrix of shape (N, N).
        """
        device = self._device
        n_atoms = self.n_atoms

        # Initialize ring detection buffers
        ring_member_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        ring_group_id = torch.full((n_atoms,), -1, dtype=torch.long, device=device)

        # Find back edges (edges that are not in the spanning tree)
        # A back edge connects an atom to a non-parent ancestor
        back_edges = []

        for i in range(n_atoms):
            neighbors = torch.where(adjacency[i])[0]
            for j in neighbors:
                j = j.item()
                if j <= i:
                    continue  # Only check each edge once
                # Check if this edge is not in the spanning tree
                if self.parent_indices[i] != j and self.parent_indices[j] != i:
                    # This is a back edge - indicates a ring
                    back_edges.append((i, j))

        # For each back edge, find the ring members
        ring_anchors = []
        ring_members_list = []
        ring_idx = 0

        for edge_i, edge_j in back_edges:
            # Find path from i to j through the tree
            # The ring consists of atoms on both paths to their common ancestor

            # Find ancestors of i
            ancestors_i = set()
            current = edge_i
            while current >= 0:
                ancestors_i.add(current)
                current = self.parent_indices[current].item()

            # Find common ancestor with j
            ring_atoms = []
            current = edge_j
            while current >= 0:
                ring_atoms.append(current)
                if current in ancestors_i:
                    # Found common ancestor
                    common_ancestor = current
                    break
                current = self.parent_indices[current].item()

            # Add path from i to common ancestor
            current = edge_i
            while current != common_ancestor:
                ring_atoms.append(current)
                current = self.parent_indices[current].item()

            # Mark ring atoms
            for atom in ring_atoms:
                ring_member_mask[atom] = True
                if ring_group_id[atom] < 0:  # Only set if not already in a ring
                    ring_group_id[atom] = ring_idx

            # The anchor is the atom closest to root (smallest depth)
            depths = torch.tensor(
                [self.atom_depth[a].item() for a in ring_atoms], device=device
            )
            anchor_idx = ring_atoms[depths.argmin().item()]
            ring_anchors.append(anchor_idx)
            ring_members_list.append(ring_atoms)
            ring_idx += 1

        self.register_buffer("ring_member_mask", ring_member_mask)
        self.register_buffer("ring_group_id", ring_group_id)
        self.n_rings = len(ring_anchors)

        if self.n_rings > 0:
            self.register_buffer(
                "ring_anchor_atoms",
                torch.tensor(ring_anchors, dtype=torch.long, device=device),
            )
            # Store ring members as padded tensor
            max_ring_size = max(len(r) for r in ring_members_list)
            ring_members_tensor = torch.full(
                (self.n_rings, max_ring_size), -1, dtype=torch.long, device=device
            )
            ring_sizes = torch.zeros(self.n_rings, dtype=torch.long, device=device)
            for i, members in enumerate(ring_members_list):
                ring_sizes[i] = len(members)
                ring_members_tensor[i, : len(members)] = torch.tensor(
                    members, dtype=torch.long, device=device
                )
            self.register_buffer("ring_members", ring_members_tensor)
            self.register_buffer("ring_sizes", ring_sizes)
        else:
            self.register_buffer(
                "ring_anchor_atoms", torch.tensor([], dtype=torch.long, device=device)
            )
            self.register_buffer(
                "ring_members",
                torch.tensor([], dtype=torch.long, device=device).reshape(0, 0),
            )
            self.register_buffer(
                "ring_sizes", torch.tensor([], dtype=torch.long, device=device)
            )

    def _extract_internal_coords(
        self, xyz: torch.Tensor, requires_grad: bool = True
    ) -> None:
        """
        Extract internal coordinates from Cartesian coordinates.

        Computes bond lengths, angles, and torsions from the initial xyz
        and creates parameters for them.

        Parameters
        ----------
        xyz : torch.Tensor
            Atomic coordinates of shape (N, 3).
        requires_grad : bool
            Whether parameters should have gradients.
        """
        device = self._device
        dtype = self._dtype

        # Extract bond lengths for atoms with depth >= 1
        bond_mask = self.atom_depth >= 1
        if bond_mask.any():
            child_xyz = xyz[bond_mask]
            parent_xyz = xyz[self.parent_indices[bond_mask]]
            bond_lengths = torch.linalg.norm(child_xyz - parent_xyz, dim=-1)
        else:
            bond_lengths = torch.tensor([], dtype=dtype, device=device)

        # Extract angles for atoms with depth >= 2
        angle_mask = self.atom_depth >= 2
        if angle_mask.any():
            child_xyz = xyz[angle_mask]
            parent_xyz = xyz[self.parent_indices[angle_mask]]
            grandparent_xyz = xyz[self.grandparent_indices[angle_mask]]

            # Angle at parent between child and grandparent
            v1 = child_xyz - parent_xyz
            v2 = grandparent_xyz - parent_xyz
            cos_angles = torch.sum(v1 * v2, dim=-1) / (
                torch.linalg.norm(v1, dim=-1) * torch.linalg.norm(v2, dim=-1) + 1e-10
            )
            cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
            angles = torch.acos(cos_angles)
        else:
            angles = torch.tensor([], dtype=dtype, device=device)

        # Extract torsions for atoms with depth >= 3
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

        # Extract chain positions (positions of root atoms)
        chain_positions = xyz[self.chain_roots].clone()

        # Initialize chain orientations to zero (no rotation)
        chain_orientations = torch.zeros(self.n_chains, 3, dtype=dtype, device=device)

        # Extract ring local coordinates if there are rings
        if self.n_rings > 0:
            ring_local_coords = self._extract_ring_local_coords(xyz)
            self.register_buffer("ring_local_coords", ring_local_coords)
        else:
            self.register_buffer(
                "ring_local_coords",
                torch.tensor([], dtype=dtype, device=device).reshape(0, 0, 3),
            )

        # Store second atom directions for each chain (for placing second atoms)
        second_atom_dirs = []
        for chain_idx in range(self.n_chains):
            root = self.chain_roots[chain_idx].item()
            # Find atoms with depth 1 in this chain
            mask = (self.chain_ids == chain_idx) & (self.atom_depth == 1)
            if mask.any():
                first_child = torch.where(mask)[0][0].item()
                direction = xyz[first_child] - xyz[root]
                norm = torch.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                else:
                    direction = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
            else:
                direction = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
            second_atom_dirs.append(direction)

        self.register_buffer(
            "second_atom_dirs", torch.stack(second_atom_dirs) if second_atom_dirs else
            torch.zeros(0, 3, dtype=dtype, device=device)
        )

        # For depth-2 atoms, we need to store reference perpendiculars
        # These are computed for each individual depth-2 atom
        depth2_mask = self.atom_depth == 2
        n_depth2 = depth2_mask.sum().item()

        # Create mapping from atom index to depth2_perps index
        # -1 for atoms that aren't depth-2
        depth2_atom_to_perp_idx = torch.full(
            (self.n_atoms,), -1, dtype=torch.long, device=device
        )

        if n_depth2 > 0:
            depth2_perps = torch.zeros(n_depth2, 3, dtype=dtype, device=device)
            depth2_indices = torch.where(depth2_mask)[0]

            for i, atom_idx in enumerate(depth2_indices):
                atom_idx_val = atom_idx.item()
                depth2_atom_to_perp_idx[atom_idx_val] = i

                parent = self.parent_indices[atom_idx_val].item()
                grandparent = self.grandparent_indices[atom_idx_val].item()

                # Direction from grandparent to parent
                v_bond = xyz[parent] - xyz[grandparent]
                v_bond_norm = torch.linalg.norm(v_bond)
                if v_bond_norm > 1e-10:
                    v_bond = v_bond / v_bond_norm
                else:
                    v_bond = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)

                # Direction from parent to child
                v_child = xyz[atom_idx_val] - xyz[parent]

                # Get perpendicular component
                v_perp = v_child - torch.dot(v_child, v_bond) * v_bond
                v_perp_norm = torch.linalg.norm(v_perp)

                if v_perp_norm > 1e-10:
                    v_perp = v_perp / v_perp_norm
                else:
                    # Collinear case - use arbitrary perpendicular
                    if abs(v_bond[0]) < 0.9:
                        v_perp = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
                    else:
                        v_perp = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
                    v_perp = v_perp - torch.dot(v_perp, v_bond) * v_bond
                    v_perp = v_perp / torch.linalg.norm(v_perp)

                depth2_perps[i] = v_perp

            self.register_buffer("depth2_perps", depth2_perps)
        else:
            self.register_buffer(
                "depth2_perps", torch.zeros(0, 3, dtype=dtype, device=device)
            )

        self.register_buffer("depth2_atom_to_perp_idx", depth2_atom_to_perp_idx)

        # Register parameters
        self.bond_lengths = nn.Parameter(bond_lengths.clone(), requires_grad=requires_grad)
        self.angles = nn.Parameter(angles.clone(), requires_grad=requires_grad)
        self.torsions = nn.Parameter(torsions.clone(), requires_grad=requires_grad)
        self.chain_positions = nn.Parameter(
            chain_positions.clone(), requires_grad=requires_grad
        )
        self.chain_orientations = nn.Parameter(
            chain_orientations.clone(), requires_grad=requires_grad
        )

        # Initialize refinable mask (all atoms refinable by default)
        # True = use internal coordinates (refinable), False = use fixed xyz (frozen)
        self.register_buffer(
            "refinable_mask",
            torch.ones(self.n_atoms, dtype=torch.bool, device=device)
        )
        # Buffer to store frozen atom coordinates
        self.register_buffer(
            "fixed_xyz",
            xyz.clone()
        )

    def _extract_ring_local_coords(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Extract local coordinates for ring atoms relative to ring anchors.

        Parameters
        ----------
        xyz : torch.Tensor
            Atomic coordinates of shape (N, 3).

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

        for ring_idx in range(self.n_rings):
            anchor = self.ring_anchor_atoms[ring_idx].item()
            anchor_pos = xyz[anchor]

            # Compute local frame at anchor (using parent direction)
            if self.parent_indices[anchor] >= 0:
                parent_pos = xyz[self.parent_indices[anchor]]
                z_axis = anchor_pos - parent_pos
                z_norm = torch.linalg.norm(z_axis)
                if z_norm > 1e-10:
                    z_axis = z_axis / z_norm
                else:
                    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
            else:
                z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

            # Arbitrary perpendicular axes
            if abs(z_axis[0]) < 0.9:
                x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
            else:
                x_axis = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
            x_axis = x_axis - torch.dot(x_axis, z_axis) * z_axis
            x_axis = x_axis / torch.linalg.norm(x_axis)
            y_axis = torch.linalg.cross(z_axis, x_axis)

            R = torch.stack([x_axis, y_axis, z_axis], dim=1)  # (3, 3)

            # Store local coords for each ring member
            for local_idx in range(self.ring_sizes[ring_idx].item()):
                atom_idx = self.ring_members[ring_idx, local_idx].item()
                if atom_idx >= 0:
                    global_offset = xyz[atom_idx] - anchor_pos
                    local_offset = R.T @ global_offset  # Transform to local frame
                    ring_local_coords[ring_idx, local_idx] = local_offset

        return ring_local_coords

    @staticmethod
    def _compute_torsion_angles(
        p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute torsion angles for batched positions.

        Parameters
        ----------
        p1, p2, p3, p4 : torch.Tensor
            Atom positions of shape (N, 3) defining torsion angle p1-p2-p3-p4.

        Returns
        -------
        torch.Tensor
            Torsion angles in radians, shape (N,).
        """
        # Vectors along the backbone
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3

        # Normal vectors to planes
        n1 = torch.linalg.cross(b1, b2)
        n2 = torch.linalg.cross(b2, b3)

        # Normalize
        n1 = n1 / torch.linalg.norm(n1, dim=-1, keepdim=True).clamp(min=1e-10)
        n2 = n2 / torch.linalg.norm(n2, dim=-1, keepdim=True).clamp(min=1e-10)

        # Unit vector along b2
        b2_norm = b2 / torch.linalg.norm(b2, dim=-1, keepdim=True).clamp(min=1e-10)

        # m1 is perpendicular to n1 and b2
        m1 = torch.linalg.cross(n1, b2_norm)

        # Compute torsion angle using atan2
        x = torch.sum(n1 * n2, dim=-1)
        y = torch.sum(m1 * n2, dim=-1)

        return torch.atan2(y, x)

    def _place_second_atoms(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Place second atoms of each chain (depth=1) in parallel.

        Parameters
        ----------
        xyz : torch.Tensor
            Current coordinates of shape (N, 3).

        Returns
        -------
        torch.Tensor
            Updated coordinates.
        """
        # Find atoms with depth 1
        mask = self.atom_depth == 1
        if not mask.any():
            return xyz

        xyz = xyz.clone()

        # Get chain id for each depth-1 atom
        chain_ids = self.chain_ids[mask]

        # Get root positions for these chains
        root_positions = self.chain_positions[chain_ids]  # (n_second, 3)

        # Get chain orientations and apply to second atom directions
        chain_orients = self.chain_orientations[chain_ids]  # (n_second, 3)
        R_matrices = axis_angle_to_rotation_matrix(chain_orients)  # (n_second, 3, 3)

        # Get base directions for these chains
        base_dirs = self.second_atom_dirs[chain_ids]  # (n_second, 3)

        # Rotate directions
        rotated_dirs = torch.bmm(R_matrices, base_dirs.unsqueeze(-1)).squeeze(-1)

        # Get bond lengths for these atoms
        bond_idx = self.bond_param_indices[mask]
        bond_lengths = self.bond_lengths[bond_idx]  # (n_second,)

        # Compute positions
        new_positions = root_positions + bond_lengths.unsqueeze(-1) * rotated_dirs

        xyz[mask] = new_positions
        return xyz

    def _place_third_atoms(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Place third atoms of each chain (depth=2) in parallel.

        Parameters
        ----------
        xyz : torch.Tensor
            Current coordinates of shape (N, 3).

        Returns
        -------
        torch.Tensor
            Updated coordinates.
        """
        # Find atoms with depth 2
        mask = self.atom_depth == 2
        if not mask.any():
            return xyz

        xyz = xyz.clone()

        # Get parent and grandparent positions
        parent_idx = self.parent_indices[mask]
        grandparent_idx = self.grandparent_indices[mask]

        parent_pos = xyz[parent_idx]  # (n_third, 3)
        grandparent_pos = xyz[grandparent_idx]  # (n_third, 3)

        # Get bond lengths and angles
        bond_idx = self.bond_param_indices[mask]
        angle_idx = self.angle_param_indices[mask]

        bond_lengths = self.bond_lengths[bond_idx]  # (n_third,)
        angles = self.angles[angle_idx]  # (n_third,)

        # Build local frame at parent
        bc = parent_pos - grandparent_pos  # (n_third, 3)
        bc_norm = torch.linalg.norm(bc, dim=-1, keepdim=True).clamp(min=1e-10)
        bc_unit = bc / bc_norm

        # Get stored perpendicular directions for depth-2 atoms
        # Use the atom-to-perp index mapping
        atom_indices = torch.where(mask)[0]
        perp_idx = self.depth2_atom_to_perp_idx[atom_indices]
        ref_perp = self.depth2_perps[perp_idx]  # (n_third, 3)

        # Apply chain rotation to reference perpendicular
        chain_ids = self.chain_ids[mask]
        chain_orients = self.chain_orientations[chain_ids]
        R_matrices = axis_angle_to_rotation_matrix(chain_orients)
        rotated_ref = torch.bmm(R_matrices, ref_perp.unsqueeze(-1)).squeeze(-1)

        # Make sure rotated_ref is perpendicular to bc_unit
        proj = torch.sum(rotated_ref * bc_unit, dim=-1, keepdim=True)
        rotated_ref = rotated_ref - proj * bc_unit
        ref_norm = torch.linalg.norm(rotated_ref, dim=-1, keepdim=True).clamp(min=1e-10)
        rotated_ref = rotated_ref / ref_norm

        # Compute new position using bond length and angle
        # The angle is the bond angle at parent: grandparent-parent-child
        # For angle=pi (linear), child continues along the bc direction
        # For angle < pi, child deviates by (pi - angle) from the bc direction
        theta_internal = torch.pi - angles
        dx = bond_lengths * torch.cos(theta_internal)
        dy = bond_lengths * torch.sin(theta_internal)

        # Child position: along bc direction (continuing from grandparent through parent)
        # plus perpendicular offset for non-linear angles
        new_positions = parent_pos + dx.unsqueeze(-1) * bc_unit + dy.unsqueeze(-1) * rotated_ref

        xyz[mask] = new_positions
        return xyz

    def _place_atoms_at_depth(self, xyz: torch.Tensor, depth: int) -> torch.Tensor:
        """
        Place all atoms at given depth in parallel (fully vectorized).

        Uses the NeRF algorithm to reconstruct Cartesian coordinates from
        bond lengths, angles, and torsions.

        Parameters
        ----------
        xyz : torch.Tensor
            Current coordinates of shape (N, 3).
        depth : int
            Tree depth to process.

        Returns
        -------
        torch.Tensor
            Updated coordinates.
        """
        # Get all atom indices at this depth (pre-computed mask)
        mask = self.atom_depth == depth
        n_atoms_at_depth = mask.sum()

        if n_atoms_at_depth == 0:
            return xyz

        xyz = xyz.clone()

        # Gather reference positions via advanced indexing (batched)
        parent_idx = self.parent_indices[mask]
        gp_idx = self.grandparent_indices[mask]
        ggp_idx = self.great_grandparent_indices[mask]

        p1 = xyz[ggp_idx]  # (n_atoms_at_depth, 3)
        p2 = xyz[gp_idx]  # (n_atoms_at_depth, 3)
        p3 = xyz[parent_idx]  # (n_atoms_at_depth, 3)

        # Gather internal coordinates (batched)
        bond_idx = self.bond_param_indices[mask]
        angle_idx = self.angle_param_indices[mask]
        torsion_idx = self.torsion_param_indices[mask]

        d = self.bond_lengths[bond_idx]  # (n_atoms_at_depth,)
        theta = self.angles[angle_idx]  # (n_atoms_at_depth,)
        phi = self.torsions[torsion_idx]  # (n_atoms_at_depth,)

        # Build local coordinate frame (batched, all in parallel)
        bc = p3 - p2  # (n_atoms_at_depth, 3)
        bc = bc / torch.linalg.norm(bc, dim=-1, keepdim=True).clamp(min=1e-10)

        ab = p2 - p1
        n = torch.linalg.cross(ab, bc)
        n = n / torch.linalg.norm(n, dim=-1, keepdim=True).clamp(min=1e-10)

        m = torch.linalg.cross(n, bc)

        # Compute positions (batched trigonometry)
        theta_internal = torch.pi - theta
        sin_theta = torch.sin(theta_internal)
        cos_theta = torch.cos(theta_internal)

        dx = d * cos_theta  # (n_atoms_at_depth,)
        dy = d * sin_theta * torch.cos(phi)
        dz = d * sin_theta * torch.sin(phi)

        # Global positions (batched)
        # Child continues along bc direction (from grandparent through parent)
        # The torsion angle determines rotation around bc axis
        # Negate dz to match torsion sign convention
        new_pos = (
            p3
            + dx.unsqueeze(-1) * bc
            + dy.unsqueeze(-1) * m
            - dz.unsqueeze(-1) * n
        )  # (n_atoms_at_depth, 3)

        # Scatter back to xyz
        xyz[mask] = new_pos

        return xyz

    def _place_atoms_at_depth_fast(
        self, xyz: torch.Tensor, depth_idx: int
    ) -> torch.Tensor:
        """
        Place atoms at a depth level using pre-computed indices (fast path).

        This avoids recomputing masks during the forward pass.

        Parameters
        ----------
        xyz : torch.Tensor
            Current coordinates of shape (N, 3).
        depth_idx : int
            Index into pre-computed depth arrays (0 = depth 3, 1 = depth 4, etc.)

        Returns
        -------
        torch.Tensor
            Updated coordinates.
        """
        atom_idx = self._depth_atom_indices[depth_idx]

        if atom_idx.numel() == 0:
            return xyz

        xyz = xyz.clone()

        # Use pre-computed indices
        parent_idx = self._depth_parent_indices[depth_idx]
        gp_idx = self._depth_gp_indices[depth_idx]
        ggp_idx = self._depth_ggp_indices[depth_idx]

        p1 = xyz[ggp_idx]
        p2 = xyz[gp_idx]
        p3 = xyz[parent_idx]

        # Use pre-computed parameter indices
        bond_idx = self._depth_bond_param_indices[depth_idx]
        angle_idx = self._depth_angle_param_indices[depth_idx]
        torsion_idx = self._depth_torsion_param_indices[depth_idx]

        d = self.bond_lengths[bond_idx]
        theta = self.angles[angle_idx]
        phi = self.torsions[torsion_idx]

        # Build local coordinate frame
        bc = p3 - p2
        bc = bc / torch.linalg.norm(bc, dim=-1, keepdim=True).clamp(min=1e-10)

        ab = p2 - p1
        n = torch.linalg.cross(ab, bc)
        n = n / torch.linalg.norm(n, dim=-1, keepdim=True).clamp(min=1e-10)

        m = torch.linalg.cross(n, bc)

        # Compute positions
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

        # Scatter back using pre-computed atom indices
        xyz[atom_idx] = new_pos

        return xyz

    def _place_rigid_rings(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Place ring atoms as rigid groups around anchor (vectorized).

        Ring atoms (except anchor) are placed at fixed offsets from
        the anchor atom in its local coordinate frame.

        Parameters
        ----------
        xyz : torch.Tensor
            Current coordinates of shape (N, 3).

        Returns
        -------
        torch.Tensor
            Updated coordinates with ring atoms placed.
        """
        if self.n_rings == 0:
            return xyz

        # Get anchor positions and parent positions (vectorized)
        anchor_pos = xyz[self.ring_anchor_atoms]  # (n_rings, 3)
        parent_idx = self.parent_indices[self.ring_anchor_atoms]  # (n_rings,)

        # Handle anchors with valid parents
        valid_parent_mask = parent_idx >= 0
        parent_pos = torch.zeros_like(anchor_pos)
        parent_pos[valid_parent_mask] = xyz[parent_idx[valid_parent_mask]]

        # Compute z-axis for all rings (vectorized)
        z_axis = anchor_pos - parent_pos
        z_norm = torch.linalg.norm(z_axis, dim=-1, keepdim=True)
        z_axis = torch.where(
            z_norm > 1e-10,
            z_axis / z_norm.clamp(min=1e-10),
            torch.tensor([0.0, 0.0, 1.0], dtype=self._dtype, device=self._device)
        )

        # For anchors without valid parents, use default z-axis
        z_axis[~valid_parent_mask] = torch.tensor(
            [0.0, 0.0, 1.0], dtype=self._dtype, device=self._device
        )

        # Compute perpendicular axes (vectorized)
        # Use [1,0,0] unless z is nearly parallel to it
        x_base = torch.zeros_like(z_axis)
        x_base[:, 0] = 1.0  # Default to [1,0,0]
        nearly_x = torch.abs(z_axis[:, 0]) >= 0.9
        x_base[nearly_x, 0] = 0.0
        x_base[nearly_x, 1] = 1.0  # Use [0,1,0] when z is near x-axis

        # Gram-Schmidt: x = x_base - (x_base . z) * z
        x_axis = x_base - (x_base * z_axis).sum(dim=-1, keepdim=True) * z_axis
        x_axis = x_axis / torch.linalg.norm(x_axis, dim=-1, keepdim=True).clamp(min=1e-10)
        y_axis = torch.linalg.cross(z_axis, x_axis)

        # Build rotation matrices: R = [x, y, z] as columns
        R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (n_rings, 3, 3)

        # Now place all ring atoms at once
        # ring_members: (n_rings, max_ring_size) - atom indices
        # ring_local_coords: (n_rings, max_ring_size, 3) - local offsets
        # We need to transform each local offset by the ring's rotation matrix

        # Get valid ring member mask (exclude padding -1 and anchor atoms)
        valid_mask = self.ring_members >= 0  # (n_rings, max_ring_size)
        # Also exclude anchors (they're already placed)
        anchor_expanded = self.ring_anchor_atoms.unsqueeze(1)  # (n_rings, 1)
        valid_mask = valid_mask & (self.ring_members != anchor_expanded)

        # Transform local coords to global: global = R @ local + anchor_pos
        # local_coords: (n_rings, max_ring_size, 3)
        # R: (n_rings, 3, 3)
        # anchor_pos: (n_rings, 3)

        # Batch matrix multiply: (n_rings, max_ring_size, 3) @ (n_rings, 3, 3).T
        # -> (n_rings, max_ring_size, 3)
        global_offsets = torch.bmm(
            self.ring_local_coords,
            R.transpose(-1, -2)
        )  # (n_rings, max_ring_size, 3)

        global_pos = global_offsets + anchor_pos.unsqueeze(1)  # (n_rings, max_ring_size, 3)

        # Scatter the results back to xyz
        # Flatten valid entries and their positions
        ring_indices, local_indices = torch.where(valid_mask)
        atom_indices = self.ring_members[ring_indices, local_indices]
        positions = global_pos[ring_indices, local_indices]

        xyz[atom_indices] = positions

        return xyz

    def forward_slow(self) -> torch.Tensor:
        """
        Reconstruct Cartesian xyz from internal coordinates.

        Fully vectorized - processes each depth level in parallel.
        Only log(max_depth) sequential steps required.

        Returns
        -------
        torch.Tensor
            Reconstructed Cartesian coordinates of shape (N, 3).
        """
        xyz = torch.zeros(
            self.n_atoms, 3, dtype=self._dtype, device=self._device
        )

        # Place chain roots (vectorized) with chain orientations
        # Apply rotation to initial positions if needed
        xyz[self.chain_roots] = self.chain_positions

        # Place second atoms of each chain (vectorized)
        xyz = self._place_second_atoms(xyz)

        # Place third atoms of each chain (vectorized)
        xyz = self._place_third_atoms(xyz)

        # Place remaining atoms by depth using pre-computed indices (fast path)
        for depth_idx in range(len(self._depth_atom_indices)):
            xyz = self._place_atoms_at_depth_fast(xyz, depth_idx)

        # Place rigid ring atoms (vectorized)
        xyz = self._place_rigid_rings(xyz)

        # Apply frozen coordinates
        # For atoms that are not refinable (frozen), use fixed_xyz
        if not self.refinable_mask.all():
            frozen_mask = ~self.refinable_mask
            xyz[frozen_mask] = self.fixed_xyz[frozen_mask]

        return xyz

    def forward(self) -> torch.Tensor:
        """
        Reconstruct Cartesian xyz from internal coordinates.

        Uses optimized parallel scan method for efficiency.

        Returns
        -------
        torch.Tensor
            Reconstructed Cartesian coordinates of shape (N, 3).
        """
        return self.forward_parallel()

    def shake(self, magnitude: float = 0.1) -> torch.Tensor:
        """
        Add Gaussian noise to internal parameters (fully vectorized).

        All operations are batched tensor ops - no loops.

        Parameters
        ----------
        magnitude : float, optional
            Standard deviation of Gaussian noise. Default is 0.1.
            For bond lengths, this is in Angstroms.
            For angles and torsions, this is in radians.

        Returns
        -------
        torch.Tensor
            New Cartesian coordinates after perturbation.
        """
        with torch.no_grad():
            # Perturb bond lengths (vectorized)
            if self.bond_lengths.numel() > 0:
                self.bond_lengths.data += (
                    torch.randn_like(self.bond_lengths) * magnitude
                )
                self.bond_lengths.data = torch.clamp(self.bond_lengths.data, min=0.5)

            # Perturb angles (vectorized)
            if self.angles.numel() > 0:
                self.angles.data += torch.randn_like(self.angles) * magnitude
                self.angles.data = torch.clamp(
                    self.angles.data, min=0.1, max=torch.pi - 0.1
                )

            # Perturb torsions with wrap (vectorized)
            if self.torsions.numel() > 0:
                self.torsions.data += torch.randn_like(self.torsions) * magnitude
                self.torsions.data = torch.atan2(
                    torch.sin(self.torsions.data), torch.cos(self.torsions.data)
                )

            # Perturb chain positions (vectorized)
            if self.chain_positions.numel() > 0:
                self.chain_positions.data += (
                    torch.randn_like(self.chain_positions) * magnitude
                )

        return self.forward()

    # ===== Freeze/Unfreeze Methods (MixedTensor-style interface) =====

    def fix(
        self,
        selection: Union[torch.Tensor, slice, None] = None,
        freeze_at_current: bool = True
    ) -> None:
        """
        Fix (freeze) atoms to use fixed xyz coordinates instead of internal coordinates.

        Fixed atoms will not be updated during reconstruction from internal coordinates.
        Their positions will remain at the stored fixed_xyz values.

        Parameters
        ----------
        selection : torch.Tensor, slice, or None
            Boolean mask (shape n_atoms) or indices of atoms to fix.
            If None, fixes all atoms.
        freeze_at_current : bool, optional
            If True (default), store current reconstructed xyz for the selected atoms.
            If False, use the existing fixed_xyz values.
        """
        if selection is None:
            selection = slice(None)

        # Convert boolean mask or indices to proper indexing
        if isinstance(selection, torch.Tensor) and selection.dtype == torch.bool:
            mask = selection
        elif isinstance(selection, torch.Tensor):
            mask = torch.zeros(self.n_atoms, dtype=torch.bool, device=self._device)
            mask[selection] = True
        elif isinstance(selection, slice):
            mask = torch.zeros(self.n_atoms, dtype=torch.bool, device=self._device)
            mask[selection] = True
        else:
            raise TypeError(f"selection must be Tensor, slice, or None, got {type(selection)}")

        if freeze_at_current:
            # Compute current coordinates and store them
            with torch.no_grad():
                current_xyz = self.forward()
                self.fixed_xyz[mask] = current_xyz[mask]

        # Mark atoms as not refinable (frozen)
        self.refinable_mask[mask] = False

    def freeze(
        self,
        selection: Union[torch.Tensor, slice, None] = None,
        freeze_at_current: bool = True
    ) -> None:
        """
        Alias for fix(). Freeze atoms to use fixed xyz coordinates.

        See fix() for full documentation.
        """
        self.fix(selection, freeze_at_current)

    def refine(
        self,
        selection: Union[torch.Tensor, slice, None] = None,
        rebuild: bool = True
    ) -> None:
        """
        Make atoms refinable by computing their positions from internal coordinates.

        This unfreezes atoms, meaning their positions will be computed from
        bond lengths, angles, and torsions during forward pass.

        Parameters
        ----------
        selection : torch.Tensor, slice, or None
            Boolean mask (shape n_atoms) or indices of atoms to make refinable.
            If None, makes all atoms refinable.
        rebuild : bool, optional
            If True (default), rebuild internal coordinates from current fixed_xyz
            for the selected atoms. This ensures the internal coordinates match
            the current atom positions before unfreezing.
        """
        if selection is None:
            selection = slice(None)

        # Convert boolean mask or indices to proper indexing
        if isinstance(selection, torch.Tensor) and selection.dtype == torch.bool:
            mask = selection
        elif isinstance(selection, torch.Tensor):
            mask = torch.zeros(self.n_atoms, dtype=torch.bool, device=self._device)
            mask[selection] = True
        elif isinstance(selection, slice):
            mask = torch.zeros(self.n_atoms, dtype=torch.bool, device=self._device)
            mask[selection] = True
        else:
            raise TypeError(f"selection must be Tensor, slice, or None, got {type(selection)}")

        if rebuild:
            # Update internal coordinates for selected atoms from fixed_xyz
            self._update_internal_coords_from_xyz(self.fixed_xyz, mask)

        # Mark atoms as refinable
        self.refinable_mask[mask] = True

    def unfreeze(
        self,
        selection: Union[torch.Tensor, slice, None] = None,
        rebuild: bool = True
    ) -> None:
        """
        Alias for refine(). Unfreeze atoms to use internal coordinates.

        See refine() for full documentation.
        """
        self.refine(selection, rebuild)

    def fix_all(self, freeze_at_current: bool = True) -> None:
        """
        Fix (freeze) all atoms.

        Parameters
        ----------
        freeze_at_current : bool, optional
            If True (default), store current reconstructed xyz for all atoms.
        """
        self.fix(None, freeze_at_current)

    def freeze_all(self, freeze_at_current: bool = True) -> None:
        """
        Alias for fix_all(). Freeze all atoms.
        """
        self.fix_all(freeze_at_current)

    def refine_all(self, rebuild: bool = True) -> None:
        """
        Make all atoms refinable.

        Parameters
        ----------
        rebuild : bool, optional
            If True (default), rebuild internal coordinates from current fixed_xyz.
        """
        self.refine(None, rebuild)

    def unfreeze_all(self, rebuild: bool = True) -> None:
        """
        Alias for refine_all(). Unfreeze all atoms.
        """
        self.refine_all(rebuild)

    def _update_internal_coords_from_xyz(
        self,
        xyz: torch.Tensor,
        mask: torch.Tensor
    ) -> None:
        """
        Update internal coordinate parameters from xyz for masked atoms.

        This re-extracts bond lengths, angles, and torsions from the provided
        xyz coordinates for atoms where mask is True.

        Parameters
        ----------
        xyz : torch.Tensor
            Atomic coordinates of shape (N, 3).
        mask : torch.Tensor
            Boolean mask of shape (N,) indicating which atoms to update.
        """
        # Update bond lengths for masked atoms with depth >= 1
        bond_update_mask = mask & (self.atom_depth >= 1)
        if bond_update_mask.any():
            child_xyz = xyz[bond_update_mask]
            parent_xyz = xyz[self.parent_indices[bond_update_mask]]
            new_bonds = torch.linalg.norm(child_xyz - parent_xyz, dim=-1)
            bond_idx = self.bond_param_indices[bond_update_mask]
            with torch.no_grad():
                self.bond_lengths.data[bond_idx] = new_bonds

        # Update angles for masked atoms with depth >= 2
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

        # Update torsions for masked atoms with depth >= 3
        torsion_update_mask = mask & (self.atom_depth >= 3)
        if torsion_update_mask.any():
            child_xyz = xyz[torsion_update_mask]
            parent_xyz = xyz[self.parent_indices[torsion_update_mask]]
            grandparent_xyz = xyz[self.grandparent_indices[torsion_update_mask]]
            great_grandparent_xyz = xyz[self.great_grandparent_indices[torsion_update_mask]]

            new_torsions = self._compute_torsion_angles(
                great_grandparent_xyz, grandparent_xyz, parent_xyz, child_xyz
            )
            torsion_idx = self.torsion_param_indices[torsion_update_mask]
            with torch.no_grad():
                self.torsions.data[torsion_idx] = new_torsions

        # Update chain positions if any chain roots are in the mask
        for chain_idx in range(self.n_chains):
            root = self.chain_roots[chain_idx]
            if mask[root]:
                with torch.no_grad():
                    self.chain_positions.data[chain_idx] = xyz[root]

    @property
    def n_refinable(self) -> int:
        """Return the number of refinable (unfrozen) atoms."""
        return self.refinable_mask.sum().item()

    @property
    def n_fixed(self) -> int:
        """Return the number of fixed (frozen) atoms."""
        return (~self.refinable_mask).sum().item()

    def __repr__(self) -> str:
        return (
            f"InternalCoordinateTensor(n_atoms={self.n_atoms}, "
            f"n_refinable={self.n_refinable}, n_fixed={self.n_fixed}, "
            f"n_chains={self.n_chains}, n_bonds={self.n_bonds}, "
            f"n_angles={self.n_angles}, n_torsions={self.n_torsions}, "
            f"n_rings={self.n_rings}, max_depth={self.max_depth}, "
            f"device={self._device})"
        )

    # ===== Parallel Scan Methods for Optimized Forward Pass =====

    def _find_longest_path(self) -> list:
        """
        Find the longest path (backbone) in the spanning tree.

        Returns list of atom indices from root to leaf.
        """
        n_atoms = self.n_atoms
        parent = self.parent_indices

        # Count children for each atom
        children = [[] for _ in range(n_atoms)]
        for i in range(n_atoms):
            p = parent[i].item()
            if p >= 0:
                children[p].append(i)

        # Find leaves
        leaves = [i for i in range(n_atoms) if len(children[i]) == 0]

        # Find longest path from any leaf to root
        longest_path = []
        for leaf in leaves:
            path = [leaf]
            current = leaf
            while parent[current].item() >= 0:
                current = parent[current].item()
                path.append(current)
            path.reverse()
            if len(path) > len(longest_path):
                longest_path = path

        return longest_path

    def _compute_delta_transform(
        self, d: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor
    ) -> tuple:
        """
        Compute delta transforms for NeRF in local frame.

        For each atom, computes (R_delta, t_delta) where:
        - t_delta is the local position of the atom in parent's frame
        - R_delta is the rotation from parent's frame to child's frame

        Parameters
        ----------
        d : torch.Tensor
            Bond lengths, shape (N,)
        theta : torch.Tensor
            Bond angles in radians, shape (N,)
        phi : torch.Tensor
            Torsion angles in radians, shape (N,)

        Returns
        -------
        R : torch.Tensor
            Rotation matrices, shape (N, 3, 3)
        t : torch.Tensor
            Translation vectors, shape (N, 3)
        """
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        zeros = torch.zeros_like(theta)

        # Rotation matrix columns in parent's (m, n, bc) frame
        col0 = torch.stack([-cos_phi * cos_theta, sin_phi * cos_theta, -sin_theta], dim=-1)
        col1 = torch.stack([sin_phi, cos_phi, zeros], dim=-1)
        col2 = torch.stack([sin_theta * cos_phi, -sin_theta * sin_phi, -cos_theta], dim=-1)

        R = torch.stack([col0, col1, col2], dim=-1)  # (N, 3, 3)
        t = d.unsqueeze(-1) * col2  # (N, 3)

        return R, t

    def _compose_transforms(
        self,
        R1: torch.Tensor, t1: torch.Tensor,
        R2: torch.Tensor, t2: torch.Tensor
    ) -> tuple:
        """
        Compose rigid transforms: T1 * T2 = (R1 @ R2, R1 @ t2 + t1)
        """
        R = R1 @ R2
        t = (R1 @ t2.unsqueeze(-1)).squeeze(-1) + t1
        return R, t

    def _parallel_scan(
        self, R_deltas: torch.Tensor, t_deltas: torch.Tensor
    ) -> tuple:
        """
        Parallel prefix scan of rigid transforms (Hillis-Steele style).

        Computes G[i] = T_0 * T_1 * ... * T_{i-1} for each i.

        Parameters
        ----------
        R_deltas : torch.Tensor
            Per-step rotations, shape (N, 3, 3)
        t_deltas : torch.Tensor
            Per-step translations, shape (N, 3)

        Returns
        -------
        R_cum : torch.Tensor
            Cumulative rotations, shape (N+1, 3, 3)
        t_cum : torch.Tensor
            Cumulative translations, shape (N+1, 3)
        """
        N = R_deltas.shape[0]
        device = R_deltas.device
        dtype = R_deltas.dtype

        if N == 0:
            R_cum = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
            t_cum = torch.zeros(1, 3, dtype=dtype, device=device)
            return R_cum, t_cum

        # Work with N+1 elements (element 0 is identity)
        # Use double buffering to avoid clone() overhead
        R = torch.zeros(N + 1, 3, 3, dtype=dtype, device=device)
        t = torch.zeros(N + 1, 3, dtype=dtype, device=device)
        R_buf = torch.zeros(N + 1, 3, 3, dtype=dtype, device=device)
        t_buf = torch.zeros(N + 1, 3, dtype=dtype, device=device)

        R[0] = torch.eye(3, dtype=dtype, device=device)
        R[1:] = R_deltas
        t[1:] = t_deltas

        # Hillis-Steele inclusive scan with double buffering
        offset = 1
        use_buf = False
        while offset < N + 1:
            # Select source and destination buffers
            if use_buf:
                R_src_buf, t_src_buf = R_buf, t_buf
                R_dst_buf, t_dst_buf = R, t
            else:
                R_src_buf, t_src_buf = R, t
                R_dst_buf, t_dst_buf = R_buf, t_buf

            # Copy unchanged elements
            R_dst_buf[:offset] = R_src_buf[:offset]
            t_dst_buf[:offset] = t_src_buf[:offset]

            # Compute and store updated elements
            idx = torch.arange(offset, N + 1, device=device)
            idx_src = idx - offset

            R_src = R_src_buf[idx_src]
            t_src = t_src_buf[idx_src]
            R_dst = R_src_buf[idx]
            t_dst = t_src_buf[idx]

            R_new, t_new = self._compose_transforms(R_src, t_src, R_dst, t_dst)
            R_dst_buf[idx] = R_new
            t_dst_buf[idx] = t_new

            use_buf = not use_buf
            offset *= 2

        # Return the buffer that has the final result
        if use_buf:
            return R_buf, t_buf
        else:
            return R, t

    def _build_backbone_data(self):
        """
        Pre-compute backbone and side chain data for parallel forward.

        This identifies the longest path (backbone), computes which atoms
        are side chains, and determines their branching points.
        """
        # Find the longest path
        backbone = self._find_longest_path()
        backbone_set = set(backbone)

        # Count children for each atom
        children = [[] for _ in range(self.n_atoms)]
        for i in range(self.n_atoms):
            p = self.parent_indices[i].item()
            if p >= 0:
                children[p].append(i)

        # Find side chain atoms (not on backbone)
        side_chain_atoms = [i for i in range(self.n_atoms) if i not in backbone_set]

        # Compute side chain depths (depth from branching point)
        side_chain_depth = torch.full((self.n_atoms,), -1, dtype=torch.long, device=self._device)
        side_chain_depth[backbone] = 0  # Backbone atoms have "depth 0" from themselves

        # For each backbone atom, find non-backbone children and do BFS
        max_sc_depth = 0
        for bp_atom in backbone:
            non_backbone_children = [c for c in children[bp_atom] if c not in backbone_set]
            for start in non_backbone_children:
                queue = [(start, 1)]
                while queue:
                    atom, depth = queue.pop(0)
                    side_chain_depth[atom] = depth
                    max_sc_depth = max(max_sc_depth, depth)
                    for child in children[atom]:
                        queue.append((child, depth + 1))

        # Store as buffers
        self.register_buffer(
            "_backbone",
            torch.tensor(backbone, dtype=torch.long, device=self._device)
        )
        self.register_buffer("_side_chain_depth", side_chain_depth)
        self._max_side_chain_depth = max_sc_depth
        self._backbone_set = backbone_set

        # Pre-compute backbone internal coordinates indices
        if len(backbone) > 3:
            self._backbone_bond_idx = []
            self._backbone_angle_idx = []
            self._backbone_torsion_idx = []

            for i in range(3, len(backbone)):
                atom_idx = backbone[i]
                bond_idx = self.bond_param_indices[atom_idx].item()
                angle_idx = self.angle_param_indices[atom_idx].item()
                torsion_idx = self.torsion_param_indices[atom_idx].item()
                self._backbone_bond_idx.append(bond_idx)
                self._backbone_angle_idx.append(angle_idx)
                self._backbone_torsion_idx.append(torsion_idx)

            self.register_buffer(
                "_backbone_bond_idx_tensor",
                torch.tensor(self._backbone_bond_idx, dtype=torch.long, device=self._device)
            )
            self.register_buffer(
                "_backbone_angle_idx_tensor",
                torch.tensor(self._backbone_angle_idx, dtype=torch.long, device=self._device)
            )
            self.register_buffer(
                "_backbone_torsion_idx_tensor",
                torch.tensor(self._backbone_torsion_idx, dtype=torch.long, device=self._device)
            )

        # Pre-compute side chain depth indices (similar to _build_depth_indices but for SC depth)
        self._sc_depth_atom_indices = []
        self._sc_depth_parent_indices = []
        self._sc_depth_gp_indices = []
        self._sc_depth_ggp_indices = []
        self._sc_depth_bond_idx = []
        self._sc_depth_angle_idx = []
        self._sc_depth_torsion_idx = []

        for d in range(1, max_sc_depth + 1):
            mask = (side_chain_depth == d) & (self.atom_depth >= 3)
            if mask.any():
                atom_idx = torch.where(mask)[0]
                self._sc_depth_atom_indices.append(atom_idx)
                self._sc_depth_parent_indices.append(self.parent_indices[atom_idx])
                self._sc_depth_gp_indices.append(self.grandparent_indices[atom_idx])
                self._sc_depth_ggp_indices.append(self.great_grandparent_indices[atom_idx])
                self._sc_depth_bond_idx.append(self.bond_param_indices[atom_idx])
                self._sc_depth_angle_idx.append(self.angle_param_indices[atom_idx])
                self._sc_depth_torsion_idx.append(self.torsion_param_indices[atom_idx])
            else:
                empty = torch.tensor([], dtype=torch.long, device=self._device)
                self._sc_depth_atom_indices.append(empty)
                self._sc_depth_parent_indices.append(empty)
                self._sc_depth_gp_indices.append(empty)
                self._sc_depth_ggp_indices.append(empty)
                self._sc_depth_bond_idx.append(empty)
                self._sc_depth_angle_idx.append(empty)
                self._sc_depth_torsion_idx.append(empty)

        # Pre-compute merged indices for batched side chain placement
        # Merge all SC atoms (depth >= 3) into single tensors for efficiency
        all_sc_atoms = []
        all_sc_parents = []
        all_sc_gp = []
        all_sc_ggp = []
        all_sc_bond_idx = []
        all_sc_angle_idx = []
        all_sc_torsion_idx = []
        all_sc_depths = []

        for d in range(max_sc_depth):
            if self._sc_depth_atom_indices[d].numel() > 0:
                all_sc_atoms.append(self._sc_depth_atom_indices[d])
                all_sc_parents.append(self._sc_depth_parent_indices[d])
                all_sc_gp.append(self._sc_depth_gp_indices[d])
                all_sc_ggp.append(self._sc_depth_ggp_indices[d])
                all_sc_bond_idx.append(self._sc_depth_bond_idx[d])
                all_sc_angle_idx.append(self._sc_depth_angle_idx[d])
                all_sc_torsion_idx.append(self._sc_depth_torsion_idx[d])
                all_sc_depths.append(torch.full_like(self._sc_depth_atom_indices[d], d + 1))

        if all_sc_atoms:
            self.register_buffer("_all_sc_atoms", torch.cat(all_sc_atoms))
            self.register_buffer("_all_sc_parents", torch.cat(all_sc_parents))
            self.register_buffer("_all_sc_gp", torch.cat(all_sc_gp))
            self.register_buffer("_all_sc_ggp", torch.cat(all_sc_ggp))
            self.register_buffer("_all_sc_bond_idx", torch.cat(all_sc_bond_idx))
            self.register_buffer("_all_sc_angle_idx", torch.cat(all_sc_angle_idx))
            self.register_buffer("_all_sc_torsion_idx", torch.cat(all_sc_torsion_idx))
            self.register_buffer("_all_sc_depths", torch.cat(all_sc_depths))
        else:
            empty = torch.tensor([], dtype=torch.long, device=self._device)
            self.register_buffer("_all_sc_atoms", empty)
            self.register_buffer("_all_sc_parents", empty)
            self.register_buffer("_all_sc_gp", empty)
            self.register_buffer("_all_sc_ggp", empty)
            self.register_buffer("_all_sc_bond_idx", empty)
            self.register_buffer("_all_sc_angle_idx", empty)
            self.register_buffer("_all_sc_torsion_idx", empty)
            self.register_buffer("_all_sc_depths", empty)

    def _place_atoms_nerf(
        self, xyz: torch.Tensor, atom_idx: torch.Tensor,
        parent_idx: torch.Tensor, gp_idx: torch.Tensor, ggp_idx: torch.Tensor,
        bond_idx: torch.Tensor, angle_idx: torch.Tensor, torsion_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Place atoms using NeRF algorithm (vectorized helper).

        Parameters
        ----------
        xyz : torch.Tensor
            Current coordinates.
        atom_idx, parent_idx, gp_idx, ggp_idx : torch.Tensor
            Atom indices.
        bond_idx, angle_idx, torsion_idx : torch.Tensor
            Parameter indices.

        Returns
        -------
        torch.Tensor
            Updated coordinates.
        """
        p1 = xyz[ggp_idx]
        p2 = xyz[gp_idx]
        p3 = xyz[parent_idx]

        d = self.bond_lengths[bond_idx]
        theta = self.angles[angle_idx]
        phi = self.torsions[torsion_idx]

        # Build local frame
        bc = p3 - p2
        bc = bc / torch.linalg.norm(bc, dim=-1, keepdim=True).clamp(min=1e-10)
        ab = p2 - p1
        n = torch.linalg.cross(ab, bc)
        n = n / torch.linalg.norm(n, dim=-1, keepdim=True).clamp(min=1e-10)
        m = torch.linalg.cross(n, bc)

        # Compute positions
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

    def _place_side_chains_batched(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Place side chain atoms using batched depth processing.

        Optimizes by using pre-computed indices and a helper function.

        Parameters
        ----------
        xyz : torch.Tensor
            Current coordinates with backbone already placed.

        Returns
        -------
        torch.Tensor
            Updated coordinates with side chains placed.
        """
        # Use the original per-depth lists which are already optimally pre-computed
        for sc_depth_idx in range(self._max_side_chain_depth):
            atom_idx = self._sc_depth_atom_indices[sc_depth_idx]
            if atom_idx.numel() == 0:
                continue

            xyz = self._place_atoms_nerf(
                xyz, atom_idx,
                self._sc_depth_parent_indices[sc_depth_idx],
                self._sc_depth_gp_indices[sc_depth_idx],
                self._sc_depth_ggp_indices[sc_depth_idx],
                self._sc_depth_bond_idx[sc_depth_idx],
                self._sc_depth_angle_idx[sc_depth_idx],
                self._sc_depth_torsion_idx[sc_depth_idx]
            )

        return xyz

    def forward_parallel(self) -> torch.Tensor:
        """
        Reconstruct Cartesian xyz using parallel scan for backbone.

        This is an optimized forward pass that:
        1. Places backbone atoms using parallel prefix scan (O(log N) steps)
        2. Places side chain atoms using depth iterations (O(max_sc_depth) steps)

        For deep trees where backbone is long but side chains are short,
        this can be significantly faster than the standard forward().

        Returns
        -------
        torch.Tensor
            Reconstructed Cartesian coordinates of shape (N, 3).
        """
        # Build backbone data if not already done
        if not hasattr(self, '_backbone'):
            self._build_backbone_data()

        xyz = torch.zeros(
            self.n_atoms, 3, dtype=self._dtype, device=self._device
        )

        backbone = self._backbone
        n_backbone = len(backbone)

        # === Phase 1: Place first 3 backbone atoms ===
        # Use standard placement for atoms at depth 0, 1, 2
        xyz[self.chain_roots] = self.chain_positions
        xyz = self._place_second_atoms(xyz)
        xyz = self._place_third_atoms(xyz)

        # === Phase 2: Place remaining backbone atoms via parallel scan ===
        if n_backbone > 3:
            # Get internal coords for backbone atoms at depth >= 3
            d = self.bond_lengths[self._backbone_bond_idx_tensor]
            theta = self.angles[self._backbone_angle_idx_tensor]
            phi = self.torsions[self._backbone_torsion_idx_tensor]

            # Compute delta transforms
            R_deltas, t_deltas = self._compute_delta_transform(d, theta, phi)

            # Parallel scan to get cumulative transforms
            R_cum, t_cum = self._parallel_scan(R_deltas, t_deltas)

            # Initial frame at backbone atom 2 (parent of first depth-3 backbone atom)
            p0 = xyz[backbone[0]]
            p1 = xyz[backbone[1]]
            p2 = xyz[backbone[2]]

            bc_init = (p2 - p1) / torch.linalg.norm(p2 - p1).clamp(min=1e-10)
            ab_init = p1 - p0
            n_init = torch.linalg.cross(ab_init, bc_init)
            n_init = n_init / torch.linalg.norm(n_init).clamp(min=1e-10)
            m_init = torch.linalg.cross(n_init, bc_init)

            R_init = torch.stack([m_init, n_init, bc_init], dim=-1)
            t_init = p2

            # Place backbone atoms 3, 4, ..., n_backbone-1 (vectorized)
            # Position = R_init @ t_cum[i+1] + t_init for all i at once
            local_positions = t_cum[1:n_backbone - 2]  # (n_backbone-3, 3)
            global_positions = (R_init @ local_positions.T).T + t_init  # (n_backbone-3, 3)
            xyz[backbone[3:]] = global_positions

        # === Phase 3: Place side chain atoms at depth 1, 2 (special handling) ===
        # These are atoms with sc_depth == 1 or 2 that need the standard method
        # for atoms at tree depth 1 or 2
        # Actually, _place_second_atoms and _place_third_atoms handle all depth-1 and depth-2
        # atoms regardless of whether they're on backbone or side chains.
        # So we just need to handle side chain atoms at depth >= 3.

        # === Phase 4: Place side chain atoms by SC-depth ===
        # Use batched placement for efficiency
        xyz = self._place_side_chains_batched(xyz)

        # === Phase 5: Place rigid rings ===
        xyz = self._place_rigid_rings(xyz)

        # === Phase 6: Apply frozen coordinates ===
        # For atoms that are not refinable (frozen), use fixed_xyz
        if not self.refinable_mask.all():
            frozen_mask = ~self.refinable_mask
            xyz[frozen_mask] = self.fixed_xyz[frozen_mask]

        return xyz
