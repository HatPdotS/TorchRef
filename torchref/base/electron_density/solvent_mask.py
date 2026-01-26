"""
Solvent mask generation functions.

Functions for creating solvent masks and identifying void regions
in crystallographic unit cells.
"""

import numpy as np
import torch

from torchref.config import dtypes
from torchref.base.coordinates.periodic_boundary import smallest_diff
from .map_building import scatter_add_nd


def add_to_solvent_mask(
    surrounding_coords, voxel_indices, mask, xyz, radius, inv_frac_matrix, frac_matrix
):
    """
    Create solvent mask by placing spheres around atom positions.

    Parameters
    ----------
    surrounding_coords : torch.Tensor
        Coordinates of voxels around each atom of shape (N_atoms, N_voxels, 3).
    voxel_indices : torch.Tensor
        Indices of voxels in the map of shape (N_atoms, N_voxels, 3).
    mask : torch.Tensor
        Solvent mask to be updated of shape (nx, ny, nz).
    xyz : torch.Tensor
        Atom positions of shape (N_atoms, 3).
    radius : float
        Radius of the sphere around each atom in Angstroms.
    inv_frac_matrix : torch.Tensor
        Inverse fractionalization matrix of shape (3, 3).
    frac_matrix : torch.Tensor
        Fractionalization matrix of shape (3, 3).

    Returns
    -------
    torch.Tensor
        Updated solvent mask as boolean tensor.
    """
    mask = mask.to(dtype=dtypes.int)
    # Calculate squared distances with periodic boundary conditions
    diff_coords_squared = smallest_diff(
        surrounding_coords - xyz.unsqueeze(1), inv_frac_matrix, frac_matrix
    )

    # Create boolean mask where distance squared is less than radius squared
    within_sphere = diff_coords_squared <= radius**2  # (N_atoms, N_voxels)

    # Convert boolean to float for addition
    values_to_add = within_sphere.to(dtype=mask.dtype).flatten()
    voxel_indices_flat = voxel_indices.reshape(-1, 3).to(dtypes.int)

    # Add to mask
    mask = scatter_add_nd(values_to_add, voxel_indices_flat, mask)

    # Ensure mask is binary (0 or 1)
    mask = torch.clamp(mask, max=1.0)

    return mask.to(torch.bool)


def add_to_phenix_mask(
    surrounding_coords,
    voxel_indices,
    xyz,
    vdw_radii,
    solvent_radius,
    inv_frac_matrix,
    frac_matrix,
    grid_shape,
    device,
):
    """
    Create Phenix-style three-valued mask by placing spheres around atom positions.

    This is a vectorized implementation that processes all atoms and voxels at once.
    Creates two binary masks:
    - protein_mask: 1 where inside VdW radius (protein core)
    - boundary_mask: 1 where between VdW and VdW+solvent_radius (accessible surface)

    Final three-valued mask:
    - 0: protein_mask == 1 (protein core)
    - -1: boundary_mask == 1 and protein_mask == 0 (accessible surface)
    - 1: both masks == 0 (bulk solvent)

    Parameters
    ----------
    surrounding_coords : torch.Tensor
        Fractional coordinates of voxels around each atom of shape (N_atoms, N_voxels, 3).
    voxel_indices : torch.Tensor
        Grid indices of voxels in the map of shape (N_atoms, N_voxels, 3).
    xyz : torch.Tensor
        Atom positions in fractional coordinates of shape (N_atoms, 3).
    vdw_radii : torch.Tensor
        VdW radius for each atom in Angstroms of shape (N_atoms,).
    solvent_radius : float
        Probe radius in Angstroms (added to VdW to get accessible surface).
    inv_frac_matrix : torch.Tensor
        Inverse fractional matrix for distance calculations of shape (3, 3).
    frac_matrix : torch.Tensor
        Fractional matrix for distance calculations of shape (3, 3).
    grid_shape : tuple
        Shape of the output mask (nx, ny, nz).
    device : torch.device
        Device for tensor operations.

    Returns
    -------
    protein_mask : torch.Tensor
        Boolean mask for protein core of shape grid_shape.
    boundary_mask : torch.Tensor
        Boolean mask for accessible surface of shape grid_shape.
    """
    # Calculate distances for all atom-voxel pairs
    diff = surrounding_coords - xyz.unsqueeze(1)
    diff_coords_squared = smallest_diff(diff, inv_frac_matrix, frac_matrix)
    distances = torch.sqrt(diff_coords_squared)  # (N_atoms, N_voxels)

    # Expand VdW radii for broadcasting
    vdw_radii_expanded = vdw_radii.unsqueeze(1)  # (N_atoms, 1)
    r_cutoff = vdw_radii_expanded + solvent_radius  # (N_atoms, 1)

    # Create two binary classifications
    in_protein_core = distances < vdw_radii_expanded  # (N_atoms, N_voxels)
    in_accessible_surface = (distances >= vdw_radii_expanded) & (
        distances < r_cutoff
    )  # (N_atoms, N_voxels)

    # Flatten for scatter operations
    voxel_indices_flat = voxel_indices.reshape(-1, 3).to(torch.long)

    # Create protein core mask using scatter_add
    int_dtype = dtypes.int
    protein_mask = torch.zeros(grid_shape, dtype=int_dtype, device=device)
    protein_values = in_protein_core.flatten().to(dtype=int_dtype)
    protein_mask = scatter_add_nd(protein_values, voxel_indices_flat, protein_mask)
    protein_mask = protein_mask > 0  # Convert to binary: True where protein core

    # Create accessible surface (boundary) mask using scatter_add
    boundary_mask = torch.zeros(grid_shape, dtype=int_dtype, device=device)
    boundary_values = in_accessible_surface.flatten().to(dtype=int_dtype)
    boundary_mask = scatter_add_nd(boundary_values, voxel_indices_flat, boundary_mask)
    boundary_mask = (
        boundary_mask > 0
    )  # Convert to binary: True where accessible surface

    return protein_mask, boundary_mask


def find_solvent_voids(mask, periodic=True):
    """
    Identify void regions in a 3D boolean tensor using connected component analysis.

    A void is defined as a connected region of False values (solvent). With periodic
    boundary conditions, voids can wrap around the edges of the array (like in a
    crystallographic unit cell). Without periodic boundaries, only enclosed voids
    are detected.

    Parameters
    ----------
    mask : torch.Tensor or numpy.ndarray
        Boolean tensor of shape (nx, ny, nz) where True indicates solid regions
        (e.g., protein) and False indicates empty regions (e.g., solvent).
        Can be either PyTorch tensor or NumPy array.
    periodic : bool, optional
        If True, apply periodic boundary conditions (voids can wrap around edges).
        If False, only detect voids that are completely enclosed and don't touch
        the boundaries. Default is True.

    Returns
    -------
    dict
        Dictionary where keys are int volumes (number of voxels) of each void in
        the original array, and values are boolean masks (torch.Tensor or
        numpy.ndarray) of same shape as input with True only for that specific
        void region. Returns an empty dict if no voids are found.

    Examples
    --------
    ::

        import torch
        # Create a simple 5x5x5 grid with a void in the center
        mask = torch.ones(5, 5, 5, dtype=torch.bool)
        mask[2, 2, 2] = False  # Single void voxel
        voids = find_solvent_voids(mask)
        print(voids)
    {1: tensor([[[False, False, ...]], dtype=torch.bool)}

    Notes
    -----
    - Uses scipy.ndimage.label for connected component analysis.
    - Connectivity is 26-connected (face, edge, and corner neighbors).
    - With periodic=True, the array is padded by wrapping to detect cross-boundary voids.
    - Performance is O(n) where n is the total number of voxels.
    - With periodic boundaries, large percolating voids are still detected.
    """
    from scipy import ndimage

    # Check if input is torch tensor or numpy array
    is_torch = isinstance(mask, torch.Tensor)

    if is_torch:
        # Convert to numpy for scipy processing
        device = mask.device
        dtype = mask.dtype
        mask_np = mask.cpu().numpy()
    else:
        mask_np = np.asarray(mask)
        dtype = mask.dtype

    # Get original shape
    original_shape = mask_np.shape
    nx, ny, nz = original_shape

    # Invert mask: we want to label the False regions (voids)
    inverted_mask = ~mask_np

    if periodic:
        # Apply periodic boundary conditions by padding with wrapped values
        # Pad by 1 on each side to allow connections across boundaries
        padded_mask = np.pad(inverted_mask, pad_width=1, mode="wrap")

        # Label connected components using 26-connectivity
        structure = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
        labeled_array, num_features = ndimage.label(padded_mask, structure=structure)

        # Extract the central region (original array size)
        # The padding helps connect voids across boundaries
        labeled_central = labeled_array[1:-1, 1:-1, 1:-1]

        # Now we need to identify which labels in the central region correspond to
        # the same void (they might have different labels in the padded array)
        # Create a mapping from labels in the central region to unique void IDs

        # For periodic boundaries, voids can wrap around
        # Check all 6 boundary pairs for potential wrapping
        label_equivalences = {}

        def add_equivalence(label1, label2):
            """Track that two labels are the same void."""
            if label1 == 0 or label2 == 0:  # Ignore background
                return
            if label1 == label2:
                return
            # Find root labels
            while label1 in label_equivalences:
                label1 = label_equivalences[label1]
            while label2 in label_equivalences:
                label2 = label_equivalences[label2]
            if label1 != label2:
                # Make label1 point to label2
                label_equivalences[label1] = label2

        # Check x-boundaries (x=0 and x=max-1)
        for j in range(ny):
            for k in range(nz):
                add_equivalence(labeled_central[0, j, k], labeled_central[-1, j, k])

        # Check y-boundaries (y=0 and y=max-1)
        for i in range(nx):
            for k in range(nz):
                add_equivalence(labeled_central[i, 0, k], labeled_central[i, -1, k])

        # Check z-boundaries (z=0 and z=max-1)
        for i in range(nx):
            for j in range(ny):
                add_equivalence(labeled_central[i, j, 0], labeled_central[i, j, -1])

        # Create final label mapping
        def find_root(label):
            """Find the root label for equivalence class."""
            if label == 0:
                return 0
            root = label
            while root in label_equivalences:
                root = label_equivalences[root]
            return root

        # Relabel the array with equivalence classes
        final_labeled = np.zeros_like(labeled_central)
        unique_labels = {}
        next_label = 1

        for idx in np.ndindex(labeled_central.shape):
            label = labeled_central[idx]
            if label == 0:
                continue
            root = find_root(label)
            if root not in unique_labels:
                unique_labels[root] = next_label
                next_label += 1
            final_labeled[idx] = unique_labels[root]

        labeled_array = final_labeled
        num_features = len(unique_labels)

    else:
        # Non-periodic: standard connected component labeling
        structure = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
        labeled_array, num_features = ndimage.label(inverted_mask, structure=structure)

    # If no features found, return empty dict
    if num_features == 0:
        return {}

    # Create dictionary to store voids
    voids_dict = {}

    # Process each labeled region
    for label_id in range(1, num_features + 1):
        # Create mask for this specific void
        void_mask = labeled_array == label_id

        if not periodic:
            # For non-periodic, check if void touches boundary
            touches_boundary = (
                np.any(void_mask[0, :, :])
                or np.any(void_mask[-1, :, :])
                or np.any(void_mask[:, 0, :])
                or np.any(void_mask[:, -1, :])
                or np.any(void_mask[:, :, 0])
                or np.any(void_mask[:, :, -1])
            )

            # Only include voids that don't touch the boundary
            if touches_boundary:
                continue

        # Calculate volume (number of voxels)
        volume = int(np.sum(void_mask))

        if volume == 0:
            continue

        # Convert back to torch tensor if input was torch
        if is_torch:
            void_mask_tensor = torch.from_numpy(void_mask).to(
                device=device, dtype=torch.bool
            )
        else:
            void_mask_tensor = void_mask

        # Store in dictionary
        # If multiple voids have the same volume, append a counter
        key = volume
        counter = 1
        original_key = key
        while key in voids_dict:
            key = f"{original_key}_{counter}"
            counter += 1

        voids_dict[key] = void_mask_tensor

    return voids_dict
