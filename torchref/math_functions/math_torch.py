import torch
from torchref.math_functions import math_numpy as math_np
import numpy as np
import hashlib

def cartesian_to_fractional_torch(cart_coords, unit_cell):
    B_inv = math_np.get_inv_fractional_matrix(unit_cell)
    B_inv = torch.tensor(B_inv)
    fractional_vector = torch.einsum('ik,kj->ij',cart_coords,B_inv.T)
    return fractional_vector

def fractional_to_cartesian_torch(fractional_coords, unit_cell):
    B = math_np.get_fractional_matrix(unit_cell)
    B = torch.tensor(B,dtype=fractional_coords.dtype,device=fractional_coords.device)
    cart_coords = torch.einsum('ik,kj->ij',fractional_coords,B.T)
    return cart_coords

def get_real_grid(cell,max_res=0.8,gridsize=None,device='cpu'):
    if isinstance(gridsize, torch.Tensor):
        nsteps = gridsize.to(torch.int32).to(device)
    elif gridsize is not None:
        nsteps = torch.tensor(gridsize,dtype=torch.int32,device=device)
    else:
        nsteps = torch.floor(cell[:3] / max_res * 3).to(torch.int32).to(device)
    # Place grid points at grid edges: i / N (CCTBX convention)
    # This matches how CCTBX/gemmi create maps
    x = torch.arange(nsteps[0],device=device) / nsteps[0]
    y = torch.arange(nsteps[1],device=device) / nsteps[1]
    z = torch.arange(nsteps[2],device=device) / nsteps[2]
    x, y, z = torch.meshgrid(x, y, z, indexing='ij')
    array_shape = x.shape
    x = x.reshape((*x.shape, 1))
    y = y.reshape((*y.shape, 1))
    z = z.reshape((*z.shape, 1))
    xyz = torch.cat((x, y, z), axis=3).reshape(-1, 3)
    xyz_real_grid = fractional_to_cartesian_torch(xyz, cell)
    xyz_real_grid = xyz_real_grid.reshape((*array_shape, 3))
    return xyz_real_grid

def find_grid_size(cell: torch.Tensor, max_res: float):
    return torch.floor(cell[:3] / max_res * 2.3).to(torch.int32)

def rotate_coords_torch(coords,phi,rho):
    phi = phi * np.pi / 180
    rho = rho * np.pi / 180
    rot_matrix = torch.tensor([[torch.cos(phi),-torch.sin(phi),0],
                               [torch.sin(phi)*torch.cos(rho),torch.cos(phi)*torch.cos(rho),-torch.sin(rho)],
                               [torch.sin(phi)*torch.sin(rho),torch.cos(phi)*torch.sin(rho),torch.cos(rho)]],dtype=torch.float64)
    return torch.einsum('ij,kj->ki',rot_matrix,coords)

def get_rfactor_torch(fobs,fcalc):
    fobs = torch.abs(fobs)
    fcalc = torch.abs(fcalc)
    return torch.sum(torch.abs(fobs - fcalc)) / torch.sum(fobs)

def calc_outliers(fobs,fcalc,z):
    fobs = torch.abs(fobs)
    fcalc = torch.abs(fcalc)
    diff = torch.abs(fobs - fcalc) / fobs
    std = torch.std(diff)
    outliers = diff >  z * std
    return outliers

def apply_transformation(points, transformation_matrix):
    """Apply 4x4 transformation matrix to 3D points"""
    # Convert to homogeneous coordinates
    homo_points = torch.hstack((points, torch.ones((points.shape[0], 1),device=points.device)))
    last_row = torch.tensor([0,0,0,1],device=points.device)
    transformation_matrix = torch.vstack((transformation_matrix,last_row))
    # Apply transformation
    transformed = torch.matmul(homo_points, transformation_matrix.T)
    # Return 3D coordinates
    return transformed[:, :3]

def core_deformation(core_correction,s):
    return 1.0 - core_correction * torch.exp(-s*s/0.5)

def aniso_structure_factor_torched(hkl,s_vector,fractional_coords,occ,scattering_factors,U,space_group):
    fractional_coords = space_group(fractional_coords.T)
    fractional_shape = fractional_coords.shape
    fractional_coords = fractional_coords.reshape(3,-1)
    dot_product = torch.matmul(hkl.to(torch.float64), fractional_coords).reshape(hkl.shape[0],fractional_shape[1],-1)
    U_row1 = torch.stack([U[:,0],U[:,3], U[:,4]],dim=0)
    U_row2 = torch.stack([U[:,3], U[:,1], U[:,5]],dim=0)
    U_row3 = torch.stack([U[:,4], U[:,5], U[:,2]],dim=0)
    U_matrix = torch.stack([U_row1,U_row2,U_row3],dim=0)
    U_dot_s = torch.einsum('jik,li->jkl', U_matrix, s_vector)  # Shape (3, M, N)
    StUS = torch.einsum('li,ikl->lk', s_vector, U_dot_s)  # Shape (M, N)
    B = -2 * (np.pi**2) * StUS 
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    pidot = 2 * np.pi * dot_product
    sin_cos = torch.sum(1j * torch.sin(pidot) + torch.cos(pidot),axis=-1)
    return torch.sum(terms * sin_cos, axis=(1))

def iso_structure_factor_torched(hkl,s,fractional_coords,occ,scattering_factors,tempfactor,space_group):
    fractional_coords = space_group(fractional_coords.T)
    fractional_shape = fractional_coords.shape
    fractional_coords = fractional_coords.reshape(3,-1)
    dot_product = torch.matmul(hkl.to(torch.float64), fractional_coords).reshape(hkl.shape[0],fractional_shape[1],-1)
    tempfactor = tempfactor.reshape(1,-1)
    s = s.reshape(-1,1)
    B = -tempfactor * (s ** 2) / 4
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    pidot = 2 * np.pi * dot_product
    sin_cos = torch.sum(1j * torch.sin(pidot) + torch.cos(pidot),axis=-1)
    return torch.sum(terms * sin_cos, axis=(1))

def superpose_vectors_robust_torch(ref_coords, mov_coords, weights=None, max_iterations=10):
    if weights is None:
        weights = torch.ones((ref_coords.shape[0],1), device=ref_coords.device)
    weights = weights / torch.sum(weights)

    mobile_coords_current = mov_coords.clone()
    best_matrix = torch.eye(4, device=mobile_coords_current.device, dtype=mobile_coords_current.dtype)
    best_rmsd = torch.tensor(float('inf'))
    for iteration in range(max_iterations):
        # Calculate centroids
        target_centroid = torch.sum(weights * ref_coords, axis=0)
        mobile_centroid = torch.sum(weights * mobile_coords_current, axis=0)
        
        # Center coordinates
        target_centered = ref_coords - target_centroid
        mobile_centered = mobile_coords_current - mobile_centroid
        
        # Calculate the covariance matrix with weights
        covariance = torch.zeros((3, 3),dtype=mobile_coords_current.dtype, device=mobile_coords_current.device)
        for i in range(len(weights)):
            covariance += weights[i] * torch.outer(mobile_centered[i], target_centered[i])
        
        # SVD of covariance matrix
    
        U, S, Vt = torch.linalg.svd(covariance)
        
        # Check for reflection case (determinant < 0)
        det = torch.linalg.det(torch.matmul(Vt.T, U.T))
        correction = torch.eye(3, dtype=mobile_coords_current.dtype, device=mobile_coords_current.device)
        if det < 0:
            correction[2, 2] = -1
            
        # Calculate rotation matrix
        rotation_matrix = torch.matmul(torch.matmul(Vt.T, correction), U.T)
        
        # FIXED: Calculate translation correctly
        # The correct way is: translation = target_centroid - (rotation_matrix @ mobile_centroid)
        # In NumPy notation with row vectors, this is:
        rotated_mobile_centroid = torch.matmul(mobile_centroid, rotation_matrix.T)
        translation = target_centroid - rotated_mobile_centroid
        
        # Compute 4x4 transformation matrix
        transformation_matrix = torch.zeros((3,4), device=mobile_coords_current.device, dtype=mobile_coords_current.dtype)
        transformation_matrix[:, :3] = rotation_matrix
        transformation_matrix[:, 3] = translation
        
        # Apply transformation and calculate RMSD
        # Using the correct transformation application
        mobile_transformed = torch.matmul(mov_coords, rotation_matrix.T) + translation
        
        squared_diffs = torch.sum((ref_coords - mobile_transformed)**2, axis=1)
        rmsd = torch.sqrt(torch.sum(weights * squared_diffs))
        
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_matrix = transformation_matrix
        # Update mobile coords for next iteration if doing iterative refinement
        if max_iterations > 1:
            mobile_coords_current = mobile_transformed
        return best_matrix
    
def align_torch(xyz1,xyz2,idx_to_move=None):
    if idx_to_move is not None:
        transformation_matrix1 = superpose_vectors_robust_torch(xyz1[idx_to_move],xyz2[idx_to_move])
    else:
        transformation_matrix1 = superpose_vectors_robust_torch(xyz1,xyz2)
    transformation_matrix = transformation_matrix1
    xyz_moved = apply_transformation(xyz2, transformation_matrix)
    return xyz_moved

def smallest_diff(diff: torch.Tensor,inv_frac_matrix: torch.Tensor,frac_matrix: torch.Tensor):
    diff_shape = diff.shape
    diff = diff.reshape(-1,3)
    diff_frac = torch.matmul(inv_frac_matrix,diff.T)
    translation = torch.round(diff_frac)
    diff = diff - torch.matmul(frac_matrix,translation).T
    return torch.sum(diff ** 2,axis=-1).reshape(diff_shape[:-1])

def smallest_diff_aniso(diff: torch.Tensor,inv_frac_matrix: torch.Tensor,frac_matrix: torch.Tensor):
    diff_shape = diff.shape
    diff = diff.reshape(-1,3)
    diff_frac = torch.matmul(inv_frac_matrix,diff.T)
    translation = torch.round(diff_frac)
    diff -= torch.matmul(frac_matrix,translation).T
    return torch.abs(diff).reshape(diff_shape)

def get_alignement_matrix(xyz1,xyz2,idx_to_move=None):
    if idx_to_move is not None:
        transformation_matrix = superpose_vectors_robust_torch(xyz1[idx_to_move],xyz2[idx_to_move])
    else:
        transformation_matrix = superpose_vectors_robust_torch(xyz1,xyz2)
    return transformation_matrix

def anharmonic_correction(hkl,c):
    h1, h2, h3 = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    # These third-order terms specifically address toroidal features
    C111, C222, C333, C112, C122, C113, C133, C223, C233, C123 = c
    # For toroidal features around z-axis, C111 and C222 are most important
    third_order = (
        C111 * h1**3 + 
        C222 * h2**3 + 
        C333 * h3**3 +
        3 * C112 * h1**2 * h2 +
        3 * C122 * h1 * h2**2 +
        3 * C113 * h1**2 * h3 +
        3 * C133 * h1 * h3**2 +
        3 * C223 * h2**2 * h3 +
        3 * C233 * h2 * h3**2 +
        6 * C123 * h1 * h2 * h3
    ) * (-8j * torch.pi**3) / 6e7
    return torch.exp(third_order)

def aniso_structure_factor_torched_no_complex(hkl,s_vector,fractional_coords,occ,scattering_factors,U,space_group):
    fractional_coords = space_group(fractional_coords.T)
    fractional_shape = fractional_coords.shape
    fractional_coords = fractional_coords.reshape(3,-1)
    dot_product = torch.matmul(hkl.to(torch.float64), fractional_coords).reshape(hkl.shape[0],fractional_shape[1],-1)
    U_row1 = torch.stack([U[:,0],U[:,3], U[:,4]],dim=0)
    U_row2 = torch.stack([U[:,3], U[:,1], U[:,5]],dim=0)
    U_row3 = torch.stack([U[:,4], U[:,5], U[:,2]],dim=0)
    U_matrix = torch.stack([U_row1,U_row2,U_row3],dim=0)
    U_dot_s = torch.einsum('jik,li->jkl', U_matrix, s_vector)  # Shape (3, M, N)
    StUS = torch.einsum('li,ikl->lk', s_vector, U_dot_s)  # Shape (M, N)
    B = -2 * (np.pi**2) * StUS 
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    pidot = 2 * np.pi * dot_product
    complex = torch.sum(torch.sum(torch.sin(pidot),axis=-1) * terms,axis=1)
    real = torch.sum(torch.sum(torch.cos(pidot),axis=-1) * terms,axis=1)
    return torch.vstack((real,complex))

def iso_structure_factor_torched_no_complex(hkl,s,fractional_coords,occ,scattering_factors,tempfactor,space_group):
    fractional_coords = space_group(fractional_coords.T)
    fractional_shape = fractional_coords.shape
    fractional_coords = fractional_coords.reshape(3,-1)
    dot_product = torch.matmul(hkl.to(torch.float64), fractional_coords).reshape(hkl.shape[0],fractional_shape[1],-1)
    tempfactor = tempfactor.reshape(1,-1)
    s = s.reshape(-1,1)
    B = -tempfactor * (s ** 2) / 4
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    pidot = 2 * np.pi * dot_product
    complex = torch.sum(torch.sum(torch.sin(pidot),axis=-1) * terms,axis=1)
    real = torch.sum(torch.sum(torch.cos(pidot),axis=-1) * terms,axis=1)
    return torch.vstack((real,complex))

def anharmonic_correction_no_complex(hkl,c):
    h1, h2, h3 = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    # These third-order terms specifically address toroidal features
    C111, C222, C333, C112, C122, C113, C133, C223, C233, C123 = c
    # For toroidal features around z-axis, C111 and C222 are most important
    third_order = (
        C111 * h1**3 + 
        C222 * h2**3 + 
        C333 * h3**3 +
        3 * C112 * h1**2 * h2 +
        3 * C122 * h1 * h2**2 +
        3 * C113 * h1**2 * h3 +
        3 * C133 * h1 * h3**2 +
        3 * C223 * h2**2 * h3 +
        3 * C233 * h2 * h3**2 +
        6 * C123 * h1 * h2 * h3 
    ) *  (-8 * torch.pi**3) / 6e7
    return torch.vstack((torch.cos(third_order), torch.sin(third_order)))

def multiplication_quasi_complex_tensor(a,b):
    real_part = a[0] * b[0] - a[1] * b[1]
    imag_part = a[0] * b[1] + a[1] * b[0]
    return torch.vstack((real_part, imag_part))

def french_wilson_conversion(Iobs, sigma_I=None):
    """
    Convert intensities to structure factor amplitudes using French-Wilson method
    Also converts standard deviations
    
    Parameters:
    -----------
    Iobs : torch.Tensor
        Observed intensity values
    sigma_I : torch.Tensor, optional
        Estimated standard deviations of intensities
        
    Returns:
    --------
    tuple (torch.Tensor, torch.Tensor)
        Structure factor amplitudes and their standard deviations
    """
    # If no sigmas provided, estimate them
    if sigma_I is None:
        sigma_I = torch.sqrt(torch.clamp(Iobs, min=1e-6))
    
    # Determine mean intensity for Wilson prior
    mean_I = torch.mean(torch.clamp(Iobs[~torch.isnan(Iobs)], min=0))
    
    # Strong reflections: simple square root
    strong_mask = Iobs > 3.0 * sigma_I
    F = torch.zeros_like(Iobs)
    F[strong_mask] = torch.sqrt(Iobs[strong_mask])
    
    # Weak/negative reflections: Bayesian approach
    weak_mask = ~strong_mask
    if weak_mask.any():
        # For weak reflections, use Bayesian estimate
        I_weak = Iobs[weak_mask]
        sigma_weak = sigma_I[weak_mask]
        
        # For negative intensities, we need a better prior estimate
        # The global mean_I is biased toward strong reflections
        # For negative I, use the uncertainty as a guide for the expected intensity
        # Better prior: use sigma_I as a proxy for the true intensity scale
        
        # Separate negative and weak positive
        neg_local_mask = I_weak < 0
        pos_local_mask = ~neg_local_mask
        
        F_weak = torch.zeros_like(I_weak)
        
        # For negative intensities: use sigma_I based correction
        # The idea: if I < 0, the true intensity is likely ~sigma_I in magnitude
        # So use a prior based on sigma_I rather than the global mean_I
        if neg_local_mask.any():
            I_neg = I_weak[neg_local_mask]
            sigma_neg = sigma_weak[neg_local_mask]
            
            # For negative intensities, use a correction proportional to sigma^2
            # This gives F values that scale with the uncertainty
            # Formula: F ≈ sqrt(sigma^2 / 2) for very negative
            # Blend with the global prior for moderately negative
            
            # Weight based on how negative: |I/sigma|
            epsilon = torch.abs(I_neg / torch.clamp(sigma_neg, min=1e-10))
            
            # For slightly negative (epsilon < 0.5): use small correction
            # For very negative (epsilon > 1): use sigma-based prior
            # Smooth transition with tanh
            weight_sigma_prior = torch.tanh(epsilon)
            
            # Sigma-based correction (for very negative)
            F_sigma_prior = torch.sqrt(sigma_neg**2 / 2.0)
            
            # Global prior correction (for slightly negative)
            wilson_param_global = mean_I / 2.0
            correction_global = sigma_neg**2 / (2.0 * wilson_param_global)
            F_global = torch.sqrt(torch.clamp(I_neg + correction_global, min=0))
            
            # Blend
            F_weak[neg_local_mask] = weight_sigma_prior * F_sigma_prior + (1 - weight_sigma_prior) * F_global
        
        # For weak positive intensities: use standard correction
        if pos_local_mask.any():
            I_pos = I_weak[pos_local_mask]
            sigma_pos = sigma_weak[pos_local_mask]
            
            # Simplified Bayesian estimate (posterior mean)
            wilson_param = mean_I / 2.0
            variance_correction = sigma_pos**2 / (2.0 * wilson_param)
            F_weak[pos_local_mask] = torch.sqrt(torch.clamp(I_pos + variance_correction, min=0))
        
        F[weak_mask] = F_weak
    
    # Convert sigmas using error propagation formula
    # For F = sqrt(I), σ(F) = σ(I)/(2*F)
    # Avoid division by zero
    sigma_F = torch.zeros_like(sigma_I)
    nonzero_F = F > 1e-6
    sigma_F[nonzero_F] = sigma_I[nonzero_F] / (2.0 * F[nonzero_F])
    
    # For very weak reflections where F approaches zero,
    # use an upper bound approximation to avoid huge sigma values
    tiny_F = (F <= 1e-6) & (sigma_I > 0)
    if tiny_F.any():
        # Approximate using the typical Wilson distribution variance
        sigma_F[tiny_F] = torch.sqrt(wilson_param / 2.0)
    
    return F, sigma_F

def reciprocal_basis_matrix(unit_cell: torch.Tensor):
    # Extract unit cell parameters

    angles_rad =  torch.deg2rad(unit_cell[3:])
    # Compute real-space basis vectors
    angles_cos = torch.cos(angles_rad)
    cos_squared = angles_cos ** 2
    sin_gamma = torch.sin(angles_rad[2])
    volume = torch.sqrt(1 - cos_squared[0] - cos_squared[1] - cos_squared[2] + 2 * cos_squared[0] * cos_squared[1] * cos_squared[2])
    a_vec = torch.tensor([unit_cell[0], 0, 0], dtype=unit_cell.dtype,device=unit_cell.device)
    b_vec = torch.tensor([unit_cell[1] * angles_cos[2], unit_cell[1] * sin_gamma, 0], dtype=unit_cell.dtype,device=unit_cell.device)
    c_vec = torch.tensor([
        unit_cell[2] * angles_cos[1],
        unit_cell[2] * (angles_cos[0] - angles_cos[1] * angles_cos[2]) / sin_gamma,
        unit_cell[2] * volume / sin_gamma
    ],dtype=unit_cell.dtype,device=unit_cell.device)
    # Compute reciprocal basis vectors
    volume_real = torch.dot(a_vec, torch.linalg.cross(b_vec, c_vec))
    a_star = torch.linalg.cross(b_vec, c_vec) / volume_real
    b_star = torch.linalg.cross(c_vec, a_vec) / volume_real
    c_star = torch.linalg.cross(a_vec, b_vec) / volume_real
    # Assemble reciprocal basis matrix
    return torch.stack([a_star, b_star, c_star])

def get_scattering_vectors(hkl: torch.Tensor, unit_cell: torch.Tensor, recB=None):
    if recB is None:
        recB = reciprocal_basis_matrix(unit_cell)
    s = torch.matmul(hkl.to(unit_cell.dtype),recB)
    return s

def get_d_spacing(hkl: torch.Tensor, unit_cell: torch.Tensor, recB=None):
    s = get_scattering_vectors(hkl,unit_cell,recB)
    d_spacing = 1.0 / torch.linalg.norm(s,axis=1)
    return d_spacing


def find_relevant_voxels(real_space_grid, xyz, radius_angstrom=4, inv_frac_matrix=None):
    """
    Vectorized function to identify the surrounding voxels of atoms in a real space grid.
    
    Parameters:
    -----------
    real_space_grid : torch.Tensor, shape (nx, ny, nz, 3)
        Real space grid containing xyz coordinates at each grid point
    xyz : torch.Tensor, shape (N, 3) or (3,)
        Atom coordinates in real space (Cartesian coordinates)
    radius : int
        Half-size of the box (in voxels) around each atom.
        The box will have size (2*radius+1)³
    inv_frac_matrix : torch.Tensor, shape (3, 3), optional
        Matrix to convert Cartesian to fractional coordinates.
        Required for proper handling of non-orthogonal cells.
    
    Returns:
    --------
    tuple of:
        surrounding_coords : torch.Tensor, shape (N, R³, 3)
            Coordinates of surrounding voxels for each atom, where R = 2*radius+1
            Returns the real-space coordinates from the grid for each voxel
        voxel_indices_wrapped : torch.Tensor, shape (N, R³, 3)
            Wrapped voxel indices
    
    Note:
    -----
    Atom coordinates are NOT wrapped here - periodic boundary conditions are handled
    in smallest_diff() which finds the minimum image distance. We only wrap voxel indices
    to ensure they're valid array indices.
    """
    # Ensure xyz is 2D (N, 3)
    if xyz.ndim == 1:
        xyz = xyz.unsqueeze(0)
    
    grid_shape = torch.tensor(real_space_grid.shape[:3], device=xyz.device)
    
    # Get grid origin (first voxel corner)
    grid_origin = real_space_grid[0, 0, 0]
    
    # Convert atom positions to grid indices
    # For non-orthogonal cells, we must use fractional coordinates
    if inv_frac_matrix is not None:
        # Proper way: Cartesian -> Fractional -> Wrap to [0,1] -> Grid indices
        # This ensures atoms outside the unit cell are correctly wrapped
        xyz_frac = torch.matmul(inv_frac_matrix, xyz.T).T  # (N, 3)
        xyz_frac = xyz_frac % 1.0  # Wrap to [0, 1]
        center_idx = torch.round(xyz_frac * grid_shape.unsqueeze(0)).to(torch.int64)
    else:
        # Fallback for orthogonal cells (less accurate for non-orthogonal)
        voxelsize = real_space_grid[3, 3, 3] - real_space_grid[2, 2, 2]
        center_idx = torch.round((xyz - grid_origin.unsqueeze(0)) / voxelsize.unsqueeze(0)).to(torch.int64)
    
    voxel_indices_wrapped = excise_angstrom_radius_around_coord(real_space_grid, center_idx, radius_angstrom)
    
    # Extract coordinates from real_space_grid
    # For each atom, get all surrounding voxel coordinates
    surrounding_coords = real_space_grid[
        voxel_indices_wrapped[..., 0],
        voxel_indices_wrapped[..., 1],
        voxel_indices_wrapped[..., 2]
    ]
    
    return surrounding_coords, voxel_indices_wrapped

def excise_angstrom_radius_around_coord(real_space_grid, start_indices, radius_angstrom=4.0):
    """
    Vectorized function to identify the surrounding voxels of atoms in a real space grid.
    
    Parameters:
    -----------
    real_space_grid : torch.Tensor, shape (nx, ny, nz, 3)
        Real space grid containing xyz coordinates at each grid point
    xyz : torch.Tensor, shape (N, 3) or (3,)
        Atom coordinates in real space (Cartesian coordinates)
    radius : int
        Half-size of the box (in voxels) around each atom.
        The box will have size (2*radius+1)³
    inv_frac_matrix : torch.Tensor, shape (3, 3), optional
        Matrix to convert Cartesian to fractional coordinates.
        Required for proper handling of non-orthogonal cells.
    
    Returns:
    --------
    tuple of:
        voxel_indices_wrapped : torch.Tensor, shape (N, R³, 3)
            Wrapped voxel indices
    
    Note:
    -----
    Atom coordinates are NOT wrapped here - periodic boundary conditions are handled
    in smallest_diff() which finds the minimum image distance. We only wrap voxel indices
    to ensure they're valid array indices.
    """
    # Ensure xyz is 2D (N, 3)
    if start_indices.ndim == 1:
        start_indices = start_indices.unsqueeze(0)
    grid_shape = torch.tensor(real_space_grid.shape[:3], device=start_indices.device)
    # Get grid origin (first voxel corner)
    voxelsize = real_space_grid[3, 3, 3] - real_space_grid[2, 2, 2]

    min_box_radius = torch.ceil(radius_angstrom / torch.min(voxelsize)).to(torch.int32)

    gridx = torch.arange(-min_box_radius, min_box_radius + 1, device=start_indices.device)
    gridy = torch.arange(-min_box_radius, min_box_radius + 1, device=start_indices.device)
    gridz = torch.arange(-min_box_radius, min_box_radius + 1, device=start_indices.device)
    x, y, z = torch.meshgrid(gridx, gridy, gridz, indexing='ij')
    coords = torch.stack((x, y, z), dim=-1)

    distance_map = torch.sqrt(torch.sum((coords * voxelsize.unsqueeze(0)) ** 2, axis=-1))
    within_radius_mask = distance_map <= radius_angstrom
    local_offsets = coords[within_radius_mask]  # Shape: (N_voxels_within_radius, 3)

    voxel_indices = local_offsets.unsqueeze(0) + start_indices.unsqueeze(1)
    voxel_indices_wrapped = voxel_indices % grid_shape.unsqueeze(0).unsqueeze(0)
    return voxel_indices_wrapped

def vectorized_add_to_map(surrounding_coords, voxel_indices, map, xyz, b, inv_frac_matrix, frac_matrix, A, B,occ):
    """
    Add atoms to density map using ITC92 Gaussian parameterization.
    
    Parameters:
    -----------
    surrounding_coords : torch.Tensor, shape (N_atoms, N_voxels, 3)
        Coordinates of voxels around each atom
    voxel_indices : torch.Tensor, shape (N_atoms, N_voxels, 3)
        Indices of voxels in the map
    map : torch.Tensor, shape (nx, ny, nz)
        Electron density map
    xyz : torch.Tensor, shape (N_atoms, 3)
        Atom positions
    b : torch.Tensor, shape (N_atoms,)
        B-factors (thermal parameters) in Å²
    A : torch.Tensor, shape (N_atoms, 5)
        ITC92 amplitude coefficients for each atom
    B : torch.Tensor, shape (N_atoms, 5)
        ITC92 width coefficients (b parameters) in Å² for each atom
    inv_frac_matrix : torch.Tensor, shape (3, 3)
        Inverse fractionalization matrix
    frac_matrix : torch.Tensor, shape (3, 3)
        Fractionalization matrix
    occ: torch.Tensor, shape (N_atoms,)
        Occupancies for each atom
    """
    # Calculate squared distances with periodic boundary conditions
    # diff_coords shape: (N_atoms, N_voxels)

    diff_coords_squared = smallest_diff(surrounding_coords - xyz.unsqueeze(1), inv_frac_matrix, frac_matrix)
    
    B_total = ((B + b.unsqueeze(1)) / 4).clamp(min=1e-1)
    
    # Normalization constant: (π/B_total)^(3/2)
    normalization = (np.pi / B_total) ** 1.5
    
    # Scale amplitudes by occupancy and normalization
    A_normalized = A * occ.unsqueeze(1) * normalization
    
    # Calculate Gaussian with exponent: exp(-π²r²/B_total)
    # Note: diff_coords_squared already contains r²
    gaussian_terms = torch.exp(-(np.pi**2) * diff_coords_squared.unsqueeze(2) / B_total.unsqueeze(1))
    
    # Sum over the 4 Gaussian components
    density = torch.sum(A_normalized.unsqueeze(1) * gaussian_terms, dim=2)
    
    # Flatten to (N_atoms * N_voxels,)
    density_flat = density.flatten()
    voxel_indices_flat = voxel_indices.reshape(-1, 3)
    
    # Add to map
    map = scatter_add_nd(density_flat, voxel_indices_flat, map)
    return map

def vectorized_add_to_map_aniso(surrounding_coords, voxel_indices, map, xyz, U, inv_frac_matrix, frac_matrix, A, B, occ):
    """
    Add anisotropic atoms to density map using ITC92 Gaussian parameterization with anisotropic displacement.
    
    For anisotropic atoms, the Gaussian is:
    ρ(r) = Σᵢ Aᵢ * exp(-2π² * Δr^T * (U + Uᵢ) * Δr)
    
    where:
    - U is the atomic displacement parameter tensor (6 components: u11, u22, u33, u12, u13, u23)
    - Uᵢ is the ITC92 Gaussian width tensor derived from Bᵢ parameter
    - Δr is the distance vector from atom center
    
    Parameters:
    -----------
    surrounding_coords : torch.Tensor, shape (N_atoms, N_voxels, 3)
        Coordinates of voxels around each atom
    voxel_indices : torch.Tensor, shape (N_atoms, N_voxels, 3)
        Indices of voxels in the map
    map : torch.Tensor, shape (nx, ny, nz)
        Electron density map
    xyz : torch.Tensor, shape (N_atoms, 3)
        Atom positions in Cartesian coordinates
    U : torch.Tensor, shape (N_atoms, 6)
        Anisotropic displacement parameters in Å² (u11, u22, u33, u12, u13, u23)
        Note: These are already in Cartesian space from PDB
    inv_frac_matrix : torch.Tensor, shape (3, 3)
        Inverse fractionalization matrix
    frac_matrix : torch.Tensor, shape (3, 3)
        Fractionalization matrix
    A : torch.Tensor, shape (N_atoms, 4)
        ITC92 amplitude coefficients for each atom
    B : torch.Tensor, shape (N_atoms, 4)
        ITC92 width coefficients (b parameters) in Å² for each atom
    occ : torch.Tensor, shape (N_atoms,)
        Occupancies for each atom
        
    Returns:
    --------
    map : torch.Tensor
        Updated electron density map
    """
    # Calculate distance vectors with periodic boundary conditions
    # diff_coords shape: (N_atoms, N_voxels, 3)
    diff_coords = surrounding_coords - xyz.unsqueeze(1)
    
    diff_coords = smallest_diff_aniso(diff_coords, inv_frac_matrix, frac_matrix)
    
    # Convert ITC92 B parameters and atomic U to anisotropic formulation
    # 
    # Isotropic version uses: exp(-r² / (2σ²_total)) where σ²_total = σ²_ITC + σ²_atomic
    #   σ²_ITC = B/(4π²)
    #   σ²_atomic = U_PDB (since B-factor = 8π²*U, and σ²_B = B/(8π²) = U)
    #
    # Anisotropic version uses: exp(-2π² * r^T * U_total * r)
    # For isotropic limit: -2π² * u_total * r² = -r² / (2σ²_total)
    # Solving: u_total = 1 / (4π² σ²_total)
    #
    # KEY INSIGHT: We can't convert σ²_ITC and σ²_atomic separately and add!
    # Instead: u_total = 1 / (4π² * (σ²_ITC + σ²_atomic))
    #         = 1 / (4π² * (B/(4π²) + U_PDB))
    #         = 1 / (B + 4π²*U_PDB)
    #
    # For each ITC92 Gaussian component:
    # σ²_total = B[i]/(4π²) + U_PDB[j]
    # u_total[i,j] = 1 / (4π² * (B[i]/(4π²) + U_PDB[j]))
    #              = 1 / (B[i] + 4π²*U_PDB[j])
    #
    # B shape: (N_atoms, 4)
    # U shape: (N_atoms, 6) with format [u11, u22, u33, u12, u13, u23]
    
    # For diagonal terms of U, compute total u values
    # Need to broadcast: B is (N_atoms, 4), U diagonal is (N_atoms, 3)
    # Result should be (N_atoms, 4, 3) for u11, u22, u33 of each Gaussian component
    
    four_pi_sq = 4 * np.pi**2
    
    # Diagonal U terms
    U_diag = U[:, :3]  # (N_atoms, 3) - u11, u22, u33
    
    # Compute u_total for each combination of ITC92 component and U diagonal
    # u_total[atom, component, direction] = 1 / (B[atom, component] + 4π²*U[atom, direction])
    B_expanded = B.unsqueeze(2)  # (N_atoms, 4, 1)
    U_diag_expanded = U_diag.unsqueeze(1)  # (N_atoms, 1, 3)
    
    U_total_diag = 1.0 / (B_expanded + four_pi_sq * U_diag_expanded)  # (N_atoms, 4, 3)
    
    # Build U_total tensor with proper format [u11, u22, u33, u12, u13, u23]
    # Shape: (N_atoms, 4, 6)
    U_total = torch.zeros(B.shape[0], B.shape[1], 6, device=B.device, dtype=B.dtype)
    U_total[:, :, :3] = U_total_diag  # u11, u22, u33
    
    # For off-diagonal terms, use similar approach
    # But for simplicity, if they're close to zero, keep them zero
    # Otherwise would need to handle the full matrix inversion
    # For now: u12_total ≈ U12 / (4π²) as approximation
    U_off_diag = U[:, 3:]  # (N_atoms, 3) - u12, u13, u23
    U_total[:, :, 3:] = (U_off_diag.unsqueeze(1) / four_pi_sq).expand(-1, B.shape[1], -1)  # Approximate
    # U_total shape: (N_atoms, 4, 6)
    
    
    U_matrix = torch.zeros(U_total.shape[0], U_total.shape[1], 3, 3,
                           device=U_total.device, dtype=U_total.dtype)
    U_matrix[:, :, 0, 0] = U_total[:, :, 0]  # u11
    U_matrix[:, :, 1, 1] = U_total[:, :, 1]  # u22
    U_matrix[:, :, 2, 2] = U_total[:, :, 2]  # u33
    U_matrix[:, :, 0, 1] = U_total[:, :, 3]  # u12
    U_matrix[:, :, 1, 0] = U_total[:, :, 3]  # u12 (symmetric)
    U_matrix[:, :, 0, 2] = U_total[:, :, 4]  # u13
    U_matrix[:, :, 2, 0] = U_total[:, :, 4]  # u13 (symmetric)
    U_matrix[:, :, 1, 2] = U_total[:, :, 5]  # u23
    U_matrix[:, :, 2, 1] = U_total[:, :, 5]  # u23 (symmetric)
    # U_matrix shape: (N_atoms, 4, 3, 3)
    
    # Compute quadratic form: Δr^T * U * Δr for each Gaussian component
    # diff_coords: (N_atoms, N_voxels, 3) -> (N_atoms, N_voxels, 1, 3) for broadcasting
    # U_matrix: (N_atoms, 4, 3, 3) -> (N_atoms, 1, 4, 3, 3) for broadcasting
    diff_coords_expanded = diff_coords.unsqueeze(2)  # (N_atoms, N_voxels, 1, 3)
    U_matrix_expanded = U_matrix.unsqueeze(1)  # (N_atoms, 1, 4, 3, 3)
    
    # First: U * Δr -> (N_atoms, N_voxels, 4, 3)
    U_times_diff = torch.einsum('naijk,namk->naij', U_matrix_expanded, diff_coords_expanded)
    
    # Second: Δr^T * (U * Δr) -> (N_atoms, N_voxels, 4)
    quad_form = torch.einsum('namk,namk->nam', diff_coords_expanded, U_times_diff)
    
    # Apply occupancy scaling to amplitudes
    A_scaled = A * occ.unsqueeze(1)  # Shape: (N_atoms, 4)
    
    # Calculate Gaussian density for each component
    # quad_form shape: (N_atoms, N_voxels, 4)
    # Exponent: -2π² * quad_form
    gaussian_terms = torch.exp(-2 * np.pi**2 * quad_form)  # (N_atoms, N_voxels, 4)
    
    # Sum over 4 Gaussian components
    # A_scaled: (N_atoms, 4) -> (N_atoms, 1, 4) for broadcasting
    density = torch.sum(A_scaled.unsqueeze(1) * gaussian_terms, dim=2)  # (N_atoms, N_voxels)
    
    # Flatten to (N_atoms * N_voxels,)
    density_flat = density.flatten()
    voxel_indices_flat = voxel_indices.reshape(-1, 3)
    
    # Add to map
    map = scatter_add_nd(density_flat, voxel_indices_flat, map)
    return map

def scatter_add_nd_super_slow(source, index, map):
    for i in range(source.shape[0]):
        idx = tuple(index[i].tolist())
        map[idx] += source[i]
    return map

def scatter_add_nd(source, index, map):
    """
    Vectorized n-dimensional scatter add operation.
    
    Parameters:
    -----------
    source : torch.Tensor, shape (N,)
        Values to add to the map
    index : torch.Tensor, shape (N, ndim)
        Indices where values should be added
    map : torch.Tensor, shape (d1, d2, ..., dn)
        N-dimensional tensor to add values into
        
    Returns:
    --------
    map : torch.Tensor
        Modified map with values added
    """
    map_shape = torch.tensor(map.shape, device=index.device, dtype=torch.int64)
    
    # Convert n-dimensional indices to flat indices
    # For shape (d1, d2, d3, ..., dn), flat_index = i0 * (d1*d2*...*dn) + i1 * (d2*d3*...*dn) + ... + in
    strides = torch.ones(len(map_shape), device=index.device, dtype=torch.int64)
    for i in range(len(map_shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * map_shape[i + 1]
    
    index_flat = torch.sum(index * strides.unsqueeze(0), dim=-1)
    
    map_flat = map.view(-1)
    try:
        map_flat.scatter_add_(0, index_flat, source)
    except RuntimeError as e:
        print("Error during scatter_add_: ", e)
        print("Source shape: ", source.shape, 'device: ', source.device, 'dtype: ', source.dtype)
        print("Index shape: ", index.shape, 'device: ', index.device, 'dtype: ', index.dtype)
        print("Map shape: ", map.shape, 'device: ', map.device, 'dtype: ', map.dtype)
        raise e
    return map

def place_on_grid(hkls, structure_factor, grid_size, enforce_hermitian: bool = True) -> torch.Tensor:
    """
    Vectorized placement of batched structure factors on reciprocal-space grid.
    
    Args:
        enforce_hermitian: Whether to enforce Hermitian symmetry
        
    Returns:
        Complex tensor grid of structure factors
    """
    batch_mode = True
    if structure_factor.ndim == 1:
        structure_factor = structure_factor.unsqueeze(0)  # Add batch dimension
        batch_mode = False
    B = structure_factor.shape[0]
    device = structure_factor.device
    dtype = structure_factor.dtype
    Nx, Ny, Nz = [int(x) for x in grid_size]
    # Prepare Miller indices and linear indices
    hkls = hkls.to(device=device)
    h = hkls[:, 0].to(torch.int64)
    k = hkls[:, 1].to(torch.int64)
    l = hkls[:, 2].to(torch.int64)
    
    hi = torch.remainder(h, Nx)
    ki = torch.remainder(k, Ny)
    li = torch.remainder(l, Nz)
    lin = (hi * (Ny * Nz) + ki * Nz + li).to(torch.int64)  # (N,)
    # Vectorized scatter-add to grid
    grid = torch.zeros((B, Nx * Ny * Nz), dtype=dtype, device=device)
    grid = grid.index_add(1, lin, structure_factor)  # (B, Nx*Ny*Nz)
    
    if enforce_hermitian:
        hi_sym = torch.remainder(-h, Nx)
        ki_sym = torch.remainder(-k, Ny)
        li_sym = torch.remainder(-l, Nz)
        lin_sym = (hi_sym * (Ny * Nz) + ki_sym * Nz + li_sym).to(torch.int64)
        vals_conj = torch.conj(structure_factor)
        grid = grid.index_add(1, lin_sym, vals_conj)
        
    grid = grid.view(B, Nx, Ny, Nz)
    if not batch_mode:
        grid = grid.squeeze(0)
    return grid

def fft(reciprocal_grid) -> torch.Tensor:
    """
    Perform FFT to obtain real space electron density.
    
    Returns:
        Real-valued tensor of electron density
        
    Raises:
        ValueError: If grid not initialized
    """
    if reciprocal_grid.ndim == 4:
        rs = torch.fft.ifftn(reciprocal_grid, dim=(1, 2, 3)).real
        rs = torch.flip(rs, dims=(1, 2, 3))
        rs = torch.roll(rs, shifts=(1, 1, 1), dims=(1, 2, 3))
    else:
        rs = torch.fft.ifftn(reciprocal_grid, dim=(0, 1, 2)).real
        rs = torch.flip(rs, dims=(0, 1, 2))
        rs = torch.roll(rs, shifts=(1, 1, 1), dims=(0, 1, 2))
    return rs

def ifft(real_space_map) -> torch.Tensor:
    """
    Perform inverse FFT to obtain reciprocal space structure factors.
    
    Returns:
        Complex-valued tensor of structure factors
        
    Raises:
        ValueError: If grid not initialized
    """
    if real_space_map.ndim == 4:
        rg = torch.roll(real_space_map, shifts=(-1, -1, -1), dims=(1, 2, 3))
        rg = torch.flip(rg, dims=(1, 2, 3))
        rg = torch.fft.fftn(rg, dim=(1, 2, 3))
    else:
        rg = torch.roll(real_space_map, shifts=(-1, -1, -1), dims=(0, 1, 2))
        rg = torch.flip(rg, dims=(0, 1, 2))
        rg = torch.fft.fftn(rg, dim=(0, 1, 2))
    return rg

def extract_structure_factor_from_grid(reciprocal_grid, hkls) -> torch.Tensor:
    """
    Extract structure factors from reciprocal space grid at given Miller indices.
    
    Args:
        reciprocal_grid: Complex tensor of shape (Nx, Ny, Nz) or (B, Nx, Ny, Nz)
        hkls: Tensor of Miller indices of shape (N, 3)

    Returns:
        Tensor of structure factors of shape (N,) or (B, N,)

    """
    device = reciprocal_grid.device
    dtype = reciprocal_grid.dtype
    
    # Handle both batched and unbatched input
    if reciprocal_grid.ndim == 3:
        reciprocal_grid = reciprocal_grid.unsqueeze(0)  # Add batch dimension
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, Nx, Ny, Nz = reciprocal_grid.shape
    
    # Convert Miller indices to grid positions using same convention as place_on_grid
    hkls = hkls.to(device=device)
    h = hkls[:, 0].to(torch.int64)
    k = hkls[:, 1].to(torch.int64)
    l = hkls[:, 2].to(torch.int64)
    
    # Map to grid indices with periodic wrapping
    hi = torch.remainder(h, Nx)
    ki = torch.remainder(k, Ny)
    li = torch.remainder(l, Nz)
    
    # Extract structure factors at these positions
    # For batched: (B, Nx, Ny, Nz) -> (B, N)
    structure_factors = reciprocal_grid[:, hi, ki, li]  # (B, N)
    
    if squeeze_output:
        structure_factors = structure_factors.squeeze(0)  # (N,)
    
    return structure_factors

def add_to_solvent_mask(surrounding_coords, voxel_indices, mask, xyz, radius, inv_frac_matrix, frac_matrix):
    """
    Create solvent mask by placing spheres around atom positions.
    
    Parameters:
    -----------
    surrounding_coords : torch.Tensor, shape (N_atoms, N_voxels, 3)
        Coordinates of voxels around each atom
    voxel_indices : torch.Tensor, shape (N_atoms, N_voxels, 3)
        Indices of voxels in the map
    mask : torch.Tensor, shape (nx, ny, nz)
        Solvent mask to be updated
    xyz : torch.Tensor, shape (N_atoms, 3)
        Atom positions
    radius : float
        Radius of the sphere around each atom in Å
        
    Returns:
    --------
    mask : torch.Tensor
        Updated solvent mask
    """
    mask = mask.to(dtype=torch.int32)
    # Calculate squared distances with periodic boundary conditions
    diff_coords_squared = smallest_diff(surrounding_coords - xyz.unsqueeze(1), inv_frac_matrix, frac_matrix)
    
    # Create boolean mask where distance squared is less than radius squared
    within_sphere = diff_coords_squared <= radius**2  # (N_atoms, N_voxels)
    
    # Convert boolean to float for addition
    values_to_add = within_sphere.to(dtype=mask.dtype).flatten()
    voxel_indices_flat = voxel_indices.reshape(-1, 3).to(torch.int32)
    
    # Add to mask
    mask = scatter_add_nd(values_to_add, voxel_indices_flat, mask)
    
    # Ensure mask is binary (0 or 1)
    mask = torch.clamp(mask, max=1.0)
    
    return mask.to(torch.bool)

def add_to_phenix_mask(surrounding_coords, voxel_indices, xyz, vdw_radii, solvent_radius, 
                        inv_frac_matrix, frac_matrix, grid_shape, device):
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
    
    Parameters:
    -----------
    surrounding_coords : torch.Tensor, shape (N_atoms, N_voxels, 3)
        Fractional coordinates of voxels around each atom
    voxel_indices : torch.Tensor, shape (N_atoms, N_voxels, 3)
        Grid indices of voxels in the map
    xyz : torch.Tensor, shape (N_atoms, 3)
        Atom positions in fractional coordinates
    vdw_radii : torch.Tensor, shape (N_atoms,)
        VdW radius for each atom in Angstroms
    solvent_radius : float
        Probe radius in Angstroms (added to VdW to get accessible surface)
    inv_frac_matrix : torch.Tensor
        Inverse fractional matrix for distance calculations
    frac_matrix : torch.Tensor
        Fractional matrix for distance calculations
    grid_shape : tuple
        Shape of the output mask (nx, ny, nz)
    device : torch.device
        Device for tensor operations
        
    Returns:
    --------
    mask : torch.Tensor, dtype=torch.int8, shape grid_shape
        Three-valued mask {-1, 0, 1}
    """
    # Calculate distances for all atom-voxel pairs
    diff = (surrounding_coords - xyz.unsqueeze(1))
    diff_coords_squared = smallest_diff(diff, inv_frac_matrix, frac_matrix)
    distances = torch.sqrt(diff_coords_squared)  # (N_atoms, N_voxels)
    # Expand VdW radii for broadcasting
    vdw_radii_expanded = vdw_radii.unsqueeze(1)  # (N_atoms, 1)
    r_cutoff = vdw_radii_expanded + solvent_radius  # (N_atoms, 1)
    
    # Create two binary classifications
    in_protein_core = distances < vdw_radii_expanded  # (N_atoms, N_voxels)
    in_accessible_surface = (distances >= vdw_radii_expanded) & (distances < r_cutoff)  # (N_atoms, N_voxels)
    
    # Flatten for scatter operations
    voxel_indices_flat = voxel_indices.reshape(-1, 3).to(torch.long)
    
    # Create protein core mask using scatter_add
    protein_mask = torch.zeros(grid_shape, dtype=torch.int32, device=device)
    protein_values = in_protein_core.flatten().to(dtype=torch.int32)
    protein_mask = scatter_add_nd(protein_values, voxel_indices_flat, protein_mask)
    protein_mask = (protein_mask > 0)  # Convert to binary: True where protein core
    
    # Create accessible surface (boundary) mask using scatter_add
    boundary_mask = torch.zeros(grid_shape, dtype=torch.int32, device=device)
    boundary_values = in_accessible_surface.flatten().to(dtype=torch.int32)
    boundary_mask = scatter_add_nd(boundary_values, voxel_indices_flat, boundary_mask)
    boundary_mask = (boundary_mask > 0)  # Convert to binary: True where accessible surface
    
    return protein_mask, boundary_mask

def nll_xray(F_obs: torch.Tensor, F_calc: torch.Tensor, sigma_F_obs: torch.Tensor) -> torch.Tensor:
    # Compute amplitude of calculated structure factors
    F_calc_amp = torch.abs(F_calc)

    # Compute residual
    diff = F_obs - F_calc_amp
    # Avoid division by zero by setting a minimum sigma
    eps = torch.median(sigma_F_obs) * 1e-1
    # Compute Gaussian NLL: 0.5*(x-μ)²/σ² + log(σ) + 0.5*log(2π)
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi))
    sigma_save = torch.clamp(sigma_F_obs, min=eps)
    nll = 0.5 * (diff**2) / (sigma_save**2) + torch.log(sigma_save) + 0.5 * log_2pi

    return nll.mean()


def nll_xray_sum(F_obs: torch.Tensor, F_calc: torch.Tensor, sigma_F_obs: torch.Tensor) -> torch.Tensor:
    # Compute amplitude of calculated structure factors
    F_calc_amp = torch.abs(F_calc)

    # Compute residual
    diff = F_obs - F_calc_amp
    # Avoid division by zero by setting a minimum sigma
    eps = torch.median(sigma_F_obs) * 1e-1
    # Compute Gaussian NLL: 0.5*(x-μ)²/σ² + log(σ) + 0.5*log(2π)
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi))
    sigma_save = torch.clamp(sigma_F_obs, min=eps)
    nll = 0.5 * (diff**2) / (sigma_save**2) + torch.log(sigma_save) + 0.5 * log_2pi

    return nll.sum()

def log_loss(F_obs: torch.Tensor, F_calc: torch.Tensor, sigma_F_obs: torch.Tensor) -> torch.Tensor:
    # Compute amplitude of calculated structure factors
    F_calc_amp = torch.abs(F_calc)

    # Compute residual
    diff = torch.log(F_obs) - torch.log(F_calc_amp)
    return torch.mean(torch.abs(diff))

def estimate_sigma_I(I):
    """
    Estimate standard deviation of intensities, separating positive and negative values.
    """
    if torch.any(I < 0):
        neg_I_sig = torch.mean(I[I < 0] ** 2) ** 0.5
        sigma = I * 0.05 + neg_I_sig
    else:
        sigma = I * 0.05 + torch.mean(I) * 0.01
    return sigma

def estimate_sigma_F(F):
    """
    Estimate standard deviation of structure factor amplitudes.
    """
    sigma = F * 0.05 + torch.mean(F) * 0.01
    return sigma

def nll_xray_lognormal(F_obs: torch.Tensor, F_calc: torch.Tensor, 
                       sigma_F_obs: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute X-ray negative log-likelihood assuming lognormal distribution.
    
    This is a more realistic model for structure factor amplitudes, which must be positive.
    
    For a lognormal distribution LogNormal(μ, σ²), the NLL is:
    NLL = 0.5*(log(x) - μ)²/σ² + log(x) + log(σ) + 0.5*log(2π)
    
    Where μ and σ are derived from F_obs and sigma_F_obs using:
    - σ = √(log(1 + (sigma_F/F)²))
    - μ = log(F) - σ²/2
    
    Args:
        F_obs: Observed structure factor amplitudes
        F_calc: Calculated structure factors (complex)
        sigma_F_obs: Standard deviations of observed amplitudes
        eps: Small value to avoid numerical issues
        
    Returns:
        torch.Tensor: Mean negative log-likelihood
    """
    # Compute amplitude of calculated structure factors
    F_calc_amp = torch.abs(F_calc)
    
    # Ensure positive values
    F_obs_safe = torch.clamp(F_obs, min=eps)
    F_calc_safe = torch.clamp(F_calc_amp, min=eps)
    sigma_F_safe = torch.clamp(sigma_F_obs, min=eps)
    
    # Convert Gaussian parameters to lognormal parameters
    # For lognormal: CV² = exp(σ²) - 1, where CV = sigma_F/F
    CV = sigma_F_safe / F_obs_safe
    CV_squared = CV ** 2
    sigma_ln = torch.sqrt(torch.log1p(CV_squared))  # σ of lognormal
    
    # μ = log(F) - σ²/2
    mu_ln = torch.log(F_obs_safe) - 0.5 * sigma_ln ** 2
    
    # Lognormal NLL: 0.5*(log(x) - μ)²/σ² + log(x) + log(σ) + 0.5*log(2π)
    log_F_calc = torch.log(F_calc_safe)
    diff = log_F_calc - mu_ln
    
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=F_obs.device))
    nll = (0.5 * (diff**2) / (sigma_ln**2 + eps) + 
           log_F_calc + 
           torch.log(sigma_ln + eps) + 
           0.5 * log_2pi)
    
    # Mean over all reflections
    return nll.mean()

def U_to_matrix(U: torch.Tensor) -> torch.Tensor:
    """
    Convert anisotropic displacement parameters from 6-component vector to 3x3 matrix.
    
    Parameters:
    -----------
    U : torch.Tensor, shape (..., 6)
        Anisotropic displacement parameters in the order:
        [u11, u22, u33, u12, u13, u23]
        
    Returns:
    --------
    U_matrix : torch.Tensor, shape (..., 3, 3)
        Anisotropic displacement parameter matrices.
    """
    u11 = U[..., 0]
    u22 = U[..., 1]
    u33 = U[..., 2]
    u12 = U[..., 3]
    u13 = U[..., 4]
    u23 = U[..., 5]
    if U.ndim == 1:
        U_matrix = torch.zeros((3, 3), device=U.device, dtype=U.dtype)
    else:
        U_matrix = torch.zeros(U.shape[:-1] + (3, 3), device=U.device, dtype=U.dtype)
    U_matrix[..., 0, 0] = u11
    U_matrix[..., 1, 1] = u22
    U_matrix[..., 2, 2] = u33
    U_matrix[..., 0, 1] = u12
    U_matrix[..., 1, 0] = u12
    U_matrix[..., 0, 2] = u13
    U_matrix[..., 2, 0] = u13
    U_matrix[..., 1, 2] = u23
    U_matrix[..., 2, 1] = u23
    
    return U_matrix

def deterministic_tensor_digest(t: torch.Tensor, n_chunks: int = 16) -> torch.Tensor:
    """
    Compute a deterministic digest vector for tensor `t` directly on GPU.

    - Deterministic across devices and runs
    - Sensitive to all tensor values and order  
    - Fully vectorized with no Python loops
    - Suitable for large GPU tensors
    
    Uses a simple mean/std approach per chunk which is fully deterministic.
    """
    # Flatten and cast to a stable type
    flat = t.detach().reshape(-1)
    if not torch.is_floating_point(flat):
        flat = flat.float()

    # If tensor smaller than n_chunks, just pad
    n = flat.numel()
    if n < n_chunks:
        flat = torch.nn.functional.pad(flat, (0, n_chunks - n))
        n = n_chunks

    # Reshape into chunks directly (pad if needed)
    chunk_size = (n + n_chunks - 1) // n_chunks
    padded_size = chunk_size * n_chunks
    
    if n < padded_size:
        flat = torch.nn.functional.pad(flat, (0, padded_size - n))
    
    # Reshape to (n_chunks, chunk_size) and compute stats per chunk
    chunks = flat[:chunk_size * n_chunks].reshape(n_chunks, chunk_size)
    
    # Create digest from mean and std of each chunk (both are deterministic)
    # Interleave for better sensitivity
    digest_mean = chunks.mean(dim=1)
    digest_std = chunks.std(dim=1)
    
    # Combine into single digest vector (alternate mean/std would double size)
    # Instead, use weighted combination
    digest = digest_mean + 0.61803398875 * digest_std

    return digest

def hash_tensors(tensors) -> str:
    h = hashlib.sha1()
    for t in tensors:
        if t is None:
            h.update(b"<None>")
            continue
        digest = deterministic_tensor_digest(t)
        # Bring only digest (small) to CPU for hashing
        h.update(digest.cpu().numpy().tobytes())
        h.update(str(t.shape).encode())
        h.update(str(t.dtype).encode())
    return h.hexdigest()

def rfactor(F_obs: torch.Tensor, F_calc: torch.Tensor) -> float:
    """
    Calculate R-factor between observed and calculated structure factors.
    
    Parameters:
    -----------
    F_obs : torch.Tensor, shape (N,)
        Observed structure factor amplitudes
    F_calc : torch.Tensor, shape (N,)
        Calculated structure factor amplitudes
        
    Returns:
    --------
    r_factor : float
        R-factor value
    """
    numerator = torch.sum(torch.abs(F_obs - F_calc))
    denominator = torch.sum(torch.abs(F_obs))
    r_factor = (numerator / denominator).item()
    return r_factor

def get_rfactors(F_obs: torch.Tensor, F_calc: torch.Tensor, rfree: torch.Tensor) -> tuple:
    """
    Get R-factors for working and test sets.

    Parameters:
    -----------
    F_obs : torch.Tensor, shape (N,)
        Observed structure factor amplitudes
    F_calc : torch.Tensor, shape (N,)
        Calculated structure factor amplitudes
    rfree : torch.Tensor, shape (N,)
        Boolean mask indicating R-free reflections (1 is Working set, 0 is Test set)

    Returns:
    --------
    r_factors : tuple
        (r_work, r_test) where
        r_work : float
            R-factor for working set
        r_test : float
            R-factor for test set
    """
    rfree = rfree.to(torch.bool)
    r_work = rfactor(F_obs[rfree], F_calc[rfree])
    r_test = rfactor(F_obs[~rfree], F_calc[~rfree])
    return r_work, r_test

def bin_wise_rfactors(F_obs: torch.Tensor, F_calc: torch.Tensor, rfree: torch.Tensor, bins: torch.Tensor) -> tuple:
    """
    Calculate bin-wise R-factors between observed and calculated structure factors.

    Args:
        F_obs (torch.Tensor): Observed structure factors.
        F_calc (torch.Tensor): Calculated structure factors.
        rfree (torch.Tensor): R-free mask.
        bins (torch.Tensor): Bin indices for each reflection.

    Returns:
        tuple: (r_work_bins, r_test_bins) where
            r_work_bins : torch.Tensor
                R-factors for working set (per bin)
            r_test_bins : torch.Tensor
                R-factors for test set (per bin)
    """
    r_work_bins = []
    r_test_bins = []
    for b in range(bins.max().item() + 1):
        mask = bins == b
        r_work = rfactor(F_obs[mask & rfree], F_calc[mask & rfree])
        r_test = rfactor(F_obs[mask & ~rfree], F_calc[mask & ~rfree])
        r_work_bins.append(r_work)
        r_test_bins.append(r_test)
    return torch.tensor(r_work_bins), torch.tensor(r_test_bins)

def find_solvent_voids(mask, periodic=True):
    """
    Identify void regions in a 3D boolean tensor using connected component analysis.
    
    A void is defined as a connected region of False values (solvent). With periodic
    boundary conditions, voids can wrap around the edges of the array (like in a
    crystallographic unit cell). Without periodic boundaries, only enclosed voids
    are detected.
    
    Parameters:
    -----------
    mask : torch.Tensor or numpy.ndarray, shape (nx, ny, nz)
        Boolean tensor where True indicates solid regions (e.g., protein) and 
        False indicates empty regions (e.g., solvent). Can be either PyTorch tensor
        or NumPy array.
    
    periodic : bool, optional (default=True)
        If True, apply periodic boundary conditions (voids can wrap around edges).
        If False, only detect voids that are completely enclosed and don't touch
        the boundaries.
        
    Returns:
    --------
    voids_dict : dict
        Dictionary where:
        - keys: int, volume (number of voxels) of each void in the original array
        - values: torch.Tensor or numpy.ndarray, boolean mask of same shape as input
                  with True only for that specific void region
        
        Returns an empty dict if no voids are found.
        
    Examples:
    ---------
    >>> import torch
    >>> # Create a simple 5x5x5 grid with a void in the center
    >>> mask = torch.ones(5, 5, 5, dtype=torch.bool)
    >>> mask[2, 2, 2] = False  # Single void voxel
    >>> voids = identify_voids(mask)
    >>> print(voids)
    {1: tensor([[[False, False, ...]], dtype=torch.bool)}
    
    >>> # Multiple voids
    >>> mask = torch.ones(10, 10, 10, dtype=torch.bool)
    >>> mask[2:4, 2:4, 2:4] = False  # First void
    >>> mask[6:8, 6:8, 6:8] = False  # Second void
    >>> voids = identify_voids(mask)
    >>> print(f"Found {len(voids)} voids with volumes: {list(voids.keys())}")
    Found 2 voids with volumes: [8, 8]
    
    >>> # Void wrapping around periodic boundaries
    >>> mask = torch.ones(10, 10, 10, dtype=torch.bool)
    >>> mask[0:2, 5, 5] = False  # Near x=0
    >>> mask[8:10, 5, 5] = False  # Near x=max (wraps to connect with x=0)
    >>> voids = identify_voids(mask, periodic=True)
    >>> print(f"Found {len(voids)} void (wraps around)")
    Found 1 void (wraps around)
    
    Notes:
    ------
    - Uses scipy.ndimage.label for connected component analysis
    - Connectivity is 26-connected (face, edge, and corner neighbors)
    - With periodic=True, the array is padded by wrapping to detect cross-boundary voids
    - Performance is O(n) where n is the total number of voxels
    - With periodic boundaries, large percolating voids are still detected
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
        padded_mask = np.pad(inverted_mask, pad_width=1, mode='wrap')
        
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
        void_mask = (labeled_array == label_id)
        
        if not periodic:
            # For non-periodic, check if void touches boundary
            touches_boundary = (
                np.any(void_mask[0, :, :]) or np.any(void_mask[-1, :, :]) or
                np.any(void_mask[:, 0, :]) or np.any(void_mask[:, -1, :]) or
                np.any(void_mask[:, :, 0]) or np.any(void_mask[:, :, -1])
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
            void_mask_tensor = torch.from_numpy(void_mask).to(device=device, dtype=torch.bool)
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

def gaussian_to_lognormal_sigma(F: torch.Tensor, sigma_F: torch.Tensor, 
                                eps: float = 1e-10) -> torch.Tensor:
    """
    Approximate the sigma parameter of a lognormal distribution from Gaussian statistics.
    
    If we assume F comes from a lognormal distribution X ~ LogNormal(μ, σ²), then:
    - Mean: E[X] ≈ F
    - Std:  √Var[X] ≈ sigma_F
    
    For lognormal distribution:
    - E[X] = exp(μ + σ²/2)
    - Var(X) = exp(2μ + σ²)(exp(σ²) - 1)
    
    We can derive:
    - CV² = Var[X]/E[X]² = exp(σ²) - 1
    - σ = √(log(1 + CV²))
    
    where CV = sigma_F/F is the coefficient of variation.
    
    Args:
        F: Structure factor amplitudes (mean of the distribution)
        sigma_F: Standard deviations
        eps: Small value to avoid division by zero
        
    Returns:
        torch.Tensor: Sigma parameter for lognormal distribution
        
    References:
        - Lognormal distribution: https://en.wikipedia.org/wiki/Log-normal_distribution
        - This approximation assumes F and sigma_F represent the mean and std of 
          the lognormal distribution (not of the underlying normal distribution)
    """
    # Avoid division by zero
    F_safe = torch.clamp(F, min=eps)
    sigma_F_safe = torch.clamp(sigma_F, min=eps)
    
    # Compute coefficient of variation (CV)
    CV = sigma_F_safe / F_safe
    
    # Compute CV²
    CV_squared = CV ** 2
    
    # For lognormal: CV² = exp(σ²) - 1
    # Therefore: σ = √(log(1 + CV²))
    sigma_lognormal = torch.sqrt(torch.log1p(CV_squared))
    
    return sigma_lognormal


def gaussian_to_lognormal_mu(F: torch.Tensor, sigma_lognormal: torch.Tensor, 
                             eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate the mu parameter of a lognormal distribution given F and sigma.
    
    For lognormal distribution X ~ LogNormal(μ, σ²):
    - E[X] = exp(μ + σ²/2)
    
    Solving for μ:
    - μ = log(E[X]) - σ²/2
    
    Args:
        F: Structure factor amplitudes (mean of the distribution)
        sigma_lognormal: Sigma parameter from lognormal distribution
        eps: Small value to avoid log of zero
        
    Returns:
        torch.Tensor: Mu parameter for lognormal distribution
    """
    F_safe = torch.clamp(F, min=eps)
    mu_lognormal = torch.log(F_safe) - 0.5 * sigma_lognormal ** 2
    return mu_lognormal
