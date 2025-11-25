import numpy as np


def rotate_coords_numpy(coords,phi,rho):
    phi = float(phi * np.pi / 180)
    rho = float(rho * np.pi / 180)
    rot_matrix = np.array([[np.cos(phi),-np.sin(phi),0],
                               [np.sin(phi)*np.cos(rho),np.cos(phi)*np.cos(rho),-np.sin(rho)],
                               [np.sin(phi)*np.sin(rho),np.cos(phi)*np.sin(rho),np.cos(rho)]],dtype=np.float64)
    return np.einsum('ij,kj->ki',rot_matrix,coords)

def get_rfactor(fobs,fcalc):
    fobs = np.abs(fobs)
    fcalc = np.abs(fcalc)
    return np.sum(np.abs(fobs - fcalc)) / np.sum(fobs)

def get_s(hkl, unit_cell):
    s = get_scattering_vectors(hkl, unit_cell)
    s = np.sum(s**2,axis=1)**0.5
    return s

def get_scattering_vectors(hkl, unit_cell):
    recB = reciprocal_basis_matrix(unit_cell)
    hkl = np.array(hkl)  # Ensure hkl is a numpy array
    s = np.dot(hkl,recB)
    return s

def get_fractional_matrix(unit_cell):
    a, b, c = unit_cell[:3]
    alpha, beta, gamma = np.radians(unit_cell[3:])
    cos_alpha, cos_beta, cos_gamma = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sin_gamma = np.sin(gamma)
    volume = np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma)
    B = np.array([
        [a, b * cos_gamma, c * cos_beta],
        [0, b * sin_gamma, c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma],
        [0, 0, c * volume / sin_gamma]
    ])   
    return B

def get_res_for_dataset(ds):
    cell = ds.cell
    cell_np = np.array([cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma])
    hkl = ds.reset_index().loc[:, ['H', 'K', 'L']].values.astype(np.float64)
    s = get_scattering_vectors(hkl, cell_np)
    res = 1.0 / np.sqrt(np.sum(s**2, axis=1))
    return res

def cartesian_to_fractional(cart_coords, unit_cell):
    B_inv = get_inv_fractional_matrix(unit_cell)
    fractional_vector = np.dot(cart_coords,B_inv.T)
    return fractional_vector

def get_inv_fractional_matrix(unit_cell):
    B = get_fractional_matrix(unit_cell)
    B_inv = np.linalg.inv(B)
    return B_inv



def fractional_to_cartesian(fractional_coords, unit_cell):
    B = get_fractional_matrix(unit_cell)
    cart_coords = np.dot(fractional_coords,B.T)
    return cart_coords

def convert_coords_to_fractional(df, unit_cell):
    xyz = df[['x', 'y', 'z']].values
    fractional_coordinates = cartesian_to_fractional(xyz, unit_cell)
    return fractional_coordinates

def reciprocal_basis_matrix(unit_cell):
    # Extract unit cell parameters
    a, b, c, alpha, beta, gamma = unit_cell
    alpha, beta, gamma = np.radians([alpha, beta, gamma])
    # Compute real-space basis vectors
    cos_alpha, cos_beta, cos_gamma = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sin_gamma = np.sin(gamma)
    volume = np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma)
    a_vec = np.array([a, 0, 0])
    b_vec = np.array([b * cos_gamma, b * sin_gamma, 0])
    c_vec = np.array([
        c * cos_beta,
        c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
        c * volume / sin_gamma
    ])
    # Compute reciprocal basis vectors
    volume_real = np.dot(a_vec, np.cross(b_vec, c_vec))
    a_star = np.cross(b_vec, c_vec) / volume_real
    b_star = np.cross(c_vec, a_vec) / volume_real
    c_star = np.cross(a_vec, b_vec) / volume_real
    # Assemble reciprocal basis matrix
    return np.array([a_star, b_star, c_star])

def calc_outliers(fobs,fcalc,z):
    fobs = np.abs(fobs)
    fcalc = np.abs(fcalc)
    diff = np.abs(fobs - fcalc) / fobs * np.mean(fobs)
    std = np.std(diff)
    outliers = diff >  z * std
    return outliers

def get_grids(cell,max_res=0.8):
    nsteps = np.astype(np.floor(cell[:3] / max_res * 3), int)
    x = np.arange(nsteps[0]) / nsteps[0]
    y = np.arange(nsteps[1]) / nsteps[1]
    z = np.arange(nsteps[2]) / nsteps[2]
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    array_shape = x.shape
    x = x.reshape((*x.shape, 1))
    y = y.reshape((*y.shape, 1))
    z = z.reshape((*z.shape, 1))
    xyz = np.concatenate((x, y, z), axis=3).reshape(-1, 3)
    xyz_real_grid = fractional_to_cartesian(xyz, cell)
    xyz_real_grid = xyz_real_grid.reshape((*array_shape, 3))
    recgrid = np.zeros(array_shape, dtype=float)
    return recgrid, xyz_real_grid

def get_real_grid(cell,max_res=0.8,gridsize=None):
    if gridsize is not None:
        nsteps = np.array(gridsize,dtype=int)
    else:
        nsteps = np.astype(np.floor(cell[:3] / max_res * 3), int)
    # Place grid points at grid edges: i / N (CCTBX convention)
    # This matches how CCTBX/gemmi create maps
    x = np.arange(nsteps[0]) / nsteps[0]
    y = np.arange(nsteps[1]) / nsteps[1]
    z = np.arange(nsteps[2]) / nsteps[2]
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    array_shape = x.shape
    x = x.reshape((*x.shape, 1))
    y = y.reshape((*y.shape, 1))
    z = z.reshape((*z.shape, 1))
    xyz = np.concatenate((x, y, z), axis=3).reshape(-1, 3)
    xyz_real_grid = fractional_to_cartesian(xyz, cell)
    xyz_real_grid = xyz_real_grid.reshape((*array_shape, 3))
    return xyz_real_grid

def put_hkl_on_grid(real_space_grid,diff,hkl):
    rec_space = np.zeros(real_space_grid.shape[:3],dtype=np.complex128)
    f = diff
    rec_space[hkl[:,0],hkl[:,1],hkl[:,2]] = f
    return rec_space

def align_pdbs(pdb1,pdb2,Atoms=None):
    # align to pointclouds
    if Atoms is None:
        xyz1 = pdb1[['x','y','z']].values
        xyz2 = pdb2[['x','y','z']].values
        temp = pdb2['tempfactor'].values
    else:
        xyz1 = pdb1.loc[pdb1['name'].isin(Atoms),['x','y','z']].values
        xyz2 = pdb2.loc[pdb2['name'].isin(Atoms),['x','y','z']].values
        temp = pdb2.loc[pdb2['name'].isin(Atoms),'tempfactor'].values
    transformation_matrix1,rmsd1 = superpose_vectors_robust(xyz1,xyz2,weights=1/temp)
    transformation_matrix = transformation_matrix1
    rmsd = rmsd1
    xyz_moved = apply_transformation(pdb2[['x','y','z']].values,transformation_matrix)
    pdb2.loc[:,['x','y','z']] = xyz_moved
    xyz1 = pdb1[['x','y','z']].values
    rmsd = np.sqrt(np.mean(np.sum((xyz1 - xyz_moved)**2, axis=1)))
    return pdb2, rmsd


def get_alignment_matrix(pdb1,pdb2,Atoms=None):
    # align to pointclouds
    if Atoms is None:
        xyz1 = pdb1[['x','y','z']].values
        xyz2 = pdb2[['x','y','z']].values
        temp = pdb2['tempfactor'].values
    else:
        xyz1 = pdb1.loc[pdb1['name'].isin(Atoms),['x','y','z']].values
        xyz2 = pdb2.loc[pdb2['name'].isin(Atoms),['x','y','z']].values
        temp = pdb2.loc[pdb2['name'].isin(Atoms),'tempfactor'].values
    transformation_matrix1,rmsd1 = superpose_vectors_robust(xyz1,xyz2,weights=1/temp)
    transformation_matrix = transformation_matrix1
    return transformation_matrix, rmsd1

def superpose_vectors_robust(target_coords, mobile_coords, weights=None, max_iterations=1):
    """
    Superpose mobile_coords onto target_coords with robust handling of special cases.
    
    Parameters:
    -----------
    target_coords : numpy.ndarray
        Target coordinates with shape (N, 3)
    mobile_coords : numpy.ndarray
        Mobile coordinates with shape (N, 3) to be superposed onto target
    weights : numpy.ndarray, optional
        Per-atom weights for the superposition (N,)
    max_iterations : int, optional
        Number of iterations for refinement (1 = standard Kabsch algorithm)
        
    Returns:
    --------
    transformation_matrix : numpy.ndarray
        4x4 transformation matrix that maps mobile_coords onto target_coords
    rmsd : float
        Root-mean-square deviation after superposition
    """
    # Check input dimensions
    if target_coords.shape != mobile_coords.shape:
        raise ValueError(f"Input coordinate arrays must have the same shape: {target_coords.shape} vs {mobile_coords.shape}")
    
    if weights is None:
        weights = np.ones(len(target_coords))
    
    # Normalize weights
    weights = weights / np.sum(weights)
    weights_reshape = weights.reshape(-1, 1)
    
    # Initial mobile coords copy
    mobile_coords_current = mobile_coords.copy()
    best_rmsd = float('inf')
    best_matrix = np.eye(4)
    
    for iteration in range(max_iterations):
        # Calculate centroids
        target_centroid = np.sum(weights_reshape * target_coords, axis=0)
        mobile_centroid = np.sum(weights_reshape * mobile_coords_current, axis=0)
        
        # Center coordinates
        target_centered = target_coords - target_centroid
        mobile_centered = mobile_coords_current - mobile_centroid
        
        # Calculate the covariance matrix with weights
        covariance = np.zeros((3, 3))
        for i in range(len(weights)):
            covariance += weights[i] * np.outer(mobile_centered[i], target_centered[i])
        
        # SVD of covariance matrix
        try:
            U, S, Vt = np.linalg.svd(covariance)
            
            # Check for reflection case (determinant < 0)
            det = np.linalg.det(np.dot(Vt.T, U.T))
            correction = np.eye(3)
            if det < 0:
                correction[2, 2] = -1
                
            # Calculate rotation matrix
            rotation_matrix = np.dot(np.dot(Vt.T, correction), U.T)
            
            # FIXED: Calculate translation correctly
            # The correct way is: translation = target_centroid - (rotation_matrix @ mobile_centroid)
            # In NumPy notation with row vectors, this is:
            rotated_mobile_centroid = np.dot(mobile_centroid, rotation_matrix.T)
            translation = target_centroid - rotated_mobile_centroid
            
            # Compute 4x4 transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation
            
            # Apply transformation and calculate RMSD
            # Using the correct transformation application
            mobile_transformed = np.dot(mobile_coords, rotation_matrix.T) + translation
            
            squared_diffs = np.sum((target_coords - mobile_transformed)**2, axis=1)
            rmsd = np.sqrt(np.sum(weights * squared_diffs))
            
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_matrix = transformation_matrix
                
            # Update mobile coords for next iteration if doing iterative refinement
            if max_iterations > 1:
                mobile_coords_current = mobile_transformed
                
        except np.linalg.LinAlgError:
            print("SVD computation failed, falling back to identity transformation")
            return np.eye(4), np.sqrt(np.mean(np.sum((target_coords - mobile_coords)**2, axis=1)))
    return best_matrix, best_rmsd



def apply_transformation(points, transformation_matrix):
    """Apply 4x4 transformation matrix to 3D points"""
    # Convert to homogeneous coordinates
    homo_points = np.hstack((points, np.ones((points.shape[0], 1))))
    # Apply transformation
    transformed = np.dot(homo_points, transformation_matrix.T)
    # Return 3D coordinates
    return transformed[:, :3]

def invert_transformation_matrix(transformation_matrix):
    """
    Compute the inverse of a 4x4 transformation matrix.
    
    Parameters:
    -----------
    transformation_matrix : numpy.ndarray
        4x4 transformation matrix
        
    Returns:
    --------
    inverse_matrix : numpy.ndarray
        Inverse 4x4 transformation matrix
    """
    # Extract rotation and translation
    rotation = transformation_matrix[:3, :3]
    translation = transformation_matrix[:3, 3]
    
    # Calculate inverse rotation (transpose) and inverse translation
    inverse_rotation = rotation.T
    inverse_translation = -np.dot(inverse_rotation, translation)
    
    # Build inverse transformation matrix
    inverse_matrix = np.eye(4)
    inverse_matrix[:3, :3] = inverse_rotation
    inverse_matrix[:3, 3] = inverse_translation
    
    return inverse_matrix
