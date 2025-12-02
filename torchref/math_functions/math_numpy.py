import numpy as np


def rotate_coords_numpy(coords, phi, rho):
    """
    Rotate 3D coordinates by phi and rho angles.

    Applies a rotation transformation to a set of 3D coordinates using
    two rotation angles (phi and rho) in degrees.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of 3D coordinates with shape (N, 3).
    phi : float
        First rotation angle in degrees.
    rho : float
        Second rotation angle in degrees.

    Returns
    -------
    numpy.ndarray
        Rotated coordinates with the same shape as input (N, 3).
    """
    phi = float(phi * np.pi / 180)
    rho = float(rho * np.pi / 180)
    rot_matrix = np.array([[np.cos(phi),-np.sin(phi),0],
                               [np.sin(phi)*np.cos(rho),np.cos(phi)*np.cos(rho),-np.sin(rho)],
                               [np.sin(phi)*np.sin(rho),np.cos(phi)*np.sin(rho),np.cos(rho)]],dtype=np.float64)
    return np.einsum('ij,kj->ki',rot_matrix,coords)


def get_rfactor(fobs, fcalc):
    """
    Calculate the R-factor between observed and calculated structure factors.

    The R-factor is a measure of agreement between observed and calculated
    structure factor amplitudes, defined as sum(|Fobs - Fcalc|) / sum(Fobs).

    Parameters
    ----------
    fobs : numpy.ndarray
        Observed structure factor amplitudes.
    fcalc : numpy.ndarray
        Calculated structure factor amplitudes.

    Returns
    -------
    float
        R-factor value between 0 and 1.
    """
    fobs = np.abs(fobs)
    fcalc = np.abs(fcalc)
    return np.sum(np.abs(fobs - fcalc)) / np.sum(fobs)


def get_s(hkl, unit_cell):
    """
    Calculate the magnitude of scattering vectors for given Miller indices.

    Computes |s| = 1/d where d is the interplanar spacing for each reflection.

    Parameters
    ----------
    hkl : numpy.ndarray
        Miller indices with shape (N, 3).
    unit_cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    numpy.ndarray
        Magnitude of scattering vectors with shape (N,).
    """
    s = get_scattering_vectors(hkl, unit_cell)
    s = np.sum(s**2, axis=1)**0.5
    return s


def get_scattering_vectors(hkl, unit_cell):
    """
    Calculate scattering vectors from Miller indices and unit cell.

    Transforms Miller indices to reciprocal space scattering vectors
    using the reciprocal basis matrix.

    Parameters
    ----------
    hkl : numpy.ndarray or list
        Miller indices with shape (N, 3).
    unit_cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    numpy.ndarray
        Scattering vectors in reciprocal space with shape (N, 3).
    """
    recB = reciprocal_basis_matrix(unit_cell)
    hkl = np.array(hkl)  # Ensure hkl is a numpy array
    s = np.dot(hkl, recB)
    return s


def get_fractional_matrix(unit_cell):
    """
    Calculate the fractional-to-Cartesian transformation matrix.

    Constructs the matrix B that transforms fractional coordinates to
    Cartesian coordinates based on the unit cell parameters.

    Parameters
    ----------
    unit_cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    numpy.ndarray
        3x3 transformation matrix B such that cart = frac @ B.T.
    """
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
    """
    Calculate resolution for each reflection in a dataset.

    Computes the resolution (d-spacing) for each reflection based on the
    unit cell and Miller indices from a reciprocalspaceship dataset.

    Parameters
    ----------
    ds : reciprocalspaceship.DataSet
        Crystallographic dataset containing Miller indices and unit cell.

    Returns
    -------
    numpy.ndarray
        Resolution values (d-spacing) for each reflection.
    """
    cell = ds.cell
    cell_np = np.array([cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma])
    hkl = ds.reset_index().loc[:, ['H', 'K', 'L']].values.astype(np.float64)
    s = get_scattering_vectors(hkl, cell_np)
    res = 1.0 / np.sqrt(np.sum(s**2, axis=1))
    return res
def cartesian_to_fractional(cart_coords, unit_cell):
    """
    Convert Cartesian coordinates to fractional coordinates.

    Parameters
    ----------
    cart_coords : numpy.ndarray
        Cartesian coordinates with shape (N, 3).
    unit_cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    numpy.ndarray
        Fractional coordinates with shape (N, 3).
    """
    B_inv = get_inv_fractional_matrix(unit_cell)
    fractional_vector = np.dot(cart_coords, B_inv.T)
    return fractional_vector


def get_inv_fractional_matrix(unit_cell):
    """
    Calculate the Cartesian-to-fractional transformation matrix.

    Computes the inverse of the fractional matrix for converting Cartesian
    coordinates to fractional coordinates.

    Parameters
    ----------
    unit_cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    numpy.ndarray
        3x3 inverse transformation matrix B_inv such that frac = cart @ B_inv.T.
    """
    B = get_fractional_matrix(unit_cell)
    B_inv = np.linalg.inv(B)
    return B_inv


def fractional_to_cartesian(fractional_coords, unit_cell):
    """
    Convert fractional coordinates to Cartesian coordinates.

    Parameters
    ----------
    fractional_coords : numpy.ndarray
        Fractional coordinates with shape (N, 3).
    unit_cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    numpy.ndarray
        Cartesian coordinates with shape (N, 3).
    """
    B = get_fractional_matrix(unit_cell)
    cart_coords = np.dot(fractional_coords, B.T)
    return cart_coords


def convert_coords_to_fractional(df, unit_cell):
    """
    Convert coordinates from a DataFrame to fractional coordinates.

    Extracts x, y, z columns from a DataFrame and converts them from
    Cartesian to fractional coordinates.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'x', 'y', 'z' columns with Cartesian coordinates.
    unit_cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    numpy.ndarray
        Fractional coordinates with shape (N, 3).
    """
    xyz = df[['x', 'y', 'z']].values
    fractional_coordinates = cartesian_to_fractional(xyz, unit_cell)
    return fractional_coordinates


def reciprocal_basis_matrix(unit_cell):
    """
    Calculate the reciprocal basis matrix from unit cell parameters.

    Computes the reciprocal space basis vectors (a*, b*, c*) that define
    the transformation from Miller indices to scattering vectors.

    Parameters
    ----------
    unit_cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    numpy.ndarray
        3x3 matrix containing reciprocal basis vectors as rows [a*, b*, c*].
    """
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


def calc_outliers(fobs, fcalc, z):
    """
    Identify outlier reflections based on structure factor differences.

    Detects reflections where the normalized difference between observed
    and calculated structure factors exceeds z standard deviations.

    Parameters
    ----------
    fobs : numpy.ndarray
        Observed structure factor amplitudes.
    fcalc : numpy.ndarray
        Calculated structure factor amplitudes.
    z : float
        Number of standard deviations for outlier threshold.

    Returns
    -------
    numpy.ndarray
        Boolean array where True indicates an outlier reflection.
    """
    fobs = np.abs(fobs)
    fcalc = np.abs(fcalc)
    diff = np.abs(fobs - fcalc) / fobs * np.mean(fobs)
    std = np.std(diff)
    outliers = diff > z * std
    return outliers


def get_grids(cell, max_res=0.8):
    """
    Generate real-space and reciprocal-space grids for Fourier transforms.

    Creates a 3D grid in fractional coordinates and converts it to Cartesian
    coordinates, along with an empty reciprocal space grid.

    Parameters
    ----------
    cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.
    max_res : float, optional
        Maximum resolution in Angstroms for grid spacing. Default is 0.8.

    Returns
    -------
    recgrid : numpy.ndarray
        Empty reciprocal space grid with shape determined by resolution.
    xyz_real_grid : numpy.ndarray
        Real-space grid coordinates with shape (nx, ny, nz, 3).
    """
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


def get_real_grid(cell, max_res=0.8, gridsize=None):
    """
    Generate a real-space grid of Cartesian coordinates.

    Creates a 3D grid in fractional coordinates and converts it to Cartesian
    coordinates. Grid points are placed at cell edges following CCTBX convention.

    Parameters
    ----------
    cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.
    max_res : float, optional
        Maximum resolution in Angstroms for grid spacing. Default is 0.8.
        Ignored if gridsize is provided.
    gridsize : list or numpy.ndarray, optional
        Explicit grid dimensions [nx, ny, nz]. If provided, overrides max_res.

    Returns
    -------
    numpy.ndarray
        Real-space grid coordinates with shape (nx, ny, nz, 3).
    """
    if gridsize is not None:
        nsteps = np.array(gridsize, dtype=int)
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


def put_hkl_on_grid(real_space_grid, diff, hkl):
    """
    Place structure factors on a reciprocal space grid.

    Maps structure factor values to their corresponding positions on a
    3D reciprocal space grid based on Miller indices.

    Parameters
    ----------
    real_space_grid : numpy.ndarray
        Real-space grid used to determine the reciprocal grid dimensions.
        Shape should be (nx, ny, nz, 3) or similar.
    diff : numpy.ndarray
        Structure factor values (complex) to place on the grid.
    hkl : numpy.ndarray
        Miller indices with shape (N, 3), used as grid indices.

    Returns
    -------
    numpy.ndarray
        Complex reciprocal space grid with shape (nx, ny, nz).
    """
    rec_space = np.zeros(real_space_grid.shape[:3], dtype=np.complex128)
    f = diff
    rec_space[hkl[:, 0], hkl[:, 1], hkl[:, 2]] = f
    return rec_space


def align_pdbs(pdb1, pdb2, Atoms=None):
    """
    Align two PDB structures using the Kabsch algorithm.

    Superimposes pdb2 onto pdb1 by minimizing the RMSD between corresponding
    atoms. The transformation is applied in-place to pdb2.

    Parameters
    ----------
    pdb1 : pandas.DataFrame
        Reference PDB structure with 'x', 'y', 'z', 'name', and 'tempfactor' columns.
    pdb2 : pandas.DataFrame
        Mobile PDB structure to be aligned onto pdb1.
    Atoms : list, optional
        List of atom names to use for alignment. If None, all atoms are used.

    Returns
    -------
    pdb2 : pandas.DataFrame
        Transformed pdb2 with updated coordinates.
    rmsd : float
        Root-mean-square deviation after alignment.
    """
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


def get_alignment_matrix(pdb1, pdb2, Atoms=None):
    """
    Calculate the transformation matrix to align two PDB structures.

    Computes the 4x4 transformation matrix that would superimpose pdb2 onto
    pdb1 without actually applying the transformation.

    Parameters
    ----------
    pdb1 : pandas.DataFrame
        Reference PDB structure with 'x', 'y', 'z', 'name', and 'tempfactor' columns.
    pdb2 : pandas.DataFrame
        Mobile PDB structure.
    Atoms : list, optional
        List of atom names to use for alignment. If None, all atoms are used.

    Returns
    -------
    transformation_matrix : numpy.ndarray
        4x4 transformation matrix.
    rmsd : float
        Root-mean-square deviation that would result from the alignment.
    """
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
    Superpose mobile coordinates onto target coordinates using the Kabsch algorithm.

    Computes the optimal rotation and translation to minimize the weighted
    RMSD between two sets of 3D coordinates, with robust handling of special
    cases such as reflection.

    Parameters
    ----------
    target_coords : numpy.ndarray
        Target coordinates with shape (N, 3).
    mobile_coords : numpy.ndarray
        Mobile coordinates with shape (N, 3) to be superposed onto target.
    weights : numpy.ndarray, optional
        Per-atom weights for the superposition with shape (N,). Default is
        uniform weights.
    max_iterations : int, optional
        Number of iterations for refinement. Default is 1 (standard Kabsch).

    Returns
    -------
    transformation_matrix : numpy.ndarray
        4x4 transformation matrix that maps mobile_coords onto target_coords.
    rmsd : float
        Weighted root-mean-square deviation after superposition.

    Raises
    ------
    ValueError
        If input coordinate arrays have different shapes.

    Notes
    -----
    The algorithm uses SVD decomposition of the covariance matrix and handles
    the reflection case by checking the determinant of the rotation matrix.
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
    """
    Apply a 4x4 transformation matrix to 3D points.

    Converts points to homogeneous coordinates, applies the transformation,
    and returns the transformed 3D coordinates.

    Parameters
    ----------
    points : numpy.ndarray
        3D coordinates with shape (N, 3).
    transformation_matrix : numpy.ndarray
        4x4 transformation matrix containing rotation and translation.

    Returns
    -------
    numpy.ndarray
        Transformed 3D coordinates with shape (N, 3).
    """
    # Convert to homogeneous coordinates
    homo_points = np.hstack((points, np.ones((points.shape[0], 1))))
    # Apply transformation
    transformed = np.dot(homo_points, transformation_matrix.T)
    # Return 3D coordinates
    return transformed[:, :3]


def invert_transformation_matrix(transformation_matrix):
    """
    Compute the inverse of a 4x4 transformation matrix.

    Efficiently inverts a rigid-body transformation matrix by transposing
    the rotation component and computing the inverse translation.

    Parameters
    ----------
    transformation_matrix : numpy.ndarray
        4x4 transformation matrix containing rotation (top-left 3x3) and
        translation (top-right 3x1).

    Returns
    -------
    numpy.ndarray
        Inverse 4x4 transformation matrix.

    Notes
    -----
    This function assumes the input is a valid rigid-body transformation
    (rotation + translation). For such matrices, the inverse rotation is
    simply the transpose, and the inverse translation is computed as
    -R^T @ t.
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
