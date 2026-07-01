"""
Superposition functions for coordinate alignment.

Functions for computing optimal superposition of coordinate sets
using the Kabsch algorithm and related methods.
"""

import numpy as np
import torch


def superpose_vectors_robust_torch(
    ref_coords, mov_coords, weights=None, max_iterations=10
):
    """
    Perform weighted superposition of two coordinate sets using SVD (PyTorch version).

    Parameters
    ----------
    ref_coords : torch.Tensor
        Reference coordinates of shape (N, 3).
    mov_coords : torch.Tensor
        Mobile coordinates of shape (N, 3) to be superposed onto reference.
    weights : torch.Tensor, optional
        Weights for each atom of shape (N, 1). Default is uniform weights.
    max_iterations : int, optional
        Maximum number of weighted Kabsch steps. Default is 10. The function
        runs up to this many steps and returns the transform with the lowest
        weighted RMSD seen.

        .. note::
           The per-atom ``weights`` are held fixed across steps (this is not an
           iteratively-reweighted / robust estimator), and a single weighted
           Kabsch step is already optimal for fixed weights, so additional
           iterations do not normally change the returned transform.

    Returns
    -------
    torch.Tensor
        Transformation matrix of shape (3, 4): the top-left (3, 3) block is
        the rotation and the last column is the translation. Unlike the NumPy
        sibling :func:`superpose_vectors_robust`, only the matrix is returned
        (no RMSD).
    """
    if weights is None:
        weights = torch.ones((ref_coords.shape[0], 1), device=ref_coords.device)
    weights = weights / torch.sum(weights)

    mobile_coords_current = mov_coords.clone()
    best_matrix = torch.eye(
        4, device=mobile_coords_current.device, dtype=mobile_coords_current.dtype
    )
    best_rmsd = torch.tensor(float("inf"))
    for iteration in range(max_iterations):
        # Calculate centroids
        target_centroid = torch.sum(weights * ref_coords, axis=0)
        mobile_centroid = torch.sum(weights * mobile_coords_current, axis=0)

        # Center coordinates
        target_centered = ref_coords - target_centroid
        mobile_centered = mobile_coords_current - mobile_centroid

        # Calculate the covariance matrix with weights
        covariance = torch.zeros(
            (3, 3),
            dtype=mobile_coords_current.dtype,
            device=mobile_coords_current.device,
        )
        for i in range(len(weights)):
            covariance += weights[i] * torch.outer(
                mobile_centered[i], target_centered[i]
            )

        # SVD of covariance matrix
        U, S, Vt = torch.linalg.svd(covariance)

        # Check for reflection case (determinant < 0)
        det = torch.linalg.det(torch.matmul(Vt.T, U.T))
        correction = torch.eye(
            3, dtype=mobile_coords_current.dtype, device=mobile_coords_current.device
        )
        if det < 0:
            correction[2, 2] = -1

        # Calculate rotation matrix
        rotation_matrix = torch.matmul(torch.matmul(Vt.T, correction), U.T)

        # Calculate translation correctly
        rotated_mobile_centroid = torch.matmul(mobile_centroid, rotation_matrix.T)
        translation = target_centroid - rotated_mobile_centroid

        # Compute 4x4 transformation matrix
        transformation_matrix = torch.zeros(
            (3, 4),
            device=mobile_coords_current.device,
            dtype=mobile_coords_current.dtype,
        )
        transformation_matrix[:, :3] = rotation_matrix
        transformation_matrix[:, 3] = translation

        # Apply transformation and calculate RMSD
        mobile_transformed = torch.matmul(mov_coords, rotation_matrix.T) + translation

        squared_diffs = torch.sum((ref_coords - mobile_transformed) ** 2, axis=1)
        rmsd = torch.sqrt(torch.sum(weights * squared_diffs))

        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_matrix = transformation_matrix
        # Update mobile coords for next iteration if doing iterative refinement
        if max_iterations > 1:
            mobile_coords_current = mobile_transformed
    return best_matrix


def superpose_vectors_robust(
    target_coords, mobile_coords, weights=None, max_iterations=1
):
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
        raise ValueError(
            f"Input coordinate arrays must have the same shape: {target_coords.shape} vs {mobile_coords.shape}"
        )

    if weights is None:
        weights = np.ones(len(target_coords))

    # Normalize weights
    weights = weights / np.sum(weights)
    weights_reshape = weights.reshape(-1, 1)

    # Initial mobile coords copy
    mobile_coords_current = mobile_coords.copy()
    best_rmsd = float("inf")
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

            # Calculate translation correctly
            rotated_mobile_centroid = np.dot(mobile_centroid, rotation_matrix.T)
            translation = target_centroid - rotated_mobile_centroid

            # Compute 4x4 transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation

            # Apply transformation and calculate RMSD
            mobile_transformed = np.dot(mobile_coords, rotation_matrix.T) + translation

            squared_diffs = np.sum((target_coords - mobile_transformed) ** 2, axis=1)
            rmsd = np.sqrt(np.sum(weights * squared_diffs))

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_matrix = transformation_matrix

            # Update mobile coords for next iteration if doing iterative refinement
            if max_iterations > 1:
                mobile_coords_current = mobile_transformed

        except np.linalg.LinAlgError:
            print("SVD computation failed, falling back to identity transformation")
            return np.eye(4), np.sqrt(
                np.mean(np.sum((target_coords - mobile_coords) ** 2, axis=1))
            )
    return best_matrix, best_rmsd


def align_torch(xyz1, xyz2, idx_to_move=None):
    """
    Align two coordinate sets using superposition (PyTorch version).

    Parameters
    ----------
    xyz1 : torch.Tensor
        Target coordinates of shape (N, 3).
    xyz2 : torch.Tensor
        Coordinates to be aligned of shape (N, 3).
    idx_to_move : torch.Tensor, optional
        Indices of atoms to use for alignment. If None, uses all atoms.

    Returns
    -------
    torch.Tensor
        Aligned coordinates of shape (N, 3).
    """
    if idx_to_move is not None:
        transformation_matrix1 = superpose_vectors_robust_torch(
            xyz1[idx_to_move], xyz2[idx_to_move]
        )
    else:
        transformation_matrix1 = superpose_vectors_robust_torch(xyz1, xyz2)
    transformation_matrix = transformation_matrix1
    xyz_moved = apply_transformation(xyz2, transformation_matrix)
    return xyz_moved


def get_alignement_matrix(xyz1, xyz2, idx_to_move=None):
    """
    Get the alignment transformation matrix between two coordinate sets.

    Parameters
    ----------
    xyz1 : torch.Tensor
        Target coordinates of shape (N, 3).
    xyz2 : torch.Tensor
        Coordinates to be aligned of shape (N, 3).
    idx_to_move : torch.Tensor, optional
        Indices of atoms to use for alignment. If None, uses all atoms.

    Returns
    -------
    torch.Tensor
        Transformation matrix of shape (3, 4).
    """
    if idx_to_move is not None:
        transformation_matrix = superpose_vectors_robust_torch(
            xyz1[idx_to_move], xyz2[idx_to_move]
        )
    else:
        transformation_matrix = superpose_vectors_robust_torch(xyz1, xyz2)
    return transformation_matrix


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

    Notes
    -----
    The superposition is B-factor weighted with ``weights = 1 / tempfactor``,
    so the 'tempfactor' column must be nonzero for every atom used (a zero
    value triggers a division by zero).
    """
    # align to pointclouds
    if Atoms is None:
        xyz1 = pdb1[["x", "y", "z"]].values
        xyz2 = pdb2[["x", "y", "z"]].values
        temp = pdb2["tempfactor"].values
    else:
        xyz1 = pdb1.loc[pdb1["name"].isin(Atoms), ["x", "y", "z"]].values
        xyz2 = pdb2.loc[pdb2["name"].isin(Atoms), ["x", "y", "z"]].values
        temp = pdb2.loc[pdb2["name"].isin(Atoms), "tempfactor"].values
    transformation_matrix1, rmsd1 = superpose_vectors_robust(
        xyz1, xyz2, weights=1 / temp
    )
    transformation_matrix = transformation_matrix1
    rmsd = rmsd1
    xyz_moved = apply_transformation_numpy(
        pdb2[["x", "y", "z"]].values, transformation_matrix
    )
    pdb2.loc[:, ["x", "y", "z"]] = xyz_moved
    xyz1 = pdb1[["x", "y", "z"]].values
    rmsd = np.sqrt(np.mean(np.sum((xyz1 - xyz_moved) ** 2, axis=1)))
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
        xyz1 = pdb1[["x", "y", "z"]].values
        xyz2 = pdb2[["x", "y", "z"]].values
        temp = pdb2["tempfactor"].values
    else:
        xyz1 = pdb1.loc[pdb1["name"].isin(Atoms), ["x", "y", "z"]].values
        xyz2 = pdb2.loc[pdb2["name"].isin(Atoms), ["x", "y", "z"]].values
        temp = pdb2.loc[pdb2["name"].isin(Atoms), "tempfactor"].values
    transformation_matrix1, rmsd1 = superpose_vectors_robust(
        xyz1, xyz2, weights=1 / temp
    )
    transformation_matrix = transformation_matrix1
    return transformation_matrix, rmsd1


def apply_transformation(points, transformation_matrix):
    """
    Apply a 4x4 transformation matrix to 3D points (PyTorch version).

    Parameters
    ----------
    points : torch.Tensor
        3D points of shape (N, 3).
    transformation_matrix : torch.Tensor
        Transformation matrix of shape (3, 4) or (4, 4).

    Returns
    -------
    torch.Tensor
        Transformed 3D points of shape (N, 3).
    """
    # Convert to homogeneous coordinates
    homo_points = torch.hstack(
        (points, torch.ones((points.shape[0], 1), device=points.device))
    )
    last_row = torch.tensor([0, 0, 0, 1], device=points.device)
    transformation_matrix = torch.vstack((transformation_matrix, last_row))
    # Apply transformation
    transformed = torch.matmul(homo_points, transformation_matrix.T)
    # Return 3D coordinates
    return transformed[:, :3]


def apply_transformation_numpy(points, transformation_matrix):
    """
    Apply a 4x4 transformation matrix to 3D points (NumPy version).

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
