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
    """Weighted SVD (Kabsch) superposition of two coordinate sets (PyTorch).

    Parameters
    ----------
    ref_coords : torch.Tensor
        Reference coordinates of shape (N, 3).
    mov_coords : torch.Tensor
        Mobile coordinates of shape (N, 3) to be superposed onto reference.
    weights : torch.Tensor, optional
        Weights for each atom of shape (N, 1). Default is uniform weights.
    max_iterations : int, optional
        Weighted Kabsch steps, best weighted RMSD wins. Weights are never
        reweighted, so past the first step this normally changes nothing.

    Returns
    -------
    torch.Tensor
        Transformation matrix of shape (3, 4): rotation block plus translation
        column. Unlike the NumPy sibling :func:`superpose_vectors_robust`, no
        RMSD is returned.
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
    """Superpose mobile onto target coordinates with the weighted Kabsch algorithm.

    Reflections are rejected by determinant correction, so the result is always
    a proper rotation.

    Parameters
    ----------
    target_coords : numpy.ndarray
        Target coordinates with shape (N, 3).
    mobile_coords : numpy.ndarray
        Mobile coordinates with shape (N, 3) to be superposed onto target.
    weights : numpy.ndarray, optional
        Per-atom weights with shape (N,). Default is uniform weights.
    max_iterations : int, optional
        Number of iterations for refinement. Default is 1 (standard Kabsch).

    Returns
    -------
    transformation_matrix : numpy.ndarray
        4x4 matrix mapping mobile_coords onto target_coords. If the SVD fails
        this degrades to the identity (printed, not raised).
    rmsd : float
        Weighted RMSD after superposition -- unweighted on the SVD-failure path.

    Raises
    ------
    ValueError
        If input coordinate arrays have different shapes.
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
    """Align pdb2 onto pdb1 by weighted Kabsch, rewriting pdb2's coordinates in place.

    Weighting is ``1 / tempfactor``, so every atom used must have a nonzero
    'tempfactor' or the call divides by zero.

    Parameters
    ----------
    pdb1 : pandas.DataFrame
        Reference structure with 'x', 'y', 'z', 'name' and 'tempfactor' columns.
    pdb2 : pandas.DataFrame
        Mobile structure; its coordinate columns are overwritten.
    Atoms : list, optional
        Atom names to use for alignment. If None, all atoms are used.

    Returns
    -------
    pdb2 : pandas.DataFrame
        The same object, with updated coordinates.
    rmsd : float
        Unweighted all-atom RMSD after alignment.
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
    """Transformation matrix that would superimpose pdb2 onto pdb1, without applying it.

    Same ``1 / tempfactor`` weighting as :func:`align_pdbs`, so 'tempfactor'
    must be nonzero for every atom used.

    Parameters
    ----------
    pdb1 : pandas.DataFrame
        Reference structure with 'x', 'y', 'z', 'name' and 'tempfactor' columns.
    pdb2 : pandas.DataFrame
        Mobile PDB structure.
    Atoms : list, optional
        Atom names to use for alignment. If None, all atoms are used.

    Returns
    -------
    transformation_matrix : numpy.ndarray
        4x4 transformation matrix.
    rmsd : float
        Weighted RMSD that would result from the alignment.
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
    homo_points = torch.hstack(
        (points, torch.ones((points.shape[0], 1), device=points.device))
    )
    last_row = torch.tensor([0, 0, 0, 1], device=points.device)
    transformation_matrix = torch.vstack((transformation_matrix, last_row))
    transformed = torch.matmul(homo_points, transformation_matrix.T)
    return transformed[:, :3]


def apply_transformation_numpy(points, transformation_matrix):
    """Apply a 4x4 transformation matrix to 3D points (NumPy version).

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
    homo_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed = np.dot(homo_points, transformation_matrix.T)
    return transformed[:, :3]


def invert_transformation_matrix(transformation_matrix):
    """Invert a rigid-body 4x4 transformation matrix.

    Uses ``R^-1 = R^T``, so a matrix that is not a pure rotation plus
    translation (any scale or shear) is inverted silently wrongly.

    Parameters
    ----------
    transformation_matrix : numpy.ndarray
        4x4 matrix with rotation in the top-left 3x3 and translation in the
        top-right 3x1.

    Returns
    -------
    numpy.ndarray
        Inverse 4x4 transformation matrix.
    """
    rotation = transformation_matrix[:3, :3]
    translation = transformation_matrix[:3, 3]

    inverse_rotation = rotation.T
    inverse_translation = -np.dot(inverse_rotation, translation)

    inverse_matrix = np.eye(4)
    inverse_matrix[:3, :3] = inverse_rotation
    inverse_matrix[:3, 3] = inverse_translation

    return inverse_matrix
