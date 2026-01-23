"""
NumPy implementations of coordinate transformation functions.

These functions provide CPU-based coordinate transformations
for use when GPU acceleration is not needed or available.
"""

import numpy as np


def get_fractional_matrix(cell):
    """
    Calculate the fractional-to-Cartesian transformation matrix.

    Constructs the matrix B that transforms fractional coordinates to
    Cartesian coordinates based on the unit cell parameters.

    Parameters
    ----------
    cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    numpy.ndarray
        3x3 transformation matrix B such that cart = frac @ B.T.
    """
    a, b, c = cell[:3]
    alpha, beta, gamma = np.radians(cell[3:])
    cos_alpha, cos_beta, cos_gamma = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sin_gamma = np.sin(gamma)
    volume = np.sqrt(
        1
        - cos_alpha**2
        - cos_beta**2
        - cos_gamma**2
        + 2 * cos_alpha * cos_beta * cos_gamma
    )
    B = np.array(
        [
            [a, b * cos_gamma, c * cos_beta],
            [0, b * sin_gamma, c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma],
            [0, 0, c * volume / sin_gamma],
        ]
    )
    return B


def get_inv_fractional_matrix(cell):
    """
    Calculate the Cartesian-to-fractional transformation matrix.

    Computes the inverse of the fractional matrix for converting Cartesian
    coordinates to fractional coordinates.

    Parameters
    ----------
    cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    numpy.ndarray
        3x3 inverse transformation matrix B_inv such that frac = cart @ B_inv.T.
    """
    B = get_fractional_matrix(cell)
    B_inv = np.linalg.inv(B)
    return B_inv


def cartesian_to_fractional(xyz, cell):
    """
    Convert Cartesian coordinates to fractional coordinates.

    Parameters
    ----------
    xyz : numpy.ndarray
        Cartesian coordinates with shape (N, 3).
    cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    numpy.ndarray
        Fractional coordinates with shape (N, 3).
    """
    B_inv = get_inv_fractional_matrix(cell)
    xyz_fractional = np.dot(xyz, B_inv.T)
    return xyz_fractional


def fractional_to_cartesian(xyz_fractional, cell):
    """
    Convert fractional coordinates to Cartesian coordinates.

    Parameters
    ----------
    xyz_fractional : numpy.ndarray
        Fractional coordinates with shape (N, 3).
    cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    numpy.ndarray
        Cartesian coordinates with shape (N, 3).
    """
    B = get_fractional_matrix(cell)
    xyz = np.dot(xyz_fractional, B.T)
    return xyz


def convert_coords_to_fractional(df, cell):
    """
    Convert coordinates from a DataFrame to fractional coordinates.

    Extracts x, y, z columns from a DataFrame and converts them from
    Cartesian to fractional coordinates.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'x', 'y', 'z' columns with Cartesian coordinates.
    cell : numpy.ndarray or list
        Unit cell parameters [a, b, c, alpha, beta, gamma] where lengths are
        in Angstroms and angles are in degrees.

    Returns
    -------
    numpy.ndarray
        Fractional coordinates with shape (N, 3).
    """
    xyz = df[["x", "y", "z"]].values
    xyz_fractional = cartesian_to_fractional(xyz, cell)
    return xyz_fractional
