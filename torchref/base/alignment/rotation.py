"""
Rotation utility functions.

Functions for coordinate rotations and rotation matrix conversions
including axis-angle, quaternion, and Euler representations.
"""

import numpy as np
import torch

from torchref.config import dtypes, get_float_dtype, normalize_device


def rotate_coords_torch(coords, phi, rho):
    """
    Rotate coordinates using phi and rho angles (PyTorch version).

    Parameters
    ----------
    coords : torch.Tensor
        Coordinates of shape (N, 3) to rotate.
    phi : torch.Tensor
        Rotation angle phi in degrees (scalar tensor; ``torch.cos``/``sin``
        are applied to it, so a Python float is not accepted).
    rho : torch.Tensor
        Rotation angle rho in degrees (scalar tensor).

    Returns
    -------
    torch.Tensor
        Rotated coordinates of shape (N, 3).

    Notes
    -----
    The two-angle (phi, rho) convention composes a rotation by ``phi`` about
    the z axis followed by a rotation by ``rho`` about the resulting x axis.
    """
    phi = phi * np.pi / 180
    rho = rho * np.pi / 180
    rot_matrix = torch.tensor(
        [
            [torch.cos(phi), -torch.sin(phi), 0],
            [
                torch.sin(phi) * torch.cos(rho),
                torch.cos(phi) * torch.cos(rho),
                -torch.sin(rho),
            ],
            [
                torch.sin(phi) * torch.sin(rho),
                torch.cos(phi) * torch.sin(rho),
                torch.cos(rho),
            ],
        ],
        dtype=coords.dtype,
        device=coords.device,
    )
    return torch.einsum("ij,kj->ki", rot_matrix, coords)


def rotate_coords_numpy(coords, phi, rho):
    """
    Rotate 3D coordinates by phi and rho angles (NumPy version).

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
    rot_matrix = np.array(
        [
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi) * np.cos(rho), np.cos(phi) * np.cos(rho), -np.sin(rho)],
            [np.sin(phi) * np.sin(rho), np.cos(phi) * np.sin(rho), np.cos(rho)],
        ],
        dtype=np.float64,
    )
    return np.einsum("ij,kj->ki", rot_matrix, coords)


def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle representation to 3x3 rotation matrix.

    Uses Rodrigues' formula. Supports batched input and gradients.

    Parameters
    ----------
    axis_angle : torch.Tensor
        Axis-angle representation with shape (3,) or (N, 3).
        Direction is rotation axis, magnitude is angle in radians.

    Returns
    -------
    torch.Tensor
        Rotation matrix with shape (3, 3) or (N, 3, 3).
    """
    if axis_angle.dim() == 1:
        axis_angle = axis_angle.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    # Handle zero rotation case
    axis = torch.where(
        angle > 1e-10,
        axis_angle / angle,
        torch.tensor(
            [0.0, 0.0, 1.0], device=axis_angle.device, dtype=axis_angle.dtype
        ).expand_as(axis_angle),
    )

    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    # where K is the skew-symmetric matrix of the axis
    K = torch.zeros(
        axis_angle.shape[0], 3, 3, device=axis_angle.device, dtype=axis_angle.dtype
    )
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0)
    angle = angle.unsqueeze(-1)

    R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.bmm(K, K)

    if squeeze:
        R = R.squeeze(0)

    return R


def rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to axis-angle representation.

    Parameters
    ----------
    R : torch.Tensor
        Rotation matrix with shape (3, 3) or (N, 3, 3).

    Returns
    -------
    torch.Tensor
        Axis-angle representation with shape (3,) or (N, 3).
    """
    if R.dim() == 2:
        R = R.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # Compute angle from trace: trace(R) = 1 + 2*cos(θ)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))

    # Compute axis from skew-symmetric part: (R - R^T) / (2*sin(θ))
    axis = torch.stack(
        [R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]],
        dim=-1,
    )

    # Normalize axis (handle small angles)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    axis = torch.where(
        axis_norm > 1e-10,
        axis / axis_norm,
        torch.tensor([0.0, 0.0, 1.0], device=R.device, dtype=R.dtype).expand_as(axis),
    )

    # Scale by angle
    result = axis * angle.unsqueeze(-1)

    if squeeze:
        result = result.squeeze(0)

    return result


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.

    Parameters
    ----------
    q : torch.Tensor
        Quaternion with shape (4,) or (N, 4). Format: [w, x, y, z].

    Returns
    -------
    torch.Tensor
        Rotation matrix with shape (3, 3) or (N, 3, 3).
    """
    if q.dim() == 1:
        q = q.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # Normalize quaternion
    q = q / torch.norm(q, dim=-1, keepdim=True)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    if squeeze:
        R = R.squeeze(0)

    return R


def random_rotation_uniform(
    n: int = 1,
    device: str = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Generate uniform random rotations over SO(3).

    Uses Shoemake's quaternion-based uniform sampling.

    Parameters
    ----------
    n : int, optional
        Number of rotations to generate. Default is 1.
    device : str or torch.device, optional
        Device for output tensor. Defaults to the configured default device.
    dtype : torch.dtype, optional
        Data type for output tensor. Default is dtypes.float.

    Returns
    -------
    torch.Tensor
        Rotation matrices with shape (n, 3, 3) or (3, 3) if n=1.
    """
    device = normalize_device(device)
    if dtype is None:
        dtype = get_float_dtype()
    # Sample uniform random numbers
    u = torch.rand(n, 3, device=device, dtype=dtype)

    # Convert to quaternion using Shoemake's method
    q = torch.zeros(n, 4, device=device, dtype=dtype)
    q[:, 0] = torch.sqrt(1 - u[:, 0]) * torch.sin(2 * np.pi * u[:, 1])
    q[:, 1] = torch.sqrt(1 - u[:, 0]) * torch.cos(2 * np.pi * u[:, 1])
    q[:, 2] = torch.sqrt(u[:, 0]) * torch.sin(2 * np.pi * u[:, 2])
    q[:, 3] = torch.sqrt(u[:, 0]) * torch.cos(2 * np.pi * u[:, 2])

    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(q)

    if n == 1:
        R = R.squeeze(0)

    return R


def rotation_matrix_euler_zyz(
    angles: torch.Tensor,
) -> torch.Tensor:
    """
    Create rotation matrix from ZYZ Euler angles (differentiable PyTorch version).

    R = Rz(alpha) @ Ry(beta) @ Rz(gamma)

    Parameters
    ----------
    angles : torch.Tensor
        Tensor of three rotation angles (alpha, beta, gamma) in radians.
        Or shape (B, 3) for batched input. The function will return (B, 3, 3) in that case.

    Returns
    -------
    torch.Tensor
        Rotation matrix of shape (3, 3), or (B, 3, 3) for batched input.
    """

    batched = True

    if angles.dim() == 1:
        angles = angles.unsqueeze(0)
        batched = False

    ca, sa = torch.cos(angles[:, 0]), torch.sin(angles[:, 0])
    cb, sb = torch.cos(angles[:, 1]), torch.sin(angles[:, 1])
    cg, sg = torch.cos(angles[:, 2]), torch.sin(angles[:, 2])

    # Build rotation matrix element by element
    R = torch.stack(
        [
            torch.stack(
                [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb], dim=1
            ),
            torch.stack(
                [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb], dim=1
            ),
            torch.stack([-sb * cg, sb * sg, cb], dim=1),
        ],
        dim=1,
    )

    return R if batched else R.squeeze(0)


def rotation_matrix_euler_xyz(
    angles: torch.Tensor,
) -> torch.Tensor:
    """
    Create rotation matrix from XYZ Euler angles (differentiable PyTorch version).

    R = Rz(gamma) @ Ry(beta) @ Rx(alpha)

    Parameters
    ----------
    angles : torch.Tensor
        Tensor of three rotation angles (alpha, beta, gamma) in radians,
        applied as Rx(alpha), Ry(beta), Rz(gamma) — outer product Rz·Ry·Rx.
        Shape (3,) or (B, 3); returns (3, 3) or (B, 3, 3) respectively.

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix (or batched (B, 3, 3)).

    Notes
    -----
    Rotating about distinct world axes avoids the gimbal-lock singularity at
    the origin that ZYZ has when beta=0 (where alpha and gamma both rotate
    about Z).
    """
    batched = True
    if angles.dim() == 1:
        angles = angles.unsqueeze(0)
        batched = False

    ca, sa = torch.cos(angles[:, 0]), torch.sin(angles[:, 0])
    cb, sb = torch.cos(angles[:, 1]), torch.sin(angles[:, 1])
    cg, sg = torch.cos(angles[:, 2]), torch.sin(angles[:, 2])

    # R = Rz(g) @ Ry(b) @ Rx(a). Expansion of the product:
    R = torch.stack([
        torch.stack([cg * cb,
                     cg * sb * sa - sg * ca,
                     cg * sb * ca + sg * sa], dim=1),
        torch.stack([sg * cb,
                     sg * sb * sa + cg * ca,
                     sg * sb * ca - cg * sa], dim=1),
        torch.stack([-sb,
                     cb * sa,
                     cb * ca], dim=1),
    ], dim=1)

    return R if batched else R.squeeze(0)
