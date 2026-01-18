"""
Rigid body transformations for crystallographic alignment.

Provides unified handling of rotations and translations with quaternion-based
internal storage and multiple representation formats.
"""

from typing import Optional, Union

import torch
import torch.nn as nn


# =============================================================================
# Quaternion Helper Functions
# =============================================================================

def get_inverse_rotation_matrix(R: torch.Tensor) -> torch.Tensor:
    """
    Compute inverse of rotation matrix (transpose for orthogonal matrices).

    Parameters
    ----------
    R : torch.Tensor
        Rotation matrix of shape (3, 3) or (N, 3, 3).

    Returns
    -------
    torch.Tensor
        Inverse rotation matrix of same shape.
    """
    return R.transpose(-2, -1)

def sample_angles(sampling_pitch_rad, max_angles_rad):
    """
    Sample Euler angles (in radians) up to the specified maximum angles with the given sampling pitch.
    Returns a tensor of shape (N, 3) where N is the number of sampled angles.
    
    Args:
        sampling_pitch_rad (float): Sampling pitch in radians.
        max_angles_rad (tuple): Maximum angles (alpha, beta, gamma) in radians.
    Returns:
        torch.Tensor: Sampled angles of shape (N, 3).

    """

    angles = []
    max_alpha, max_beta, max_gamma = max_angles_rad
    alpha = torch.arange(0, max_alpha + 1e-6, sampling_pitch_rad, dtype=torch.float32)
    beta = torch.arange(0, max_beta + 1e-6, sampling_pitch_rad, dtype=torch.float32)
    gamma = torch.arange(0, max_gamma + 1e-6, sampling_pitch_rad, dtype=torch.float32)
    alpha, beta, gamma = torch.meshgrid(alpha, beta, gamma, indexing='ij')

    return torch.stack([alpha.flatten(), beta.flatten(), gamma.flatten()], dim=-1)

def rotation_matrix_from_euler(angles):
    """
    Compute rotation matrices from Euler angles (in radians).
    Angles should be of shape (N, 3) where N is the number of angle sets.
    
    Args:
        angles (torch.Tensor): Euler angles of shape (N, 3).
    Returns:
        torch.Tensor: Rotation matrices of shape (N, 3, 3).
    """
    alpha = angles[:, 0]
    beta = angles[:, 1]
    gamma = angles[:, 2]

    R_alpha = torch.stack([
        torch.cos(alpha), -torch.sin(alpha), torch.zeros_like(alpha),
        torch.sin(alpha), torch.cos(alpha), torch.zeros_like(alpha),
        torch.zeros_like(alpha), torch.zeros_like(alpha), torch.ones_like(alpha)
    ], dim=-1).reshape(-1, 3, 3)

    R_beta = torch.stack([
        torch.cos(beta), torch.zeros_like(beta), torch.sin(beta),
        torch.zeros_like(beta), torch.ones_like(beta), torch.zeros_like(beta),
        -torch.sin(beta), torch.zeros_like(beta), torch.cos(beta)
    ], dim=-1).reshape(-1, 3, 3)

    R_gamma = torch.stack([
        torch.ones_like(gamma), torch.zeros_like(gamma), torch.zeros_like(gamma),
        torch.zeros_like(gamma), torch.cos(gamma), -torch.sin(gamma),
        torch.zeros_like(gamma), torch.sin(gamma), torch.cos(gamma)
    ], dim=-1).reshape(-1, 3, 3)

    R = torch.einsum('rij,rjk,rkl->ril', R_gamma, R_beta, R_alpha)

    return R

def quaternion_normalize(q: torch.Tensor) -> torch.Tensor:
    """
    Normalize quaternion to unit length.

    Parameters
    ----------
    q : torch.Tensor
        Quaternion(s) of shape (4,) or (N, 4).

    Returns
    -------
    torch.Tensor
        Normalized quaternion(s) of same shape.
    """
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion conjugate (inverse for unit quaternions).

    For q = [w, x, y, z], conjugate is [w, -x, -y, -z].

    Parameters
    ----------
    q : torch.Tensor
        Quaternion(s) of shape (4,) or (N, 4).

    Returns
    -------
    torch.Tensor
        Conjugate quaternion(s) of same shape.
    """
    conj = q.clone()
    conj[..., 1:] = -conj[..., 1:]
    return conj


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Compute Hamilton product of two quaternions.

    Parameters
    ----------
    q1, q2 : torch.Tensor
        Quaternions of shape (4,) or (N, 4). Format: [w, x, y, z].

    Returns
    -------
    torch.Tensor
        Product quaternion of same shape.
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def quaternion_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector(s) by quaternion using q * v * q^*.

    Parameters
    ----------
    q : torch.Tensor
        Unit quaternion of shape (4,).
    v : torch.Tensor
        Vector(s) of shape (3,) or (N, 3).

    Returns
    -------
    torch.Tensor
        Rotated vector(s) of same shape as v.
    """
    # Convert vector to pure quaternion [0, x, y, z]
    v_shape = v.shape
    v_flat = v.reshape(-1, 3)

    v_quat = torch.zeros(v_flat.shape[0], 4, dtype=q.dtype, device=q.device)
    v_quat[:, 1:] = v_flat

    # q * v * q^*
    q_conj = quaternion_conjugate(q)
    result = quaternion_multiply(quaternion_multiply(q.unsqueeze(0), v_quat), q_conj.unsqueeze(0))

    # Extract vector part
    return result[:, 1:].reshape(v_shape)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternion to 3x3 rotation matrix.

    Parameters
    ----------
    q : torch.Tensor
        Unit quaternion of shape (4,) or (N, 4). Format: [w, x, y, z].

    Returns
    -------
    torch.Tensor
        Rotation matrix of shape (3, 3) or (N, 3, 3).
    """
    q = quaternion_normalize(q)

    batched = q.dim() == 2
    if not batched:
        q = q.unsqueeze(0)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Rotation matrix from quaternion
    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], dim=-1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], dim=-1),
    ], dim=-2)

    if not batched:
        R = R.squeeze(0)

    return R


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to unit quaternion.

    Uses Shepperd's method for numerical stability.

    Parameters
    ----------
    R : torch.Tensor
        Rotation matrix of shape (3, 3) or (N, 3, 3).

    Returns
    -------
    torch.Tensor
        Unit quaternion of shape (4,) or (N, 4). Format: [w, x, y, z].
    """
    batched = R.dim() == 3
    if not batched:
        R = R.unsqueeze(0)

    batch_size = R.shape[0]
    dtype, device = R.dtype, R.device

    # Shepperd's method - choose largest diagonal element
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    q = torch.zeros(batch_size, 4, dtype=dtype, device=device)

    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

    # Case 2: R[0,0] is largest diagonal
    mask2 = ~mask1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

    # Case 3: R[1,1] is largest diagonal
    mask3 = ~mask1 & ~mask2 & (R[:, 1, 1] > R[:, 2, 2])
    if mask3.any():
        s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

    # Case 4: R[2,2] is largest diagonal
    mask4 = ~mask1 & ~mask2 & ~mask3
    if mask4.any():
        s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        q[mask4, 3] = 0.25 * s

    # Ensure positive w (canonical form)
    q = torch.where(q[:, 0:1] < 0, -q, q)

    if not batched:
        q = q.squeeze(0)

    return quaternion_normalize(q)


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle representation to quaternion.

    Parameters
    ----------
    axis_angle : torch.Tensor
        Axis-angle vector of shape (3,) or (N, 3).
        Direction is rotation axis, magnitude is angle in radians.

    Returns
    -------
    torch.Tensor
        Unit quaternion of shape (4,) or (N, 4).
    """
    batched = axis_angle.dim() == 2
    if not batched:
        axis_angle = axis_angle.unsqueeze(0)

    angle = axis_angle.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    axis = axis_angle / angle

    half_angle = angle / 2
    w = torch.cos(half_angle)
    xyz = axis * torch.sin(half_angle)

    q = torch.cat([w, xyz], dim=-1)

    if not batched:
        q = q.squeeze(0)

    return q


def quaternion_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to axis-angle representation.

    Parameters
    ----------
    q : torch.Tensor
        Unit quaternion of shape (4,) or (N, 4).

    Returns
    -------
    torch.Tensor
        Axis-angle vector of shape (3,) or (N, 3).
    """
    q = quaternion_normalize(q)

    batched = q.dim() == 2
    if not batched:
        q = q.unsqueeze(0)

    # Ensure positive w for numerical stability
    q = torch.where(q[:, 0:1] < 0, -q, q)

    w = q[:, 0].clamp(-1 + 1e-7, 1 - 1e-7)
    xyz = q[:, 1:]

    angle = 2 * torch.acos(w)
    sin_half = torch.sqrt(1 - w * w).clamp(min=1e-12)

    axis = xyz / sin_half.unsqueeze(-1)
    axis_angle = axis * angle.unsqueeze(-1)

    # Handle near-identity case
    near_identity = angle < 1e-6
    axis_angle = torch.where(
        near_identity.unsqueeze(-1),
        2 * xyz,  # Small angle approximation
        axis_angle
    )

    if not batched:
        axis_angle = axis_angle.squeeze(0)

    return axis_angle


def quaternion_to_euler_zyz(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to Euler angles (ZYZ convention).

    Parameters
    ----------
    q : torch.Tensor
        Unit quaternion of shape (4,).

    Returns
    -------
    torch.Tensor
        Euler angles [alpha, beta, gamma] of shape (3,).
        Ranges: alpha in [0, 2pi), beta in [0, pi], gamma in [0, 2pi).
    """
    # Convert to matrix first, then extract ZYZ angles
    R = quaternion_to_matrix(q)

    # ZYZ convention: R = Rz(alpha) @ Ry(beta) @ Rz(gamma)
    # beta = acos(R[2,2])
    # alpha = atan2(R[1,2], R[0,2])
    # gamma = atan2(R[2,1], -R[2,0])

    beta = torch.acos(R[2, 2].clamp(-1, 1))

    # Handle gimbal lock cases
    if torch.abs(torch.sin(beta)) < 1e-6:
        # beta ≈ 0 or pi, gimbal lock
        alpha = torch.atan2(R[1, 0], R[0, 0])
        gamma = torch.zeros_like(alpha)
    else:
        alpha = torch.atan2(R[1, 2], R[0, 2])
        gamma = torch.atan2(R[2, 1], -R[2, 0])

    # Normalize to [0, 2pi) for alpha and gamma
    alpha = torch.remainder(alpha, 2 * torch.pi)
    gamma = torch.remainder(gamma, 2 * torch.pi)

    return torch.stack([alpha, beta, gamma])


def euler_zyz_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles (ZYZ convention) to quaternion.

    Parameters
    ----------
    euler : torch.Tensor
        Euler angles [alpha, beta, gamma] of shape (3,).

    Returns
    -------
    torch.Tensor
        Unit quaternion of shape (4,).
    """
    alpha, beta, gamma = euler[0], euler[1], euler[2]

    # ZYZ: q = qz(alpha) * qy(beta) * qz(gamma)
    ca, sa = torch.cos(alpha / 2), torch.sin(alpha / 2)
    cb, sb = torch.cos(beta / 2), torch.sin(beta / 2)
    cg, sg = torch.cos(gamma / 2), torch.sin(gamma / 2)

    # qz(alpha) = [cos(a/2), 0, 0, sin(a/2)]
    # qy(beta) = [cos(b/2), 0, sin(b/2), 0]
    # qz(gamma) = [cos(g/2), 0, 0, sin(g/2)]

    q_alpha = torch.stack([ca, torch.zeros_like(ca), torch.zeros_like(ca), sa])
    q_beta = torch.stack([cb, torch.zeros_like(cb), sb, torch.zeros_like(cb)])
    q_gamma = torch.stack([cg, torch.zeros_like(cg), torch.zeros_like(cg), sg])

    return quaternion_multiply(quaternion_multiply(q_alpha, q_beta), q_gamma)


# =============================================================================
# RigidTransform Class
# =============================================================================


class RigidTransform(nn.Module):
    """
    Rigid body transformation with quaternion-based rotation storage.

    Stores rotation internally as unit quaternion [w, x, y, z] and
    translation as 3D vector. Provides methods for various representations
    and transformation operations.

    Parameters
    ----------
    quaternion : torch.Tensor, optional
        Rotation as quaternion [w, x, y, z] of shape (4,).
    translation : torch.Tensor, optional
        Translation vector of shape (3,). Defaults to zeros.
    rotation_matrix : torch.Tensor, optional
        Alternative: initialize from rotation matrix of shape (3, 3).
    axis_angle : torch.Tensor, optional
        Alternative: initialize from axis-angle vector of shape (3,).

    Examples
    --------
    >>> T = RigidTransform.random()
    >>> coords = torch.randn(100, 3)
    >>> coords_transformed = T.apply(coords)
    >>> coords_back = T.inverse().apply(coords_transformed)
    """

    def __init__(
        self,
        quaternion: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None,
        rotation_matrix: Optional[torch.Tensor] = None,
        axis_angle: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Determine dtype and device from inputs
        dtype = torch.float64
        device = torch.device("cpu")

        if quaternion is not None:
            dtype, device = quaternion.dtype, quaternion.device
        elif rotation_matrix is not None:
            dtype, device = rotation_matrix.dtype, rotation_matrix.device
        elif axis_angle is not None:
            dtype, device = axis_angle.dtype, axis_angle.device
        elif translation is not None:
            dtype, device = translation.dtype, translation.device

        # Convert to quaternion from alternative representations
        if quaternion is not None:
            q = quaternion_normalize(quaternion)
        elif rotation_matrix is not None:
            q = matrix_to_quaternion(rotation_matrix)
        elif axis_angle is not None:
            q = axis_angle_to_quaternion(axis_angle)
        else:
            # Identity rotation
            q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)

        # Set translation
        if translation is not None:
            t = translation.to(dtype=dtype, device=device)
        else:
            t = torch.zeros(3, dtype=dtype, device=device)

        # Register as buffers (not parameters by default)
        self.register_buffer("_quaternion", q)
        self.register_buffer("_translation", t)

    # =========================================================================
    # Properties for different representations
    # =========================================================================

    @property
    def quaternion(self) -> torch.Tensor:
        """Get rotation as quaternion [w, x, y, z]."""
        return self._quaternion

    @property
    def rotation_matrix(self) -> torch.Tensor:
        """Get rotation as 3x3 matrix."""
        return quaternion_to_matrix(self._quaternion)

    @property
    def axis_angle(self) -> torch.Tensor:
        """Get rotation as axis-angle vector."""
        return quaternion_to_axis_angle(self._quaternion)

    @property
    def euler_zyz(self) -> torch.Tensor:
        """Get rotation as Euler angles (ZYZ convention)."""
        return quaternion_to_euler_zyz(self._quaternion)

    @property
    def translation(self) -> torch.Tensor:
        """Get translation vector."""
        return self._translation

    @property
    def dtype(self) -> torch.dtype:
        """Get data type."""
        return self._quaternion.dtype

    @property
    def device(self) -> torch.device:
        """Get device."""
        return self._quaternion.device

    # =========================================================================
    # Transformation methods
    # =========================================================================

    def apply(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation to coordinates: x' = R @ x + t.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates of shape (N, 3) or (3,).

        Returns
        -------
        torch.Tensor
            Transformed coordinates of same shape.
        """
        R = self.rotation_matrix
        coords_dtype = coords.dtype
        coords = coords.to(dtype=self.dtype)

        if coords.dim() == 1:
            result = R @ coords + self._translation
        else:
            result = coords @ R.T + self._translation

        return result.to(dtype=coords_dtype)

    def apply_rotation_only(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation only: x' = R @ x.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates of shape (N, 3) or (3,).

        Returns
        -------
        torch.Tensor
            Rotated coordinates of same shape.
        """
        R = self.rotation_matrix
        coords_dtype = coords.dtype
        coords = coords.to(dtype=self.dtype)

        if coords.dim() == 1:
            result = R @ coords
        else:
            result = coords @ R.T

        return result.to(dtype=coords_dtype)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """nn.Module forward = apply()."""
        return self.apply(coords)

    # =========================================================================
    # Composition and inversion
    # =========================================================================

    def inverse(self) -> "RigidTransform":
        """
        Compute inverse transformation.

        For T(x) = R @ x + t, inverse is T^{-1}(x) = R^T @ (x - t).

        Returns
        -------
        RigidTransform
            Inverse transformation.
        """
        q_inv = quaternion_conjugate(self._quaternion)
        # t_inv = -R^T @ t = -R_inv @ t
        t_inv = -quaternion_rotate(q_inv, self._translation)
        return RigidTransform(quaternion=q_inv, translation=t_inv)

    def compose(self, other: "RigidTransform") -> "RigidTransform":
        """
        Compose with another transformation: (self @ other)(x) = self(other(x)).

        Parameters
        ----------
        other : RigidTransform
            Transformation to compose with.

        Returns
        -------
        RigidTransform
            Composed transformation.
        """
        q_new = quaternion_multiply(self._quaternion, other._quaternion)
        # t_new = R_self @ t_other + t_self
        t_new = quaternion_rotate(self._quaternion, other._translation) + self._translation
        return RigidTransform(quaternion=q_new, translation=t_new)

    def __matmul__(self, other: Union["RigidTransform", torch.Tensor]):
        """
        Support @ operator for composition or application.

        Parameters
        ----------
        other : RigidTransform or torch.Tensor
            If RigidTransform, composes transformations.
            If tensor, applies transformation to coordinates.

        Returns
        -------
        RigidTransform or torch.Tensor
            Composed transformation or transformed coordinates.
        """
        if isinstance(other, RigidTransform):
            return self.compose(other)
        return self.apply(other)

    # =========================================================================
    # Factory methods
    # =========================================================================

    @classmethod
    def identity(
        cls,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "RigidTransform":
        """
        Create identity transformation.

        Parameters
        ----------
        device : str or torch.device
            Device for tensors.
        dtype : torch.dtype
            Data type for tensors.

        Returns
        -------
        RigidTransform
            Identity transformation.
        """
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
        t = torch.zeros(3, dtype=dtype, device=device)
        return cls(quaternion=q, translation=t)

    @classmethod
    def from_matrix(
        cls,
        R: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> "RigidTransform":
        """
        Create from rotation matrix and translation.

        Parameters
        ----------
        R : torch.Tensor
            Rotation matrix of shape (3, 3).
        t : torch.Tensor, optional
            Translation vector of shape (3,).

        Returns
        -------
        RigidTransform
            Transformation from matrix representation.
        """
        return cls(rotation_matrix=R, translation=t)

    @classmethod
    def from_axis_angle(
        cls,
        axis_angle: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> "RigidTransform":
        """
        Create from axis-angle and translation.

        Parameters
        ----------
        axis_angle : torch.Tensor
            Axis-angle vector of shape (3,).
        t : torch.Tensor, optional
            Translation vector of shape (3,).

        Returns
        -------
        RigidTransform
            Transformation from axis-angle representation.
        """
        return cls(axis_angle=axis_angle, translation=t)

    @classmethod
    def from_euler_zyz(
        cls,
        euler: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> "RigidTransform":
        """
        Create from Euler angles (ZYZ convention) and translation.

        Parameters
        ----------
        euler : torch.Tensor
            Euler angles [alpha, beta, gamma] of shape (3,).
        t : torch.Tensor, optional
            Translation vector of shape (3,).

        Returns
        -------
        RigidTransform
            Transformation from Euler angle representation.
        """
        q = euler_zyz_to_quaternion(euler)
        return cls(quaternion=q, translation=t)

    @classmethod
    def random(
        cls,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float64,
        translation_scale: float = 0.0,
    ) -> "RigidTransform":
        """
        Create random transformation with uniform rotation over SO(3).

        Uses Shoemake's quaternion-based uniform sampling.

        Parameters
        ----------
        device : str or torch.device
            Device for tensors.
        dtype : torch.dtype
            Data type for tensors.
        translation_scale : float
            Scale for random translation (0 for no translation).

        Returns
        -------
        RigidTransform
            Random transformation.
        """
        # Shoemake's uniform random quaternion
        u1, u2, u3 = torch.rand(3, dtype=dtype, device=device)

        q = torch.stack([
            torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2),
            torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2),
            torch.sqrt(u1) * torch.sin(2 * torch.pi * u3),
            torch.sqrt(u1) * torch.cos(2 * torch.pi * u3),
        ])

        # Reorder to [w, x, y, z] format
        q = torch.stack([q[3], q[0], q[1], q[2]])

        if translation_scale > 0:
            t = torch.randn(3, dtype=dtype, device=device) * translation_scale
        else:
            t = torch.zeros(3, dtype=dtype, device=device)

        return cls(quaternion=q, translation=t)

    # =========================================================================
    # Utility methods
    # =========================================================================

    def detach(self) -> "RigidTransform":
        """
        Return detached copy (no gradient tracking).

        Returns
        -------
        RigidTransform
            Detached transformation.
        """
        return RigidTransform(
            quaternion=self._quaternion.detach(),
            translation=self._translation.detach(),
        )

    def clone(self) -> "RigidTransform":
        """
        Return deep copy.

        Returns
        -------
        RigidTransform
            Cloned transformation.
        """
        return RigidTransform(
            quaternion=self._quaternion.clone(),
            translation=self._translation.clone(),
        )

    def __repr__(self) -> str:
        q = self._quaternion
        t = self._translation
        return (
            f"RigidTransform(\n"
            f"  quaternion=[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}],\n"
            f"  translation=[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]\n"
            f")"
        )
