"""Local orthonormal frames anchored on three atoms, and points expressed in them.

A hydrogen that rides on its parent is stored as a constant offset in a frame built
from the parent ``p`` and two reference heavy atoms ``n1``, ``n2``::

    e1 = unit(n1 - p)
    e2 = unit((n2 - p) orthogonal to e1)
    e3 = e1 x e2
    h  = p + lx*e1 + ly*e2 + lz*e3

Every function here is a pure tensor op on already-gathered positions, so autograd
carries a force on ``h`` back onto ``p``, ``n1`` and ``n2`` through the exact frame
Jacobian. Coordinates are Cartesian Angstroms unless the caller chooses otherwise; the
only scale-dependent constant is ``eps``, which floors norms before division.
"""

from typing import Tuple

import torch

#: Norm floor in the same units as the coordinates (Angstroms here).
DEFAULT_EPS = 1e-8

#: A frame whose reference bonds are shorter than this, or whose ``n1-p-n2`` angle has
#: a sine below ``MIN_FRAME_SINE``, is treated as degenerate and placed rigidly.
MIN_FRAME_NORM = 1e-3
MIN_FRAME_SINE = 0.1


def rotate_vectors(vectors: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate Cartesian vectors by axis-angle rotation vectors.

    Parameters
    ----------
    vectors : torch.Tensor
        Cartesian vectors, shape ``(..., 3)``, in Å.
    rotation : torch.Tensor
        Rotation vectors with the same shape as ``vectors``, in radians.
        Direction gives the axis and length gives the right-handed angle.

    Returns
    -------
    torch.Tensor
        Rotated vectors in Å. First and second derivatives are finite at zero.
    """
    angle2 = rotation.square().sum(-1, keepdim=True)
    # Both torch.where branches must be safe at zero. The series also avoids
    # cancellation in (1 - cos(angle)) / angle**2 in single precision.
    angle = angle2.clamp_min(1e-4).sqrt()
    small = angle2 < 1e-4
    a = torch.where(
        small, 1 - angle2 / 6 + angle2.square() / 120, torch.sin(angle) / angle
    )
    b = torch.where(
        small,
        0.5 - angle2 / 24 + angle2.square() / 720,
        0.5 * torch.sinc(angle / (2 * torch.pi)).square(),
    )
    cross = torch.cross(rotation, vectors, dim=-1)
    return vectors + a * cross + b * torch.cross(rotation, cross, dim=-1)


def local_frame_axes(
    p: torch.Tensor,
    n1: torch.Tensor,
    n2: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-handed orthonormal axes of the frame anchored at ``p``.

    Parameters
    ----------
    p, n1, n2 : torch.Tensor
        Cartesian positions, shape ``(H, 3)``, same dtype and device.
    eps : float
        Norm floor guarding a collapsed reference bond.

    Returns
    -------
    e1, e2, e3 : torch.Tensor
        Unit vectors, each ``(H, 3)``. ``e1`` points along ``n1 - p``, ``e2`` lies in
        the ``p, n1, n2`` plane, ``e3 = e1 x e2``.
    """
    a = n1 - p
    e1 = a / a.norm(dim=-1, keepdim=True).clamp(min=eps)
    b = n2 - p
    b_perp = b - (b * e1).sum(-1, keepdim=True) * e1
    e2 = b_perp / b_perp.norm(dim=-1, keepdim=True).clamp(min=eps)
    e3 = torch.cross(e1, e2, dim=-1)
    return e1, e2, e3


def place_local_frame(
    p: torch.Tensor,
    n1: torch.Tensor,
    n2: torch.Tensor,
    local_offset: torch.Tensor,
    frame_valid: torch.Tensor,
    rigid_offset: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Positions of points stored as local-frame offsets.

    Parameters
    ----------
    p, n1, n2 : torch.Tensor
        Frame atoms, each ``(H, 3)``. Rows flagged invalid may hold any in-bounds
        position; their frame is not used.
    local_offset : torch.Tensor
        Coordinates in the ``(e1, e2, e3)`` frame, shape ``(H, 3)``.
    frame_valid : torch.Tensor
        Boolean ``(H,)``; where False the point is placed rigidly at
        ``p + rigid_offset``.
    rigid_offset : torch.Tensor
        Cartesian ``p -> point`` vector for the rigid fallback, shape ``(H, 3)``.
    eps : float
        Norm floor guarding degenerate frames.

    Returns
    -------
    torch.Tensor
        Cartesian positions, shape ``(H, 3)``, differentiable in ``p``, ``n1``, ``n2``.
    """
    e1, e2, e3 = local_frame_axes(p, n1, n2, eps)
    h_frame = (
        p
        + local_offset[:, 0:1] * e1
        + local_offset[:, 1:2] * e2
        + local_offset[:, 2:3] * e3
    )
    h_rigid = p + rigid_offset
    return torch.where(frame_valid.unsqueeze(-1), h_frame, h_rigid)


def local_frame_coordinates(
    p: torch.Tensor,
    n1: torch.Tensor,
    n2: torch.Tensor,
    point: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Inverse of :func:`place_local_frame`: express ``point`` in the frame at ``p``.

    Parameters
    ----------
    p, n1, n2, point : torch.Tensor
        Cartesian positions, each ``(H, 3)``.
    eps : float
        Norm floor guarding degenerate frames.

    Returns
    -------
    torch.Tensor
        Local coordinates ``(lx, ly, lz)``, shape ``(H, 3)``, such that
        ``place_local_frame(p, n1, n2, result, True, ...)`` returns ``point``.
    """
    e1, e2, e3 = local_frame_axes(p, n1, n2, eps)
    d = point - p
    return torch.stack([(d * e1).sum(-1), (d * e2).sum(-1), (d * e3).sum(-1)], dim=-1)


def frame_is_degenerate(
    p: torch.Tensor,
    n1: torch.Tensor,
    n2: torch.Tensor,
    min_norm: float = MIN_FRAME_NORM,
    min_sine: float = MIN_FRAME_SINE,
) -> torch.Tensor:
    """Frames too ill-conditioned to carry an offset.

    A frame is degenerate when either reference bond is shorter than ``min_norm`` or
    the two reference bonds are within ``asin(min_sine)`` of collinear, in which case
    ``e2`` is set by numerical noise and a riding point would swing with it.

    Parameters
    ----------
    p, n1, n2 : torch.Tensor
        Cartesian positions, each ``(H, 3)``.
    min_norm : float
        Shortest acceptable reference bond, same units as the coordinates.
    min_sine : float
        Smallest acceptable ``|sin(angle(n1 - p, n2 - p))|``.

    Returns
    -------
    torch.Tensor
        Boolean ``(H,)``, True where the frame must not be used.
    """
    a = n1 - p
    b = n2 - p
    na = a.norm(dim=-1)
    nb = b.norm(dim=-1)
    cross = torch.cross(a, b, dim=-1).norm(dim=-1)
    sine = cross / (na * nb).clamp(min=min_norm * min_norm)
    return (na < min_norm) | (nb < min_norm) | (sine < min_sine)


__all__ = [
    "DEFAULT_EPS",
    "MIN_FRAME_NORM",
    "MIN_FRAME_SINE",
    "rotate_vectors",
    "local_frame_axes",
    "place_local_frame",
    "local_frame_coordinates",
    "frame_is_degenerate",
]
