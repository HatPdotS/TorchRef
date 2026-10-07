"""Euler-angle rotation matrices, differentiable and batched."""

import torch


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

    R = Rz(gamma) @ Ry(beta) @ Rx(alpha). Distinct world axes, so unlike ZYZ
    there is no gimbal-lock singularity at beta=0.

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
