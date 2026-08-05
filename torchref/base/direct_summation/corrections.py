"""
Structure factor correction terms.

Functions for applying various correction terms to structure factors,
including anharmonic corrections and core deformation.
"""

import torch


def core_deformation(core_correction, s):
    """
    Apply core deformation correction to scattering.

    Parameters
    ----------
    core_correction : float or torch.Tensor
        Core correction factor.
    s : torch.Tensor
        Scattering vector magnitudes.

    Returns
    -------
    torch.Tensor
        Core deformation correction factors.
    """
    return 1.0 - core_correction * torch.exp(-s * s / 0.5)


def anharmonic_correction(hkl, c):
    """
    Apply anharmonic (third-order) correction to structure factors.

    Parameters
    ----------
    hkl : torch.Tensor
        Miller indices of shape (N_reflections, 3).
    c : tuple or list
        Ten anharmonic coefficients (C111, C222, C333, C112, C122, C113, C133, C223, C233, C123).

    Returns
    -------
    torch.Tensor
        Complex anharmonic correction factors of shape (N_reflections,).
    """
    h1, h2, h3 = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    # Gram-Charlier third-order tensor: symmetric cubic in (h1, h2, h3) using
    # all ten coefficients with their multiplicities (1 for diagonal terms, 3
    # for two-index terms, 6 for C123). The (-8j*pi**3) prefactor is the
    # Gram-Charlier (2*pi*i)**3 / 3! phase factor; the 6e7 divisor folds the
    # 3! = 6 with a 1e7 coefficient scale.
    C111, C222, C333, C112, C122, C113, C133, C223, C233, C123 = c
    third_order = (
        (
            C111 * h1**3
            + C222 * h2**3
            + C333 * h3**3
            + 3 * C112 * h1**2 * h2
            + 3 * C122 * h1 * h2**2
            + 3 * C113 * h1**2 * h3
            + 3 * C133 * h1 * h3**2
            + 3 * C223 * h2**2 * h3
            + 3 * C233 * h2 * h3**2
            + 6 * C123 * h1 * h2 * h3
        )
        * (-8j * torch.pi**3)
        / 6e7
    )
    return torch.exp(third_order)


def anharmonic_correction_no_complex(hkl, c):
    """
    Apply anharmonic (third-order) correction without complex numbers.

    Parameters
    ----------
    hkl : torch.Tensor
        Miller indices of shape (N_reflections, 3).
    c : tuple or list
        Ten anharmonic coefficients (C111, C222, C333, C112, C122, C113, C133, C223, C233, C123).

    Returns
    -------
    torch.Tensor
        Correction factors as [cos, sin] of shape (2, N_reflections).
    """
    h1, h2, h3 = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    # Same Gram-Charlier third-order tensor as anharmonic_correction, but the
    # purely real phase argument is returned as (cos, sin) rows instead of a
    # complex exponential. The (-8*pi**3)/6e7 factor matches the complex path
    # (3! = 6 folded with a 1e7 coefficient scale); see anharmonic_correction.
    C111, C222, C333, C112, C122, C113, C133, C223, C233, C123 = c
    third_order = (
        (
            C111 * h1**3
            + C222 * h2**3
            + C333 * h3**3
            + 3 * C112 * h1**2 * h2
            + 3 * C122 * h1 * h2**2
            + 3 * C113 * h1**2 * h3
            + 3 * C133 * h1 * h3**2
            + 3 * C223 * h2**2 * h3
            + 3 * C233 * h2 * h3**2
            + 6 * C123 * h1 * h2 * h3
        )
        * (-8 * torch.pi**3)
        / 6e7
    )
    return torch.vstack((torch.cos(third_order), torch.sin(third_order)))


def multiplication_quasi_complex_tensor(a, b):
    """
    Multiply two quasi-complex tensors represented as [real, imag] rows.

    Parameters
    ----------
    a : torch.Tensor
        First quasi-complex tensor of shape (2, N).
    b : torch.Tensor
        Second quasi-complex tensor of shape (2, N).

    Returns
    -------
    torch.Tensor
        Product as [real, imag] of shape (2, N).
    """
    real_part = a[0] * b[0] - a[1] * b[1]
    imag_part = a[0] * b[1] + a[1] * b[0]
    return torch.vstack((real_part, imag_part))
