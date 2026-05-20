"""
Normalization functions for E-value conversion and anisotropy correction.

This module provides GPU-accelerated PyTorch implementations of:
- Radial shell computation and assignment
- Anisotropy correction for F² values
- E-value normalization within resolution shells

These functions are used for molecular replacement and related analyses.
"""

from typing import Optional, Tuple

import torch

from torchref.base.math_torch import U_to_matrix
from torchref.config import get_default_device


def compute_radial_shells(
    d_min: float,
    d_max: float,
    n_shells: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute uniform radial shell boundaries in reciprocal space.

    Shells are spaced uniformly in 1/d (|s|) for even coverage of resolution.

    Parameters
    ----------
    d_min : float
        High resolution limit in Angstroms.
    d_max : float
        Low resolution limit in Angstroms.
    n_shells : int
        Number of radial shells.
    device : torch.device, optional
        Device for output tensors. Default is CPU.

    Returns
    -------
    shell_edges : torch.Tensor
        Shell boundaries in Angstroms^-1, shape (n_shells+1,).
    shell_centers : torch.Tensor
        Shell centers in Angstroms^-1, shape (n_shells,).
    """
    if device is None:
        device = get_default_device()

    s_min = 1.0 / d_max  # Low resolution end
    s_max = 1.0 / d_min  # High resolution end

    shell_edges = torch.linspace(s_min, s_max, n_shells + 1, device=device)
    shell_centers = 0.5 * (shell_edges[:-1] + shell_edges[1:])

    return shell_edges, shell_centers


def assign_to_shells(
    s_mag: torch.Tensor,
    shell_edges: torch.Tensor,
) -> torch.Tensor:
    """
    Assign reflections to radial shells.

    Parameters
    ----------
    s_mag : torch.Tensor
        |s| values in Angstroms^-1, shape (N,).
    shell_edges : torch.Tensor
        Shell boundaries in Angstroms^-1, shape (n_shells+1,).

    Returns
    -------
    shell_idx : torch.Tensor
        Shell index for each reflection, shape (N,).
        Values 0 to n_shells-1, or -1 for out-of-range.
    """
    # bucketize gives index of the bin (right edge)
    # We subtract 1 to get the shell index
    shell_idx = torch.bucketize(s_mag, shell_edges[1:-1])

    n_shells = len(shell_edges) - 1

    # Mark out-of-range as -1
    out_of_range = (s_mag < shell_edges[0]) | (s_mag >= shell_edges[-1])
    shell_idx = torch.where(out_of_range, torch.tensor(-1, device=s_mag.device), shell_idx)

    return shell_idx


def compute_anisotropy_correction(
    s_vectors: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """
    Compute anisotropic correction factor for F² values.

    The correction is: exp(-2*pi^2 * s^T U s)

    This scales F² values to correct for anisotropic diffraction
    before normalizing to E-values.

    Parameters
    ----------
    s_vectors : torch.Tensor
        Reciprocal space vectors in Angstroms^-1, shape (N, 3).
    U : torch.Tensor
        Anisotropic parameters [u11, u22, u33, u12, u13, u23], shape (6,).

    Returns
    -------
    correction : torch.Tensor
        Correction factors, shape (N,).
    """
    U_matrix = U_to_matrix(U)  # (3, 3)

    # exp(-2*pi^2 * s^T U s)
    # sUs = s @ U @ s^T for each s
    sUs = torch.einsum("ni,ij,nj->n", s_vectors, U_matrix, s_vectors)
    exponent = -2 * (torch.pi**2) * sUs

    # Clamp to avoid numerical issues
    exponent = torch.clamp(exponent, -10.0, 10.0)

    return torch.exp(exponent)


def compute_shell_cv(
    F_squared: torch.Tensor,
    shell_idx: torch.Tensor,
    n_shells: int,
    min_count: int = 10,
) -> float:
    """
    Compute mean coefficient of variation of F² values within resolution shells.

    After proper anisotropy correction, F² should have similar CV
    in all directions within each resolution shell.

    Parameters
    ----------
    F_squared : torch.Tensor
        F² values, shape (N,).
    shell_idx : torch.Tensor
        Shell assignments, shape (N,).
    n_shells : int
        Number of shells.
    min_count : int
        Minimum reflections per shell to include in calculation.

    Returns
    -------
    mean_cv : float
        Mean coefficient of variation across shells.
    """
    total_cv = 0.0
    n_valid_shells = 0

    for p in range(n_shells):
        mask = shell_idx == p
        count = mask.sum().item()

        if count < min_count:
            continue

        F2_shell = F_squared[mask]
        mean_shell = F2_shell.mean()

        if mean_shell > 1e-10:
            cv = F2_shell.std() / mean_shell
            total_cv += cv.item()
            n_valid_shells += 1

    return total_cv / max(n_valid_shells, 1)


def fit_anisotropy_correction(
    F_squared: torch.Tensor,
    s_vectors: torch.Tensor,
    n_shells: int = 20,
    d_min: float = 4.0,
    d_max: float = 50.0,
    n_iterations: int = 100,
    lr: float = 0.01,
    verbose: bool = True,
) -> Tuple[torch.Tensor, float]:
    """
    Fit anisotropy correction parameters to minimize variance within shells.

    Optimizes U parameters so that corrected F² values have minimal
    coefficient of variation within each resolution shell, making the
    distribution more isotropic before E-value conversion.

    Uses PyTorch's LBFGS optimizer for efficient optimization.

    Parameters
    ----------
    F_squared : torch.Tensor
        F² values, shape (N,).
    s_vectors : torch.Tensor
        Reciprocal space vectors in Angstroms^-1, shape (N, 3).
    n_shells : int
        Number of resolution shells for variance calculation.
    d_min : float
        High resolution limit in Angstroms.
    d_max : float
        Low resolution limit in Angstroms.
    n_iterations : int
        Number of optimization iterations.
    lr : float
        Learning rate for optimizer.
    verbose : bool
        Print progress.

    Returns
    -------
    U_optimal : torch.Tensor
        Optimal anisotropic parameters, shape (6,).
    final_cv : float
        Final mean coefficient of variation after correction.
    """
    device = F_squared.device

    # Compute shell assignments
    shell_edges, _ = compute_radial_shells(d_min, d_max, n_shells, device=device)
    s_mag = torch.linalg.norm(s_vectors, dim=1)
    shell_idx = assign_to_shells(s_mag, shell_edges)

    # Filter to valid shells
    valid = shell_idx >= 0
    F2_valid = F_squared[valid]
    s_valid = s_vectors[valid]
    shell_valid = shell_idx[valid]

    # Initial CV
    initial_cv = compute_shell_cv(F2_valid, shell_valid, n_shells)

    if verbose:
        print("Anisotropy correction fitting:")
        print(f"  Initial mean CV: {initial_cv:.6f}")

    # Initialize U parameters (start from identity = no anisotropy)
    U = torch.zeros(6, dtype=F_squared.dtype, device=device, requires_grad=True)

    # Use LBFGS optimizer
    optimizer = torch.optim.LBFGS([U], lr=lr, max_iter=20, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        correction = compute_anisotropy_correction(s_valid, U)
        F2_corrected = F2_valid * correction

        # Compute loss as mean CV (we need gradients, so compute differentiably)
        total_loss = torch.tensor(0.0, device=device)
        n_valid_shells = 0

        for p in range(n_shells):
            mask = shell_valid == p
            count = mask.sum()

            if count < 10:
                continue

            F2_shell = F2_corrected[mask]
            mean_shell = F2_shell.mean()

            if mean_shell > 1e-10:
                cv = F2_shell.std() / mean_shell
                total_loss = total_loss + cv
                n_valid_shells += 1

        if n_valid_shells > 0:
            total_loss = total_loss / n_valid_shells

        total_loss.backward()
        return total_loss

    # Run optimization
    for _ in range(n_iterations // 20):  # LBFGS does multiple steps per call
        optimizer.step(closure)

    # Get final CV
    with torch.no_grad():
        correction = compute_anisotropy_correction(s_valid, U)
        F2_corrected = F2_valid * correction
        final_cv = compute_shell_cv(F2_corrected, shell_valid, n_shells)

    U_optimal = U.detach()

    if verbose:
        print(f"  Final mean CV: {final_cv:.6f}")
        print(f"  CV reduction: {100*(1 - final_cv/initial_cv):.1f}%")
        print(f"  U parameters: [{', '.join(f'{u:.4f}' for u in U_optimal.cpu().numpy())}]")

        # Print the correction magnitude in principal directions
        U_matrix = U_to_matrix(U_optimal)
        eigenvalues = torch.linalg.eigvalsh(U_matrix)
        print(f"  U eigenvalues: [{', '.join(f'{e:.4f}' for e in eigenvalues.cpu().numpy())}]")

    return U_optimal, final_cv


def apply_anisotropy_correction(
    F_squared: torch.Tensor,
    s_vectors: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """
    Apply anisotropic correction to F² values.

    Should be called BEFORE converting F² to E-values.

    Parameters
    ----------
    F_squared : torch.Tensor
        F² values, shape (N,).
    s_vectors : torch.Tensor
        Reciprocal space vectors in Angstroms^-1, shape (N, 3).
    U : torch.Tensor
        Anisotropic parameters, shape (6,).

    Returns
    -------
    F2_corrected : torch.Tensor
        Corrected F² values, shape (N,).
    """
    correction = compute_anisotropy_correction(s_vectors, U)
    return F_squared * correction


def F_squared_to_E_values(
    F_squared: torch.Tensor,
    s_vectors: torch.Tensor,
    n_shells: int = 20,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert F² values to E-values by normalizing within resolution shells.

    E-values are normalized structure factors where <E²> = 1 within each
    resolution shell.

    Parameters
    ----------
    F_squared : torch.Tensor
        F² values, shape (N,).
    s_vectors : torch.Tensor
        Reciprocal space vectors in Angstroms^-1, shape (N, 3).
    n_shells : int
        Number of resolution shells for normalization.
    d_min : float, optional
        High resolution limit in Angstroms. If None, derived from s_vectors.
    d_max : float, optional
        Low resolution limit in Angstroms. If None, derived from s_vectors.

    Returns
    -------
    E_values : torch.Tensor
        Normalized E-values, shape (N,).
    E_squared : torch.Tensor
        E² values (for correlation calculations), shape (N,).
    shell_idx : torch.Tensor
        Shell assignments for each reflection, shape (N,).
    """
    device = F_squared.device
    s_mag = torch.linalg.norm(s_vectors, dim=1)

    # Determine resolution limits if not provided
    if d_min is None:
        d_min = 1.0 / s_mag.max().item()
    if d_max is None:
        d_max = 1.0 / s_mag[s_mag > 0].min().item()

    shell_edges, _ = compute_radial_shells(d_min, d_max, n_shells, device=device)
    shell_idx = assign_to_shells(s_mag, shell_edges)

    # Initialize output
    E_squared = torch.zeros_like(F_squared)
    E_values = torch.zeros_like(F_squared)

    for p in range(n_shells):
        mask = shell_idx == p
        count = mask.sum().item()

        if count < 2:
            continue

        F2_shell = F_squared[mask]
        mean_F2 = F2_shell.mean()

        if mean_F2 > 1e-10:
            # E² = F² / <F²> so that <E²> = 1
            E2_shell = F2_shell / mean_F2
            E_squared[mask] = E2_shell
            # E = sqrt(E²) with sign preservation isn't meaningful for intensities
            # E is typically defined as sqrt(E²)
            E_values[mask] = torch.sqrt(E2_shell.clamp(min=0))

    return E_values, E_squared, shell_idx
