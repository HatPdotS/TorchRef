"""
PyTorch implementations of mathematical functions for crystallography.

.. deprecated::
    This module is deprecated. Please import from domain-specific submodules:

    - ``torchref.base.coordinates`` - Coordinate transformations
    - ``torchref.base.reciprocal`` - Reciprocal space calculations
    - ``torchref.base.structure_factors`` - Structure factor calculations
    - ``torchref.base.electron_density`` - Electron density map building
    - ``torchref.base.fourier`` - FFT operations
    - ``torchref.base.scattering`` - Atomic scattering factors
    - ``torchref.base.alignment`` - Coordinate alignment
    - ``torchref.base.metrics`` - R-factors and loss functions
    - ``torchref.base.kernels`` - Optimized kernels

This module is maintained for backward compatibility and re-exports all
functions from the new submodules.

Example (new style - recommended)::

    from torchref.base.coordinates import cartesian_to_fractional_torch
    from torchref.base.metrics import get_rfactor_torch

Example (old style - still works)::

    from torchref.base.math_torch import cartesian_to_fractional_torch
"""

import hashlib
import torch

# =============================================================================
# Re-exports from coordinates submodule
# =============================================================================
from torchref.base.coordinates import (
    cartesian_to_fractional_torch,
    fractional_to_cartesian_torch,
    get_fractional_matrix,
    get_inv_fractional_matrix_torch,
    smallest_diff,
    smallest_diff_aniso,
)

# =============================================================================
# Re-exports from reciprocal submodule
# =============================================================================
from torchref.base.reciprocal import (
    reciprocal_basis_matrix,
    get_scattering_vectors,
    get_d_spacing,
    place_on_grid,
    extract_structure_factor_from_grid,
    apply_translation_phase,
    interpolate_structure_factor_from_grid,
    interpolate_complex_from_grid,
    trilinear_interpolate_patterson,
)

# =============================================================================
# Re-exports from structure_factors submodule
# =============================================================================
from torchref.base.structure_factors import (
    iso_structure_factor_torched,
    iso_structure_factor_torched_no_complex,
    aniso_structure_factor_torched,
    aniso_structure_factor_torched_no_complex,
    anharmonic_correction,
    anharmonic_correction_no_complex,
    core_deformation,
    multiplication_quasi_complex_tensor,
)

# =============================================================================
# Re-exports from electron_density submodule
# =============================================================================
from torchref.base.electron_density import (
    vectorized_add_to_map,
    vectorized_add_to_map_aniso,
    scatter_add_nd,
    scatter_add_nd_super_slow,
    find_relevant_voxels,
    excise_angstrom_radius_around_coord,
    add_to_solvent_mask,
    add_to_phenix_mask,
    find_solvent_voids,
)

# =============================================================================
# Re-exports from fourier submodule
# =============================================================================
from torchref.base.fourier import (
    fft,
    ifft,
    get_real_grid,
    find_grid_size,
)

# =============================================================================
# Re-exports from alignment submodule
# =============================================================================
from torchref.base.alignment import (
    rotate_coords_torch,
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_axis_angle,
    quaternion_to_rotation_matrix,
    random_rotation_uniform,
    superpose_vectors_robust_torch,
    align_torch,
    get_alignement_matrix,
    apply_transformation,
)

# =============================================================================
# Re-exports from metrics submodule
# =============================================================================
from torchref.base.metrics import (
    get_rfactor_torch,
    rfactor,
    get_rfactors,
    bin_wise_rfactors,
    calc_outliers,
    nll_xray,
    nll_xray_sum,
    nll_xray_lognormal,
    log_loss,
    estimate_sigma_I,
    estimate_sigma_F,
    gaussian_to_lognormal_sigma,
    gaussian_to_lognormal_mu,
)

# =============================================================================
# Utility functions (kept here as they don't fit a specific domain)
# =============================================================================


def U_to_matrix(U: torch.Tensor) -> torch.Tensor:
    """
    Convert anisotropic displacement parameters from 6-component vector to 3x3 matrix.

    Parameters
    ----------
    U : torch.Tensor
        Anisotropic displacement parameters in the order
        [u11, u22, u33, u12, u13, u23] of shape (..., 6).

    Returns
    -------
    torch.Tensor
        Anisotropic displacement parameter matrices of shape (..., 3, 3).
    """
    u11 = U[..., 0]
    u22 = U[..., 1]
    u33 = U[..., 2]
    u12 = U[..., 3]
    u13 = U[..., 4]
    u23 = U[..., 5]
    if U.ndim == 1:
        U_matrix = torch.zeros((3, 3), device=U.device, dtype=U.dtype)
    else:
        U_matrix = torch.zeros(U.shape[:-1] + (3, 3), device=U.device, dtype=U.dtype)
    U_matrix[..., 0, 0] = u11
    U_matrix[..., 1, 1] = u22
    U_matrix[..., 2, 2] = u33
    U_matrix[..., 0, 1] = u12
    U_matrix[..., 1, 0] = u12
    U_matrix[..., 0, 2] = u13
    U_matrix[..., 2, 0] = u13
    U_matrix[..., 1, 2] = u23
    U_matrix[..., 2, 1] = u23

    return U_matrix


def deterministic_tensor_digest(t: torch.Tensor, n_chunks: int = 16) -> torch.Tensor:
    """
    Compute a deterministic digest vector for tensor directly on GPU.

    This function is deterministic across devices and runs, sensitive to all
    tensor values and order, fully vectorized with no Python loops, and suitable
    for large GPU tensors. Uses a simple mean/std approach per chunk which is
    fully deterministic.

    Parameters
    ----------
    t : torch.Tensor
        Input tensor to compute digest for.
    n_chunks : int, optional
        Number of chunks to divide the tensor into. Default is 16.

    Returns
    -------
    torch.Tensor
        Digest vector of length n_chunks.
    """
    # Flatten and cast to a stable type
    flat = t.detach().reshape(-1)
    if not torch.is_floating_point(flat):
        flat = flat.float()

    # If tensor smaller than n_chunks, just pad
    n = flat.numel()
    if n < n_chunks:
        flat = torch.nn.functional.pad(flat, (0, n_chunks - n))
        n = n_chunks

    # Reshape into chunks directly (pad if needed)
    chunk_size = (n + n_chunks - 1) // n_chunks
    padded_size = chunk_size * n_chunks

    if n < padded_size:
        flat = torch.nn.functional.pad(flat, (0, padded_size - n))

    # Reshape to (n_chunks, chunk_size) and compute stats per chunk
    chunks = flat[: chunk_size * n_chunks].reshape(n_chunks, chunk_size)

    # Create digest from mean and std of each chunk (both are deterministic)
    # Interleave for better sensitivity
    digest_mean = chunks.mean(dim=1)
    if chunks.size(1) > 1:
        digest_std = chunks.std(dim=1)
    else:
        digest_std = torch.zeros_like(digest_mean)

    # Combine into single digest vector (alternate mean/std would double size)
    # Instead, use weighted combination
    digest = digest_mean + 0.61803398875 * digest_std

    return digest


def hash_tensors(tensors) -> str:
    """
    Compute a hash of multiple tensors for caching purposes.

    Parameters
    ----------
    tensors : list of torch.Tensor or None
        List of tensors to hash. None values are handled.

    Returns
    -------
    str
        SHA-1 hash of the tensor contents.
    """
    h = hashlib.sha1()
    for t in tensors:
        if t is None:
            h.update(b"<None>")
            continue
        digest = deterministic_tensor_digest(t)
        # Bring only digest (small) to CPU for hashing
        h.update(digest.cpu().numpy().tobytes())
        h.update(str(t.shape).encode())
        h.update(str(t.dtype).encode())
    return h.hexdigest()


def french_wilson_conversion(Iobs, sigma_I=None):
    """
    Convert intensities to structure factor amplitudes using French-Wilson method.

    Also converts standard deviations.

    Parameters
    ----------
    Iobs : torch.Tensor
        Observed intensity values.
    sigma_I : torch.Tensor, optional
        Estimated standard deviations of intensities.

    Returns
    -------
    F : torch.Tensor
        Structure factor amplitudes.
    sigma_F : torch.Tensor
        Standard deviations of structure factor amplitudes.
    """
    # If no sigmas provided, estimate them
    if sigma_I is None:
        sigma_I = torch.sqrt(torch.clamp(Iobs, min=1e-6))

    # Determine mean intensity for Wilson prior
    mean_I = torch.mean(torch.clamp(Iobs[~torch.isnan(Iobs)], min=0))

    # Strong reflections: simple square root
    strong_mask = Iobs > 3.0 * sigma_I
    F = torch.zeros_like(Iobs)
    F[strong_mask] = torch.sqrt(Iobs[strong_mask])

    # Weak/negative reflections: Bayesian approach
    weak_mask = ~strong_mask
    if weak_mask.any():
        # For weak reflections, use Bayesian estimate
        I_weak = Iobs[weak_mask]
        sigma_weak = sigma_I[weak_mask]

        # For negative intensities, we need a better prior estimate
        # The global mean_I is biased toward strong reflections
        # For negative I, use the uncertainty as a guide for the expected intensity
        # Better prior: use sigma_I as a proxy for the true intensity scale

        # Separate negative and weak positive
        neg_local_mask = I_weak < 0
        pos_local_mask = ~neg_local_mask

        F_weak = torch.zeros_like(I_weak)

        # For negative intensities: use sigma_I based correction
        # The idea: if I < 0, the true intensity is likely ~sigma_I in magnitude
        # So use a prior based on sigma_I rather than the global mean_I
        if neg_local_mask.any():
            I_neg = I_weak[neg_local_mask]
            sigma_neg = sigma_weak[neg_local_mask]

            # For negative intensities, use a correction proportional to sigma^2
            # This gives F values that scale with the uncertainty
            # Formula: F ≈ sqrt(sigma^2 / 2) for very negative
            # Blend with the global prior for moderately negative

            # Weight based on how negative: |I/sigma|
            epsilon = torch.abs(I_neg / torch.clamp(sigma_neg, min=1e-10))

            # For slightly negative (epsilon < 0.5): use small correction
            # For very negative (epsilon > 1): use sigma-based prior
            # Smooth transition with tanh
            weight_sigma_prior = torch.tanh(epsilon)

            # Sigma-based correction (for very negative)
            F_sigma_prior = torch.sqrt(sigma_neg**2 / 2.0)

            # Global prior correction (for slightly negative)
            wilson_param_global = mean_I / 2.0
            correction_global = sigma_neg**2 / (2.0 * wilson_param_global)
            F_global = torch.sqrt(torch.clamp(I_neg + correction_global, min=0))

            # Blend
            F_weak[neg_local_mask] = (
                weight_sigma_prior * F_sigma_prior + (1 - weight_sigma_prior) * F_global
            )

        # For weak positive intensities: use standard correction
        if pos_local_mask.any():
            I_pos = I_weak[pos_local_mask]
            sigma_pos = sigma_weak[pos_local_mask]

            # Simplified Bayesian estimate (posterior mean)
            wilson_param = mean_I / 2.0
            variance_correction = sigma_pos**2 / (2.0 * wilson_param)
            F_weak[pos_local_mask] = torch.sqrt(
                torch.clamp(I_pos + variance_correction, min=0)
            )

        F[weak_mask] = F_weak

    # Convert sigmas using error propagation formula
    # For F = sqrt(I), σ(F) = σ(I)/(2*F)
    # Avoid division by zero
    sigma_F = torch.zeros_like(sigma_I)
    nonzero_F = F > 1e-6
    sigma_F[nonzero_F] = sigma_I[nonzero_F] / (2.0 * F[nonzero_F])

    # For very weak reflections where F approaches zero,
    # use an upper bound approximation to avoid huge sigma values
    tiny_F = (F <= 1e-6) & (sigma_I > 0)
    if tiny_F.any():
        # Approximate using the typical Wilson distribution variance
        wilson_param = mean_I / 2.0
        sigma_F[tiny_F] = torch.sqrt(wilson_param / 2.0)

    return F, sigma_F


# =============================================================================
# __all__ - Public API
# =============================================================================
__all__ = [
    # Coordinate transforms
    "cartesian_to_fractional_torch",
    "fractional_to_cartesian_torch",
    "get_fractional_matrix",
    "get_inv_fractional_matrix_torch",
    "smallest_diff",
    "smallest_diff_aniso",
    # Reciprocal space
    "reciprocal_basis_matrix",
    "get_scattering_vectors",
    "get_d_spacing",
    "place_on_grid",
    "extract_structure_factor_from_grid",
    "apply_translation_phase",
    "interpolate_structure_factor_from_grid",
    "interpolate_complex_from_grid",
    "trilinear_interpolate_patterson",
    # Structure factors
    "iso_structure_factor_torched",
    "iso_structure_factor_torched_no_complex",
    "aniso_structure_factor_torched",
    "aniso_structure_factor_torched_no_complex",
    "anharmonic_correction",
    "anharmonic_correction_no_complex",
    "core_deformation",
    "multiplication_quasi_complex_tensor",
    # Electron density
    "vectorized_add_to_map",
    "vectorized_add_to_map_aniso",
    "scatter_add_nd",
    "scatter_add_nd_super_slow",
    "find_relevant_voxels",
    "excise_angstrom_radius_around_coord",
    "add_to_solvent_mask",
    "add_to_phenix_mask",
    "find_solvent_voids",
    # Fourier
    "fft",
    "ifft",
    "get_real_grid",
    "find_grid_size",
    # Alignment
    "rotate_coords_torch",
    "axis_angle_to_rotation_matrix",
    "rotation_matrix_to_axis_angle",
    "quaternion_to_rotation_matrix",
    "random_rotation_uniform",
    "superpose_vectors_robust_torch",
    "align_torch",
    "get_alignement_matrix",
    "apply_transformation",
    # Metrics
    "get_rfactor_torch",
    "rfactor",
    "get_rfactors",
    "bin_wise_rfactors",
    "calc_outliers",
    "nll_xray",
    "nll_xray_sum",
    "nll_xray_lognormal",
    "log_loss",
    "estimate_sigma_I",
    "estimate_sigma_F",
    "gaussian_to_lognormal_sigma",
    "gaussian_to_lognormal_mu",
    # Utility functions
    "U_to_matrix",
    "deterministic_tensor_digest",
    "hash_tensors",
    "french_wilson_conversion",
]
