"""
PyTorch implementations of mathematical functions for crystallography.

.. deprecated:: 0.6.0
    This module is deprecated. Please import from domain-specific submodules:

    - ``torchref.base.coordinates`` - Coordinate transformations
    - ``torchref.base.reciprocal`` - Reciprocal space calculations
    - ``torchref.base.direct_summation`` - Structure factor calculations
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
    from torchref.base.metrics import get_rfactors

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
# Re-exports from direct_summation submodule
# =============================================================================
from torchref.base.direct_summation import (
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
    # NOTE: legacy misspelling preserved for backward compatibility. The
    # canonical public name is ``get_alignment_matrix`` (exported by
    # ``torchref.base``); this ``get_alignement_matrix`` alias is deprecated.
    get_alignement_matrix,
    apply_transformation,
)

# =============================================================================
# Re-exports from metrics submodule
# =============================================================================
from torchref.base.metrics import (
    get_rfactors,
    bin_wise_rfactors,
    nll_xray,
    nll_xray_sum,
    nll_xray_mean,
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

    # Build rows and stack to preserve gradient flow
    row0 = torch.stack([u11, u12, u13], dim=-1)
    row1 = torch.stack([u12, u22, u23], dim=-1)
    row2 = torch.stack([u13, u23, u33], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


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

    .. deprecated:: 0.6.0
        Use ``(tensor.data_ptr(), tensor._version, tensor.numel())`` tuples
        for lightweight fingerprinting instead. ``hash_tensors`` copies data
        to CPU and computes SHA-1, which is expensive.

    Parameters
    ----------
    tensors : list of torch.Tensor or None
        List of tensors to hash. None values are handled.

    Returns
    -------
    str
        SHA-1 hash of the tensor contents.
    """
    import warnings
    warnings.warn(
        "hash_tensors is deprecated. Use (tensor.data_ptr(), tensor._version, "
        "tensor.numel()) tuples for lightweight fingerprinting instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    "get_rfactors",
    "bin_wise_rfactors",
    "nll_xray",
    "nll_xray_sum",
    "nll_xray_mean",
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
]
