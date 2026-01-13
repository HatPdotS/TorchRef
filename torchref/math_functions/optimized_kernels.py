"""
Optimized versions of map building functions with kernel fusion
to reduce CPU-GPU synchronization overhead.
"""

import torch


@torch.jit.script
def fused_gaussian_density(
    diff_coords_squared: torch.Tensor,
    B: torch.Tensor,
    b: torch.Tensor,
    A: torch.Tensor,
    occ: torch.Tensor,
) -> torch.Tensor:
    """
    Fused computation of Gaussian density to reduce kernel launches.

    This combines:
    - B_total calculation
    - Normalization
    - A normalization
    - Gaussian evaluation
    - Summation over components

    Into a single fused operation that PyTorch can optimize.

    Parameters:
    -----------
    diff_coords_squared : torch.Tensor (N_atoms, N_voxels)
        Squared distances from atoms to voxels
    B : torch.Tensor (N_atoms, 5)
        ITC92 B parameters
    b : torch.Tensor (N_atoms,)
        Atomic B-factors
    A : torch.Tensor (N_atoms, 5)
        ITC92 A parameters
    occ : torch.Tensor (N_atoms,)
        Occupancies

    Returns:
    --------
    density : torch.Tensor (N_atoms, N_voxels)
        Computed density values
    """
    pi = 3.14159265359
    pi_sq = 9.86960440109

    # All operations in one expression for better fusion
    B_total = torch.clamp((B + b.unsqueeze(1)) / 4.0, min=0.1)

    # Compute in a way that encourages fusion
    normalization = torch.pow(pi / B_total, 1.5)
    A_norm = A * occ.unsqueeze(1) * normalization

    # Gaussian evaluation
    exponent = -pi_sq * diff_coords_squared.unsqueeze(2) / B_total.unsqueeze(1)
    gaussian = torch.exp(exponent)

    # Sum over Gaussian components
    density = torch.sum(A_norm.unsqueeze(1) * gaussian, dim=2)

    return density


@torch.jit.script
def fused_aniso_gaussian_density(
    diff_coords: torch.Tensor,
    U_matrix: torch.Tensor,
    A: torch.Tensor,
    occ: torch.Tensor,
) -> torch.Tensor:
    """
    Fused anisotropic Gaussian density calculation.

    Parameters:
    -----------
    diff_coords : torch.Tensor (N_atoms, N_voxels, 3)
        Distance vectors
    U_matrix : torch.Tensor (N_atoms, 4, 3, 3)
        Anisotropic U tensors for each Gaussian component
    A : torch.Tensor (N_atoms, 4)
        ITC92 A parameters
    occ : torch.Tensor (N_atoms,)
        Occupancies

    Returns:
    --------
    density : torch.Tensor (N_atoms, N_voxels)
    """
    two_pi_sq = 19.7392088022

    # Compute r^T U r for all components at once
    # diff_coords: (N_atoms, N_voxels, 3)
    # U_matrix: (N_atoms, 4, 3, 3)

    # Expand for broadcasting
    diff_expanded = diff_coords.unsqueeze(2)  # (N_atoms, N_voxels, 1, 3)

    # Matrix multiply: r^T U r
    U_expanded = U_matrix.unsqueeze(1)  # (N_atoms, 1, 4, 3, 3)

    # r^T U
    rT_U = torch.matmul(diff_expanded, U_expanded)  # (N_atoms, N_voxels, 4, 1, 3)

    # (r^T U) r
    quad_form = torch.matmul(
        rT_U, diff_expanded.unsqueeze(-1)
    )  # (N_atoms, N_voxels, 4, 1, 1)
    quad_form = quad_form.squeeze(-1).squeeze(-1)  # (N_atoms, N_voxels, 4)

    # Compute Gaussian
    exponent = -two_pi_sq * quad_form
    gaussian = torch.exp(exponent)

    # Weight by amplitude and occupancy
    A_occ = A * occ.unsqueeze(1)  # (N_atoms, 4)
    density = torch.sum(A_occ.unsqueeze(1) * gaussian, dim=2)

    return density


def warmup_cuda_operations(device="cuda"):
    """
    Warm up CUDA kernels to avoid lazy loading overhead.

    This function runs dummy operations to trigger CUDA kernel compilation
    and loading, so subsequent operations don't incur this overhead.

    Call this once after moving model to GPU.
    """
    if device == "cpu":
        return

    # Create dummy tensors with correct shapes for broadcasting
    dummy_a = torch.randn(1000, 100, device=device)
    dummy_b = torch.randn(1000, 100, device=device)
    dummy_c = torch.randn(1000, device=device)
    dummy_d = torch.randn(1000, 5, device=device)

    # Trigger common operations
    _ = dummy_a + dummy_b
    _ = dummy_a * dummy_c.unsqueeze(1)
    _ = torch.exp(dummy_a)
    _ = torch.sum(dummy_a, dim=1)
    _ = dummy_a / dummy_b.clamp(min=0.1)
    _ = torch.matmul(dummy_d, dummy_d.T)

    # FFT operations
    dummy_3d = torch.randn(64, 64, 64, device=device, dtype=torch.complex64)
    _ = torch.fft.fftn(dummy_3d)
    _ = torch.fft.ifftn(dummy_3d)

    # Scatter operations
    dummy_map = torch.zeros(100, 100, 100, device=device)
    dummy_indices = torch.randint(0, 100, (1000, 3), device=device)
    dummy_values = torch.randn(1000, device=device)
    dummy_map.view(-1).index_add_(
        0,
        dummy_indices[:, 0] * 10000 + dummy_indices[:, 1] * 100 + dummy_indices[:, 2],
        dummy_values,
    )

    # Synchronize to ensure all kernels are loaded
    torch.cuda.synchronize()


@torch.jit.script
def compute_smallest_diff_squared(
    diff: torch.Tensor, inv_frac_matrix: torch.Tensor, frac_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Fused computation of periodic distance squared.

    Combines fractional coordinate conversion, wrapping, and
    distance calculation into a single fused operation.
    """
    # Convert to fractional
    diff_frac = torch.matmul(diff, inv_frac_matrix.T)

    # Wrap to [-0.5, 0.5]
    diff_frac_wrapped = diff_frac - torch.round(diff_frac)

    # Convert back to Cartesian
    diff_cart = torch.matmul(diff_frac_wrapped, frac_matrix.T)

    # Compute squared distance
    r_squared = torch.sum(diff_cart * diff_cart, dim=-1)

    return r_squared


class CachedRadiusMask:
    """
    Cache the radius mask computation to avoid recomputing for every atom batch.

    Usage:
        cache = CachedRadiusMask()
        offsets = cache.get_offsets(voxel_size, radius_angstrom, device)
    """

    def __init__(self):
        self._cache = {}

    def get_offsets(
        self, voxel_size: torch.Tensor, radius_angstrom: float, device: torch.device
    ) -> torch.Tensor:
        """
        Get cached offset grid for given parameters.

        Returns:
            offsets : torch.Tensor (N_voxels, 3)
                Voxel offsets within radius
        """
        # Create cache key
        voxel_min = voxel_size.min().item()
        key = (device, radius_angstrom, round(voxel_min, 6))

        if key not in self._cache:
            # Compute radius in voxels
            min_box_radius = int(torch.ceil(radius_angstrom / voxel_size.min()).item())

            # Create offset grid
            gridx = torch.arange(-min_box_radius, min_box_radius + 1, device=device)
            gridy = torch.arange(-min_box_radius, min_box_radius + 1, device=device)
            gridz = torch.arange(-min_box_radius, min_box_radius + 1, device=device)

            x, y, z = torch.meshgrid(gridx, gridy, gridz, indexing="ij")
            coords = torch.stack((x, y, z), dim=-1)

            # Compute distances
            distance_map = torch.sqrt(
                torch.sum((coords * voxel_size.unsqueeze(0)) ** 2, dim=-1)
            )
            within_radius_mask = distance_map <= radius_angstrom

            # Store offsets
            self._cache[key] = coords[within_radius_mask].contiguous()

        return self._cache[key]


# Global cache instance
_radius_mask_cache = CachedRadiusMask()


def get_cached_radius_offsets(
    voxel_size: torch.Tensor, radius_angstrom: float, device: torch.device
) -> torch.Tensor:
    """
    Get cached radius offsets to avoid recomputation.

    This eliminates redundant computation when processing multiple atoms
    with the same voxel size and radius.
    """
    return _radius_mask_cache.get_offsets(voxel_size, radius_angstrom, device)


def vectorized_add_to_map_optimized(
    surrounding_coords: torch.Tensor,
    voxel_indices: torch.Tensor,
    map: torch.Tensor,
    xyz: torch.Tensor,
    b: torch.Tensor,
    inv_frac_matrix: torch.Tensor,
    frac_matrix: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    occ: torch.Tensor,
) -> torch.Tensor:
    """
    Optimized version of vectorized_add_to_map using fused Gaussian calculation.

    This is a drop-in replacement that uses the fused_gaussian_density function
    to reduce kernel launches.

    Parameters: Same as vectorized_add_to_map
    """
    from torchref.math_functions.math_torch import scatter_add_nd, smallest_diff

    # Calculate squared distances with periodic boundary conditions
    diff_coords_squared = smallest_diff(
        surrounding_coords - xyz.unsqueeze(1), inv_frac_matrix, frac_matrix
    )

    # Use fused Gaussian density calculation
    density = fused_gaussian_density(diff_coords_squared, B, b, A, occ)

    # Flatten and scatter to map
    density_flat = density.flatten()
    voxel_indices_flat = voxel_indices.reshape(-1, 3)

    # Add to map
    map = scatter_add_nd(density_flat, voxel_indices_flat, map)
    return map
