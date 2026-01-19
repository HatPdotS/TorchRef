"""
Maximum Likelihood molecular replacement aligner.

This module provides the main API for ML-based rigid body alignment,
combining quaternion-based transformations with ML target functions
for gradient-based optimization.

Main classes:
- RigidBodyMLTarget: Differentiable target function for rigid body refinement
- MLOrientationAligner: High-level API for ML molecular replacement
- MLAlignmentResult: Container for alignment results

The workflow is:
1. Create an MLOrientationAligner with observed data and model
2. Optionally use Patterson pre-screening for initial rotation
3. Run LBFGS optimization to refine rotation and translation
4. Retrieve the aligned model and result statistics

Examples
--------
::

    from torchref.alignment import MLOrientationAligner
    from torchref.model import ModelFT
    from torchref.io.datasets.reflection_data import ReflectionData
    
    data = ReflectionData().load_mtz('observed.mtz')
    model = ModelFT().load_pdb('search_model.pdb')
    
    aligner = MLOrientationAligner(data, model)
    aligned_model, result = aligner.align()
    print(f"Final LLG: {result.llg:.2f}")
"""

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn

from torchref.alignment.likelihood import MLTargetFunction, compute_d_factors
from torchref.alignment.transform import (
    quaternion_normalize,
    quaternion_to_matrix,
)
from torchref.math_functions.french_wilson import is_centric_from_hkl
from torchref.math_functions.math_torch import get_d_spacing

if TYPE_CHECKING:
    from torchref.io.datasets.reflection_data import ReflectionData
    from torchref.model.model_ft import ModelFT


@dataclass
class MLAlignmentResult:
    """
    Container for ML alignment results.

    Attributes
    ----------
    llg : float
        Final log-likelihood gain.
    rotation : torch.Tensor
        Final rotation quaternion [w, x, y, z].
    translation : torch.Tensor
        Final translation vector (fractional coordinates).
    rotation_matrix : torch.Tensor
        Final rotation as 3x3 matrix.
    n_iterations : int
        Number of optimization iterations performed.
    converged : bool
        Whether the optimization converged.
    initial_llg : float
        LLG at the start of refinement.
    llg_history : list
        LLG values at each iteration.
    """

    llg: float = 0.0
    rotation: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0, 0.0, 0.0, 0.0]))
    translation: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    rotation_matrix: torch.Tensor = field(default_factory=lambda: torch.eye(3))
    n_iterations: int = 0
    converged: bool = False
    initial_llg: float = 0.0
    llg_history: list = field(default_factory=list)

    def rotation_degrees(self) -> float:
        """Get rotation angle in degrees from quaternion."""
        w = torch.clamp(torch.abs(self.rotation[0]), max=1.0)
        angle_rad = 2.0 * torch.acos(w)
        return float(angle_rad * 180.0 / 3.14159265)


class RigidBodyMLTarget(nn.Module):
    """
    Differentiable ML target function for rigid body refinement.

    Combines rigid body transformation (rotation + translation) with
    structure factor calculation and ML likelihood evaluation.

    The forward pass:
    1. Normalizes the quaternion
    2. Converts to rotation matrix
    3. Transforms model coordinates: xyz' = xyz @ R.T + t
    4. Computes structure factors via ModelFT
    5. Returns negative log-likelihood (for minimization)

    Parameters
    ----------
    model : ModelFT
        Model with structure factor computation capability.
    data : ReflectionData
        Observed reflection data.
    rms_error : float, optional
        Estimated RMS coordinate error in Angstroms. Default is 1.0.
    use_rfree : bool, optional
        If True, evaluate on R-free set only. Default is False.
    high_resolution_limit : float, optional
        High resolution cutoff in Angstroms. Reflections with d-spacing below
        this limit are excluded. Default is 0.0 (no limit).
        For MR searches, 3.0-4.0 Å is typical for initial searches.

    Attributes
    ----------
    ml_target : MLTargetFunction
        Precomputed ML target function.
    original_xyz : torch.Tensor
        Original model coordinates (stored for restoration).
    """

    def __init__(
        self,
        model: "ModelFT",
        data: "ReflectionData",
        rms_error: float = 1.0,
        use_rfree: bool = False,
        high_resolution_limit: float = 0.0,
    ):
        super().__init__()

        self.model = model
        self.data = data
        self.rms_error = rms_error
        self.use_rfree = use_rfree

        # Store original coordinates
        self.register_buffer("original_xyz", model.xyz().clone())

        # Get HKL and observed amplitudes
        hkl = data.hkl
        F_obs = data.F

        # Apply R-free mask if requested
        if use_rfree and data.rfree_flags is not None:
            mask = data.rfree_flags
            hkl = hkl[mask]
            F_obs = F_obs[mask]

        # Apply resolution cutoff (high_resolution_limit in Angstroms)
        # Lower d-spacing = higher resolution, so we exclude d < limit
        if high_resolution_limit > 0:
            resolution = get_d_spacing(hkl, data.cell)
            res_mask = resolution >= high_resolution_limit
            hkl = hkl[res_mask]
            F_obs = F_obs[res_mask]

        self.register_buffer("hkl", hkl)
        self.register_buffer("F_obs", F_obs)

        # Compute resolution
        resolution = get_d_spacing(hkl, data.cell)
        self.register_buffer("resolution", resolution)

        # Compute D factors
        D = compute_d_factors(resolution, rms_error=rms_error)
        self.register_buffer("D", D)

        # Get centric flags
        spacegroup = data.spacegroup if data.spacegroup else "P1"
        centric_flags = is_centric_from_hkl(hkl, spacegroup)
        self.register_buffer("centric_flags", centric_flags)

        # Compute epsilon factors
        # For simplicity, use 1.0 for all; proper epsilon requires symmetry grid
        epsilon = torch.ones_like(F_obs)
        self.register_buffer("epsilon", epsilon)

        # Create ML target function
        self.ml_target = MLTargetFunction(
            F_obs=F_obs,
            resolution=resolution,
            epsilon=epsilon,
            centric_flags=centric_flags,
            rms_error=rms_error,
        )

    def transform_coordinates(
        self,
        rotation_quat: torch.Tensor,
        translation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rigid body transformation to model coordinates.

        Parameters
        ----------
        rotation_quat : torch.Tensor
            Rotation quaternion [w, x, y, z] of shape (4,).
        translation : torch.Tensor
            Translation vector in Cartesian coordinates of shape (3,).

        Returns
        -------
        torch.Tensor
            Transformed coordinates of shape (n_atoms, 3).
        """
        # Normalize quaternion
        quat_norm = quaternion_normalize(rotation_quat)

        # Convert to rotation matrix
        R = quaternion_to_matrix(quat_norm)

        # Get coordinates - use detach to ensure no graph connection from previous iterations
        # The original_xyz is a buffer that shouldn't have gradients
        xyz = self.original_xyz.detach()

        # Apply transformation: xyz' = xyz @ R.T + t
        xyz_transformed = xyz @ R.T + translation

        return xyz_transformed

    def forward(
        self,
        rotation_quat: torch.Tensor,
        translation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative LLG for the given transformation.

        Parameters
        ----------
        rotation_quat : torch.Tensor
            Rotation quaternion [w, x, y, z].
        translation : torch.Tensor
            Translation vector in Cartesian coordinates.

        Returns
        -------
        torch.Tensor
            Negative log-likelihood gain (scalar, for minimization).
        """
        # Transform coordinates
        xyz_transformed = self.transform_coordinates(rotation_quat, translation)

        # Update model coordinates in-place (faster than copy)
        # Note: This modifies the model, so caller should be aware
        self.model.xyz[:] = xyz_transformed

        # Reset cache and compute structure factors
        self.model.reset_cache()
        F_calc = self.model.get_structure_factor(self.hkl, recalc=True)
        F_calc_amp = torch.abs(F_calc)

        # Scale F_calc to match F_obs (LSQ scaling)
        # k = sum(F_obs * F_calc) / sum(F_calc^2)
        scale = (self.F_obs * F_calc_amp).sum() / (F_calc_amp**2).sum().clamp(min=1e-10)
        F_calc_scaled = scale * F_calc_amp

        # Compute negative LLG
        neg_llg = self.ml_target(F_calc_scaled)

        return neg_llg

    def get_llg(
        self,
        rotation_quat: torch.Tensor,
        translation: torch.Tensor,
    ) -> float:
        """Get LLG (not negated) for evaluation."""
        with torch.no_grad():
            neg_llg = self.forward(rotation_quat, translation)
            return float(-neg_llg)

    def restore_original_coordinates(self) -> None:
        """Restore model to original coordinates."""
        self.model.xyz[:] = self.original_xyz.clone()


class InterpolatedMLTarget(nn.Module):
    """
    Fast ML target using pre-computed reciprocal space grid with interpolation.

    Instead of recalculating structure factors for each rotation, this class:
    1. Pre-computes the reciprocal space grid once via FFT
    2. For each rotation R, samples F at rotated HKL positions using interpolation
    3. For translation, applies a phase factor (very fast)

    This is ~100x faster than RigidBodyMLTarget for rotation searches.

    Parameters
    ----------
    model : ModelFT
        Model with structure factor computation capability.
    data : ReflectionData
        Observed reflection data.
    rms_error : float, optional
        Estimated RMS coordinate error in Angstroms. Default is 1.0.
    high_resolution_limit : float, optional
        High resolution cutoff in Angstroms. Reflections with d-spacing below
        this limit are excluded. Default is 0.0 (no limit).
        For rotation searches, 3.0-4.0 Å is typical.

    Notes
    -----
    The math behind interpolation-based rotation:
    - Original structure factor: F(hkl) = FT{ρ(r)}
    - After rotating molecule by R: F'(hkl) = F(R^T @ hkl)
    - For translation t (fractional): F'(hkl) = F(hkl) * exp(2πi * hkl · t)
    """

    def __init__(
        self,
        model: "ModelFT",
        data: "ReflectionData",
        rms_error: float = 1.0,
        high_resolution_limit: float = 0.0,
    ):
        super().__init__()

        from torchref.math_functions.math_torch import (
            apply_translation_phase,
            interpolate_structure_factor_from_grid,
        )

        self.model = model
        self.data = data
        self.rms_error = rms_error
        self._interpolate_sf = interpolate_structure_factor_from_grid
        self._apply_phase = apply_translation_phase

        # Get HKL and observed amplitudes
        hkl = data.hkl
        F_obs = data.F

        # Apply resolution cutoff (high_resolution_limit in Angstroms)
        if high_resolution_limit > 0:
            resolution = get_d_spacing(hkl, data.cell)
            res_mask = resolution >= high_resolution_limit
            hkl = hkl[res_mask]
            F_obs = F_obs[res_mask]

        self.register_buffer("hkl", hkl.float())  # Need float for rotation
        self.register_buffer("F_obs", F_obs)

        # Pre-compute the reciprocal space grid (expensive, but only once)
        model.build_complete_map()
        from torchref.math_functions.math_torch import ifft

        reciprocal_grid = ifft(model.map)
        self.register_buffer("reciprocal_grid", reciprocal_grid)

        # Store grid dimensions for HKL validation
        self.grid_shape = reciprocal_grid.shape

        # Compute resolution
        resolution = get_d_spacing(hkl.long(), data.cell)
        self.register_buffer("resolution", resolution)

        # Compute D factors
        D = compute_d_factors(resolution, rms_error=rms_error)
        self.register_buffer("D", D)

        # Get centric flags
        spacegroup = data.spacegroup if data.spacegroup else "P1"
        centric_flags = is_centric_from_hkl(hkl.long(), spacegroup)
        self.register_buffer("centric_flags", centric_flags)

        # Epsilon factors
        epsilon = torch.ones_like(F_obs)
        self.register_buffer("epsilon", epsilon)

        # Create ML target function
        self.ml_target = MLTargetFunction(
            F_obs=F_obs,
            resolution=resolution,
            epsilon=epsilon,
            centric_flags=centric_flags,
            rms_error=rms_error,
        )

    def forward(
        self,
        rotation_quat: torch.Tensor,
        translation_frac: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative LLG using interpolation (fast).

        Parameters
        ----------
        rotation_quat : torch.Tensor
            Rotation quaternion [w, x, y, z].
        translation_frac : torch.Tensor
            Translation vector in fractional coordinates.
            Note: Translation has no effect because we interpolate amplitudes
            (phases are discarded). This parameter is kept for API compatibility.

        Returns
        -------
        torch.Tensor
            Negative log-likelihood gain (scalar, for minimization).
        """
        # Normalize quaternion and get rotation matrix
        quat_norm = quaternion_normalize(rotation_quat)
        R = quaternion_to_matrix(quat_norm)

        # Rotate HKL indices: for rotation R of molecule, sample at R^T @ hkl
        # Which is equivalent to hkl @ R
        hkl_rotated = self.hkl @ R

        # Interpolate structure factor AMPLITUDES at rotated positions
        # We interpolate amplitudes (not complex values) to avoid phase cancellation
        # issues where F=1 and F=-1 would average to 0 instead of 1
        F_calc_amp = self._interpolate_sf(
            self.reciprocal_grid, hkl_rotated, interpolate_amplitude=True
        )

        # Note: Translation phase shift is NOT applied because we're working with
        # amplitudes only. Translation would need a different approach (e.g.,
        # Patterson-based translation function or direct SF recalculation).

        # Scale F_calc to match F_obs
        scale = (self.F_obs * F_calc_amp).sum() / (F_calc_amp**2).sum().clamp(min=1e-10)
        F_calc_scaled = scale * F_calc_amp

        # Compute negative LLG
        neg_llg = self.ml_target(F_calc_scaled)

        return neg_llg

    def get_llg(
        self,
        rotation_quat: torch.Tensor,
        translation_frac: torch.Tensor,
    ) -> float:
        """Get LLG (not negated) for evaluation."""
        with torch.no_grad():
            neg_llg = self.forward(rotation_quat, translation_frac)
            return float(-neg_llg)

    def scan_rotations_batched(
        self,
        rotation_matrices: torch.Tensor,
        batch_size: int = 100,
    ) -> torch.Tensor:
        """
        Compute LLG for multiple rotations in batched mode (fast).

        This method efficiently evaluates many rotations by:
        1. Batching the HKL rotation: (batch, N_refl, 3) = (N_refl, 3) @ (batch, 3, 3)
        2. Batching the interpolation lookup
        3. Batching the LLG computation

        Parameters
        ----------
        rotation_matrices : torch.Tensor
            Rotation matrices of shape (N_rotations, 3, 3).
        batch_size : int, optional
            Number of rotations to process at once. Default is 100.
            Larger batches are faster but use more memory.

        Returns
        -------
        torch.Tensor
            LLG values for each rotation, shape (N_rotations,).
        """
        device = self.reciprocal_grid.device
        n_rotations = rotation_matrices.shape[0]
        n_reflections = self.hkl.shape[0]

        # Move rotation matrices to device
        R = rotation_matrices.to(device=device, dtype=torch.float32)

        # Store results
        llg_values = torch.zeros(n_rotations, device=device)

        # Process in batches
        for batch_start in range(0, n_rotations, batch_size):
            batch_end = min(batch_start + batch_size, n_rotations)
            R_batch = R[batch_start:batch_end]  # (batch, 3, 3)
            actual_batch_size = R_batch.shape[0]

            # Rotate HKL for all rotations in batch
            # hkl: (N_refl, 3) -> (1, N_refl, 3)
            # R_batch: (batch, 3, 3)
            # Result: (batch, N_refl, 3)
            hkl_rotated = torch.einsum("ij,bjk->bik", self.hkl, R_batch)

            # Interpolate amplitudes for all rotations in batch
            # This is the key batched operation
            F_calc_batch = self._interpolate_batched(
                self.reciprocal_grid, hkl_rotated
            )  # (batch, N_refl)

            # Compute LLG for each rotation in batch
            for i in range(actual_batch_size):
                F_calc_amp = F_calc_batch[i]

                # Scale F_calc to match F_obs
                scale = (self.F_obs * F_calc_amp).sum() / (
                    F_calc_amp**2
                ).sum().clamp(min=1e-10)
                F_calc_scaled = scale * F_calc_amp

                # Compute LLG (positive = good)
                neg_llg = self.ml_target(F_calc_scaled)
                llg_values[batch_start + i] = -neg_llg

        return llg_values

    def _interpolate_batched(
        self,
        reciprocal_grid: torch.Tensor,
        hkl_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched trilinear interpolation of amplitudes.

        Parameters
        ----------
        reciprocal_grid : torch.Tensor
            Complex reciprocal space grid of shape (Nx, Ny, Nz).
        hkl_batch : torch.Tensor
            Batched HKL positions of shape (batch, N_refl, 3).

        Returns
        -------
        torch.Tensor
            Interpolated amplitudes of shape (batch, N_refl).
        """
        batch_size, n_refl, _ = hkl_batch.shape
        Nx, Ny, Nz = reciprocal_grid.shape
        device = reciprocal_grid.device

        # Flatten batch dimension for interpolation
        hkl_flat = hkl_batch.reshape(-1, 3)  # (batch * N_refl, 3)

        h = hkl_flat[:, 0]
        k = hkl_flat[:, 1]
        l = hkl_flat[:, 2]

        # Floor and ceil indices
        h0 = torch.floor(h).long()
        k0 = torch.floor(k).long()
        l0 = torch.floor(l).long()
        h1 = h0 + 1
        k1 = k0 + 1
        l1 = l0 + 1

        # Fractional parts (weights)
        hd = h - h0.float()
        kd = k - k0.float()
        ld = l - l0.float()

        # Wrap indices (periodic boundary)
        h0 = torch.remainder(h0, Nx)
        h1 = torch.remainder(h1, Nx)
        k0 = torch.remainder(k0, Ny)
        k1 = torch.remainder(k1, Ny)
        l0 = torch.remainder(l0, Nz)
        l1 = torch.remainder(l1, Nz)

        # Get amplitudes at 8 corners
        a000 = torch.abs(reciprocal_grid[h0, k0, l0])
        a001 = torch.abs(reciprocal_grid[h0, k0, l1])
        a010 = torch.abs(reciprocal_grid[h0, k1, l0])
        a011 = torch.abs(reciprocal_grid[h0, k1, l1])
        a100 = torch.abs(reciprocal_grid[h1, k0, l0])
        a101 = torch.abs(reciprocal_grid[h1, k0, l1])
        a110 = torch.abs(reciprocal_grid[h1, k1, l0])
        a111 = torch.abs(reciprocal_grid[h1, k1, l1])

        # Trilinear interpolation of amplitudes
        a00 = a000 * (1 - ld) + a001 * ld
        a01 = a010 * (1 - ld) + a011 * ld
        a10 = a100 * (1 - ld) + a101 * ld
        a11 = a110 * (1 - ld) + a111 * ld
        a0 = a00 * (1 - kd) + a01 * kd
        a1 = a10 * (1 - kd) + a11 * kd
        result = a0 * (1 - hd) + a1 * hd

        # Reshape back to (batch, N_refl)
        return result.reshape(batch_size, n_refl)

    def rotation_search(
        self,
        angular_step_deg: float = 10.0,
        batch_size: int = 100,
        verbose: bool = True,
    ) -> tuple:
        """
        Perform a full rotation search over the non-degenerate rotation space.

        Uses the crystal point group symmetry to determine the minimal search
        space (asymmetric unit of SO(3)).

        Parameters
        ----------
        angular_step_deg : float, optional
            Angular sampling step in degrees. Default is 10.0.
        batch_size : int, optional
            Number of rotations to process at once. Default is 100.
        verbose : bool, optional
            Print progress information. Default is True.

        Returns
        -------
        llg_values : torch.Tensor
            LLG for each rotation, shape (N_rotations,).
        euler_angles : torch.Tensor
            Euler angles (alpha, beta, gamma) for each rotation, shape (N_rotations, 3).
        rotation_matrices : torch.Tensor
            Rotation matrices for each rotation, shape (N_rotations, 3, 3).
        best_idx : int
            Index of the best rotation (highest LLG).
        """
        import math

        from .sampling import get_rotation_sampling_range
        from .transform import rotation_matrix_from_euler, sample_angles

        # Get symmetry operations from data
        sym_matrices = None

        # Try to get from data.symmetry
        if hasattr(self.data, "symmetry") and self.data.symmetry is not None:
            sym_matrices = self.data.symmetry.matrices

        # Try to create from spacegroup if available
        if sym_matrices is None and hasattr(self.data, "spacegroup"):
            try:
                from torchref.symmetry import Symmetry

                sg = self.data.spacegroup
                # Handle gemmi spacegroup objects
                if hasattr(sg, "xhm"):
                    sg = sg.xhm()
                symmetry = Symmetry(sg)
                sym_matrices = symmetry.matrices
            except Exception:
                pass

        # Default to P1 (identity only)
        if sym_matrices is None:
            sym_matrices = torch.eye(3).unsqueeze(0)

        # Determine rotation sampling range based on point group symmetry
        max_angles = get_rotation_sampling_range(sym_matrices)

        if verbose:
            print(f"Point group symmetry reduces search space to:")
            print(
                f"  alpha: [0, {math.degrees(max_angles[0]):.1f}°], "
                f"beta: [0, {math.degrees(max_angles[1]):.1f}°], "
                f"gamma: [0, {math.degrees(max_angles[2]):.1f}°]"
            )

        # Generate Euler angles with specified sampling
        angular_step_rad = math.radians(angular_step_deg)
        euler_angles = sample_angles(angular_step_rad, max_angles)
        n_rotations = euler_angles.shape[0]

        if verbose:
            print(f"Sampling {n_rotations} rotations at {angular_step_deg}° step")

        # Convert to rotation matrices
        rotation_matrices = rotation_matrix_from_euler(euler_angles)

        # Run batched rotation search
        if verbose:
            print(f"Running batched search (batch_size={batch_size})...")

        import time

        start_time = time.time()

        with torch.no_grad():
            llg_values = self.scan_rotations_batched(rotation_matrices, batch_size)

        elapsed = time.time() - start_time

        # Find best rotation
        best_idx = llg_values.argmax().item()
        best_llg = llg_values[best_idx].item()
        best_euler = euler_angles[best_idx]

        if verbose:
            print(f"Search completed in {elapsed:.2f}s")
            print(f"  Rate: {n_rotations / elapsed:.1f} rotations/sec")
            print(f"  Best LLG: {best_llg:.2f}")
            print(
                f"  Best Euler angles: α={math.degrees(best_euler[0]):.1f}°, "
                f"β={math.degrees(best_euler[1]):.1f}°, "
                f"γ={math.degrees(best_euler[2]):.1f}°"
            )

        return llg_values, euler_angles, rotation_matrices, best_idx


class MLOrientationAligner:
    """
    High-level API for Maximum Likelihood molecular replacement.

    This class provides a complete workflow for ML-based alignment:
    1. Optionally pre-screen rotations using Patterson correlation
    2. Refine rotation and translation using LBFGS optimization
    3. Return aligned model and detailed results

    Parameters
    ----------
    data : ReflectionData
        Observed reflection data.
    model : ModelFT
        Search model with structure factor capability.
    rms_error : float, optional
        Estimated RMS coordinate error in Angstroms. Default is 1.0.
    verbose : int, optional
        Verbosity level (0=silent, 1=normal, 2=debug). Default is 1.

    Attributes
    ----------
    target : RigidBodyMLTarget
        The differentiable ML target function.

    Examples
    --------
    ::

        aligner = MLOrientationAligner(data, model)
        aligned_model, result = aligner.align()
        print(f"LLG: {result.llg:.2f}, Iterations: {result.n_iterations}")
    """

    def __init__(
        self,
        data: "ReflectionData",
        model: "ModelFT",
        rms_error: float = 1.0,
        verbose: int = 1,
        high_resolution_limit: float = 0.0,
    ):
        self.data = data
        self.model = model
        self.rms_error = rms_error
        self.verbose = verbose

        # Create target function
        self.target = RigidBodyMLTarget(
            model, data, rms_error=rms_error, high_resolution_limit=high_resolution_limit
        )

    def align(
        self,
        init_rotation: Optional[torch.Tensor] = None,
        init_translation: Optional[torch.Tensor] = None,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        use_patterson_prescreen: bool = False,
    ) -> Tuple["ModelFT", MLAlignmentResult]:
        """
        Perform ML alignment of the model to the data.

        Parameters
        ----------
        init_rotation : torch.Tensor, optional
            Initial rotation quaternion. If None, uses identity.
        init_translation : torch.Tensor, optional
            Initial translation. If None, uses zeros.
        max_iter : int, optional
            Maximum LBFGS iterations. Default is 100.
        tolerance : float, optional
            Convergence tolerance for loss change. Default is 1e-6.
        use_patterson_prescreen : bool, optional
            If True, use Patterson correlation to find initial rotation.
            Default is False.

        Returns
        -------
        model : ModelFT
            Model with aligned coordinates.
        result : MLAlignmentResult
            Detailed alignment results.
        """
        device = self.model.device
        dtype = self.model.dtype_float

        # Initialize rotation and translation
        if init_rotation is None:
            rotation = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device, requires_grad=True
            )
        else:
            rotation = init_rotation.clone().to(dtype=dtype, device=device)
            rotation.requires_grad_(True)

        if init_translation is None:
            translation = torch.zeros(3, dtype=dtype, device=device, requires_grad=True)
        else:
            translation = init_translation.clone().to(dtype=dtype, device=device)
            translation.requires_grad_(True)

        # Patterson pre-screening (placeholder for now)
        if use_patterson_prescreen:
            if self.verbose > 0:
                print("Patterson pre-screening not yet implemented, using initial rotation")

        # Run LBFGS refinement
        result = self.refine_orientation(
            rotation, translation, max_iter=max_iter, tolerance=tolerance
        )

        # Apply final transformation to model
        final_xyz = self.target.transform_coordinates(result.rotation, result.translation)
        self.model.xyz[:] = final_xyz

        return self.model, result

    def refine_orientation(
        self,
        init_rotation: torch.Tensor,
        init_translation: torch.Tensor,
        max_iter: int = 100,
        tolerance: float = 1e-6,
    ) -> MLAlignmentResult:
        """
        Refine rotation and translation using numerical gradient optimization.

        Uses finite differences to compute gradients since the model's MixedTensor
        design doesn't propagate gradients through coordinate assignment.

        Parameters
        ----------
        init_rotation : torch.Tensor
            Initial rotation quaternion.
        init_translation : torch.Tensor
            Initial translation.
        max_iter : int, optional
            Maximum iterations. Default is 100.
        tolerance : float, optional
            Convergence tolerance. Default is 1e-6.

        Returns
        -------
        MLAlignmentResult
            Optimization results including final parameters and LLG.
        """
        # Work with detached tensors for numerical gradient computation
        rotation = init_rotation.detach().clone()
        rotation = rotation / rotation.norm()  # Ensure normalized
        translation = init_translation.detach().clone()

        # Track progress
        llg_history = []
        converged = False

        # Get initial LLG
        initial_llg = self.target.get_llg(rotation, translation)
        llg_history.append(initial_llg)
        prev_llg = initial_llg

        if self.verbose > 0:
            print(f"Initial LLG: {initial_llg:.2f}")

        # Numerical optimization parameters
        eps_rot = 1e-3  # Step size for rotation gradient (larger for more stable gradient)
        eps_trans = 0.1  # Step size for translation gradient (Angstroms)
        lr_rot = 0.01  # Learning rate for rotation (smaller for stability)
        lr_trans = 0.1  # Learning rate for translation

        # Optimization loop using numerical gradients
        for iteration in range(max_iter):
            # Compute numerical gradient for rotation (4 components)
            grad_rot = torch.zeros(4)
            for i in range(4):
                rot_plus = rotation.clone()
                rot_plus[i] += eps_rot
                rot_plus = rot_plus / rot_plus.norm()

                rot_minus = rotation.clone()
                rot_minus[i] -= eps_rot
                rot_minus = rot_minus / rot_minus.norm()

                # Gradient of negative LLG (we want to maximize LLG = minimize neg_llg)
                llg_plus = self.target.get_llg(rot_plus, translation)
                llg_minus = self.target.get_llg(rot_minus, translation)
                grad_rot[i] = -(llg_plus - llg_minus) / (2 * eps_rot)

            # Compute numerical gradient for translation (3 components)
            grad_trans = torch.zeros(3)
            for i in range(3):
                trans_plus = translation.clone()
                trans_plus[i] += eps_trans

                trans_minus = translation.clone()
                trans_minus[i] -= eps_trans

                llg_plus = self.target.get_llg(rotation, trans_plus)
                llg_minus = self.target.get_llg(rotation, trans_minus)
                grad_trans[i] = -(llg_plus - llg_minus) / (2 * eps_trans)

            # Update parameters (gradient descent on negative LLG = ascent on LLG)
            rotation = rotation - lr_rot * grad_rot
            rotation = rotation / rotation.norm()  # Renormalize quaternion

            translation = translation - lr_trans * grad_trans

            # Compute current LLG
            current_llg = self.target.get_llg(rotation, translation)
            llg_history.append(current_llg)

            if self.verbose > 1:
                grad_norm = (grad_rot.norm() + grad_trans.norm()).item()
                print(f"  Iter {iteration + 1}: LLG = {current_llg:.2f}, grad_norm = {grad_norm:.6f}")

            # Check convergence
            if abs(current_llg - prev_llg) < tolerance:
                converged = True
                if self.verbose > 0:
                    print(f"Converged after {iteration + 1} iterations")
                break

            prev_llg = current_llg

        # Final LLG
        final_llg = self.target.get_llg(rotation, translation)

        if self.verbose > 0 and not converged:
            print(f"Reached max iterations ({max_iter})")
        if self.verbose > 0:
            print(f"Final LLG: {final_llg:.2f}")

        # Build result
        result = MLAlignmentResult(
            llg=final_llg,
            rotation=rotation.clone(),
            translation=translation.clone(),
            rotation_matrix=quaternion_to_matrix(quaternion_normalize(rotation)),
            n_iterations=iteration + 1 if 'iteration' in dir() else 0,
            converged=converged,
            initial_llg=initial_llg,
            llg_history=llg_history,
        )

        return result

    def evaluate(
        self,
        rotation: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Evaluate LLG for given transformation without optimization.

        Parameters
        ----------
        rotation : torch.Tensor, optional
            Rotation quaternion. If None, uses identity.
        translation : torch.Tensor, optional
            Translation vector. If None, uses zeros.

        Returns
        -------
        float
            Log-likelihood gain for the transformation.
        """
        device = self.model.device
        dtype = self.model.dtype_float

        if rotation is None:
            rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
        if translation is None:
            translation = torch.zeros(3, dtype=dtype, device=device)

        return self.target.get_llg(rotation, translation)

    def scan_rotations(
        self,
        rotations: torch.Tensor,
        translation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluate LLG for multiple rotations (rotation function).

        Parameters
        ----------
        rotations : torch.Tensor
            Rotation quaternions of shape (N, 4).
        translation : torch.Tensor, optional
            Translation to use for all rotations. If None, uses zeros.

        Returns
        -------
        torch.Tensor
            LLG values for each rotation, shape (N,).
        """
        device = self.model.device
        dtype = self.model.dtype_float

        if translation is None:
            translation = torch.zeros(3, dtype=dtype, device=device)

        n_rotations = rotations.shape[0]
        llg_values = torch.zeros(n_rotations, dtype=dtype, device=device)

        for i in range(n_rotations):
            llg_values[i] = self.target.get_llg(rotations[i], translation)

        return llg_values

    def scan_translations(
        self,
        translations: torch.Tensor,
        rotation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluate LLG for multiple translations (translation function).

        Parameters
        ----------
        translations : torch.Tensor
            Translation vectors of shape (N, 3).
        rotation : torch.Tensor, optional
            Rotation to use for all translations. If None, uses identity.

        Returns
        -------
        torch.Tensor
            LLG values for each translation, shape (N,).
        """
        device = self.model.device
        dtype = self.model.dtype_float

        if rotation is None:
            rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)

        n_translations = translations.shape[0]
        llg_values = torch.zeros(n_translations, dtype=dtype, device=device)

        for i in range(n_translations):
            llg_values[i] = self.target.get_llg(rotation, translations[i])

        return llg_values


# =============================================================================
# MRResult Dataclass
# =============================================================================


@dataclass
class MRResult:
    """
    Container for Molecular Replacement pipeline results.

    Attributes
    ----------
    llg : float
        Final LLG after joint refinement.
    rotation_quat : torch.Tensor
        Final rotation quaternion [w, x, y, z].
    rotation_matrix : torch.Tensor
        Final rotation as 3x3 matrix.
    translation_cart : torch.Tensor
        Final translation in Cartesian coordinates (Angstroms).
    translation_frac : torch.Tensor
        Final translation in fractional coordinates [0,1)³.
    r_factor : float
        R-factor: R = Σ||F_obs| - k|F_calc|| / Σ|F_obs|.
    r_free : Optional[float]
        R-free if available (computed on test set).
    rotation_search_llg : float
        Best LLG from stage 1 (rotation search).
    rotation_refined_llg : float
        Best LLG after stage 2 (rotation refinement).
    translation_search_corr : float
        Best correlation from stage 3 (translation search).
    joint_refined_llg : float
        LLG after stage 5 (joint refinement).
    converged : bool
        Whether joint refinement converged.
    runtime_seconds : float
        Total pipeline runtime.
    stage_timings : Dict[str, float]
        Per-stage timing breakdown.
    """

    llg: float = 0.0
    rotation_quat: torch.Tensor = field(
        default_factory=lambda: torch.tensor([1.0, 0.0, 0.0, 0.0])
    )
    rotation_matrix: torch.Tensor = field(default_factory=lambda: torch.eye(3))
    translation_cart: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    translation_frac: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    r_factor: float = 1.0
    r_free: Optional[float] = None
    rotation_search_llg: float = 0.0
    rotation_refined_llg: float = 0.0
    translation_search_corr: float = 0.0
    joint_refined_llg: float = 0.0
    converged: bool = False
    runtime_seconds: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)

    def rotation_degrees(self) -> float:
        """Get rotation angle in degrees from quaternion."""
        w = torch.clamp(torch.abs(self.rotation_quat[0]), max=1.0)
        angle_rad = 2.0 * torch.acos(w)
        return float(angle_rad * 180.0 / math.pi)


# =============================================================================
# TranslationSearchTarget Class
# =============================================================================


class TranslationSearchTarget(nn.Module):
    """
    FFT-based translation search using symmetry-derived structure factors.

    For a given rotation R:
    1. Calculate F_calc in P1 by interpolating from reciprocal grid at hkl @ R
    2. For each symmetry operation S (with rotation R_s and translation t_s):
       - Rotate HKL: hkl_s = hkl @ R_s
       - Interpolate COMPLEX F_calc at hkl_s (preserves phase)
       - Apply symmetry phase: F_s = F_calc(hkl_s) * exp(2πi * hkl · t_s)
    3. Sum contributions: F_total = Σ_s F_s
    4. Compute Patterson correlation via FFT for all translations simultaneously

    Parameters
    ----------
    model : ModelFT
        Model with pre-computed reciprocal space grid.
    data : ReflectionData
        Observed reflection data.
    grid_oversampling : int, optional
        Oversampling factor for translation grid. Default is 2.

    Attributes
    ----------
    reciprocal_grid : torch.Tensor
        Complex-valued reciprocal space grid from model.
    hkl : torch.Tensor
        Miller indices from data.
    F_obs : torch.Tensor
        Observed structure factor amplitudes.
    symmetry : Symmetry
        Space group symmetry operations.
    grid_shape : Tuple[int, int, int]
        Shape of the translation search grid.
    """

    def __init__(
        self,
        model: "ModelFT",
        data: "ReflectionData",
        grid_oversampling: int = 2,
    ):
        super().__init__()

        from torchref.math_functions.math_torch import ifft, interpolate_complex_from_grid
        from torchref.symmetry import Symmetry

        self.model = model
        self.data = data
        self._interpolate_complex = interpolate_complex_from_grid

        # Store HKL and observed amplitudes
        hkl = data.hkl
        F_obs = data.F

        self.register_buffer("hkl", hkl.float())
        self.register_buffer("F_obs", F_obs)

        # Pre-compute the reciprocal space grid (expensive, but only once)
        model.build_complete_map()
        reciprocal_grid = ifft(model.map)
        self.register_buffer("reciprocal_grid", reciprocal_grid)

        # Store grid dimensions
        self.model_grid_shape = reciprocal_grid.shape

        # Set up symmetry
        spacegroup = data.spacegroup if data.spacegroup else "P1"
        # Handle gemmi spacegroup objects
        if hasattr(spacegroup, "xhm"):
            spacegroup = spacegroup.xhm()
        self.symmetry = Symmetry(spacegroup)

        # Translation grid shape (for FFT-based search)
        # Use oversampling for finer translation sampling
        self.grid_oversampling = grid_oversampling
        Nx, Ny, Nz = self.model_grid_shape
        self.translation_grid_shape = (
            Nx * grid_oversampling,
            Ny * grid_oversampling,
            Nz * grid_oversampling,
        )

        # Store cell parameters for coordinate conversion
        self.cell = data.cell

    def translation_function(
        self,
        rotation_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute translation function for all translations via FFT.

        This is the key optimization: we compute the correlation between
        F_obs and F_calc for ALL translations simultaneously using FFT.

        Parameters
        ----------
        rotation_matrix : torch.Tensor
            Rotation matrix of shape (3, 3).

        Returns
        -------
        correlation_map : torch.Tensor
            3D tensor of correlation values at each grid point.
        best_translation_frac : torch.Tensor
            Fractional coordinates [0,1)³ of best translation.
        best_correlation : float
            Correlation value at best translation.
        """
        device = self.reciprocal_grid.device
        dtype = self.reciprocal_grid.dtype

        # Step 1: Compute F_calc with symmetry for this rotation
        F_calc_sym = self._compute_symmetry_fcalc(rotation_matrix)

        # Step 2: FFT-based translation search
        # For translation t: F(hkl, t) = F(hkl) * exp(2πi * hkl · t)
        # Correlation C(t) = Σ_hkl |F_obs(hkl)|² * |F_calc(hkl)|² * cos(2π * hkl · t + phase_diff)
        # For Patterson-like correlation, we use: C(t) = IFFT{F_obs * F_calc*}
        # This gives us the correlation at all translations in O(N log N)

        # Create product grid: place F_obs * F_calc* on reciprocal space grid
        Nx, Ny, Nz = self.translation_grid_shape
        product_grid = torch.zeros((Nx, Ny, Nz), dtype=dtype, device=device)

        # Place products at HKL positions
        hkl_int = self.hkl.long()
        for i in range(len(hkl_int)):
            h, k, l = hkl_int[i]
            hi = int(h) % Nx
            ki = int(k) % Ny
            li = int(l) % Nz
            product_grid[hi, ki, li] += self.F_obs[i] * F_calc_sym[i].conj()

        # IFFT gives correlation at all translations
        correlation_map = torch.fft.ifftn(product_grid).real

        # Normalize by sum of |F_obs|² for consistent scale
        norm_factor = (self.F_obs**2).sum().sqrt()
        if norm_factor > 0:
            correlation_map = correlation_map / norm_factor

        # Also compute F_calc norm for later use in proper CC comparison
        self._last_fcalc_norm = (torch.abs(F_calc_sym)**2).sum().sqrt()

        # Find best translation
        best_idx = correlation_map.argmax()
        best_idx_tuple = (
            best_idx // (Ny * Nz),
            (best_idx % (Ny * Nz)) // Nz,
            best_idx % Nz,
        )
        best_translation_frac = torch.tensor(
            [
                float(best_idx_tuple[0]) / Nx,
                float(best_idx_tuple[1]) / Ny,
                float(best_idx_tuple[2]) / Nz,
            ],
            device=device,
        )

        best_correlation = float(correlation_map.flatten()[best_idx].detach())

        return correlation_map, best_translation_frac, best_correlation

    def _compute_symmetry_fcalc(
        self,
        rotation_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute F_calc with symmetry via HKL rotation + phase shifts.

        For each symmetry operation (R_sym, t_sym):
        1. Rotate HKL by symmetry, then by MR rotation: hkl' = hkl @ R_sym @ R_mr
        2. Interpolate complex F_calc at hkl'
        3. Apply symmetry translation phase: exp(2πi * hkl · t_sym)
        4. Sum contributions

        Parameters
        ----------
        rotation_matrix : torch.Tensor
            MR rotation matrix of shape (3, 3).

        Returns
        -------
        torch.Tensor
            Complex F_calc with symmetry applied, shape (N_reflections,).
        """
        device = self.reciprocal_grid.device
        dtype = self.reciprocal_grid.dtype
        n_refl = len(self.hkl)

        # Ensure rotation matrix is on the correct device
        rotation_matrix = rotation_matrix.to(device=device, dtype=torch.float32)

        F_calc_sym = torch.zeros(n_refl, dtype=dtype, device=device)

        # Loop over symmetry operations
        for i in range(self.symmetry.n_ops):
            R_sym = self.symmetry.matrices[i].to(device=device)  # (3, 3)
            t_sym = self.symmetry.translations[i].to(device=device)  # (3,)

            # Combined rotation: first symmetry, then MR rotation
            # For HKL transformation: hkl @ R_sym @ R_mr
            combined_R = R_sym.float() @ rotation_matrix.float()

            # Rotate HKL
            hkl_rotated = self.hkl @ combined_R

            # Interpolate complex F_calc at rotated positions
            F_sym = self._interpolate_complex(self.reciprocal_grid, hkl_rotated)

            # Apply symmetry translation phase: exp(2πi * hkl · t_sym)
            # This accounts for the phase shift due to the symmetry translation
            sym_phase = 2.0 * math.pi * (self.hkl @ t_sym.float())
            phase_factor = torch.complex(torch.cos(sym_phase), torch.sin(sym_phase))
            F_sym = F_sym * phase_factor.to(dtype)

            F_calc_sym = F_calc_sym + F_sym

        return F_calc_sym

    def get_correlation_at_translation(
        self,
        rotation_matrix: torch.Tensor,
        translation_frac: torch.Tensor,
    ) -> float:
        """
        Compute correlation at a specific translation.

        Parameters
        ----------
        rotation_matrix : torch.Tensor
            Rotation matrix of shape (3, 3).
        translation_frac : torch.Tensor
            Translation in fractional coordinates.

        Returns
        -------
        float
            Correlation value.
        """
        # Compute F_calc with symmetry
        F_calc_sym = self._compute_symmetry_fcalc(rotation_matrix)

        # Apply translation phase
        trans_phase = 2.0 * math.pi * (self.hkl @ translation_frac.float())
        phase_factor = torch.complex(
            torch.cos(trans_phase), torch.sin(trans_phase)
        ).to(F_calc_sym.dtype)
        F_calc_translated = F_calc_sym * phase_factor

        # Compute correlation
        # C = Re(Σ F_obs * F_calc*) / sqrt(Σ|F_obs|² * Σ|F_calc|²)
        numerator = (self.F_obs * F_calc_translated.conj()).sum().real
        denom_obs = (self.F_obs**2).sum().sqrt()
        denom_calc = (torch.abs(F_calc_translated) ** 2).sum().sqrt()

        if denom_obs * denom_calc > 0:
            correlation = float((numerator / (denom_obs * denom_calc)).detach())
        else:
            correlation = 0.0

        return correlation


# =============================================================================
# MolecularReplacementPipeline Class
# =============================================================================


class MolecularReplacementPipeline:
    """
    Complete Molecular Replacement pipeline with fast FFT interpolation.

    Pipeline stages:
    1. Rotation search: Fast amplitude interpolation over rotation space
    2. Rotation refinement: Refine top N candidates using interpolation
    3. Translation search: FFT-based correlation with symmetry
    4. Translation refinement: Gradient-based fine-tuning
    5. Joint refinement: Simultaneous rotation + translation using RigidBodyMLTarget

    Parameters
    ----------
    model : ModelFT
        Model with structure factor computation capability.
    data : ReflectionData
        Observed reflection data.
    n_rotation_candidates : int, optional
        Number of top rotations to refine. Default is 10.
    rotation_angular_step : float, optional
        Angular step for rotation search in degrees. Default is 10.0.
    max_refinement_iter : int, optional
        Maximum iterations for refinement stages. Default is 50.
    rms_error : float, optional
        Estimated RMS coordinate error in Angstroms. Default is 1.0.
    high_resolution_limit : float, optional
        High resolution cutoff in Angstroms for fast search targets.
        Reflections with d-spacing below this limit are excluded.
        Default is 3.5 Å (typical for MR rotation search).
        Set to 0.0 to use all reflections.
    verbose : int, optional
        Verbosity level (0=silent, 1=normal, 2=debug). Default is 1.

    Attributes
    ----------
    interpolated_target : InterpolatedMLTarget
        Fast target for rotation search and refinement.
    translation_target : TranslationSearchTarget
        FFT-based target for translation search.
    rigid_target : RigidBodyMLTarget
        Accurate target for final joint refinement.
    """

    def __init__(
        self,
        model: "ModelFT",
        data: "ReflectionData",
        n_rotation_candidates: int = 10,
        rotation_angular_step: float = 10.0,
        max_refinement_iter: int = 50,
        rms_error: float = 1.0,
        high_resolution_limit: float = 3.5,
        verbose: int = 1,
    ):
        self.model = model
        self.data = data
        self.n_rotation_candidates = n_rotation_candidates
        self.rotation_angular_step = rotation_angular_step
        self.max_refinement_iter = max_refinement_iter
        self.rms_error = rms_error
        self.high_resolution_limit = high_resolution_limit
        self.verbose = verbose

        # Store original coordinates for restoration
        self.original_xyz = model.xyz().clone()

        # Create fast target for rotation search and refinement
        # Use resolution limit for speed during search
        self.interpolated_target = InterpolatedMLTarget(
            model, data, rms_error=rms_error, high_resolution_limit=high_resolution_limit
        )

        # Create translation search target (uses all reflections)
        self.translation_target = TranslationSearchTarget(model, data)

        # Create accurate target for final joint refinement
        # Use higher resolution for refinement (limit / 2, or all if limit is 0)
        refinement_limit = high_resolution_limit / 2 if high_resolution_limit > 0 else 0.0
        self.rigid_target = RigidBodyMLTarget(
            model, data, rms_error=rms_error, high_resolution_limit=refinement_limit
        )

    def run(self) -> MRResult:
        """
        Execute complete MR pipeline.

        Returns
        -------
        MRResult
            Complete results including LLG, transformation, and R-factor.
        """
        from torchref.alignment.transform import matrix_to_quaternion

        total_start = time.time()
        stage_timings = {}

        if self.verbose > 0:
            print("=" * 60)
            print("MOLECULAR REPLACEMENT PIPELINE")
            print("=" * 60)

        # Stage 1: Rotation Search
        if self.verbose > 0:
            print("\n[Stage 1] Rotation Search")
            print("-" * 40)

        stage_start = time.time()
        top_llg, top_euler, top_R, best_idx = self.rotation_search()
        stage_timings["rotation_search"] = time.time() - stage_start

        rotation_search_llg = float(top_llg[best_idx])

        if self.verbose > 0:
            print(f"  Best LLG: {rotation_search_llg:.2f}")
            print(f"  Time: {stage_timings['rotation_search']:.2f}s")

        # Stage 2: Rotation Refinement
        if self.verbose > 0:
            print("\n[Stage 2] Rotation Refinement")
            print("-" * 40)

        stage_start = time.time()
        best_quat, rotation_refined_llg, best_rot_idx = self.refine_rotation(
            top_llg, top_R
        )
        stage_timings["rotation_refinement"] = time.time() - stage_start

        if self.verbose > 0:
            print(f"  Best LLG after refinement: {rotation_refined_llg:.2f}")
            print(f"  Time: {stage_timings['rotation_refinement']:.2f}s")

        # Stage 3: Translation Search
        if self.verbose > 0:
            print("\n[Stage 3] Translation Search")
            print("-" * 40)

        stage_start = time.time()
        best_R = quaternion_to_matrix(quaternion_normalize(best_quat))
        best_trans_frac, best_corr = self.translation_search(best_R)
        stage_timings["translation_search"] = time.time() - stage_start

        if self.verbose > 0:
            print(f"  Best correlation: {best_corr:.4f}")
            print(f"  Translation (frac): [{best_trans_frac[0]:.3f}, {best_trans_frac[1]:.3f}, {best_trans_frac[2]:.3f}]")
            print(f"  Time: {stage_timings['translation_search']:.2f}s")

        # Stage 4: Translation Refinement
        if self.verbose > 0:
            print("\n[Stage 4] Translation Refinement")
            print("-" * 40)

        stage_start = time.time()
        refined_trans_frac, trans_llg, trans_converged = self.refine_translation(
            best_quat, best_trans_frac
        )
        stage_timings["translation_refinement"] = time.time() - stage_start

        if self.verbose > 0:
            print(f"  LLG after translation refinement: {trans_llg:.2f}")
            print(f"  Converged: {trans_converged}")
            print(f"  Time: {stage_timings['translation_refinement']:.2f}s")

        # Stage 5: Joint Refinement
        if self.verbose > 0:
            print("\n[Stage 5] Joint Refinement")
            print("-" * 40)

        stage_start = time.time()
        final_quat, final_trans_frac, final_llg, joint_converged = self.joint_refinement(
            best_quat, refined_trans_frac
        )
        stage_timings["joint_refinement"] = time.time() - stage_start

        if self.verbose > 0:
            print(f"  Final LLG: {final_llg:.2f}")
            print(f"  Converged: {joint_converged}")
            print(f"  Time: {stage_timings['joint_refinement']:.2f}s")

        # Compute R-factor
        if self.verbose > 0:
            print("\n[Computing R-factor]")
            print("-" * 40)

        r_work, r_free = self._compute_rfactor(final_quat, final_trans_frac)

        if self.verbose > 0:
            print(f"  R-work: {r_work:.4f}")
            if r_free is not None:
                print(f"  R-free: {r_free:.4f}")

        # Convert translation to Cartesian
        from torchref.math_functions.math_torch import fractional_to_cartesian_torch

        final_trans_cart = fractional_to_cartesian_torch(
            final_trans_frac.unsqueeze(0), self.data.cell
        ).squeeze(0)

        # Build result
        total_time = time.time() - total_start

        if self.verbose > 0:
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETE")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Final LLG: {final_llg:.2f}")
            print(f"  R-factor: {r_work:.4f}")
            print("=" * 60)

        result = MRResult(
            llg=final_llg,
            rotation_quat=final_quat.clone(),
            rotation_matrix=quaternion_to_matrix(quaternion_normalize(final_quat)),
            translation_cart=final_trans_cart.clone(),
            translation_frac=final_trans_frac.clone(),
            r_factor=r_work,
            r_free=r_free,
            rotation_search_llg=rotation_search_llg,
            rotation_refined_llg=rotation_refined_llg,
            translation_search_corr=best_corr,
            joint_refined_llg=final_llg,
            converged=joint_converged,
            runtime_seconds=total_time,
            stage_timings=stage_timings,
        )

        return result

    def rotation_search(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Stage 1: Fast amplitude interpolation rotation search.

        Returns
        -------
        llg_values : torch.Tensor
            LLG values for all rotations.
        euler_angles : torch.Tensor
            Euler angles for all rotations.
        rotation_matrices : torch.Tensor
            Rotation matrices for all rotations.
        best_idx : int
            Index of best rotation.
        """
        llg_values, euler_angles, rotation_matrices, best_idx = (
            self.interpolated_target.rotation_search(
                angular_step_deg=self.rotation_angular_step,
                batch_size=100,
                verbose=self.verbose > 1,
            )
        )

        return llg_values, euler_angles, rotation_matrices, best_idx

    def refine_rotation(
        self,
        llg_values: torch.Tensor,
        rotation_matrices: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, int]:
        """
        Stage 2: Refine top N rotation candidates using interpolation.

        Parameters
        ----------
        llg_values : torch.Tensor
            LLG values from rotation search.
        rotation_matrices : torch.Tensor
            Rotation matrices from rotation search.

        Returns
        -------
        best_quat : torch.Tensor
            Best refined quaternion.
        best_llg : float
            Best LLG after refinement.
        best_idx : int
            Index of best candidate.
        """
        from torchref.alignment.transform import matrix_to_quaternion

        # Get top N candidates
        top_indices = llg_values.argsort(descending=True)[: self.n_rotation_candidates]

        best_llg = -float("inf")
        best_quat = None
        best_idx = 0

        zero_trans = torch.zeros(3, device=rotation_matrices.device)

        for i, idx in enumerate(top_indices):
            R = rotation_matrices[idx]
            quat = matrix_to_quaternion(R)

            if self.verbose > 1:
                print(f"  Refining candidate {i + 1}/{len(top_indices)}...")

            # Refine using numerical gradients on interpolated LLG
            refined_quat, refined_llg = self._refine_rotation_gradient(quat)

            if refined_llg > best_llg:
                best_llg = refined_llg
                best_quat = refined_quat
                best_idx = int(idx)

            if self.verbose > 1:
                print(f"    Initial LLG: {llg_values[idx]:.2f} -> Refined: {refined_llg:.2f}")

        return best_quat, best_llg, best_idx

    def _refine_rotation_gradient(
        self,
        quat: torch.Tensor,
        max_iter: int = 30,
    ) -> Tuple[torch.Tensor, float]:
        """
        Gradient ascent on InterpolatedMLTarget (fast).

        Parameters
        ----------
        quat : torch.Tensor
            Initial quaternion.
        max_iter : int, optional
            Maximum iterations. Default is 30.

        Returns
        -------
        quat : torch.Tensor
            Refined quaternion.
        llg : float
            Final LLG.
        """
        quat = quat.clone()
        zero_trans = torch.zeros(3, device=quat.device)

        eps = 1e-3
        lr = 0.02
        prev_llg = self.interpolated_target.get_llg(quat, zero_trans)

        for iteration in range(max_iter):
            # Numerical gradient for quaternion
            grad = torch.zeros(4, device=quat.device)
            for j in range(4):
                q_plus = quat.clone()
                q_plus[j] += eps
                q_plus = q_plus / q_plus.norm()

                q_minus = quat.clone()
                q_minus[j] -= eps
                q_minus = q_minus / q_minus.norm()

                llg_plus = self.interpolated_target.get_llg(q_plus, zero_trans)
                llg_minus = self.interpolated_target.get_llg(q_minus, zero_trans)
                grad[j] = (llg_plus - llg_minus) / (2 * eps)

            # Gradient ascent (maximize LLG)
            quat = quat + lr * grad
            quat = quat / quat.norm()

            # Check convergence
            current_llg = self.interpolated_target.get_llg(quat, zero_trans)
            if abs(current_llg - prev_llg) < 0.01:
                break
            prev_llg = current_llg

        return quat, self.interpolated_target.get_llg(quat, zero_trans)

    def translation_search(
        self,
        rotation_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Stage 3: FFT-based translation search with symmetry.

        Parameters
        ----------
        rotation_matrix : torch.Tensor
            Rotation matrix from rotation refinement.

        Returns
        -------
        best_trans_frac : torch.Tensor
            Best translation in fractional coordinates.
        best_corr : float
            Best correlation value.
        """
        _, best_trans_frac, best_corr = self.translation_target.translation_function(
            rotation_matrix
        )

        return best_trans_frac, best_corr

    def refine_translation(
        self,
        rotation_quat: torch.Tensor,
        initial_trans_frac: torch.Tensor,
        max_iter: int = 30,
    ) -> Tuple[torch.Tensor, float, bool]:
        """
        Stage 4: Gradient-based translation refinement.

        Parameters
        ----------
        rotation_quat : torch.Tensor
            Fixed rotation quaternion.
        initial_trans_frac : torch.Tensor
            Initial translation in fractional coordinates.
        max_iter : int, optional
            Maximum iterations. Default is 30.

        Returns
        -------
        trans_frac : torch.Tensor
            Refined translation in fractional coordinates.
        llg : float
            Final LLG.
        converged : bool
            Whether optimization converged.
        """
        from torchref.math_functions.math_torch import fractional_to_cartesian_torch

        trans_frac = initial_trans_frac.clone()
        rotation_matrix = quaternion_to_matrix(quaternion_normalize(rotation_quat))

        eps = 0.002  # Small step in fractional coordinates
        lr = 0.01
        converged = False

        # Use correlation from translation target for refinement
        prev_corr = self.translation_target.get_correlation_at_translation(
            rotation_matrix, trans_frac
        )

        for iteration in range(max_iter):
            # Numerical gradient
            grad = torch.zeros(3, device=trans_frac.device)
            for j in range(3):
                t_plus = trans_frac.clone()
                t_plus[j] += eps

                t_minus = trans_frac.clone()
                t_minus[j] -= eps

                corr_plus = self.translation_target.get_correlation_at_translation(
                    rotation_matrix, t_plus
                )
                corr_minus = self.translation_target.get_correlation_at_translation(
                    rotation_matrix, t_minus
                )
                grad[j] = (corr_plus - corr_minus) / (2 * eps)

            # Gradient ascent
            trans_frac = trans_frac + lr * grad

            # Keep in [0, 1) with periodic boundary
            trans_frac = torch.remainder(trans_frac, 1.0)

            # Check convergence
            current_corr = self.translation_target.get_correlation_at_translation(
                rotation_matrix, trans_frac
            )
            if abs(current_corr - prev_corr) < 1e-5:
                converged = True
                break
            prev_corr = current_corr

        # Compute LLG at final position using rigid target
        trans_cart = fractional_to_cartesian_torch(
            trans_frac.unsqueeze(0), self.data.cell
        ).squeeze(0)
        llg = self.rigid_target.get_llg(rotation_quat, trans_cart)

        return trans_frac, llg, converged

    def joint_refinement(
        self,
        rotation_quat: torch.Tensor,
        translation_frac: torch.Tensor,
        max_iter: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, bool]:
        """
        Stage 5: Simultaneous rotation + translation refinement.

        Uses RigidBodyMLTarget for accurate gradients.

        Parameters
        ----------
        rotation_quat : torch.Tensor
            Initial rotation quaternion.
        translation_frac : torch.Tensor
            Initial translation in fractional coordinates.
        max_iter : int, optional
            Maximum iterations. Default is 50.

        Returns
        -------
        final_quat : torch.Tensor
            Final rotation quaternion.
        final_trans_frac : torch.Tensor
            Final translation in fractional coordinates.
        final_llg : float
            Final LLG.
        converged : bool
            Whether optimization converged.
        """
        from torchref.math_functions.math_torch import (
            cartesian_to_fractional_torch,
            fractional_to_cartesian_torch,
        )

        quat = rotation_quat.clone()
        trans_cart = fractional_to_cartesian_torch(
            translation_frac.unsqueeze(0), self.data.cell
        ).squeeze(0)

        # Optimization parameters
        eps_rot = 1e-3
        eps_trans = 0.1
        lr_rot = 0.01
        lr_trans = 0.1
        tolerance = 0.1

        converged = False
        prev_llg = self.rigid_target.get_llg(quat, trans_cart)

        for iteration in range(max_iter):
            # Numerical gradient for quaternion
            grad_rot = torch.zeros(4, device=quat.device)
            for j in range(4):
                q_plus = quat.clone()
                q_plus[j] += eps_rot
                q_plus = q_plus / q_plus.norm()

                q_minus = quat.clone()
                q_minus[j] -= eps_rot
                q_minus = q_minus / q_minus.norm()

                llg_plus = self.rigid_target.get_llg(q_plus, trans_cart)
                llg_minus = self.rigid_target.get_llg(q_minus, trans_cart)
                grad_rot[j] = (llg_plus - llg_minus) / (2 * eps_rot)

            # Numerical gradient for translation
            grad_trans = torch.zeros(3, device=trans_cart.device)
            for j in range(3):
                t_plus = trans_cart.clone()
                t_plus[j] += eps_trans

                t_minus = trans_cart.clone()
                t_minus[j] -= eps_trans

                llg_plus = self.rigid_target.get_llg(quat, t_plus)
                llg_minus = self.rigid_target.get_llg(quat, t_minus)
                grad_trans[j] = (llg_plus - llg_minus) / (2 * eps_trans)

            # Gradient ascent
            quat = quat + lr_rot * grad_rot
            quat = quat / quat.norm()

            trans_cart = trans_cart + lr_trans * grad_trans

            # Check convergence
            current_llg = self.rigid_target.get_llg(quat, trans_cart)

            if self.verbose > 1 and iteration % 10 == 0:
                print(f"    Iter {iteration}: LLG = {current_llg:.2f}")

            if abs(current_llg - prev_llg) < tolerance:
                converged = True
                break
            prev_llg = current_llg

        # Convert translation back to fractional
        final_trans_frac = cartesian_to_fractional_torch(
            trans_cart.unsqueeze(0), self.data.cell
        ).squeeze(0)

        # Keep in [0, 1) with periodic boundary
        final_trans_frac = torch.remainder(final_trans_frac, 1.0)

        final_llg = self.rigid_target.get_llg(quat, trans_cart)

        return quat, final_trans_frac, final_llg, converged

    def _compute_rfactor(
        self,
        rotation_quat: torch.Tensor,
        translation_frac: torch.Tensor,
    ) -> Tuple[float, Optional[float]]:
        """
        Compute R-factor for validation.

        Parameters
        ----------
        rotation_quat : torch.Tensor
            Rotation quaternion.
        translation_frac : torch.Tensor
            Translation in fractional coordinates.

        Returns
        -------
        r_work : float
            R-factor on working set.
        r_free : Optional[float]
            R-factor on free set (if available).
        """
        from torchref.math_functions.math_torch import fractional_to_cartesian_torch

        # Apply transformation
        trans_cart = fractional_to_cartesian_torch(
            translation_frac.unsqueeze(0), self.data.cell
        ).squeeze(0)

        xyz_transformed = self.rigid_target.transform_coordinates(
            rotation_quat, trans_cart
        )
        self.model.xyz[:] = xyz_transformed

        # Compute F_calc for all reflections
        self.model.reset_cache()
        F_calc = self.model.get_structure_factor(self.data.hkl, recalc=True)
        F_calc_amp = torch.abs(F_calc)

        F_obs = self.data.F

        # Scale factor: k = Σ(F_obs * F_calc) / Σ(F_calc²)
        scale = (F_obs * F_calc_amp).sum() / (F_calc_amp**2).sum().clamp(min=1e-10)

        # R-factor: R = Σ||F_obs| - k|F_calc|| / Σ|F_obs|
        r_work = float(
            (torch.abs(F_obs - scale * F_calc_amp).sum() / F_obs.sum()).detach()
        )

        # Compute R-free if available
        r_free = None
        if hasattr(self.data, "rfree_flags") and self.data.rfree_flags is not None:
            mask = self.data.rfree_flags
            F_obs_free = F_obs[mask]
            F_calc_free = F_calc_amp[mask]

            if len(F_obs_free) > 0:
                scale_free = (
                    (F_obs_free * F_calc_free).sum()
                    / (F_calc_free**2).sum().clamp(min=1e-10)
                )
                r_free = float(
                    (torch.abs(F_obs_free - scale_free * F_calc_free).sum()
                    / F_obs_free.sum()).detach()
                )

        # Restore original coordinates
        self.model.xyz[:] = self.original_xyz

        return r_work, r_free
