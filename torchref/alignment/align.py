"""
Main Patterson alignment class for TorchRef.

Aligns predicted structures to observed diffraction data via Patterson matching.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from torchref.io.datasets.reflection_data import ReflectionData
from torchref.math_functions.math_torch import (
    axis_angle_to_rotation_matrix,
    cartesian_to_fractional_torch,
    fractional_to_cartesian_torch,
    rotation_matrix_to_axis_angle,
    trilinear_interpolate_patterson,
)
from torchref.model.model import Model

from .clashscore import AtomSampler, ClashScoreCalculator
from .sampling import VectorSampler
from .transform import RigidTransform


@dataclass
class AlignmentResult:
    """
    Result of Patterson alignment.

    Attributes
    ----------
    transform : RigidTransform
        Rigid body transformation (rotation + translation).
    score : float
        Patterson correlation score (higher = better match).
    n_starts : int
        Number of random starts used.
    converged : bool
        Whether optimization converged.
    correlation : float, optional
        Correlation coefficient between F_obs and F_calc (if computed).

    Properties
    ----------
    rotation : torch.Tensor
        Rotation matrix (3, 3) in Cartesian space. (Backward compatibility)
    translation : torch.Tensor
        Translation vector (3,) in Cartesian coordinates. (Backward compatibility)
    """

    transform: RigidTransform
    score: float
    n_starts: int
    converged: bool
    correlation: Optional[float] = None

    @property
    def rotation(self) -> torch.Tensor:
        """Get rotation matrix (3, 3) for backward compatibility."""
        return self.transform.rotation_matrix

    @property
    def translation(self) -> torch.Tensor:
        """Get translation vector (3,) for backward compatibility."""
        return self.transform.translation

    def apply(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation to coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates with shape (N, 3).

        Returns
        -------
        torch.Tensor
            Transformed coordinates with shape (N, 3).
        """
        return self.transform.apply(coords)

    def as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (R, t) as numpy arrays.

        Returns
        -------
        R : np.ndarray
            Rotation matrix (3, 3).
        t : np.ndarray
            Translation vector (3,).
        """
        return self.rotation.cpu().numpy(), self.translation.cpu().numpy()


class PattersonAligner:
    """
    Align predicted structures to diffraction data via Patterson matching.

    This class takes TorchRef Model and ReflectionData objects directly,
    following the TorchRef API conventions. It also provides clash scoring
    to detect steric clashes between symmetry-related molecules.

    Parameters
    ----------
    data : ReflectionData
        Reflection data containing observed amplitudes (F), Miller indices (hkl),
        unit cell, and space group.
    model : Model
        Model object for extracting symmetry and initial structure.
        Water molecules are automatically excluded.
    n_vectors : int, optional
        Number of atom pairs to sample for scoring. Default is 10000.
    weighting : str, optional
        Sampling weight scheme: 'uniform', 'Z2', or 'Z2_distance'. Default is 'Z2'.
    verbose : int, optional
        Verbosity level (0=silent, 1=normal, 2=debug). Default is 1.

    Attributes
    ----------
    patterson : torch.Tensor
        Precomputed Patterson map (nx, ny, nz) from data.calc_patterson().
    cell : torch.Tensor
        Unit cell parameters from data.
    symmetry : Symmetry
        Symmetry operations object from model.symmetry.
    model : Model
        Reference to the input model.
    device : torch.device
        Computation device (from data).
    clash_calculator : ClashScoreCalculator
        Pre-initialized clash score calculator for the space group.

    Examples
    --------
    ::

        from torchref.model import Model
        from torchref.io.datasets.reflection_data import ReflectionData
        from torchref.alignment import PattersonAligner

        # Load data and model
        data = ReflectionData(verbose=1).load_mtz('observed.mtz')
        model = Model(verbose=1).load_pdb('predicted.pdb')

        # Create aligner (precomputes Patterson map and clash calculator)
        aligner = PattersonAligner(data, model)

        # Align model to data
        aligned_model, result = aligner.align(n_starts=20)
        print(f"Score: {result.score:.4f}")

        # Check for clashes in the aligned model
        clash_score = aligner.compute_clash_score(model=aligned_model)
        print(f"Clash score: {clash_score.item():.4f}")

        # Save aligned structure
        aligned_model.write_pdb('aligned.pdb')
    """

    def __init__(
        self,
        data: ReflectionData,
        model: Model,
        n_vectors: int = 1000000,
        weighting: str = "Z2",
        verbose: int = 1,
    ):
        # Store references
        self.data = data
        self.verbose = verbose
        self.device = (
            torch.device(data.device) if isinstance(data.device, str) else data.device
        )
        self.n_vectors = n_vectors
        self.weighting = weighting

        # Get cell parameters (use data.cell as source of truth)
        self.cell = data.cell.clone()

        # Store symmetry from model
        self.symmetry = model.symmetry

        # Store reference to model (filtering done in align() method)
        self.model = model

        # Precompute Patterson map and normalize for consistent scoring
        patterson_raw = data.calc_patterson()
        # Normalize to zero mean and unit variance for stable optimization
        self.patterson_mean = patterson_raw.mean()
        self.patterson_std = patterson_raw.std()
        self.patterson = (patterson_raw - self.patterson_mean) / (
            self.patterson_std + 1e-8
        )

        # Initialize clash score calculator for symmetry-based clash detection
        self.clash_calculator = ClashScoreCalculator(
            symmetry=model.spacegroup,
            device=self.device,
        )

        # Pre-compute atom selection mask for clash scoring
        self._clash_atom_mask = AtomSampler.from_model(model, mode="auto")

    def _local_search_around_quaternion(
        self,
        q: torch.Tensor,
        angular_delta: float = 5.0,
        n_samples: int = 10,
    ) -> torch.Tensor:
        """
        Generate nearby rotations around a given quaternion.

        Creates a local search grid by perturbing Euler angles around the
        rotation represented by the input quaternion.

        Parameters
        ----------
        q : torch.Tensor
            Center quaternion [w, x, y, z] with shape (4,).
        angular_delta : float, default 5.0
            Angular perturbation range in degrees.
        n_samples : int, default 10
            Number of nearby quaternions to generate.

        Returns
        -------
        torch.Tensor
            Quaternions with shape (n_samples, 4).
        """
        device = q.device
        dtype = q.dtype

        # Convert quaternion to Euler angles (ZYZ convention)
        # This allows us to perturb in a meaningful way
        R = self._quaternion_to_matrix(q)

        # Extract Euler angles from rotation matrix (ZYZ convention)
        # R = Rz(alpha) @ Ry(beta) @ Rz(gamma)
        beta = torch.acos(torch.clamp(R[2, 2], -1.0, 1.0))

        if torch.abs(torch.sin(beta)) < 1e-6:
            # Gimbal lock: beta ≈ 0 or pi
            alpha = torch.atan2(R[1, 0], R[0, 0])
            gamma = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            alpha = torch.atan2(R[2, 1], R[2, 0])
            gamma = torch.atan2(R[1, 2], -R[0, 2])

        # Generate perturbations
        delta_rad = angular_delta * np.pi / 180.0
        quaternions = []

        # Sample perturbations uniformly in the local neighborhood
        for _ in range(n_samples):
            # Random perturbations in each Euler angle
            d_alpha = (torch.rand(1, device=device, dtype=dtype) - 0.5) * 2 * delta_rad
            d_beta = (torch.rand(1, device=device, dtype=dtype) - 0.5) * 2 * delta_rad
            d_gamma = (torch.rand(1, device=device, dtype=dtype) - 0.5) * 2 * delta_rad

            new_alpha = alpha + d_alpha.item()
            new_beta = torch.clamp(beta + d_beta.item(), 0.0, np.pi)
            new_gamma = gamma + d_gamma.item()

            # Convert back to quaternion
            ca, sa = torch.cos(new_alpha / 2), torch.sin(new_alpha / 2)
            cb, sb = torch.cos(new_beta / 2), torch.sin(new_beta / 2)
            cg, sg = torch.cos(new_gamma / 2), torch.sin(new_gamma / 2)

            w = ca * cb * cg - sa * cb * sg
            x = ca * sb * cg + sa * sb * sg
            y = sa * sb * cg - ca * sb * sg
            z = sa * cb * cg + ca * cb * sg

            q_new = torch.stack([w, x, y, z])
            q_new = q_new / q_new.norm()  # Normalize
            quaternions.append(q_new)

        return torch.stack(quaternions)

    def compute_clash_score(
        self,
        xyz: Optional[torch.Tensor] = None,
        model: Optional[Model] = None,
        clash_radius: float = 5.0,
    ) -> torch.Tensor:
        """
        Compute clash score for given coordinates or model.

        Parameters
        ----------
        xyz : torch.Tensor, optional
            Cartesian coordinates (N, 3). If None, uses model.
        model : Model, optional
            Model to compute clash score for. If None, uses self.model.
        clash_radius : float, default 5.0
            Minimum allowed distance between atoms in Angstroms.

        Returns
        -------
        torch.Tensor
            Scalar clash score. Lower values indicate fewer clashes.
        """
        if xyz is None:
            if model is None:
                model = self.model
            xyz = model.xyz().detach()
            # Use pre-computed mask only if using self.model
            atom_mask = (
                self._clash_atom_mask
                if model is self.model
                else AtomSampler.from_model(model)
            )
        else:
            # When xyz is provided directly, don't use atom mask
            # (caller is responsible for providing appropriate coordinates)
            atom_mask = None

        return self.clash_calculator(
            xyz=xyz,
            cell=self.cell,
            atom_mask=atom_mask,
            clash_radius=clash_radius,
        )

    def compute_correlation_for_transform(
        self,
        transform: RigidTransform,
        model: Optional[Model] = None,
    ) -> float:
        """
        Compute correlation coefficient between F_obs and F_calc for a transformation.

        Creates a temporary ModelFT, applies the transformation, computes
        structure factors, and evaluates correlation with observed data.
        Correlation is scale-invariant, making it more robust than R-factor
        for candidate ranking.

        Parameters
        ----------
        transform : RigidTransform
            Transformation to apply to the model.
        model : Model, optional
            Model to transform. If None, uses self.model.

        Returns
        -------
        float
            Correlation coefficient between F_obs and F_calc (higher = better).
        """
        from torchref.model.model_ft import ModelFT

        if model is None:
            model = self.model

        # Create ModelFT by copying the input model
        model_ft = model.copy()

        # Apply transformation to coordinates
        transformed_xyz = transform.apply(model_ft.xyz)
        model_ft.xyz[:] = transformed_xyz

        # Convert to ModelFT if not already (for structure factor calculation)
        if not isinstance(model_ft, ModelFT):
            # Need to wrap as ModelFT for SF calculation
            temp = ModelFT(verbose=0)
            temp.pdb = model_ft.pdb
            temp.cell = model_ft.cell
            temp.xyz = model_ft.xyz
            temp.adp = model_ft.adp
            temp.occupancy = model_ft.occupancy
            temp.spacegroup = model_ft.spacegroup
            temp._symmetry = model_ft._symmetry
            temp._build_parametrization()
            temp.setup_grid()
            model_ft = temp
        else:
            model_ft._build_parametrization()
            model_ft.setup_grid()

        # Compute structure factors
        hkl = self.data.hkl
        with torch.no_grad():
            F_calc_complex = model_ft.get_structure_factor(hkl, recalc=True)
            F_calc = torch.abs(F_calc_complex)

        # Get F_obs
        F_obs = self.data.F

        # Compute Pearson correlation (scale-invariant)
        F_obs_centered = F_obs - F_obs.mean()
        F_calc_centered = F_calc - F_calc.mean()

        numerator = (F_obs_centered * F_calc_centered).sum()
        denominator = torch.sqrt((F_obs_centered**2).sum() * (F_calc_centered**2).sum())

        correlation = (numerator / (denominator + 1e-8)).item()

        return correlation

    def interpolate_patterson(self, fractional_vectors: torch.Tensor) -> torch.Tensor:
        """
        Interpolate Patterson map at fractional coordinate positions.

        Uses trilinear interpolation with periodic boundary handling.
        The Patterson function is periodic, so vectors are wrapped to [0, 1).

        Parameters
        ----------
        fractional_vectors : torch.Tensor
            Fractional coordinate vectors with shape (n_vectors, 3).
            These are difference vectors in fractional space.

        Returns
        -------
        torch.Tensor
            Interpolated Patterson values with shape (n_vectors,).
        """
        # Ensure vectors are on the same device as Patterson map
        vecs = fractional_vectors.to(
            device=self.patterson.device, dtype=self.patterson.dtype
        )

        # Use trilinear_interpolate with periodic wrapping
        # This function handles wrapping to [0, 1) internally
        return trilinear_interpolate_patterson(self.patterson, vecs)

    def evaluate_vectors_on_coords(
        self, idx1: torch.Tensor, idx2: torch.Tensor, xyz_fractional: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate Patterson score for intra-ASU atom pair vectors.

        Parameters
        ----------
        idx1 : torch.Tensor
            Indices of first atoms in pairs (n_vectors,).
        idx2 : torch.Tensor
            Indices of second atoms in pairs (n_vectors,).
        xyz_fractional : torch.Tensor
            Fractional coordinates of atoms (N, 3).

        Returns
        -------
        torch.Tensor
            Mean Patterson score for intra-ASU vectors.
        """
        # Compute difference vectors within the ASU
        vecs = xyz_fractional[idx1] - xyz_fractional[idx2]

        # Interpolate Patterson map at vector positions
        scores = self.interpolate_patterson(vecs)

        return scores.mean()

    def evaluate_intercopy_vectors(
        self,
        idx1: torch.Tensor,
        idx2: torch.Tensor,
        xyz_fractional: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate Patterson score for inter-copy vectors between ASU and symmetry mates.

        These vectors ARE translation-dependent (unlike intra-ASU vectors) because:
        inter-copy vector = r_i - (R_sym @ r_j + t_sym)

        When ASU is translated by T:
        new vector = (r_i + T) - (R_sym @ (r_j + T) + t_sym)
                   = r_i - R_sym @ r_j - t_sym + (I - R_sym) @ T

        The (I - R_sym) @ T term makes this translation-dependent.

        Parameters
        ----------
        idx1 : torch.Tensor
            Indices of atoms in ASU (n_vectors,).
        idx2 : torch.Tensor
            Indices of atoms to transform via symmetry (n_vectors,).
        xyz_fractional : torch.Tensor
            Fractional coordinates of ASU atoms (N, 3).

        Returns
        -------
        torch.Tensor
            Mean Patterson score for inter-copy vectors.
        """
        all_scores = []

        # Get symmetry operations (skip identity at index 0)
        n_ops = self.symmetry.n_ops
        dtype = xyz_fractional.dtype
        device = xyz_fractional.device

        for op_idx in range(1, n_ops):  # Skip identity
            R_sym = self.symmetry.matrices[op_idx].to(dtype=dtype, device=device)
            t_sym = self.symmetry.translations[op_idx].to(dtype=dtype, device=device)

            # Generate symmetry mate positions for selected atoms
            # xyz_mate = R_sym @ xyz + t_sym
            xyz_mate = xyz_fractional @ R_sym.T + t_sym

            # Inter-copy vectors: ASU atom i to symmetry mate of atom j
            vecs = xyz_fractional[idx1] - xyz_mate[idx2]

            # Score these vectors
            scores = self.interpolate_patterson(vecs)
            all_scores.append(scores)

        if len(all_scores) == 0:
            # Only identity operation (P1 space group)
            return torch.tensor(0.0, device=device, dtype=dtype)

        # Combine scores from all symmetry operations
        all_scores = torch.cat(all_scores)
        return all_scores.mean()

    def score_transformation(
        self,
        xyz_cartesian: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        idx1: torch.Tensor,
        idx2: torch.Tensor,
        intercopy_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Score a rotation and translation by Patterson vector matching.

        Combines intra-ASU vectors (rotation-dependent only) with inter-copy
        vectors (both rotation and translation dependent) for a complete
        Patterson correlation score.

        This method is differentiable for use with gradient-based optimization.

        Parameters
        ----------
        xyz_cartesian : torch.Tensor
            Cartesian coordinates of atoms (N, 3).
        rotation : torch.Tensor
            Rotation matrix (3, 3) in Cartesian space.
        translation : torch.Tensor
            Translation vector (3,) in Cartesian coordinates.
        idx1 : torch.Tensor
            Indices of first atoms in pairs (n_vectors,).
        idx2 : torch.Tensor
            Indices of second atoms in pairs (n_vectors,).
        intercopy_weight : float, default 1.0
            Weight for inter-copy vectors relative to intra-ASU vectors.

        Returns
        -------
        torch.Tensor
            Patterson score (scalar tensor for gradient computation).
        """
        # Apply rotation and translation
        xyz_transformed = xyz_cartesian @ rotation.T + translation

        # Convert to fractional coordinates
        xyz_fractional = cartesian_to_fractional_torch(xyz_transformed, self.cell)

        # Evaluate intra-ASU Patterson score (rotation-dependent only)
        intra_score = self.evaluate_vectors_on_coords(idx1, idx2, xyz_fractional)

        # Evaluate inter-copy Patterson score (rotation AND translation dependent)
        inter_score = self.evaluate_intercopy_vectors(idx1, idx2, xyz_fractional)

        # Combine scores
        return intra_score + intercopy_weight * inter_score

    def _score_quaternions(
        self,
        quaternions: torch.Tensor,
        xyz_cartesian: torch.Tensor,
        idx1: torch.Tensor,
        idx2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a batch of quaternions using Patterson vector matching.

        Parameters
        ----------
        quaternions : torch.Tensor
            Quaternions with shape (N, 4).
        xyz_cartesian : torch.Tensor
            Cartesian coordinates (M, 3).
        idx1, idx2 : torch.Tensor
            Atom pair indices for Patterson scoring.

        Returns
        -------
        torch.Tensor
            Scores for each quaternion (N,).
        """
        device = xyz_cartesian.device
        dtype = xyz_cartesian.dtype
        n_quats = quaternions.shape[0]
        scores = torch.zeros(n_quats, device=device, dtype=dtype)

        with torch.no_grad():
            for i, q in enumerate(quaternions):
                R = self._quaternion_to_matrix(q)
                xyz_rot = xyz_cartesian @ R.T
                xyz_frac = cartesian_to_fractional_torch(xyz_rot, self.cell)
                score = self.evaluate_vectors_on_coords(idx1, idx2, xyz_frac)
                scores[i] = score

        return scores

    def _grid_search_rotation(
        self,
        xyz_cartesian: torch.Tensor,
        idx1: Optional[torch.Tensor] = None,
        idx2: Optional[torch.Tensor] = None,
        angular_step: float = 15.0,
        n_top: int = 10,
        n_vectors_coarse: int = 100000,
        n_vectors_fine: int = 10000000,
        n_top_coarse: int = 100,
        n_local_samples: int = 10,
        local_angular_delta: float = 5.0,
        use_symmetry: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Three-stage grid search over rotations for efficient Patterson alignment.

        This method implements an efficient three-stage search strategy:

        Stage 1 (Coarse Grid): Screen all rotations with fewer vectors (100k).
                              Identify top 100 candidates quickly.

        Stage 2 (Local Search): For each of the top 100 candidates, test 10
                               nearby rotations with many vectors (10M).
                               Total: 1000 rotations with high precision.

        Stage 3 (Select Top): Return the top N candidates for LBFGS refinement.

        This is much more efficient than testing all rotations with 10M vectors.

        Parameters
        ----------
        xyz_cartesian : torch.Tensor
            Cartesian coordinates (N, 3), already centered.
        idx1, idx2 : torch.Tensor, optional
            Deprecated - atom pair indices are now created internally.
        angular_step : float, default 15.0
            Angular resolution in degrees for coarse grid.
        n_top : int, default 10
            Number of final candidates to return for LBFGS refinement.
        n_vectors_coarse : int, default 100000
            Number of vectors for Stage 1 coarse screening.
        n_vectors_fine : int, default 10000000
            Number of vectors for Stage 2 local search.
        n_top_coarse : int, default 100
            Number of candidates to advance from Stage 1 to Stage 2.
        n_local_samples : int, default 10
            Number of local perturbations per coarse candidate in Stage 2.
        local_angular_delta : float, default 5.0
            Angular perturbation range in degrees for local search.
        use_symmetry : bool, default True
            If True, reduce grid based on point group symmetry.
        seed : int, optional
            Random seed for reproducibility of sampling.

        Returns
        -------
        top_quaternions : torch.Tensor
            Top quaternions with shape (n_top, 4).
        top_scores : torch.Tensor
            Corresponding Patterson scores (n_top,).
        """
        device = xyz_cartesian.device
        dtype = xyz_cartesian.dtype

        # Create samplers for coarse and fine stages
        model_no_water = self.model.select("not resname HOH")
        sampler = VectorSampler(model_no_water, weighting=self.weighting, seed=seed)

        # ========================================
        # Stage 1: Coarse grid screening
        # ========================================
        if self.verbose > 0:
            print(f"\n  Stage 1: Coarse grid screening ({n_vectors_coarse:,} vectors)")

        # Sample vectors for coarse search
        idx1_coarse, idx2_coarse = sampler.sample(n_vectors_coarse)

        # Generate quaternion grid (optionally reduced by symmetry)
        quaternions = self.generate_quaternion_grid(
            angular_step=angular_step,
            device=str(device),
            dtype=dtype,
            use_symmetry_reduction=use_symmetry,
        )
        n_quats = quaternions.shape[0]

        if self.verbose > 0:
            print(f"    Grid: {n_quats} rotations at {angular_step}° resolution")
            if use_symmetry:
                print("    (reduced by point group symmetry)")

        # Score all rotations with coarse vectors
        scores_coarse = self._score_quaternions(
            quaternions, xyz_cartesian, idx1_coarse, idx2_coarse
        )

        if self.verbose > 1:
            print(
                f"    Coarse scores range: [{scores_coarse.min():.4f}, {scores_coarse.max():.4f}]"
            )

        # Select top candidates from Stage 1
        top_coarse_indices = torch.argsort(scores_coarse, descending=True)[
            :n_top_coarse
        ]
        top_coarse_quats = quaternions[top_coarse_indices]
        top_coarse_scores = scores_coarse[top_coarse_indices]

        if self.verbose > 0:
            print(
                f"    Top {n_top_coarse} coarse scores: {top_coarse_scores[:5].tolist()}"
            )

        # ========================================
        # Stage 2: Local search around top candidates
        # ========================================
        if self.verbose > 0:
            print(f"\n  Stage 2: Local search ({n_vectors_fine:,} vectors)")
            print(
                f"    Testing {n_local_samples} points around each of {n_top_coarse} candidates"
            )
            print(f"    Total: {n_top_coarse * n_local_samples} rotations")

        # Sample vectors for fine search (resample with new seed to avoid correlation)
        if seed is not None:
            sampler_fine = VectorSampler(
                model_no_water, weighting=self.weighting, seed=seed + 1
            )
        else:
            sampler_fine = VectorSampler(model_no_water, weighting=self.weighting)
        idx1_fine, idx2_fine = sampler_fine.sample(n_vectors_fine)

        # Generate local perturbations for each top candidate
        local_quats = []
        local_source_indices = (
            []
        )  # Track which coarse candidate each local quat came from

        for i, q in enumerate(top_coarse_quats):
            # Include the original quaternion
            local_quats.append(q.unsqueeze(0))
            local_source_indices.append(i)

            # Generate nearby perturbations
            nearby = self._local_search_around_quaternion(
                q, angular_delta=local_angular_delta, n_samples=n_local_samples - 1
            )
            local_quats.append(nearby)
            local_source_indices.extend([i] * (n_local_samples - 1))

        local_quats = torch.cat(local_quats, dim=0)
        n_local = local_quats.shape[0]

        if self.verbose > 1:
            print(f"    Scoring {n_local} local rotations...")

        # Score all local rotations with fine vectors
        scores_fine = self._score_quaternions(
            local_quats, xyz_cartesian, idx1_fine, idx2_fine
        )

        if self.verbose > 1:
            print(
                f"    Fine scores range: [{scores_fine.min():.4f}, {scores_fine.max():.4f}]"
            )

        # ========================================
        # Stage 3: Select final top candidates
        # ========================================
        if self.verbose > 0:
            print(f"\n  Stage 3: Selecting top {n_top} candidates for refinement")

        top_final_indices = torch.argsort(scores_fine, descending=True)[:n_top]
        top_quaternions = local_quats[top_final_indices]
        top_scores = scores_fine[top_final_indices]

        if self.verbose > 0:
            print(f"    Final top {n_top} scores: {top_scores.tolist()}")

        return top_quaternions, top_scores

    def _quaternion_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion [w, x, y, z] to 3x3 rotation matrix.

        Parameters
        ----------
        q : torch.Tensor
            Quaternion with shape (4,).

        Returns
        -------
        torch.Tensor
            Rotation matrix with shape (3, 3).
        """
        w, x, y, z = q[0], q[1], q[2], q[3]

        # Rotation matrix from quaternion
        R = torch.stack(
            [
                torch.stack(
                    [
                        1 - 2 * y * y - 2 * z * z,
                        2 * x * y - 2 * w * z,
                        2 * x * z + 2 * w * y,
                    ]
                ),
                torch.stack(
                    [
                        2 * x * y + 2 * w * z,
                        1 - 2 * x * x - 2 * z * z,
                        2 * y * z - 2 * w * x,
                    ]
                ),
                torch.stack(
                    [
                        2 * x * z - 2 * w * y,
                        2 * y * z + 2 * w * x,
                        1 - 2 * x * x - 2 * y * y,
                    ]
                ),
            ]
        )

        return R

    def _optimize_single_start(
        self,
        xyz_cartesian: torch.Tensor,
        idx1: torch.Tensor,
        idx2: torch.Tensor,
        R_init: torch.Tensor,
        t_init_frac: torch.Tensor,
        max_iter: int = 100,
        lr: float = 0.1,
        clash_weight: float = 1.0,
        clash_radius: float = 5.0,
        clash_atom_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float, bool]:
        """
        Run single optimization from given starting point.

        Parameters
        ----------
        xyz_cartesian : torch.Tensor
            Cartesian coordinates (N, 3).
        idx1, idx2 : torch.Tensor
            Atom pair indices.
        R_init : torch.Tensor
            Initial rotation matrix (3, 3).
        t_init_frac : torch.Tensor
            Initial translation vector (3,) in FRACTIONAL coordinates [0, 1).
        max_iter : int
            Maximum LBFGS iterations.
        lr : float
            Learning rate for LBFGS.
        clash_weight : float
            Weight for clash penalty in loss function.
        clash_radius : float
            Minimum allowed distance between atoms for clash detection.
        clash_atom_mask : torch.Tensor, optional
            Boolean mask selecting atoms for clash detection (e.g., CA only).
            If None, uses all atoms.

        Returns
        -------
        R_opt : torch.Tensor
            Optimized rotation matrix.
        t_opt : torch.Tensor
            Optimized translation vector in Cartesian coordinates.
        patterson_score : float
            Final Patterson score.
        clash_score : float
            Final clash score.
        converged : bool
            Whether optimization converged.
        """
        device = xyz_cartesian.device
        dtype = xyz_cartesian.dtype

        # Convert rotation to axis-angle parameterization
        rot_params = rotation_matrix_to_axis_angle(R_init).to(
            dtype=dtype, device=device
        )
        rot_params = rot_params.clone().requires_grad_(True)

        # Translation parameters in FRACTIONAL space for better optimization
        # Patterson is periodic in fractional coords, so [0,1) is the natural search space
        trans_frac_params = (
            t_init_frac.clone().to(dtype=dtype, device=device).requires_grad_(True)
        )

        optimizer = torch.optim.LBFGS(
            [rot_params, trans_frac_params],
            max_iter=max_iter,
            lr=lr,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=20,
        )

        final_patterson_score = 0.0
        final_clash_score = 0.0

        def closure():
            nonlocal final_patterson_score, final_clash_score
            optimizer.zero_grad()

            # Convert axis-angle to rotation matrix
            R = axis_angle_to_rotation_matrix(rot_params)

            # Convert fractional translation to Cartesian
            # Wrap to [0, 1) for periodicity
            trans_frac_wrapped = trans_frac_params % 1.0
            trans_cart = fractional_to_cartesian_torch(
                trans_frac_wrapped.unsqueeze(0), self.cell
            ).squeeze(0)

            # Compute Patterson score (we want to maximize)
            patterson_score = self.score_transformation(
                xyz_cartesian, R, trans_cart, idx1, idx2
            )

            # Compute clash score (we want to minimize)
            # Use CA atoms only for efficiency (clash_atom_mask)
            xyz_transformed = xyz_cartesian @ R.T + trans_cart
            clash_score = self.clash_calculator(
                xyz=xyz_transformed,
                cell=self.cell,
                atom_mask=clash_atom_mask,
                clash_radius=clash_radius,
            )

            # Combined loss: minimize -patterson + clash_weight * clash
            loss = -patterson_score + clash_weight * clash_score
            loss.backward()

            final_patterson_score = patterson_score.item()
            final_clash_score = clash_score.item()
            return loss

        try:
            optimizer.step(closure)
            converged = True
        except RuntimeError as e:
            if self.verbose > 1:
                print(f"    Optimization warning: {e}")
            converged = False

        # Get final rotation matrix and translation
        with torch.no_grad():
            R_final = axis_angle_to_rotation_matrix(rot_params)
            # Convert final fractional translation to Cartesian
            trans_frac_final = trans_frac_params % 1.0
            t_final = fractional_to_cartesian_torch(
                trans_frac_final.unsqueeze(0), self.cell
            ).squeeze(0)

        return R_final, t_final, final_patterson_score, final_clash_score, converged

    def align(
        self,
        model: Optional[Model] = None,
        n_refine: int = 10,
        n_vectors: Optional[int] = None,
        max_iter: int = 50,
        seed: Optional[int] = None,
        clash_weight: float = 1.0,
        clash_radius: float = 5.0,
        grid_angular_step: float = 15.0,
        use_correlation_scoring: bool = False,
        n_vectors_coarse: int = 100000,
        n_vectors_fine: int = 10000000,
        n_top_coarse: int = 100,
        n_local_samples: int = 10,
        local_angular_delta: float = 5.0,
        use_symmetry: bool = True,
    ) -> Tuple[Model, AlignmentResult]:
        """
        Align a model to the diffraction data via Patterson matching.

        Uses an efficient three-stage approach:

        Stage 1 (Coarse Grid): Screen all rotations with fewer vectors (100k).
                              Optionally reduced by crystallographic symmetry.

        Stage 2 (Local Search): For each top candidate from Stage 1, test nearby
                               rotations with many vectors (10M) for discrimination.

        Stage 3 (LBFGS Refinement): Refine top candidates from Stage 2 with
                                   gradient-based optimization.

        Stage 4 (Optional): Correlation scoring to select the best candidate.

        The Patterson function has inherent ambiguities that can lead to false peaks.
        When `use_correlation_scoring=True`, the method computes F_obs/F_calc correlation
        for all refined candidates and selects the one with highest correlation.

        Parameters
        ----------
        model : Model, optional
            Model to align. If None, uses self.model.
        n_refine : int, optional
            Number of top candidates to refine with LBFGS. Default is 10.
            When use_correlation_scoring=True and n_refine < 50, automatically
            increased to 50 for better coverage.
        n_vectors : int, optional
            Number of atom pairs for LBFGS refinement. Default is self.n_vectors.
        max_iter : int, optional
            Maximum iterations per LBFGS optimization. Default is 50.
        seed : int, optional
            Random seed for reproducibility. Default is None.
        clash_weight : float, optional
            Weight for clash penalty in optimization. Default is 1.0.
        clash_radius : float, optional
            Minimum allowed distance between atoms (Angstroms). Default is 5.0.
        grid_angular_step : float, optional
            Angular step for coarse grid search in degrees. Default is 15.0.
        use_correlation_scoring : bool, optional
            If True, compute F_obs/F_calc correlation for all refined candidates
            and select the one with highest correlation. Default is False.
        n_vectors_coarse : int, optional
            Number of vectors for Stage 1 coarse screening. Default is 100000.
        n_vectors_fine : int, optional
            Number of vectors for Stage 2 local search. Default is 10000000.
        n_top_coarse : int, optional
            Number of candidates from Stage 1 to advance to Stage 2. Default is 100.
        n_local_samples : int, optional
            Number of local rotations to test around each Stage 1 candidate. Default is 10.
        local_angular_delta : float, optional
            Angular range for local search perturbations in degrees. Default is 5.0.
        use_symmetry : bool, optional
            If True, reduce rotation grid based on point group symmetry. Default is True.

        Returns
        -------
        aligned_model : Model
            Copy of model with aligned coordinates.
        result : AlignmentResult
            Alignment result containing transformation, score, and optionally
            correlation if use_correlation_scoring=True.
        """
        if model is None:
            model = self.model

        if n_vectors is None:
            n_vectors = self.n_vectors

        # When using correlation scoring, we need more candidates to ensure
        # the correct solution is among them
        if use_correlation_scoring and n_refine < 50:
            if self.verbose > 0:
                print(
                    f"Note: Increasing n_refine from {n_refine} to 50 for correlation scoring"
                )
            n_refine = 50

        # Use model without waters for alignment scoring
        model_no_water = model.select("not resname HOH")

        # Get coordinates from model without waters
        xyz_cartesian = model_no_water.xyz().detach().clone()
        device = xyz_cartesian.device
        dtype = xyz_cartesian.dtype

        # Create CA atom mask for efficient clash scoring
        clash_atom_mask = AtomSampler.from_model(model_no_water, mode="auto")

        # Center coordinates for better optimization
        centroid = xyz_cartesian.mean(dim=0)
        xyz_centered = xyz_cartesian - centroid

        if self.verbose > 0:
            n_clash_atoms = clash_atom_mask.sum().item()
            print("Starting Patterson alignment (three-stage search)...")
            print(f"  Stage 1: {n_vectors_coarse:,} vectors for coarse screening")
            print(f"  Stage 2: {n_vectors_fine:,} vectors for local search")
            print(f"  Stage 3: {n_vectors:,} vectors for LBFGS refinement")
            print(f"  Clash weight: {clash_weight}, radius: {clash_radius}A")
            print(f"  Using {n_clash_atoms} atoms for clash detection")
            if use_correlation_scoring:
                print("  Correlation scoring enabled")
            if use_symmetry:
                print("  Using point group symmetry for grid reduction")

        # Stages 1-2: Three-stage grid search
        top_quaternions, top_scores = self._grid_search_rotation(
            xyz_centered,
            idx1=None,  # Will be created internally
            idx2=None,
            angular_step=grid_angular_step,
            n_top=n_refine,
            n_vectors_coarse=n_vectors_coarse,
            n_vectors_fine=n_vectors_fine,
            n_top_coarse=n_top_coarse,
            n_local_samples=n_local_samples,
            local_angular_delta=local_angular_delta,
            use_symmetry=use_symmetry,
            seed=seed,
        )

        # Stage 3: Refine top candidates with LBFGS
        if self.verbose > 0:
            print(f"\n  Stage 3: Refining top {n_refine} candidates with LBFGS")

        # Create sampler for LBFGS refinement (separate from grid search)
        if seed is not None:
            sampler_refine = VectorSampler(
                model_no_water, weighting=self.weighting, seed=seed + 2
            )
        else:
            sampler_refine = VectorSampler(model_no_water, weighting=self.weighting)
        idx1, idx2 = sampler_refine.sample(n_vectors)

        # Collect all refined candidates
        candidates = []

        for i, q in enumerate(top_quaternions):
            # Convert quaternion to rotation matrix for initial guess
            R_init = self._quaternion_to_matrix(q)

            # Start with zero fractional translation (will be optimized)
            t_init_frac = torch.zeros(3, device=device, dtype=dtype)

            # Run optimization
            R_opt, t_opt, patterson, clash, converged = self._optimize_single_start(
                xyz_centered,
                idx1,
                idx2,
                R_init,
                t_init_frac,
                max_iter=max_iter,
                clash_weight=clash_weight,
                clash_radius=clash_radius,
                clash_atom_mask=clash_atom_mask,
            )

            # Adjust translation to account for centering
            t_final = t_opt - R_opt @ centroid + centroid

            candidates.append(
                {
                    "R": R_opt,
                    "t": t_final,
                    "patterson": patterson,
                    "clash": clash,
                    "converged": converged,
                    "grid_score": top_scores[i].item(),
                }
            )

            if self.verbose > 1:
                status = "converged" if converged else "not converged"
                print(
                    f"  Refine {i + 1}/{n_refine}: "
                    f"grid={top_scores[i].item():.4f} -> refined={patterson:.4f}, "
                    f"clash={clash:.6f} ({status})"
                )

        # Stage 3: Select best candidate
        if use_correlation_scoring:
            # Score all candidates with correlation (scale-invariant)
            if self.verbose > 0:
                print(f"\nStage 3: Correlation scoring of {len(candidates)} candidates")

            for i, cand in enumerate(candidates):
                transform = RigidTransform.from_matrix(
                    cand["R"].detach(), cand["t"].detach()
                )
                try:
                    correlation = self.compute_correlation_for_transform(
                        transform, model
                    )
                    cand["correlation"] = correlation
                except Exception as e:
                    if self.verbose > 0:
                        print(
                            f"  Warning: Correlation computation failed for candidate {i+1}: {e}"
                        )
                    cand["correlation"] = float("-inf")

                if self.verbose > 1:
                    print(f"  Candidate {i + 1}: correlation={cand['correlation']:.4f}")

            # Select by highest correlation
            best_cand = max(candidates, key=lambda c: c["correlation"])
            if self.verbose > 0:
                print(
                    f"\nBest by correlation: corr={best_cand['correlation']:.4f}, "
                    f"patterson={best_cand['patterson']:.4f}"
                )
        else:
            # Select by Patterson score (original behavior)
            best_cand = max(
                candidates, key=lambda c: c["patterson"] - clash_weight * c["clash"]
            )
            best_cand["correlation"] = None
            if self.verbose > 0:
                print(
                    f"\nBest by Patterson: score={best_cand['patterson']:.4f}, "
                    f"clash={best_cand['clash']:.6f}"
                )

        # Create result with RigidTransform
        transform = RigidTransform.from_matrix(
            best_cand["R"].detach(), best_cand["t"].detach()
        )
        result = AlignmentResult(
            transform=transform,
            score=best_cand["patterson"],
            n_starts=n_refine,
            converged=best_cand["converged"],
            correlation=best_cand["correlation"],
        )

        # Apply transformation to the FULL model (including waters)
        aligned_model = model.copy()
        full_xyz = model.xyz().detach().clone()
        aligned_coords = result.apply(full_xyz)

        aligned_model.xyz[:] = aligned_coords

        return aligned_model, result

    def generate_rotation_grid(
        self,
        angular_step: float = 10.0,
    ) -> torch.Tensor:
        """
        Generate a uniform grid of rotations on SO(3).

        Uses Euler angle sampling with specified angular step size.

        Parameters
        ----------
        angular_step : float
            Angular step size in degrees. Default is 10.0.
        device : str, optional
            Device for output tensors.
        dtype : torch.dtype, optional
            Data type for output tensors.

        Returns
        -------
        torch.Tensor
            Rotation matrices with shape (N, 3, 3).
        """
        from .sampling import get_rotation_sampling_range
        from .transform import sample_angles, rotation_matrix_from_euler

        max_angles = get_rotation_sampling_range(self.symmetry.matrices)
        euler_angles = sample_angles(
            angular_step * (3.141592653589793 / 180.0), max_angles
        )
        rotations = rotation_matrix_from_euler(euler_angles)

        return rotations

    def score_rotation_batch(
        self,
        xyz_cartesian: torch.Tensor,
        rotations: torch.Tensor,
        idx1: torch.Tensor,
        idx2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score multiple rotations efficiently (no gradient computation).

        Parameters
        ----------
        xyz_cartesian : torch.Tensor
            Cartesian coordinates (N, 3).
        rotations : torch.Tensor
            Rotation matrices (M, 3, 3).
        idx1, idx2 : torch.Tensor
            Atom pair indices.

        Returns
        -------
        torch.Tensor
            Scores for each rotation (M,).
        """
        n_rotations = rotations.shape[0]
        scores = torch.zeros(
            n_rotations, device=xyz_cartesian.device, dtype=xyz_cartesian.dtype
        )

        with torch.no_grad():
            for i in range(n_rotations):
                R = rotations[i]
                xyz_rotated = xyz_cartesian @ R.T
                xyz_frac = cartesian_to_fractional_torch(xyz_rotated, self.cell)
                vecs = xyz_frac[idx1] - xyz_frac[idx2]
                scores[i] = self.interpolate_patterson(vecs).mean()

        return scores

    def align_grid_search(
        self,
        model: Optional[Model] = None,
        angular_step: float = 10.0,
        n_refine: int = 5,
        n_vectors: Optional[int] = None,
        max_iter: int = 100,
        seed: Optional[int] = None,
    ) -> Tuple[Model, AlignmentResult]:
        """
        Align a model using systematic rotation grid search.

        This method performs:
        1. Coarse grid search over all rotations
        2. Refinement of top candidates with gradient descent

        Parameters
        ----------
        model : Model, optional
            Model to align. If None, uses self.model.
        angular_step : float
            Angular step for grid search in degrees. Default is 10.0.
        n_refine : int
            Number of top rotations to refine. Default is 5.
        n_vectors : int, optional
            Number of atom pairs for scoring.
        max_iter : int
            Maximum iterations for refinement.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        aligned_model : Model
            Copy of model with aligned coordinates.
        result : AlignmentResult
            Alignment result containing transformation and score.
        """
        if model is None:
            model = self.model

        if n_vectors is None:
            n_vectors = self.n_vectors

        # Use model without waters for alignment scoring
        model_no_water = model.select("not resname HOH")
        xyz_cartesian = model_no_water.xyz().detach().clone()
        device = xyz_cartesian.device
        dtype = xyz_cartesian.dtype

        # Create sampler
        sampler = VectorSampler(model_no_water, weighting=self.weighting, seed=seed)
        idx1, idx2 = sampler.sample(n_vectors)

        # Generate rotation grid
        if self.verbose > 0:
            print(f"Generating rotation grid with {angular_step}° step...")
        rotations = self.generate_rotation_grid(
            angular_step, device=str(device), dtype=dtype
        )
        n_rotations = rotations.shape[0]
        if self.verbose > 0:
            print(f"  Generated {n_rotations} rotations")

        # Score all rotations (coarse search)
        if self.verbose > 0:
            print("Scoring rotations...")
        scores = self.score_rotation_batch(xyz_cartesian, rotations, idx1, idx2)

        # Get top candidates
        top_indices = torch.argsort(scores, descending=True)[:n_refine]
        top_scores = scores[top_indices]

        if self.verbose > 0:
            print(f"Top {n_refine} coarse scores: {top_scores.tolist()}")

        # Refine top candidates
        if self.verbose > 0:
            print("Refining top candidates...")

        # Center coordinates for refinement
        centroid = xyz_cartesian.mean(dim=0)
        xyz_centered = xyz_cartesian - centroid

        best_score = float("-inf")
        best_R = torch.eye(3, device=device, dtype=dtype)
        best_t = torch.zeros(3, device=device, dtype=dtype)
        best_converged = False

        for i, idx in enumerate(top_indices):
            R_init = rotations[idx]
            t_init_frac = torch.zeros(3, device=device, dtype=dtype)

            R_opt, t_opt, score, converged = self._optimize_single_start(
                xyz_centered, idx1, idx2, R_init, t_init_frac, max_iter=max_iter
            )

            if self.verbose > 1:
                print(
                    f"  Refined {i+1}/{n_refine}: coarse={top_scores[i]:.4f} -> refined={score:.4f}"
                )

            if score > best_score:
                best_score = score
                best_R = R_opt
                best_t = t_opt - best_R @ centroid + centroid
                best_converged = converged

        if self.verbose > 0:
            print(f"Best alignment: score={best_score:.4f}")

        # Create result with RigidTransform
        transform = RigidTransform.from_matrix(best_R.detach(), best_t.detach())
        result = AlignmentResult(
            transform=transform,
            score=best_score,
            n_starts=n_refine,
            converged=best_converged,
        )

        # Apply transformation to full model
        aligned_model = model.copy()
        full_xyz = model.xyz().detach().clone()
        aligned_coords = result.apply(full_xyz)
        aligned_model.xyz[:] = aligned_coords

        return aligned_model, result

    def get_random_orientations(self, n_angles) -> torch.Tensor:
        """
        Generate a random rotation matrix.

        Parameters
        ----------
        device : str, optional
            Device for output tensor.
        dtype : torch.dtype, optional
            Data type for output tensor.

        Returns
        -------
        torch.Tensor
            Random rotation matrix (3, 3).
        """

        from .sampling import get_rotation_sampling_range
        from .transform import rotation_matrix_from_euler

        max_angles = get_rotation_sampling_range(self.symmetry.matrices)
        euler_angles = torch.rand(n_angles, 3, device=self.device) * torch.tensor(
            max_angles, device=self.device
        ).unsqueeze(0)
        R = rotation_matrix_from_euler(euler_angles)
        return R
