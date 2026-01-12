"""
Main Patterson alignment class for TorchRef.

Aligns predicted structures to observed diffraction data via Patterson matching.
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import torch
import numpy as np

from torchref.model.model import Model
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.symmetrie.symmetrie import Symmetry
from torchref.math_functions.math_torch import (
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_axis_angle,
    random_rotation_uniform,
    cartesian_to_fractional_torch,
    fractional_to_cartesian_torch,
    trilinear_interpolate,
)

from .sampling import VectorSampler


@dataclass
class AlignmentResult:
    """
    Result of Patterson alignment.

    Attributes
    ----------
    rotation : torch.Tensor
        Rotation matrix (3, 3) in Cartesian space.
    translation : torch.Tensor
        Translation vector (3,) in Cartesian coordinates (Angstroms).
    score : float
        Patterson correlation score (higher = better match).
    n_starts : int
        Number of random starts used.
    converged : bool
        Whether optimization converged.
    """
    rotation: torch.Tensor
    translation: torch.Tensor
    score: float
    n_starts: int
    converged: bool

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
        return coords @ self.rotation.T + self.translation

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
    following the TorchRef API conventions.

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

    Examples
    --------
    >>> from torchref.model import Model
    >>> from torchref.io.datasets.reflection_data import ReflectionData
    >>> from torchref.alignment import PattersonAligner
    >>>
    >>> # Load data and model
    >>> data = ReflectionData(verbose=1).load_mtz('observed.mtz')
    >>> model = Model(verbose=1).load_pdb('predicted.pdb')
    >>>
    >>> # Create aligner (precomputes Patterson map)
    >>> aligner = PattersonAligner(data, model)
    >>>
    >>> # Align model to data
    >>> aligned_model, result = aligner.align(n_starts=20)
    >>> print(f"Score: {result.score:.4f}")
    >>>
    >>> # Save aligned structure
    >>> aligned_model.write_pdb('aligned.pdb')
    """

    def __init__(
        self,
        data: ReflectionData,
        model: Model,
        n_vectors: int = 10000,
        weighting: str = 'Z2',
        verbose: int = 1
    ):
        # Store references
        self.data = data
        self.verbose = verbose
        self.device = torch.device(data.device) if isinstance(data.device, str) else data.device
        self.n_vectors = n_vectors
        self.weighting = weighting

        # Get cell parameters (use data.cell as source of truth)
        self.cell = data.cell.clone()

        # Store symmetry from model
        self.symmetry = model.symmetry

        # Store reference to model (filtering done in align() method)
        self.model = model

        # Precompute Patterson map
        self.patterson = data.calc_patterson()


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
        vecs = fractional_vectors.to(device=self.patterson.device, dtype=self.patterson.dtype)

        # Use trilinear_interpolate with periodic wrapping
        # This function handles wrapping to [0, 1) internally
        return trilinear_interpolate(self.patterson, vecs, mode='wrap')

    def evaluate_vectors_on_coords(
        self,
        idx1: torch.Tensor,
        idx2: torch.Tensor,
        xyz_fractional: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate Patterson score for atom pair vectors on fractional coordinates.

        The Patterson map already contains all symmetry information, so we
        compute vectors directly from the ASU coordinates without symmetry
        expansion.

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
            Mean Patterson score for all vectors.
        """
        # Compute difference vectors directly from ASU coordinates
        # The Patterson map already encodes symmetry, so no expansion needed
        vecs = xyz_fractional[idx1] - xyz_fractional[idx2]

        # Interpolate Patterson map at vector positions
        scores = self.interpolate_patterson(vecs)

        return scores.mean()

    def score_transformation(
        self,
        xyz_cartesian: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        idx1: torch.Tensor,
        idx2: torch.Tensor
    ) -> torch.Tensor:
        """
        Score a rotation and translation by Patterson vector matching.

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

        Returns
        -------
        torch.Tensor
            Patterson score (scalar tensor for gradient computation).
        """
        # Apply rotation and translation
        xyz_transformed = xyz_cartesian @ rotation.T + translation

        # Convert to fractional coordinates
        xyz_fractional = cartesian_to_fractional_torch(xyz_transformed, self.cell)

        # Evaluate Patterson score
        return self.evaluate_vectors_on_coords(idx1, idx2, xyz_fractional)

    def _optimize_single_start(
        self,
        xyz_cartesian: torch.Tensor,
        idx1: torch.Tensor,
        idx2: torch.Tensor,
        R_init: torch.Tensor,
        t_init: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor, float, bool]:
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
        t_init : torch.Tensor
            Initial translation vector (3,).
        max_iter : int
            Maximum LBFGS iterations.
        lr : float
            Learning rate for LBFGS.

        Returns
        -------
        R_opt : torch.Tensor
            Optimized rotation matrix.
        t_opt : torch.Tensor
            Optimized translation vector.
        score : float
            Final Patterson score.
        converged : bool
            Whether optimization converged.
        """
        device = xyz_cartesian.device
        dtype = xyz_cartesian.dtype

        # Convert rotation to axis-angle parameterization
        rot_params = rotation_matrix_to_axis_angle(R_init).to(dtype=dtype, device=device)
        rot_params = rot_params.clone().requires_grad_(True)

        # Translation parameters
        trans_params = t_init.clone().to(dtype=dtype, device=device).requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [rot_params, trans_params],
            max_iter=max_iter,
            lr=lr,
            line_search_fn='strong_wolfe'
        )

        final_score = 0.0

        def closure():
            nonlocal final_score
            optimizer.zero_grad()

            # Convert axis-angle to rotation matrix
            R = axis_angle_to_rotation_matrix(rot_params)

            # Compute score (we want to maximize, so negate for minimization)
            score = self.score_transformation(
                xyz_cartesian, R, trans_params, idx1, idx2
            )

            loss = -score
            loss.backward()

            final_score = score.item()
            return loss

        try:
            optimizer.step(closure)
            converged = True
        except RuntimeError as e:
            if self.verbose > 1:
                print(f"    Optimization warning: {e}")
            converged = False

        # Get final rotation matrix
        with torch.no_grad():
            R_final = axis_angle_to_rotation_matrix(rot_params)
            t_final = trans_params.clone()

        return R_final, t_final, final_score, converged

    def align(
        self,
        model: Optional[Model] = None,
        n_starts: int = 10,
        n_vectors: Optional[int] = None,
        max_iter: int = 50,
        seed: Optional[int] = None
    ) -> Tuple[Model, AlignmentResult]:
        """
        Align a model to the diffraction data via Patterson matching.

        Uses multi-start optimization with random initial rotations and
        translations to find the best alignment.

        Parameters
        ----------
        model : Model, optional
            Model to align. If None, uses self.model.
        n_starts : int, optional
            Number of random starting orientations. Default is 10.
        n_vectors : int, optional
            Number of atom pairs for scoring. Default is self.n_vectors.
        max_iter : int, optional
            Maximum iterations per optimization. Default is 50.
        seed : int, optional
            Random seed for reproducibility. Default is None.

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
        # (VectorSampler also excludes waters internally)
        model_no_water = model.select('not resname HOH')

        # Get coordinates from model without waters
        xyz_cartesian = model_no_water.xyz().detach().clone()
        device = xyz_cartesian.device
        dtype = xyz_cartesian.dtype

        # Create sampler using the same filtered model
        sampler = VectorSampler(model_no_water, weighting=self.weighting, seed=seed)
        idx1, idx2 = sampler.sample(n_vectors)

        # Center coordinates for better optimization
        centroid = xyz_cartesian.mean(dim=0)
        xyz_centered = xyz_cartesian - centroid

        # Best result tracking
        best_score = float('-inf')
        best_R = torch.eye(3, device=device, dtype=dtype)
        best_t = torch.zeros(3, device=device, dtype=dtype)
        best_converged = False

        if self.verbose > 0:
            print(f"Starting Patterson alignment with {n_starts} random starts...")
            print(f"  Using {n_vectors} atom pair vectors")

        # Random generator for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        for start_idx in range(n_starts):
            # Random initial rotation (uniform over SO(3))
            R_init = random_rotation_uniform(1, device=str(device), dtype=dtype)

            # Random initial translation within unit cell
            t_init = torch.rand(3, device=device, dtype=dtype) * self.cell[:3].to(device=device, dtype=dtype)

            # Run optimization
            R_opt, t_opt, score, converged = self._optimize_single_start(
                xyz_centered, idx1, idx2, R_init, t_init, max_iter=max_iter
            )

            if self.verbose > 1:
                status = "converged" if converged else "not converged"
                print(f"  Start {start_idx + 1}/{n_starts}: score={score:.4f} ({status})")

            if score > best_score:
                best_score = score
                best_R = R_opt
                # Adjust translation to account for centering
                best_t = t_opt - best_R @ centroid + centroid
                best_converged = converged

        if self.verbose > 0:
            print(f"Best alignment: score={best_score:.4f}")

        # Create result
        result = AlignmentResult(
            rotation=best_R.detach(),
            translation=best_t.detach(),
            score=best_score,
            n_starts=n_starts,
            converged=best_converged
        )

        # Apply transformation to the FULL model (including waters)
        # The transformation found on non-water atoms applies to all atoms
        aligned_model = model.copy()
        full_xyz = model.xyz().detach().clone()
        aligned_coords = result.apply(full_xyz)

        aligned_model.xyz[:] = aligned_coords

        return aligned_model, result


def clashscore(xyz_fractional, symm):
    """
    Compute simple clashscore for given coordinates and symmetry.

    Parameters
    ----------
    xyz : torch.Tensor
        Cartesian coordinates (N, 3).
    symm : Symmetry
        Symmetry operations object.

    Returns
    -------
    float
        Clashscore (number of clashes per 1000 atoms).
    """

    
    
    pass