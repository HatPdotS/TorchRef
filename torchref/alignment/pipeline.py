"""
Molecular replacement pipeline integrating rotation, translation, and refinement.

This module provides a unified pipeline for molecular replacement that chains:
1. Fast rotation function (ball transform)
2. FFT-based translation search
3. Clash filtering
4. Rigid body refinement

The pipeline supports early stopping when a good solution is found.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import torch

from torchref.config import get_default_device, get_float_dtype

from .ball_transform import (
    ball_rotation_search_torch,
    rotation_matrix_from_euler_zyz,
)
from .translation import fft_translation_search_torch, TranslationPeak
from .rigid_body import RigidBodyRefinement, RigidBodyResult
from .clashscore import ClashScoreCalculator, AtomSampler

if TYPE_CHECKING:
    from torchref.model import ModelFT
    from torchref.io.datasets import ReflectionData


def rotation_angular_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute angular distance between two rotation matrices in degrees.

    The angular distance is the angle of the rotation R2 @ R1.T.

    Parameters
    ----------
    R1, R2 : np.ndarray
        3x3 rotation matrices.

    Returns
    -------
    float
        Angular distance in degrees.
    """
    R_diff = R2 @ R1.T
    # Clamp trace to valid range for arccos
    trace = np.trace(R_diff)
    trace = np.clip(trace, -1.0, 3.0)
    angle_rad = np.arccos((trace - 1.0) / 2.0)
    return np.degrees(angle_rad)


def euler_angular_distance(
    euler1: Tuple[float, float, float],
    euler2: Tuple[float, float, float],
) -> float:
    """
    Compute angular distance between two ZYZ Euler angle sets.

    Parameters
    ----------
    euler1, euler2 : tuple
        (alpha, beta, gamma) Euler angles in radians.

    Returns
    -------
    float
        Angular distance in degrees.
    """
    R1 = rotation_matrix_from_euler_zyz(euler1[0], euler1[1], euler1[2])
    R2 = rotation_matrix_from_euler_zyz(euler2[0], euler2[1], euler2[2])
    return rotation_angular_distance(R1, R2)


def cluster_rotation_peaks(
    peaks: list,
    threshold_deg: float = 6.0,
    symmetry_matrices: Optional[np.ndarray] = None,
) -> list:
    """
    Cluster rotation peaks by angular distance.

    Peaks within threshold_deg of each other are considered the same solution.
    Only the highest-scoring peak from each cluster is kept.

    Parameters
    ----------
    peaks : list
        List of rotation peaks as tuples (alpha, beta, gamma, score, sigma).
    threshold_deg : float
        Angular distance threshold for clustering in degrees.
    symmetry_matrices : np.ndarray, optional
        Point group symmetry matrices (N, 3, 3) to check symmetry equivalents.

    Returns
    -------
    list
        Clustered peaks with one representative per cluster.
    """
    if not peaks:
        return []

    # Sort by sigma (descending) so we keep highest-scoring peaks
    sorted_peaks = sorted(peaks, key=lambda p: p[4], reverse=True)

    clustered = []
    used_rotations = []  # Store rotation matrices of accepted peaks

    for peak in sorted_peaks:
        alpha, beta, gamma, score, sigma = peak
        R = rotation_matrix_from_euler_zyz(alpha, beta, gamma)

        # Check if this rotation is too close to any already accepted rotation
        is_new = True
        for R_used in used_rotations:
            dist = rotation_angular_distance(R, R_used)
            if dist < threshold_deg:
                is_new = False
                break

            # Also check symmetry equivalents if provided
            if symmetry_matrices is not None and is_new:
                for sym_op in symmetry_matrices:
                    R_sym = sym_op @ R
                    dist_sym = rotation_angular_distance(R_sym, R_used)
                    if dist_sym < threshold_deg:
                        is_new = False
                        break
                if not is_new:
                    break

        if is_new:
            clustered.append(peak)
            used_rotations.append(R)

    return clustered


@dataclass
class MRSolution:
    """
    Molecular replacement solution.

    Attributes
    ----------
    rotation : np.ndarray
        ZYZ Euler angles (radians), shape (3,).
    translation : np.ndarray
        Fractional coordinates, shape (3,).
    rotation_score : float
        FRF sigma (Z-score).
    translation_score : float
        Translation function correlation.
    clash_score : float
        Steric clash score (lower is better).
    r_factor : float
        R-factor after refinement.
    refined_rotation : np.ndarray, optional
        Refined Euler angles (radians).
    refined_translation : np.ndarray, optional
        Refined fractional translation.
    """
    rotation: np.ndarray
    translation: np.ndarray
    rotation_score: float
    translation_score: float
    clash_score: float
    r_factor: float
    refined_rotation: Optional[np.ndarray] = None
    refined_translation: Optional[np.ndarray] = None


class MolecularReplacementPipeline:
    """
    Unified MR pipeline: Rotation -> Translation -> Rigid Body Refinement.

    This pipeline integrates the fast rotation function (ball transform),
    FFT-based translation search, clash filtering, and rigid body refinement
    into a single workflow with early stopping.

    Parameters
    ----------
    data : ReflectionData
        Observed reflection data.
    model : ModelFT
        Search model with atomic coordinates.
    device : torch.device, optional
        Computation device. Default is CPU.
    verbose : int
        Verbosity level (0=silent, 1=summary, 2=detailed).

    Examples
    --------
    ::

        from torchref.alignment import MolecularReplacementPipeline
        from torchref.model import ModelFT
        from torchref.io.datasets import ReflectionData

        data = ReflectionData().load_mtz('observed.mtz')
        model = ModelFT().load_pdb('search_model.pdb')
        pipeline = MolecularReplacementPipeline(data, model)
        solutions = pipeline.run(n_rotation_peaks=50, min_tries=3, max_tries=10)
        print(f'Best R: {solutions[0].r_factor:.3f}')
    """

    def __init__(
        self,
        data: "ReflectionData",
        model: "ModelFT",
        device: Optional[torch.device] = None,
        verbose: int = 1,
    ):
        self.data = data
        self.model = model
        self.device = device or get_default_device()
        self.verbose = verbose

        # Lazy caches
        self._clash_calc = None
        self._e_obs = None
        self._e_calc = None
        self._s_vectors = None
        self._mask = None

    def run(
        self,
        n_rotation_peaks: int = 100,
        n_translation_peaks: int = 5,
        min_tries: int = 3,
        max_tries: int = 10,
        rfactor_converged: float = 0.45,
        max_clash_score: float = 100.0,
        d_min: float = 4.0,
        d_max: float = 50.0,
        L: int = 32,
        P: int = 20,
        cluster_threshold_deg: float = 6.0,
    ) -> List[MRSolution]:
        """
        Run full MR pipeline with early stopping.

        Parameters
        ----------
        n_rotation_peaks : int
            Number of rotation peaks to try.
        n_translation_peaks : int
            Number of translation peaks per rotation.
        min_tries : int
            Minimum number of candidates to refine before early stopping.
        max_tries : int
            Maximum number of candidates to refine.
        rfactor_converged : float
            R-factor threshold for convergence (early stopping).
        max_clash_score : float
            Maximum clash score to accept a candidate.
        d_min : float
            High resolution limit for rotation search (Angstroms).
        d_max : float
            Low resolution limit for rotation search (Angstroms).
        L : int
            Angular bandlimit for rotation search.
        P : int
            Radial bandlimit for rotation search.
        cluster_threshold_deg : float
            Angular distance threshold for clustering rotation peaks (degrees).
            Peaks within this angular distance are considered the same solution.
            Default 6.0 degrees matches rigid body refinement convergence radius.

        Returns
        -------
        List[MRSolution]
            Solutions sorted by R-factor. Early stops if converged.
        """
        # Step 1: Rotation search
        if self.verbose:
            print("Step 1: Rotation search...")
        rotation_peaks = self._rotation_search(n_rotation_peaks, d_min, d_max, L, P)

        if not rotation_peaks:
            if self.verbose:
                print("  No rotation peaks found!")
            return []

        # Step 1b: Cluster rotation peaks
        if self.verbose:
            print(f"  Found {len(rotation_peaks)} raw peaks, clustering with {cluster_threshold_deg}° threshold...")

        # Get symmetry matrices for clustering
        sym_matrices = None
        if hasattr(self.model, 'symmetry') and self.model.symmetry is not None:
            sym_matrices = np.array([s.numpy() for s in self.model.symmetry.matrices])

        rotation_peaks = cluster_rotation_peaks(
            rotation_peaks,
            threshold_deg=cluster_threshold_deg,
            symmetry_matrices=sym_matrices,
        )

        if self.verbose:
            print(f"  {len(rotation_peaks)} unique rotation clusters")

        if not rotation_peaks:
            if self.verbose:
                print("  No rotation peaks after clustering!")
            return []

        # Step 2: Translation search for each rotation
        if self.verbose:
            print(f"Step 2: Translation search for {len(rotation_peaks)} rotations...")
        candidates = []
        for i, rot in enumerate(rotation_peaks):
            trans_peaks = self._translation_search(rot, n_translation_peaks)
            for trans in trans_peaks:
                candidates.append((rot, trans))
            if self.verbose > 1 and (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(rotation_peaks)} rotations...")

        if self.verbose:
            print(f"  Generated {len(candidates)} rotation+translation candidates")

        # Step 3: Score and filter by clash
        if self.verbose:
            print("Step 3: Clash filtering...")
        candidates = self._score_and_filter(candidates, max_clash_score)

        if not candidates:
            if self.verbose:
                print("  All candidates rejected by clash filter!")
            return []

        if self.verbose:
            print(f"  {len(candidates)} candidates passed clash filter")

        # Step 4: Rigid body refinement with early stopping
        if self.verbose:
            print(f"Step 4: Refining candidates (min={min_tries}, max={max_tries})...")

        solutions = []
        best_r_factor = float('inf')
        converged = False

        n_to_refine = min(len(candidates), max_tries)
        for i, (rot, trans, clash) in enumerate(candidates[:n_to_refine]):
            if self.verbose > 1:
                print(f"  Refining candidate {i+1}/{n_to_refine}...")

            try:
                result = self._rigid_body_refine(rot, trans)

                solution = MRSolution(
                    rotation=np.array([rot[0], rot[1], rot[2]]),
                    translation=trans.translation,
                    rotation_score=rot[4],  # sigma
                    translation_score=trans.score,
                    clash_score=clash,
                    r_factor=result.final_r_factor,
                    refined_rotation=np.array(result.final_rotation),
                    refined_translation=np.array(result.final_translation_frac),
                )
                solutions.append(solution)

                # Track best
                if result.final_r_factor < best_r_factor:
                    best_r_factor = result.final_r_factor

                # Check early stopping
                n_refined = i + 1
                if n_refined >= min_tries and best_r_factor < rfactor_converged:
                    converged = True
                    if self.verbose:
                        print(f"  Converged! R-factor {best_r_factor:.4f} < {rfactor_converged}")
                    break

            except Exception as e:
                if self.verbose > 1:
                    print(f"  Refinement failed: {e}")
                continue

        if self.verbose:
            status = "converged" if converged else f"completed {len(solutions)}/{n_to_refine}"
            print(f"  Refinement {status}")
            if solutions:
                print(f"  Best R-factor: {best_r_factor:.4f}")

        solutions.sort(key=lambda s: s.r_factor)
        return solutions

    def _rotation_search(
        self,
        n_peaks: int,
        d_min: float,
        d_max: float,
        L: int,
        P: int,
    ) -> list:
        """Run ball rotation search."""
        # Prepare E-values
        E_obs, s_obs = self._get_e_values_obs(d_min, d_max)
        E_calc, s_calc = self._get_e_values_calc(d_min, d_max)

        _, _, peaks = ball_rotation_search_torch(
            E_obs, s_obs, E_calc, s_calc,
            L=L, P=P, d_min=d_min, d_max=d_max, n_peaks=n_peaks,
            verbose=self.verbose > 1,
        )
        return peaks

    def _translation_search(
        self,
        rotation_peak: tuple,
        n_peaks: int,
    ) -> List[TranslationPeak]:
        """Run translation search for a rotation."""
        alpha, beta, gamma, _, _ = rotation_peak

        # Apply rotation to model coordinates
        R = torch.tensor(
            rotation_matrix_from_euler_zyz(alpha, beta, gamma),
            dtype=get_float_dtype(),
            device=self.device,
        )
        xyz = self.model.xyz()
        xyz_centered = xyz - xyz.mean(dim=0)
        xyz_rotated = xyz_centered @ R.T

        # Temporarily update model coordinates and compute F_calc
        original_xyz = self.model.xyz().clone()
        self.model.xyz[:] = xyz_rotated

        try:
            hkl = self.data.hkl
            F_obs = self.data.F
            mask = self.data.get_valid_mask()

            with torch.no_grad():
                F_calc = self.model(hkl)

            # Apply mask
            F_obs_masked = F_obs[mask]
            F_calc_masked = F_calc[mask]
            hkl_masked = hkl[mask]

            _, _, peaks = fft_translation_search_torch(
                F_obs_masked, F_calc_masked, hkl_masked, n_peaks=n_peaks
            )
        finally:
            # Restore original coordinates
            self.model.xyz[:] = original_xyz

        return peaks

    def _score_and_filter(
        self,
        candidates: list,
        max_clash: float,
    ) -> list:
        """Score candidates and filter by clash."""
        if self._clash_calc is None:
            self._clash_calc = ClashScoreCalculator(
                symmetry=self.data.spacegroup,
                default_clash_radius=4.0,
                device=self.device,
            )

        scored = []
        for rot, trans in candidates:
            clash = self._compute_clash(rot, trans)
            if clash <= max_clash:
                scored.append((rot, trans, clash))

        # Sort by combined score (rotation_sigma + trans_sigma - clash_penalty)
        def combined_score(x):
            rot, trans, clash = x
            return rot[4] + trans.sigma - clash / 100.0

        scored.sort(key=combined_score, reverse=True)
        return scored

    def _compute_clash(
        self,
        rotation_peak: tuple,
        trans_peak: TranslationPeak,
    ) -> float:
        """Compute clash score for a solution."""
        alpha, beta, gamma, _, _ = rotation_peak
        R = torch.tensor(
            rotation_matrix_from_euler_zyz(alpha, beta, gamma),
            dtype=get_float_dtype(),
            device=self.device,
        )

        xyz = self.model.xyz()
        xyz_centered = xyz - xyz.mean(dim=0)
        xyz_rotated = xyz_centered @ R.T

        # Apply translation (fractional -> Cartesian)
        trans_frac = torch.tensor(
            trans_peak.translation,
            dtype=get_float_dtype(),
            device=self.device,
        )
        trans_cart = trans_frac @ self.data.cell.fractional_matrix.to(self.device)
        xyz_final = xyz_rotated + trans_cart

        atom_mask = AtomSampler.from_model(self.model, mode='ca_only')
        with torch.no_grad():
            clash = self._clash_calc(
                xyz=xyz_final,
                cell=self.data.cell.data,
                atom_mask=atom_mask,
            ).item()
        return clash

    def _rigid_body_refine(
        self,
        rotation_peak: tuple,
        trans_peak: TranslationPeak,
    ) -> RigidBodyResult:
        """Run rigid body refinement."""
        alpha, beta, gamma, _, _ = rotation_peak

        rb = RigidBodyRefinement(
            model=self.model,
            data=self.data,
            initial_rotation=torch.tensor([alpha, beta, gamma], dtype=get_float_dtype()),
            initial_translation=torch.tensor(
                trans_peak.translation, dtype=get_float_dtype()
            ),
            device=self.device,
            verbose=max(0, self.verbose - 1),
        )
        return rb.refine()

    def _get_e_values_obs(
        self,
        d_min: float,
        d_max: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get E-values for observed data (cached)."""
        if self._e_obs is None:
            from torchref.base.alignment import F_squared_to_E_values

            F_obs = self.data.F
            mask = self.data.get_valid_mask()
            F_obs_masked = F_obs[mask]

            F2 = (F_obs_masked ** 2).to(torch.float64)
            s = self._get_s_vectors()[mask]

            E_values, E_squared, _ = F_squared_to_E_values(
                F2, s, n_shells=20, d_min=d_min, d_max=d_max
            )
            self._e_obs = E_squared  # Use E² for correlation
            self._s_obs = s
            self._mask = mask  # Cache mask for later use
        return self._e_obs, self._s_obs

    def _get_e_values_calc(
        self,
        d_min: float,
        d_max: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get E-values for calculated data (cached)."""
        if self._e_calc is None:
            from torchref.base.alignment import F_squared_to_E_values

            hkl = self.data.hkl
            mask = self.data.get_valid_mask()

            with torch.no_grad():
                F_calc = self.model(hkl).abs()

            F_calc_masked = F_calc[mask]
            F2 = (F_calc_masked ** 2).to(torch.float64)
            s = self._get_s_vectors()[mask]

            E_values, E_squared, _ = F_squared_to_E_values(
                F2, s, n_shells=20, d_min=d_min, d_max=d_max
            )
            self._e_calc = E_squared
            self._s_calc = s
        return self._e_calc, self._s_calc

    def _get_s_vectors(self) -> torch.Tensor:
        """Get scattering vectors (cached)."""
        if self._s_vectors is None:
            from torchref.base import reciprocal_basis_matrix
            rec_basis = reciprocal_basis_matrix(self.model.cell)
            self._s_vectors = self.data.hkl.to(torch.float64) @ rec_basis.to(torch.float64)
        return self._s_vectors

    def clear_cache(self):
        """Clear cached E-values and s-vectors."""
        self._e_obs = None
        self._e_calc = None
        self._s_vectors = None
        self._s_obs = None
        self._s_calc = None
        self._mask = None
