"""
Atom pair sampling for efficient Patterson vector generation.

Supports weighted sampling to prioritize informative pairs
(heavy atoms, close distances).
"""

from typing import Optional, Tuple

import numpy as np
import torch


class VectorSampler:
    """
    Samples atom pairs for Patterson vector matching.

    Supports weighted sampling to prioritize informative pairs
    (heavy atoms via Z-weighting). Samples pairs from the asymmetric
    unit (ASU) only - symmetry is already encoded in the Patterson map.

    Parameters
    ----------
    model : Model
        TorchRef Model object. The caller is responsible for filtering
        atoms (e.g., excluding waters) before passing to this class.
    weighting : str, optional
        Weighting scheme: 'uniform' or 'Z2' (weight by atomic number squared).
        Default is 'Z2'.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Attributes
    ----------
    model : Model
        The model used for sampling.
    n_atoms : int
        Number of atoms in the model.
    weighting : str
        Weighting scheme used.
    weights : torch.Tensor
        Sampling weights for each atom (n_atoms,).
    rng : torch.Generator
        Random number generator.
    """

    def __init__(self, model, weighting: str = "Z2", seed: int = None):
        """
        Initialize the VectorSampler.

        Parameters
        ----------
        model : Model
            TorchRef Model object. The caller is responsible for filtering
            atoms (e.g., excluding waters) before passing to this class.
        weighting : str
            Weighting scheme for sampling.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.model = model
        self.n_atoms = len(model.pdb)
        self.weighting = weighting
        self.rng = (
            torch.Generator().manual_seed(seed)
            if seed is not None
            else torch.Generator()
        )
        self.weights = self._compute_weights()

    def _compute_weights(
        self,
    ) -> torch.Tensor:
        """
        Compute sampling probability for each atom based on atomic number and B-factor.

        Weights are computed as: Z^2 / B (for Z2 weighting) or 1/B (for uniform).
        Atoms with lower B-factors (more ordered) get higher weights since they
        contribute more signal to the Patterson map.

        Returns
        -------
        torch.Tensor
            Weight for each atom with shape (n_atoms,).
        """
        from torchref.utils.pse import PERIODIC_TABLE

        elements = self.model.pdb.element.values

        Zs = torch.tensor(
            [PERIODIC_TABLE[el]["number"] for el in elements],
            dtype=torch.float32,
            device=self.model.device,
        )

        # Get B-factors and compute reciprocal weights
        # Use 1/B so atoms with lower B-factors get higher weights
        B_factors = torch.tensor(
            self.model.pdb["tempfactor"].values,
            dtype=torch.float32,
            device=self.model.device,
        )
        # Clamp B-factors to avoid division by zero or very small values
        B_factors = torch.clamp(B_factors, min=1.0)
        B_weights = 1.0 / B_factors

        if self.weighting == "Z2":
            weights = Zs**2 * B_weights
        else:  # uniform
            weights = B_weights

        weights = weights / weights.sum()  # Normalize to probabilities
        return weights

    def sample(
        self, n_vectors: int, weights: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample atom pairs according to weighting scheme.

        Parameters
        ----------
        n_vectors : int
            Number of atom pairs to sample.
        weights : torch.Tensor, optional
            Override weights for sampling. If None, uses self.weights.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Two tensors of shape (n_vectors,) containing
            the indices of the sampled atom pairs.
        """
        w = weights if weights is not None else self.weights

        # Sample first indices according to weights
        idx1 = torch.multinomial(w, n_vectors, replacement=True, generator=self.rng)

        # Sample second indices according to weights
        idx2 = torch.multinomial(w, n_vectors, replacement=True, generator=self.rng)

        # Redraw idx2 where it equals idx1
        same_mask = idx1 == idx2
        max_attempts = 100  # Prevent infinite loop
        attempt = 0
        while same_mask.any() and attempt < max_attempts:
            n_resample = same_mask.sum().item()
            idx2[same_mask] = torch.multinomial(
                w, n_resample, replacement=True, generator=self.rng
            )
            same_mask = idx1 == idx2
            attempt += 1

        return idx1, idx2


def get_rotation_sampling_range(
    rotation_matrices: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    Determine rotation angle sampling ranges given point group symmetry operations.

    Given the rotation matrices from a spacegroup's point group, this function
    computes the asymmetric unit in rotation space (SO(3)) and returns the
    maximum Euler angles (alpha, beta, gamma) needed to cover the asymmetric unit.

    The function uses ZYZ Euler angle convention where:
    - alpha: rotation about Z axis, range [0, 2*pi)
    - beta: rotation about Y axis, range [0, pi]
    - gamma: rotation about Z axis, range [0, 2*pi)

    For crystals, the point group symmetry reduces the search space:
    - Triclinic (1): Full SO(3) - (2*pi, pi, 2*pi)
    - Monoclinic (2): Half of SO(3) - (2*pi, pi, pi)
    - Orthorhombic (222): 1/4 of SO(3) - (pi, pi, pi)
    - Tetragonal (4, 422): 1/8 or 1/16 of SO(3)
    - Trigonal (3, 32): 1/6 or 1/12 of SO(3)
    - Hexagonal (6, 622): 1/12 or 1/24 of SO(3)
    - Cubic (23, 432): 1/12 or 1/24 of SO(3)

    Parameters
    ----------
    rotation_matrices : torch.Tensor
        Point group rotation matrices with shape (N, 3, 3), where N is the
        number of symmetry operations. These should be the pure rotation
        parts of the spacegroup operations (no translations).

    Returns
    -------
    Tuple[float, float, float]
        Maximum values for (alpha, beta, gamma) Euler angles in radians.
        These define the asymmetric unit in rotation space that needs to
        be sampled during molecular replacement searches.

    Examples
    --------
    >>> from torchref.symmetry import Symmetry
    >>> S = Symmetry('P212121')  # Orthorhombic
    >>> ranges = get_rotation_sampling_range(S.matrices)
    >>> print(f"alpha: {ranges[0]:.4f}, beta: {ranges[1]:.4f}, gamma: {ranges[2]:.4f}")
    alpha: 3.1416, beta: 3.1416, gamma: 3.1416

    >>> S = Symmetry('P1')  # Triclinic - need full SO(3)
    >>> ranges = get_rotation_sampling_range(S.matrices)
    >>> print(f"alpha: {ranges[0]:.4f}, beta: {ranges[1]:.4f}, gamma: {ranges[2]:.4f}")
    alpha: 6.2832, beta: 3.1416, gamma: 6.2832
    """
    n_ops = rotation_matrices.shape[0]

    # Convert to numpy for analysis
    if isinstance(rotation_matrices, torch.Tensor):
        R_ops = rotation_matrices.detach().cpu().numpy()
    else:
        R_ops = np.array(rotation_matrices)

    # Analyze the point group to determine fold symmetries
    # We look for rotation axes and their orders

    # Default: full SO(3) coverage
    alpha_max = 2 * np.pi
    beta_max = np.pi
    gamma_max = 2 * np.pi

    # Identity only (P1) - need full search
    if n_ops == 1:
        return (alpha_max, beta_max, gamma_max)

    # Analyze rotation axes and angles
    axes_and_angles = []
    for R in R_ops:
        # Skip identity
        trace = np.trace(R)
        if np.abs(trace - 3.0) < 1e-6:
            continue

        # Rotation angle from trace: trace = 1 + 2*cos(theta)
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.arccos(cos_theta)

        if angle < 1e-6:
            continue

        # Get rotation axis from antisymmetric part of R
        # axis is proportional to (R - R^T)
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])
        norm = np.linalg.norm(axis)
        if norm > 1e-6:
            axis = axis / norm
        else:
            # 180-degree rotation - get axis from R + I
            # For 180° rotation, axis is eigenvector with eigenvalue 1
            eigvals, eigvecs = np.linalg.eig(R)
            idx = np.argmin(np.abs(eigvals - 1.0))
            axis = np.real(eigvecs[:, idx])
            axis = axis / np.linalg.norm(axis)

        axes_and_angles.append((axis, angle))

    # Determine fold along principal axes
    z_axis = np.array([0, 0, 1])
    y_axis = np.array([0, 1, 0])
    x_axis = np.array([1, 0, 0])

    z_fold = 1
    y_fold = 1
    x_fold = 1

    for axis, angle in axes_and_angles:
        # Check if axis is along z
        if np.abs(np.abs(np.dot(axis, z_axis)) - 1.0) < 0.1:
            fold = int(round(2 * np.pi / angle))
            z_fold = max(z_fold, fold)
        # Check if axis is along y
        elif np.abs(np.abs(np.dot(axis, y_axis)) - 1.0) < 0.1:
            fold = int(round(2 * np.pi / angle))
            y_fold = max(y_fold, fold)
        # Check if axis is along x
        elif np.abs(np.abs(np.dot(axis, x_axis)) - 1.0) < 0.1:
            fold = int(round(2 * np.pi / angle))
            x_fold = max(x_fold, fold)

    # Also check for 2-fold along diagonal (orthorhombic has 3 perpendicular 2-folds)
    has_three_twofolds = False
    twofold_count = 0
    for axis, angle in axes_and_angles:
        fold = int(round(2 * np.pi / angle))
        if fold == 2:
            twofold_count += 1
    if twofold_count >= 3:
        has_three_twofolds = True

    # Determine sampling ranges based on symmetry analysis
    # For ZYZ Euler angles:
    # - z_fold reduces alpha range
    # - 2-fold perpendicular to z reduces beta range to [0, pi/2] in some cases
    # - Combined symmetry reduces gamma range

    # Alpha reduction based on z-axis fold
    if z_fold > 1:
        alpha_max = 2 * np.pi / z_fold

    # Check for 2-fold perpendicular to z-axis (reduces gamma)
    has_perp_twofold = False
    for axis, angle in axes_and_angles:
        fold = int(round(2 * np.pi / angle))
        if fold == 2:
            # Check if axis is perpendicular to z
            if np.abs(np.dot(axis, z_axis)) < 0.1:
                has_perp_twofold = True
                break

    if has_perp_twofold:
        gamma_max = np.pi

    # For point groups with higher symmetry, apply additional reductions
    # Based on number of operations (proxy for point group order)
    if n_ops >= 24:
        # Cubic or high-symmetry hexagonal
        alpha_max = min(alpha_max, np.pi / 2)
        gamma_max = min(gamma_max, np.pi / 2)
    elif n_ops >= 12:
        # Hexagonal 622, Cubic 23, etc.
        alpha_max = min(alpha_max, np.pi)
        gamma_max = min(gamma_max, np.pi)
    elif n_ops >= 8:
        # Tetragonal 422
        gamma_max = min(gamma_max, np.pi)
    elif has_three_twofolds:
        # Orthorhombic 222
        alpha_max = np.pi
        gamma_max = np.pi

    return (alpha_max, beta_max, gamma_max)
