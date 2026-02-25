import torch
from typing import TYPE_CHECKING, Dict

from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

from .base import ADPTarget

if TYPE_CHECKING:
    from torchref.model.model import Model


class ADPLocalityTarget(ADPTarget):
    """
    Proximity-based ADP restraint using K nearest neighbors.

    ADPs of nearby atoms should be similar because:

    1. Core residues (buried) tend to have low ADP
    2. Surface/loop residues tend to have high ADP
    3. Disorder propagates through space

    This restraint finds the K nearest neighbors for each atom and computes
    a weighted MSE on log(B) differences. The weight decays exponentially
    with distance, based on a correlation length parameter::

        w_ij = exp(-d_ij / xi)

    Where xi (correlation length) controls how quickly the restraint weakens
    with distance. Typical values: 4-8 Å for proteins.

    The loss is weighted MSE in log-space::

        loss = scale * mean_ij [w_ij * (log(B_i) - log(B_j))^2]

    This is simpler than NLL - no fake probabilistic sigma. The exponential
    decay has a clear physical interpretation: atoms within correlation
    length xi should have similar B-factors.

    Tunable parameters (as buffers):
    - _k_neighbors: int, number of nearest neighbors
    - _correlation_length: float, distance scale for weight decay (Å)
    - _scale: float, scaling factor for loss magnitude

    Parameters
    ----------
    model : Model
        Reference to Model object.
    k_neighbors : int, optional
        Number of nearest neighbors to consider. Default is 50.
    correlation_length : float, optional
        Distance scale for weight decay in Angstrom. Default is 5.0.
    scale : float, optional
        Scaling factor for loss magnitude. Default is 5.0.
    exclude_bonded : bool, optional
        Exclude directly bonded atoms. Default is True.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    name: str = "adp/locality"

    def __init__(
        self,
        model: "Model" = None,
        k_neighbors: int = 50,
        correlation_length: float = 5.0,  # xi in Angstrom
        scale: float = 5.0,  # Scale factor for loss magnitude (reduced from 10.0)
        exclude_bonded: bool = True,
        verbose: int = 0,
    ):
        super().__init__(model, verbose, target_value=0.3, sigma=0.2)
        # Register tunable parameters as buffers
        self.register_buffer(
            "_k_neighbors", torch.tensor(k_neighbors, dtype=torch.int64)
        )
        self.register_buffer("_correlation_length", torch.tensor(correlation_length))
        self.register_buffer("_scale", torch.tensor(scale))
        self.exclude_bonded = exclude_bonded

        # Cache for neighbor indices and distances
        self._neighbor_indices = None  # (N, k_neighbors)
        self._neighbor_distances = None  # (N, k_neighbors)
        self._last_xyz_hash = None

    @property
    def k_neighbors(self) -> int:
        """Get k_neighbors value."""
        return self._k_neighbors.item()

    @k_neighbors.setter
    def k_neighbors(self, value: int):
        """Set k_neighbors value."""
        self._k_neighbors.fill_(value)

    @property
    def correlation_length(self) -> float:
        """Get correlation_length value."""
        return self._correlation_length.item()

    @correlation_length.setter
    def correlation_length(self, value: float):
        """Set correlation_length value."""
        self._correlation_length.fill_(value)

    @property
    def scale(self) -> float:
        """Get scale value."""
        return self._scale.item()

    @scale.setter
    def scale(self, value: float):
        """Set scale value."""
        self._scale.fill_(value)

    def _build_neighbor_list(self) -> None:
        """
        Build list of K nearest neighbors for each atom.

        Stores:
            _neighbor_indices: (N, k_neighbors) indices of neighbors
            _neighbor_distances: (N, k_neighbors) distances to neighbors
        """
        xyz = self.model.xyz()
        device = xyz.device
        n_atoms = xyz.shape[0]

        # Compute all pairwise distances (O(N^2) but simple and fast for proteins)
        k = min(self.k_neighbors, n_atoms - 1)

        # Compute distance matrix
        diff = xyz.unsqueeze(0) - xyz.unsqueeze(1)  # (N, N, 3)
        dist_matrix = torch.sqrt((diff**2).sum(dim=-1)).detach()  # (N, N)

        # Create mask for diagonal (self-distances) and bonded pairs
        # Use non-inplace operations to avoid gradient issues
        mask = torch.eye(n_atoms, device=device, dtype=torch.bool)

        # Apply mask using torch.where (non-inplace)
        dist_matrix = torch.where(
            mask, torch.tensor(float("inf"), device=device), dist_matrix
        )

        # Get k nearest neighbors
        distances, indices = torch.topk(dist_matrix, k, dim=1, largest=False)

        self._neighbor_indices = indices  # (N, k)
        self._neighbor_distances = distances  # (N, k)

        if self.verbose > 1:
            mean_dist = distances.mean().item()
            min_dist = distances.min().item()
            max_dist = distances[:, -1].mean().item()  # Farthest of k neighbors
            print(
                f"    Built K-NN list: k={k}, mean dist={mean_dist:.2f}A, "
                f"min={min_dist:.2f}A, max (kth)={max_dist:.2f}A"
            )

    def forward(self, recompute_neighbors: bool = False) -> torch.Tensor:
        """
        Compute weighted MSE on log(B) differences with exponential decay.

        loss = scale * mean_ij [w_ij * (log(B_i) - log(B_j))^2]
        where w_ij = exp(-d_ij / correlation_length)
        """
        # Check if cached tensors are on wrong device and need rebuilding
        model_device = self.model.xyz().device
        cache_stale = (
            self._neighbor_indices is not None
            and self._neighbor_indices.device != model_device
        )
        if recompute_neighbors or self._neighbor_indices is None or cache_stale:
            self._build_neighbor_list()

        adp = self.model.adp()
        device = adp.device
        n_atoms = len(adp)

        if n_atoms == 0 or self._neighbor_indices is None:
            return torch.tensor(0.0, device=device)

        log_adp = torch.log(adp.clamp(min=1e-3))

        indices = self._neighbor_indices  # (N, k)
        distances = self._neighbor_distances  # (N, k)

        # Gather neighbor log(ADP) values
        neighbor_log_adp = log_adp[indices]  # (N, k)

        # Compute pairwise differences: diff_ij = log(ADP_i) - log(ADP_j)
        diff = log_adp.unsqueeze(1) - neighbor_log_adp  # (N, k)

        weights = 1 / (distances + 1e-6)  # Avoid div by zero

        weights = weights / (weights.mean() + 1e-8)  # Normalize weights per atom

        # Weighted MSE
        weighted_sq_diff = weights * (diff / 0.5) ** 2  # (N, k)

        # Normalize by sum of weights to get weighted average
        loss = weighted_sq_diff.mean()

        return loss

    def stats(self) -> Dict[str, any]:
        """Get locality restraint statistics."""
        self._build_neighbor_list()

        if self._neighbor_indices is None:
            return {}

        adp = self.model.adp().detach()
        log_adp = torch.log(adp.clamp(min=1e-3))

        indices = self._neighbor_indices
        distances = self._neighbor_distances

        neighbor_log_adp = log_adp[indices]
        diff = log_adp.unsqueeze(1) - neighbor_log_adp

        # Exponential weights
        weights = torch.exp(-distances / self.correlation_length)

        # Weighted RMS
        weighted_sq_diff = weights * (diff**2)
        weighted_rms = torch.sqrt(weighted_sq_diff.sum() / weights.sum()).item()
        loss = self.forward()

        return {
            "loss": stat(loss.item(), VERBOSITY_STANDARD),
            "n_atoms": stat(len(adp), VERBOSITY_DEBUG),
            "weighted_rms_log": stat(weighted_rms, VERBOSITY_DETAILED),
            "rms_deviation_log": stat(
                torch.sqrt((diff**2).mean()).item(), VERBOSITY_DETAILED
            ),
            "max_deviation_log": stat(diff.abs().max().item(), VERBOSITY_DETAILED),
            "k_neighbors": stat(self.k_neighbors, VERBOSITY_DEBUG),
            "correlation_length": stat(self.correlation_length, VERBOSITY_DEBUG),
            "scale": stat(self.scale, VERBOSITY_DEBUG),
            "avg_neighbor_dist": stat(distances.mean().item(), VERBOSITY_DEBUG),
            "max_neighbor_dist": stat(distances.max().item(), VERBOSITY_DEBUG),
            "avg_weight": stat(weights.mean().item(), VERBOSITY_DEBUG),
        }
