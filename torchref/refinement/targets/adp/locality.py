"""K-nearest-neighbour spatial smoothness restraint on the ADPs."""

import numpy as np
import torch
from scipy.spatial import cKDTree
from typing import TYPE_CHECKING, Dict

from torchref.base.targets.adp import adp_locality_aniso_math
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    stat,
)

from .base import ADPTarget

if TYPE_CHECKING:
    from torchref.model.model import Model


class ADPLocalityTarget(ADPTarget):
    """
    Proximity-based ADP restraint over each atom's K nearest neighbours.

    Built on a spatial cell-list (O(N) memory, O(N·k) time) rather than a full
    N×N distance matrix, so it scales to arbitrarily large structures. Bonded
    neighbours are included; SIMU
    (:class:`~torchref.refinement.targets.adp.ADPSimilarityTarget`) restrains them
    separately.

    Parameters
    ----------
    model : Model
        Reference to Model object.
    k_neighbors : int, optional
        Number of nearest neighbors to consider. Default is 50.
    correlation_length : float, optional
        Weight-decay distance scale (Å), default 5.0. Used **only** by ``stats()``;
        ``forward()`` weights by inverse distance instead.
    sigma_aniso : float, optional
        Sigma for the deviatoric (anisotropy) channel, used only when anisotropic
        atoms are present. Default 0.5, dimensionless and on the same scale as the
        magnitude channel's fixed 0.5 log-sigma.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    name: str = "adp/locality"

    def __init__(
        self,
        model: "Model" = None,
        k_neighbors: int = 50,
        correlation_length: float = 5.0,
        sigma_aniso: float = 0.5,
        verbose: int = 0,
        device=None,
    ):
        super().__init__(model, verbose, device=device)
        # Host-side, deliberately not buffers: every consumer (k-NN sizing, the
        # exp() falloff, stats) is host-side, so a device tensor would only add a
        # sync per access.
        self._k_neighbors = int(k_neighbors)
        self._correlation_length = float(correlation_length)
        # This one *is* a buffer, unlike the two above: adp_locality_aniso_math
        # takes it as a tensor. It restrains fractional anisotropy dev/B_eq, the
        # analogue of log B_eq, hence the shared 0.5 scale.
        self._register_scalar("_sigma_aniso", float(sigma_aniso))

        # Cache for neighbor indices and distances
        self._neighbor_indices = None  # (N, k_neighbors)
        self._neighbor_distances = None  # (N, k_neighbors)
        self._last_xyz_hash = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Accept checkpoints that store ``_k_neighbors``/``_correlation_length`` as
        tensors, restoring their values, and drop a stored ``_scale``, which nothing
        reads; a ``strict=True`` load would otherwise reject them as unexpected keys.
        """
        for key, cast in (("_k_neighbors", int), ("_correlation_length", float)):
            saved = state_dict.pop(prefix + key, None)
            if saved is not None:
                setattr(self, key, cast(saved.item()))
        state_dict.pop(prefix + "_scale", None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def k_neighbors(self) -> int:
        return self._k_neighbors

    @k_neighbors.setter
    def k_neighbors(self, value: int):
        self._k_neighbors = int(value)

    @property
    def correlation_length(self) -> float:
        return self._correlation_length

    @correlation_length.setter
    def correlation_length(self, value: float):
        self._correlation_length = float(value)

    @property
    def sigma_aniso(self) -> float:
        return self._sigma_aniso.item()

    @sigma_aniso.setter
    def sigma_aniso(self, value: float):
        self._sigma_aniso.fill_(value)

    # ------------------------------------------------------------------
    # k-NN list
    # ------------------------------------------------------------------

    def _build_neighbor_list(self) -> None:
        """Build each atom's list of its ``k`` nearest other atoms, nearest first.

        Uses a k-d tree. ``stats()`` rebuilds the list on every call, which is also what
        keeps ``forward()``'s list current as atoms move, so this runs every time
        metrics are collected and has to be cheap. The per-atom Python loop over a cell
        list that it replaces was one of the largest costs of a refinement, and more so
        with hydrogens, which double the atom count.

        Distances are recomputed from the coordinates in their own dtype, the way the
        loss sees them, rather than taken from the tree.
        """
        xyz = self.model.xyz()
        device = xyz.device
        n_atoms = xyz.shape[0]
        k = min(self.k_neighbors, n_atoms - 1)

        coords = xyz.detach().cpu().numpy()

        if k <= 0:
            all_neighbor_idx = np.zeros((n_atoms, 0), dtype=np.int64)
            all_neighbor_dist = np.zeros((n_atoms, 0), dtype=np.float32)
        else:
            # k + 1, because each atom is its own nearest point.
            _, idx = cKDTree(coords).query(coords, k=k + 1)
            # Drop the atom itself by index, not by column: a coincident atom can sort
            # ahead of it. Where it is absent (more than k coincident atoms) the
            # farthest candidate goes instead.
            keep = idx != np.arange(n_atoms)[:, None]
            keep[keep.all(axis=1), -1] = False
            all_neighbor_idx = idx[keep].reshape(n_atoms, k).astype(np.int64)
            diff = coords[:, None, :] - coords[all_neighbor_idx]
            all_neighbor_dist = np.sqrt((diff * diff).sum(axis=-1)).astype(np.float32)

        self._neighbor_indices = torch.from_numpy(all_neighbor_idx).to(device)
        self._neighbor_distances = torch.from_numpy(all_neighbor_dist).to(device)

        if self.verbose > 1 and all_neighbor_dist.size:
            print(
                f"    Built K-NN list (k-d tree): k={k}, "
                f"mean dist={float(all_neighbor_dist.mean()):.2f}A"
            )

    def forward(self, recompute_neighbors: bool = False) -> torch.Tensor:
        """
        Inverse-distance-weighted sum of squared log(B) differences.

        ``loss = sum_ij w_ij ((log B_i - log B_j) / 0.5)^2`` with
        ``w_ij = 1/(d_ij + eps)`` over each atom's k nearest neighbours -- the
        isotropic path. With anisotropic atoms present the loss instead routes to
        ``adp_locality_aniso_math`` on the unified U6 basis: a B_eq magnitude channel
        reproducing the above, plus a fractional-anisotropy channel at ``sigma_aniso``.

        Parameters
        ----------
        recompute_neighbors : bool, optional
            Rebuild the k-nearest-neighbor list before evaluating the loss.
            Default is False; the list is also rebuilt automatically when no
            cache exists or it lives on a different device than the model.

        Returns
        -------
        torch.Tensor
            Scalar loss (the summed weighted squared log-B differences).
        """
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

        indices = self._neighbor_indices
        distances = self._neighbor_distances

        # With any anisotropic atom, restrain the full U tensors; isotropic-only
        # models take the cheaper B-factor path below, numerically unchanged.
        if not getattr(self.model, "_aniso_is_empty", True):
            u6 = self.model.adp_u6()
            return adp_locality_aniso_math(
                u6, indices, distances, self._sigma_aniso
            )

        log_adp = torch.log(adp.clamp(min=1e-3))

        neighbor_log_adp = log_adp[indices]
        diff = log_adp.unsqueeze(1) - neighbor_log_adp

        weights = 1 / (distances + 1e-6)

        weighted_sq_diff = weights * (diff / 0.5) ** 2
        loss = weighted_sq_diff.sum()

        return loss

    def stats(self) -> Dict[str, any]:
        """Locality restraint statistics.

        Caution: ``weighted_rms_log`` and ``avg_weight`` use exponential-decay
        weights ``exp(-d / correlation_length)``, **not** the inverse-distance
        weights ``forward()`` uses, so they do not describe the loss's weighting.
        """
        self._build_neighbor_list()

        if self._neighbor_indices is None:
            return {}

        adp = self.model.adp().detach()
        log_adp = torch.log(adp.clamp(min=1e-3))

        indices = self._neighbor_indices
        distances = self._neighbor_distances

        neighbor_log_adp = log_adp[indices]
        diff = log_adp.unsqueeze(1) - neighbor_log_adp

        weights = torch.exp(-distances / self.correlation_length)

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
            "avg_neighbor_dist": stat(distances.mean().item(), VERBOSITY_DEBUG),
            "max_neighbor_dist": stat(distances.max().item(), VERBOSITY_DEBUG),
            "avg_weight": stat(weights.mean().item(), VERBOSITY_DEBUG),
        }
