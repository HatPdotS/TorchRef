import numpy as np
import torch
from typing import TYPE_CHECKING, Dict

from torchref.base.targets.adp import adp_locality_aniso_math
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

    Uses a spatial cell-list (O(N) memory, O(N·k) time) instead of
    a full N×N distance matrix, so it scales to arbitrarily large
    structures without memory issues.

    Parameters
    ----------
    model : Model
        Reference to Model object.
    k_neighbors : int, optional
        Number of nearest neighbors to consider. Default is 50.
    correlation_length : float, optional
        Distance scale for weight decay in Angstrom. Default is 5.0. Only
        used in ``stats()`` (for the exponential-decay reported weights);
        ``forward()`` weights neighbors by inverse distance instead.
    scale : float, optional
        Default is 5.0. Currently informational only: ``forward()`` does not
        multiply the loss by ``scale`` (it is surfaced only in ``stats()``),
        so it is not an active loss-magnitude lever.
    sigma_aniso : float, optional
        Sigma for the deviatoric (anisotropy) channel, used only when
        anisotropic atoms are present. Default is 0.5. Dimensionless, on the
        same scale as the magnitude channel's fixed 0.5 log-sigma.
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
        correlation_length: float = 5.0,
        scale: float = 5.0,
        sigma_aniso: float = 0.5,
        exclude_bonded: bool = True,
        verbose: int = 0,
        device=None,
    ):
        super().__init__(model, verbose, device=device)
        # Host-side, deliberately not buffers: all three are only ever reached
        # through the properties below, which immediately ``.item()`` them, and
        # their consumers (k-NN sizing, the exp() falloff, stats reporting) are
        # host-side. Keeping them on the device bought a sync per access.
        self._k_neighbors = int(k_neighbors)
        self._correlation_length = float(correlation_length)
        self._scale = float(scale)
        # Sigma for the deviatoric (anisotropy) channel; only used when
        # anisotropic atoms are present. Dimensionless and on the same scale as
        # the magnitude channel's fixed 0.5 log-sigma (the channel restrains
        # fractional anisotropy dev/B_eq, the analogue of log B_eq). This one
        # *is* a buffer: it is handed to adp_locality_aniso_math as a tensor.
        self._register_scalar("_sigma_aniso", float(sigma_aniso))
        self.exclude_bonded = exclude_bonded

        # Cache for neighbor indices and distances
        self._neighbor_indices = None  # (N, k_neighbors)
        self._neighbor_distances = None  # (N, k_neighbors)
        self._last_xyz_hash = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Absorb the retired ``_k_neighbors`` / ``_correlation_length`` /
        ``_scale`` buffers.

        All three are host-side Python scalars now (see ``__init__``).
        Checkpoints predating that change still carry them, and a
        ``strict=True`` load would reject them as unexpected keys -- so restore
        their values rather than discarding a user's tuning.
        """
        for legacy, cast in (
            ("_k_neighbors", int),
            ("_correlation_length", float),
            ("_scale", float),
        ):
            saved = state_dict.pop(prefix + legacy, None)
            if saved is not None:
                setattr(self, legacy, cast(saved.item()))
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
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float):
        self._scale = float(value)

    @property
    def sigma_aniso(self) -> float:
        return self._sigma_aniso.item()

    @sigma_aniso.setter
    def sigma_aniso(self, value: float):
        self._sigma_aniso.fill_(value)

    # ------------------------------------------------------------------
    # Spatial-hash k-NN (O(N) memory)
    # ------------------------------------------------------------------

    def _build_neighbor_list(self) -> None:
        """
        Build k-nearest-neighbor list using a spatial cell-list.

        Memory usage is O(N·k) for the output arrays and O(N) for the
        cell-list bookkeeping, instead of O(N²) for a full distance matrix.
        """
        xyz = self.model.xyz()
        device = xyz.device
        n_atoms = xyz.shape[0]
        k = min(self.k_neighbors, n_atoms - 1)

        coords = xyz.detach().cpu().numpy()

        # Cell size should cover the kth-neighbor distance.  For proteins
        # k=50 neighbors are typically within ~8-10 Å, so 12 Å is safe.
        cell_size = 12.0

        xyz_min = coords.min(axis=0)
        cell_idx = ((coords - xyz_min) / cell_size).astype(np.int64)

        grid_dims = cell_idx.max(axis=0) + 1
        gx, gy, gz = int(grid_dims[0]), int(grid_dims[1]), int(grid_dims[2])
        gyz = gy * gz

        flat = cell_idx[:, 0] * gyz + cell_idx[:, 1] * gz + cell_idx[:, 2]

        order = np.argsort(flat)
        sorted_flat = flat[order]

        unique_cells, first_idx, counts = np.unique(
            sorted_flat, return_index=True, return_counts=True
        )
        n_unique = len(unique_cells)

        # start[i] .. start[i+1] are the atoms in unique cell i
        starts = np.empty(n_unique + 1, dtype=np.int64)
        starts[0] = 0
        starts[1:] = np.cumsum(counts)

        # flat_cell -> unique index (-1 = empty)
        n_grid = gx * gyz
        cell_lookup = np.full(n_grid, -1, dtype=np.int64)
        cell_lookup[unique_cells] = np.arange(n_unique, dtype=np.int64)

        # 27 neighbor offsets (self + all adjacent cells)
        offsets = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    offsets.append((dx, dy, dz, dx * gyz + dy * gz + dz))

        # For each atom, collect candidate neighbors and keep top-k
        all_neighbor_idx = np.zeros((n_atoms, k), dtype=np.int64)
        all_neighbor_dist = np.full((n_atoms, k), np.inf, dtype=np.float32)

        # atom_cell[i] = unique-cell index for atom i
        atom_cell = np.empty(n_atoms, dtype=np.int64)
        atom_cell[order] = np.repeat(np.arange(n_unique), counts)

        for ci in range(n_unique):
            cell_flat = int(unique_cells[ci])
            sa, ea = int(starts[ci]), int(starts[ci + 1])
            atoms_a = order[sa:ea]
            xyz_a = coords[atoms_a]

            cx = cell_flat // gyz
            cy = (cell_flat % gyz) // gz
            cz = cell_flat % gz

            # Collect all candidate neighbor atoms from adjacent cells
            cand_atoms_list = []
            cand_xyz_list = []
            for dx, dy, dz, _ in offsets:
                ncx, ncy, ncz = cx + dx, cy + dy, cz + dz
                if ncx < 0 or ncx >= gx or ncy < 0 or ncy >= gy or ncz < 0 or ncz >= gz:
                    continue
                nb_flat = ncx * gyz + ncy * gz + ncz
                nb_ci = int(cell_lookup[nb_flat])
                if nb_ci < 0:
                    continue
                sb, eb = int(starts[nb_ci]), int(starts[nb_ci + 1])
                cand_atoms_list.append(order[sb:eb])
                cand_xyz_list.append(coords[order[sb:eb]])

            if not cand_atoms_list:
                continue

            cand_atoms = np.concatenate(cand_atoms_list)
            cand_xyz = np.concatenate(cand_xyz_list, axis=0)

            # Distances from each atom in this cell to all candidates
            # shape: (len(atoms_a), len(cand_atoms))
            diff = xyz_a[:, None, :] - cand_xyz[None, :, :]
            dist = np.sqrt((diff * diff).sum(axis=-1))

            for li, ai in enumerate(atoms_a):
                d = dist[li]
                # Mask self
                self_mask = cand_atoms == ai
                d[self_mask] = np.inf

                if len(d) <= k:
                    top_k_idx = np.argsort(d)[:k]
                else:
                    top_k_idx = np.argpartition(d, k)[:k]
                    # Sort the top-k for deterministic order
                    sub_order = np.argsort(d[top_k_idx])
                    top_k_idx = top_k_idx[sub_order]

                n_valid = min(k, len(top_k_idx))
                all_neighbor_idx[ai, :n_valid] = cand_atoms[top_k_idx[:n_valid]]
                all_neighbor_dist[ai, :n_valid] = d[top_k_idx[:n_valid]]

        self._neighbor_indices = torch.from_numpy(all_neighbor_idx).to(device)
        self._neighbor_distances = torch.from_numpy(all_neighbor_dist).to(device)

        if self.verbose > 1:
            mean_dist = float(all_neighbor_dist[all_neighbor_dist < np.inf].mean())
            print(
                f"    Built K-NN list (spatial hash): k={k}, "
                f"mean dist={mean_dist:.2f}A"
            )

    def forward(self, recompute_neighbors: bool = False) -> torch.Tensor:
        """
        Compute the inverse-distance-weighted sum of squared log(B) differences.

        loss = sum_ij [w_ij * ((log(B_i) - log(B_j)) / 0.5)^2]
        where w_ij = 1 / (d_ij + eps) and the inner sum runs over the k
        nearest neighbors j of each atom i. This formula describes the
        isotropic path only. When anisotropic atoms are present
        (``model._aniso_is_empty`` is False) the loss instead routes to
        ``adp_locality_aniso_math`` on the unified U6 basis: a magnitude
        channel on B_eq that reproduces the isotropic loss above, plus a
        deviatoric/fractional-anisotropy channel scaled by ``sigma_aniso``.

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

        # Anisotropic atoms present: restrain the full U tensors (magnitude
        # channel on B_eq -- which reproduces the isotropic loss below -- plus a
        # deviatoric/anisotropy channel). Isotropic-only models keep the
        # original B-factor path (numerically unchanged, zero overhead).
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
        """Get locality restraint statistics.

        Notes
        -----
        The ``weighted_rms_log`` / ``avg_weight`` entries use exponential
        decay weights ``exp(-d / correlation_length)``, which is a
        **different** weighting scheme from ``forward()`` (inverse distance
        ``1 / (d + eps)``). The reported "weighted" stats therefore do not
        correspond to the weights used in the loss.
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
            "scale": stat(self.scale, VERBOSITY_DEBUG),
            "avg_neighbor_dist": stat(distances.mean().item(), VERBOSITY_DEBUG),
            "max_neighbor_dist": stat(distances.max().item(), VERBOSITY_DEBUG),
            "avg_weight": stat(weights.mean().item(), VERBOSITY_DEBUG),
        }
