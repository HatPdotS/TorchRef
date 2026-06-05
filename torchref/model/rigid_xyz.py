"""
Rigid-body xyz container.

Drop-in replacement for the ``MixedTensor`` that ``ModelFT`` uses for atomic
coordinates. Instead of refining every atom independently, atoms are grouped
into rigid bodies (one per chain) and only a per-body ZYZ-Euler rotation and a
per-body translation are refinable. ``forward()`` reconstructs full Cartesian
coordinates by rotating each chain around its centroid and translating it.
"""

from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from torchref.base.alignment.rotation import rotation_matrix_euler_zyz
from torchref.utils.caching import CachedForwardMixin
from torchref.utils.device_mixin import DeviceMixin


class RigidXYZTensor(DeviceMixin, CachedForwardMixin, nn.Module):
    """
    Per-chain rigid-body parametrization of atomic coordinates.

    Parameters
    ----------
    original_xyz : torch.Tensor
        ``(N, 3)`` reference coordinates. Stored as a buffer; rotation /
        translation are applied on top.
    chain_ids : Sequence
        Length-``N`` sequence of chain identifiers (one per atom). Chains are
        assigned an integer index in order of first appearance.
    dtype : torch.dtype, optional
        Floating dtype. Defaults to ``original_xyz.dtype``.
    device : torch.device, optional
        Defaults to ``original_xyz.device``.
    name : str, optional
        Wrapper name. Defaults to ``"xyz"`` so ``Model`` consumers find it.
    """

    def __init__(
        self,
        original_xyz: Optional[torch.Tensor] = None,
        chain_ids: Optional[Sequence] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        name: str = "xyz",
        mobile_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self._name = name

        if original_xyz is None:
            # Empty init for state_dict loading.
            self.register_buffer("original_xyz", torch.empty(0, 3))
            self.register_buffer("chain_indices", torch.empty(0, dtype=torch.long))
            self.register_buffer("chain_centers", torch.empty(0, 3))
            self.register_buffer("mobile_mask", torch.empty(0, dtype=torch.bool))
            self.euler_angles = nn.Parameter(torch.empty(0, 3))
            self.translations = nn.Parameter(torch.empty(0, 3))
            self._n_chains = 0
            self._chain_id_order: list = []
            return

        if dtype is None:
            dtype = original_xyz.dtype
        if device is None:
            device = original_xyz.device

        if chain_ids is None or len(chain_ids) != original_xyz.shape[0]:
            raise ValueError(
                "chain_ids must be provided with length matching original_xyz "
                f"({original_xyz.shape[0]} atoms)"
            )

        N = original_xyz.shape[0]
        if mobile_mask is None:
            mobile_arr = np.ones(N, dtype=bool)
        else:
            mobile_arr = (
                mobile_mask.detach().cpu().numpy().astype(bool)
                if isinstance(mobile_mask, torch.Tensor)
                else np.asarray(mobile_mask, dtype=bool)
            )
            if mobile_arr.shape[0] != N:
                raise ValueError(
                    f"mobile_mask length {mobile_arr.shape[0]} != n_atoms {N}"
                )

        # Build chain index: -1 sentinel for fixed atoms.
        chain_id_order: list = []
        chain_id_to_idx: dict = {}
        idx_list = np.empty(N, dtype=np.int64)
        for i, cid in enumerate(chain_ids):
            if not mobile_arr[i]:
                idx_list[i] = -1
                continue
            if cid not in chain_id_to_idx:
                chain_id_to_idx[cid] = len(chain_id_order)
                chain_id_order.append(cid)
            idx_list[i] = chain_id_to_idx[cid]

        n_chains = len(chain_id_order)
        if n_chains == 0:
            raise ValueError(
                "No mobile atoms after filtering — every atom was masked out."
            )

        original_xyz_t = original_xyz.to(dtype=dtype, device=device).detach().clone()
        mobile_t = torch.from_numpy(mobile_arr).to(device=device)
        # Safe chain index buffer: replace -1 with 0 (still gathers a valid
        # entry; forward() blends with mobile_mask so the rotated/translated
        # result is discarded for non-mobile atoms).
        safe_idx_np = np.where(idx_list >= 0, idx_list, 0)
        chain_indices = torch.from_numpy(safe_idx_np).to(device=device)

        # Per-chain centroid: only over MOBILE atoms of each chain.
        mobile_idx = torch.from_numpy(idx_list[mobile_arr]).to(device=device)
        mobile_xyz = original_xyz_t[mobile_t]
        chain_centers = torch.zeros((n_chains, 3), dtype=dtype, device=device)
        chain_centers.index_add_(0, mobile_idx, mobile_xyz)
        counts = torch.zeros(n_chains, dtype=dtype, device=device)
        counts.index_add_(
            0,
            mobile_idx,
            torch.ones(mobile_xyz.shape[0], dtype=dtype, device=device),
        )
        chain_centers = chain_centers / counts.unsqueeze(1).clamp(min=1)

        self.register_buffer("original_xyz", original_xyz_t)
        self.register_buffer("chain_indices", chain_indices)
        self.register_buffer("chain_centers", chain_centers)
        self.register_buffer("mobile_mask", mobile_t)

        self.euler_angles = nn.Parameter(
            torch.zeros((n_chains, 3), dtype=dtype, device=device)
        )
        self.translations = nn.Parameter(
            torch.zeros((n_chains, 3), dtype=dtype, device=device)
        )

        self._n_chains = n_chains
        self._chain_id_order = list(chain_id_order)

    # -----------------------------------------------------------------------
    # Forward — reconstruct full xyz
    # -----------------------------------------------------------------------
    def forward(self) -> torch.Tensor:
        # rotation_matrix_euler_zyz accepts batched (B, 3) → (B, 3, 3).
        R = rotation_matrix_euler_zyz(self.euler_angles)  # (n_chains, 3, 3)

        per_atom_R = R[self.chain_indices]  # (N, 3, 3)
        per_atom_center = self.chain_centers[self.chain_indices]  # (N, 3)
        per_atom_trans = self.translations[self.chain_indices]  # (N, 3)

        centered = self.original_xyz - per_atom_center  # (N, 3)
        rotated = torch.einsum("nij,nj->ni", per_atom_R, centered)
        transformed = rotated + per_atom_center + per_atom_trans
        # Fixed atoms (mobile_mask == False) retain their original positions
        # regardless of the chain's rigid-body parameters.
        return torch.where(
            self.mobile_mask.unsqueeze(1), transformed, self.original_xyz
        )

    # -----------------------------------------------------------------------
    # Interface compatibility with MixedTensor
    # -----------------------------------------------------------------------
    @property
    def shape(self):
        return tuple(self.original_xyz.shape)

    @property
    def dtype(self):
        return self.original_xyz.dtype

    @property
    def device(self):
        return self.original_xyz.device

    @property
    def n_chains(self) -> int:
        return self._n_chains

    @property
    def chain_id_order(self) -> list:
        return list(self._chain_id_order)

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def fixed_values(self) -> torch.Tensor:
        # ``Model.get_aniso`` (model.py) reaches into ``self.xyz.fixed_values``
        # to allocate empty placeholder tensors when no anisotropic atoms are
        # present. Returning ``original_xyz`` here gives it the right dtype /
        # device / shape without exposing a buffer for direct mutation.
        return self.original_xyz

    def get_refinable_count(self) -> int:
        return self.euler_angles.numel() + self.translations.numel()

    def get_fixed_count(self) -> int:
        return 0

    def __getitem__(self, key) -> torch.Tensor:
        return self()[key]

    # Rigid container: per-atom mutation makes no sense.
    def __setitem__(self, key, value) -> None:
        raise NotImplementedError(
            "RigidXYZTensor does not support per-atom assignment; "
            "set euler_angles / translations directly or commit back to a ModelFT."
        )

    def set(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        raise NotImplementedError(
            "RigidXYZTensor does not support per-atom assignment."
        )

    def refine(self, *args, **kwargs):
        raise NotImplementedError(
            "RigidXYZTensor refines per-chain rotation+translation only."
        )

    def fix(self, *args, **kwargs):
        raise NotImplementedError(
            "RigidXYZTensor refines per-chain rotation+translation only."
        )

    def fix_all(self, freeze_at_current: bool = True):
        self.euler_angles.requires_grad_(False)
        self.translations.requires_grad_(False)

    def refine_all(self):
        self.euler_angles.requires_grad_(True)
        self.translations.requires_grad_(True)

    def update_refinable_mask(self, *args, **kwargs):
        # No-op: the rigid container has no per-atom mask.
        return

    def update_fixed_values(self, new_values: torch.Tensor):
        # Update the reference coordinates and reset the per-chain transforms
        # so forward() reproduces these new values.
        if new_values.shape != self.shape:
            raise ValueError(
                f"new_values shape {tuple(new_values.shape)} must match {self.shape}"
            )
        with torch.no_grad():
            self.original_xyz.copy_(new_values.to(dtype=self.dtype, device=self.device))
            # Recompute chain centers over MOBILE atoms only.
            mobile = self.mobile_mask
            mobile_idx = self.chain_indices[mobile]
            mobile_xyz = self.original_xyz[mobile]
            counts = torch.zeros(self._n_chains, dtype=self.dtype, device=self.device)
            counts.index_add_(
                0,
                mobile_idx,
                torch.ones(mobile_xyz.shape[0], dtype=self.dtype, device=self.device),
            )
            centers = torch.zeros(
                (self._n_chains, 3), dtype=self.dtype, device=self.device
            )
            centers.index_add_(0, mobile_idx, mobile_xyz)
            self.chain_centers.copy_(centers / counts.unsqueeze(1).clamp(min=1))
            self.euler_angles.zero_()
            self.translations.zero_()
        self.reset_forward_cache()

    def detach(self) -> torch.Tensor:
        return self().detach()

    def parameters(self, recurse: bool = True):
        # Match MixedTensor.parameters() return convention (a list).
        return [self.euler_angles, self.translations]

    def copy(self) -> "RigidXYZTensor":
        new = RigidXYZTensor(
            original_xyz=self.original_xyz.clone(),
            chain_ids=self._chain_id_order_for_atoms(),
            dtype=self.dtype,
            device=self.device,
            name=self._name,
            mobile_mask=self.mobile_mask.clone(),
        )
        with torch.no_grad():
            new.euler_angles.copy_(self.euler_angles)
            new.translations.copy_(self.translations)
        return new

    def _chain_id_order_for_atoms(self) -> list:
        # Reconstruct per-atom chain ids from chain_indices + chain_id_order.
        # Non-mobile atoms get a dummy "_FIXED_" tag so the constructor can
        # reproduce the same mobile_mask via the mobile_mask kwarg.
        idx_cpu = self.chain_indices.detach().cpu().tolist()
        mob_cpu = self.mobile_mask.detach().cpu().tolist()
        return [
            self._chain_id_order[i] if m else "_FIXED_"
            for i, m in zip(idx_cpu, mob_cpu)
        ]

    # -----------------------------------------------------------------------
    # Materialize back into a regular MixedTensor.
    # -----------------------------------------------------------------------
    def to_mixed_tensor(self):
        from torchref.model.parameter_wrappers import MixedTensor

        with torch.no_grad():
            full = self().detach().clone()
        return MixedTensor(full, name=self._name, device=self.device)

    def __repr__(self) -> str:
        return f"RigidXYZTensor(n_atoms={self.shape[0]}, n_chains={self._n_chains})"
