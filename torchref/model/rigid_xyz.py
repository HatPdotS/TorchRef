"""
Rigid-body xyz container.

Drop-in replacement for the ``MixedTensor`` that ``ModelFT`` uses for atomic
coordinates. Instead of refining every atom independently, atoms are grouped
into rigid bodies (one per chain) and only a per-body XYZ-Euler rotation and
a per-body translation are refinable. ``forward()`` reconstructs full
Cartesian coordinates by rotating each chain around its mass-weighted
centroid and translating it. XYZ Euler matches Phenix's default
``euler_angle_convention`` and keeps the rotation Jacobian full-rank at the
origin (no gimbal lock when angles reset to zero after ``bake()``).
"""

from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from torchref.base.alignment.rotation import rotation_matrix_euler_xyz
from torchref.config import get_float_dtype, normalize_device
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
        atom_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self._name = name

        if original_xyz is None:
            # Empty init for state_dict loading. Honour the requested
            # device/dtype: otherwise every buffer lands on CPU regardless of
            # what the caller asked for, and only a later ``.to()`` repairs it.
            device = normalize_device(device)
            dtype = dtype if dtype is not None else get_float_dtype()
            self.register_buffer("original_xyz", torch.empty(0, 3, device=device, dtype=dtype))
            self.register_buffer(
                "chain_indices", torch.empty(0, dtype=torch.long, device=device)  # dtype-ok: empty chain_indices buffer; indexing requires long
            )
            self.register_buffer("chain_centers", torch.empty(0, 3, device=device, dtype=dtype))
            self.register_buffer(
                "mobile_mask", torch.empty(0, dtype=torch.bool, device=device)
            )
            self.register_buffer("atom_weights", torch.empty(0, device=device, dtype=dtype))
            self.euler_angles = nn.Parameter(torch.empty(0, 3, device=device, dtype=dtype))
            self.translations = nn.Parameter(torch.empty(0, 3, device=device, dtype=dtype))
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

        # Per-atom weights for the centroid (= rotation center). Defaults
        # to uniform; pass atomic Z (or true masses) to use a mass-weighted
        # center of mass — matches Phenix's
        # ``apply_rigid_body_shift_obj(..., atomic_weights=...)`` in
        # mmtbx/refinement/rigid_body.py.
        if atom_weights is None:
            atom_weights_t = torch.ones(N, dtype=dtype, device=device)
        else:
            if isinstance(atom_weights, torch.Tensor):
                atom_weights_t = atom_weights.to(dtype=dtype, device=device).detach().clone()
            else:
                atom_weights_t = torch.as_tensor(atom_weights, dtype=dtype, device=device)
            if atom_weights_t.shape[0] != N:
                raise ValueError(
                    f"atom_weights length {atom_weights_t.shape[0]} != n_atoms {N}"
                )
            atom_weights_t = atom_weights_t.reshape(-1).contiguous()

        # Per-chain mass-weighted center: only over MOBILE atoms of each chain.
        mobile_idx = torch.from_numpy(idx_list[mobile_arr]).to(device=device)
        mobile_xyz = original_xyz_t[mobile_t]
        mobile_w = atom_weights_t[mobile_t]
        chain_centers = torch.zeros((n_chains, 3), dtype=dtype, device=device)
        chain_centers.index_add_(0, mobile_idx, mobile_xyz * mobile_w.unsqueeze(1))
        w_sum = torch.zeros(n_chains, dtype=dtype, device=device)
        w_sum.index_add_(0, mobile_idx, mobile_w)
        chain_centers = chain_centers / w_sum.unsqueeze(1).clamp(min=1e-12)

        self.register_buffer("original_xyz", original_xyz_t)
        self.register_buffer("chain_indices", chain_indices)
        self.register_buffer("chain_centers", chain_centers)
        self.register_buffer("mobile_mask", mobile_t)
        self.register_buffer("atom_weights", atom_weights_t)

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
        """Full ``(N, 3)`` coordinates: each chain rotated about its weighted
        centroid and translated. Non-mobile atoms keep ``original_xyz``.
        """
        # XYZ Euler — same convention as Phenix's default rigid-body
        # parametrization. Critical near macro-cycle resets (after bake()
        # the angles are exactly zero): XYZ keeps the Jacobian full-rank
        # at the origin, while ZYZ has a gimbal-lock singularity there
        # (dR/dα_1 and dR/dα_3 collapse onto z-axis rotations when β=0).
        R = rotation_matrix_euler_xyz(self.euler_angles)  # (n_chains, 3, 3)

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

    @property
    def refinable_params(self) -> torch.Tensor:
        # Compat with ``MixedTensor`` consumers (e.g. ModelFT's dtype guard
        # in ``_check_forward_dtype``). This is a dtype-probe shim only: it
        # returns just one leaf (``euler_angles``) so callers can inspect the
        # float dtype, NOT the full refinable set. The actual refinable count
        # spans both leaves (euler_angles + translations); see
        # ``get_refinable_count``. Both leaves share the model's float dtype.
        return self.euler_angles

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

    def bake(self) -> None:
        """Bake the current rigid transformation into ``original_xyz``.

        Replaces the reference coordinates with the current
        rotated+translated pose, recomputes per-chain centroids, and
        zeros ``euler_angles + translations``. The next ``forward()``
        reproduces the same Cartesian coordinates with all rigid
        parameters at zero — so subsequent LBFGS steps stay in the
        small-angle regime, matching Phenix's per-macro-cycle reset
        (``r_initial=t_initial=[0,0,0]`` in ``mmtbx.refinement.rigid_body``).
        """
        with torch.no_grad():
            new_xyz = self().detach()
        self.update_fixed_values(new_xyz)

    def update_fixed_values(self, new_values: torch.Tensor):
        """Adopt ``new_values`` as the reference pose, ZEROING the rigid-body
        parameters and recomputing the chain centroids (see :meth:`bake`).
        """
        # Update the reference coordinates and reset the per-chain transforms
        # so forward() reproduces these new values.
        if new_values.shape != self.shape:
            raise ValueError(
                f"new_values shape {tuple(new_values.shape)} must match {self.shape}"
            )
        with torch.no_grad():
            self.original_xyz.copy_(new_values.to(dtype=self.dtype, device=self.device))
            # Recompute mass-weighted chain centers over MOBILE atoms only,
            # using the per-atom weights stored at construction (uniform by
            # default; atomic Z when threaded from Model.use_rigid_xyz).
            mobile = self.mobile_mask
            mobile_idx = self.chain_indices[mobile]
            mobile_xyz = self.original_xyz[mobile]
            mobile_w = self.atom_weights[mobile]
            w_sum = torch.zeros(self._n_chains, dtype=self.dtype, device=self.device)
            w_sum.index_add_(0, mobile_idx, mobile_w)
            centers = torch.zeros(
                (self._n_chains, 3), dtype=self.dtype, device=self.device
            )
            centers.index_add_(0, mobile_idx, mobile_xyz * mobile_w.unsqueeze(1))
            self.chain_centers.copy_(centers / w_sum.unsqueeze(1).clamp(min=1e-12))
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
            atom_weights=self.atom_weights.clone(),
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
        """A per-atom :class:`MixedTensor` holding the current transformed
        coordinates, for handing per-atom refinement back to ``Model``.
        """
        from torchref.model.parameter_wrappers import MixedTensor

        with torch.no_grad():
            full = self().detach().clone()
        return MixedTensor(full, name=self._name, device=self.device)

    def __repr__(self) -> str:
        return f"RigidXYZTensor(n_atoms={self.shape[0]}, n_chains={self._n_chains})"
