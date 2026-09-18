"""Coordinate wrapper whose hydrogen rows ride on their parents.

A :class:`RidingXYZTensor` looks like a :class:`~torchref.model.parameter_wrappers.MixedTensor`
over the whole atom table -- ``forward()`` returns ``(N, 3)``, masks are given in atom
space -- but only the non-riding rows are stored and refined. Each riding hydrogen is a
reference offset in a frame built from its parent and two reference heavy atoms
(:mod:`torchref.base.coordinates.local_frame`), rebuilt from the current heavy
coordinates on every forward. A force on a hydrogen therefore lands on the atoms that
carry it, which is the riding-hydrogen convention of Phenix and Refmac.

Two index spaces meet here and callers must not mix them: everything public -- masks
passed to ``update_refinable_mask`` / ``refine`` / ``fix`` / ``set``, the result of
``forward()``, ``__getitem__`` / ``__setitem__`` -- is in FULL atom space, while the
inherited ``refinable_mask`` / ``fixed_values`` / ``refinable_params`` and the counts
from ``get_refinable_count()`` are in STORAGE space (the non-riding rows). Independent
angles are held in ``torsions`` and ``rotations`` and exposed together with the stored
coordinates by ``optimization_parameters()``. This is the
same contract :class:`~torchref.model.parameter_wrappers.OccupancyTensor` uses for its
collapsed groups.
"""

from typing import Iterator, Optional, Union

import numpy as np
import torch
from torchref.config import get_int_dtype
import torch.nn as nn
from torchref.base.coordinates.local_frame import (
    frame_is_degenerate,
    local_frame_coordinates,
    place_local_frame,
    rotate_vectors,
)
from torchref.model.parameter_wrappers import MixedTensor
from torchref.topology.hydrogens import HydrogenFrames


class _DerivedRowsMixin:
    """Bookkeeping shared by wrappers that store some rows and derive the rest.

    Registers the row maps as buffers and maintains the storage-space gathers the
    derivation needs. Subclasses call :meth:`_register_rows` after the parent
    constructor has run and :meth:`_rebuild_row_cache` from ``_build_index_cache``.
    """

    #: Full-space row of every stored row, ``(N_base,)``.
    base_row: torch.Tensor
    #: Full-space row of every derived (riding) row, ``(H,)``.
    h_row: torch.Tensor
    #: Frame atoms per riding row, full-space, ``(H,)`` each. Absent -> clamped to 0
    #: and masked through ``frame_valid``.
    parent_row: torch.Tensor
    n1_row: torch.Tensor
    n2_row: torch.Tensor
    frame_valid: torch.Tensor

    def _register_rows(self, n_full: int, frames: HydrogenFrames, device) -> None:
        h = np.asarray(frames.h_row, dtype=np.int64)
        if len(h) and (h.min() < 0 or h.max() >= n_full):
            raise ValueError("frames carry a hydrogen row outside the atom table")
        is_riding = np.zeros(n_full, dtype=bool)
        is_riding[h] = True
        base = np.nonzero(~is_riding)[0]
        for name in ("parent_row", "n1_row", "n2_row"):
            rows = np.asarray(getattr(frames, name), dtype=np.int64)
            if len(rows) and is_riding[rows[rows >= 0]].any():
                raise ValueError(f"{name} must reference stored rows, not riding ones")

        long = dict(
            dtype=get_int_dtype(), device=device
        )
        self.register_buffer("base_row", torch.as_tensor(base, **long))
        self.register_buffer("h_row", torch.as_tensor(h, **long))
        self.register_buffer(
            "parent_row",
            torch.as_tensor(np.asarray(frames.parent_row, dtype=np.int64), **long),
        )
        self.register_buffer(
            "n1_row", torch.as_tensor(np.asarray(frames.n1_row, dtype=np.int64), **long)
        )
        self.register_buffer(
            "n2_row", torch.as_tensor(np.asarray(frames.n2_row, dtype=np.int64), **long)
        )
        self.register_buffer(
            "frame_valid",
            torch.as_tensor(np.asarray(frames.frame_valid, dtype=bool), device=device),
        )
        self._rebuild_row_cache()

    def _rebuild_row_cache(self) -> None:
        """Derive the storage-space gathers from the row buffers."""
        base = getattr(self, "base_row", None)
        if base is None or getattr(self, "h_row", None) is None:
            self._n_full = 0
            return
        device = base.device
        n_full = int(base.numel() + self.h_row.numel())
        self._n_full = n_full
        full_to_base = torch.full(
            (max(n_full, 1),), -1, dtype=get_int_dtype(), device=device
        )
        full_to_base[base] = torch.arange(
            base.numel(), dtype=get_int_dtype(), device=device
        )
        self._parent_bidx = full_to_base[self.parent_row.clamp(min=0)].clamp(min=0)
        self._n1_bidx = full_to_base[self.n1_row.clamp(min=0)].clamp(min=0)
        self._n2_bidx = full_to_base[self.n2_row.clamp(min=0)].clamp(min=0)
        # ``cat([base, derived])[gather]`` lays the full table out in one gather.
        order = torch.empty(
            n_full, dtype=get_int_dtype(), device=device
        )
        order[base] = torch.arange(
            base.numel(), dtype=get_int_dtype(), device=device
        )
        order[self.h_row] = base.numel() + torch.arange(
            self.h_row.numel(),
            dtype=get_int_dtype(),
            device=device,
        )
        self._gather_order = order

    @property
    def n_hydrogens(self) -> int:
        """How many rows ride."""
        return 0 if getattr(self, "h_row", None) is None else int(self.h_row.numel())

    @property
    def n_base(self) -> int:
        """How many rows are stored."""
        return (
            0 if getattr(self, "base_row", None) is None else int(self.base_row.numel())
        )

    def hydrogen_frames(self) -> HydrogenFrames:
        """The frames as a CPU record, full-space rows."""
        return HydrogenFrames.from_tensors(
            self.h_row,
            self.parent_row,
            self.n1_row,
            self.n2_row,
            self.frame_valid,
            self.torsion_group,
            self.rotation_group,
        )

    def _to_full_bool(self, selection) -> torch.Tensor:
        """Any selection (bool mask, slice, indices) as a full-space bool mask."""
        if isinstance(selection, torch.Tensor) and selection.dtype == torch.bool:
            if selection.ndim != 1 or selection.shape[0] != self._n_full:
                raise ValueError(
                    f"Boolean selection shape {tuple(selection.shape)} must be "
                    f"({self._n_full},)"
                )
            return selection.to(device=self.base_row.device)
        mask = torch.zeros(self._n_full, dtype=torch.bool, device=self.base_row.device)
        mask[selection] = True
        return mask

    def _project(self, full_mask: torch.Tensor) -> torch.Tensor:
        """Storage-space view of a full-space bool mask (riding rows dropped)."""
        return full_mask.to(device=self.base_row.device, dtype=torch.bool)[
            self.base_row
        ]

    def _expand_mask(self, base_mask: torch.Tensor) -> torch.Tensor:
        """Full-space bool mask from a storage-space one; riding rows False."""
        out = torch.zeros(self._n_full, dtype=torch.bool, device=base_mask.device)
        out[self.base_row] = base_mask
        return out


class RidingXYZTensor(_DerivedRowsMixin, MixedTensor):
    """Coordinates with riding hydrogen rows derived from their parents.

    Parameters
    ----------
    initial_values : torch.Tensor, optional
        Full atom table, Cartesian Angstroms, shape ``(N, 3)``. The riding rows fix the
        local offsets; every other row is stored. None gives the empty shell that
        ``load_state_dict`` fills.
    frames : HydrogenFrames, optional
        Which rows ride and on which atoms; required with ``initial_values``.
    refinable_mask : torch.Tensor, optional
        Boolean ``(N,)`` in full atom space, or
        ``(N_base,)`` in storage space with ``mask_in_base_space``. None: every stored
        row and every orientation refinable.
    mask_in_base_space : bool, default False
        Whether ``refinable_mask`` is already in storage space, as a saved state dict
        hands it back.
    requires_grad, dtype, device, name
        As for :class:`~torchref.model.parameter_wrappers.MixedTensor`.
    eps : float
        Norm floor of the frame kernel, Angstroms.

    Notes
    -----
    ``shape`` is the full ``(N, 3)``; ``refinable_mask``, ``fixed_values`` and
    ``refinable_params`` are storage-space (``N_base`` rows). Frames whose reference
    atoms are collinear or missing fall back to a Cartesian ``parent + offset``.
    ``torsions`` holds one angle in radians per freely rotating bonded group;
    ``rotations`` holds one Cartesian rotation vector in radians per unanchored
    group (for example water). Both are :class:`MixedTensor` submodules. Selecting
    a parent or any member hydrogen selects that group's orientation. Coordinate
    assignment adopts the supplied orientation as the reference and zeros angles;
    copies and checkpoints preserve the parameter values and references.
    """

    def __init__(
        self,
        initial_values: Optional[torch.Tensor] = None,
        frames: Optional[HydrogenFrames] = None,
        refinable_mask: Optional[torch.Tensor] = None,
        *,
        mask_in_base_space: bool = False,
        requires_grad: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        name: Optional[str] = "xyz",
        eps: float = 1e-8,
    ):
        self._eps = float(eps)
        if initial_values is None:
            super().__init__(
                None, requires_grad=requires_grad, dtype=dtype, device=device, name=name
            )
            for buffer in ("base_row", "h_row", "parent_row", "n1_row", "n2_row"):
                self.register_buffer(
                    buffer,
                    torch.zeros(
                        0, dtype=get_int_dtype(), device=self.device
                    ),
                )
            self.register_buffer(
                "frame_valid", torch.zeros(0, dtype=torch.bool, device=self.device)
            )
            self.register_buffer(
                "local_offset", torch.zeros(0, 3, dtype=self.dtype, device=self.device)
            )
            self.register_buffer(
                "rigid_offset", torch.zeros(0, 3, dtype=self.dtype, device=self.device)
            )
            self.register_buffer(
                "virtual_reference",
                torch.zeros(0, 3, dtype=self.dtype, device=self.device),
            )
            self._initialize_orientations(HydrogenFrames.empty(), None)
            self.register_load_state_dict_post_hook(self._after_load)
            self._build_index_cache()
            return

        if frames is None:
            raise ValueError("frames are required with initial_values")
        if initial_values.ndim != 2 or initial_values.shape[1] != 3:
            raise ValueError(
                f"initial_values must be (N, 3), got {tuple(initial_values.shape)}"
            )
        dtype = dtype if dtype is not None else initial_values.dtype
        device = device if device is not None else initial_values.device
        values = initial_values.detach().to(dtype=dtype, device=device)
        n_full = values.shape[0]

        is_riding = np.zeros(n_full, dtype=bool)
        is_riding[np.asarray(frames.h_row, dtype=np.int64)] = True
        base_rows = torch.as_tensor(
            np.nonzero(~is_riding)[0], dtype=get_int_dtype(), device=device
        )

        if refinable_mask is None:
            base_mask = None
        elif mask_in_base_space:
            base_mask = refinable_mask.to(device=device, dtype=torch.bool)
        else:
            if refinable_mask.shape[0] != n_full:
                raise ValueError(
                    f"refinable_mask has {refinable_mask.shape[0]} rows, table has {n_full}"
                )
            base_mask = refinable_mask.to(device=device, dtype=torch.bool)[base_rows]

        super().__init__(
            values.index_select(0, base_rows),
            base_mask,
            requires_grad=requires_grad,
            dtype=dtype,
            device=device,
            name=name,
        )
        self._register_rows(n_full, frames, device)
        self.register_buffer(
            "local_offset", torch.zeros(self.n_hydrogens, 3, dtype=dtype, device=device)
        )
        self.register_buffer(
            "rigid_offset", torch.zeros(self.n_hydrogens, 3, dtype=dtype, device=device)
        )
        self.register_buffer("virtual_reference", torch.zeros_like(self.local_offset))
        full_mask = None if refinable_mask is None else refinable_mask.to(device=device)
        if full_mask is not None and mask_in_base_space:
            full_mask = self._expand_mask(full_mask)
        self._initialize_orientations(frames, full_mask)
        self.refresh_offsets(values)
        self.register_load_state_dict_post_hook(self._after_load)
        self._build_index_cache()

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _build_index_cache(self):
        super()._build_index_cache()
        self._rebuild_row_cache()
        if hasattr(self, "torsion_group"):
            self._rebuild_orientation_cache()

    def _initialize_orientations(self, frames, full_mask):
        for name in ("torsion_group", "rotation_group"):
            labels = np.asarray(getattr(frames, name))
            selected = labels >= 0
            unique, inverse = np.unique(labels[selected], return_inverse=True)
            compact = np.full(len(labels), -1, dtype=np.int64)
            compact[selected] = inverse
            for group in unique:
                members = np.flatnonzero(labels == group)
                if len(np.unique(frames.parent_row[members])) != 1:
                    raise ValueError("An orientation group must share one parent")
                if name == "torsion_group" and (
                    (frames.n1_row[members] < 0).any()
                    or len(np.unique(frames.n1_row[members])) != 1
                ):
                    raise ValueError("A torsion group must share one bonded axis")
            self.register_buffer(name, torch.as_tensor(compact, device=self.device))
        self._rebuild_orientation_cache()
        requires_grad = self.refinable_params.requires_grad
        self.torsions = MixedTensor(
            torch.zeros(
                self._torsion_parents.numel(), dtype=self.dtype, device=self.device
            ),
            requires_grad=requires_grad,
            name="hydrogen_torsions",
        )
        self.rotations = MixedTensor(
            torch.zeros(
                self._rotation_parents.numel(), 3, dtype=self.dtype, device=self.device
            ),
            requires_grad=requires_grad,
            name="hydrogen_rotations",
        )
        if full_mask is not None:
            for wrapper, mask in zip(
                (self.torsions, self.rotations), self._orientation_selection(full_mask)
            ):
                wrapper.update_refinable_mask(mask)

    def _rebuild_orientation_cache(self):
        self._virtual_frame = (self.torsion_group >= 0) & (self.n2_row < 0)
        self._has_virtual_frames = bool(self._virtual_frame.any())
        for kind in ("torsion", "rotation"):
            labels = getattr(self, kind + "_group")
            rows = (labels >= 0).nonzero(as_tuple=True)[0]
            groups = labels[rows]
            if rows.numel():
                order = torch.argsort(groups, stable=True)
                sorted_groups = groups[order]
                first = torch.cat(
                    [
                        torch.ones(1, dtype=torch.bool, device=groups.device),
                        sorted_groups[1:] != sorted_groups[:-1],
                    ]
                )
                first_rows = rows[order[first]]
                parents = self.parent_row[first_rows]
            else:
                first_rows = rows
                parents = self.parent_row[:0]
            setattr(self, "_" + kind + "_h", rows)
            setattr(self, "_" + kind + "_inverse", groups)
            setattr(self, "_" + kind + "_parents", parents)
            setattr(self, "_" + kind + "_first", first_rows)

    def _orientation_selection(self, full_mask):
        selections = []
        for kind in ("torsion", "rotation"):
            parents = getattr(self, "_" + kind + "_parents")
            rows = getattr(self, "_" + kind + "_h")
            groups = getattr(self, "_" + kind + "_inverse")
            selected = full_mask[parents].to(get_int_dtype())
            selected.index_add_(0, groups, full_mask[self.h_row[rows]].to(get_int_dtype()))
            selections.append(selected > 0)
        return selections

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Yield stored-coordinate and orientation leaves, including frozen shells."""
        return nn.Module.parameters(self, recurse=recurse)

    def optimization_parameters(self) -> list[nn.Parameter]:
        """Return coordinate, torsion and rotation leaves for the xyz optimizer."""
        return [
            self.refinable_params,
            self.torsions.refinable_params,
            self.rotations.refinable_params,
        ]

    @property
    def _storage_rows(self) -> int:
        return 0 if self.fixed_values is None else int(self.fixed_values.shape[0])

    def _storage_values(self) -> torch.Tensor:
        """The stored rows assembled, ``(N_base, 3)``."""
        return MixedTensor.forward(self)

    def evaluate(
        self,
        base_xyz: torch.Tensor,
        torsions: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full coordinates from stored ones: the pure, differentiable part of forward.

        Parameters
        ----------
        base_xyz : torch.Tensor
            Stored Cartesian coordinates in Å, shape ``(N_base, 3)``.
        torsions : torch.Tensor, optional
            Full group angles in radians, shape ``(n_torsions,)``. Defaults to
            the stored torsion parameters.
        rotations : torch.Tensor, optional
            Full group rotation vectors in radians, shape ``(n_rotations, 3)``.
            Defaults to the stored orientation parameters.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 3)``; riding rows placed from their frames. Gradients on any
            row reach ``base_xyz`` through the frame Jacobian.
        """
        if self.n_hydrogens == 0:
            return base_xyz
        local = self.local_offset
        if self._torsion_h.numel():
            angles = self.torsions.forward() if torsions is None else torsions
            cs = torch.stack((angles.cos(), angles.sin()), dim=-1)[
                self._torsion_inverse
            ]
            offsets = local[self._torsion_h]
            x, y, z = offsets.unbind(-1)
            c, sn = cs.unbind(-1)
            turned = torch.stack((x, c * y - sn * z, sn * y + c * z), dim=-1)
            local = local.index_copy(0, self._torsion_h, turned)
        p = base_xyz.index_select(0, self._parent_bidx)
        n2 = base_xyz.index_select(0, self._n2_bidx)
        if self._has_virtual_frames:
            n2 = torch.where(
                self._virtual_frame[:, None], p + self.virtual_reference, n2
            )
        h = place_local_frame(
            p,
            base_xyz.index_select(0, self._n1_bidx),
            n2,
            local,
            self.frame_valid,
            self.rigid_offset,
            eps=self._eps,
        )
        if self._rotation_h.numel():
            vectors = self.rotations.forward() if rotations is None else rotations
            offsets = rotate_vectors(
                self.rigid_offset[self._rotation_h], vectors[self._rotation_inverse]
            )
            h = h.index_copy(0, self._rotation_h, p[self._rotation_h] + offsets)
        return torch.cat([base_xyz, h], dim=0).index_select(0, self._gather_order)

    def forward(self) -> torch.Tensor:
        """The full ``(N, 3)`` table, riding rows derived from the stored ones."""
        return self.evaluate(self._storage_values())

    @property
    def shape(self):
        """Full-space shape ``(N, 3)``."""
        if self.fixed_values is None:
            return ()
        return (self._n_full, int(self.fixed_values.shape[1]))

    @property
    def base_shape(self):
        """Storage-space shape ``(N_base, 3)``."""
        return () if self.fixed_values is None else tuple(self.fixed_values.shape)

    @property
    def full_refinable_mask(self) -> torch.Tensor:
        """Refinable rows in full atom space; riding rows are never refinable."""
        return self._expand_mask(self.refinable_mask)

    # ------------------------------------------------------------------
    # Offsets
    # ------------------------------------------------------------------

    @torch.no_grad()
    def refresh_offsets(self, full_xyz: Optional[torch.Tensor] = None) -> None:
        """Re-derive every local offset from full-space coordinates.

        Parameters
        ----------
        full_xyz : torch.Tensor, optional
            Shape ``(N, 3)``; defaults to the current ``forward()``, which leaves the
            coordinates unchanged while rebasing the angular parameters. Pass the
            table after an external re-placement (a
            torsion re-scan, a hydrogen written by ``__setitem__``) to adopt it.

        Notes
        -----
        Frames that are geometrically degenerate at these coordinates are demoted to
        the rigid fallback. Rewrites buffers in place, so the forward cache is
        invalidated automatically.
        """
        if self.n_hydrogens == 0:
            return
        if full_xyz is None:
            full_xyz = self.forward()
        full_xyz = full_xyz.detach().to(dtype=self.dtype, device=self.device)
        p = full_xyz.index_select(0, self.parent_row)
        n1 = full_xyz.index_select(0, self.n1_row.clamp(min=0))
        n2 = full_xyz.index_select(0, self.n2_row.clamp(min=0))
        h = full_xyz.index_select(0, self.h_row)
        if self._has_virtual_frames:
            first = self._torsion_first[self._torsion_inverse]
            self.virtual_reference[self._torsion_h] = (h - p)[first]
            n2 = torch.where(
                self._virtual_frame[:, None], p + self.virtual_reference, n2
            )
        topological = (self.n1_row >= 0) & ((self.n2_row >= 0) | self._virtual_frame)
        valid = topological & ~frame_is_degenerate(p, n1, n2)
        local = local_frame_coordinates(p, n1, n2, h, eps=self._eps)
        self.frame_valid.copy_(valid)
        self.local_offset.copy_(
            torch.where(valid.unsqueeze(-1), local, torch.zeros_like(local))
        )
        self.rigid_offset.copy_(h - p)
        for orientation in (self.torsions, self.rotations):
            orientation.refinable_params.zero_()
            orientation.fixed_values.zero_()
            orientation.reset_forward_cache()

    def set_hydrogen_positions(self, h_xyz: torch.Tensor) -> None:
        """Adopt new positions for the riding rows, in ``h_row`` order, ``(H, 3)``."""
        full = self.forward().detach()
        full[self.h_row] = h_xyz.to(dtype=self.dtype, device=self.device)
        self.refresh_offsets(full)

    # ------------------------------------------------------------------
    # Mutation in full space
    # ------------------------------------------------------------------

    def _set_values(self, key, value: torch.Tensor) -> None:
        """Write full-space values; stored rows update, riding rows become new offsets."""
        full = self.forward().detach()
        full[key] = value
        super()._set_values(slice(None), full.index_select(0, self.base_row))
        self.refresh_offsets(full)

    def set(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        """Write ``values`` at the True rows of a full-space ``mask``."""
        if mask.ndim != 1 or mask.shape[0] != self._n_full:
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} must be ({self._n_full},)"
            )
        mask = mask.to(device=self.device, dtype=torch.bool)
        n_selected = int(mask.sum().item())
        if tuple(values.shape) != (n_selected, 3):
            raise ValueError(
                f"Values shape {tuple(values.shape)} doesn't match ({n_selected}, 3)"
            )
        self._set_values(mask, values.to(dtype=self.dtype, device=self.device))

    def update_refinable_mask(
        self, new_mask: torch.Tensor, reset_refinable: bool = False
    ):
        """Repartition coordinates and orientations with a full- or storage-space mask."""
        full_mask = new_mask.to(device=self.device)
        if new_mask.shape[0] == self._n_full:
            new_mask = self._project(new_mask)
        elif new_mask.shape[0] != self._storage_rows:
            raise ValueError(
                f"new_mask has {new_mask.shape[0]} rows; expected {self._n_full} "
                f"(atom space) or {self._storage_rows} (storage space)"
            )
        if full_mask.shape[0] != self._n_full:
            full_mask = self._expand_mask(full_mask)
        super().update_refinable_mask(new_mask, reset_refinable=reset_refinable)
        for wrapper, mask in zip(
            (self.torsions, self.rotations), self._orientation_selection(full_mask)
        ):
            wrapper.update_refinable_mask(mask, reset_refinable=reset_refinable)

    def refine(
        self, selection: Union[slice, torch.Tensor, tuple], reset_values: bool = False
    ):
        """Add a full-space selection to the refinable set."""
        full_mask = self._to_full_bool(selection)
        super().refine(self._project(full_mask), reset_values)
        for wrapper, mask in zip(
            (self.torsions, self.rotations), self._orientation_selection(full_mask)
        ):
            wrapper.refine(mask, reset_values)

    def fix(
        self,
        selection: Union[slice, torch.Tensor, tuple],
        freeze_at_current: bool = True,
    ):
        """Remove a full-space selection from the refinable set."""
        full_mask = self._to_full_bool(selection)
        super().fix(self._project(full_mask), freeze_at_current)
        for wrapper, mask in zip(
            (self.torsions, self.rotations), self._orientation_selection(full_mask)
        ):
            wrapper.fix(mask, freeze_at_current)

    def refine_all(self):
        """Make every stored row and orientation refinable."""
        self.refine(torch.ones(self._n_full, dtype=torch.bool, device=self.device))

    def fix_all(self, freeze_at_current: bool = True):
        """Fix every stored row and orientation."""
        self.fix(
            torch.ones(self._n_full, dtype=torch.bool, device=self.device),
            freeze_at_current=freeze_at_current,
        )

    def update_fixed_values(self, new_values: torch.Tensor):
        """Replace the stored rows' fixed buffer from a full-space ``(N, 3)`` table."""
        if tuple(new_values.shape) == self.shape:
            new_values = new_values.index_select(0, self.base_row.to(new_values.device))
        super().update_fixed_values(new_values)

    # ------------------------------------------------------------------
    # Conversions and copies
    # ------------------------------------------------------------------

    def to_mixed_tensor(self) -> MixedTensor:
        """Materialise as a plain per-atom wrapper; hydrogens follow their parent's mask."""
        mask = self.full_refinable_mask.clone()
        mask[self.h_row] = mask[self.parent_row]
        return MixedTensor(
            self.forward().detach(),
            mask,
            requires_grad=self.refinable_params.requires_grad,
            dtype=self.dtype,
            device=self.device,
            name=self.name,
        )

    @classmethod
    def from_mixed_tensor(
        cls, xyz: MixedTensor, frames: HydrogenFrames, **kwargs
    ) -> "RidingXYZTensor":
        """Wrap an existing per-atom coordinate tensor with riding frames."""
        return cls(
            xyz.forward().detach(),
            frames,
            refinable_mask=xyz.refinable_mask,
            requires_grad=xyz.refinable_params.requires_grad,
            dtype=xyz.dtype,
            device=xyz.device,
            name=xyz.name,
            **kwargs,
        )

    def with_values(self, full_xyz: torch.Tensor) -> "RidingXYZTensor":
        """Same frames and mask, new coordinates ``(N, 3)``."""
        result = RidingXYZTensor(
            full_xyz,
            self.hydrogen_frames(),
            refinable_mask=self.refinable_mask.clone(),
            mask_in_base_space=True,
            requires_grad=self.refinable_params.requires_grad,
            dtype=self.dtype,
            device=self.device,
            name=self.name,
            eps=self._eps,
        )
        for name in ("torsions", "rotations"):
            getattr(result, name).update_refinable_mask(
                getattr(self, name).refinable_mask.clone()
            )
        return result

    def select_rows(self, keep: torch.Tensor) -> "RidingXYZTensor":
        """The wrapper over the rows where ``keep`` is True, frames remapped.

        A hydrogen whose parent is not kept becomes an ordinary stored row.
        """
        keep_np = np.asarray(keep.detach().cpu().numpy(), dtype=bool)
        old_to_new = np.full(len(keep_np), -1, dtype=np.int64)
        old_to_new[keep_np] = np.arange(int(keep_np.sum()))
        frames = self.hydrogen_frames().remap(old_to_new)
        keep_t = torch.as_tensor(keep_np, device=self.device)
        result = RidingXYZTensor(
            self.forward().detach()[keep_t],
            frames,
            refinable_mask=self.full_refinable_mask[keep_t],
            requires_grad=self.refinable_params.requires_grad,
            dtype=self.dtype,
            device=self.device,
            name=self.name,
            eps=self._eps,
        )
        for name, labels in (
            ("torsions", frames.torsion_group),
            ("rotations", frames.rotation_group),
        ):
            retained = torch.as_tensor(
                np.unique(labels[labels >= 0]), device=self.device
            )
            getattr(result, name).update_refinable_mask(
                getattr(self, name).refinable_mask[retained]
            )
        return result

    def clone(self) -> "RidingXYZTensor":
        """Independent copy, offsets carried over bit for bit."""
        out = self.with_values(self.forward().detach())
        with torch.no_grad():
            out.local_offset.copy_(self.local_offset)
            out.rigid_offset.copy_(self.rigid_offset)
            out.frame_valid.copy_(self.frame_valid)
            out.virtual_reference.copy_(self.virtual_reference)
        out.torsions = self.torsions.copy()
        out.rotations = self.rotations.copy()
        return out

    def copy(self) -> "RidingXYZTensor":
        """Alias for :meth:`clone`."""
        return self.clone()

    def clip(self, min_value=None, max_value=None) -> "RidingXYZTensor":
        """Clip the full table; riding rows re-derive from the clipped heavy atoms."""
        full = self.forward().detach()
        if min_value is not None:
            full = torch.clamp(full, min=min_value)
        if max_value is not None:
            full = torch.clamp(full, max=max_value)
        return self.with_values(full)

    def _after_load(self, module, incompatible_keys):
        self._build_index_cache()
        self.reset_forward_cache()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + "torsion_group" not in state_dict:
            # A checkpoint without group metadata describes fixed orientations.
            hydrogen_rows = state_dict[prefix + "h_row"]
            defaults = self.state_dict()
            for name in ("torsion_group", "rotation_group"):
                defaults[name] = torch.full_like(hydrogen_rows, -1)
            defaults["virtual_reference"] = torch.zeros_like(
                state_dict[prefix + "rigid_offset"]
            )
            for name in ("torsions", "rotations"):
                shape = (0,) if name == "torsions" else (0, 3)
                empty = MixedTensor(
                    torch.empty(shape, dtype=self.dtype, device=self.device)
                )
                defaults.update(
                    {
                        name + "." + key: value
                        for key, value in empty.state_dict().items()
                    }
                )
            for name, value in defaults.items():
                if name in (
                    "torsion_group",
                    "rotation_group",
                    "virtual_reference",
                ) or name.startswith(("torsions.", "rotations.")):
                    state_dict.setdefault(prefix + name, value)
        for name, buffer in list(self._buffers.items()):
            saved = state_dict.get(prefix + name)
            if saved is not None and (buffer is None or saved.shape != buffer.shape):
                value = (
                    torch.empty_like(saved, device=self.device)
                    if buffer is None
                    else buffer.new_empty(saved.shape)
                )
                setattr(self, name, value)
        saved_params = state_dict.get(prefix + "refinable_params")
        if (
            saved_params is not None
            and saved_params.shape != self.refinable_params.shape
        ):
            self.refinable_params = nn.Parameter(
                self.refinable_params.new_empty(saved_params.shape),
                requires_grad=self.refinable_params.requires_grad,
            )
        for name in ("torsions", "rotations"):
            saved = state_dict.get(prefix + name + ".fixed_values")
            if saved is not None:
                mask = state_dict[prefix + name + ".refinable_mask"].to(self.device)
                setattr(
                    self,
                    name,
                    MixedTensor(
                        saved.to(device=self.device, dtype=self.dtype),
                        mask,
                        requires_grad=self.refinable_params.requires_grad,
                        name="hydrogen_" + name,
                    ),
                )
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __repr__(self) -> str:
        name_str = f"'{self.name}', " if self.name is not None else ""
        return (
            f"RidingXYZTensor({name_str}shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device}, refinable={self.get_refinable_count()}, "
            f"fixed={self.get_fixed_count()}, riding_h={self.n_hydrogens})"
        )


__all__ = ["RidingXYZTensor"]
