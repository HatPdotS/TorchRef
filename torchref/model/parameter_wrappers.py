"""Parametrization wrappers for the refinable crystallographic parameters.

Each wrapper is an ``nn.Module`` whose ``forward()`` rebuilds the full per-atom
tensor from a fixed buffer plus a refinable leaf, so a subset of atoms can be
frozen: :class:`MixedTensor` (xyz), :class:`PositiveMixedTensor` (isotropic B,
log-space), :class:`CholeskyMixedTensor` (anisotropic U, positive-definite) and
:class:`OccupancyTensor` (occupancies in [0, 1], with sharing groups). Assigning
into one *replaces* its ``refinable_params``, which invalidates optimizer state.
"""

import warnings
from typing import Optional, Union

import torch
from torch import nn

from torchref.config import get_float_dtype, normalize_device
from torchref.utils.caching import CachedForwardMixin
from torchref.utils.device_mixin import DeviceMixin


class _AssembleMixedTensor(torch.autograd.Function):
    """Scatter refinable values into a clone of ``fixed_values``.

    Custom op purely for the backward: the gradient w.r.t. ``refinable_params``
    is exactly ``grad_output[indices]``, one ``index_select``, instead of
    PyTorch's default ``index_put_`` backward (radix sort + atomic scatter).
    """

    @staticmethod
    def forward(ctx, refinable, fixed, indices):
        # fixed is a buffer; the .clone() is required so callers cannot
        # mutate it. index_copy_ is the canonical fast scatter for dim=0.
        result = fixed.clone()
        result.index_copy_(0, indices, refinable)
        ctx.save_for_backward(indices)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        # Only `refinable` (input 0) needs grad; fixed is a buffer and
        # indices is integer-typed.
        d_refinable = grad_output.index_select(0, indices)
        return d_refinable, None, None


class MixedTensor(DeviceMixin, CachedForwardMixin, nn.Module):
    """
    A wrapper class for tensors with mixed fixed and refinable elements.

    Keeps the fixed values in a buffer and the refinable ones in a parameter,
    reassembling the full tensor on each call. Construct it either with values or
    empty (a shell for ``load_state_dict``); ``refinable_params`` is what an
    optimizer should be given.

    Parameters
    ----------
    initial_values : torch.Tensor, optional
        Initial tensor values for all elements. Optional for empty init.
    refinable_mask : torch.Tensor, optional
        Boolean mask indicating which elements can be refined.
        If None, all elements are refinable.
    requires_grad : bool, optional
        Whether refinable parameters should have gradients. Default is True.
    dtype : torch.dtype, optional
        Data type for the tensor. Default is same as initial_values.
    device : torch.device, optional
        Device for the tensor. Default is same as initial_values.
    name : str, optional
        Optional name for this parameter (useful for debugging/logging).

    Attributes
    ----------
    refinable_mask : torch.Tensor
        Boolean mask indicating refinable elements.
    fixed_mask : torch.Tensor
        Boolean mask indicating fixed elements (inverse of refinable_mask).
    fixed_values : torch.Tensor
        Buffer containing fixed values.
    refinable_params : nn.Parameter
        Parameter containing refinable values.
    name : str or None
        Optional name for this parameter, exposed via the ``name`` property
        (useful for debugging/logging).
    """

    def __init__(
        self,
        initial_values: torch.Tensor = None,
        refinable_mask: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize a MixedTensor.

        With ``initial_values``, fully initializes; without, creates a shell ready
        for ``load_state_dict``.

        Parameters
        ----------
        initial_values : torch.Tensor, optional
            Initial tensor values for all elements. Optional for empty init.
        refinable_mask : torch.Tensor, optional
            Boolean mask indicating which elements can be refined.
            If None, all elements are refinable.
        requires_grad : bool, optional
            Whether refinable parameters should have gradients. Default is True.
        dtype : torch.dtype, optional
            Data type for the tensor. Default is same as initial_values.
        device : torch.device, optional
            Device for the tensor. Default is same as initial_values.
        name : str, optional
            Optional name for this parameter (useful for debugging/logging).
        """
        super().__init__()

        self._name = name

        if initial_values is None:
            # Honour the requested device/dtype even with no values yet: the empty
            # shell is the documented ``load_state_dict`` entry point, and a caller
            # that asked for a device should not get a CPU parameter (nor a
            # ``.device`` property that raises because every buffer is ``None``).
            device = normalize_device(device)
            dtype = dtype if dtype is not None else get_float_dtype()
            self.register_buffer("refinable_mask", None)
            self.register_buffer("fixed_mask", None)
            self.register_buffer("fixed_values", None)
            self.refinable_params = nn.Parameter(
                torch.empty(0, device=device, dtype=dtype),
                requires_grad=requires_grad,
            )
            self._has_refinable = False
            self._refinable_indices = None
            return

        if dtype is None:
            dtype = initial_values.dtype
        if device is None:
            device = initial_values.device

        initial_values = initial_values.to(dtype=dtype, device=device)

        # The mask indexes the FIRST dimension for multi-dimensional values
        # (one entry per atom), and is elementwise for 1-D values.
        if refinable_mask is None:
            if initial_values.ndim > 1:
                refinable_mask = torch.ones(
                    initial_values.shape[0], dtype=torch.bool, device=device
                )
            else:
                refinable_mask = torch.ones_like(initial_values, dtype=torch.bool)
        else:
            refinable_mask = refinable_mask.to(device=device)

        if initial_values.ndim > 1:
            if (
                refinable_mask.ndim != 1
                or refinable_mask.shape[0] != initial_values.shape[0]
            ):
                raise ValueError(
                    f"For {initial_values.ndim}D tensor with shape {initial_values.shape}, "
                    f"refinable_mask must be 1D with shape ({initial_values.shape[0]},), "
                    f"got shape {refinable_mask.shape}"
                )
        else:
            if refinable_mask.shape != initial_values.shape:
                raise ValueError(
                    f"refinable_mask shape {refinable_mask.shape} must match "
                    f"initial_values shape {initial_values.shape}"
                )

        self.register_buffer("refinable_mask", refinable_mask)
        self.register_buffer("fixed_mask", ~refinable_mask)

        # Store fixed values as a buffer. Force row-major contiguous layout:
        # pandas DataFrame selections (used for xyz / u init) hand us
        # column-major (..., 3) arrays whose stride is (1, N). Downstream
        # consumers (e.g. Triton kernels) assume the canonical stride.
        fixed_values = initial_values.clone().detach().contiguous()
        self.register_buffer("fixed_values", fixed_values)

        refinable_values = initial_values[refinable_mask].clone().detach()
        self.refinable_params = nn.Parameter(
            refinable_values, requires_grad=requires_grad
        )

        # Shape stays host-side metadata derived from ``fixed_values``, never a
        # registered buffer: ``.to()`` would drag it onto the accelerator and every
        # ``.shape`` access would then cost a device sync.

        # Pre-compute integer indices so the hot path never boolean-indexes.
        self._build_index_cache()

    def _build_index_cache(self):
        """Pre-compute integer indices from refinable_mask to avoid GPU sync."""
        if (
            self.refinable_mask is not None
            and self.refinable_mask.numel() > 0
            and self.refinable_params is not None
            and self.refinable_params.numel() > 0
        ):
            self._has_refinable = bool(self.refinable_mask.any().item())
            if self._has_refinable:
                # Keep the legacy tuple form for callers that read
                # ``_refinable_indices`` directly (used by ``__setitem__`` etc).
                self._refinable_indices = self.refinable_mask.nonzero(as_tuple=True)
                # Pre-compute a 1-D int64 index tensor for the fast path —
                # ``index_copy_`` / ``index_select`` take a 1-D LongTensor.
                self._refinable_idx_1d = self._refinable_indices[0]
                self._all_refinable = bool(
                    self.refinable_mask.numel()
                    == int(self.refinable_params.shape[0])
                )
            else:
                self._refinable_indices = None
                self._refinable_idx_1d = None
                self._all_refinable = False
        else:
            self._has_refinable = False
            self._refinable_indices = None
            self._refinable_idx_1d = None
            self._all_refinable = False

    def forward(self) -> torch.Tensor:
        """
        Reconstruct and return the full tensor.

        Three paths: all atoms refinable (``refinable_params`` straight through,
        no scatter), none refinable (a clone of ``fixed_values`` -- detached, but
        cloned so callers cannot mutate the buffer), or mixed, via
        :class:`_AssembleMixedTensor` for the cheap gather backward.
        """
        if self._all_refinable:
            # `.clone()` turns the Parameter into a plain Tensor, without which
            # CachedForwardMixin trips ``nn.Module.__setattr__`` caching a
            # Parameter under a non-Parameter slot. Its backward is the identity,
            # so the gradient reaches ``refinable_params`` directly.
            return self.refinable_params.clone()

        if not self._has_refinable or self.refinable_params.numel() == 0:
            return self.fixed_values.clone()

        return _AssembleMixedTensor.apply(
            self.refinable_params, self.fixed_values, self._refinable_idx_1d,
        )

    def __getitem__(self, key) -> torch.Tensor:
        """
        Get values at specified indices/mask from the full tensor.

        Parameters
        ----------
        key : int, slice, torch.Tensor, or tuple
            Any tensor index: integer, slice, boolean mask, integer indices, or a
            tuple for multi-dimensional indexing (``model.xyz[:, 0]``).

        Returns
        -------
        torch.Tensor
            Selected values from the full tensor, in the wrapper's public space
            (subclasses override ``_get_values``).
        """
        return self._get_values(key)

    def _get_values(self, key) -> torch.Tensor:
        """Index the assembled tensor; override to customize retrieval."""
        return self()[key]

    def __setitem__(self, key, value) -> None:
        """
        Set values at specified indices/mask, updating fixed and refinable parts.

        Parameters
        ----------
        key : int, slice, torch.Tensor, or tuple
            Any tensor index, including tuples for multi-dimensional writes
            (``model.xyz[:, 0] += 1.0``).
        value : torch.Tensor, float, or int
            Scalar (broadcast) or a tensor matching the selected region.

        Notes
        -----
        Mutates in place and **replaces** ``refinable_params`` with a new
        Parameter, so any existing optimizer state for it is stale. Values are in
        the wrapper's public space: subclasses override ``_set_values`` to
        re-encode (log for :class:`PositiveMixedTensor`, Cholesky for
        :class:`CholeskyMixedTensor`, collapsed logits for
        :class:`OccupancyTensor`).
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=self.dtype, device=self.device)
        else:
            value = value.to(dtype=self.dtype, device=self.device)

        self._set_values(key, value)

    def _set_values(self, key, value: torch.Tensor) -> None:
        """Write already-cast values into the storage; override to re-encode.

        Rebuilds ``fixed_values`` and re-extracts ``refinable_params``.
        """
        current_full = self.forward().detach()
        current_full[key] = value

        self.fixed_values = current_full.clone()

        if self.refinable_mask.any():
            new_refinable = current_full[self.refinable_mask].clone()
            self.refinable_params = nn.Parameter(
                new_refinable, requires_grad=self.refinable_params.requires_grad
            )

    def set(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Set values at positions specified by a boolean mask.

        For applying coordinate shifts, B-factor corrections and similar updates
        to a subset of atoms::

            mask = model.get_selection_mask("chain A")
            model.xyz.set(model.xyz()[mask] + shift, mask)

        Parameters
        ----------
        values : torch.Tensor
            New values, shape ``(n_selected,)`` for 1-D wrappers or
            ``(n_selected, d)`` for 2-D ones, with ``n_selected = mask.sum()``.
        mask : torch.Tensor
            Boolean mask of shape ``(n_atoms,)``; True positions are written.

        Raises
        ------
        ValueError
            If ``mask`` does not match the tensor's first dimension, or ``values``
            does not match the number of selected elements.

        Notes
        -----
        Mutates in place and replaces ``refinable_params``, invalidating existing
        optimizer state.
        """
        if mask.shape[0] != self.shape[0]:
            raise ValueError(
                f"Mask shape {mask.shape} must match tensor's first dimension {self.shape[0]}"
            )

        if mask.ndim != 1:
            raise ValueError(f"Mask must be 1D, got shape {mask.shape}")

        mask = mask.to(device=self.device, dtype=torch.bool)

        n_selected = mask.sum().item()
        expected_shape = (
            (n_selected,) if len(self.shape) == 1 else (n_selected, self.shape[1])
        )

        if values.shape != expected_shape:
            raise ValueError(
                f"Values shape {values.shape} doesn't match expected shape {expected_shape} "
                f"for {n_selected} selected elements"
            )

        # Must go through _set_values so subclass re-encoding is honoured
        # (log-space, collapsed logits, ...); inlining the base logic here would
        # write public-space values straight into the internal storage.
        self._set_values(mask, values.to(dtype=self.dtype, device=self.device))

    @property
    def shape(self):
        """Return the shape of the full tensor."""
        if self.fixed_values is None:
            return ()
        return tuple(self.fixed_values.shape)

    # ``fixed_values`` is ``None`` on the empty-shell path, so both properties
    # fall back to the (empty) parameter, which always exists and carries the
    # device/dtype the constructor was given. Without the fallback the
    # ``AttributeError`` raised by ``None.device`` is intercepted by
    # ``nn.Module.__getattr__`` and re-raised as the thoroughly misleading
    # "'MixedTensor' object has no attribute 'device'".

    @property
    def dtype(self):
        """Return the dtype of the tensor."""
        if self.fixed_values is not None:
            return self.fixed_values.dtype
        return self.refinable_params.dtype

    @property
    def device(self):
        """Return the device of the tensor."""
        if self.fixed_values is not None:
            return self.fixed_values.device
        return self.refinable_params.device

    def get_refinable_count(self) -> int:
        """Return the number of refinable parameters."""
        return self.refinable_mask.sum().item()

    def get_fixed_count(self) -> int:
        """Return the number of fixed parameters."""
        return self.fixed_mask.sum().item()

    def update_fixed_values(self, new_values: torch.Tensor):
        """Replace the whole ``fixed_values`` buffer, leaving
        ``refinable_params`` untouched -- so only the fixed positions actually
        change what ``forward()`` returns. Raises ``ValueError`` on a shape
        mismatch.
        """
        if new_values.shape != self.shape:
            raise ValueError(
                f"new_values shape {new_values.shape} must match "
                f"tensor shape {self.shape}"
            )
        self.fixed_values = new_values.to(dtype=self.dtype, device=self.device).detach()

    def _normalize_refinable_mask(self, new_mask: torch.Tensor) -> torch.Tensor:
        """Coerce an incoming mask onto this wrapper's device as ``bool``.

        Apply *before* deriving ``fixed_mask = ~new_mask``: negating the caller's
        un-migrated tensor splits the two masks across devices, which then fails
        far from here in whichever op first uses both.
        """
        return new_mask.to(device=self.device, dtype=torch.bool)

    def update_refinable_mask(
        self, new_mask: torch.Tensor, reset_refinable: bool = False
    ):
        """
        Repartition elements between refinable and fixed.

        Replaces ``refinable_params``, so any optimizer built on the old one must
        be rebuilt.

        Parameters
        ----------
        new_mask : torch.Tensor
            New boolean mask indicating refinable elements.
        reset_refinable : bool, optional
            If True, also re-baseline ``fixed_values`` to the current values.
            Default is False.
        """
        if new_mask.shape[0] != self.shape[0]:
            raise ValueError(
                f"new_mask shape {new_mask.shape} must match "
                f"tensor shape {self.shape}"
            )

        current_full = self.forward().detach()

        new_mask = self._normalize_refinable_mask(new_mask)
        self.refinable_mask = new_mask
        self.fixed_mask = ~new_mask

        if reset_refinable:
            self.fixed_values = current_full.clone()
            new_refinable = current_full[self.refinable_mask].clone()
        else:
            new_refinable = current_full[self.refinable_mask].clone()

        self.refinable_params = nn.Parameter(
            new_refinable, requires_grad=self.refinable_params.requires_grad
        )

        self._build_index_cache()

    def detach(self) -> torch.Tensor:
        """Return a detached copy of the full tensor."""
        return self.forward().detach()

    def clone(self) -> "MixedTensor":
        """Create a deep copy of this MixedTensor."""
        new_mixed = MixedTensor(
            self.forward().detach(),
            self.refinable_mask.clone(),
            requires_grad=self.refinable_params.requires_grad,
            dtype=self.dtype,
            device=self.device,
            name=self.name,
        )
        return new_mixed

    def copy(self) -> "MixedTensor":
        """Deep copy; alias for :meth:`clone`."""
        return self.clone()

    def clip(self, min_value=None, max_value=None) -> "MixedTensor":
        """Clip the full tensor values between min_value and max_value."""
        full_tensor = self.forward()
        clipped_tensor = full_tensor
        if min_value is not None:
            clipped_tensor = torch.clamp(clipped_tensor, min=min_value)
        if max_value is not None:
            clipped_tensor = torch.clamp(clipped_tensor, max=max_value)
        new_mixed = MixedTensor(
            clipped_tensor.detach(),
            self.refinable_mask.clone(),
            requires_grad=self.refinable_params.requires_grad,
            dtype=self.dtype,
            device=self.device,
            name=self.name,
        )
        return new_mixed

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Drop the legacy ``_shape`` key, which a ``strict=True`` load would
        otherwise reject as unexpected (shape now derives from ``fixed_values``).
        """
        state_dict.pop(prefix + "_shape", None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _after_device_apply(self, *args, device_changed, dtype_changed, **kwargs):
        """Rebuild the index cache after a real device/dtype change.

        ``_refinable_indices`` / ``_refinable_idx_1d`` are plain attributes holding
        tensors, so they must be regenerated on the new device. The movement hook
        (not ``to()``, which ``_apply`` bypasses; not ``reset_cache()``, which
        fires every optimizer step) is the right place.
        """
        self._build_index_cache()

    def refine(
        self, selection: Union[slice, torch.Tensor, tuple], reset_values: bool = False
    ):
        """
        Add a selection to the refinable set (union with the current mask).

        Parameters
        ----------
        selection : slice, torch.Tensor, or tuple
            Boolean mask (1-D over the first dimension), slice, integer indices,
            or index tuple.
        reset_values : bool, optional
            If True, re-baseline ``fixed_values`` to the current values first.
            Default is False.
        """
        current_full = self.forward().detach()

        # Union of the current refinable mask with the new selection.
        new_mask = self.refinable_mask.clone()

        if isinstance(selection, torch.Tensor):
            if selection.dtype == torch.bool:
                if len(self.shape) > 1:
                    if selection.shape[0] != self.shape[0] or len(selection.shape) != 1:
                        raise ValueError(
                            f"Boolean selection shape {selection.shape} must be 1D "
                            f"matching first dimension {self.shape[0]} for multi-dimensional "
                            f"tensor with shape {self.shape}"
                        )
                else:
                    if selection.shape != self.shape:
                        raise ValueError(
                            f"Boolean selection shape {selection.shape} must match "
                            f"tensor shape {self.shape}"
                        )
                new_mask |= selection.to(device=self.device)
            else:
                temp_mask = torch.zeros_like(new_mask)
                temp_mask[selection] = True
                new_mask |= temp_mask
        else:
            temp_mask = torch.zeros_like(new_mask)
            temp_mask[selection] = True
            new_mask |= temp_mask

        new_mask = self._normalize_refinable_mask(new_mask)
        self.refinable_mask = new_mask
        self.fixed_mask = ~new_mask

        if reset_values:
            self.fixed_values = current_full.clone()

        new_refinable = current_full[self.refinable_mask].clone()
        self.refinable_params = nn.Parameter(
            new_refinable, requires_grad=self.refinable_params.requires_grad
        )

        self._build_index_cache()

    def fix(
        self,
        selection: Union[slice, torch.Tensor, tuple],
        freeze_at_current: bool = True,
    ):
        """
        Remove a selection from the refinable set.

        Parameters
        ----------
        selection : slice, torch.Tensor, or tuple
            Boolean mask (1-D over the first dimension), slice, integer indices,
            or index tuple.
        freeze_at_current : bool, optional
            If True (default), freeze at the current values; if False, the
            selected elements revert to the stored ``fixed_values``.
        """
        current_full = self.forward().detach()

        # Current refinable mask minus the selection.
        new_mask = self.refinable_mask.clone()

        if isinstance(selection, torch.Tensor):
            if selection.dtype == torch.bool:
                if len(self.shape) > 1:
                    if selection.shape[0] != self.shape[0] or len(selection.shape) != 1:
                        raise ValueError(
                            f"Boolean selection shape {selection.shape} must be 1D "
                            f"matching first dimension {self.shape[0]} for multi-dimensional "
                            f"tensor with shape {self.shape}"
                        )
                else:
                    if selection.shape != self.shape:
                        raise ValueError(
                            f"Boolean selection shape {selection.shape} must match "
                            f"tensor shape {self.shape}"
                        )
                new_mask &= ~selection.to(device=self.device)
            else:
                temp_mask = torch.zeros_like(new_mask)
                temp_mask[selection] = True
                new_mask &= ~temp_mask
        else:
            temp_mask = torch.zeros_like(new_mask)
            temp_mask[selection] = True
            new_mask &= ~temp_mask

        new_mask = self._normalize_refinable_mask(new_mask)
        self.refinable_mask = new_mask
        self.fixed_mask = ~new_mask

        if freeze_at_current:
            self.fixed_values = current_full.clone()

        if self.refinable_mask.any():
            new_refinable = current_full[self.refinable_mask].clone()
            self.refinable_params = nn.Parameter(
                new_refinable, requires_grad=self.refinable_params.requires_grad
            )
        else:
            self.refinable_params = nn.Parameter(
                torch.tensor([], dtype=self.dtype, device=self.device),
                requires_grad=self.refinable_params.requires_grad,
            )

        self._build_index_cache()

    def refine_all(self):
        """Make all elements refinable."""
        all_true = torch.ones_like(self.refinable_mask)
        self.refine(all_true)

    def fix_all(self, freeze_at_current: bool = True):
        """Make all elements fixed."""
        all_true = torch.ones_like(self.refinable_mask)
        self.fix(all_true, freeze_at_current=freeze_at_current)

    @property
    def name(self) -> Optional[str]:
        """Return the name of this parameter."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Set the name of this parameter."""
        self._name = value

    def __repr__(self) -> str:
        name_str = f"'{self.name}', " if self.name is not None else ""
        return (
            f"MixedTensor({name_str}shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device}, refinable={self.get_refinable_count()}, "
            f"fixed={self.get_fixed_count()})"
        )

    def __str__(self) -> str:
        """More detailed string representation."""
        name_str = f" '{self.name}'" if self.name is not None else ""
        return (
            f"MixedTensor{name_str}:\n"
            f"  Shape: {self.shape}\n"
            f"  Dtype: {self.dtype}\n"
            f"  Device: {self.device}\n"
            f"  Refinable: {self.get_refinable_count()} / {self.refinable_mask.numel()}\n"
            f"  Fixed: {self.get_fixed_count()} / {self.refinable_mask.numel()}\n"
            f"  Requires grad: {self.refinable_params.requires_grad}"
        )

    def parameters(self):
        """Return refinable parameters for optimizer."""
        yield self.refinable_params


class PositiveMixedTensor(MixedTensor):
    """
    A MixedTensor keeping all values positive via a log-space parametrization.

    For strictly positive parameters (B-factors, scales, sigmas): storage is
    ``log(clamp(value, min=epsilon))`` and ``forward()`` returns ``exp`` of it, so
    the output is positive for any parameter value -- bounded below by ~epsilon
    rather than exactly 0 -- with smooth gradients. Everything crossing the
    public API (``set``, ``__setitem__``, ``forward``) is in NORMAL space.

    Parameters
    ----------
    initial_values : torch.Tensor, optional
        Initial tensor values in NORMAL space. Optional for empty init.
    refinable_mask : torch.Tensor, optional
        Boolean mask indicating which elements can be refined.
    requires_grad : bool, optional
        Whether refinable parameters should have gradients. Default is True.
    dtype : torch.dtype, optional
        Data type for the tensor.
    device : torch.device, optional
        Device for the tensor.
    name : str, optional
        Optional name for this parameter.
    epsilon : float, optional
        Clamp floor applied before the log; also the effective lower bound on the
        output. Default is 1e-1.
    """

    def __init__(
        self,
        initial_values: torch.Tensor = None,
        refinable_mask: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        name: Optional[str] = None,
        epsilon: float = 1e-1,
    ):
        """
        Initialize a PositiveMixedTensor.

        With ``initial_values``, fully initializes; without, creates a shell ready
        for ``load_state_dict``.

        Parameters
        ----------
        initial_values : torch.Tensor, optional
            Initial tensor values in NORMAL space. Optional for empty init.
        refinable_mask : torch.Tensor, optional
            Boolean mask indicating which elements can be refined.
        requires_grad : bool, optional
            Whether refinable parameters should have gradients. Default is True.
        dtype : torch.dtype, optional
            Data type for the tensor.
        device : torch.device, optional
            Device for the tensor.
        name : str, optional
            Optional name for this parameter.
        epsilon : float, optional
            Clamp floor applied before the log; also the effective lower bound on
            the output. Default is 1e-1. Non-positive inputs are clamped up to it
            rather than rejected.
        """
        self.epsilon = epsilon

        if initial_values is None:
            super().__init__(None, refinable_mask, requires_grad, dtype, device, name)
            return

        initial_values = torch.clamp(initial_values, min=epsilon)

        # epsilon is a plain float attribute: not a buffer, so it neither moves
        # with .to(device) nor appears in state_dict.
        log_initial_values = torch.log(initial_values)

        super().__init__(
            initial_values=log_initial_values,
            refinable_mask=refinable_mask,
            requires_grad=requires_grad,
            dtype=dtype,
            device=device,
            name=name,
        )

    def forward(self) -> torch.Tensor:
        """The full tensor in NORMAL space (``exp`` of the log-space storage)."""
        log_values = super().forward()
        return torch.exp(log_values)

    def _set_values(self, key, value: torch.Tensor) -> None:
        """Write NORMAL-space (strictly positive) values, storing their log.

        Raises ``ValueError`` on any non-positive value.
        """
        if (value <= 0).any():
            raise ValueError("All values must be positive for PositiveMixedTensor")

        current_normal = self.forward().detach()
        current_normal[key] = value

        current_log = torch.log(current_normal.clamp(min=self.epsilon))
        self.fixed_values = current_log.clone()

        if self.refinable_mask.any():
            new_refinable = current_log[self.refinable_mask].clone()
            self.refinable_params = nn.Parameter(
                new_refinable, requires_grad=self.refinable_params.requires_grad
            )

    def fix(self, mask: torch.Tensor, freeze_at_current: bool = True):
        """Freeze the masked elements, storing their value in log space.

        ``freeze_at_current=True`` (default) freezes them at their current values;
        ``False`` leaves ``fixed_values`` alone, so they revert to whatever was
        stored there.
        """
        if freeze_at_current:
            with torch.no_grad():
                current_normal = self.forward()
                current_log = torch.log(current_normal.clamp(min=self.epsilon))

            if current_log.ndim > 1:
                self.fixed_values[mask] = current_log[mask]
            else:
                self.fixed_values = torch.where(mask, current_log, self.fixed_values)

        # freeze_at_current=False: the log-space values are already written.
        super().fix(mask, freeze_at_current=False)

    def refine(self, mask: torch.Tensor):
        """Make the masked elements refinable, preserving their current value."""
        with torch.no_grad():
            current_normal = self.forward()
            current_log = torch.log(current_normal.clamp(min=self.epsilon))

        if current_log.ndim > 1:
            self.fixed_values[mask] = current_log[mask]
        else:
            self.fixed_values = torch.where(mask, current_log, self.fixed_values)

        super().refine(mask)

    def set(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Set masked positions from NORMAL-space (positive) values, e.g.::

            mask = model.get_selection_mask("name CA")
            model.b.set(torch.full((int(mask.sum()),), 30.0), mask)

        Parameters
        ----------
        values : torch.Tensor
            New values in NORMAL space, shape ``(n_selected,)`` with
            ``n_selected = mask.sum()``. Must be positive.
        mask : torch.Tensor
            Boolean mask of shape ``(n_atoms,)``; True positions are written.

        Raises
        ------
        ValueError
            If the shapes disagree or any value is non-positive.
        """
        if mask.shape[0] != self.shape[0]:
            raise ValueError(
                f"Mask shape {mask.shape} must match tensor's first dimension {self.shape[0]}"
            )

        if mask.ndim != 1:
            raise ValueError(f"Mask must be 1D, got shape {mask.shape}")

        mask = mask.to(device=self.device, dtype=torch.bool)

        n_selected = mask.sum().item()
        expected_shape = (n_selected,)

        if values.shape != expected_shape:
            raise ValueError(
                f"Values shape {values.shape} doesn't match expected shape {expected_shape} "
                f"for {n_selected} selected elements"
            )

        values = values.to(dtype=self.dtype, device=self.device)
        if (values <= 0).any():
            raise ValueError("All values must be positive for PositiveMixedTensor")

        log_values = torch.log(values.clamp(min=self.epsilon))

        # super().forward() is the LOG-space tensor; self.forward() would be normal.
        current_log = super().forward().detach()
        current_log[mask] = log_values

        self.fixed_values = current_log.clone()

        if self.refinable_mask.any():
            new_refinable = current_log[self.refinable_mask].clone()
            self.refinable_params = nn.Parameter(
                new_refinable, requires_grad=self.refinable_params.requires_grad
            )

    def get_log_values(self) -> torch.Tensor:
        """The internal log-space representation (for debugging/introspection)."""
        return super().forward()

    def update_refinable_mask(
        self, new_mask: torch.Tensor, reset_refinable: bool = False
    ):
        """Repartition refinable/fixed elements, keeping values in log space.

        ``reset_refinable`` is accepted for signature compatibility; the values
        are always re-baselined from the current state.
        """
        if new_mask.shape[0] != self.shape[0]:
            raise ValueError(
                f"new_mask shape {new_mask.shape} must match "
                f"tensor shape {self.shape}"
            )

        with torch.no_grad():
            current_normal = self.forward()
            current_log = torch.log(current_normal.clamp(min=self.epsilon))

        new_mask = self._normalize_refinable_mask(new_mask)
        self.refinable_mask = new_mask
        self.fixed_mask = ~new_mask

        self.fixed_values = current_log.clone()
        new_refinable_log = current_log[self.refinable_mask].clone()

        self.refinable_params = nn.Parameter(
            new_refinable_log, requires_grad=self.refinable_params.requires_grad
        )

        self._build_index_cache()

    def copy(self) -> "PositiveMixedTensor":
        """Deep-copy, rebuilt from NORMAL-space values so the log
        parametrization (and ``epsilon``) is preserved.
        """
        current_normal = self.forward().detach()

        new_tensor = PositiveMixedTensor(
            initial_values=current_normal,
            refinable_mask=self.refinable_mask.clone(),
            requires_grad=self.refinable_params.requires_grad,
            dtype=self.dtype,
            device=self.device,
            name=self._name,
            epsilon=self.epsilon,
        )
        return new_tensor

    def __repr__(self) -> str:
        name_str = f"'{self.name}', " if self.name is not None else ""
        return (
            f"PositiveMixedTensor({name_str}shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device}, refinable={self.get_refinable_count()}, "
            f"fixed={self.get_fixed_count()}, epsilon={self.epsilon})"
        )

    def __str__(self) -> str:
        """More detailed string representation."""
        name_str = f" '{self.name}'" if self.name is not None else ""
        return (
            f"PositiveMixedTensor{name_str}:\n"
            f"  Shape: {self.shape}\n"
            f"  Dtype: {self.dtype}\n"
            f"  Device: {self.device}\n"
            f"  Refinable: {self.get_refinable_count()} / {self.refinable_mask.numel()}\n"
            f"  Fixed: {self.get_fixed_count()} / {self.refinable_mask.numel()}\n"
            f"  Requires grad: {self.refinable_params.requires_grad}\n"
            f"  Parametrization: log space (output = exp(internal))\n"
            f"  Epsilon: {self.epsilon}"
        )


class CholeskyMixedTensor(MixedTensor):
    """A MixedTensor for anisotropic ADPs (U tensors) kept positive-definite.

    The six U components (u11, u22, u33, u12, u13, u23) are stored internally as
    the six free parameters of a lower-triangular Cholesky factor ``L``, and the
    public tensor is reconstructed as ``U = L Lᵀ``. With the diagonal of ``L``
    mapped through ``exp(x) + epsilon`` (strictly positive), ``U`` is positive-
    definite by construction for *any* value of the free parameters -- so
    unconstrained optimisation (e.g. LBFGS line search) can never drive ``U``
    indefinite. An indefinite ``U`` otherwise makes the per-atom anisotropic
    B-matrix singular, so its inverse and the Gaussian exponent blow up and the
    structure-factor FFT returns NaN. This is the anisotropic analogue of
    :class:`PositiveMixedTensor`, which keeps isotropic B positive the same way.

    Rows that are entirely non-finite (isotropic atoms carry ``U = NaN``) are
    passed through unchanged in both directions, preserving the iso/aniso split.
    The eigen-decomposition / Cholesky mapping ``U -> L`` runs only at
    construction and on freeze/unfreeze, never in the forward path, so no matrix
    factorisation enters the autograd graph.
    """

    def __init__(
        self,
        initial_values: torch.Tensor = None,
        refinable_mask: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        name: Optional[str] = None,
        epsilon: float = 1e-3,
    ):
        """Build from U-space values ``(N, 6)``, converted to Cholesky parameters.

        ``epsilon`` is the floor on the diagonal of ``L`` (``exp(x) + epsilon``),
        so it bounds the smallest eigenvalue of ``U``; other arguments are as
        :class:`MixedTensor`. Rows of NaN pass through as isotropic atoms.
        """
        self.epsilon = epsilon
        if initial_values is None:
            super().__init__(None, refinable_mask, requires_grad, dtype, device, name)
            return
        raw = self._u6_to_raw6(initial_values)
        super().__init__(
            initial_values=raw,
            refinable_mask=refinable_mask,
            requires_grad=requires_grad,
            dtype=dtype,
            device=device,
            name=name,
        )

    # ------------------------------------------------------------------
    # U (6-vector) <-> Cholesky free-parameter (6-vector) transforms.
    # Both operate on (..., 6) tensors and pass NaN rows through untouched.
    # ------------------------------------------------------------------
    @staticmethod
    def _u6_to_matrix(U: torch.Tensor) -> torch.Tensor:
        M = U.new_zeros(*U.shape[:-1], 3, 3)
        M[..., 0, 0] = U[..., 0]
        M[..., 1, 1] = U[..., 1]
        M[..., 2, 2] = U[..., 2]
        M[..., 0, 1] = M[..., 1, 0] = U[..., 3]
        M[..., 0, 2] = M[..., 2, 0] = U[..., 4]
        M[..., 1, 2] = M[..., 2, 1] = U[..., 5]
        return M

    def _u6_to_raw6(self, U: torch.Tensor) -> torch.Tensor:
        """U components -> Cholesky free parameters [log(L_ii - eps); L_offdiag]."""
        eps = self.epsilon
        finite = torch.isfinite(U).all(dim=-1)
        M = self._u6_to_matrix(torch.nan_to_num(U, nan=0.0))
        eye = torch.eye(3, dtype=M.dtype, device=M.device).expand_as(M)
        M = torch.where(finite[..., None, None], M, eye)
        # Project to positive-definite: symmetrise, clamp eigenvalues off zero.
        # No-op for well-conditioned deposited U; rescues marginally non-PD input.
        M = 0.5 * (M + M.transpose(-1, -2))
        # eigh + Cholesky forced onto the CPU: cuSolver's *batched* kernels fail
        # (CUSOLVER_STATUS_INVALID_VALUE) on the large degenerate batches an
        # isotropic ensemble produces (U ≡ 0), while LAPACK handles them. This
        # runs only at load / mask change, never per optimizer step.
        src_device = M.device
        M = M.cpu()
        w, V = torch.linalg.eigh(M)
        w = w.clamp(min=eps * eps)
        M = (V * w.unsqueeze(-2)) @ V.transpose(-1, -2)
        L = torch.linalg.cholesky(M)
        diag = torch.stack([L[..., 0, 0], L[..., 1, 1], L[..., 2, 2]], dim=-1)
        off = torch.stack([L[..., 1, 0], L[..., 2, 0], L[..., 2, 1]], dim=-1)
        raw_diag = torch.log((diag - eps).clamp(min=1e-12))  # invert exp(x)+eps
        raw = torch.cat([raw_diag, off], dim=-1).to(src_device)
        nan = torch.full_like(raw, float("nan"))
        return torch.where(finite.unsqueeze(-1), raw, nan)

    def _raw6_to_u6(self, raw: torch.Tensor) -> torch.Tensor:
        """Cholesky free parameters -> U components (U = L Lᵀ). PD by construction."""
        eps = self.epsilon
        diag, off = raw[..., :3], raw[..., 3:]
        L11 = torch.exp(diag[..., 0]) + eps
        L22 = torch.exp(diag[..., 1]) + eps
        L33 = torch.exp(diag[..., 2]) + eps
        L21, L31, L32 = off[..., 0], off[..., 1], off[..., 2]
        U11 = L11 * L11
        U22 = L21 * L21 + L22 * L22
        U33 = L31 * L31 + L32 * L32 + L33 * L33
        U12 = L21 * L11
        U13 = L31 * L11
        U23 = L31 * L21 + L32 * L22
        return torch.stack([U11, U22, U33, U12, U13, U23], dim=-1)

    def forward(self) -> torch.Tensor:
        """Return the full U tensor (positive-definite per finite row)."""
        return self._raw6_to_u6(super().forward())

    def _set_values(self, key, value: torch.Tensor) -> None:
        """Set U-space values at ``key``; stored internally as Cholesky params."""
        current = self.forward().detach()
        current[key] = value
        raw = self._u6_to_raw6(current)
        self.fixed_values = raw.clone()
        if self.refinable_mask.any():
            self.refinable_params = nn.Parameter(
                raw[self.refinable_mask].clone(),
                requires_grad=self.refinable_params.requires_grad,
            )

    def fix(self, mask: torch.Tensor, freeze_at_current: bool = True):
        """Freeze rows, storing their current value in Cholesky space."""
        if freeze_at_current:
            with torch.no_grad():
                raw = self._u6_to_raw6(self.forward())
            self.fixed_values[mask] = raw[mask]
        super().fix(mask, freeze_at_current=False)

    def refine(self, mask: torch.Tensor):
        """Make rows refinable, preserving their current value in Cholesky space."""
        with torch.no_grad():
            raw = self._u6_to_raw6(self.forward())
        self.fixed_values[mask] = raw[mask]
        super().refine(mask)

    def set(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        """Set U-space values for masked rows (converted to Cholesky internally)."""
        self._set_values(mask, values)

    def update_refinable_mask(
        self, new_mask: torch.Tensor, reset_refinable: bool = False
    ):
        """Repartition refinable/fixed elements, preserving values in U space.

        The base implementation re-stores ``forward()`` output directly, which
        would double-transform here (U written back into Cholesky-parameter
        storage); convert to Cholesky parameters first, mirroring
        :meth:`PositiveMixedTensor.update_refinable_mask`.
        """
        if new_mask.shape[0] != self.shape[0]:
            raise ValueError(
                f"new_mask shape {new_mask.shape} must match tensor shape {self.shape}"
            )
        with torch.no_grad():
            current_raw = self._u6_to_raw6(self.forward())
        new_mask = self._normalize_refinable_mask(new_mask)
        self.refinable_mask = new_mask
        self.fixed_mask = ~new_mask
        self.fixed_values = current_raw.clone()
        new_refinable = current_raw[self.refinable_mask].clone()
        self.refinable_params = nn.Parameter(
            new_refinable, requires_grad=self.refinable_params.requires_grad
        )
        self._build_index_cache()

    def copy(self) -> "CholeskyMixedTensor":
        """Deep-copy, preserving the Cholesky parametrization.

        Rebuilds from the U-space values (``__init__`` reconverts to Cholesky
        parameters), so the copy stays positive-definite rather than degrading
        to a plain unconstrained :class:`MixedTensor`.
        """
        return CholeskyMixedTensor(
            initial_values=self.forward().detach(),
            refinable_mask=self.refinable_mask.clone(),
            requires_grad=self.refinable_params.requires_grad,
            dtype=self.dtype,
            device=self.device,
            name=self._name,
            epsilon=self.epsilon,
        )


class OccupancyTensor(MixedTensor):
    """
    A MixedTensor for occupancies: bounded, shared between atoms, altloc-constrained.

    Values are kept in [0, 1] by a sigmoid reparametrization; atoms may share one
    occupancy (e.g. a whole residue), so storage is *collapsed* to one parameter
    per sharing group and expanded per atom in ``forward()``; alternative
    conformations are normalized to sum to 1.0 on the way out.

    Two index spaces meet here and callers must not mix them: masks passed to
    :meth:`freeze` / :meth:`unfreeze` / :meth:`set` are in FULL atom space, while
    ``refinable_mask`` and the counts from :meth:`get_refinable_count` are in
    COLLAPSED group space. Freezing or unfreezing any atom of a group applies to
    the whole group.

    Parameters
    ----------
    initial_values : torch.Tensor, optional
        Occupancies for ALL atoms, in [0, 1]. Omit for an empty shell.
    sharing_groups : torch.Tensor, optional
        ``(n_atoms,)`` collapsed index per atom; ``None`` = one per atom.
    altloc_groups : list of tuple, optional
        One tuple per altloc set, holding the atom indices of each conformation.
    refinable_mask : torch.Tensor, optional
        Boolean mask of refinable ATOMS (full space), collapsed internally.
    requires_grad : bool, optional
        Whether refinable parameters should have gradients. Default is True.
    dtype, device : optional
        Dtype and device for the tensors.
    name : str, optional
        Optional name for this parameter. Defaults to ``"occupancy"``.
    use_sigmoid : bool, optional
        Bound values to [0, 1] via sigmoid. Default True; False stores raw values
        and skips the range check.

    Attributes
    ----------
    expansion_mask : torch.Tensor
        Buffer mapping each atom to its collapsed index.
    collapse_counts : torch.Tensor
        Buffer holding the number of atoms per collapsed index.
    linked_occ_sizes : list
        Sorted altloc group sizes present (a plain list, not a buffer). Only set
        when ``altloc_groups`` were supplied and absent after an empty init, so
        consumers must guard with ``hasattr(self, "linked_occ_sizes")``.

    Examples
    --------
    ::

        occ = OccupancyTensor(
            initial_values=torch.tensor([1.0, 1.0, 0.7, 0.7, 0.3, 0.3]),
            sharing_groups=torch.tensor([0, 0, 1, 1, 2, 2]),
            altloc_groups=[([2, 3], [4, 5])],
        )
        result = occ()  # atoms 2-3 and 4-5 sum to 1.0
    """

    def __init__(
        self,
        initial_values: torch.Tensor = None,
        sharing_groups: Optional[torch.Tensor] = None,
        altloc_groups: Optional[list] = None,
        refinable_mask: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        name: Optional[str] = None,
        use_sigmoid: bool = True,
    ):
        """
        Initialize an OccupancyTensor with collapsed storage and altloc support.

        With ``initial_values``, fully initializes; without, creates a shell for
        ``load_state_dict``. Occupancies outside [0, 1] are clamped with a warning
        (deposited PDBs do carry them), not rejected.

        Parameters
        ----------
        initial_values : torch.Tensor, optional
            Occupancies for ALL atoms. Omit for an empty shell.
        sharing_groups : torch.Tensor, optional
            ``(n_atoms,)`` collapsed index per atom. ``tensor([0, 0, 0, 1, 1, 2])``
            = atoms 0-2 share one occupancy, 3-4 another, 5 independent.
        altloc_groups : list of tuple, optional
            ``[([10, 11], [12, 13])]`` = atoms 10-11 (conf A) and 12-13 (conf B)
            are altlocs whose occupancies sum to 1.0.
        refinable_mask : torch.Tensor, optional
            Boolean mask of refinable ATOMS (full space); any refinable atom makes
            its whole group refinable.
        requires_grad : bool, optional
            Whether refinable parameters should have gradients. Default is True.
        dtype, device : optional
            Dtype and device for the tensors.
        name : str, optional
            Optional name for this parameter. Defaults to ``"occupancy"``.
        use_sigmoid : bool, optional
            Bound values to [0, 1] via sigmoid. Default True.
        """
        self.use_sigmoid = use_sigmoid

        # Must precede any register_buffer call.
        nn.Module.__init__(self)

        self._name = name or "occupancy"

        # The empty shell is built here rather than delegated to ``MixedTensor``,
        # so the device/dtype handling is repeated: fixing only the base class
        # would leave this path on CPU.
        if initial_values is None:
            device = normalize_device(device)
            dtype = dtype if dtype is not None else get_float_dtype()
            self._full_shape = 0
            self._collapsed_shape = 0
            self.register_buffer("refinable_mask", None)
            self.register_buffer("fixed_mask", None)
            self.register_buffer("fixed_values", None)
            self.register_buffer("expansion_mask", None)
            self.refinable_params = nn.Parameter(
                torch.empty(0, device=device, dtype=dtype),
                requires_grad=requires_grad,
            )
            self._has_refinable = False
            self._refinable_indices = None
            return

        self._full_shape = initial_values.shape[0]

        if dtype is None:
            dtype = initial_values.dtype
        if device is None:
            device = initial_values.device

        initial_values = initial_values.to(device=device, dtype=dtype)

        # Clamp rather than reject: deposited PDBs carry occupancies slightly
        # outside [0, 1] (waters/ligands refined above 1.0) and must still load.
        if self.use_sigmoid:
            if torch.any(initial_values < 0) or torch.any(initial_values > 1):
                n_out = int(
                    (torch.sum(initial_values < 0) + torch.sum(initial_values > 1)).item()
                )
                warnings.warn(
                    f"{n_out} occupancy value(s) outside [0, 1]; clamping to range.",
                    stacklevel=2,
                )
                initial_values = torch.clamp(initial_values, min=0.0, max=1.0)

        self._setup_sharing_groups_and_expansion(
            initial_values, sharing_groups, altloc_groups, device
        )

        # Logit space, clamped off the asymptotes.
        if self.use_sigmoid:
            clamped_values = torch.clamp(initial_values, min=1e-6, max=1 - 1e-6)
            logit_values = torch.logit(clamped_values)
        else:
            logit_values = initial_values.clone()

        collapsed_logits = self._collapse_values_vectorized(logit_values)

        if refinable_mask is not None:
            if refinable_mask.shape[0] != self._full_shape:
                raise ValueError(
                    f"refinable_mask shape {refinable_mask.shape} must match "
                    f"initial_values shape {initial_values.shape}"
                )
            collapsed_refinable_mask = self._collapse_mask_vectorized(
                refinable_mask.to(device=device)
            )
        else:
            collapsed_refinable_mask = torch.ones(
                self._collapsed_shape, dtype=torch.bool, device=device
            )

        # All altloc members stay refinable; the sum-to-1 constraint is enforced
        # by normalization in forward(), not by freezing one of them.
        self.register_buffer("refinable_mask", collapsed_refinable_mask)
        self.register_buffer("fixed_mask", ~collapsed_refinable_mask)

        self.register_buffer("fixed_values", collapsed_logits.clone().detach())

        refinable_values = collapsed_logits[collapsed_refinable_mask].clone().detach()
        self.refinable_params = nn.Parameter(
            refinable_values, requires_grad=requires_grad
        )

        # ``_collapsed_shape`` stays a host-side int, not a buffer, for the same
        # reason as ``MixedTensor``'s shape: a buffer would cost a device sync.

        # Pre-compute integer indices so the hot path never boolean-indexes.
        self._build_index_cache()

    def _setup_sharing_groups_and_expansion(
        self,
        initial_values: torch.Tensor,
        sharing_groups: Optional[torch.Tensor],
        altloc_groups: Optional[list],
        device: torch.device,
    ):
        """Register ``expansion_mask`` / ``collapse_counts`` and the per-size
        ``linked_occ_<n>`` altloc buffers.

        All atoms of one altloc conformation must map to the same collapsed
        index; a violation raises ``AssertionError`` rather than silently
        normalizing the wrong group.
        """
        n_atoms = initial_values.shape[0]

        # Use sharing_groups directly as the expansion mask
        if sharing_groups is None:
            # No sharing - each atom maps to its own index
            expansion_mask = torch.arange(n_atoms, dtype=torch.long, device=device)
            self._collapsed_shape = n_atoms
        else:
            # Use the provided index tensor
            expansion_mask = sharing_groups.to(device=device, dtype=torch.long)
            self._collapsed_shape = expansion_mask.max().item() + 1

        self.register_buffer("expansion_mask", expansion_mask)

        # Process altloc groups: convert to collapsed indices and group by size
        # linked_occupancies[n] = tensor of shape (N_groups, n) where n is number of conformations
        linked_occupancies = {}

        if altloc_groups is not None and len(altloc_groups) > 0:
            for altloc_idx, conf_groups in enumerate(altloc_groups):
                n_conformations = len(conf_groups)
                if n_conformations < 2:
                    raise ValueError(
                        f"Altloc group {altloc_idx} must have at least 2 conformations"
                    )

                # Get collapsed indices for each conformation
                collapsed_indices = []
                for conf_atoms in conf_groups:
                    if isinstance(conf_atoms, (list, tuple)):
                        conf_atoms = torch.tensor(
                            conf_atoms, dtype=torch.long, device=device
                        )
                    else:
                        conf_atoms = conf_atoms.to(device=device, dtype=torch.long)

                    # Get collapsed index for first atom
                    collapsed_idx = expansion_mask[conf_atoms[0]].item()

                    # ASSERT: All atoms in this conformation map to the same collapsed index
                    for atom_idx in conf_atoms:
                        atom_collapsed_idx = expansion_mask[atom_idx].item()
                        if atom_collapsed_idx != collapsed_idx:
                            raise AssertionError(
                                f"Altloc group {altloc_idx}, conformation {len(collapsed_indices)}: "
                                f"atom {atom_idx} maps to collapsed index {atom_collapsed_idx}, "
                                f"but first atom maps to {collapsed_idx}. "
                                f"All atoms in a conformation must share the same collapsed index."
                            )

                    collapsed_indices.append(collapsed_idx)

                # Add to the appropriate group based on number of conformations
                if n_conformations not in linked_occupancies:
                    linked_occupancies[n_conformations] = []

                linked_occupancies[n_conformations].append(collapsed_indices)

        # Convert lists to tensors and register as buffers
        # Store as dictionary with keys like 'linked_occ_2', 'linked_occ_3', etc.
        for n_conf, groups in linked_occupancies.items():
            # Shape: (N_groups, n_conf)
            tensor = torch.tensor(groups, dtype=torch.long, device=device)
            self.register_buffer(f"linked_occ_{n_conf}", tensor)

        # Store which sizes we have
        self.linked_occ_sizes = sorted(linked_occupancies.keys())

        # Create count buffer for vectorized collapse operations
        # counts[i] = number of atoms that map to collapsed index i
        counts = torch.zeros(self._collapsed_shape, dtype=torch.long, device=device)
        counts.scatter_add_(0, expansion_mask, torch.ones_like(expansion_mask))
        self.register_buffer("collapse_counts", counts)

    def _collapse_values_vectorized(self, full_values: torch.Tensor) -> torch.Tensor:
        """Full (per-atom) -> collapsed (per-group), by scatter_add then mean.

        A group whose atoms disagree therefore collapses to their average.
        """
        collapsed_sum = torch.zeros(
            self._collapsed_shape, dtype=full_values.dtype, device=full_values.device
        )
        collapsed_sum.scatter_add_(0, self.expansion_mask, full_values)

        collapsed = collapsed_sum / self.collapse_counts.float().clamp(min=1)

        return collapsed

    def _collapse_mask_vectorized(self, full_mask: torch.Tensor) -> torch.Tensor:
        """Full -> collapsed boolean mask: a group is True if ANY of its atoms is."""
        collapsed_sum = torch.zeros(
            self._collapsed_shape, dtype=torch.float, device=full_mask.device
        )
        collapsed_sum.scatter_add_(0, self.expansion_mask, full_mask.float())

        collapsed = collapsed_sum > 0

        return collapsed

    def _collapse_values(self, full_values: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`_collapse_values_vectorized`."""
        return self._collapse_values_vectorized(full_values)

    def _collapse_mask(self, full_mask: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`_collapse_mask_vectorized`."""
        return self._collapse_mask_vectorized(full_mask)

    def _expand_values(self, collapsed_values: torch.Tensor) -> torch.Tensor:
        """Collapsed (per-group) -> full (per-atom), via ``expansion_mask``."""
        return collapsed_values[self.expansion_mask]

    def forward(self) -> torch.Tensor:
        """
        Reconstruct full occupancy tensor with sigmoid and altloc constraints.

        Sigmoid on the collapsed logits, sum-to-1 normalization within each altloc
        group, then expansion to one value per atom.

        Returns
        -------
        torch.Tensor
            Full occupancy tensor with values in [0, 1] and shape (n_atoms,).
        """
        # Integer indices, not the boolean mask: boolean indexing forces a GPU sync.
        result = self.fixed_values.clone()
        if self._has_refinable and self.refinable_params.numel() > 0:
            result[self._refinable_indices] = self.refinable_params

        if self.use_sigmoid:
            collapsed_occs = torch.sigmoid(result)
        else:
            collapsed_occs = result.clone()

        # Altloc groups: normalize to sum 1 within each, one group size at a time
        # (2-way, 3-way, ...) since each size has its own (N_groups, n_conf) buffer.
        if hasattr(self, "linked_occ_sizes") and len(self.linked_occ_sizes) > 0:
            updated_occs = collapsed_occs.clone()

            for n_conf in self.linked_occ_sizes:
                linked_indices = getattr(self, f"linked_occ_{n_conf}")
                linked_occs = collapsed_occs[linked_indices]

                sums = linked_occs.sum(dim=1, keepdim=True).clamp(min=1e-10)
                normalized_occs = linked_occs / sums

                indices_flat = linked_indices.flatten()
                occs_flat = normalized_occs.flatten()
                updated_occs[indices_flat] = occs_flat

            collapsed_occs = updated_occs

        full_occs = self._expand_values(collapsed_occs)

        return full_occs.contiguous()

    def _set_values(self, key, value: torch.Tensor) -> None:
        """Write occupancies in [0, 1], keyed in FULL atom space.

        Values are logit-transformed and collapsed into the internal storage, so
        atoms sharing a group end up at their average. Raises ``ValueError``
        outside [0, 1] when ``use_sigmoid``.
        """
        if self.use_sigmoid:
            if (value < 0).any() or (value > 1).any():
                raise ValueError("Occupancy values must be in range [0, 1]")

        current_full = self.forward().detach()
        current_full[key] = value

        if self.use_sigmoid:
            clamped_values = torch.clamp(current_full, min=1e-6, max=1 - 1e-6)
            logit_values = torch.logit(clamped_values)
        else:
            logit_values = current_full.clone()

        collapsed_logits = self._collapse_values_vectorized(logit_values)
        self.fixed_values = collapsed_logits.clone()

        if self.refinable_mask.any():
            new_refinable = collapsed_logits[self.refinable_mask].clone()
            self.refinable_params = nn.Parameter(
                new_refinable, requires_grad=self.refinable_params.requires_grad
            )

    @property
    def shape(self):
        """Return the shape of the FULL tensor (not collapsed)."""
        return (self._full_shape,)

    @property
    def collapsed_shape(self):
        """Return the shape of the collapsed internal storage."""
        return (self._collapsed_shape,)

    def clamp(
        self, min_value: float = 0.0, max_value: float = 1.0
    ) -> "OccupancyTensor":
        """
        Return a NEW OccupancyTensor with values clamped to the range (not in place).

        Parameters
        ----------
        min_value : float, optional
            Minimum occupancy value. Default is 0.0.
        max_value : float, optional
            Maximum occupancy value. Default is 1.0.

        Returns
        -------
        OccupancyTensor
            New OccupancyTensor with clamped values.
        """
        # Get current occupancy values in full space
        current_occ = self.forward().detach()

        # Clamp in occupancy space
        clamped_occ = torch.clamp(current_occ, min=min_value, max=max_value)

        # Reconstruct refinable mask in full space
        full_refinable_mask = self._expand_values(self.refinable_mask.float()).bool()

        # Create new OccupancyTensor
        new_occ = OccupancyTensor(
            initial_values=clamped_occ,
            sharing_groups=self.expansion_mask.clone(),
            refinable_mask=full_refinable_mask,
            requires_grad=self.refinable_params.requires_grad,
            dtype=self.dtype,
            device=self.device,
            name=self.name,
            use_sigmoid=self.use_sigmoid,
        )

        return new_occ

    def set_group_occupancy(self, group_idx: int, value: float):
        """
        Set the occupancy for all atoms in a specific collapsed group.

        Emits a ``UserWarning`` on every call: this path is still NumPy-based and
        not production-hardened.

        Parameters
        ----------
        group_idx : int
            Collapsed index of the group.
        value : float
            Occupancy value to set (must be in [0, 1]).

        Raises
        ------
        ValueError
            If group_idx is out of range or value is not in [0, 1].
        """
        #   to fix numpy usage

        import numpy as np
        import warnings

        warnings.warn(
            "Using numpy inside torchref/model/parameter_wrappers.py, @Peter please fix",
            UserWarning,
        )

        if group_idx < 0 or group_idx >= self._collapsed_shape:
            raise ValueError(f"Invalid group index {group_idx}")

        if value < 0 or value > 1:
            raise ValueError(f"Occupancy value must be in [0, 1], got {value}")

        # Convert value to logit space
        clamped_value = np.clip(value, 1e-6, 1 - 1e-6)
        logit_value = np.log(clamped_value / (1 - clamped_value))
        logit_tensor = torch.tensor(logit_value, dtype=self.dtype, device=self.device)

        # The group occupies collapsed_idx = group_idx (groups are first in collapsed storage)
        collapsed_idx = group_idx

        # Get current collapsed logits
        result = self.fixed_values.clone()
        result[self.refinable_mask] = self.refinable_params.data

        # Update the collapsed value for this group
        result[collapsed_idx] = logit_tensor

        # Update fixed values and refinable params
        self.fixed_values = result.clone().detach()
        if self.refinable_mask[collapsed_idx]:
            # This group is refinable, update refinable params
            self.refinable_params.data = result[self.refinable_mask].clone()

    def get_group_occupancy(self, group_idx: int) -> float:
        """
        Get the current occupancy value for a collapsed group.

        Parameters
        ----------
        group_idx : int
            Collapsed index of the group.

        Returns
        -------
        float
            Current occupancy value for the group.

        Raises
        ------
        ValueError
            If group_idx is out of range.
        """
        if group_idx < 0 or group_idx >= self._collapsed_shape:
            raise ValueError(f"Invalid group index {group_idx}")

        # Get current occupancies in full space
        occupancies = self.forward()

        # Find first atom that maps to this collapsed index
        atom_idx = (self.expansion_mask == group_idx).nonzero()[0].item()
        return occupancies[atom_idx].item()

    def freeze(self, mask: Optional[torch.Tensor] = None):
        """
        Freeze occupancy parameters, making them non-refinable.

        Parameters
        ----------
        mask : torch.Tensor, optional
            Boolean mask in FULL (uncompressed) atom space, shape ``(n_atoms,)``,
            collapsed internally. ``None`` freezes everything.

        Notes
        -----
        Freezing ANY atom of a sharing group freezes the whole group -- they share
        one compressed parameter.
        """
        if mask is None:
            mask = torch.ones(self._full_shape, dtype=torch.bool, device=self.device)
        else:
            if mask.shape[0] != self._full_shape:
                raise ValueError(
                    f"Freeze mask must have shape ({self._full_shape},) to match full atom space, "
                    f"got shape {mask.shape}"
                )
            mask = mask.to(device=self.device, dtype=torch.bool)

        collapsed_freeze_mask = self._collapse_mask_vectorized(mask)

        # Snapshot the live state (fixed buffer overwritten with refined values)
        # before repartitioning, so nothing refined so far is lost.
        current_logits = self.fixed_values.clone()
        current_logits[self.refinable_mask] = self.refinable_params.data

        new_refinable_mask = self.refinable_mask & ~collapsed_freeze_mask

        self.fixed_values = current_logits.clone().detach()

        if new_refinable_mask.any():
            new_refinable_values = current_logits[new_refinable_mask].clone().detach()
            self.refinable_params = nn.Parameter(
                new_refinable_values, requires_grad=self.refinable_params.requires_grad
            )
        else:
            self.refinable_params = nn.Parameter(
                torch.empty(0, dtype=self.dtype, device=self.device),
                requires_grad=False,
            )

        self.refinable_mask = new_refinable_mask
        self.fixed_mask = ~new_refinable_mask

        self._build_index_cache()

    def unfreeze(self, mask: Optional[torch.Tensor] = None):
        """
        Unfreeze occupancy parameters, making them refinable.

        Parameters
        ----------
        mask : torch.Tensor, optional
            Boolean mask in FULL (uncompressed) atom space, shape ``(n_atoms,)``,
            collapsed internally. ``None`` unfreezes everything.

        Notes
        -----
        Unfreezing ANY atom of a sharing group unfreezes the whole group -- they
        share one compressed parameter.
        """
        if mask is None:
            mask = torch.ones(self._full_shape, dtype=torch.bool, device=self.device)
        else:
            if mask.shape[0] != self._full_shape:
                raise ValueError(
                    f"Unfreeze mask must have shape ({self._full_shape},) to match full atom space, "
                    f"got shape {mask.shape}"
                )
            mask = mask.to(device=self.device, dtype=torch.bool)

        collapsed_unfreeze_mask = self._collapse_mask_vectorized(mask)

        # Snapshot the live state before repartitioning (see freeze()).
        current_logits = self.fixed_values.clone()
        if self.refinable_mask.any():
            current_logits[self.refinable_mask] = self.refinable_params.data

        new_refinable_mask = self.refinable_mask | collapsed_unfreeze_mask

        self.fixed_values = current_logits.clone().detach()

        if new_refinable_mask.any():
            new_refinable_values = current_logits[new_refinable_mask].clone().detach()
            self.refinable_params = nn.Parameter(
                new_refinable_values,
                requires_grad=True,  # Unfrozen parameters should have gradients
            )
        else:
            self.refinable_params = nn.Parameter(
                torch.empty(0, dtype=self.dtype, device=self.device),
                requires_grad=False,
            )

        self.refinable_mask = new_refinable_mask
        self.fixed_mask = ~new_refinable_mask

        self._build_index_cache()

    def freeze_all(self):
        """Freeze all occupancy parameters (``freeze(None)``)."""
        self.freeze(None)

    def unfreeze_all(self):
        """Unfreeze all occupancy parameters (``unfreeze(None)``)."""
        self.unfreeze(None)

    def get_refinable_atoms(self) -> torch.Tensor:
        """Boolean ``(n_atoms,)`` mask in FULL atom space of refinable atoms.

        True means the atom's group is refinable; the value is still shared with
        the rest of its group.
        """
        return self._expand_values(self.refinable_mask.float()).bool()

    def get_frozen_atoms(self) -> torch.Tensor:
        """Boolean ``(n_atoms,)`` mask in FULL atom space of frozen atoms."""
        return self._expand_values(self.fixed_mask.float()).bool()

    def get_refinable_count(self) -> int:
        """Number of refinable *groups* (COMPRESSED space), not atoms.

        For atoms, use ``get_refinable_atoms().sum()``.
        """
        return self.refinable_mask.sum().item()

    def get_fixed_count(self) -> int:
        """Number of fixed *groups* (COMPRESSED space), not atoms.

        For atoms, use ``get_frozen_atoms().sum()``.
        """
        return self.fixed_mask.sum().item()

    def update_refinable_mask(
        self, new_mask: torch.Tensor, in_compressed_space: bool = False
    ):
        """
        Replace the refinable mask outright (unlike freeze/unfreeze, which
        combine with the current one).

        Parameters
        ----------
        new_mask : torch.Tensor
            Boolean mask of refinable parameters: shape ``(n_atoms,)`` in full
            atom space, or ``(n_groups,)`` when ``in_compressed_space``.
        in_compressed_space : bool, optional
            Whether ``new_mask`` is already collapsed. Default False, i.e. full
            atom space, collapsed here (any refinable atom makes its group
            refinable).
        """
        if not in_compressed_space:
            if new_mask.shape[0] != self._full_shape:
                raise ValueError(
                    f"Mask in full atom space must have shape ({self._full_shape},), "
                    f"got shape {new_mask.shape}"
                )
            new_mask = new_mask.to(device=self.device, dtype=torch.bool)
            collapsed_mask = self._collapse_mask_vectorized(new_mask)
        else:
            if new_mask.shape[0] != self._collapsed_shape:
                raise ValueError(
                    f"Mask in compressed space must have shape ({self._collapsed_shape},), "
                    f"got shape {new_mask.shape}"
                )
            new_mask = new_mask.to(device=self.device, dtype=torch.bool)
            collapsed_mask = new_mask

        # Snapshot the live state before repartitioning (see freeze()).
        current_logits = self.fixed_values.clone()
        if self.refinable_mask.any():
            current_logits[self.refinable_mask] = self.refinable_params.data

        self.fixed_values = current_logits.clone().detach()

        if collapsed_mask.any():
            new_refinable_values = current_logits[collapsed_mask].clone().detach()
            self.refinable_params = nn.Parameter(
                new_refinable_values, requires_grad=True
            )
        else:
            self.refinable_params = nn.Parameter(
                torch.empty(0, dtype=self.dtype, device=self.device),
                requires_grad=False,
            )

        self.refinable_mask = collapsed_mask
        self.fixed_mask = ~collapsed_mask

        self._build_index_cache()

    @staticmethod
    def from_residue_groups(
        initial_values: torch.Tensor,
        pdb_dataframe,
        refinable_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "OccupancyTensor":
        """
        Create an OccupancyTensor where all atoms in each residue share occupancy.

        Residues are grouped by ``(resname, resseq, chainid, altloc)``.

        Parameters
        ----------
        initial_values : torch.Tensor
            Initial occupancy values for all atoms.
        pdb_dataframe : pandas.DataFrame
            DataFrame with PDB data (must have 'resname', 'resseq', 'chainid').
        refinable_mask : torch.Tensor, optional
            Mask for refinable atoms.
        **kwargs
            Additional arguments passed to OccupancyTensor constructor.

        Returns
        -------
        OccupancyTensor
            OccupancyTensor with residue-based sharing groups.
        """
        # Group atoms by residue
        grouped = pdb_dataframe.groupby(["resname", "resseq", "chainid", "altloc"])

        n_atoms = len(initial_values)
        sharing_groups_tensor = torch.arange(n_atoms, dtype=torch.long)
        # Singletons keep their arange ids (0..n_atoms-1); start multi-atom
        # group ids past that range so a group id can never collide with a
        # singleton's leftover arange id (the torch.unique compaction below
        # would otherwise silently merge them into one sharing group).
        collapsed_idx = n_atoms

        for (resname, resseq, chainid, altloc), group in grouped:
            indices = group["index"].tolist()
            if len(indices) > 1:  # Only create group if more than one atom
                sharing_groups_tensor[indices] = collapsed_idx
                collapsed_idx += 1

        # Compact the indices
        unique_indices = torch.unique(sharing_groups_tensor, sorted=True)
        for new_idx, old_idx in enumerate(unique_indices):
            mask = sharing_groups_tensor == old_idx
            sharing_groups_tensor[mask] = new_idx

        return OccupancyTensor(
            initial_values=initial_values,
            sharing_groups=sharing_groups_tensor,
            refinable_mask=refinable_mask,
            name="occupancy",
            **kwargs,
        )

    def copy(self) -> "OccupancyTensor":
        """
        Deep-copy, rebuilding the sharing groups, altloc groups and collapsed
        storage from the current occupancies.

        Returns
        -------
        OccupancyTensor
            New, fully independent OccupancyTensor.
        """
        current_occ = self.forward().detach()

        full_refinable_mask = self._expand_values(self.refinable_mask.float()).bool()

        # Rebuild the altloc groups from the linked_occ buffers.
        altloc_groups = []
        if hasattr(self, "linked_occ_sizes"):
            for n_conf in self.linked_occ_sizes:
                linked_indices = getattr(
                    self, f"linked_occ_{n_conf}"
                )  # shape (N_groups, n_conf)

                for group_collapsed_indices in linked_indices:
                    conf_atom_lists = []
                    for collapsed_idx in group_collapsed_indices:
                        atom_indices = (
                            (self.expansion_mask == collapsed_idx)
                            .nonzero(as_tuple=False)
                            .squeeze(-1)
                        )
                        conf_atom_lists.append(atom_indices.tolist())

                    altloc_groups.append(tuple(conf_atom_lists))

        new_tensor = OccupancyTensor(
            initial_values=current_occ,
            sharing_groups=self.expansion_mask.clone(),
            altloc_groups=altloc_groups if altloc_groups else None,
            refinable_mask=full_refinable_mask,
            requires_grad=self.refinable_params.requires_grad,
            dtype=self.dtype,
            device=self.device,
            name=self._name,
            use_sigmoid=self.use_sigmoid,
        )
        return new_tensor

    def __repr__(self) -> str:
        name_str = f"'{self.name}', " if self.name is not None else ""
        n_groups = self._collapsed_shape
        return (
            f"OccupancyTensor({name_str}shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device}, refinable={self.get_refinable_count()}, "
            f"fixed={self.get_fixed_count()}, collapsed_groups={n_groups}, "
            f"use_sigmoid={self.use_sigmoid})"
        )


class PassThroughTensor(DeviceMixin, nn.Module):
    """
    A parameter wrapper that would pass the parameter through unchanged.

    .. warning::
        **Non-functional.** ``__init__`` forwards keyword arguments to
        ``nn.Module.__init__``, which accepts none, and ``self.param`` is never
        assigned, so both construction and ``forward`` raise. A legacy stub.

    Parameters
    ----------
    initial_values : torch.Tensor
        Initial tensor values.
    requires_grad : bool, optional
        Whether the parameter requires gradients. Default is True.
    dtype : torch.dtype, optional
        Data type of the tensor.
    device : torch.device, optional
        Device to place the tensor on.
    name : str, optional
        Optional name for the parameter.
    """

    def __init__(
        self,
        initial_values: torch.Tensor,
        requires_grad: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        name: Optional[str] = None,
    ):
        """Initialize the PassThroughTensor -- see the class warning; this raises."""
        super().__init__(
            initial_values=initial_values,
            requires_grad=requires_grad,
            dtype=dtype,
            device=device,
            name=name,
        )

    def forward(self) -> torch.Tensor:
        """
        Return ``self.param`` unchanged.

        .. warning::
            Raises ``AttributeError``: ``self.param`` is never assigned. See the
            class warning.
        """
        return self.param
