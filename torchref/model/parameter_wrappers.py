"""
A file that contains wrapper classes for handling crystallographic parameters. (Occ, xyz, B, etc.)
"""

from typing import Optional, Union

import torch
from torch import nn

from torchref.utils.caching import CachedForwardMixin


class _AssembleMixedTensor(torch.autograd.Function):
    """Scatter refinable values into a clone of fixed_values, with a
    cheap (index_select) backward.

    PyTorch's default backward for ``result[idx] = refinable`` lowers to
    ``aten::_index_put_impl_``, whose backward goes through a radix-sort
    of the indices followed by an atomic scatter. Profiling on A100/1DAW
    showed this dominating the model SF backward (~370 µs/iter across
    six MixedTensors). The gradient w.r.t. ``refinable_params`` is just
    ``grad_output[indices]`` — a single ``index_select`` — so we wrap the
    assembly in a custom autograd op that returns exactly that.
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


class MixedTensor(CachedForwardMixin, nn.Module):
    """
    A wrapper class for tensors with mixed fixed and refinable elements.

    Stores a mask indicating which elements can be refined and maintains
    both fixed and refinable components separately. The full tensor is
    reconstructed on-the-fly when accessed.

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

    Examples
    --------
    Empty initialization for state_dict loading::

        mixed = MixedTensor()
        mixed.load_state_dict(torch.load('mixed.pt'))

    Full initialization with values::

        mask = torch.zeros(100, dtype=torch.bool)
        mask[20:30] = True
        initial_values = torch.randn(100)
        mixed = MixedTensor(initial_values, refinable_mask=mask, requires_grad=True)
        optimizer = torch.optim.Adam([mixed.refinable_params], lr=0.01)
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

        If initial_values is provided, fully initializes the tensor.
        If not provided (empty init), creates a shell ready for load_state_dict().

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

        # Store the name
        self._name = name

        # Empty initialization
        if initial_values is None:
            self.register_buffer("refinable_mask", None)
            self.register_buffer("fixed_mask", None)
            self.register_buffer("fixed_values", None)
            self.register_buffer("_shape", None)
            self.refinable_params = nn.Parameter(
                torch.empty(0), requires_grad=requires_grad
            )
            self._has_refinable = False
            self._refinable_indices = None
            return

        if dtype is None:
            dtype = initial_values.dtype
        if device is None:
            device = initial_values.device

        initial_values = initial_values.to(dtype=dtype, device=device)

        # Create refinable mask
        if refinable_mask is None:
            # For multi-dimensional tensors, create a mask for the first dimension
            if initial_values.ndim > 1:
                refinable_mask = torch.ones(
                    initial_values.shape[0], dtype=torch.bool, device=device
                )
            else:
                refinable_mask = torch.ones_like(initial_values, dtype=torch.bool)
        else:
            refinable_mask = refinable_mask.to(device=device)

        # Validate mask shape - it should match the first dimension for broadcasting
        if initial_values.ndim > 1:
            # For multi-dimensional tensors, mask should be 1D matching first dimension
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
            # For 1D tensors, shapes must match exactly
            if refinable_mask.shape != initial_values.shape:
                raise ValueError(
                    f"refinable_mask shape {refinable_mask.shape} must match "
                    f"initial_values shape {initial_values.shape}"
                )

        # Store the mask as a buffer (not a parameter, won't be optimized)
        self.register_buffer("refinable_mask", refinable_mask)
        self.register_buffer("fixed_mask", ~refinable_mask)

        # Store fixed values as a buffer. Force row-major contiguous layout:
        # pandas DataFrame selections (used for xyz / u init) hand us
        # column-major (..., 3) arrays whose stride is (1, N). Downstream
        # consumers (e.g. Triton kernels) assume the canonical stride.
        fixed_values = initial_values.clone().detach().contiguous()
        self.register_buffer("fixed_values", fixed_values)

        # Store refinable values as a parameter (will be optimized)
        refinable_values = initial_values[refinable_mask].clone().detach()
        self.refinable_params = nn.Parameter(
            refinable_values, requires_grad=requires_grad
        )

        # Store shape for reconstruction
        self.register_buffer("_shape", torch.tensor(initial_values.shape))

        # Pre-compute index cache to avoid boolean indexing at runtime
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

        Three fast paths, in priority order:

        1. **All atoms refinable** — return ``refinable_params`` directly
           (no clone, no scatter). Common in standard refinement where no
           atoms are frozen. Saves a full-tensor clone + an ``index_put_``
           per call and replaces the ``index_put`` backward (sort + atomic
           scatter) with a no-op identity.
        2. **No refinable atoms** — return ``fixed_values.clone()`` (the
           clone preserves the caller-must-not-mutate contract even though
           the result is detached from autograd).
        3. **Mixed** — go through :class:`_AssembleMixedTensor`, whose
           backward is a single ``index_select`` (gather) instead of
           PyTorch's default ``index_put_`` backward (radix-sort + scatter).
        """
        if self._all_refinable:
            # `.clone()` turns the Parameter into a plain Tensor (otherwise
            # the CachedForwardMixin trips ``nn.Module.__setattr__`` when
            # caching a Parameter under a non-Parameter attribute slot).
            # The clone's backward is identity, which is exactly the cheap
            # path we want — gradient flows straight to ``refinable_params``
            # without going through PyTorch's index_put backward
            # (sort + atomic scatter).
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
            Index specification. Can be:
            - int: Single element
            - slice: Range of elements (e.g., 5:10, :, ::2)
            - torch.Tensor: Boolean mask or integer indices
            - tuple: Multi-dimensional indexing

        Returns
        -------
        torch.Tensor
            Selected values from the full tensor.

        Examples
        --------
        ::

            model.b[5]           # Get B-factor for atom 5
            model.b[5:10]        # Get B-factors for atoms 5-9
            model.b[mask]        # Get B-factors where mask is True
            model.xyz[:, 0]      # Get all x-coordinates

        Notes
        -----
        Subclasses may override _get_values() to customize value retrieval.
        """
        return self._get_values(key)

    def _get_values(self, key) -> torch.Tensor:
        """
        Internal method to get values. Override in subclasses for custom behavior.

        Parameters
        ----------
        key : indexing key
            Index specification.

        Returns
        -------
        torch.Tensor
            Selected values from the full tensor.
        """
        return self()[key]

    def __setitem__(self, key, value) -> None:
        """
        Set values at specified indices/mask.

        This method updates both fixed_values and refinable_params at the
        specified positions. Supports various indexing styles including
        slices, boolean masks, and integer indices.

        Parameters
        ----------
        key : int, slice, torch.Tensor, or tuple
            Index specification. Can be:
            - int: Single element
            - slice: Range of elements (e.g., 5:10, :, ::2)
            - torch.Tensor: Boolean mask or integer indices
            - tuple: Multi-dimensional indexing
        value : torch.Tensor, float, or int
            Values to assign. Can be:
            - Scalar: Broadcast to all selected positions
            - Tensor: Must match the shape of selected region

        Examples
        --------
        ::

            model.b[:] = 30.0           # Set all B-factors to 30
            model.b[5:10] = 25.0        # Set B-factors 5-9 to 25
            model.b[mask] = new_values  # Set B-factors where mask is True
            model.xyz[mask] = new_coords  # Set coordinates for masked atoms
            model.xyz[:, 0] += 1.0      # Shift all x-coordinates (read-modify-write)

        Notes
        -----
        This method modifies the tensor in-place. The refinable_params
        parameter is replaced with a new Parameter containing the updated
        values, which may affect optimizer state.

        Subclasses may override _set_values() to customize value handling
        (e.g., PositiveMixedTensor converts to log-space).
        """
        # Convert value to tensor if needed
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=self.dtype, device=self.device)
        else:
            value = value.to(dtype=self.dtype, device=self.device)

        # Delegate to _set_values (can be overridden by subclasses)
        self._set_values(key, value)

    def _set_values(self, key, value: torch.Tensor) -> None:
        """
        Internal method to set values. Override in subclasses for custom behavior.

        Parameters
        ----------
        key : indexing key
            Index specification (already validated).
        value : torch.Tensor
            Values to assign (already converted to tensor with correct dtype/device).
        """
        # Get current full tensor
        current_full = self.forward().detach()

        # Apply the assignment
        current_full[key] = value

        # Update fixed_values buffer with the new full tensor
        self.fixed_values = current_full.clone()

        # Re-extract refinable parameters (only those in refinable_mask)
        if self.refinable_mask.any():
            new_refinable = current_full[self.refinable_mask].clone()
            self.refinable_params = nn.Parameter(
                new_refinable, requires_grad=self.refinable_params.requires_grad
            )

    def set(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Set values at positions specified by a boolean mask.

        Updates both fixed_values and refinable_params at the positions
        specified by the mask. This is useful for applying coordinate shifts,
        B-factor corrections, or any other updates to specific atoms.

        Parameters
        ----------
        values : torch.Tensor
            New values to assign. Shape must match:
            - For 1D tensors: (n_selected,) where n_selected = mask.sum()
            - For 2D tensors (e.g., xyz): (n_selected, d) where d is the
              second dimension size (e.g., 3 for coordinates)
        mask : torch.Tensor
            Boolean mask of shape (n_atoms,) indicating which elements to update.
            True positions will receive the new values.

        Raises
        ------
        ValueError
            If mask shape doesn't match tensor's first dimension, or if
            values shape doesn't match the number of selected elements.

        Examples
        --------
        ::

            # Update coordinates for selected atoms
            mask = model.get_selection_mask("chain A")
            new_coords = original_coords[mask] + shift
            model.xyz.set(new_coords, mask)

            # Update B-factors for specific residues
            mask = model.get_selection_mask("resseq 10:20")
            new_b = torch.ones(mask.sum()) * 30.0
            model.b.set(new_b, mask)

        Notes
        -----
        This method modifies the tensor in-place. The refinable_params
        parameter is replaced with a new Parameter containing the updated
        values, which may affect optimizer state.
        """
        # Validate mask shape
        if mask.shape[0] != self.shape[0]:
            raise ValueError(
                f"Mask shape {mask.shape} must match tensor's first dimension {self.shape[0]}"
            )

        if mask.ndim != 1:
            raise ValueError(f"Mask must be 1D, got shape {mask.shape}")

        # Move mask to correct device
        mask = mask.to(device=self.device, dtype=torch.bool)

        # Validate values shape
        n_selected = mask.sum().item()
        expected_shape = (
            (n_selected,) if len(self.shape) == 1 else (n_selected, self.shape[1])
        )

        if values.shape != expected_shape:
            raise ValueError(
                f"Values shape {values.shape} doesn't match expected shape {expected_shape} "
                f"for {n_selected} selected elements"
            )

        # Get current full tensor
        current_full = self.forward().detach()

        # Update the full tensor at masked positions
        current_full[mask] = values.to(dtype=self.dtype, device=self.device)

        # Update fixed_values buffer with the new full tensor
        self.fixed_values = current_full.clone()

        # Re-extract refinable parameters (only those in refinable_mask)
        if self.refinable_mask.any():
            new_refinable = current_full[self.refinable_mask].clone()
            self.refinable_params = nn.Parameter(
                new_refinable, requires_grad=self.refinable_params.requires_grad
            )

    @property
    def shape(self):
        """Return the shape of the full tensor."""
        return tuple(self._shape.tolist())

    @property
    def dtype(self):
        """Return the dtype of the tensor."""
        return self.fixed_values.dtype

    @property
    def device(self):
        """Return the device of the tensor."""
        return self.fixed_values.device

    def get_refinable_count(self) -> int:
        """Return the number of refinable parameters."""
        return self.refinable_mask.sum().item()

    def get_fixed_count(self) -> int:
        """Return the number of fixed parameters."""
        return self.fixed_mask.sum().item()

    def update_fixed_values(self, new_values: torch.Tensor):
        """
        Update the fixed values (does not affect refinable parameters).

        Parameters
        ----------
        new_values : torch.Tensor
            New tensor values. Only fixed positions will be updated.

        Raises
        ------
        ValueError
            If new_values shape doesn't match tensor shape.
        """
        if new_values.shape != self.shape:
            raise ValueError(
                f"new_values shape {new_values.shape} must match "
                f"tensor shape {self.shape}"
            )
        self.fixed_values = new_values.to(dtype=self.dtype, device=self.device).detach()

    def update_refinable_mask(
        self, new_mask: torch.Tensor, reset_refinable: bool = False
    ):
        """
        Update which elements are refinable.

        This is an advanced operation that modifies the refinable/fixed split.

        Parameters
        ----------
        new_mask : torch.Tensor
            New boolean mask indicating refinable elements.
        reset_refinable : bool, optional
            If True, reset refinable parameters to current fixed values.
            If False, keep existing refinable parameter values where possible.
            Default is False.
        """
        if new_mask.shape[0] != self.shape[0]:
            raise ValueError(
                f"new_mask shape {new_mask.shape} must match "
                f"tensor shape {self.shape}"
            )

        current_full = self.forward().detach()

        self.refinable_mask = new_mask.to(device=self.device)
        self.fixed_mask = ~new_mask

        if reset_refinable:
            # Reset refinable params to current fixed values
            self.fixed_values = current_full.clone()
            new_refinable = current_full[self.refinable_mask].clone()
        else:
            # Try to preserve existing refinable values where masks overlap
            new_refinable = current_full[self.refinable_mask].clone()

        # Replace the parameter
        self.refinable_params = nn.Parameter(
            new_refinable, requires_grad=self.refinable_params.requires_grad
        )

        # Rebuild index cache after mask change
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
        """
        Create a deep copy of this MixedTensor.

        Creates a complete independent copy with all buffers and parameters.
        Alias for clone().

        Returns
        -------
        MixedTensor
            New MixedTensor instance with copied data.
        """
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

    def to(self, device=None, dtype=None) -> "MixedTensor":
        """
        Move tensor to a different device or dtype.

        Parameters
        ----------
        device : torch.device or str, optional
            Target device.
        dtype : torch.dtype, optional
            Target data type.

        Returns
        -------
        MixedTensor
            Self for method chaining.
        """
        # Explicitly move all internal tensors to ensure proper device placement
        if device is not None or dtype is not None:
            # Move buffers
            if self.fixed_values is not None:
                self.fixed_values = self.fixed_values.to(device=device, dtype=dtype)
            if self.refinable_mask is not None:
                self.refinable_mask = self.refinable_mask.to(device=device)
            if self.fixed_mask is not None:
                self.fixed_mask = self.fixed_mask.to(device=device)
            if self._shape is not None:
                self._shape = self._shape.to(device=device)

            # Move refinable parameters
            if self.refinable_params is not None and self.refinable_params.numel() > 0:
                new_param = self.refinable_params.to(device=device, dtype=dtype)
                self.refinable_params = nn.Parameter(
                    new_param, requires_grad=self.refinable_params.requires_grad
                )

        result = super().to(device=device, dtype=dtype)
        # Rebuild index cache after device move
        self._build_index_cache()
        return result

    def parameters(self):
        parameter = super().parameters()
        parameter_valid = [param for param in parameter if param.numel() > 0]
        yield from parameter_valid

    def refine(
        self, selection: Union[slice, torch.Tensor, tuple], reset_values: bool = False
    ):
        """
        Make a selection of the tensor refinable.

        Parameters
        ----------
        selection : slice, torch.Tensor, or tuple
            Selection indicating which elements should become refinable. Can be:
            - Boolean tensor of same shape as the full tensor
            - Slice object (e.g., slice(10, 20))
            - Tuple of indices for multidimensional tensors
            - Integer indices
        reset_values : bool, optional
            If True, reset the selected elements to their current fixed values
            before making them refinable. Default is False.

        Examples
        --------
        ::

            mixed.refine(slice(10, 20))  # Make elements 10-19 refinable
            mixed.refine(mask)  # Make elements where mask is True refinable
        """
        # Get current full tensor
        current_full = self.forward().detach()

        # Create a new mask that combines old refinable + new selection
        new_mask = self.refinable_mask.clone()

        if isinstance(selection, torch.Tensor):
            if selection.dtype == torch.bool:
                # For multi-dimensional tensors, mask should match first dimension
                # For 1D tensors, must match exactly
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
                # Integer indices
                temp_mask = torch.zeros_like(new_mask)
                temp_mask[selection] = True
                new_mask |= temp_mask
        else:
            # Handle slice or tuple indices
            temp_mask = torch.zeros_like(new_mask)
            temp_mask[selection] = True
            new_mask |= temp_mask

        # Update the mask
        self.refinable_mask = new_mask
        self.fixed_mask = ~new_mask

        # Update values
        if reset_values:
            self.fixed_values = current_full.clone()

        # Reconstruct refinable parameters with new selection
        new_refinable = current_full[self.refinable_mask].clone()
        self.refinable_params = nn.Parameter(
            new_refinable, requires_grad=self.refinable_params.requires_grad
        )

        # Rebuild index cache after mask change
        self._build_index_cache()

    def fix(
        self,
        selection: Union[slice, torch.Tensor, tuple],
        freeze_at_current: bool = True,
    ):
        """
        Make a selection of the tensor fixed (non-refinable).

        Parameters
        ----------
        selection : slice, torch.Tensor, or tuple
            Selection indicating which elements should become fixed. Can be:
            - Boolean tensor of same shape as the full tensor
            - Slice object (e.g., slice(10, 20))
            - Tuple of indices for multidimensional tensors
            - Integer indices
        freeze_at_current : bool, optional
            If True (default), freeze the selected elements at their current values.
            If False, they revert to the original fixed values.

        Examples
        --------
        ::

            mixed.fix(slice(10, 20))  # Fix elements 10-19
            mixed.fix(mask)  # Fix elements where mask is True
        """
        # Get current full tensor
        current_full = self.forward().detach()

        # Create a new mask that removes the selection from refinable
        new_mask = self.refinable_mask.clone()

        if isinstance(selection, torch.Tensor):
            if selection.dtype == torch.bool:
                # For multi-dimensional tensors, mask should match first dimension
                # For 1D tensors, must match exactly
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
                # Integer indices
                temp_mask = torch.zeros_like(new_mask)
                temp_mask[selection] = True
                new_mask &= ~temp_mask
        else:
            # Handle slice or tuple indices
            temp_mask = torch.zeros_like(new_mask)
            temp_mask[selection] = True
            new_mask &= ~temp_mask

        # Update the mask
        self.refinable_mask = new_mask
        self.fixed_mask = ~new_mask

        # Update fixed values
        if freeze_at_current:
            self.fixed_values = current_full.clone()

        # Reconstruct refinable parameters without the fixed selection
        if self.refinable_mask.any():
            new_refinable = current_full[self.refinable_mask].clone()
            self.refinable_params = nn.Parameter(
                new_refinable, requires_grad=self.refinable_params.requires_grad
            )
        else:
            # All fixed, create empty parameter
            self.refinable_params = nn.Parameter(
                torch.tensor([], dtype=self.dtype, device=self.device),
                requires_grad=self.refinable_params.requires_grad,
            )

        # Rebuild index cache after mask change
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
    A MixedTensor subclass ensuring all values are positive via log-space parametrization.

    Useful for parameters that must be strictly positive (e.g., B-factors,
    scale factors, sigma values). Values are stored as logarithms internally
    and converted to normal space via exp() when accessed.

    Reparametrization::

        internal_value = log(desired_value)
        output_value = exp(internal_value)

    This ensures output_value > 0 always, with smooth gradient flow.

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
        Small value to add before taking log to avoid log(0). Default is 1e-1.

    Examples
    --------
    Empty initialization for state_dict loading::

        b = PositiveMixedTensor()
        b.load_state_dict(torch.load('b_factors.pt'))

    Full initialization with values::

        initial_b = torch.tensor([20.0, 30.0, 15.0])
        b = PositiveMixedTensor(initial_b)
        output = b()  # Returns exp(log_b) = positive values
        assert (b() > 0).all()
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

        If initial_values is provided, fully initializes the tensor.
        If not provided (empty init), creates a shell ready for load_state_dict().

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
            Small value to add before taking log to avoid log(0). Default is 1e-1.

        Raises
        ------
        ValueError
            If any initial values are not positive.
        """
        # Store epsilon
        self.epsilon = epsilon

        # Empty initialization
        if initial_values is None:
            super().__init__(None, refinable_mask, requires_grad, dtype, device, name)
            return

        # Full initialization - clip initial values to be positive
        initial_values = torch.clamp(initial_values, min=epsilon)

        # Store epsilon as buffer (not parameter)
        self.epsilon = epsilon

        # Convert initial values to log space
        log_initial_values = torch.log(initial_values.clamp(min=epsilon))

        # Initialize parent class with log-space values
        super().__init__(
            initial_values=log_initial_values,
            refinable_mask=refinable_mask,
            requires_grad=requires_grad,
            dtype=dtype,
            device=device,
            name=name,
        )

    def forward(self) -> torch.Tensor:
        """
        Return the full tensor in NORMAL space.

        Applies exponential transformation to the log-space values.

        Returns
        -------
        torch.Tensor
            Tensor with positive values.
        """
        # Get log-space values from parent
        log_values = super().forward()

        # Convert to normal space via exp
        return torch.exp(log_values)

    def _set_values(self, key, value: torch.Tensor) -> None:
        """
        Internal method to set values with log-space conversion.

        Values are provided in NORMAL space (positive values) and
        automatically converted to log-space for internal storage.

        Parameters
        ----------
        key : indexing key
            Index specification (already validated).
        value : torch.Tensor
            Values to assign in NORMAL space. Must be positive.

        Raises
        ------
        ValueError
            If any values are not positive.
        """
        # Ensure values are positive
        if (value <= 0).any():
            raise ValueError("All values must be positive for PositiveMixedTensor")

        # Get current full tensor in NORMAL space
        current_normal = self.forward().detach()

        # Update in normal space first
        current_normal[key] = value

        # Convert entire tensor to log space
        current_log = torch.log(current_normal.clamp(min=self.epsilon))

        # Update fixed_values buffer with the new log values
        self.fixed_values = current_log.clone()

        # Re-extract refinable parameters (in log space)
        if self.refinable_mask.any():
            new_refinable = current_log[self.refinable_mask].clone()
            self.refinable_params = nn.Parameter(
                new_refinable, requires_grad=self.refinable_params.requires_grad
            )

    def fix(self, mask: torch.Tensor, freeze_at_current: bool = True):
        """
        Fix (freeze) specific elements.

        Converts current normal-space values to log space for storage.

        Parameters
        ----------
        mask : torch.Tensor
            Boolean mask indicating which elements to fix.
        freeze_at_current : bool, optional
            If True, freeze at current values. Default is True.
        """
        if freeze_at_current:
            # Get current log-space values WITHOUT creating computation graph
            with torch.no_grad():
                current_normal = self.forward()
                current_log = torch.log(current_normal.clamp(min=self.epsilon))

            # Update fixed_values with current log-space values
            if current_log.ndim > 1:
                self.fixed_values[mask] = current_log[mask]
            else:
                self.fixed_values = torch.where(mask, current_log, self.fixed_values)

        # Call parent's fix method with freeze_at_current=False since we already updated
        super().fix(mask, freeze_at_current=False)

    def refine(self, mask: torch.Tensor):
        """
        Make specific elements refinable.

        Preserves current log-space values.

        Parameters
        ----------
        mask : torch.Tensor
            Boolean mask indicating which elements to make refinable.
        """
        # Get current log-space values WITHOUT creating computation graph
        with torch.no_grad():
            current_normal = self.forward()
            current_log = torch.log(current_normal.clamp(min=self.epsilon))

        # Update fixed_values with current log-space values
        if current_log.ndim > 1:
            self.fixed_values[mask] = current_log[mask]
        else:
            self.fixed_values = torch.where(mask, current_log, self.fixed_values)

        # Call parent's refine method
        super().refine(mask)

    def set(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Set values at positions specified by a boolean mask.

        Values are provided in NORMAL space (e.g., actual B-factors) and
        automatically converted to log-space for internal storage.

        Parameters
        ----------
        values : torch.Tensor
            New values to assign in NORMAL space (positive values).
            Shape must be (n_selected,) where n_selected = mask.sum().
        mask : torch.Tensor
            Boolean mask of shape (n_atoms,) indicating which elements to update.
            True positions will receive the new values.

        Raises
        ------
        ValueError
            If mask shape doesn't match tensor's first dimension, if
            values shape doesn't match the number of selected elements,
            or if any values are not positive.

        Examples
        --------
        ::

            # Update B-factors for selected atoms
            mask = model.get_selection_mask("name CA")
            new_b = torch.ones(mask.sum()) * 30.0  # Set CA B-factors to 30
            model.b.set(new_b, mask)

        Notes
        -----
        This method modifies the tensor in-place. Values are automatically
        converted to log-space internally to maintain the positivity constraint.
        """
        # Validate mask shape
        if mask.shape[0] != self.shape[0]:
            raise ValueError(
                f"Mask shape {mask.shape} must match tensor's first dimension {self.shape[0]}"
            )

        if mask.ndim != 1:
            raise ValueError(f"Mask must be 1D, got shape {mask.shape}")

        # Move mask to correct device
        mask = mask.to(device=self.device, dtype=torch.bool)

        # Validate values shape
        n_selected = mask.sum().item()
        expected_shape = (n_selected,)

        if values.shape != expected_shape:
            raise ValueError(
                f"Values shape {values.shape} doesn't match expected shape {expected_shape} "
                f"for {n_selected} selected elements"
            )

        # Ensure values are positive
        values = values.to(dtype=self.dtype, device=self.device)
        if (values <= 0).any():
            raise ValueError("All values must be positive for PositiveMixedTensor")

        # Convert values to log space
        log_values = torch.log(values.clamp(min=self.epsilon))

        # Get current full tensor in LOG space
        current_log = super().forward().detach()

        # Update the log tensor at masked positions
        current_log[mask] = log_values

        # Update fixed_values buffer with the new log values
        self.fixed_values = current_log.clone()

        # Re-extract refinable parameters (in log space)
        if self.refinable_mask.any():
            new_refinable = current_log[self.refinable_mask].clone()
            self.refinable_params = nn.Parameter(
                new_refinable, requires_grad=self.refinable_params.requires_grad
            )

    def get_log_values(self) -> torch.Tensor:
        """
        Return the internal log-space representation.

        Useful for debugging or when direct access to the parametrization space
        is needed.

        Returns
        -------
        torch.Tensor
            Tensor with log-space values.
        """
        return super().forward()

    def update_refinable_mask(
        self, new_mask: torch.Tensor, reset_refinable: bool = False
    ):
        """
        Update which elements are refinable.

        Properly handles log-space conversion.

        Parameters
        ----------
        new_mask : torch.Tensor
            New boolean mask indicating refinable elements.
        reset_refinable : bool, optional
            If True, reset refinable parameters to current fixed values.
            If False, keep existing refinable parameter values where possible.
            Default is False.
        """
        if new_mask.shape[0] != self.shape[0]:
            raise ValueError(
                f"new_mask shape {new_mask.shape} must match "
                f"tensor shape {self.shape}"
            )

        # Get current values in NORMAL space
        with torch.no_grad():
            current_normal = self.forward()
            # Convert to log space
            current_log = torch.log(current_normal.clamp(min=self.epsilon))

        self.refinable_mask = new_mask.to(device=self.device)
        self.fixed_mask = ~new_mask

        # Update fixed_values with log-space values
        self.fixed_values = current_log.clone()

        # Extract refinable portion (in log space)
        new_refinable_log = current_log[self.refinable_mask].clone()

        # Replace the parameter with log-space values
        self.refinable_params = nn.Parameter(
            new_refinable_log, requires_grad=self.refinable_params.requires_grad
        )

        # Rebuild index cache after mask change
        self._build_index_cache()

    def copy(self) -> "PositiveMixedTensor":
        """
        Create a deep copy of this PositiveMixedTensor.

        Properly handles the log-space reparametrization.

        Returns
        -------
        PositiveMixedTensor
            New PositiveMixedTensor instance with copied data.
        """
        # Get current values in normal space
        current_normal = self.forward().detach()

        # Create new instance (will convert to log space internally)
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


class OccupancyTensor(MixedTensor):
    """
    A specialized MixedTensor for handling occupancy parameters in crystallographic refinement.

    Handles specific constraints and requirements for occupancy including value bounds
    [0, 1] via sigmoid reparameterization, atom sharing, and alternative conformation
    sum-to-1 constraints.

    Features:
    - Values bounded between 0 and 1 using sigmoid reparameterization
    - Atoms can share occupancies (e.g., all atoms in a residue)
    - Alternative conformations automatically sum to 1.0 via normalization
    - Memory-efficient collapsed storage (one parameter per sharing group)
    - Fully vectorized collapse/expand operations

    Parameters
    ----------
    initial_values : torch.Tensor, optional
        Initial occupancy values for ALL atoms (should be in [0, 1]).
        Optional for empty init.
    sharing_groups : torch.Tensor, optional
        Tensor of shape (n_atoms,) where each value is the collapsed index
        for that atom. If None, each atom has independent occupancy.
    altloc_groups : list of tuple, optional
        List of tuples of atom index lists representing alternative
        conformations. Each tuple contains the atom indices for each
        conformation.
    refinable_mask : torch.Tensor, optional
        Boolean mask for which ATOMS can be refined (in full tensor space).
    requires_grad : bool, optional
        Whether refinable parameters should have gradients. Default is True.
    dtype : torch.dtype, optional
        Data type for the tensor.
    device : torch.device, optional
        Device for the tensor.
    name : str, optional
        Optional name for this parameter.
    use_sigmoid : bool, optional
        If True, use sigmoid parameterization to bound values to [0,1].
        Default is True.

    Attributes
    ----------
    expansion_mask : torch.Tensor
        Maps atoms to collapsed indices.
    linked_occ_sizes : list
        List of altloc group sizes present.
    collapse_counts : torch.Tensor
        Count of atoms per collapsed index.

    Examples
    --------
    ::

        sharing_groups = torch.tensor([0, 0, 1, 1, 2, 2])
        occ = OccupancyTensor(
            initial_values=torch.tensor([1.0, 1.0, 0.7, 0.7, 0.3, 0.3]),
            sharing_groups=sharing_groups,
            altloc_groups=[([2, 3], [4, 5])],
        )
        result = occ()  # Atoms 2-3 and 4-5 will sum to 1.0
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

        If initial_values is provided, fully initializes the tensor.
        If not provided (empty init), creates a shell ready for load_state_dict().

        Parameters
        ----------
        initial_values : torch.Tensor, optional
            Initial occupancy values for ALL atoms (should be in [0, 1]).
            Optional for empty init.
        sharing_groups : torch.Tensor, optional
            Tensor of shape (n_atoms,) where each value is the collapsed index
            for that atom. If None, each atom has independent occupancy.
            Example: tensor([0, 0, 0, 1, 1, 2]) means atoms 0,1,2 share one
            occupancy, atoms 3,4 share another, and atom 5 is independent.
        altloc_groups : list of tuple, optional
            List of tuples of atom index lists representing alternative
            conformations. Example: [([10,11], [12,13])] means atoms 10,11
            (conf A) and 12,13 (conf B) are altlocs that sum to 1.0.
        refinable_mask : torch.Tensor, optional
            Boolean mask for which ATOMS can be refined (in full tensor space).
            If any atom in a group is refinable, the entire group becomes refinable.
        requires_grad : bool, optional
            Whether refinable parameters should have gradients. Default is True.
        dtype : torch.dtype, optional
            Data type for the tensor.
        device : torch.device, optional
            Device for the tensor.
        name : str, optional
            Optional name for this parameter.
        use_sigmoid : bool, optional
            If True, use sigmoid parameterization to bound values to [0,1].
            Default is True.
        """
        # Store configuration
        self.use_sigmoid = use_sigmoid

        # Initialize Module first (required before register_buffer)
        nn.Module.__init__(self)

        self._name = name or "occupancy"

        # Empty initialization
        if initial_values is None:
            self._full_shape = 0
            self._collapsed_shape = 0
            self.register_buffer("refinable_mask", None)
            self.register_buffer("fixed_mask", None)
            self.register_buffer("fixed_values", None)
            self.register_buffer("_shape", None)
            self.register_buffer("expansion_mask", None)
            self.refinable_params = nn.Parameter(
                torch.empty(0), requires_grad=requires_grad
            )
            self._has_refinable = False
            self._refinable_indices = None
            return

        # Full initialization
        self._full_shape = initial_values.shape[0]

        if dtype is None:
            dtype = initial_values.dtype
        if device is None:
            device = initial_values.device

        # Ensure initial_values is on the correct device and dtype
        initial_values = initial_values.to(device=device, dtype=dtype)

        # Validate initial values are in valid range
        if self.use_sigmoid:
            if torch.any(initial_values < 0) or torch.any(initial_values > 1):
                raise ValueError("Initial occupancy values must be in range [0, 1]")

        # Process sharing groups and altlocs, create expansion mask
        self._setup_sharing_groups_and_expansion(
            initial_values, sharing_groups, altloc_groups, device
        )

        # Convert initial occupancies to logit space (full space)
        if self.use_sigmoid:
            clamped_values = torch.clamp(initial_values, min=1e-6, max=1 - 1e-6)
            logit_values = torch.logit(clamped_values)
        else:
            logit_values = initial_values.clone()

        # Collapse logit values using vectorized operation
        collapsed_logits = self._collapse_values_vectorized(logit_values)

        # Handle refinable mask - collapse it too
        if refinable_mask is not None:
            if refinable_mask.shape[0] != self._full_shape:
                raise ValueError(
                    f"refinable_mask shape {refinable_mask.shape} must match "
                    f"initial_values shape {initial_values.shape}"
                )
            # Collapse the refinable mask
            collapsed_refinable_mask = self._collapse_mask_vectorized(
                refinable_mask.to(device=device)
            )
        else:
            collapsed_refinable_mask = torch.ones(
                self._collapsed_shape, dtype=torch.bool, device=device
            )

        # Note: With the new normalization approach, all altloc members are refinable
        # The sum-to-1 constraint is enforced during forward() via normalization

        # Store masks as buffers
        self.register_buffer("refinable_mask", collapsed_refinable_mask)
        self.register_buffer("fixed_mask", ~collapsed_refinable_mask)

        # Store fixed values as buffer (collapsed)
        self.register_buffer("fixed_values", collapsed_logits.clone().detach())

        # Store refinable values as parameter (collapsed, excluding placeholders)
        refinable_values = collapsed_logits[collapsed_refinable_mask].clone().detach()
        self.refinable_params = nn.Parameter(
            refinable_values, requires_grad=requires_grad
        )

        # Store collapsed shape
        self.register_buffer("_shape", torch.tensor([self._collapsed_shape]))

        # Pre-compute index cache to avoid boolean indexing at runtime
        self._build_index_cache()

    def _setup_sharing_groups_and_expansion(
        self,
        initial_values: torch.Tensor,
        sharing_groups: Optional[torch.Tensor],
        altloc_groups: Optional[list],
        device: torch.device,
    ):
        """
        Setup sharing groups, altlocs, and create expansion mask.

        Creates the index tensor for efficient collapse/expand operations and
        processes altloc groups into tensors grouped by number of conformations.

        Parameters
        ----------
        initial_values : torch.Tensor
            Initial occupancy values for all atoms.
        sharing_groups : torch.Tensor or None
            Tensor of shape (n_atoms,) giving collapsed index for each atom.
        altloc_groups : list or None
            List of tuples of atom index lists for alternative conformations.
        device : torch.device
            Device to place tensors on.
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
        """
        Collapse full tensor to collapsed storage using vectorized scatter_add.

        Parameters
        ----------
        full_values : torch.Tensor
            Tensor in full space (one value per atom).

        Returns
        -------
        torch.Tensor
            Tensor in collapsed space (one value per group + ungrouped atoms).
        """
        # Sum values at each collapsed index using scatter_add
        collapsed_sum = torch.zeros(
            self._collapsed_shape, dtype=full_values.dtype, device=full_values.device
        )
        collapsed_sum.scatter_add_(0, self.expansion_mask, full_values)

        # Divide by counts to get mean (avoid division by zero)
        collapsed = collapsed_sum / self.collapse_counts.float().clamp(min=1)

        return collapsed

    def _collapse_mask_vectorized(self, full_mask: torch.Tensor) -> torch.Tensor:
        """
        Collapse boolean mask to collapsed storage using vectorized operations.

        If ANY atom in a collapsed position is refinable, the position is refinable.

        Parameters
        ----------
        full_mask : torch.Tensor
            Boolean mask in full space.

        Returns
        -------
        torch.Tensor
            Boolean mask in collapsed space.
        """
        # Use scatter_add with float tensors, then check if any > 0
        collapsed_sum = torch.zeros(
            self._collapsed_shape, dtype=torch.float, device=full_mask.device
        )
        collapsed_sum.scatter_add_(0, self.expansion_mask, full_mask.float())

        # If sum > 0, at least one atom in that collapsed position was True
        collapsed = collapsed_sum > 0

        return collapsed

    def _collapse_values(self, full_values: torch.Tensor) -> torch.Tensor:
        """
        Legacy collapse function - redirects to vectorized version.
        Kept for backward compatibility.
        """
        return self._collapse_values_vectorized(full_values)

    def _collapse_mask(self, full_mask: torch.Tensor) -> torch.Tensor:
        """
        Legacy collapse mask function - redirects to vectorized version.
        Kept for backward compatibility.
        """
        return self._collapse_mask_vectorized(full_mask)

    def _expand_values(self, collapsed_values: torch.Tensor) -> torch.Tensor:
        """
        Expand collapsed storage to full tensor using expansion mask.

        Parameters
        ----------
        collapsed_values : torch.Tensor
            Tensor in collapsed space.

        Returns
        -------
        torch.Tensor
            Tensor in full space (one value per atom).
        """
        return collapsed_values[self.expansion_mask]

    def forward(self) -> torch.Tensor:
        """
        Reconstruct full occupancy tensor with sigmoid and altloc constraints.

        For alternative conformations, applies sigmoid then normalizes within
        each group to enforce sum-to-1 constraint.

        Returns
        -------
        torch.Tensor
            Full occupancy tensor with values in [0, 1] and shape (n_atoms,).
        """
        # Get collapsed logit values (combining fixed and refinable)
        # Uses pre-computed integer indices to avoid boolean indexing GPU sync
        result = self.fixed_values.clone()
        if self._has_refinable and self.refinable_params.numel() > 0:
            result[self._refinable_indices] = self.refinable_params

        # Apply sigmoid transformation to get raw occupancies
        if self.use_sigmoid:
            collapsed_occs = torch.sigmoid(result)
        else:
            collapsed_occs = result.clone()

        # Handle linked occupancies: normalize within each altloc group
        # Process each group size separately (2-way, 3-way, etc.)
        if hasattr(self, "linked_occ_sizes") and len(self.linked_occ_sizes) > 0:
            # Start with a copy of collapsed_occs that we'll update
            updated_occs = collapsed_occs.clone()

            for n_conf in self.linked_occ_sizes:
                # Get the tensor of linked indices: shape (N_groups, n_conf)
                linked_indices = getattr(self, f"linked_occ_{n_conf}")

                # Gather occupancies for all linked groups: shape (N_groups, n_conf)
                linked_occs = collapsed_occs[linked_indices]

                # Normalize: divide each by the sum across conformations
                # Shape: (N_groups, 1) for broadcasting
                sums = linked_occs.sum(dim=1, keepdim=True).clamp(min=1e-10)
                normalized_occs = linked_occs / sums

                # this should be vectorized assignment back to updated_occs
                indices_flat = linked_indices.flatten()
                occs_flat = normalized_occs.flatten()
                updated_occs[indices_flat] = occs_flat

            collapsed_occs = updated_occs

        # Expand to full space
        full_occs = self._expand_values(collapsed_occs)

        return full_occs.contiguous()

    def _set_values(self, key, value: torch.Tensor) -> None:
        """
        Internal method to set occupancy values with sigmoid reparameterization.

        Values are provided in NORMAL space (occupancies in [0, 1]) and
        automatically converted to logit-space for internal storage.

        Parameters
        ----------
        key : indexing key
            Index specification (in full atom space).
        value : torch.Tensor
            Occupancy values to assign. Must be in [0, 1].

        Raises
        ------
        ValueError
            If any values are not in [0, 1].

        Notes
        -----
        This operates in FULL atom space. Values are collapsed to internal
        storage using the sharing groups. For atoms that share occupancies,
        the value is averaged across the group.
        """
        # Validate values are in valid range
        if self.use_sigmoid:
            if (value < 0).any() or (value > 1).any():
                raise ValueError("Occupancy values must be in range [0, 1]")

        # Get current full occupancies
        current_full = self.forward().detach()

        # Update in full space
        current_full[key] = value

        # Convert to logit space
        if self.use_sigmoid:
            clamped_values = torch.clamp(current_full, min=1e-6, max=1 - 1e-6)
            logit_values = torch.logit(clamped_values)
        else:
            logit_values = current_full.clone()

        # Collapse to internal storage
        collapsed_logits = self._collapse_values_vectorized(logit_values)

        # Update fixed_values buffer
        self.fixed_values = collapsed_logits.clone()

        # Re-extract refinable parameters
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
        Clamp occupancy values to specified range and return a new OccupancyTensor.

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

        The mask is supplied in UNCOMPRESSED (full atom) form but freezing
        operates on the COMPRESSED data structure. This method handles the
        conversion.

        Parameters
        ----------
        mask : torch.Tensor, optional
            Boolean mask in FULL (uncompressed) atom space indicating which
            atoms to freeze. If None, freeze all parameters.
            Shape must be (n_atoms,).

        Notes
        -----
        If ANY atom in a sharing group is frozen, the ENTIRE group is frozen
        because all atoms in a group share the same compressed parameter.

        Examples
        --------
        ::

            # Freeze atoms 0-10 (in full atom space)
            freeze_mask = torch.zeros(n_atoms, dtype=torch.bool)
            freeze_mask[0:11] = True
            occ.freeze(freeze_mask)
            # Freeze all atoms
            occ.freeze()
        """
        if mask is None:
            # Freeze all - set refinable_mask to all False
            mask = torch.ones(self._full_shape, dtype=torch.bool, device=self.device)
        else:
            # Validate mask shape
            if mask.shape[0] != self._full_shape:
                raise ValueError(
                    f"Freeze mask must have shape ({self._full_shape},) to match full atom space, "
                    f"got shape {mask.shape}"
                )
            mask = mask.to(device=self.device, dtype=torch.bool)

        # Collapse the freeze mask to compressed space
        # If ANY atom in a group should be frozen, the group is frozen
        collapsed_freeze_mask = self._collapse_mask_vectorized(mask)

        # Get current full state (collapsed logits)
        current_logits = self.fixed_values.clone()
        current_logits[self.refinable_mask] = self.refinable_params.data

        # Update masks: positions to freeze become non-refinable
        new_refinable_mask = self.refinable_mask & ~collapsed_freeze_mask

        # Update fixed values with current state
        self.fixed_values = current_logits.clone().detach()

        # Update refinable params - only keep parameters that are still refinable
        if new_refinable_mask.any():
            new_refinable_values = current_logits[new_refinable_mask].clone().detach()
            self.refinable_params = nn.Parameter(
                new_refinable_values, requires_grad=self.refinable_params.requires_grad
            )
        else:
            # All parameters frozen - create empty parameter
            self.refinable_params = nn.Parameter(
                torch.empty(0, dtype=self.dtype, device=self.device),
                requires_grad=False,
            )

        # Update masks
        self.refinable_mask = new_refinable_mask
        self.fixed_mask = ~new_refinable_mask

        # Rebuild index cache after mask change
        self._build_index_cache()

    def unfreeze(self, mask: Optional[torch.Tensor] = None):
        """
        Unfreeze occupancy parameters, making them refinable.

        The mask is supplied in UNCOMPRESSED (full atom) form but unfreezing
        operates on the COMPRESSED data structure. This method handles the
        conversion.

        Parameters
        ----------
        mask : torch.Tensor, optional
            Boolean mask in FULL (uncompressed) atom space indicating which
            atoms to unfreeze. If None, unfreeze all parameters.
            Shape must be (n_atoms,).

        Notes
        -----
        If ANY atom in a sharing group is unfrozen, the ENTIRE group becomes
        refinable because all atoms in a group share the same compressed parameter.

        Examples
        --------
        ::

            # Unfreeze atoms 100-200 (in full atom space)
            unfreeze_mask = torch.zeros(n_atoms, dtype=torch.bool)
            unfreeze_mask[100:201] = True
            occ.unfreeze(unfreeze_mask)
            # Unfreeze all atoms
            occ.unfreeze()
        """
        if mask is None:
            # Unfreeze all - set refinable_mask to all True
            mask = torch.ones(self._full_shape, dtype=torch.bool, device=self.device)
        else:
            # Validate mask shape
            if mask.shape[0] != self._full_shape:
                raise ValueError(
                    f"Unfreeze mask must have shape ({self._full_shape},) to match full atom space, "
                    f"got shape {mask.shape}"
                )
            mask = mask.to(device=self.device, dtype=torch.bool)

        # Collapse the unfreeze mask to compressed space
        # If ANY atom in a group should be unfrozen, the group becomes refinable
        collapsed_unfreeze_mask = self._collapse_mask_vectorized(mask)

        # Get current full state (collapsed logits)
        current_logits = self.fixed_values.clone()
        if self.refinable_mask.any():
            current_logits[self.refinable_mask] = self.refinable_params.data

        # Update masks: positions to unfreeze become refinable
        new_refinable_mask = self.refinable_mask | collapsed_unfreeze_mask

        # Update fixed values with current state
        self.fixed_values = current_logits.clone().detach()

        # Update refinable params - include newly unfrozen parameters
        if new_refinable_mask.any():
            new_refinable_values = current_logits[new_refinable_mask].clone().detach()
            self.refinable_params = nn.Parameter(
                new_refinable_values,
                requires_grad=True,  # Unfrozen parameters should have gradients
            )
        else:
            # No refinable parameters
            self.refinable_params = nn.Parameter(
                torch.empty(0, dtype=self.dtype, device=self.device),
                requires_grad=False,
            )

        # Update masks
        self.refinable_mask = new_refinable_mask
        self.fixed_mask = ~new_refinable_mask

        # Rebuild index cache after mask change
        self._build_index_cache()

    def freeze_all(self):
        """
        Freeze all occupancy parameters.

        Convenience method equivalent to freeze(None).
        """
        self.freeze(None)

    def unfreeze_all(self):
        """
        Unfreeze all occupancy parameters.

        Convenience method equivalent to unfreeze(None).
        """
        self.unfreeze(None)

    def get_refinable_atoms(self) -> torch.Tensor:
        """
        Get a boolean mask in FULL atom space indicating refinable atoms.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (n_atoms,) where True indicates the atom's
            occupancy is refinable (though it shares with others in its group).
        """
        return self._expand_values(self.refinable_mask.float()).bool()

    def get_frozen_atoms(self) -> torch.Tensor:
        """
        Get a boolean mask in FULL atom space indicating frozen atoms.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape (n_atoms,) where True indicates the atom's
            occupancy is frozen.
        """
        return self._expand_values(self.fixed_mask.float()).bool()

    def get_refinable_count(self) -> int:
        """
        Get the number of refinable parameters in COMPRESSED space.

        This is the number of refinable groups, not atoms.
        Use get_refinable_atoms().sum() to get the number of refinable atoms.

        Returns
        -------
        int
            Number of refinable compressed parameters.
        """
        return self.refinable_mask.sum().item()

    def get_fixed_count(self) -> int:
        """
        Get the number of fixed parameters in COMPRESSED space.

        This is the number of fixed groups, not atoms.
        Use get_frozen_atoms().sum() to get the number of frozen atoms.

        Returns
        -------
        int
            Number of fixed compressed parameters.
        """
        return self.fixed_mask.sum().item()

    def update_refinable_mask(
        self, new_mask: torch.Tensor, in_compressed_space: bool = False
    ):
        """
        Directly update the refinable mask with a new mask.

        Allows more direct control over which parameters are refinable,
        compared to freeze/unfreeze which modify the existing state.

        Parameters
        ----------
        new_mask : torch.Tensor
            Boolean tensor indicating which parameters should be refinable.
            If in_compressed_space=False: shape (n_atoms,) in full atom space.
            If in_compressed_space=True: shape (n_groups,) in compressed space.
        in_compressed_space : bool, optional
            If True, new_mask is in compressed space.
            If False (default), new_mask is in full atom space and will be collapsed.

        Examples
        --------
        Full atom space::

            atom_mask = torch.zeros(n_atoms, dtype=torch.bool)
            atom_mask[:100] = True
            occ.update_refinable_mask(atom_mask, in_compressed_space=False)

        Compressed space::

            group_mask = torch.zeros(n_groups, dtype=torch.bool)
            group_mask[::2] = True
            occ.update_refinable_mask(group_mask, in_compressed_space=True)
        """
        # Validate and convert mask
        if not in_compressed_space:
            # Mask is in full atom space, need to collapse
            if new_mask.shape[0] != self._full_shape:
                raise ValueError(
                    f"Mask in full atom space must have shape ({self._full_shape},), "
                    f"got shape {new_mask.shape}"
                )
            new_mask = new_mask.to(device=self.device, dtype=torch.bool)
            collapsed_mask = self._collapse_mask_vectorized(new_mask)
        else:
            # Mask is already in compressed space
            if new_mask.shape[0] != self._collapsed_shape:
                raise ValueError(
                    f"Mask in compressed space must have shape ({self._collapsed_shape},), "
                    f"got shape {new_mask.shape}"
                )
            new_mask = new_mask.to(device=self.device, dtype=torch.bool)
            collapsed_mask = new_mask

        # Get current full state (collapsed logits)
        current_logits = self.fixed_values.clone()
        if self.refinable_mask.any():
            current_logits[self.refinable_mask] = self.refinable_params.data

        # Update fixed values with current state
        self.fixed_values = current_logits.clone().detach()

        # Create new refinable params based on new mask
        if collapsed_mask.any():
            new_refinable_values = current_logits[collapsed_mask].clone().detach()
            self.refinable_params = nn.Parameter(
                new_refinable_values, requires_grad=True
            )
        else:
            # No refinable parameters
            self.refinable_params = nn.Parameter(
                torch.empty(0, dtype=self.dtype, device=self.device),
                requires_grad=False,
            )

        # Update masks
        self.refinable_mask = collapsed_mask
        self.fixed_mask = ~collapsed_mask

        # Rebuild index cache after mask change
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

        Common use case where all atoms in a residue should have the same occupancy.

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
        collapsed_idx = 0

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
        Create a deep copy of this OccupancyTensor.

        Creates a complete independent copy with all buffers and parameters,
        including sharing groups, altloc groups, and collapsed storage structures.

        Returns
        -------
        OccupancyTensor
            New OccupancyTensor instance with copied data.
        """
        # Get current occupancy values in normal space (full atom space)
        current_occ = self.forward().detach()

        # Reconstruct refinable mask in full space
        full_refinable_mask = self._expand_values(self.refinable_mask.float()).bool()

        # Reconstruct altloc groups from the linked_occ buffers
        altloc_groups = []
        if hasattr(self, "linked_occ_sizes"):
            for n_conf in self.linked_occ_sizes:
                linked_indices = getattr(
                    self, f"linked_occ_{n_conf}"
                )  # shape (N_groups, n_conf)

                # For each group of linked conformations
                for group_collapsed_indices in linked_indices:
                    # Find all atoms that map to each collapsed index
                    conf_atom_lists = []
                    for collapsed_idx in group_collapsed_indices:
                        atom_indices = (
                            (self.expansion_mask == collapsed_idx)
                            .nonzero(as_tuple=False)
                            .squeeze(-1)
                        )
                        conf_atom_lists.append(atom_indices.tolist())

                    altloc_groups.append(tuple(conf_atom_lists))

        # Create new OccupancyTensor with the same configuration
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

    def to(self, device=None, dtype=None) -> "OccupancyTensor":
        """
        Move tensor to a different device or dtype.

        Parameters
        ----------
        device : torch.device or str, optional
            Target device.
        dtype : torch.dtype, optional
            Target data type.

        Returns
        -------
        OccupancyTensor
            Self for method chaining.
        """
        # Move OccupancyTensor-specific buffers
        if device is not None or dtype is not None:
            # Move expansion_mask and collapse_counts
            if hasattr(self, "expansion_mask") and self.expansion_mask is not None:
                self.expansion_mask = self.expansion_mask.to(device=device)
            if hasattr(self, "collapse_counts") and self.collapse_counts is not None:
                self.collapse_counts = self.collapse_counts.to(device=device)

            # Move linked occupancy buffers
            if hasattr(self, "linked_occ_sizes"):
                for n_conf in self.linked_occ_sizes:
                    buf_name = f"linked_occ_{n_conf}"
                    if hasattr(self, buf_name):
                        buf = getattr(self, buf_name)
                        if buf is not None:
                            setattr(self, buf_name, buf.to(device=device))

        # Call parent's to() which handles the MixedTensor buffers
        return super().to(device=device, dtype=dtype)

    def __repr__(self) -> str:
        name_str = f"'{self.name}', " if self.name is not None else ""
        n_groups = self._collapsed_shape
        return (
            f"OccupancyTensor({name_str}shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device}, refinable={self.get_refinable_count()}, "
            f"fixed={self.get_fixed_count()}, collapsed_groups={n_groups}, "
            f"use_sigmoid={self.use_sigmoid})"
        )


class PassThroughTensor(nn.Module):
    """
    A simple parameter wrapper that passes the parameter through unchanged.

    Useful as a placeholder or for parameters that do not require any
    special handling.

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
        """
        Initialize the PassThroughTensor.

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
        super().__init__(
            initial_values=initial_values,
            requires_grad=requires_grad,
            dtype=dtype,
            device=device,
            name=name,
        )

    def forward(self) -> torch.Tensor:
        """
        Return the parameter value unchanged.

        Returns
        -------
        torch.Tensor
            The parameter tensor.
        """
        return self.param
