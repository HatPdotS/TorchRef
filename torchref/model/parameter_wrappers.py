"""
A file that contains wrapper classes for handling crystallographic parameters. (Occ, xyz, B, etc.)
"""

import torch
from torch import nn
from typing import Optional, Union

class MixedTensor(nn.Module):
    """
    A wrapper class to handle tensors where part of the tensor should be fixed 
    and another part should be optimized during refinement.
    
    This class stores a mask indicating which elements can be refined, and maintains
    both fixed and refinable components separately. The full tensor is reconstructed
    on-the-fly when accessed.
    
    Example:
        >>> # Create a tensor with 100 elements, where only indices 20-30 are refinable
        >>> mask = torch.zeros(100, dtype=torch.bool)
        >>> mask[20:30] = True
        >>> initial_values = torch.randn(100)
        >>> mixed = MixedTensor(initial_values, refinable_mask=mask, requires_grad=True)
        >>> 
        >>> # Use it in optimization
        >>> optimizer = torch.optim.Adam([mixed.refinable_params], lr=0.01)
        >>> loss = (mixed() - target).pow(2).sum()
        >>> loss.backward()
        >>> optimizer.step()
    """
    
    def __init__(
        self, 
        initial_values: torch.Tensor, 
        refinable_mask: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        name: Optional[str] = None
    ):
        """
        Initialize a MixedTensor.
        
        Args:
            initial_values: Initial tensor values for all elements
            refinable_mask: Boolean mask indicating which elements can be refined.
                          If None, all elements are refinable.
            requires_grad: Whether refinable parameters should have gradients
            dtype: Data type for the tensor (default: same as initial_values)
            device: Device for the tensor (default: same as initial_values)
            name: Optional name for this parameter (useful for debugging/logging)
        """
        super().__init__()
        
        # Store the name
        self._name = name
        
        if dtype is None:
            dtype = initial_values.dtype
        if device is None:
            device = initial_values.device
            
        initial_values = initial_values.to(dtype=dtype, device=device)
        
        # Create refinable mask
        if refinable_mask is None:
            # For multi-dimensional tensors, create a mask for the first dimension
            if initial_values.ndim > 1:
                refinable_mask = torch.ones(initial_values.shape[0], dtype=torch.bool, device=device)
            else:
                refinable_mask = torch.ones_like(initial_values, dtype=torch.bool)
        else:
            refinable_mask = refinable_mask.to(device=device)
            
        # Validate mask shape - it should match the first dimension for broadcasting
        if initial_values.ndim > 1:
            # For multi-dimensional tensors, mask should be 1D matching first dimension
            if refinable_mask.ndim != 1 or refinable_mask.shape[0] != initial_values.shape[0]:
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
        self.register_buffer('refinable_mask', refinable_mask)
        self.register_buffer('fixed_mask', ~refinable_mask)
        
        # Store fixed values as a buffer
        fixed_values = initial_values.clone().detach()
        self.register_buffer('fixed_values', fixed_values)
        
        # Store refinable values as a parameter (will be optimized)
        refinable_values = initial_values[refinable_mask].clone().detach()
        self.refinable_params = nn.Parameter(
            refinable_values, 
            requires_grad=requires_grad
        )
        
        # Store shape for reconstruction
        self.register_buffer('_shape', torch.tensor(initial_values.shape))
    
    def forward(self) -> torch.Tensor:
        """
        Reconstruct and return the full tensor by combining fixed and refinable parts.
        
        For multi-dimensional tensors (e.g., xyz with shape [N, 3]), the mask
        broadcasts correctly: refinable_mask with shape [N] selects rows.
        
        Returns:
            Full tensor with fixed values in non-refinable positions and 
            current refinable parameter values in refinable positions.
        """
        # Start with fixed values
        result = self.fixed_values.clone()
        
        # Insert refinable values at their positions
        # For multi-dimensional tensors, mask broadcasts automatically
        # Only assign if there are refinable parameters
        if self.refinable_mask.any():
            result[self.refinable_mask] = self.refinable_params
        
        return result
    
    def __call__(self) -> torch.Tensor:
        """Allow instance to be called like a function."""
        return self.forward()
    
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
        
        Args:
            new_values: New tensor values. Only fixed positions will be updated.
        """
        if new_values.shape != self.shape:
            raise ValueError(
                f"new_values shape {new_values.shape} must match "
                f"tensor shape {self.shape}"
            )
        self.fixed_values = new_values.to(dtype=self.dtype, device=self.device).detach()
    
    def update_refinable_mask(self, new_mask: torch.Tensor, reset_refinable: bool = False):
        """
        Update which elements are refinable. This is an advanced operation.
        
        Args:
            new_mask: New boolean mask indicating refinable elements
            reset_refinable: If True, reset refinable parameters to current fixed values.
                           If False, keep existing refinable parameter values where possible.
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
            new_refinable,
            requires_grad=self.refinable_params.requires_grad
        )
    
    def detach(self) -> torch.Tensor:
        """Return a detached copy of the full tensor."""
        return self.forward().detach()
    
    def clone(self) -> 'MixedTensor':
        """Create a deep copy of this MixedTensor."""
        new_mixed = MixedTensor(
            self.forward().detach(),
            self.refinable_mask.clone(),
            requires_grad=self.refinable_params.requires_grad,
            dtype=self.dtype,
            device=self.device,
            name=self.name
        )
        return new_mixed
    
    def copy(self) -> 'MixedTensor':
        """
        Create a deep copy of this MixedTensor.
        
        This is an alias for clone() to follow common Python conventions.
        Creates a complete independent copy with all buffers and parameters.
        
        Returns:
            New MixedTensor instance with copied data
        """
        return self.clone()
    
    def clip(self, min_value=None, max_value=None) -> 'MixedTensor':
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
            name=self.name
        )
        return new_mixed
    
    def to(self, *args, **kwargs) -> 'MixedTensor':
        """Move tensor to a different device or dtype."""
        super().to(*args, **kwargs)
        return self
    
    def parameters(self):
        parameter = super().parameters()
        parameter_valid = [param for param in parameter if param.numel() > 0]
        yield from parameter_valid

    def refine(self, selection: Union[slice, torch.Tensor, tuple], reset_values: bool = False):
        """
        Make a selection of the tensor refinable.
        
        Args:
            selection: Boolean mask, slice, or index selection indicating which 
                      elements should become refinable. Can be:
                      - Boolean tensor of same shape as the full tensor
                      - Slice object (e.g., slice(10, 20))
                      - Tuple of indices for multidimensional tensors
                      - Integer indices
            reset_values: If True, reset the selected elements to their current 
                         fixed values before making them refinable.
        
        Example:
            >>> mixed.make_refinable(slice(10, 20))  # Make elements 10-19 refinable
            >>> mixed.make_refinable(mask)  # Make elements where mask is True refinable
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
            new_refinable,
            requires_grad=self.refinable_params.requires_grad
        )
    
    def fix(self, selection: Union[slice, torch.Tensor, tuple], freeze_at_current: bool = True):
        """
        Make a selection of the tensor fixed (non-refinable).
        
        Args:
            selection: Boolean mask, slice, or index selection indicating which 
                      elements should become fixed. Can be:
                      - Boolean tensor of same shape as the full tensor
                      - Slice object (e.g., slice(10, 20))
                      - Tuple of indices for multidimensional tensors
                      - Integer indices
            freeze_at_current: If True (default), freeze the selected elements at 
                             their current values. If False, they revert to the 
                             original fixed values.
        
        Example:
            >>> mixed.make_fixed(slice(10, 20))  # Fix elements 10-19
            >>> mixed.make_fixed(mask)  # Fix elements where mask is True
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
                new_refinable,
                requires_grad=self.refinable_params.requires_grad
            )
        else:
            # All fixed, create empty parameter
            self.refinable_params = nn.Parameter(
                torch.tensor([], dtype=self.dtype, device=self.device),
                requires_grad=self.refinable_params.requires_grad
            )
    
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
    A MixedTensor subclass that ensures all values are positive by parametrizing in log space.
    
    This class is useful for parameters that must be strictly positive (e.g., B-factors,
    scale factors, sigma values). Values are stored as logarithms internally and 
    converted to normal space via exp() when accessed.
    
    Reparametrization:
        internal_value = log(desired_value)
        output_value = exp(internal_value)
    
    This ensures:
        - output_value > 0 always (never negative or zero)
        - Gradient flow is smooth and well-behaved
        - No need for manual clamping or constraints
    
    Example:
        >>> # Create positive-only B-factors
        >>> initial_b = torch.tensor([20.0, 30.0, 15.0])  # Positive B-factors
        >>> b = PositiveMixedTensor(initial_b)
        >>> 
        >>> # Values are automatically in normal space when accessed
        >>> output = b()  # Returns exp(log_b) = positive values
        >>> 
        >>> # Optimization works in log space
        >>> optimizer = torch.optim.Adam([b.refinable_params], lr=0.01)
        >>> loss = (b() - target_b).pow(2).sum()
        >>> loss.backward()
        >>> optimizer.step()
        >>> 
        >>> # Values remain positive after optimization
        >>> assert (b() > 0).all()
    """
    
    def __init__(
        self,
        initial_values: torch.Tensor,
        refinable_mask: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        name: Optional[str] = None,
        epsilon: float = 1e-1
    ):
        """
        Initialize a PositiveMixedTensor.
        
        Args:
            initial_values: Initial tensor values in NORMAL space (must be positive)
            refinable_mask: Boolean mask indicating which elements can be refined
            requires_grad: Whether refinable parameters should have gradients
            dtype: Data type for the tensor (default: same as initial_values)
            device: Device for the tensor (default: same as initial_values)
            name: Optional name for this parameter
            epsilon: Small value to add before taking log to avoid log(0) (default: 1e-10)
        
        Raises:
            ValueError: If any initial values are not positive
        """
        """Clip initial values to be positive"""

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
            name=name
        )
    
    def forward(self) -> torch.Tensor:
        """
        Return the full tensor in NORMAL space (exponential of log-space values).
        
        Returns:
            Tensor with positive values
        """
        # Get log-space values from parent
        log_values = super().forward()
        
        # Convert to normal space via exp
        return torch.exp(log_values)
    
    def fix(self, mask: torch.Tensor, freeze_at_current: bool = True):
        """
        Fix (freeze) specific elements, converting current normal-space values to log space.
        
        Args:
            mask: Boolean mask indicating which elements to fix
            freeze_at_current: If True, freeze at current values
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
        Make specific elements refinable, preserving their current log-space values.
        
        Args:
            mask: Boolean mask indicating which elements to make refinable
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
    
    def get_log_values(self) -> torch.Tensor:
        """
        Return the internal log-space representation.
        
        This is useful for debugging or when you need direct access to the
        parametrization space.
        
        Returns:
            Tensor with log-space values
        """
        return super().forward()
    
    def update_refinable_mask(self, new_mask: torch.Tensor, reset_refinable: bool = False):
        """
        Update which elements are refinable, properly handling log-space conversion.
        
        Args:
            new_mask: New boolean mask indicating refinable elements
            reset_refinable: If True, reset refinable parameters to current fixed values.
                           If False, keep existing refinable parameter values where possible.
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
            new_refinable_log,
            requires_grad=self.refinable_params.requires_grad
        )
    
    def copy(self) -> 'PositiveMixedTensor':
        """
        Create a deep copy of this PositiveMixedTensor.
        
        Creates a complete independent copy with all buffers and parameters,
        properly handling the log-space reparametrization.
        
        Returns:
            New PositiveMixedTensor instance with copied data
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
            epsilon=self.epsilon
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
    
    This class handles the specific constraints and requirements for occupancy:
    1. Values are bounded between 0 and 1 using sigmoid reparameterization
    2. Atoms can share occupancies (e.g., all atoms in a residue)
    3. Alternative conformations automatically sum to 1.0 via normalization
    4. Supports per-atom, per-residue, or custom grouping schemes
    5. Memory-efficient: stores only one parameter per sharing group
    6. Static (fixed) occupancies that never change during refinement
    7. Fully vectorized collapse/expand operations using scatter_add and indexing
    
    The internal representation:
    - Stores COLLAPSED logit values (one per unique group, plus one per ungrouped atom)
    - Uses an index tensor (expansion_mask) to map atoms to collapsed indices
    - Collapse: scatter_add to sum values by collapsed index (O(n))
    - Expand: direct indexing collapsed_values[expansion_mask] (O(n))
    - Transforms to [0,1] occupancies via sigmoid function during forward pass
    
    Alternative Conformation Handling (New Normalization Approach):
    - Altloc groups are represented as tensors of shape (N_groups, M_conformations)
    - Grouped by number of conformations: linked_occ_2, linked_occ_3, etc.
    - During forward(), all members are passed through sigmoid, then normalized:
      occupancy_i = sigmoid(logit_i) / sum_j(sigmoid(logit_j))
    - This enforces sum-to-1 constraint while keeping all parameters refinable
    - More stable gradients than placeholder approach
    - Handles arbitrary N-way splits (not just 2-way)
    
    Example:
        >>> # Create occupancy with sharing groups as an index tensor
        >>> # Atoms 0,1 share (index 0), atoms 2,3 share (index 1), atoms 4,5 share (index 2)
        >>> sharing_groups = torch.tensor([0, 0, 1, 1, 2, 2])
        >>> occ = OccupancyTensor(
        ...     initial_values=torch.tensor([1.0, 1.0, 0.7, 0.7, 0.3, 0.3]),
        ...     sharing_groups=sharing_groups,
        ...     altloc_groups=[([2, 3], [4, 5])],  # indices 1 and 2 are altlocs
        ... )
        >>> # All parameters refinable, normalization ensures sum-to-1
        >>> result = occ()  # Atoms 2-3 and 4-5 will sum to 1.0
        >>> # Stored as linked_occ_2 = tensor([[1, 2]]) - 2-way split
    """
    
    def __init__(
        self,
        initial_values: torch.Tensor,
        sharing_groups: Optional[torch.Tensor] = None,
        altloc_groups: Optional[list] = None,
        refinable_mask: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        name: Optional[str] = None,
        use_sigmoid: bool = True
    ):
        """
        Initialize an OccupancyTensor with collapsed storage and altloc support.
        
        Args:
            initial_values: Initial occupancy values for ALL atoms (should be in [0, 1])
            sharing_groups: Tensor of shape (n_atoms,) where each value is the collapsed index
                          for that atom. If None, each atom has independent occupancy.
                          Example: tensor([0, 0, 0, 1, 1, 2]) means atoms 0,1,2 share one occupancy,
                          atoms 3,4 share another, and atom 5 is independent.
            altloc_groups: List of tuples of atom index lists representing alternative
                          conformations. Each tuple contains the atom indices for each
                          conformation. Example: [([10,11], [12,13]), ([20,21], [22,23])]
                          means atoms 10,11 (conf A) and 12,13 (conf B) are altlocs,
                          and atoms 20,21 (conf C) and 22,23 (conf D) are altlocs.
                          These automatically sum to 1.0.
            refinable_mask: Boolean mask for which ATOMS can be refined (in full tensor space).
                          Applied after sharing groups. If any atom in a group is refinable,
                          the entire group becomes refinable. Altloc placeholders are never
                          directly refinable (computed from other altlocs).
            requires_grad: Whether refinable parameters should have gradients
            dtype: Data type for the tensor
            device: Device for the tensor
            name: Optional name for this parameter
            use_sigmoid: If True, use sigmoid parameterization to bound values to [0,1].
        """
        # Store configuration
        self.use_sigmoid = use_sigmoid
        self._full_shape = initial_values.shape[0]
        
        # Initialize Module first (required before register_buffer)
        nn.Module.__init__(self)
        
        self._name = name or 'occupancy'
        
        if dtype is None:
            dtype = initial_values.dtype
        if device is None:
            device = initial_values.device
        
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
            clamped_values = torch.clamp(initial_values, min=1e-6, max=1-1e-6)
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
            collapsed_refinable_mask = self._collapse_mask_vectorized(refinable_mask.to(device=device))
        else:
            collapsed_refinable_mask = torch.ones(self._collapsed_shape, dtype=torch.bool, device=device)
        
        # Note: With the new normalization approach, all altloc members are refinable
        # The sum-to-1 constraint is enforced during forward() via normalization
        
        # Store masks as buffers
        self.register_buffer('refinable_mask', collapsed_refinable_mask)
        self.register_buffer('fixed_mask', ~collapsed_refinable_mask)
        
        # Store fixed values as buffer (collapsed)
        self.register_buffer('fixed_values', collapsed_logits.clone().detach())
        
        # Store refinable values as parameter (collapsed, excluding placeholders)
        refinable_values = collapsed_logits[collapsed_refinable_mask].clone().detach()
        self.refinable_params = nn.Parameter(refinable_values, requires_grad=requires_grad)
        
        # Store collapsed shape
        self.register_buffer('_shape', torch.tensor([self._collapsed_shape]))
    
    def _setup_sharing_groups_and_expansion(
        self, 
        initial_values: torch.Tensor, 
        sharing_groups: Optional[torch.Tensor],
        altloc_groups: Optional[list],
        device: torch.device
    ):
        """
        Setup sharing groups, altlocs, and create expansion mask for memory-efficient storage.
        
        Now uses a simple index tensor approach:
        - sharing_groups is a tensor of shape (n_atoms,) mapping each atom to its collapsed index
        - Collapse: scatter_add to sum values by collapsed index
        - Expand: direct indexing collapsed_values[sharing_groups]
        
        For altlocs, we create tensors of shape (N_pairs, M_conformations) where each row
        contains the collapsed indices for conformations that must sum to 1.0.
        These are stored grouped by number of conformations (2-way, 3-way, etc.).
        
        Args:
            initial_values: Initial occupancy values for all atoms
            sharing_groups: Tensor of shape (n_atoms,) giving collapsed index for each atom
            altloc_groups: List of tuples of atom index lists for alternative conformations
            device: Device to place tensors on
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
        
        self.register_buffer('expansion_mask', expansion_mask)
        
        # Process altloc groups: convert to collapsed indices and group by size
        # linked_occupancies[n] = tensor of shape (N_groups, n) where n is number of conformations
        linked_occupancies = {}
        
        if altloc_groups is not None and len(altloc_groups) > 0:
            for altloc_idx, conf_groups in enumerate(altloc_groups):
                n_conformations = len(conf_groups)
                if n_conformations < 2:
                    raise ValueError(f"Altloc group {altloc_idx} must have at least 2 conformations")
                
                # Get collapsed indices for each conformation
                collapsed_indices = []
                for conf_atoms in conf_groups:
                    if isinstance(conf_atoms, (list, tuple)):
                        conf_atoms = torch.tensor(conf_atoms, dtype=torch.long, device=device)
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
            self.register_buffer(f'linked_occ_{n_conf}', tensor)
        
        # Store which sizes we have
        self.linked_occ_sizes = sorted(linked_occupancies.keys())
        
        # Create count buffer for vectorized collapse operations
        # counts[i] = number of atoms that map to collapsed index i
        counts = torch.zeros(self._collapsed_shape, dtype=torch.long, device=device)
        counts.scatter_add_(0, expansion_mask, torch.ones_like(expansion_mask))
        self.register_buffer('collapse_counts', counts)
    
    def _collapse_values_vectorized(self, full_values: torch.Tensor) -> torch.Tensor:
        """
        Collapse full tensor to collapsed storage using vectorized scatter_add.
        
        Uses the expansion_mask directly for O(n) collapse operation.
        
        Args:
            full_values: Tensor in full space (one value per atom)
        
        Returns:
            Tensor in collapsed space (one value per group + ungrouped atoms)
        """
        # Sum values at each collapsed index using scatter_add
        collapsed_sum = torch.zeros(
            self._collapsed_shape, 
            dtype=full_values.dtype, 
            device=full_values.device
        )
        collapsed_sum.scatter_add_(0, self.expansion_mask, full_values)
        
        # Divide by counts to get mean (avoid division by zero)
        collapsed = collapsed_sum / self.collapse_counts.float().clamp(min=1)
        
        return collapsed
    
    def _collapse_mask_vectorized(self, full_mask: torch.Tensor) -> torch.Tensor:
        """
        Collapse boolean mask to collapsed storage using vectorized operations.
        
        If ANY atom in a collapsed position is refinable, the position is refinable.
        
        Args:
            full_mask: Boolean mask in full space
        
        Returns:
            Boolean mask in collapsed space
        """
        # Use scatter_add with float tensors, then check if any > 0
        collapsed_sum = torch.zeros(
            self._collapsed_shape,
            dtype=torch.float,
            device=full_mask.device
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
        
        Args:
            collapsed_values: Tensor in collapsed space
        
        Returns:
            Tensor in full space (one value per atom)
        """
        return collapsed_values[self.expansion_mask]
    
    def forward(self) -> torch.Tensor:
        """
        Reconstruct full occupancy tensor with sigmoid transformation and altloc constraints.
        
        For alternative conformations, we apply sigmoid then normalize within each group
        to enforce the sum-to-1 constraint. This is done separately for each group size
        (2-way, 3-way, etc.) for efficiency.
        
        Returns:
            Full occupancy tensor with values in [0, 1] (shape: [n_atoms])
        """
        # Get collapsed logit values (combining fixed and refinable)
        result = self.fixed_values.clone()
        result[self.refinable_mask] = self.refinable_params
        
        # Apply sigmoid transformation to get raw occupancies
        if self.use_sigmoid:
            collapsed_occs = torch.sigmoid(result)
        else:
            collapsed_occs = result.clone()
        
        # Handle linked occupancies: normalize within each altloc group
        # Process each group size separately (2-way, 3-way, etc.)
        if hasattr(self, 'linked_occ_sizes') and len(self.linked_occ_sizes) > 0:
            # Start with a copy of collapsed_occs that we'll update
            updated_occs = collapsed_occs.clone()
            
            for n_conf in self.linked_occ_sizes:
                # Get the tensor of linked indices: shape (N_groups, n_conf)
                linked_indices = getattr(self, f'linked_occ_{n_conf}')
                
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
        
        return full_occs
    
    @property
    def shape(self):
        """Return the shape of the FULL tensor (not collapsed)."""
        return (self._full_shape,)
    
    @property
    def collapsed_shape(self):
        """Return the shape of the collapsed internal storage."""
        return (self._collapsed_shape,)
    
    def clamp(self, min_value: float = 0.0, max_value: float = 1.0) -> 'OccupancyTensor':
        """
        Clamp occupancy values to specified range and return a new OccupancyTensor.
        
        Args:
            min_value: Minimum occupancy value (default: 0.0)
            max_value: Maximum occupancy value (default: 1.0)
        
        Returns:
            New OccupancyTensor with clamped values
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
            use_sigmoid=self.use_sigmoid
        )
        
        return new_occ
    
    def set_group_occupancy(self, group_idx: int, value: float):
        """
        Set the occupancy for all atoms in a specific collapsed group.
        
        Args:
            group_idx: Collapsed index of the group
            value: Occupancy value to set (must be in [0, 1])
        """
        if group_idx < 0 or group_idx >= self._collapsed_shape:
            raise ValueError(f"Invalid group index {group_idx}")
        
        if value < 0 or value > 1:
            raise ValueError(f"Occupancy value must be in [0, 1], got {value}")
        
        # Convert value to logit space
        clamped_value = np.clip(value, 1e-6, 1-1e-6)
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
        
        Args:
            group_idx: Collapsed index of the group
        
        Returns:
            Current occupancy value for the group
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
        
        IMPORTANT: The mask is supplied in UNCOMPRESSED (full atom) form but freezing
        operates on the COMPRESSED data structure. This method handles the conversion.
        
        Args:
            mask: Optional boolean mask in FULL (uncompressed) atom space indicating which
                  atoms to freeze. If None, freeze all parameters.
                  Shape must be (n_atoms,) where n_atoms is the full number of atoms.
                  
        Behavior:
            - If ANY atom in a sharing group is frozen, the ENTIRE group is frozen
            - This is because all atoms in a group share the same compressed parameter
            - The mask is collapsed using the same logic as initial mask setup
        
        Example:
            >>> # Freeze atoms 0-10 (in full atom space)
            >>> freeze_mask = torch.zeros(n_atoms, dtype=torch.bool)
            >>> freeze_mask[0:11] = True
            >>> occ.freeze(freeze_mask)
            >>> 
            >>> # Freeze all atoms
            >>> occ.freeze()
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
                new_refinable_values, 
                requires_grad=self.refinable_params.requires_grad
            )
        else:
            # All parameters frozen - create empty parameter
            self.refinable_params = nn.Parameter(
                torch.empty(0, dtype=self.dtype, device=self.device),
                requires_grad=False
            )
        
        # Update masks
        self.refinable_mask = new_refinable_mask
        self.fixed_mask = ~new_refinable_mask
    
    def unfreeze(self, mask: Optional[torch.Tensor] = None):
        """
        Unfreeze occupancy parameters, making them refinable.
        
        IMPORTANT: The mask is supplied in UNCOMPRESSED (full atom) form but unfreezing
        operates on the COMPRESSED data structure. This method handles the conversion.
        
        Args:
            mask: Optional boolean mask in FULL (uncompressed) atom space indicating which
                  atoms to unfreeze. If None, unfreeze all parameters.
                  Shape must be (n_atoms,) where n_atoms is the full number of atoms.
                  
        Behavior:
            - If ANY atom in a sharing group is unfrozen, the ENTIRE group becomes refinable
            - This is because all atoms in a group share the same compressed parameter
            - The mask is collapsed using the same logic as initial mask setup
        
        Example:
            >>> # Unfreeze atoms 100-200 (in full atom space)
            >>> unfreeze_mask = torch.zeros(n_atoms, dtype=torch.bool)
            >>> unfreeze_mask[100:201] = True
            >>> occ.unfreeze(unfreeze_mask)
            >>> 
            >>> # Unfreeze all atoms
            >>> occ.unfreeze()
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
                requires_grad=True  # Unfrozen parameters should have gradients
            )
        else:
            # No refinable parameters
            self.refinable_params = nn.Parameter(
                torch.empty(0, dtype=self.dtype, device=self.device),
                requires_grad=False
            )
        
        # Update masks
        self.refinable_mask = new_refinable_mask
        self.fixed_mask = ~new_refinable_mask
    
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
        Get a boolean mask in FULL atom space indicating which atoms are refinable.
        
        Returns:
            Boolean tensor of shape (n_atoms,) where True indicates the atom's
            occupancy is refinable (though it shares with others in its group).
        """
        return self._expand_values(self.refinable_mask.float()).bool()
    
    def get_frozen_atoms(self) -> torch.Tensor:
        """
        Get a boolean mask in FULL atom space indicating which atoms are frozen.
        
        Returns:
            Boolean tensor of shape (n_atoms,) where True indicates the atom's
            occupancy is frozen.
        """
        return self._expand_values(self.fixed_mask.float()).bool()
    
    def get_refinable_count(self) -> int:
        """
        Get the number of refinable parameters in COMPRESSED space.
        
        Note: This is the number of refinable groups, not atoms.
        Use get_refinable_atoms().sum() to get the number of refinable atoms.
        
        Returns:
            Number of refinable compressed parameters
        """
        return self.refinable_mask.sum().item()
    
    def get_fixed_count(self) -> int:
        """
        Get the number of fixed parameters in COMPRESSED space.
        
        Note: This is the number of fixed groups, not atoms.
        Use get_frozen_atoms().sum() to get the number of frozen atoms.
        
        Returns:
            Number of fixed compressed parameters
        """
        return self.fixed_mask.sum().item()
    
    def update_refinable_mask(self, new_mask: torch.Tensor, in_compressed_space: bool = False):
        """
        Directly update the refinable mask with a new mask.
        
        This method allows more direct control over which parameters are refinable,
        compared to freeze/unfreeze which modify the existing state. This is useful
        when you want to set a specific refinement pattern from scratch.
        
        Args:
            new_mask: Boolean tensor indicating which parameters should be refinable.
                     If in_compressed_space=False (default): shape (n_atoms,) in full atom space
                     If in_compressed_space=True: shape (n_groups,) in compressed space
            in_compressed_space: If True, new_mask is in compressed space.
                                If False (default), new_mask is in full atom space and will be collapsed.
        
        Example (full atom space):
            >>> # Refine only first 100 atoms
            >>> atom_mask = torch.zeros(n_atoms, dtype=torch.bool)
            >>> atom_mask[:100] = True
            >>> occ.update_refinable_mask(atom_mask, in_compressed_space=False)
        
        Example (compressed space):
            >>> # Refine only even-indexed groups
            >>> group_mask = torch.zeros(n_groups, dtype=torch.bool)
            >>> group_mask[::2] = True
            >>> occ.update_refinable_mask(group_mask, in_compressed_space=True)
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
                new_refinable_values,
                requires_grad=True
            )
        else:
            # No refinable parameters
            self.refinable_params = nn.Parameter(
                torch.empty(0, dtype=self.dtype, device=self.device),
                requires_grad=False
            )
        
        # Update masks
        self.refinable_mask = collapsed_mask
        self.fixed_mask = ~collapsed_mask

    @staticmethod
    def from_residue_groups(initial_values: torch.Tensor, 
                           pdb_dataframe,
                           refinable_mask: Optional[torch.Tensor] = None,
                           **kwargs) -> 'OccupancyTensor':
        """
        Create an OccupancyTensor where all atoms in each residue share the same occupancy.
        
        This is a common use case where all atoms in a residue should have the same occupancy.
        
        Args:
            initial_values: Initial occupancy values for all atoms
            pdb_dataframe: Pandas DataFrame with PDB data (must have 'resname', 'resseq', 'chainid')
            refinable_mask: Optional mask for refinable atoms
            **kwargs: Additional arguments passed to OccupancyTensor constructor
        
        Returns:
            OccupancyTensor with residue-based sharing groups
        """
        # Group atoms by residue
        grouped = pdb_dataframe.groupby(['resname', 'resseq', 'chainid', 'altloc'])
        
        n_atoms = len(initial_values)
        sharing_groups_tensor = torch.arange(n_atoms, dtype=torch.long)
        collapsed_idx = 0
        
        for (resname, resseq, chainid, altloc), group in grouped:
            indices = group['index'].tolist()
            if len(indices) > 1:  # Only create group if more than one atom
                sharing_groups_tensor[indices] = collapsed_idx
                collapsed_idx += 1
        
        # Compact the indices
        unique_indices = torch.unique(sharing_groups_tensor, sorted=True)
        for new_idx, old_idx in enumerate(unique_indices):
            mask = (sharing_groups_tensor == old_idx)
            sharing_groups_tensor[mask] = new_idx
        
        return OccupancyTensor(
            initial_values=initial_values,
            sharing_groups=sharing_groups_tensor,
            refinable_mask=refinable_mask,
            name='occupancy',
            **kwargs
        )
    
    def copy(self) -> 'OccupancyTensor':
        """
        Create a deep copy of this OccupancyTensor.
        
        Creates a complete independent copy with all buffers and parameters,
        including sharing groups, altloc groups, and collapsed storage structures.
        
        Returns:
            New OccupancyTensor instance with copied data
        """
        # Get current occupancy values in normal space (full atom space)
        current_occ = self.forward().detach()
        
        # Reconstruct refinable mask in full space
        full_refinable_mask = self._expand_values(self.refinable_mask.float()).bool()
        
        # Reconstruct altloc groups from the linked_occ buffers
        altloc_groups = []
        if hasattr(self, 'linked_occ_sizes'):
            for n_conf in self.linked_occ_sizes:
                linked_indices = getattr(self, f'linked_occ_{n_conf}')  # shape (N_groups, n_conf)
                
                # For each group of linked conformations
                for group_collapsed_indices in linked_indices:
                    # Find all atoms that map to each collapsed index
                    conf_atom_lists = []
                    for collapsed_idx in group_collapsed_indices:
                        atom_indices = (self.expansion_mask == collapsed_idx).nonzero(as_tuple=False).squeeze(-1)
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
            use_sigmoid=self.use_sigmoid
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

class PassThroughTensor(nn.Module):
    """
    A simple parameter wrapper that passes the parameter through unchanged.
    
    This is useful as a placeholder or for parameters that do not require
    any special handling.
    """
    def __init__(self, 
                 initial_values: torch.Tensor, 
                 requires_grad: bool = True,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 name: Optional[str] = None):
        """
        Initialize the PassThroughTensor.
        
        Args:
            initial_values: Initial tensor values
            requires_grad: Whether the parameter requires gradients
            dtype: Data type of the tensor
            device: Device to place the tensor on
            name: Optional name for the parameter
        """
        super().__init__(
            initial_values=initial_values,
            requires_grad=requires_grad,
            dtype=dtype,
            device=device,
            name=name
        )
    
    def forward(self) -> torch.Tensor:
        """
        Return the parameter value unchanged.
        
        Returns:
            The parameter tensor
        """
        return self.param