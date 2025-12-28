"""
Hyperparameter Registration for PyTorch Modules.

This module provides a mechanism for tracking hyperparameters separately from
parameters (trainable weights) and buffers (persistent state). This allows for:

1. Easy access to all hyperparameters via `module.hyperparameters()`
2. Separate state_dict for hyperparameters via `module.hyperparameter_state_dict()`
3. Clean separation between model weights, state, and configuration

Design Pattern:
- Hyperparameters are registered as buffers internally (for device tracking)
- They are also tracked in a separate `_hyperparameters` set
- This allows filtering them out or including them as needed

Usage:
    class MyModule(HyperparameterMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.register_hyperparameter('learning_rate', 0.01)
            self.register_hyperparameter('sigma', 2.0)
            
    # Access all hyperparameters
    for name, value in module.hyperparameters():
        print(f"{name}: {value}")
    
    # Get just hyperparameter state dict
    hp_state = module.hyperparameter_state_dict()
    
    # Load hyperparameters
    module.load_hyperparameter_state_dict(hp_state)
"""

import torch
from torch import nn
from typing import Dict, Iterator, Tuple, Any, Optional, Set
from collections import OrderedDict


class HyperparameterMixin:
    """
    Mixin class providing hyperparameter registration and tracking.
    
    Use this with nn.Module to add hyperparameter tracking capabilities.
    Hyperparameters are stored as buffers but tracked separately.
    
    Examples
    --------
    >>> class MyTarget(HyperparameterMixin, nn.Module):
    ...     def __init__(self, sigma=1.0, target_value=0.0):
    ...         nn.Module.__init__(self)
    ...         HyperparameterMixin.__init__(self)
    ...         self.register_hyperparameter('sigma', sigma)
    ...         self.register_hyperparameter('target_value', target_value)
    ...
    >>> target = MyTarget(sigma=2.5)
    >>> dict(target.hyperparameters())
    {'sigma': tensor(2.5), 'target_value': tensor(0.0)}
    """
    
    def __init__(self):
        """Initialize hyperparameter tracking."""
        # Set to track which buffers are hyperparameters
        # Use object.__setattr__ to avoid triggering nn.Module's __setattr__
        if not hasattr(self, '_hyperparameter_names'):
            object.__setattr__(self, '_hyperparameter_names', set())
    
    def register_hyperparameter(
        self, 
        name: str, 
        value: float,
        persistent: bool = True
    ) -> None:
        """
        Register a hyperparameter.
        
        Hyperparameters are stored as buffers (so they move with .to(device))
        but tracked separately for easy access and state_dict operations.
        
        Parameters
        ----------
        name : str
            Name of the hyperparameter. Will be prefixed with '_hp_' internally.
        value : float
            Initial value of the hyperparameter.
        persistent : bool, optional
            If True, hyperparameter will be part of module's state_dict.
            Default is True.
            
        Examples
        --------
        >>> self.register_hyperparameter('sigma', 2.0)
        >>> self.sigma  # Access via property (if defined) or _hp_sigma
        2.0
        """
        # Initialize tracking set if needed
        if not hasattr(self, '_hyperparameter_names'):
            object.__setattr__(self, '_hyperparameter_names', set())
        
        # Internal buffer name
        buffer_name = f'_hp_{name}'
        
        # Register as buffer
        tensor_value = torch.tensor(value, dtype=torch.float32)
        self.register_buffer(buffer_name, tensor_value, persistent=persistent)
        
        # Track as hyperparameter
        self._hyperparameter_names.add(name)
    
    def get_hyperparameter(self, name: str) -> torch.Tensor:
        """
        Get a hyperparameter value.
        
        Parameters
        ----------
        name : str
            Name of the hyperparameter.
            
        Returns
        -------
        torch.Tensor
            The hyperparameter tensor.
        """
        buffer_name = f'_hp_{name}'
        return getattr(self, buffer_name)
    
    def set_hyperparameter(self, name: str, value: float) -> None:
        """
        Set a hyperparameter value.
        
        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        value : float
            New value.
        """
        buffer_name = f'_hp_{name}'
        getattr(self, buffer_name).fill_(value)
    
    def hyperparameters(self, recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Return an iterator over module hyperparameters.
        
        Parameters
        ----------
        recurse : bool, optional
            If True, include hyperparameters from submodules. Default is True.
            
        Yields
        ------
        tuple
            (name, tensor) pairs for each hyperparameter.
            
        Examples
        --------
        >>> for name, hp in module.hyperparameters():
        ...     print(f"{name}: {hp.item()}")
        """
        # Yield own hyperparameters
        if hasattr(self, '_hyperparameter_names'):
            for name in self._hyperparameter_names:
                buffer_name = f'_hp_{name}'
                yield name, getattr(self, buffer_name)
        
        # Recursively yield from submodules
        if recurse:
            for module_name, module in self.named_modules():
                if module is self:
                    continue
                if hasattr(module, '_hyperparameter_names'):
                    for hp_name in module._hyperparameter_names:
                        buffer_name = f'_hp_{hp_name}'
                        full_name = f'{module_name}.{hp_name}' if module_name else hp_name
                        yield full_name, getattr(module, buffer_name)
    
    def named_hyperparameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Return an iterator over module hyperparameters with names.
        
        Same as hyperparameters() but allows prefix specification.
        
        Parameters
        ----------
        prefix : str, optional
            Prefix to prepend to names. Default is ''.
        recurse : bool, optional
            If True, include hyperparameters from submodules. Default is True.
            
        Yields
        ------
        tuple
            (name, tensor) pairs for each hyperparameter.
        """
        for name, hp in self.hyperparameters(recurse=recurse):
            full_name = f'{prefix}{name}' if prefix else name
            yield full_name, hp
    
    def hyperparameter_state_dict(self, prefix: str = '') -> Dict[str, torch.Tensor]:
        """
        Return a dictionary containing only hyperparameters.
        
        Unlike state_dict() which includes all buffers and parameters,
        this only returns the hyperparameters.
        
        Parameters
        ----------
        prefix : str, optional
            Prefix to prepend to names. Default is ''.
            
        Returns
        -------
        dict
            Dictionary mapping hyperparameter names to tensors.
            
        Examples
        --------
        >>> hp_state = module.hyperparameter_state_dict()
        >>> torch.save(hp_state, 'hyperparameters.pt')
        """
        result = OrderedDict()
        for name, hp in self.named_hyperparameters(prefix=prefix, recurse=True):
            result[name] = hp.clone()
        return result
    
    def load_hyperparameter_state_dict(
        self, 
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True
    ) -> None:
        """
        Load hyperparameters from a state dict.
        
        Parameters
        ----------
        state_dict : dict
            Dictionary of hyperparameter name -> tensor.
        strict : bool, optional
            If True, raise error on missing/unexpected keys. Default is True.
            
        Examples
        --------
        >>> hp_state = torch.load('hyperparameters.pt')
        >>> module.load_hyperparameter_state_dict(hp_state)
        """
        # Get all hyperparameter names (with module paths)
        own_hp_names = set(name for name, _ in self.named_hyperparameters(recurse=True))
        
        missing_keys = []
        unexpected_keys = []
        
        for name, value in state_dict.items():
            if name in own_hp_names:
                # Find the module and set the value
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    module_path, hp_name = parts
                    # Navigate to submodule
                    module = self
                    for part in module_path.split('.'):
                        module = getattr(module, part)
                    module.set_hyperparameter(hp_name, value.item())
                else:
                    hp_name = parts[0]
                    self.set_hyperparameter(hp_name, value.item())
            else:
                unexpected_keys.append(name)
        
        # Check for missing
        for name in own_hp_names:
            if name not in state_dict:
                missing_keys.append(name)
        
        if strict:
            if missing_keys:
                raise RuntimeError(f"Missing hyperparameters: {missing_keys}")
            if unexpected_keys:
                raise RuntimeError(f"Unexpected hyperparameters: {unexpected_keys}")
    
    def hyperparameter_dict(self) -> Dict[str, float]:
        """
        Return hyperparameters as a simple Python dict of floats.
        
        Useful for logging, serialization to JSON, etc.
        
        Returns
        -------
        dict
            Dictionary mapping hyperparameter names to float values.
            
        Examples
        --------
        >>> params = module.hyperparameter_dict()
        >>> import json
        >>> json.dumps(params)  # JSON serializable
        """
        return {name: hp.item() for name, hp in self.hyperparameters(recurse=True)}
    
    def print_hyperparameters(self, prefix: str = '') -> None:
        """
        Print all hyperparameters in a formatted way.
        
        Parameters
        ----------
        prefix : str, optional
            Prefix for indentation. Default is ''.
        """
        print(f"{prefix}Hyperparameters:")
        for name, hp in sorted(self.hyperparameters(recurse=True)):
            print(f"{prefix}  {name}: {hp.item():.6g}")


def create_hyperparameter_property(name: str) -> property:
    """
    Create a property for convenient hyperparameter access.
    
    Use this to create getter/setter properties that access
    the underlying registered hyperparameter.
    
    Parameters
    ----------
    name : str
        Name of the hyperparameter.
        
    Returns
    -------
    property
        A property object for the hyperparameter.
        
    Examples
    --------
    >>> class MyModule(HyperparameterMixin, nn.Module):
    ...     sigma = create_hyperparameter_property('sigma')
    ...     
    ...     def __init__(self, sigma=1.0):
    ...         super().__init__()
    ...         self.register_hyperparameter('sigma', sigma)
    ...
    >>> m = MyModule(sigma=2.5)
    >>> m.sigma  # Uses the property
    2.5
    >>> m.sigma = 3.0  # Sets via property
    """
    def getter(self):
        return self.get_hyperparameter(name).item()
    
    def setter(self, value):
        self.set_hyperparameter(name, value)
    
    return property(getter, setter, doc=f"Hyperparameter: {name}")
