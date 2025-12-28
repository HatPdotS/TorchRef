from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
from torch import nn
from torch.nn.functional import softplus
from torchref.utils.utils import ModuleReference
from torchref.utils.stats import (
    StatEntry, stat, filter_stats, flatten_stats,
    VERBOSITY_ESSENTIAL, VERBOSITY_STANDARD, VERBOSITY_DETAILED, VERBOSITY_DEBUG
)


class WeightingScheme(nn.Module, ABC):
    """
    Abstract base class for weighting schemes.
    
    All tunable parameters should be registered as buffers using register_buffer()
    so they can be accessed/modified via state_dict notation, e.g.:
        refinement.component_weighting.schemes.0.target_gap
    """
    name = 'base_weighting_scheme'

    def __init__(self, refinement):
        super().__init__()
        self.refinement = ModuleReference(refinement)
    
    def stats(self) -> Dict[str, StatEntry]:
        """
        Return statistics for reporting.
        
        Returns dict with StatEntry values containing verbosity levels.
        Use filter_stats() to filter by verbosity.
        """
        return {}

class TargetOffsetWeighting(WeightingScheme):
    """
    Weights components based on distance from target value.
    weight = 1 + softplus(loss - target_value) / sigma
    """
    name = 'target_offset_weighting'

    def get_weight(self, target):
        if not hasattr(target, 'target_value') or not hasattr(target, 'sigma'):
            # Return 1.0 if attributes missing
            return torch.tensor(1.0, device=self.refinement.device)
        
        # We need to compute loss to get the weight
        loss = target()
        if isinstance(loss, torch.Tensor) and loss.numel() > 1:
            loss = loss.sum()
            
        weight = 1 + softplus(loss - target.target_value) / (target.sigma + 1e-6)
        return weight.detach()
    
    def forward(self):
        weights = {}
        
        # Geometry targets
        if hasattr(self.refinement, 'geometry_target'):
            for name, target in self.refinement.geometry_target.targets().items():
                weights[name] = self.get_weight(target)
                
        # ADP targets
        if hasattr(self.refinement, 'adp_target'):
            for name, target in self.refinement.adp_target.targets().items():
                weights[name] = self.get_weight(target)
                
        return weights

    def stats(self):
        stats = {}
        if hasattr(self.refinement, 'geometry_target'):
            for name, target in self.refinement.geometry_target.targets().items():
                stats[f'{name}_weight'] = stat(self.get_weight(target).item(), VERBOSITY_STANDARD)
        if hasattr(self.refinement, 'adp_target'):
            for name, target in self.refinement.adp_target.targets().items():
                stats[f'{name}_weight'] = stat(self.get_weight(target).item(), VERBOSITY_STANDARD)
        return stats

class OverfittingWeighting(WeightingScheme):
    """
    Scale-invariant overfitting correction weight.
    
    Tunable parameters (as buffers):
    - target_gap: float, target overfitting gap threshold
    - min_weight: float, minimum weight value
    - sharpness: float, exponential sharpness
    - smoothing: float, exponential smoothing factor (0-1)
    """
    name = 'overfitting_weighting'
    def __init__(self, refinement, target_gap: float = 0.2, min_weight: float = 0.2, 
                 sharpness: float = 5.0, smoothing: float = 0.8):
        super().__init__(refinement)
        # Register tunable parameters as buffers for state_dict access
        self.register_buffer('target_gap', torch.tensor(target_gap))
        self.register_buffer('min_weight', torch.tensor(min_weight))
        self.register_buffer('sharpness', torch.tensor(sharpness))
        self.register_buffer('smoothing', torch.tensor(smoothing))
        self.register_buffer('weight_reg', torch.tensor(1.0))  # Current weight value

    def forward(self):
        train_nll = self.refinement.xray_target_work().detach()
        test_nll = self.refinement.xray_target_test().detach()
        scale = torch.abs(test_nll) + 1e-6
        relative_gap = (test_nll - train_nll) / scale
        
        # Calculate target weight with hard penalty
        target_weight = self.min_weight + torch.exp(self.sharpness * (relative_gap - self.target_gap))
        target_weight = target_weight.detach()  # Detach to prevent gradient flow through weights
        
        # Apply smoothing
        self.weight_reg = self.smoothing * self.weight_reg + (1 - self.smoothing) * target_weight
            
        # Apply to all regularization terms (geom and adp)
        return {'geom': self.weight_reg.detach(), 'adp': self.weight_reg.detach()}

    def stats(self):
        train_nll = self.refinement.xray_target_work().detach().item()
        test_nll = self.refinement.xray_target_test().detach().item()
        return {
            'overfitting_weight': stat(self.weight_reg.item(), VERBOSITY_ESSENTIAL),
            'target_gap': stat(self.target_gap.item(), VERBOSITY_DEBUG),
            'min_weight': stat(self.min_weight.item(), VERBOSITY_DEBUG),
            'sharpness': stat(self.sharpness.item(), VERBOSITY_DEBUG),
            'train_nll': stat(train_nll, VERBOSITY_STANDARD),
            'test_nll': stat(test_nll, VERBOSITY_STANDARD),
        }

class ManualWeighting(WeightingScheme):
    name = 'manual_weighting'
    def __init__(self, refinement, weights: Dict[str, float]):
        super().__init__(refinement)
        self.weights = weights
        
    def forward(self):
        # Return multipliers from manual weights
        device = self.refinement.device
        return {k: torch.tensor(v, device=device) if isinstance(v, (float, int)) else v 
                for k, v in self.weights.items()}


class XrayScaleWeighting(WeightingScheme):
    """
    Scale X-ray weight so that the effective X-ray loss is at a fixed target value.
    
    The X-ray NLL can vary widely (5-200+) depending on data quality, resolution,
    and model state. This scheme computes a weight such that:
    
        weight * xray_loss ≈ target_scale
        
    So: weight = target_scale / xray_loss
    
    This normalizes the X-ray contribution to a consistent scale, making the
    relative weights of geometry/ADP restraints more interpretable and stable.
    
    Tunable parameters (as buffers):
    - target_scale: float, target effective X-ray loss value
    - min_weight: float, minimum allowed weight
    - max_weight: float, maximum allowed weight
    - smoothing: float, exponential smoothing factor (0-1)
    
    Parameters
    ----------
    refinement : Refinement
        Reference to Refinement object.
    target_scale : float, optional
        Target effective X-ray loss value. Default is 50.0.
    min_weight : float, optional
        Minimum allowed weight (prevents division by small loss). Default is 0.01.
    max_weight : float, optional
        Maximum allowed weight (prevents explosion when loss is tiny). Default is 10.0.
    smoothing : float, optional
        Exponential smoothing factor (0-1). Higher = more smoothing. Default is 0.5.
    """
    
    name = 'xray_scale_weighting'
    
    def __init__(self, refinement, target_scale: float = 50.0, 
                 min_weight: float = 0.01, max_weight: float = 10.0,
                 smoothing: float = 0.5):
        super().__init__(refinement)
        # Register tunable parameters as buffers for state_dict access
        self.register_buffer('target_scale', torch.tensor(target_scale))
        self.register_buffer('min_weight', torch.tensor(min_weight))
        self.register_buffer('max_weight', torch.tensor(max_weight))
        self.register_buffer('smoothing', torch.tensor(smoothing))
        self.register_buffer('xray_weight', torch.tensor(1.0))  # Current weight value
        self.register_buffer('xray_loss_initial', self.refinement.xray_target_work().detach())
        self._raw_xray_loss = None
    
    def forward(self):

        # weight = target_scale / loss
        target_weight = self.target_scale / (self.xray_loss_initial + 1e-6)
        
        # Clamp to reasonable range
        target_weight = torch.clamp(target_weight, self.min_weight.item(), self.max_weight.item())
        target_weight = target_weight.detach()  # Detach to prevent gradient flow through weights
        
        # Apply smoothing
        self.xray_weight = self.smoothing * self.xray_weight + (1 - self.smoothing) * target_weight
        
        return {'xray': self.xray_weight.detach()}
    
    def stats(self):
        return {
            'xray_weight': stat(self.xray_weight.item(), VERBOSITY_ESSENTIAL),
            'raw_xray_loss': stat(self._raw_xray_loss if self._raw_xray_loss is not None else 0.0, VERBOSITY_STANDARD),
            'effective_xray_loss': stat(
                (self._raw_xray_loss * self.xray_weight.item()) if (self._raw_xray_loss and self.xray_weight is not None) else 0.0,
                VERBOSITY_STANDARD
            ),
            'target_scale': stat(self.target_scale.item(), VERBOSITY_DEBUG),
            'min_weight': stat(self.min_weight.item(), VERBOSITY_DEBUG),
            'max_weight': stat(self.max_weight.item(), VERBOSITY_DEBUG),
        }

class ComponentWeighting(nn.Module):
    """
    Combines multiple weighting schemes using nn.ModuleDict for clean organization.

    Default schemes:
    - 'xray_scale': XrayScaleWeighting - normalizes X-ray loss to consistent scale
    - 'target_offset': TargetOffsetWeighting - weights based on distance from target
    - 'overfitting': OverfittingWeighting - prevents overfitting via Rfree gap

    Parameters
    ----------
    refinement : Refinement
        Reference to Refinement object.
    weights : dict, optional
        Manual weight overrides (deprecated, use component_weights).
    component_weights : dict, optional
        Manual weight overrides for specific components.
    schemes : list of WeightingScheme, optional
        Additional custom weighting schemes.

    Attributes
    ----------
    schemes : nn.ModuleDict
        Dictionary of weighting schemes.
    weights : dict
        Current computed weights for each component.

    Examples
    --------
    >>> cw = ComponentWeighting(refinement)
    >>> cw.update_weights()
    >>> xray_scheme = cw['xray_scale']
    >>> xray_scheme.target_scale.fill_(100.0)
    """

    def __init__(self, refinement, weights: Dict[str, float] = None, component_weights: Dict[str, float] = None, schemes: List[WeightingScheme] = None):
        super().__init__()
        self.refinement = ModuleReference(refinement)
        
        # Build schemes dict
        schemes_dict = {
            'xray_scale': XrayScaleWeighting(refinement),
            'target_offset': TargetOffsetWeighting(refinement),
            'overfitting': OverfittingWeighting(refinement),
        }
        
        # Add manual weights if provided
        manual_weights_dict = {}
        if weights:
            manual_weights_dict.update(weights)
        if component_weights:
            manual_weights_dict.update(component_weights)
            
        if manual_weights_dict:
            schemes_dict['manual'] = ManualWeighting(refinement, manual_weights_dict)
            
        # Add additional schemes
        if schemes:
            for i, scheme in enumerate(schemes):
                key = getattr(scheme, 'name', f'custom_{i}')
                schemes_dict[key] = scheme
        
        self.schemes = nn.ModuleDict(schemes_dict)
        self.weights = {}
        self.update_weights()
    
    def __getitem__(self, key: str) -> WeightingScheme:
        """Get a scheme by name using dictionary-style access."""
        return self.schemes[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if a scheme exists."""
        return key in self.schemes
    
    def keys(self):
        """Return scheme names."""
        return self.schemes.keys()
    
    def values(self):
        """Return scheme instances."""
        return self.schemes.values()
    
    def items(self):
        """Return (name, scheme) pairs."""
        return self.schemes.items()
        
    def add_scheme(self, name: str, scheme: WeightingScheme):
        """
        Add a new weighting scheme.

        Parameters
        ----------
        name : str
            Name/key for the scheme.
        scheme : WeightingScheme
            The scheme to add.
        """
        self.schemes[name] = scheme
        
    def update_weights(self):
        """
        Update weights by combining all scheme outputs.

        Each scheme's weights are multiplied together for shared keys.

        Returns
        -------
        dict
            Dictionary of computed weights for each component.
        """
        # Initialize weights for all known components to 1.0
        self.weights = {}
        
        for scheme in self.schemes.values():
            scheme_weights = scheme()
            for k, v in scheme_weights.items():
                if k in self.weights:
                    self.weights[k] = self.weights[k] * v
                else:
                    self.weights[k] = v

        return self.weights


    def total_loss(self):
        total = 0.0
        
        # X-ray
        total += self.refinement.xray_target_work() * self.weights.get('xray', 1.0)
        
        # Geometry
        if hasattr(self.refinement, 'geometry_target'):
            losses = self.refinement.geometry_target.target_losses()
            for name, loss in losses.items():
                if name in self.weights:
                    total += loss * self.weights[name]
                    
        # ADP
        if hasattr(self.refinement, 'adp_target'):
            losses = self.refinement.adp_target.target_losses()
            for name, loss in losses.items():
                if name in self.weights:
                    total += loss * self.weights[name]
                    
        return total

    def stats(self):
        """
        Return statistics for reporting.
        
        Returns full stats dictionary with StatEntry values. Use filter_stats()
        at the caller level to filter by verbosity when needed.
            
        Returns
        -------
        dict
            Stats dictionary with StatEntry objects containing verbosity metadata.
        """
        stats = {}
        
        # Collect stats from schemes
        for name, scheme in self.schemes.items():
            scheme_stats = scheme.stats()
            if scheme_stats:
                stats[name] = scheme_stats
            
        # Add current weights (essential for monitoring)
        stats['weights'] = {
            k: stat(v.item() if isinstance(v, torch.Tensor) else v, VERBOSITY_STANDARD) 
            for k, v in self.weights.items()
        }

        # Add target stats
        if hasattr(self.refinement, 'geometry_target'):
            geom_stats = self.refinement.geometry_target.stats()
            if geom_stats:
                # Wrap raw values with VERBOSITY_DETAILED if not already StatEntry
                stats['geom_target'] = {
                    k: v if isinstance(v, StatEntry) else stat(v, VERBOSITY_DETAILED)
                    for k, v in geom_stats.items()
                }
        if hasattr(self.refinement, 'adp_target'):
            adp_stats = self.refinement.adp_target.stats()
            if adp_stats:
                stats['adp_target'] = {
                    k: v if isinstance(v, StatEntry) else stat(v, VERBOSITY_DETAILED)
                    for k, v in adp_stats.items()
                }

        # Add xray stats (essential)
        stats['xray'] = {
            'work_nll': stat(self.refinement.xray_target_work().item(), VERBOSITY_ESSENTIAL),
            'test_nll': stat(self.refinement.xray_target_test().item(), VERBOSITY_ESSENTIAL),
        }

        return stats


__all__ = [
    # Weighting classes
    'WeightingScheme',
    'TargetOffsetWeighting',
    'OverfittingWeighting',
    'ManualWeighting',
    'XrayScaleWeighting',
    'ComponentWeighting',
]
