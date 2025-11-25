import torch
import json
import torch.nn as nn
from torchref.utils.debug_utils import DebugMixin
# Dictionary storing all non-crystallographic symmetry operations
# Format: {spacegroup_canonical_name: (rotation_matrices, translation_vectors)}

import os
symmetry_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'caching/files/gemmi_symmetry_operations.pt')
SYMMETRY_OPERATIONS = torch.load(symmetry_path)

# Dictionary mapping different space group names/aliases to canonical identifiers
# This allows for flexible space group name input while maintaining consistency
# Loaded from JSON file for easier maintenance

mapping_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'caching/files/spacegroup_name_mapping.json')

with open(mapping_path, 'r') as f: SPACEGROUP_NAME_MAPPING = json.load(f)

class Symmetry(DebugMixin, nn.Module):
    def __init__(self, space_group, dtype=torch.float64,device=torch.device('cpu')):
        super(Symmetry, self).__init__()
        self.device = device
        self.space_group = space_group.strip().replace(' ','')
        self.canonical_space_group = self._resolve_space_group_name(self.space_group)
        matrices, translations = self._get_ops(self.canonical_space_group)
        matrices = matrices.to(dtype).to(self.device)  # Ensure matrices are of the correct dtype
        translations = translations.to(dtype).to(self.device)  # Ensure translations are of the correct dtype
        self.register_buffer('matrices', matrices)
        self.register_buffer('translations', translations)


    def _resolve_space_group_name(self, space_group):
        """
        Resolve space group name to canonical identifier using the name mapping.
        Uses space-removed canonicalization for flexible matching.
        
        Parameters:
        -----------
        space_group : str
            Input space group name (with any common variations/aliases)
            
        Returns:
        --------
        str
            Canonical space group identifier used in SYMMETRY_OPERATIONS
            
        Raises:
        -------
        ValueError
            If space group name is not recognized
        """
        # First try direct lookup
        if space_group in SPACEGROUP_NAME_MAPPING:
            return SPACEGROUP_NAME_MAPPING[space_group]
        
        # Try with spaces removed (canonical form)
        canonical_input = space_group.replace(' ', '')
        
        # Try direct lookup of canonicalized name in SYMMETRY_OPERATIONS
        if canonical_input in SYMMETRY_OPERATIONS:
            return canonical_input
        
        # Try case-insensitive lookup with spaces removed in name mapping
        for key, value in SPACEGROUP_NAME_MAPPING.items():
            if key.replace(' ', '').upper() == canonical_input.upper():
                return value
        
        # If still not found, check SYMMETRY_OPERATIONS directly (case-insensitive)
        for key in SYMMETRY_OPERATIONS:
            if key.replace(' ', '').upper() == canonical_input.upper():
                return key
        
        available_names = list(SYMMETRY_OPERATIONS.keys())[:20]  # Show first 20
        raise ValueError(f'Space group "{space_group}" not recognized. '
                       f'Available space groups (first 20): {available_names}...')

    def _get_ops(self, canonical_space_group):
        """
        Get symmetry operations for the canonical space group name.
        
        Parameters:
        -----------
        canonical_space_group : str
            Canonical space group identifier
            
        Returns:
        --------
        tuple
            (rotation_matrices, translation_vectors) as torch tensors
            
        Raises:
        -------
        ValueError
            If canonical space group is not implemented
        """
        if canonical_space_group in SYMMETRY_OPERATIONS:
            matrices, translations = SYMMETRY_OPERATIONS[canonical_space_group]
            # Return deep copies to avoid modifying the stored tensors
            return matrices.clone(), translations.clone()
        else:
            available_groups = list(SYMMETRY_OPERATIONS.keys())
            raise ValueError(f'Space group "{canonical_space_group}" not implemented. '
                           f'Available space groups: {available_groups}')

    def apply(self, fractional_coords):
        """
        Apply symmetry operations to fractional coordinates.

        Parameters:
        -----------
        fractional_coords : torch.Tensor
            Input tensor of shape (N, 3) representing fractional coordinates
        Returns:
        --------
        torch.Tensor
            Transformed coordinates of shape (3, N, ops) where ops is the number of symmetry operations
        """
        coords = fractional_coords.reshape(3, -1).to(self.matrices.device)  # (3, N)
        coords = coords.unsqueeze(0)  # (1, 3, N)
        transformed = torch.matmul(self.matrices, coords) + self.translations.unsqueeze(2)
        # transformed: (ops, 3, N)
        return transformed.permute(1, 2, 0)  # (3, N, ops)

    def forward(self, fractional_coords):
        return self.apply(fractional_coords)
    
    def get_grid_requirements(self):
        """
        Analyze symmetry operations to determine grid size requirements.
        
        Examines all rotation matrices and translations to determine which
        grid dimensions must satisfy divisibility constraints for exact
        integer indexing (interpolation-free symmetry expansion).
        
        Returns:
        --------
        dict : {'nx_mod': int, 'ny_mod': int, 'nz_mod': int}
            Required divisibility for each axis.
            For example: {'nx_mod': 1, 'ny_mod': 2, 'nz_mod': 1}
            means ny must be divisible by 2.
        
        Example:
        --------
        >>> sym = Symmetry('P21')
        >>> req = sym.get_grid_requirements()
        >>> print(req)  # {'nx_mod': 1, 'ny_mod': 2, 'nz_mod': 1}
        """
        import math
        from fractions import Fraction
        
        # Start with no requirements
        nx_lcm = 1
        ny_lcm = 1
        nz_lcm = 1
        
        # Analyze each symmetry operation
        for i in range(self.matrices.shape[0]):
            trans = self.translations[i].cpu()
            
            # For each axis, check if translation has fractional component
            for axis_idx in range(3):
                t = float(trans[axis_idx])
                
                # Convert to fraction and get denominator
                # The grid size must be divisible by this denominator
                frac = Fraction(t).limit_denominator(12)  # Limit to denominators up to 12
                denom = frac.denominator
                
                if axis_idx == 0:
                    nx_lcm = math.lcm(nx_lcm, denom)
                elif axis_idx == 1:
                    ny_lcm = math.lcm(ny_lcm, denom)
                else:
                    nz_lcm = math.lcm(nz_lcm, denom)
        
        return {
            'nx_mod': nx_lcm,
            'ny_mod': ny_lcm,
            'nz_mod': nz_lcm
        }
    
    def check_grid_compatibility(self, grid_shape):
        """
        Check if a grid size is compatible with the symmetry operations.
        
        Parameters:
        -----------
        grid_shape : tuple of int
            (nx, ny, nz) grid dimensions
        
        Returns:
        --------
        dict with keys:
            'compatible': bool
                True if grid satisfies all symmetry requirements
            'can_use_direct_indexing': bool
                True if interpolation-free expansion is possible
            'issues': list of str
                Descriptions of incompatibilities (empty if compatible)
            'requirements': dict
                Required divisibility from get_grid_requirements()
        
        Example:
        --------
        >>> sym = Symmetry('P21')
        >>> result = sym.check_grid_compatibility((131, 163, 148))
        >>> print(result['compatible'])  # False
        >>> print(result['issues'])  # ['ny=163 not divisible by 2']
        """
        nx, ny, nz = grid_shape
        requirements = self.get_grid_requirements()
        
        issues = []
        
        if nx % requirements['nx_mod'] != 0:
            issues.append(
                f"nx={nx} not divisible by {requirements['nx_mod']} "
                f"(required for {self.space_group})"
            )
        
        if ny % requirements['ny_mod'] != 0:
            issues.append(
                f"ny={ny} not divisible by {requirements['ny_mod']} "
                f"(required for {self.space_group})"
            )
        
        if nz % requirements['nz_mod'] != 0:
            issues.append(
                f"nz={nz} not divisible by {requirements['nz_mod']} "
                f"(required for {self.space_group})"
            )
        
        compatible = len(issues) == 0
        
        return {
            'compatible': compatible,
            'can_use_direct_indexing': compatible,
            'issues': issues,
            'requirements': requirements
        }
    
    def suggest_grid_size(self, min_grid_shape, make_fft_friendly=True):
        """
        Suggest an optimal grid size that satisfies symmetry requirements.
        
        Given a minimum grid size, finds the nearest larger size that:
        1. Satisfies symmetry requirements (divisibility constraints)
        2. Optionally, is FFT-friendly (factors of 2, 3, 5 only)
        
        Parameters:
        -----------
        min_grid_shape : tuple of int
            Minimum (nx, ny, nz) grid dimensions
        make_fft_friendly : bool, default True
            If True, ensures result has only factors of 2, 3, 5
        
        Returns:
        --------
        tuple of int : (nx, ny, nz)
            Suggested grid dimensions
        
        Example:
        --------
        >>> sym = Symmetry('P21')
        >>> suggested = sym.suggest_grid_size((131, 163, 148))
        >>> print(suggested)  # (135, 164, 150) or similar
        """
        requirements = self.get_grid_requirements()
        
        def find_next_valid(n, divisibility):
            """Find next number >= n that satisfies divisibility."""
            if n % divisibility == 0:
                candidate = n
            else:
                candidate = ((n // divisibility) + 1) * divisibility
            
            if not make_fft_friendly:
                return candidate
            
            # Find next FFT-friendly number
            while not self._is_fft_friendly(candidate):
                candidate += divisibility
            
            return candidate
        
        nx_min, ny_min, nz_min = min_grid_shape
        
        nx = find_next_valid(nx_min, requirements['nx_mod'])
        ny = find_next_valid(ny_min, requirements['ny_mod'])
        nz = find_next_valid(nz_min, requirements['nz_mod'])
        
        return (nx, ny, nz)
    
    @staticmethod
    def _is_fft_friendly(n):
        """Check if number has only factors of 2, 3, and 5."""
        if n <= 0:
            return False
        
        # Remove all factors of 2, 3, 5
        while n % 2 == 0:
            n //= 2
        while n % 3 == 0:
            n //= 3
        while n % 5 == 0:
            n //= 5
        
        return n == 1

    def __repr__(self):
        return f'Symmetry(space_group={self.space_group}, canonical={self.canonical_space_group})'