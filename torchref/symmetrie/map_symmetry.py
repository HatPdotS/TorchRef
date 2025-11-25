"""
Map-level symmetry operations for electron density maps.

This module provides efficient symmetry operations applied directly to density maps,
which is much faster than applying symmetry to individual atoms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchref.symmetrie.symmetrie import Symmetry


class MapSymmetry(nn.Module):
    """
    Applies crystallographic symmetry operations to electron density maps.
    
    This class handles space group symmetry by:
    1. Taking a density map calculated for the asymmetric unit
    2. Applying rotation and translation operations in fractional coordinates
    3. Interpolating the transformed maps
    4. Summing all symmetry-related maps
    
    This is much more efficient than generating symmetry mates for each atom
    and recalculating density.
    
    Example:
    --------
    >>> map_sym = MapSymmetry(space_group='P21', map_shape=(64, 64, 64), cell_params=cell)
    >>> asymmetric_map = model.build_density_map()
    >>> symmetric_map = map_sym(asymmetric_map)
    """
    
    def __init__(self, space_group, map_shape, cell_params,dtype_float=torch.float32, verbose=1,device=torch.device('cpu')):
        """
        Initialize map symmetry operator.
        
        Parameters:
        -----------
        space_group : str
            Space group name (e.g., 'P1', 'P21', 'P-1', etc.)
        map_shape : tuple of int
            Shape of the density map (nx, ny, nz)
        cell_params : array-like, shape (6,)
            Unit cell parameters [a, b, c, alpha, beta, gamma] in Å and degrees
        """
        super().__init__()
        self.dtype_float=dtype_float
        self.space_group = space_group
        self.map_shape = tuple(map_shape)
        self.cell_params = np.array(cell_params)
        self.verbose = verbose
        self.device = device
        # Get symmetry operations
        self.symmetry = Symmetry(space_group,dtype=self.dtype_float,device=self.device)
        self.n_ops = self.symmetry.matrices.shape[0]
        
        # Check if grid is compatible for direct indexing
        compat = self.symmetry.check_grid_compatibility(map_shape)
        self.can_use_direct_indexing = compat['can_use_direct_indexing']
        
        if self.verbose > 0:
            print(f"MapSymmetry initialized for {space_group}")
            print(f"  Number of symmetry operations: {self.n_ops}")
            print(f"  Map shape: {self.map_shape}")
            if self.can_use_direct_indexing:
                print(f"  ✓ Grid is compatible - using direct indexing (no interpolation)")
            else:
                print(f"  ⚠ Grid not optimal for symmetry - using interpolation")
                if compat['issues']:
                    for issue in compat['issues']:
                        print(f"    - {issue}")
                # Suggest better grid
                suggested = self.symmetry.suggest_grid_size(map_shape, make_fft_friendly=True)
                print(f"    Suggested grid: {suggested}")
        
        # Precompute grid coordinates in fractional space
        self._setup_fractional_grid()
        
        # Precompute transformed grids for each symmetry operation
        self._setup_symmetry_grids()
    
    def _setup_fractional_grid(self):
        """
        Setup fractional coordinate grid for the map.
        Each voxel is at grid edge i/N (CCTBX/gemmi convention).
        """
        nx, ny, nz = self.map_shape
        
        # Fractional coordinates at grid edges (CCTBX convention)
        fx = torch.arange(nx, dtype=self.dtype_float, device=self.device) / nx
        fy = torch.arange(ny, dtype=self.dtype_float, device=self.device) / ny
        fz = torch.arange(nz, dtype=self.dtype_float, device=self.device) / nz

        # Create 3D grid
        # IMPORTANT: We want grid shape (nx, ny, nz, 3) where last dim is [fx, fy, fz]
        # meshgrid with indexing='ij' on (fx, fy, fz) gives (nx, ny, nz)
        grid_fx, grid_fy, grid_fz = torch.meshgrid(fx, fy, fz, indexing='ij')
        grid_frac = torch.stack([grid_fx, grid_fy, grid_fz], dim=-1)
        
        # Register as buffer (will be moved to GPU with model)
        self.register_buffer('grid_frac', grid_frac)
    
    def _setup_symmetry_grids(self):
        """
        Precompute transformed grid coordinates for all symmetry operations.
        
        For each symmetry operation:
        - Apply rotation matrix to fractional coordinates
        - Add translation
        - Convert to sampling coordinates for grid_sample
        """
        nx, ny, nz = self.map_shape
        
        # Flatten grid for easier matrix operations
        # Shape: (nx*ny*nz, 3)
        grid_flat = self.grid_frac.reshape(-1, 3)
        
        # Storage for transformed grids
        # Will convert to [-1, 1] range for grid_sample
        sampling_grids_list = []
        
        for i in range(self.n_ops):
            # Apply symmetry operation: R @ coords + t
            # grid_flat.T shape: (3, nx*ny*nz)
            # matrices[i] shape: (3, 3)
            # Result shape: (3, nx*ny*nz)
            transformed = torch.matmul(self.symmetry.matrices[i], grid_flat.T)
            transformed = transformed.T  # (nx*ny*nz, 3)
            transformed = transformed + self.symmetry.translations[i]
            
            # Wrap to [0, 1) for periodic boundary conditions
            transformed = transformed - torch.floor(transformed)
            grid_shape_tensor = torch.tensor([nx, ny, nz], dtype=self.dtype_float, device=transformed.device)
            # transformed shape: (nx*ny*nz, 3)
            # For each dimension: grid_coord = -1 + 2*N/(N-1) * frac
            sampling_coords = -1.0 + 2.0 * grid_shape_tensor / (grid_shape_tensor - 1.0) * transformed
            
            # Reshape back to 3D grid
            # grid_sample expects (N, D, H, W, 3) for 3D
            sampling_grid = sampling_coords.reshape(nx, ny, nz, 3)
            
            # CRITICAL: grid_sample coordinate interpretation for 3D data
            # - Input tensor has shape (N, C, D, H, W) where D=nx, H=ny, W=nz in our case
            # - Grid coords have shape (N, D_out, H_out, W_out, 3)
            # - The last dimension [grid_x, grid_y, grid_z] maps to [W, H, D] dimensions
            # - In our fractional coords: [fx, fy, fz] should map to [W, H, D] = [nz, ny, nx]
            # - So grid_sample expects coords in order: [fz, fy, fx] NOT [fx, fy, fz]
            # Therefore we need to reorder our [fx, fy, fz] -> [fz, fy, fx]
            sampling_grid = sampling_grid[..., [2, 1, 0]]  # [fx, fy, fz] -> [fz, fy, fx]
            
            sampling_grids_list.append(sampling_grid)
        
        # Stack all grids: (n_ops, nx, ny, nz, 3)
        sampling_grids_stacked = torch.stack(sampling_grids_list, dim=0)
        
        # Register as buffer (will be moved to GPU with model)
        self.register_buffer('sampling_grids', sampling_grids_stacked)
        
        # Also precompute integer index grids for direct indexing when possible
        if self.can_use_direct_indexing:
            self._setup_direct_index_grids()
    
    def _setup_direct_index_grids(self):
        """Precompute integer index grids for interpolation-free symmetry expansion.
        
        CRITICAL: This must match the grid_sample behavior exactly.
        With align_corners=True and the grid-edge convention (voxels at i/N):
        - Fractional coordinate f maps to sampling coordinate s = -1 + 2*N/(N-1) * f
        - grid_sample maps s back to index: idx = (s + 1) * (N - 1) / 2
        - Combining: idx = N/(N-1) * f * (N-1) = f * N
        - So fractional f should map to index round(f * N) % N
        
        This ensures exact mapping when f is a grid point (i/N).
        """
        nx, ny, nz = self.map_shape
        grid_flat = self.grid_frac.reshape(-1, 3)
        
        index_grids_list = []
        
        for i in range(self.n_ops):
            # Apply symmetry operation: R @ coords + t
            transformed = torch.matmul(self.symmetry.matrices[i], grid_flat.T).T
            transformed = transformed + self.symmetry.translations[i]
            
            # Wrap to [0, 1) for periodic boundary conditions
            transformed = transformed - torch.floor(transformed)
            
            # Convert fractional coordinates to integer indices
            # Use round() to get nearest grid point, matching interpolation behavior
            grid_shape_tensor = torch.tensor([nx, ny, nz], dtype=self.dtype_float, device=self.device)
            indices = torch.round(transformed * grid_shape_tensor).to(torch.int64)
            
            # Wrap indices to ensure they're within bounds (handle periodic boundary)
            indices[:, 0] = indices[:, 0] % nx
            indices[:, 1] = indices[:, 1] % ny
            indices[:, 2] = indices[:, 2] % nz
            
            # Reshape to 3D grid
            index_grid = indices.reshape(nx, ny, nz, 3)
            index_grids_list.append(index_grid)
        
        # Stack all index grids: (n_ops, nx, ny, nz, 3)
        index_grids_stacked = torch.stack(index_grids_list, dim=0)
        self.register_buffer('index_grids', index_grids_stacked)
    
    def get_symmetry_mate(self, density_map, operation_index):
        """
        Apply a single symmetry operation to get one symmetry mate.
        
        Parameters:
        -----------
        density_map : torch.Tensor, shape (nx, ny, nz)
            Electron density map (typically from asymmetric unit)
        operation_index : int
            Index of the symmetry operation to apply (0 to n_ops-1)
        
        Returns:
        --------
        transformed_map : torch.Tensor, shape (nx, ny, nz)
            Density map after applying the symmetry operation
        """
        if operation_index < 0 or operation_index >= self.n_ops:
            raise ValueError(f"Operation index {operation_index} out of range [0, {self.n_ops-1}]")
        
        # Ensure map is correct shape
        if density_map.shape != self.map_shape:
            raise ValueError(f"Map shape {density_map.shape} doesn't match expected {self.map_shape}")
        
        # Fast path: direct indexing if grid is compatible
        if self.can_use_direct_indexing and hasattr(self, 'index_grids'):
            # Use precomputed integer indices - no interpolation needed!
            index_grid = self.index_grids[operation_index]  # (nx, ny, nz, 3)
            
            # Extract values using advanced indexing
            # This is much faster than grid_sample
            transformed_map = density_map[
                index_grid[..., 0],
                index_grid[..., 1],
                index_grid[..., 2]
            ]
            
            return transformed_map
        
        # Fallback: interpolation-based approach
        # Prepare for grid_sample
        map_5d = density_map.unsqueeze(0).unsqueeze(0)  # (1, 1, nx, ny, nz)
        
        # Get sampling grid for this operation
        sampling_grid = self.sampling_grids[operation_index]
        sampling_grid_batch = sampling_grid.unsqueeze(0)
        
        # Interpolate map at transformed coordinates
        # align_corners=True ensures that:
        #   -1 maps to index 0 (fractional coord 0)
        #   +1 maps to index N-1 (fractional coord (N-1)/N)
        # This matches the grid-edge convention (voxels at i/N)
        # padding_mode='border' handles periodic boundary conditions via the wrapping
        # we did in _setup_symmetry_grids
        transformed_map = F.grid_sample(
            map_5d, 
            sampling_grid_batch,
            mode='bilinear',  # Trilinear interpolation for 3D
            padding_mode='border',  # Use border mode since we pre-wrapped coordinates
            align_corners=True  # Critical: matches grid-edge convention
        )
        
        # Remove batch and channel dimensions
        transformed_map = transformed_map.squeeze(0).squeeze(0)
        
        return transformed_map
    
    def get_all_symmetry_mates(self, density_map):
        """
        Get all symmetry mates as a list.
        
        Parameters:
        -----------
        density_map : torch.Tensor, shape (nx, ny, nz)
            Electron density map (typically from asymmetric unit)
        
        Returns:
        --------
        mates : list of torch.Tensor
            List of symmetry-related maps, one for each operation
        """
        mates = []
        for i in range(self.n_ops):
            mates.append(self.get_symmetry_mate(density_map, i))
        return mates
    
    def forward(self, density_map, apply_symmetry=True, combine_mode='sum'):
        """
        Apply symmetry operations to density map.
        
        Parameters:
        -----------
        density_map : torch.Tensor, shape (nx, ny, nz)
            Electron density map (typically from asymmetric unit)
        apply_symmetry : bool, default True
            If True, apply all symmetry operations and combine them.
            If False, return input map unchanged (useful for P1 or debugging)
        combine_mode : str, default 'sum'
            How to combine symmetry mates:
            - 'sum': Sum all symmetry mates (for electron density)
            - 'max': Take maximum across symmetry mates (for masks/boolean data)
        
        Returns:
        --------
        symmetric_map : torch.Tensor, shape (nx, ny, nz)
            Symmetry-expanded density map (combined symmetry mates)
        """
        if not apply_symmetry or self.n_ops == 1:
            # No symmetry or P1
            return density_map
        
        # Get all symmetry mates
        mates = self.get_all_symmetry_mates(density_map)
        mates_stacked = torch.stack(mates, dim=0)
        
        # Combine according to mode
        if combine_mode == 'sum':
            symmetric_map = mates_stacked.sum(dim=0)
        elif combine_mode == 'max':
            symmetric_map = mates_stacked.max(dim=0)[0]  # max returns (values, indices)
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}. Use 'sum' or 'max'.")
        
        return symmetric_map
    
    def __call__(self, density_map, apply_symmetry=True, combine_mode='sum'):
        """Make the class callable like a PyTorch module."""
        return self.forward(density_map, apply_symmetry=apply_symmetry, combine_mode=combine_mode)
    
    def cuda(self):
        """Move to GPU."""
        super().cuda()
        self.device = torch.device('cuda')
        return self
    
    def cpu(self):
        """Move to CPU."""
        super().cpu()
        self.device = torch.device('cpu')
        return self
    
    def to(self, device):
        """Move to specified device."""
        super().to(device)
        self.device = device
        return self
    
    def get_symmetry_info(self):
        """
        Get information about symmetry operations.
        
        Returns:
        --------
        dict with keys:
            - space_group: str
            - n_operations: int
            - matrices: torch.Tensor (n_ops, 3, 3)
            - translations: torch.Tensor (n_ops, 3)
        """
        return {
            'space_group': self.space_group,
            'n_operations': self.n_ops,
            'matrices': self.symmetry.matrices,
            'translations': self.symmetry.translations
        }
    
    def __repr__(self):
        return (f"MapSymmetry(space_group='{self.space_group}', "
                f"n_ops={self.n_ops}, map_shape={self.map_shape})")

