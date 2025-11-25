import pandas as pd
import numpy as np
from torchref.math_functions import math_torch
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING
import reciprocalspaceship as rs
from torchref.model.model_ft import ModelFT
from torchref.utils.utils import TensorMasks
from torchref.io import legacy_format_readers, cif_readers
from torchref.utils.debug_utils import DebugMixin
from torchref.math_functions.french_wilson import FrenchWilson


if TYPE_CHECKING:
    from torchref.model import Model

class ReflectionData(DebugMixin, nn.Module):
    """
    Class for handling crystallographic reflection data.
    
    Initialized empty and populated via loader methods.
    All data arrays are stored as PyTorch tensors.
    """
    
    def __init__(self, verbose: int = 1, device: str = 'cpu'):
        """
        Initialize empty ReflectionData object.
        
        Args:
            verbose: Verbosity level for logging
            device: Device to store tensors on ('cpu' or 'cuda' or 'cuda:0', etc.)
        """
        super().__init__()
        
        self.verbose: int = verbose  # Verbosity level for logging
        self.device = torch.device(device)  # Device for all tensors
        
        # Register tensor buffers (will be moved to GPU with .cuda())
        # Miller indicesx
        self.register_buffer('hkl', None)  # Shape: (N, 3), dtype: int32
        
        # Amplitudes/Intensities
        self.register_buffer('F', None)  # Structure factor amplitudes, shape: (N,)
        self.register_buffer('F_sigma', None)  # Uncertainties, shape: (N,)
        self.register_buffer('I', None)  # Intensities, shape: (N,)
        self.register_buffer('I_sigma', None)  # Uncertainties, shape: (N,)

        self.masks = TensorMasks()  # Dictionary of boolean masks for filtering reflections 1 is included 0 is excluded
        
        # R-free flags
        self.register_buffer('rfree_flags', None)  # R-free test set flags, shape: (N,), dtype: int32
        self.rfree_source: Optional[str] = None  # Name of the R-free column
        
        # Outlier flags
        self.register_buffer('outlier_flags', None)  # Outlier flags, shape: (N,), dtype: bool
        self.outlier_detection_params: Optional[Dict] = None  # Parameters used for outlier detection
        
        # Metadata
        self.register_buffer('cell', None)  # Unit cell parameters [a,b,c,alpha,beta,gamma]
        self.spacegroup: Optional[str] = None
        self.register_buffer('resolution', None)  # Resolution per reflection, shape: (N,)
        
        # Data source tracking
        self.amplitude_source: Optional[str] = None
        self.intensity_source: Optional[str] = None
        self.phase_source: Optional[str] = None

    def cuda(self, device=None):
        """Move ReflectionData to CUDA, including masks child module."""
        super().cuda(device)
        self.device = torch.device('cuda') if device is None else torch.device(device)
        # Explicitly move masks since it's a child module
        if hasattr(self, 'masks'):
            self.masks.cuda(device)
        if self.verbose > 1:
            print(f"ReflectionData moved to device: {self.device}")
        return self
    
    def cpu(self):
        """Move ReflectionData to CPU, including masks child module."""
        super().cpu()
        self.device = torch.device('cpu')
        # Explicitly move masks since it's a child module
        if hasattr(self, 'masks'):
            self.masks.cpu()
        if self.verbose > 1:
            print(f"ReflectionData moved to cpu")
        return self
    
    def load(self, reader):
        '''
        Load reflection data using a data reader.
        '''

        data_dict, cell, spacegroup = reader()

        hkl = torch.tensor(data_dict['HKL'], dtype=torch.int32, device=self.device)

        self.register_buffer('hkl', hkl)

        if cell is not None:
            self.register_buffer('cell', torch.tensor(cell, dtype=torch.float32, device=self.device))
        else:
            raise ValueError("Unit cell parameters are required in the data and could not be read.")
        
        if spacegroup is not None:
            self.spacegroup = spacegroup

        self._calculate_resolution()


        if 'I' in data_dict:    
            self.register_buffer('I', torch.tensor(data_dict['I'], dtype=torch.float32, device=self.device))
            if 'SIGI' in data_dict:
                self.register_buffer('I_sigma', torch.tensor(data_dict['SIGI'], dtype=torch.float32, device=self.device))
            self.intensity_source = data_dict.get('I_col', 'Unknown')
            self.FrenchWilson = FrenchWilson(self.hkl, self.cell, self.spacegroup, verbose=self.verbose)
            F, F_sigma = self.FrenchWilson(self.I, self.I_sigma)
            self.register_buffer('F', F)
            self.register_buffer('F_sigma', F_sigma)
        elif 'F' in data_dict:
            self.register_buffer('F', torch.tensor(data_dict['F'], dtype=torch.float32, device=self.device))
            if 'SIGF' in data_dict:
                if data_dict['SIGF'] is not None:
                     self.register_buffer('F_sigma', torch.tensor(data_dict['SIGF'], dtype=torch.float32, device=self.device))
                else:
                    sigF = math_torch.estimate_sigma_F(self.F)
                    self.register_buffer('F_sigma', sigF)
            else:
                sigF = math_torch.estimate_sigma_F(self.F)
                self.register_buffer('F_sigma', sigF)
            self.amplitude_source = data_dict.get('F_col', 'Unknown')

        else:
            raise ValueError("No amplitude or intensity data found in MTZ file")

        if 'R-free-flags' in data_dict:
            rfree = torch.tensor(data_dict['R-free-flags'], device=self.device)
            flagged = rfree < 0 
            rfree = rfree.clip(min=0, max=1).to(torch.bool)
            self.register_buffer('rfree_flags', rfree)
        else:
            flagged = torch.zeros(len(self.hkl), dtype=torch.bool, device=self.device)
            self._generate_rfree_flags(free_fraction=0.02, n_bins=20, min_per_bin=100)
            

        self.masks['flagged_initial'] = ~flagged

        self.sanitize_F()
        self.flag_suspicious_sigma()
        return self

    def load_mtz(self, path: str) -> 'ReflectionData':
        """
        Load reflection data from MTZ file using reciprocalspaceship.
        
        Args:
            path: Path to MTZ file
            expand_to_p1: Whether to expand to P1 (all symmetry-related reflections)
            
        Returns:
            self (for method chaining)
        """
        reader = legacy_format_readers.MTZ(verbose=self.verbose).read(path)
        return self.load(reader)

    def load_cif(self, path: str, data_block: Optional[str] = None) -> 'ReflectionData':
        """
        Load reflection data from CIF file using ReflectionCIFReader.
        
        Args:
            path: Path to CIF file
            data_block: Optional specific data block name to read (e.g., 'r1vlmsf').
                       If None, reads the first data block. Useful for files with
                       multiple datasets.
            
        Returns:
            self (for method chaining)
        """
        self.reader = cif_readers.ReflectionCIFReader(path, verbose=self.verbose, data_block=data_block)
        return self.load(self.reader)
    
    @staticmethod
    def list_cif_data_blocks(path: str) -> List[str]:
        """
        List all data blocks available in a CIF file without loading the data.
        
        Useful for multi-dataset CIF files to see which blocks are available
        before loading a specific one.
        
        Args:
            path: Path to CIF file
            
        Returns:
            List of data block names
            
        Example:
            >>> blocks = ReflectionData.list_cif_data_blocks('1VLM-sf.cif')
            >>> print(blocks)
            ['r1vlmsf', 'r1vlmAsf', 'r1vlmBsf', ...]
            >>> data = ReflectionData().load_cif('1VLM-sf.cif', data_block=blocks[1])
        """
        reader = cif_readers.CIFReader(path)
        return reader.available_blocks
    
    def _generate_rfree_flags(self, free_fraction: float = 0.02, n_bins: int = 20, 
                             min_per_bin: int = 100, seed: Optional[int] = None) -> None:
        """
        Generate R-free flags with proper resolution binning.
        
        This ensures that free reflections are evenly distributed across resolution shells,
        which is critical for unbiased validation.
        
        Args:
            free_fraction: Fraction of reflections to mark as free (default: 0.02 = 2%)
            n_bins: Target number of resolution bins (default: 20)
            min_per_bin: Minimum number of reflections per resolution bin (default: 100)
            seed: Random seed for reproducibility (default: None)
            
        The algorithm:
        1. Bin reflections by resolution
        2. Ensure bins have at least min_per_bin reflections or 1% of data
        3. Randomly select free_fraction of reflections from each bin
        4. This ensures even distribution across all resolution ranges
        """
        if self.resolution is None:
            raise ValueError("Resolution information required to generate R-free flags")
        
        print(f"Generating R-free flags:")
        print(f"  Target free fraction: {free_fraction*100:.1f}%")
        print(f"  Target bins: {n_bins}")
        print(f"  Minimum per bin: {min_per_bin} reflections")
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        n_refl = len(self.resolution)
        
        # Create resolution bins
        bin_indices, actual_n_bins = self.get_bins(n_bins=n_bins, min_per_bin=min_per_bin)
        
        print(f"  Created {actual_n_bins} resolution bins")
        
        # Initialize all flags as work set (1)
        flags = torch.ones(n_refl, dtype=torch.int32)
        
        # Sample free reflections from each bin
        total_free = 0
        for bin_idx in range(actual_n_bins):
            bin_mask = bin_indices == bin_idx
            bin_size = bin_mask.sum().item()
            
            if bin_size == 0:
                continue
            
            # Number of free reflections in this bin
            # Ensure at least 1, but respect the free_fraction
            n_free_in_bin = max(1, int(bin_size * free_fraction))
            
            # Get indices of reflections in this bin
            bin_refl_indices = torch.where(bin_mask)[0]
            
            # Randomly select free reflections
            perm = torch.randperm(bin_size)[:n_free_in_bin]
            free_indices = bin_refl_indices[perm]
            
            # Mark as free (0)
            flags[free_indices] = 0
            total_free += n_free_in_bin
        
        # Move to correct device and register
        flags_tensor = flags.to(dtype=torch.int32, device=self.device)
        self.register_buffer('rfree_flags', flags_tensor)
        self.rfree_source = "Generated (resolution-binned)"
        
        n_free = (flags == 0).sum().item()
        n_work = (flags != 0).sum().item()
        free_pct = 100.0 * n_free / n_refl
        
        print(f"  ✓ Generated flags: {n_free} free ({free_pct:.1f}%), {n_work} work ({100-free_pct:.1f}%)")
        print(f"  Flags are resolution-binned for unbiased validation")
    
    def get_bins(self, n_bins: int = 20, min_per_bin: int = 100) -> Tuple[torch.Tensor, int]:
        """
        Create resolution bins with target number of bins.
        
        Args:
            n_bins: Target number of resolution bins (default: 20)
            min_per_bin: Minimum reflections per bin (default: 100)
            
        Returns:
            Tuple of (bin_indices, n_bins) where:
                - bin_indices: Tensor of shape (N,) with bin index for each reflection
                - n_bins: Actual number of bins created (may be less than target if dataset is small)
        """
        n_refl = len(self.resolution)
        
        # Calculate how many bins we can actually create given min_per_bin constraint
        max_possible_bins = max(1, n_refl // min_per_bin)
        actual_n_bins = min(n_bins, max_possible_bins)
        
        if actual_n_bins < n_bins and self.verbose > 0:
            print(f"  Note: Adjusted bins from {n_bins} to {actual_n_bins} (min {min_per_bin} refl/bin)")
        
        # Sort reflections by resolution
        sorted_res, sort_indices = torch.sort(self.resolution)
        
        # Create bins with approximately equal number of reflections
        bin_indices = torch.zeros(n_refl, dtype=torch.int32, device=self.device)
        reflections_per_bin = n_refl / actual_n_bins
        
        for i, idx in enumerate(sort_indices):
            # Assign to bin based on position in sorted list
            bin_idx = min(int(i / reflections_per_bin), actual_n_bins - 1)
            bin_indices[idx] = bin_idx

        if self.verbose > 1:
            # Print bin statistics
            print(f"  Resolution bins:")
            for bin_idx in range(min(actual_n_bins, 20)):  # Show all bins (up to 20)
                bin_mask = bin_indices == bin_idx
                if bin_mask.sum() > 0:
                    bin_res = self.resolution[bin_mask]
                    print(f"    Bin {bin_idx:2d}: {bin_mask.sum():6d} refl, "
                        f"resolution {bin_res.min():.2f}-{bin_res.max():.2f} Å")
        
            if actual_n_bins > 20:
                print(f"    ... ({actual_n_bins - 20} more bins)")
        self.register_buffer('bin_indices', bin_indices)
        self.register_buffer('n_bins', torch.tensor(actual_n_bins, dtype=torch.int32))
        return bin_indices, actual_n_bins
    
    def mean_res_per_bin(self) -> torch.Tensor:
        """
        Calculate mean resolution for each bin.
        
        Returns:
            List of mean resolutions per bin
        """
        if not hasattr(self, 'bin_indices') or not hasattr(self, 'resolution'):
            raise ValueError("Bins have not been created yet")
        
        mean_resolutions = torch.zeros(self.n_bins, dtype=torch.float32, device=self.device)
        count_per_bin = torch.zeros(self.n_bins, dtype=torch.int32, device=self.device)
        mask = self.masks()
        mean_resolutions = torch.scatter_add(mean_resolutions, 0, self.bin_indices[mask].to(torch.int64), self.resolution[mask])
        count_per_bin = torch.scatter_add(count_per_bin, 0, self.bin_indices[mask].to(torch.int64), torch.ones_like(self.resolution[mask], dtype=torch.int32))
        mean_resolutions = mean_resolutions / count_per_bin.clamp(min=1).float()
        return mean_resolutions

    def regenerate_rfree_flags(self, free_fraction: float = 0.02, n_bins: int = 20,
                               min_per_bin: int = 100, seed: Optional[int] = None, 
                               force: bool = False) -> None:
        """
        Regenerate R-free flags (public interface).
        
        Args:
            free_fraction: Fraction of reflections to mark as free (default: 0.02 = 2%)
            n_bins: Target number of resolution bins (default: 20)
            min_per_bin: Minimum reflections per resolution bin (default: 100)
            seed: Random seed for reproducibility (default: None)
            force: If True, regenerate even if flags already exist (default: False)
            
        Example:
            # Generate 2% free reflections with 20 bins and reproducible seed
            data.regenerate_rfree_flags(free_fraction=0.02, n_bins=20, seed=42)
            
            # Generate 5% free with 10 bins
            data.regenerate_rfree_flags(free_fraction=0.05, n_bins=10, force=True)
        """
        if self.rfree_flags is not None and not force:
            print("⚠️  WARNING: R-free flags already exist!")
            print(f"   Current source: {self.rfree_source}")
            print("   Use force=True to overwrite existing flags")
            return
        
        if self.rfree_flags is not None and force:
            print("⚠️  WARNING: Overwriting existing R-free flags")
            print(f"   Old source: {self.rfree_source}")
        
        self._generate_rfree_flags(free_fraction=free_fraction, n_bins=n_bins, 
                                   min_per_bin=min_per_bin, seed=seed)
    
    def _calculate_resolution(self) -> None:
        """Calculate resolution for each reflection."""
        if self.hkl is None:
            raise ValueError("Miller indices (hkl) are required to calculate resolution")
        if self.cell is None:
            raise ValueError("Unit cell parameters are required to calculate resolution")
        s = math_torch.get_scattering_vectors(self.hkl, self.cell)
        resolution = 1.0 / torch.linalg.norm(s, axis=1)
        self.register_buffer('resolution', resolution)
    
    def get_structure_factors(self, as_complex: bool = False) -> torch.Tensor:
        """
        Get structure factors, optionally as complex numbers.
        
        Args:
            as_complex: If True and phases available, return F*exp(i*phi)
            
        Returns:
            Tensor of amplitudes or complex structure factors
        """
        if self.F is None:
            raise ValueError("No amplitude data loaded")
        
        if as_complex and self.phase is not None:
            return self.F * torch.exp(1j * self.phase)
        else:
            return self.F
    
    def get_structure_factors_with_sigma(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get structure factor amplitudes and their uncertainties.
        
        Returns:
            Tuple of (F, F_sigma) where:
                - F: Structure factor amplitudes, shape (N,)
                - F_sigma: Uncertainties (None if not available), shape (N,)
        
        Example:
            F, sigma_F = data.get_structure_factors_with_sigma()
            if sigma_F is not None:
                weighted_residual = (F_obs - F_calc) / sigma_F
        """
        if self.F is None:
            raise ValueError("No amplitude data loaded")
        
        return self.F, self.F_sigma
    
    def get_hkl(self):
        """Return Miller indices as a tensor of shape (N, 3)."""
        if self.hkl is None:
            raise ValueError("No Miller indices loaded")
        return self.hkl[self.masks()]

    def filter_by_resolution(self, d_min: Optional[float] = None, 
                            d_max: Optional[float] = None) -> 'ReflectionData':
        """
        Filter reflections by resolution range.
        adds a boolean mask to self.masks for the specified resolution range.

        Args:
            d_min: Minimum resolution (highest resolution, e.g., 1.5 Å)
            d_max: Maximum resolution (lowest resolution, e.g., 50.0 Å)
            
        Returns:
            self (for method chaining)
        """
        if self.resolution is None:
            self._calculate_resolution()
        
        mask = torch.ones(len(self.hkl), dtype=torch.bool)
        
        if d_min is not None:
            mask &= self.resolution >= d_min
        if d_max is not None:
            mask &= self.resolution <= d_max
        
        print(f"Filtering: {mask.sum()}/{len(mask)} reflections in range "
              f"[{d_max if d_max else 'inf'} - {d_min if d_min else 'inf'}] Å")
        
        self.masks['resolution'] = mask

        return self
    
    def get_mask(self):
        return self.masks()
    
    def cut_res(self, highres: Optional[float] = None, 
                lowres: Optional[float] = None) -> 'ReflectionData':
        """
        Filter reflections by resolution range (alias for filter_by_resolution).
        
        This method uses the more intuitive naming where:
        - res_min is the minimum resolution (high resolution limit, small d-spacing)
        - res_max is the maximum resolution (low resolution limit, large d-spacing)
        
        Args:
            res_min: Minimum resolution / high resolution cutoff (e.g., 1.5 Å)
                    Keeps reflections with d >= res_min
            res_max: Maximum resolution / low resolution cutoff (e.g., 50.0 Å)
                    Keeps reflections with d <= res_max
            
        Returns:
            New ReflectionData object with filtered data
            
        Example:
            # Keep reflections between 50 Å and 1.5 Å
            filtered = data.cut_res(res_min=1.5, res_max=50.0)
            
            # Keep only high-resolution data (< 2 Å)
            high_res = data.cut_res(res_min=1.0, res_max=2.0)
        """
        return self.filter_by_resolution(d_min=highres, d_max=lowres)
    
    def get_rfree_masks(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get boolean masks for work and test (free) sets.
        
        Returns:
            Tuple of (work_mask, test_mask) where:
                - work_mask: Boolean tensor for work set (flag != 0)
                - test_mask: Boolean tensor for test/free set (flag == 0)
                - Both are None if no R-free flags are available
        
        Example:
            work_mask, test_mask = data.get_rfree_masks()
            if work_mask is not None:
                F_work = data.F[work_mask]
                F_test = data.F[test_mask]
                # Calculate R-factors separately
        """
        if self.rfree_flags is None:
            return None, None
        
        work_mask = self.rfree_flags != 0
        test_mask = self.rfree_flags == 0
        
        return work_mask, test_mask
    
    def get_work_set(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get structure factors for the work set (R-free flag != 0).
        
        Returns:
            Tuple of (F_work, sigma_work) or full dataset if no R-free flags
        """
        if self.rfree_flags is None:
            print("WARNING: No R-free flags available, returning full dataset")
            return self.F, self.F_sigma
        
        work_mask = self.rfree_flags != 0
        F_work = self.F[work_mask] if self.F is not None else None
        sigma_work = self.F_sigma[work_mask] if self.F_sigma is not None else None
        
        return F_work, sigma_work
    
    def get_test_set(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get structure factors for the test set (R-free flag == 0).
        
        Returns:
            Tuple of (F_test, sigma_test)
        
        Raises:
            ValueError if no R-free flags are available
        """
        if self.rfree_flags is None:
            raise ValueError("No R-free flags available in dataset")
        
        test_mask = self.rfree_flags == 0
        F_test = self.F[test_mask] if self.F is not None else None
        sigma_test = self.F_sigma[test_mask] if self.F_sigma is not None else None
        
        return F_test, sigma_test

    def get_max_res(self) -> Optional[float]:
        """Return maximum resolution (lowest d-spacing) in Å."""
        if self.resolution is None:
            self._calculate_resolution()
        return float(self.resolution.min().item())

    def __len__(self) -> int:
        """Return number of reflections."""
        return len(self.hkl) if self.hkl is not None else 0
    
    def __repr__(self) -> str:
        """String representation."""
        if self.hkl is None:
            return "ReflectionData(empty)"
        
        parts = [f"ReflectionData(n={len(self.hkl)}"]
        if self.amplitude_source:
            parts.append(f"F={self.amplitude_source}")
        if self.phase_source:
            parts.append(f"φ={self.phase_source}")
        if self.resolution is not None:
            parts.append(f"d={self.resolution.min():.2f}-{self.resolution.max():.2f}Å")
        parts.append(f"sg={self.spacegroup}")
        
        return ", ".join(parts) + ")"

    def forward(self, mask:bool=True)-> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """

        Return core reflection data: (hkl, F, F_sigma, rfree_flags).

        Returns:

            Tuple of (hkl, F, F_sigma, rfree_flags)

        hkl: Tensor of shape (N, 3)
        F: Tensor of shape (N,)
        F_sigma: Tensor of shape (N,) or None 
        rfree_flags: Tensor of shape (N,) or None (1 is work, 0 is free)

        """
        hkl, F, F_sigma, rfree_flags = self.hkl, self.F, self.F_sigma, self.rfree_flags
        if mask:
            to_mask = self.masks()
            hkl, F = hkl[to_mask], F[to_mask]
            if F_sigma is not None:
                F_sigma = F_sigma[to_mask]
            if rfree_flags is not None:
                rfree_flags = rfree_flags[to_mask]
            rfree_flags = rfree_flags.to(torch.bool)
        return hkl, F, F_sigma, rfree_flags
    
    def __select__(self,mask:torch.Tensor, op=None)-> 'ReflectionData':
        """Select reflections based on a boolean mask."""
        # Create new instance with same device
        selected = ReflectionData(verbose=self.verbose, device=self.device)
        
        # Register buffers for the selected data
        if self.hkl is not None:
            selected.register_buffer('hkl', self.hkl[mask])
        if self.I is not None:
            selected.register_buffer('I', self.I[mask])
        if self.F is not None:
            selected.register_buffer('F', self.F[mask])
        if self.F_sigma is not None:
            selected.register_buffer('F_sigma', self.F_sigma[mask])
        if self.phase is not None:
            selected.register_buffer('phase', self.phase[mask])
        if self.fom is not None:
            selected.register_buffer('fom', self.fom[mask])
        if self.rfree_flags is not None:
            selected.register_buffer('rfree_flags', self.rfree_flags[mask])
        if self.cell is not None:
            selected.register_buffer('cell', self.cell.clone())
        if self.resolution is not None:
            selected.register_buffer('resolution', self.resolution[mask])
        
        selected.spacegroup = self.spacegroup
        selected.amplitude_source = self.amplitude_source
        selected.intensity_source = self.intensity_source
        selected.phase_source = self.phase_source
        selected.rfree_source = self.rfree_source
        selected.verbose = self.verbose
        selected.source = self
        selected.dataset = self.dataset.iloc[mask.cpu().numpy()].copy() if self.dataset is not None else None
        self.last_op = op
        return selected

    def unpack(self):
        """
        Unpack to the original source data.
        Traverses the source chain to the original data and validates HKL.
        Marks reflections not present in the original data in self.flagged
        """
        current_hkl = self.hkl
        while self.source is not None:
            self = self.source
        _, valid_hkl = self.validate_hkl(current_hkl)
        self.flagged = ~valid_hkl
        return self        

    def sanitize_F(self):
        """Cut NA values from F and F_sigma."""
        mask = torch.zeros(len(self.F), dtype=torch.bool, device=self.device)
        if self.F is not None:
            if self.verbose > 0: print('found nan F values: ', torch.isnan(self.F).sum().item())
            mask |= torch.isnan(self.F)
        if self.F_sigma is not None:
            if self.verbose > 0: print('found nan F_sigma values: ', torch.isnan(self.F_sigma).sum().item())
            mask |= torch.isnan(self.F_sigma)
        self.masks['sanity_F'] = ~mask
        return self

    def check_all_data_types(self):
        for key in self.__dict__:
            if self.__dict__[key] is not None and isinstance(self.__dict__[key], torch.Tensor):
                print(f"{key}: {self.__dict__[key].dtype}, shape: {self.__dict__[key].shape}")
            elif self.__dict__[key] is not None:
                print(f"{key}: {type(self.__dict__[key])}, value: {self.__dict__[key]}")
            else:
                print(f"{key}: None")
    
    def validate_hkl(self, hkl_ref: torch.Tensor) -> Tuple['ReflectionData', torch.Tensor]:
        """
        Validate and filter reflections against a reference HKL set.
        
        This method filters the current dataset to only include reflections that are 
        present in the reference HKL tensor, and returns a boolean mask indicating 
        which reference reflections are present in this dataset.
        
        Args:
            hkl_ref: Reference Miller indices tensor of shape (N, 3) with dtype int32
            
        Returns:
            Tuple of (filtered_data, presence_mask) where:
                - filtered_data: New ReflectionData object containing only reflections 
                  that are present in hkl_ref
                - presence_mask: Boolean tensor of shape (N,) indicating which reflections 
                  from hkl_ref are present in this dataset (True = present, False = absent)
        
        Example:
            # Filter data to match reference HKL list
            filtered_data, present_in_data = data.validate_hkl(reference_hkl)
            print(f"Kept {len(filtered_data)} reflections out of {len(data)}")
            print(f"{present_in_data.sum()} reference reflections found in dataset")
        """
        if self.hkl is None:
            raise ValueError("No Miller indices loaded in ReflectionData")
        
        if not isinstance(hkl_ref, torch.Tensor):
            raise TypeError(f"hkl_ref must be a torch.Tensor, got {type(hkl_ref)}")
        
        if hkl_ref.shape[-1] != 3:
            raise ValueError(f"hkl_ref must have shape (N, 3), got {hkl_ref.shape}")
        
        # Ensure hkl_ref is 2D and int32
        if hkl_ref.dim() == 1:
            hkl_ref = hkl_ref.unsqueeze(0)
        hkl_ref = hkl_ref.to(dtype=torch.int32)
        
        n_ref = len(hkl_ref)
        n_data = len(self.hkl)
        
        # Convert to numpy for efficient isin-based lookup
        # We'll use structured arrays to compare HKL triplets as single entities
        hkl_ref_np = hkl_ref.cpu().numpy()
        hkl_data_np = self.hkl.cpu().numpy()
        
        # Create structured arrays to treat each (h,k,l) triplet as a single comparable unit
        # This allows numpy to efficiently check membership
        hkl_ref_structured = np.ascontiguousarray(hkl_ref_np).view(
            np.dtype((np.void, hkl_ref_np.dtype.itemsize * hkl_ref_np.shape[1]))
        )
        hkl_data_structured = np.ascontiguousarray(hkl_data_np).view(
            np.dtype((np.void, hkl_data_np.dtype.itemsize * hkl_data_np.shape[1]))
        )
        
        # Find which data reflections are present in the reference
        # np.isin is highly optimized and uses hash-based lookup internally
        data_mask_np = np.isin(hkl_data_structured, hkl_ref_structured)
        data_mask = torch.from_numpy(data_mask_np.flatten())
        
        # Find which reference reflections are present in the dataset
        # Use filtered HKL for this operation
        hkl_filtered, _, _, _ = self()
        hkl_filtered_np = hkl_filtered.cpu().numpy()
        hkl_filtered_structured = np.ascontiguousarray(hkl_filtered_np).view(
            np.dtype((np.void, hkl_filtered_np.dtype.itemsize * hkl_filtered_np.shape[1]))
        )
        presence_mask_np = np.isin(hkl_ref_structured, hkl_filtered_structured)
        presence_mask = torch.from_numpy(presence_mask_np.flatten())

        # Filter the dataset to only include reflections in the reference
        self.masks['hkl_validation'] = data_mask.to(self.device)
        
        if self.verbose > 0:
            print(f"HKL validation:")
            print(f"  Dataset reflections: {n_data}")
            print(f"  Reference reflections: {n_ref}")
            print(f"  Kept in dataset: {data_mask.sum().item()} ({100*data_mask.sum().item()/n_data:.1f}%)")
            print(f"  Found in dataset: {presence_mask.sum().item()} ({100*presence_mask.sum().item()/n_ref:.1f}%)")
        
        return self, presence_mask
    
    def find_outliers(self, model: ModelFT, scaler, z_threshold: float = 4.0) -> torch.Tensor:
        """
        Identify outlier reflections based on log-ratio distribution.
        
        Uses the fact that log(F_obs) - log(F_calc) should be normally distributed.
        Outliers are reflections where |log_ratio - mean| > z_threshold * std_dev.
        
        Args:
            model: ModelFT object to compute structure factors
            scaler: Scaler object to scale calculated structure factors  
            z_threshold: Z-score threshold to classify outliers (default: 4.0)
            
        Returns:
            torch.Tensor: Boolean mask where True indicates outliers
        """
        hkl, F_obs, _, _ = self.forward(mask=False)
        log_ratio = self.get_log_ratio(model, scaler)
        eps = 1e-10
        
        # Remove any infinite or NaN values for statistics
        valid_mask = torch.isfinite(log_ratio)
        if valid_mask.sum() == 0:
            if self.verbose > 0:
                print("Warning: No valid log-ratios found for outlier detection")
            return torch.zeros_like(F_obs, dtype=torch.bool, device=self.device)
        
        to_use = valid_mask 

        log_ratio_valid = log_ratio[to_use]
        
        # Compute mean and standard deviation of log-ratio distribution
        mean_log_ratio = torch.mean(log_ratio_valid)
        std_log_ratio = torch.std(log_ratio_valid, unbiased=True)
        
        # Identify outliers using Z-score criterion
        z_scores = torch.abs(log_ratio - mean_log_ratio) / (std_log_ratio + eps)
        outlier_mask = z_scores > z_threshold
        
        # Set invalid ratios as outliers too
        outlier_mask = outlier_mask | ~valid_mask
        
        if self.verbose > 0:
            n_outliers = outlier_mask.sum().item()
            n_total = len(F_obs)
            print(f"Outlier detection: {n_outliers}/{n_total} ({100*n_outliers/n_total:.2f}%) outliers found")
            print(f"  Log-ratio statistics: mean={mean_log_ratio:.3f}, std={std_log_ratio:.3f}")
            print(f"  Z-score threshold: {z_threshold:.1f}")
        
        # Ensure outlier_mask is on correct device and register
        outlier_mask = outlier_mask.to(self.device)
        self.masks['outliers'] = ~outlier_mask
        if self.verbose > 0: print(f"Outlier detection: {outlier_mask.sum().item()} reflections flagged as outliers out of {len(outlier_mask)}.")
    
    def get_log_ratio(self, model: ModelFT, scaler) -> torch.Tensor:
        # Get observed and calculated structure factors
        eps = 1e-6
        hkl, F_obs, _ , _ = self.forward(mask=False)
        F_calc_complex = model.forward(hkl)  # Complex structure factors
        F_calc_scaled = torch.abs(scaler(F_calc_complex,use_mask=False))  # Scaled amplitudes
        # Avoid log of zero by adding small epsilon
        F_obs_safe = torch.clamp(F_obs, min=eps)
        F_calc_safe = torch.clamp(F_calc_scaled, min=eps)
        # Compute log-ratio distribution: log(F_obs) - log(F_calc)
        log_ratio = torch.log(F_obs_safe) - torch.log(F_calc_safe)
        return log_ratio

    def get_outlier_statistics(self) -> Dict:
        """Get statistics about flagged outliers."""
        if self.outlier_flags is None:
            return {'n_outliers': 0, 'n_total': 0, 'fraction_outliers': 0.0}
        
        n_outliers = self.outlier_flags.sum().item()
        n_total = len(self.outlier_flags)
        
        stats = {
            'n_outliers': n_outliers,
            'n_total': n_total,
            'fraction_outliers': n_outliers / n_total if n_total > 0 else 0.0,
            'detection_params': self.outlier_detection_params
        }
        
        if self.resolution is not None:
            # Add resolution-dependent statistics
            outlier_resolutions = self.resolution[self.outlier_flags] if n_outliers > 0 else torch.tensor([])
            if len(outlier_resolutions) > 0:
                stats['outlier_resolution_stats'] = {
                    'min': outlier_resolutions.min().item(),
                    'max': outlier_resolutions.max().item(),
                    'mean': outlier_resolutions.mean().item(),
                    'median': outlier_resolutions.median().item()
                }
        
        return stats
    
    def unpack_one(self):
        """
        Unpack one level of source.
        Does not recurse fully.
        Also does not flag. 
        """
        if self.source is not None:
            return self.source
        return self
    
    def get_lognormal_sigma(self, F: Optional[torch.Tensor] = None, 
                           sigma_F: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convert Gaussian parameters (F, sigma_F) to lognormal sigma parameter.
        
        For a lognormal distribution, if X ~ LogNormal(μ, σ²), then:
        - E[X] = exp(μ + σ²/2)
        - Var(X) = exp(2μ + σ²)(exp(σ²) - 1)
        
        Given observed F (mean) and sigma_F (standard deviation), we can solve for σ.
        
        Args:
            F: Structure factor amplitudes (uses self.F if None)
            sigma_F: Standard deviations (uses self.F_sigma if None)
            
        Returns:
            torch.Tensor: Sigma parameter for lognormal distribution
        """
        if F is None:
            F = self.F
        if sigma_F is None:
            sigma_F = self.F_sigma
            
        if F is None or sigma_F is None:
            raise ValueError("F and sigma_F must be provided or available in self")
            
        return gaussian_to_lognormal_sigma(F, sigma_F)

    def flag_suspicious_sigma(self, z_threshold: float = 5.0) -> None:
        '''
        Sigma values from a detector should follow a pretty tight log normal distribution.
        The distribution might not be super pretty but the outlier scoring should be robust.
        All values with z_threshold sigma away from the mean of log(sigma) are flagged as suspicious.
        '''
        sigmas = self.F_sigma
        log_sigmas = torch.log(sigmas)
        flagged_initial = torch.isnan(log_sigmas) | torch.isinf(log_sigmas)
        mean_log_sigma = torch.mean(log_sigmas[~flagged_initial])
        std_log_sigma = torch.std(log_sigmas[~flagged_initial])
        z_scores = (log_sigmas - mean_log_sigma) / std_log_sigma
        flagged = torch.abs(z_scores) > z_threshold
        flagged = flagged | flagged_initial
        if self.verbose > 0:
            n_flagged = flagged.sum().item()
            n_total = len(sigmas)
            print(f"Suspicious sigma detection: {n_flagged}/{n_total} ({100*n_flagged/n_total:.2f}%) reflections flagged")
        self.masks['flagged_sigma'] = ~flagged

    def dump(self):
        """Dump all reflection data to console for debugging."""
        print("ReflectionData dump:")
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, torch.Tensor):
                print(f"  {key}: dtype={value.dtype}, shape={value.shape}, device={value.device}")
            else:
                print(f"  {key}: type={type(value)}, value={value}")

    def write_mtz(self, fname: str, fcalc: Optional[torch.Tensor] = None, 
                  model_ft: Optional[ModelFT] = None, fill_to_resolution: bool = True) -> None:
        """
        Write reflection data to MTZ file with optional calculated structure factors and map coefficients.
        
        Args:
            fname: Output MTZ filename
            fcalc: Optional complex calculated structure factors (shape: [N])
                   If provided, will compute phases and map coefficients (2mFo-DFc, mFo-DFc)
            model_ft: Optional ModelFT object to compute fcalc if not provided
            fill_to_resolution: If True and fcalc provided, fill map coefficients to resolution limit
                               (default: True). This ensures complete maps for visualization.
        
        The MTZ file will contain canonical column names for maximum compatibility:
            - FP, SIGFP: Observed amplitudes and uncertainties
            - I, SIGI: Observed intensities and uncertainties (if available)
            - FreeR_flag: R-free test set flags
            - FWT, PHWT: 2mFo-DFc map coefficients (if fcalc provided)
            - DELFWT, PHDELWT: mFo-DFc map coefficients (if fcalc provided)
        
        Map coefficients are computed using:
            - 2mFo-DFc: 2*Fo - Fc (filled to resolution limit)
            - mFo-DFc: Fo - Fc
        
        Example:
            >>> data = ReflectionData().load_mtz('observed.mtz')
            >>> model = Model().load_pdb('model.pdb')
            >>> model_ft = ModelFT(model, data.cell, data.spacegroup)
            >>> fcalc = model_ft.forward(data.hkl)
            >>> data.write_mtz('output.mtz', fcalc=fcalc)
        """
        from torchref.io import file_writers
        
        # Convert data to numpy for DataFrame creation
        hkl_np = self.hkl.detach().cpu().numpy()
        
        # Create DataFrame with HKL indices
        data_dict = {
            'H': hkl_np[:, 0],
            'K': hkl_np[:, 1],
            'L': hkl_np[:, 2],
        }
        
        # Add observed amplitudes (canonical names: FP, SIGFP)
        if self.F is not None:
            data_dict['F-obs'] = self.F.detach().cpu().numpy()
            if self.F_sigma is not None:
                data_dict['SIGF-obs'] = self.F_sigma.detach().cpu().numpy()
        
        # Add observed intensities (canonical names: I, SIGI)
        if self.I is not None:
            data_dict['I-obs'] = self.I.detach().cpu().numpy()
            if self.I_sigma is not None:
                data_dict['SIGI-obs'] = self.I_sigma.detach().cpu().numpy()
        
        # Add R-free flags (canonical name: FreeR_flag)
        if self.rfree_flags is not None:
            data_dict['R-free-flags'] = self.rfree_flags.detach().cpu().numpy().astype(int)
        
        # Compute fcalc if model_ft is provided but fcalc is not
        if fcalc is None and model_ft is not None:
            fcalc = model_ft.forward(self.hkl)
        
        # Add map coefficients if fcalc is provided
        if fcalc is not None:
            # Ensure fcalc is complex
            if not torch.is_complex(fcalc):
                raise ValueError("fcalc must be a complex tensor")
            
            # Convert to numpy
            fcalc_np = fcalc.detach().cpu().numpy()
            F_obs = self.F.detach().cpu().numpy()
            
            # Compute phases in degrees
            phases = np.angle(fcalc_np, deg=True)
            F_calc_amp = np.abs(fcalc_np)
            
            # Compute map coefficients
            # 2mFo-DFc: Use observed amplitudes with calculated phases
            two_mfo_dfc_amp = 2.0 * F_obs - F_calc_amp
            two_mfo_dfc_phase = phases
            
            # mFo-DFc: Difference map
            mfo_dfc_complex = F_obs * np.exp(1j * np.deg2rad(phases)) - fcalc_np
            mfo_dfc_amp = np.abs(mfo_dfc_complex)
            mfo_dfc_phase = np.angle(mfo_dfc_complex, deg=True)
            
            # Add 2mFo-DFc map coefficients (canonical names: FWT, PHWT)
            data_dict['2FOFCWT'] = two_mfo_dfc_amp
            data_dict['PH2FOFCWT'] = two_mfo_dfc_phase
            
            # Add mFo-DFc map coefficients (canonical names: DELFWT, PHDELWT)
            data_dict['FOFCWT'] = mfo_dfc_amp
            data_dict['PHFOFCWT'] = mfo_dfc_phase

            data_dict['F-model'] = F_calc_amp
            data_dict['PH-model'] = phases
            
            if self.verbose > 0:
                print(f"Added map coefficients:")
                print(f"  2mFo-DFc: FWT, PHWT")
                print(f"  mFo-DFc: DELFWT, PHDELWT")
                print(f"  Resolution range: {self.resolution.min().item():.2f} - {self.resolution.max().item():.2f} Å")
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Write MTZ file
        file_writers.write_mtz(df, self.cell, self.spacegroup, fname)
        
        if self.verbose > 0:
            print(f"✓ Wrote MTZ file: {fname}")
            print(f"  Reflections: {len(df)}")
            print(f"  Columns: {', '.join(df.columns)}")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Returns a dictionary containing the complete state of the ReflectionData.
        
        This includes:
        - All registered buffers (hkl, F, I, rfree_flags, etc.)
        - Masks (via TensorMasks.state_dict())
        - Metadata (spacegroup, data sources, outlier params, etc.)
        
        Args:
            destination: Optional dict to populate
            prefix: Prefix for parameter names
            keep_vars: Whether to keep variables in computational graph
            
        Returns:
            dict: Complete state dictionary
        """
        # Get parent class state_dict (includes all registered buffers)
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        # Add ReflectionData-specific metadata
        state[prefix + 'spacegroup'] = self.spacegroup
        state[prefix + 'rfree_source'] = self.rfree_source
        state[prefix + 'outlier_detection_params'] = self.outlier_detection_params
        state[prefix + 'amplitude_source'] = self.amplitude_source
        state[prefix + 'intensity_source'] = self.intensity_source
        state[prefix + 'phase_source'] = self.phase_source
        state[prefix + 'verbose'] = self.verbose
        
        return state

    def load_state_dict(self, state_dict, strict=True):
        """
        Loads the ReflectionData state from a dictionary.
        
        Args:
            state_dict: Dictionary containing reflection data state
            strict: Whether to strictly enforce that keys match
        """
        # Extract ReflectionData-specific metadata
        self.spacegroup = state_dict.pop('spacegroup', None)
        self.rfree_source = state_dict.pop('rfree_source', None)
        self.outlier_detection_params = state_dict.pop('outlier_detection_params', None)
        self.amplitude_source = state_dict.pop('amplitude_source', None)
        self.intensity_source = state_dict.pop('intensity_source', None)
        self.phase_source = state_dict.pop('phase_source', None)
        self.verbose = state_dict.pop('verbose', 1)
        
        # If loading into empty object, register buffers based on state_dict content
        # We check for 'hkl' to determine size
        if 'hkl' in state_dict and state_dict['hkl'] is not None:
            hkl = state_dict['hkl']
            n_refl = hkl.shape[0]
            
            if not hasattr(self, 'hkl') or self.hkl is None:
                self.register_buffer('hkl', torch.zeros(n_refl, 3, dtype=torch.int32, device=self.device))
                
            # Register other buffers if they are in state_dict
            buffer_names = ['F', 'F_sigma', 'I', 'I_sigma', 'rfree_flags', 'outlier_flags', 'resolution', 'bin_indices', 'n_bins']
            for name in buffer_names:
                if name in state_dict and state_dict[name] is not None:
                    if not hasattr(self, name) or getattr(self, name) is None:
                        # Use the tensor from state_dict to determine dtype and shape
                        # But we register a dummy tensor, load_state_dict will overwrite
                        tensor = state_dict[name]
                        self.register_buffer(name, torch.zeros_like(tensor, device=self.device))
            
            # Register cell
            if 'cell' in state_dict and state_dict['cell'] is not None:
                if not hasattr(self, 'cell') or self.cell is None:
                    self.register_buffer('cell', torch.zeros(6, dtype=torch.float32, device=self.device))

        # Instantiate FrenchWilson if present in state_dict
        # Check for any key starting with "FrenchWilson."
        has_french_wilson = any(k.startswith("FrenchWilson.") for k in state_dict.keys())
        if has_french_wilson and not hasattr(self, 'FrenchWilson'):
            # We need hkl and cell to instantiate FrenchWilson
            # They might be in state_dict or already registered
            if 'hkl' in state_dict and 'cell' in state_dict:
                hkl = state_dict['hkl'].to(self.device)
                cell = state_dict['cell'].to(self.device)
                # We need spacegroup. It was popped above.
                
                self.FrenchWilson = FrenchWilson(hkl, cell, self.spacegroup, verbose=self.verbose)
                self.FrenchWilson.to(self.device)
        
        # Load parent class state_dict (buffers including masks)
        return super().load_state_dict(state_dict, strict=strict)

    def save_state(self, path: str):
        """
        Save the complete state of the reflection data to a file.
        
        Args:
            path (str): Path to save the state dictionary to.
        """
        torch.save(self.state_dict(), path)
        if self.verbose > 0:
            print(f"Saved reflection data state to {path}")

    def load_state(self, path: str, strict: bool = True):
        """
        Load the complete state of the reflection data from a file.
        
        Args:
            path (str): Path to load the state dictionary from.
            strict (bool): Whether to strictly enforce that keys match.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict, strict=strict)
        if self.verbose > 0:
            print(f"Loaded reflection data state from {path}")
