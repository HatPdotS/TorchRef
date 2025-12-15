import pandas as pd
import numpy as np
import warnings
from torchref.math_functions import math_torch
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Union, TYPE_CHECKING
import reciprocalspaceship as rs
from torchref.model.model_ft import ModelFT
from torchref.utils.utils import TensorMasks
from torchref.io import legacy_format_readers, cif_readers
from torchref.utils.debug_utils import DebugMixin
from torchref.math_functions.french_wilson import FrenchWilson

# Suppress PyTorch MaskedTensor prototype warnings globally
# MaskedTensor is stable enough for our use case (aggregations, element-wise ops)
warnings.filterwarnings(
    'ignore', 
    message='.*MaskedTensors is in prototype stage.*',
    category=UserWarning
)

if TYPE_CHECKING:
    from torchref.model import Model
    from torch.masked import MaskedTensor

class ReflectionData(DebugMixin, nn.Module):
    """
    Container for crystallographic reflection data.

    This class handles loading, processing, and accessing reflection data
    including Miller indices, structure factor amplitudes, intensities,
    and R-free flags. All data is stored as PyTorch tensors for GPU
    acceleration.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level for logging (0=silent, 1=normal, 2=debug). Default is 1.
    device : str, optional
        Device to store tensors on ('cpu', 'cuda', 'cuda:0', etc.). Default is 'cpu'.

    Attributes
    ----------
    hkl : torch.Tensor
        Miller indices of shape (N, 3), dtype int32.
    F : torch.Tensor
        Structure factor amplitudes of shape (N,), dtype float32.
    F_sigma : torch.Tensor
        Amplitude uncertainties of shape (N,), dtype float32.
    I : torch.Tensor
        Intensities of shape (N,), dtype float32.
    I_sigma : torch.Tensor
        Intensity uncertainties of shape (N,), dtype float32.
    rfree_flags : torch.Tensor
        R-free test set flags of shape (N,), dtype bool.
    cell : torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    spacegroup : str
        Space group symbol.
    resolution : torch.Tensor
        Resolution per reflection in Ångströms of shape (N,).
    wilson_b : float
        Overall Wilson B-factor in Ų.

    Examples
    --------
    >>> data = ReflectionData(verbose=1, device='cuda')
    >>> data.load_mtz('data.mtz')
    >>> print(f"Loaded {len(data.hkl)} reflections")
    >>> print(f"Resolution range: {data.resolution.min():.2f} - {data.resolution.max():.2f} Å")
    """

    def __init__(self, verbose: int = 1, device: str = 'cpu'):
        """
        Initialize empty ReflectionData object.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level for logging (0=silent, 1=normal, 2=debug). Default is 1.
        device : str, optional
            Device to store tensors on ('cpu', 'cuda', 'cuda:0', etc.). Default is 'cpu'.
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
        
        # Wilson B-factors (estimated from data)
        self.wilson_b: Optional[float] = None  # Overall Wilson B-factor in Å²
        self.wilson_b_structure: Optional[float] = None  # Structure B-factor (high-res) in Å²
        self.wilson_b_solvent: Optional[float] = None  # Solvent B-factor (low-res) in Å²
        self.wilson_k_sol: Optional[float] = None  # Relative solvent contribution
        
        # Data source tracking
        self.amplitude_source: Optional[str] = None
        self.intensity_source: Optional[str] = None
        self.phase_source: Optional[str] = None

    def cuda(self, device=None):
        """
        Move ReflectionData to CUDA device.

        Parameters
        ----------
        device : torch.device or int, optional
            Target CUDA device. If None, uses default CUDA device.

        Returns
        -------
        ReflectionData
            Self, for method chaining.
        """
        super().cuda(device)
        self.device = torch.device('cuda') if device is None else torch.device(device)
        # Explicitly move masks since it's a child module
        if hasattr(self, 'masks'):
            self.masks.cuda(device)
        if self.verbose > 1:
            print(f"ReflectionData moved to device: {self.device}")
        return self
    
    def cpu(self):
        """
        Move ReflectionData to CPU.

        Returns
        -------
        ReflectionData
            Self, for method chaining.
        """
        super().cpu()
        self.device = torch.device('cpu')
        # Explicitly move masks since it's a child module
        if hasattr(self, 'masks'):
            self.masks.cpu()
        if self.verbose > 1:
            print(f"ReflectionData moved to cpu")
        return self
    
    def load(self, reader):
        """
        Load reflection data using a data reader.

        Parameters
        ----------
        reader : callable
            Data reader object that returns (data_dict, cell, spacegroup) when called.
            Can be MTZ, ReflectionCIFReader, or other compatible reader.

        Returns
        -------
        ReflectionData
            Self, for method chaining.

        Raises
        ------
        ValueError
            If unit cell parameters are missing or no amplitude/intensity data found.
        """

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
            self.masks['flagged_initial'] = ~flagged
        else:
            flagged = torch.zeros(len(self.hkl), dtype=torch.bool, device=self.device)
            self.masks['flagged_initial'] = ~flagged
            self._generate_rfree_flags(free_fraction=0.02, n_bins=20, min_per_bin=100)
            


        self.sanitize_F()
        self.flag_suspicious_sigma()
        self._calculate_wilson_b()
        return self

    def load_mtz(self, path: str) -> 'ReflectionData':
        """
        Load reflection data from MTZ file.

        Parameters
        ----------
        path : str
            Path to MTZ file.

        Returns
        -------
        ReflectionData
            Self, for method chaining.
        """
        reader = legacy_format_readers.MTZ(verbose=self.verbose).read(path)
        return self.load(reader)

    def load_cif(self, path: str, data_block: Optional[str] = None) -> 'ReflectionData':
        """
        Load reflection data from CIF file.

        Parameters
        ----------
        path : str
            Path to CIF file.
        data_block : str, optional
            Specific data block name to read (e.g., 'r1vlmsf'). If None, reads
            the first data block. Useful for multi-dataset CIF files.

        Returns
        -------
        ReflectionData
            Self, for method chaining.
        """
        self.reader = cif_readers.ReflectionCIFReader(path, verbose=self.verbose, data_block=data_block)
        return self.load(self.reader)
    
    @staticmethod
    def list_cif_data_blocks(path: str) -> List[str]:
        """
        List all data blocks available in a CIF file without loading data.

        Useful for multi-dataset CIF files to inspect available blocks
        before loading a specific one.

        Parameters
        ----------
        path : str
            Path to CIF file.

        Returns
        -------
        list of str
            Names of all data blocks in the CIF file.

        Examples
        --------
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
        Generate R-free flags with resolution-stratified sampling.

        Ensures free reflections are evenly distributed across resolution
        shells for unbiased cross-validation.

        Parameters
        ----------
        free_fraction : float, optional
            Fraction of reflections to mark as free (0.02 = 2%). Default is 0.02.
        n_bins : int, optional
            Target number of resolution bins. Default is 20.
        min_per_bin : int, optional
            Minimum reflections per resolution bin. Default is 100.
        seed : int, optional
            Random seed for reproducibility. Default is None.

        Notes
        -----
        Algorithm:
        1. Bin reflections by resolution
        2. Ensure bins have at least min_per_bin reflections
        3. Randomly select free_fraction from each bin
        4. This ensures even distribution across all resolution ranges

        Raises
        ------
        ValueError
            If resolution information is not available.
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
        Create resolution bins with approximately equal reflection counts.

        Parameters
        ----------
        n_bins : int, optional
            Target number of resolution bins. Default is 20.
        min_per_bin : int, optional
            Minimum reflections per bin. Default is 100.

        Returns
        -------
        bin_indices : torch.Tensor
            Tensor of shape (N,) with bin index for each reflection.
        n_bins : int
            Actual number of bins created (may be less than target for small datasets).
        """
        n_refl = len(self.resolution)
        valid_mask = self.masks()
        total_valid = valid_mask.sum().item()
        
        # Calculate how many bins we can actually create given min_per_bin constraint
        max_possible_bins = max(1, total_valid // min_per_bin)
        actual_n_bins = min(n_bins, max_possible_bins)
        
        if actual_n_bins < n_bins and self.verbose > 0:
            print(f"  Note: Adjusted bins from {n_bins} to {actual_n_bins} (min {min_per_bin} refl/bin)")
        
        # Sort reflections by resolution
        _, sort_indices = torch.sort(self.resolution)
        
        # Create bins with approximately equal number of VALID reflections
        bin_indices = torch.zeros(n_refl, dtype=torch.int32, device=self.device)
        reflections_per_bin = total_valid // actual_n_bins
        
        # Get the valid mask in sorted order
        valid_mask_sorted = valid_mask[sort_indices]
        
        # Cumulative sum of valid reflections in sorted order
        cumsum_valid = torch.cumsum(valid_mask_sorted.to(torch.int32), dim=0)
        
        # Create bin edges based on cumulative count of valid reflections
        # Each bin should contain approximately reflections_per_bin valid reflections
        bin_edges = [0]
        for bin_idx in range(1, actual_n_bins):
            target_count = bin_idx * reflections_per_bin
            # Find first index where cumsum >= target_count
            edge_candidates = torch.where(cumsum_valid >= target_count)[0]
            if len(edge_candidates) > 0:
                bin_edges.append(edge_candidates[0].item())
        bin_edges.append(n_refl)
        
        # Assign bin indices to sorted reflections, then map back to original order
        for bin_idx in range(len(bin_edges) - 1):
            start, end = bin_edges[bin_idx], bin_edges[bin_idx + 1]
            bin_indices[sort_indices[start:end]] = bin_idx

        if self.verbose > 1:
            # Print bin statistics
            print(f"  Resolution bins:")
            for bin_idx in range(min(actual_n_bins, 20)):  # Show all bins (up to 20)
                bin_mask = bin_indices == bin_idx
                if bin_mask.sum() > 0:
                    valid_reflexes = bin_mask & valid_mask
                    bin_res = self.resolution[bin_mask]
                    print(f"    Bin {bin_idx:2d}: {valid_reflexes.sum():6d} valid refl, "
                        f"resolution {bin_res.min():.2f}-{bin_res.max():.2f} Å")
            if actual_n_bins > 20:
                print(f"    ... ({actual_n_bins - 20} more bins)")
        self.register_buffer('bin_indices', bin_indices)
        self.register_buffer('n_bins', torch.tensor(actual_n_bins, dtype=torch.int32))
        return bin_indices, actual_n_bins
    
    def mean_res_per_bin(self) -> torch.Tensor:
        """
        Calculate mean resolution for each bin.

        Returns
        -------
        torch.Tensor
            Mean resolution for each bin in Ångströms.

        Raises
        ------
        ValueError
            If bins have not been created yet.
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
        Regenerate R-free flags with resolution-stratified sampling.

        Parameters
        ----------
        free_fraction : float, optional
            Fraction of reflections to mark as free. Default is 0.02 (2%).
        n_bins : int, optional
            Target number of resolution bins. Default is 20.
        min_per_bin : int, optional
            Minimum reflections per resolution bin. Default is 100.
        seed : int, optional
            Random seed for reproducibility. Default is None.
        force : bool, optional
            If True, regenerate even if flags already exist. Default is False.

        Examples
        --------
        >>> # Generate 2% free reflections with reproducible seed
        >>> data.regenerate_rfree_flags(free_fraction=0.02, n_bins=20, seed=42)
        >>> # Generate 5% free with 10 bins, overwriting existing
        >>> data.regenerate_rfree_flags(free_fraction=0.05, n_bins=10, force=True)
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
        """
        Calculate resolution for each reflection from Miller indices.

        Sets the `resolution` buffer with d-spacing values in Ångströms.

        Raises
        ------
        ValueError
            If Miller indices or unit cell parameters are missing.
        """
        if self.hkl is None:
            raise ValueError("Miller indices (hkl) are required to calculate resolution")
        if self.cell is None:
            raise ValueError("Unit cell parameters are required to calculate resolution")
        s = math_torch.get_scattering_vectors(self.hkl, self.cell)
        resolution = 1.0 / torch.linalg.norm(s, axis=1)
        self.register_buffer('resolution', resolution)
    
    def _calculate_wilson_b(self, n_bins: int = 30) -> None:
        """
        Calculate Wilson B-factors from structure factor amplitudes.
        
        Fits a two-component model separating structure and solvent contributions:
        <F²> ∝ A_struct * exp(-2*B_struct*s²) + A_sol * exp(-2*B_sol*s²)
        
        The fitting proceeds in three stages:
        1. Estimate B_structure from high-resolution data (d < 3.5 Å) where solvent is negligible
        2. Estimate B_solvent from low-resolution data (d > 6 Å) where solvent dominates  
        3. Refine both together with two-exponential fit across all data
        
        Args:
            n_bins: Number of resolution bins for averaging (default: 30)
        
        Sets:
            self.wilson_b: Overall Wilson B-factor (weighted average) in Å²
            self.wilson_b_structure: Structure B-factor from high-res in Å²
            self.wilson_b_solvent: Solvent B-factor from low-res in Å²
            self.wilson_k_sol: Relative solvent contribution (0-1)
        """
        if self.F is None or self.resolution is None:
            return
        
        # Get valid reflections
        F = self.F
        d = self.resolution
        valid = torch.isfinite(F) & (F > 0) & torch.isfinite(d)
        
        if valid.sum() < 100:
            if self.verbose > 0:
                print(f"  Wilson B: insufficient data ({valid.sum()} reflections), skipping")
            return
        
        F_valid = F[valid]
        d_valid = d[valid]
        
        # Calculate s² = 1/(4d²)
        s_sq = 1.0 / (4.0 * d_valid**2)
        F_sq = F_valid**2
        
        # Bin the data for noise reduction
        s_sq_min, s_sq_max = s_sq.min(), s_sq.max()
        bin_edges = torch.linspace(s_sq_min, s_sq_max, n_bins + 1, device=self.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_idx = torch.bucketize(s_sq, bin_edges[1:-1])
        
        # Calculate mean F² per bin
        bin_sums = torch.zeros(n_bins, device=self.device, dtype=F_sq.dtype)
        bin_counts = torch.zeros(n_bins, device=self.device, dtype=F_sq.dtype)
        bin_sums.scatter_add_(0, bin_idx, F_sq)
        bin_counts.scatter_add_(0, bin_idx, torch.ones_like(F_sq))
        
        valid_bins = bin_counts > 5
        if valid_bins.sum() < 5:
            if self.verbose > 0:
                print(f"  Wilson B: insufficient bins ({valid_bins.sum()}), skipping")
            return
        
        mean_F_sq = bin_sums[valid_bins] / bin_counts[valid_bins]
        s_sq_bins = bin_centers[valid_bins]
        
        # Convert s² back to d-spacing for resolution-based selection
        d_bins = 1.0 / (2.0 * torch.sqrt(s_sq_bins))
        
        # Stage 1: Fit high-resolution region (d < 3.5 Å) for structure B
        high_res_mask = d_bins < 3.5
        B_struct = self._fit_single_wilson(s_sq_bins, mean_F_sq, high_res_mask, "high-res")
        
        # Stage 2: Fit low-resolution region (d > 6 Å) for solvent B
        low_res_mask = d_bins > 6.0
        B_sol = self._fit_single_wilson(s_sq_bins, mean_F_sq, low_res_mask, "low-res")
        
        # Stage 3: Two-component fit across all data
        B_struct_final, B_sol_final, k_sol = self._fit_two_component_wilson(
            s_sq_bins, mean_F_sq, B_struct, B_sol
        )
        
        # Store results
        self.wilson_b_structure = B_struct_final
        self.wilson_b_solvent = B_sol_final
        self.wilson_k_sol = k_sol
        
        # Overall Wilson B is the structure B (what people usually mean by "Wilson B")
        self.wilson_b = B_struct_final
        
        if self.verbose > 0:
            print(f"  Wilson B-factor (structure): {B_struct_final:.1f} Å²")
            print(f"  Wilson B-factor (solvent):   {B_sol_final:.1f} Å²")
            print(f"  Solvent fraction (k_sol):    {k_sol:.3f}")
    
    def _fit_single_wilson(
        self, 
        s_sq: torch.Tensor, 
        mean_F_sq: torch.Tensor, 
        mask: torch.Tensor,
        label: str
    ) -> float:
        """
        Fit single-exponential Wilson plot to selected resolution range.
        
        Args:
            s_sq: s² values for bins
            mean_F_sq: Mean F² values for bins  
            mask: Boolean mask selecting which bins to use
            label: Label for error messages
            
        Returns:
            B-factor from fit (Å²)
        """
        if mask.sum() < 3:
            # Not enough data, return reasonable default
            if self.verbose > 1:
                print(f"    Wilson {label}: insufficient bins ({mask.sum()}), using default")
            return 50.0 if "struct" in label else 200.0
        
        x = s_sq[mask]
        y = torch.log(mean_F_sq[mask])
        
        # Linear regression: ln(F²) = const - 2B*s²
        x_mean = x.mean()
        y_mean = y.mean()
        
        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean)**2).sum()
        
        if denominator < 1e-12:
            return 50.0 if "struct" in label else 200.0
        
        slope = numerator / denominator
        B = -slope.item() / 2.0
        
        # Sanity bounds
        B = max(0.0, min(B, 300.0))
        
        return B
    
    def _fit_two_component_wilson(
        self,
        s_sq: torch.Tensor,
        mean_F_sq: torch.Tensor,
        B_struct_init: float,
        B_sol_init: float,
        n_iter: int = 50
    ) -> Tuple[float, float, float]:
        """
        Fit two-component Wilson model using iterative refinement.

        Model: F² = A_struct * exp(-2*B_struct*s²) + A_sol * exp(-2*B_sol*s²)

        Parameterized as:
            F² = A * [(1-k)*exp(-2*B_struct*s²) + k*exp(-2*B_sol*s²)]

        where k is the relative solvent contribution at s²=0.

        Parameters
        ----------
        s_sq : torch.Tensor
            s² values for bins.
        mean_F_sq : torch.Tensor
            Mean F² values for bins.
        B_struct_init : float
            Initial structure B-factor.
        B_sol_init : float
            Initial solvent B-factor.
        n_iter : int, optional
            Number of refinement iterations. Default is 50.

        Returns
        -------
        B_struct : float
            Refined structure B-factor.
        B_sol : float
            Refined solvent B-factor.
        k_sol : float
            Relative solvent contribution.
        """
        # Normalize F² for numerical stability
        F_sq_max = mean_F_sq.max()
        y = mean_F_sq / F_sq_max
        x = s_sq
        
        # Initialize parameters
        B_struct = torch.tensor(B_struct_init, device=self.device, dtype=x.dtype)
        B_sol = torch.tensor(B_sol_init, device=self.device, dtype=x.dtype)
        
        # Estimate initial k from ratio of low-res to high-res decay
        # At low resolution, solvent contributes more
        d_from_s = 1.0 / (2.0 * torch.sqrt(x))
        low_res_val = y[d_from_s > 5.0].mean() if (d_from_s > 5.0).any() else y[0]
        high_res_val = y[d_from_s < 3.0].mean() if (d_from_s < 3.0).any() else y[-1]
        
        # k estimates solvent fraction - if low res is much higher than expected
        # from structure alone, there's solvent contribution
        struct_decay = torch.exp(-2 * B_struct * x)
        expected_low = struct_decay[d_from_s > 5.0].mean() if (d_from_s > 5.0).any() else struct_decay[0]
        
        if expected_low > 1e-6 and low_res_val > expected_low:
            k_init = min(0.5, (low_res_val - expected_low).item() / low_res_val.item())
        else:
            k_init = 0.1
        
        k = torch.tensor(max(0.01, min(0.5, k_init)), device=self.device, dtype=x.dtype)
        
        # Simple gradient descent refinement
        lr = 0.1
        
        for _ in range(n_iter):
            # Compute model
            struct_term = (1 - k) * torch.exp(-2 * B_struct * x)
            sol_term = k * torch.exp(-2 * B_sol * x)
            model = struct_term + sol_term
            
            # Compute scale factor analytically
            A = (y * model).sum() / (model * model).sum()
            model_scaled = A * model
            
            # Compute gradients (simplified, using finite differences for robustness)
            eps = 0.1
            
            # B_struct gradient
            model_plus = A * ((1 - k) * torch.exp(-2 * (B_struct + eps) * x) + sol_term)
            model_minus = A * ((1 - k) * torch.exp(-2 * (B_struct - eps) * x) + sol_term)
            loss_plus = ((y - model_plus)**2).sum()
            loss_minus = ((y - model_minus)**2).sum()
            grad_B_struct = (loss_plus - loss_minus) / (2 * eps)
            
            # B_sol gradient  
            model_plus = A * (struct_term + k * torch.exp(-2 * (B_sol + eps) * x))
            model_minus = A * (struct_term + k * torch.exp(-2 * (B_sol - eps) * x))
            loss_plus = ((y - model_plus)**2).sum()
            loss_minus = ((y - model_minus)**2).sum()
            grad_B_sol = (loss_plus - loss_minus) / (2 * eps)
            
            # k gradient
            eps_k = 0.01
            k_plus = min(0.9, k + eps_k)
            k_minus = max(0.01, k - eps_k)
            model_plus = A * ((1 - k_plus) * torch.exp(-2 * B_struct * x) + k_plus * torch.exp(-2 * B_sol * x))
            model_minus = A * ((1 - k_minus) * torch.exp(-2 * B_struct * x) + k_minus * torch.exp(-2 * B_sol * x))
            loss_plus = ((y - model_plus)**2).sum()
            loss_minus = ((y - model_minus)**2).sum()
            grad_k = (loss_plus - loss_minus) / (2 * eps_k)
            
            # Update parameters
            B_struct = B_struct - lr * grad_B_struct
            B_sol = B_sol - lr * grad_B_sol
            k = k - lr * 0.1 * grad_k  # Slower learning rate for k
            
            # Enforce constraints
            B_struct = torch.clamp(B_struct, 1.0, 200.0)
            B_sol = torch.clamp(B_sol, 50.0, 500.0)
            k = torch.clamp(k, 0.01, 0.9)
            
            # Ensure B_sol > B_struct (solvent is more disordered)
            if B_sol < B_struct + 20:
                B_sol = B_struct + 20
        
        return B_struct.item(), B_sol.item(), k.item()
    
    def get_structure_factors(self, as_complex: bool = False) -> torch.Tensor:
        """
        Get structure factors, optionally as complex numbers.

        Parameters
        ----------
        as_complex : bool, optional
            If True and phases available, return F*exp(i*phi). Default is False.

        Returns
        -------
        torch.Tensor
            Structure factor amplitudes or complex structure factors.

        Raises
        ------
        ValueError
            If no amplitude data is loaded.
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

        Returns
        -------
        F : torch.Tensor
            Structure factor amplitudes of shape (N,).
        F_sigma : torch.Tensor or None
            Uncertainties of shape (N,), or None if not available.

        Raises
        ------
        ValueError
            If no amplitude data is loaded.

        Examples
        --------
        >>> F, sigma_F = data.get_structure_factors_with_sigma()
        >>> if sigma_F is not None:
        ...     weighted_residual = (F_obs - F_calc) / sigma_F
        """
        if self.F is None:
            raise ValueError("No amplitude data loaded")
        
        return self.F, self.F_sigma
    
    def get_hkl(self):
        """
        Return Miller indices for valid reflections.

        Returns
        -------
        torch.Tensor
            Miller indices of shape (N, 3), dtype int32.

        Raises
        ------
        ValueError
            If no Miller indices are loaded.
        """
        if self.hkl is None:
            raise ValueError("No Miller indices loaded")
        return self.hkl[self.masks()]

    def filter_by_resolution(self, d_min: Optional[float] = None, 
                            d_max: Optional[float] = None) -> 'ReflectionData':
        """
        Filter reflections by resolution range.

        Adds a boolean mask to self.masks for the specified resolution range.

        Parameters
        ----------
        d_min : float, optional
            Minimum resolution / high resolution cutoff (e.g., 1.5 Å).
        d_max : float, optional
            Maximum resolution / low resolution cutoff (e.g., 50.0 Å).

        Returns
        -------
        ReflectionData
            Self, for method chaining.
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
        """
        Return combined mask from all active filters.

        Returns
        -------
        torch.Tensor
            Boolean mask combining all filter conditions.
        """
    
    def cut_res(self, highres: Optional[float] = None, 
                lowres: Optional[float] = None) -> 'ReflectionData':
        """
        Filter reflections by resolution range.

        Alias for filter_by_resolution with more intuitive naming.

        Parameters
        ----------
        highres : float, optional
            High resolution cutoff (small d-spacing, e.g., 1.5 Å).
            Keeps reflections with d >= highres.
        lowres : float, optional
            Low resolution cutoff (large d-spacing, e.g., 50.0 Å).
            Keeps reflections with d <= lowres.

        Returns
        -------
        ReflectionData
            Self, for method chaining.

        Examples
        --------
        >>> # Keep reflections between 50 Å and 1.5 Å
        >>> filtered = data.cut_res(highres=1.5, lowres=50.0)
        >>> # Keep only high-resolution data (< 2 Å)
        >>> high_res = data.cut_res(highres=1.0, lowres=2.0)
        """
        return self.filter_by_resolution(d_min=highres, d_max=lowres)
    
    def get_rfree_masks(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get boolean masks for work and test (free) sets.

        Returns
        -------
        work_mask : torch.Tensor or None
            Boolean tensor for work set (flag != 0).
        test_mask : torch.Tensor or None
            Boolean tensor for test/free set (flag == 0).
            Both are None if no R-free flags are available.

        Examples
        --------
        >>> work_mask, test_mask = data.get_rfree_masks()
        >>> if work_mask is not None:
        ...     F_work = data.F[work_mask]
        ...     F_test = data.F[test_mask]
        """
        if self.rfree_flags is None:
            return None, None
        
        work_mask = self.rfree_flags != 0
        test_mask = self.rfree_flags == 0
        
        return work_mask, test_mask
    
    def get_work_set(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get structure factors for the work set (R-free flag != 0).

        Returns
        -------
        F_work : torch.Tensor
            Structure factors for work set.
        sigma_work : torch.Tensor or None
            Uncertainties for work set, or None if not available.

        Notes
        -----
        Returns full dataset with warning if no R-free flags available.
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

        Returns
        -------
        F_test : torch.Tensor
            Structure factors for test/free set.
        sigma_test : torch.Tensor or None
            Uncertainties for test set, or None if not available.

        Raises
        ------
        ValueError
            If no R-free flags are available.
        """
        if self.rfree_flags is None:
            raise ValueError("No R-free flags available in dataset")
        
        test_mask = self.rfree_flags == 0
        F_test = self.F[test_mask] if self.F is not None else None
        sigma_test = self.F_sigma[test_mask] if self.F_sigma is not None else None
        
        return F_test, sigma_test

    def get_max_res(self) -> Optional[float]:
        """
        Return maximum resolution (lowest d-spacing).

        Returns
        -------
        float
            Maximum resolution in Ångströms.
        """
        if self.resolution is None:
            self._calculate_resolution()
        return float(self.resolution.min().item())

    def __len__(self) -> int:
        """
        Return number of reflections.

        Returns
        -------
        int
            Number of reflections in the dataset.
        """
        return len(self.hkl) if self.hkl is not None else 0
    
    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            Summary of reflection data including count, sources, and resolution.
        """
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

    def forward(
        self, 
        mask: bool = True
    ) -> Tuple[
        torch.Tensor, 
        'MaskedTensor', 
        Optional['MaskedTensor'], 
        Optional[torch.Tensor]
    ]:
        """
        Return core reflection data with MaskedTensors for F and sigma.

        F and F_sigma are returned as MaskedTensors which keep all reflections
        but mark invalid ones as masked. Aggregation operations (sum, mean, etc.)
        automatically skip masked values. HKL and rfree_flags remain regular tensors.

        Parameters
        ----------
        mask : bool, optional
            If True, apply current masks to F and sigma. Default is True.

        Returns
        -------
        hkl : torch.Tensor
            Miller indices of shape (N, 3). Full size, unfiltered.
        F : MaskedTensor
            Structure factor amplitudes of shape (N,) with invalid reflections masked.
        F_sigma : MaskedTensor or None
            Uncertainties of shape (N,) with invalid reflections masked, or None.
        rfree_flags : torch.Tensor or None
            R-free flags of shape (N,) or None. Full size, unfiltered. 1=work, 0=free.

        Notes
        -----
        MaskedTensors:

        - Are PyTorch tensors with an associated boolean mask
        - Aggregations (sum, mean, etc.) skip masked values automatically
        - Element-wise operations preserve the mask
        - Use .get_data() and .get_mask() to access underlying data
        - Use .to_tensor(fill_value) to convert back to regular tensor
        - Note: MaskedTensor is in prototype stage in PyTorch

        Loss functions and targets extract valid data from MaskedTensors before
        computation to work correctly with complex F_calc values.

        Examples
        --------
        >>> hkl, F, sigma, rfree = data()
        >>> print(F.shape)  # Full shape
        >>> print(F.sum())  # Only sums valid (unmasked) values
        >>> 
        >>> # Access underlying data
        >>> valid_mask = F.get_mask()
        >>> F_values = F.get_data()[valid_mask]
        """
        from torch.masked import MaskedTensor
        
        hkl, F, F_sigma, rfree_flags = self.hkl, self.F, self.F_sigma, self.rfree_flags
        
        if mask:
            to_mask = self.masks()
            F = MaskedTensor(F, to_mask)
            if F_sigma is not None:
                F_sigma = MaskedTensor(F_sigma, to_mask)
        
        return hkl, F, F_sigma, rfree_flags

    def get_valid_mask(self) -> torch.Tensor:
        """
        Return the combined validity mask for all reflections.

        This is the mask used to filter reflections in forward(). True indicates
        a valid (included) reflection, False indicates an excluded one.

        Returns
        -------
        torch.Tensor
            Boolean mask of shape (N,) where N is the total number of reflections.
            True = valid/included, False = invalid/excluded.

        Examples
        --------
        >>> mask = data.get_valid_mask()
        >>> print(f"{mask.sum()} of {len(mask)} reflections are valid")
        """
        return self.masks()

    def forward_indexed(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Return reflection data as indexed (filtered) tensors.

        This method filters out invalid reflections and returns smaller tensors
        containing only valid data. Useful for operations that don't support
        MaskedTensors or for writing output files.

        Returns
        -------
        hkl : torch.Tensor
            Miller indices of shape (M, 3) where M is number of valid reflections.
        F : torch.Tensor
            Structure factor amplitudes of shape (M,).
        F_sigma : torch.Tensor or None
            Uncertainties of shape (M,) or None.
        rfree_flags : torch.Tensor or None
            R-free flags of shape (M,) or None.

        See Also
        --------
        forward : Main method returning MaskedTensors.

        Examples
        --------
        >>> hkl, F, sigma, rfree = data.forward_indexed()
        >>> F_np = F.cpu().numpy()  # Safe for writing to files
        """
        to_mask = self.masks()
        
        hkl = self.hkl[to_mask]
        F = self.F[to_mask]
        F_sigma = self.F_sigma[to_mask] if self.F_sigma is not None else None
        rfree_flags = self.rfree_flags[to_mask].to(torch.bool) if self.rfree_flags is not None else None
        
        return hkl, F, F_sigma, rfree_flags
    
    def __select__(self,mask:torch.Tensor, op=None)-> 'ReflectionData':
        """
        Select reflections based on a boolean mask.

        Parameters
        ----------
        mask : torch.Tensor
            Boolean mask of shape (N,) for selection.
        op : str, optional
            Operation name for tracking purposes.

        Returns
        -------
        ReflectionData
            New ReflectionData object with selected reflections.
        """
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
    

    def sanitize_F(self):
        """
        Remove invalid values from structure factors.

        Adds a mask to filter out NaN, Inf, and non-positive values
        from F and F_sigma.
        """
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
    
    def validate_hkl(self, hkl_ref: torch.Tensor) -> 'ReflectionData':
        """
        Expand dataset to match a reference HKL set.

        Reorders and expands the current dataset to match the reference HKL set.
        Reflections present in the reference but missing from this dataset are
        filled with placeholder values and masked out. This ensures all datasets
        aligned to the same reference have identical shapes and can be processed
        together without data loss from intersection operations.

        Parameters
        ----------
        hkl_ref : torch.Tensor
            Reference Miller indices tensor of shape (N, 3), dtype int32.
            This defines the canonical HKL ordering for all aligned datasets.

        Returns
        -------
        ReflectionData
            Self, modified in-place with expanded arrays matching hkl_ref.

        Notes
        -----
        After calling this method:
        - self.hkl will equal hkl_ref exactly
        - All data arrays (F, F_sigma, rfree_flags, etc.) are reordered/expanded
        - Missing reflections are filled with 0 (or appropriate defaults)
        - A mask 'hkl_present' is added marking which reflections have real data
        - forward() will return MaskedTensors that skip missing reflections

        This approach avoids the problem where intersecting many datasets with
        different outliers/missing reflections causes exponential data loss.

        Examples
        --------
        >>> # Align multiple datasets to a common HKL set
        >>> reference_hkl = data1.hkl.clone()
        >>> data1.validate_hkl(reference_hkl)
        >>> data2.validate_hkl(reference_hkl)
        >>> # Now data1 and data2 have identical shapes
        >>> assert data1.hkl.shape == data2.hkl.shape
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
        hkl_ref = hkl_ref.to(dtype=torch.int32, device=self.device)
        
        n_ref = len(hkl_ref)
        n_data = len(self.hkl)
        
        # Build lookup from data HKL to index
        # Use a dictionary with tuple keys for fast lookup
        hkl_data_np = self.hkl.cpu().numpy()
        data_hkl_to_idx = {tuple(hkl): idx for idx, hkl in enumerate(hkl_data_np)}
        
        # For each reference HKL, find the corresponding data index (or -1 if missing)
        hkl_ref_np = hkl_ref.cpu().numpy()
        ref_to_data_idx = np.array([
            data_hkl_to_idx.get(tuple(hkl), -1) for hkl in hkl_ref_np
        ], dtype=np.int64)
        
        # Create presence mask: True where data exists
        presence_mask = torch.from_numpy(ref_to_data_idx >= 0).to(device=self.device)
        valid_indices = torch.from_numpy(ref_to_data_idx).to(device=self.device)
        
        # Helper to expand a tensor to reference size
        def expand_tensor(tensor, fill_value=0.0):
            if tensor is None:
                return None
            expanded = torch.full((n_ref,) + tensor.shape[1:], fill_value, 
                                  dtype=tensor.dtype, device=self.device)
            # Copy existing data to correct positions
            mask = valid_indices >= 0
            expanded[mask] = tensor[valid_indices[mask]]
            return expanded
        
        # Expand all data arrays
        old_F = self.F
        old_F_sigma = self.F_sigma
        old_I = self.I
        old_I_sigma = getattr(self, 'I_sigma', None)
        old_rfree = self.rfree_flags
        old_phase = getattr(self, 'phase', None)
        old_fom = getattr(self, 'fom', None)
        
        # Replace HKL with reference
        self.register_buffer('hkl', hkl_ref)
        
        # Expand data tensors
        self.register_buffer('F', expand_tensor(old_F, fill_value=0.0))
        self.register_buffer('F_sigma', expand_tensor(old_F_sigma, fill_value=1.0))
        
        if old_I is not None:
            self.register_buffer('I', expand_tensor(old_I, fill_value=0.0))
        if old_I_sigma is not None:
            self.register_buffer('I_sigma', expand_tensor(old_I_sigma, fill_value=1.0))
        
        # For rfree, default missing to work set (1)
        if old_rfree is not None:
            rfree_expanded = torch.ones(n_ref, dtype=old_rfree.dtype, device=self.device)
            mask = valid_indices >= 0
            rfree_expanded[mask] = old_rfree[valid_indices[mask]]
            self.register_buffer('rfree_flags', rfree_expanded)
        
        # Recalculate resolution for new HKL set
        self._calculate_resolution()
        
        # Expand phase and fom if present
        if old_phase is not None:
            self.register_buffer('phase', expand_tensor(old_phase, fill_value=0.0))
        if old_fom is not None:
            self.register_buffer('fom', expand_tensor(old_fom, fill_value=0.0))
        
        # Transfer existing masks to new indexing
        old_masks = dict(self.masks.items())
        # Clear existing masks by removing each key
        for name in list(self.masks.keys()):
            delattr(self.masks, f"_buf_{name}")
        self.masks._keys.clear()
        self.masks.updated = True
        
        for name, old_mask in old_masks.items():
            if old_mask is not None and len(old_mask) == n_data:
                # Expand mask: missing reflections are masked out (False)
                new_mask = torch.zeros(n_ref, dtype=torch.bool, device=self.device)
                mask = valid_indices >= 0
                new_mask[mask] = old_mask[valid_indices[mask]]
                self.masks[name] = new_mask
        
        # Add presence mask - this is the key mask that marks real vs placeholder data
        self.masks['hkl_present'] = presence_mask
        
        n_present = presence_mask.sum().item()
        n_missing = n_ref - n_present
        
        if self.verbose > 0:
            print(f"HKL validation (expand mode):")
            print(f"  Original dataset: {n_data} reflections")
            print(f"  Reference set: {n_ref} reflections")
            print(f"  Present in data: {n_present} ({100*n_present/n_ref:.1f}%)")
            print(f"  Missing (masked): {n_missing} ({100*n_missing/n_ref:.1f}%)")
        
        return self
    
    def find_outliers(self, model: ModelFT, scaler, z_threshold: float = 4.0) -> torch.Tensor:
        """
        Identify outlier reflections based on log-ratio distribution.

        Uses the fact that log(F_obs) - log(F_calc) should be normally distributed.
        Outliers are reflections where |log_ratio - mean| > z_threshold * std_dev.

        Parameters
        ----------
        model : ModelFT
            ModelFT object to compute structure factors.
        scaler : Scaler
            Scaler object to scale calculated structure factors.
        z_threshold : float, optional
            Z-score threshold to classify outliers. Default is 4.0.

        Returns
        -------
        torch.Tensor
            Boolean mask where True indicates outliers.
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
        """
        Compute log-ratio between observed and calculated structure factors.

        Parameters
        ----------
        model : ModelFT
            ModelFT object to compute structure factors.
        scaler : Scaler
            Scaler object to scale calculated structure factors.

        Returns
        -------
        torch.Tensor
            Log-ratio values: log(F_obs) - log(F_calc).
        """
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
        """
        Get statistics about flagged outliers.

        Returns
        -------
        dict
            Dictionary containing:
            - n_outliers : int
            - n_total : int
            - fraction_outliers : float
            - detection_params : dict or None
            - outlier_resolution_stats : dict (if resolution available)
        """
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

        Does not recurse fully and does not flag.

        Returns
        -------
        ReflectionData
            Parent source or self if no source.
        """
        if self.source is not None:
            return self.source
        return self

    def flag_suspicious_sigma(self, z_threshold: float = 5.0) -> None:
        """
        Flag sigma values that deviate significantly from expected distribution.

        Sigma values from a detector should follow a log-normal distribution.
        Values with z-scores beyond threshold are flagged as suspicious.

        Parameters
        ----------
        z_threshold : float, optional
            Z-score threshold for flagging outliers. Default is 5.0.
        """
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
        """
        Dump all reflection data to console for debugging.

        Prints type, shape, and device information for all attributes.
        """
        print("ReflectionData dump:")
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, torch.Tensor):
                print(f"  {key}: dtype={value.dtype}, shape={value.shape}, device={value.device}")
            else:
                print(f"  {key}: type={type(value)}, value={value}")

    def write_mtz(self, fname: str, fcalc: Optional[torch.Tensor] = None, 
                  model_ft: Optional[ModelFT] = None) -> None:
        """
        Write reflection data to MTZ file with optional map coefficients.

        Parameters
        ----------
        fname : str
            Output MTZ filename.
        fcalc : torch.Tensor, optional
            Complex calculated structure factors of shape (N,).
            If provided, computes phases and map coefficients.
        model_ft : ModelFT, optional
            ModelFT object to compute fcalc if not provided.
        fill_to_resolution : bool, optional
            If True and fcalc provided, fill map coefficients to resolution
            limit. Default is True.

        Notes
        -----
        The MTZ file will contain canonical column names:
            - FP, SIGFP: Observed amplitudes and uncertainties
            - I, SIGI: Observed intensities and uncertainties (if available)
            - FreeR_flag: R-free test set flags
            - FWT, PHWT: 2mFo-DFc map coefficients (if fcalc provided)
            - DELFWT, PHDELWT: mFo-DFc map coefficients (if fcalc provided)

        Map coefficients are computed as:
            - 2mFo-DFc: 2*Fo - Fc (filled to resolution limit)
            - mFo-DFc: Fo - Fc

        Examples
        --------
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
        mask = self.masks().detach().cpu().numpy()
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
            two_mfo_dfc_amp[~mask] = 0.0  # Zero out reflections outside mask
            two_mfo_dfc_phase = phases
            
            # mFo-DFc: Difference map
            mfo_dfc_complex = F_obs * np.exp(1j * np.deg2rad(phases)) - fcalc_np
            mfo_dfc_complex[~mask] = 0.0  # Zero out reflections outside mask
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
        Return dictionary containing complete state of ReflectionData.

        Includes all registered buffers, masks, and metadata.

        Parameters
        ----------
        destination : dict, optional
            Optional dict to populate.
        prefix : str, optional
            Prefix for parameter names.
        keep_vars : bool, optional
            Whether to keep variables in computational graph.

        Returns
        -------
        dict
            Complete state dictionary.
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
        state[prefix + 'wilson_b'] = self.wilson_b
        state[prefix + 'wilson_b_structure'] = self.wilson_b_structure
        state[prefix + 'wilson_b_solvent'] = self.wilson_b_solvent
        state[prefix + 'wilson_k_sol'] = self.wilson_k_sol
        state[prefix + 'verbose'] = self.verbose
        
        return state

    def save_state(self, path: str):
        """
        Save complete state of reflection data to file.

        Parameters
        ----------
        path : str
            Path to save the state dictionary to.
        """
        torch.save(self.state_dict(), path)
        if self.verbose > 0:
            print(f"Saved reflection data state to {path}")

    def load_state(self, path: str, strict: bool = True):
        """
        Load complete state of reflection data from file.

        Parameters
        ----------
        path : str
            Path to load the state dictionary from.
        strict : bool, optional
            Whether to strictly enforce that keys match. Default is True.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict, strict=strict)
        if self.verbose > 0:
            print(f"Loaded reflection data state from {path}")
    
    @classmethod
    def create_from_state_dict(cls, state_dict: dict, device: str = 'cpu', verbose: int = 1) -> 'ReflectionData':
        """
        Create a fully initialized ReflectionData from a state dictionary.
        
        This is the recommended way to restore ReflectionData from a saved state.
        It creates an instance with properly sized buffers, then loads the state.
        
        Args:
            state_dict: State dictionary from torch.save(reflection_data.state_dict(), ...)
            device: Device to place tensors on ('cpu' or 'cuda')
            verbose: Verbosity level
            
        Returns:
            ReflectionData: Fully initialized instance with restored state
        """
        # Extract metadata (these are not tensors, need special handling)
        spacegroup = state_dict.pop('spacegroup', None)
        rfree_source = state_dict.pop('rfree_source', None)
        outlier_detection_params = state_dict.pop('outlier_detection_params', None)
        amplitude_source = state_dict.pop('amplitude_source', None)
        intensity_source = state_dict.pop('intensity_source', None)
        phase_source = state_dict.pop('phase_source', None)
        wilson_b = state_dict.pop('wilson_b', None)
        wilson_b_structure = state_dict.pop('wilson_b_structure', None)
        wilson_b_solvent = state_dict.pop('wilson_b_solvent', None)
        wilson_k_sol = state_dict.pop('wilson_k_sol', None)
        saved_verbose = state_dict.pop('verbose', verbose)
        
        # Create instance
        instance = cls(verbose=saved_verbose, device=device)
        
        # Set metadata
        instance.spacegroup = spacegroup
        instance.rfree_source = rfree_source
        instance.outlier_detection_params = outlier_detection_params
        instance.amplitude_source = amplitude_source
        instance.intensity_source = intensity_source
        instance.phase_source = phase_source
        instance.wilson_b = wilson_b
        instance.wilson_b_structure = wilson_b_structure
        instance.wilson_b_solvent = wilson_b_solvent
        instance.wilson_k_sol = wilson_k_sol
        
        # Register buffers with correct shapes before loading
        if 'hkl' in state_dict and state_dict['hkl'] is not None:
            hkl = state_dict['hkl']
            instance.register_buffer('hkl', torch.zeros_like(hkl, device=device))
        
        buffer_names = ['F', 'F_sigma', 'I', 'I_sigma', 'rfree_flags', 'outlier_flags', 
                       'resolution', 'bin_indices', 'n_bins', 'cell']
        for name in buffer_names:
            if name in state_dict and state_dict[name] is not None:
                tensor = state_dict[name]
                instance.register_buffer(name, torch.zeros_like(tensor, device=device))
        
        # Create FrenchWilson if present
        has_french_wilson = any(k.startswith('FrenchWilson.') for k in state_dict.keys())
        if has_french_wilson and 'hkl' in state_dict and 'cell' in state_dict:
            hkl = state_dict['hkl'].to(device)
            cell = state_dict['cell'].to(device)
            instance.FrenchWilson = FrenchWilson(hkl, cell, spacegroup, verbose=saved_verbose)
            instance.FrenchWilson.to(device)
        
        # Now use PyTorch's default load_state_dict
        instance.load_state_dict(state_dict, strict=False)
        
        if verbose > 0:
            n_refl = instance.hkl.shape[0] if instance.hkl is not None else 0
            print(f"Created ReflectionData from state_dict: {n_refl} reflections")
        
        return instance
    
    @property
    def centric(self):
        """
        Get boolean mask for centric reflections (full size, unfiltered).
        
        Calculates it if not already present. Returns unfiltered centric flags
        matching the full HKL array size, consistent with how forward() returns
        full-size arrays when using MaskedTensors.
        
        Returns
        -------
        torch.Tensor or None
            Boolean tensor of shape (N,) where N is total reflections.
            True indicates centric reflection, False indicates acentric.
        """
        if self.hkl is None:
            return None
            
        # Check if we already have it cached (could be stored in a buffer if we want persistence)
        if not hasattr(self, '_centric_flags') or self._centric_flags is None:
            from torchref.math_functions.french_wilson import is_centric_from_hkl
            
            # Ensure we have spacegroup
            sg = self.spacegroup if self.spacegroup else "P1"
            
            self._centric_flags = is_centric_from_hkl(self.hkl, sg)
        
        # Return full-size centric flags (no filtering)
        return self._centric_flags
