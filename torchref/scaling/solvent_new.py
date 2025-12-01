"""
A class for modelling solvent contribution to structure factors.
"""

import torch
import torch.nn as nn
from torchref.math_functions.math_torch import find_relevant_voxels, ifft, \
    extract_structure_factor_from_grid, get_scattering_vectors, \
    add_to_phenix_mask, excise_angstrom_radius_around_coord, hash_tensors

from torchref.utils.utils import TensorDict
from torchref.utils.debug_utils import DebugMixin

class SolventModel(DebugMixin, nn.Module):
    """
    SolventModel to compute solvent contribution to structure factors using Phenix-like approach.
    
    Supports two initialization patterns:
    
    1. Empty initialization (for state_dict loading):
        >>> solvent = SolventModel()  # Creates empty shell
        >>> solvent.load_state_dict(torch.load('solvent.pt'))
    
    2. Full initialization with model:
        >>> solvent = SolventModel(model, k_solvent=0.35, b_solvent=46.0)
    """
    
    def __init__(self, model=None, radius=1.1, k_solvent=1.1, b_solvent=50.0, erosion_radius=0.9, 
                 transition=None, optimize_phase=True, initial_phase_offset=0.0, verbose=1, 
                 float_type=torch.float32, device=torch.device('cpu')):
        """
        Initialize SolventModel.
        
        If model is provided, fully initializes the solvent model.
        If not provided (empty init), creates a shell ready for load_state_dict().

        Args:
            model (ModelFT): The atomic model used for structure factor calculations (optional for empty init)
            radius (float): Probe radius in Angstroms for dilation (default: 1.1 Å, water radius)
            k_solvent (float): Solvent scattering scale factor
            b_solvent (float): Solvent B-factor
            erosion_radius (float): Radius in Angstroms for erosion step (default: 0.9 Å)
            transition (float): Gaussian smoothing sigma for mask edges (default: radius/4 in voxels)
            optimize_phase (bool): Whether to optimize phase offset parameter (default: True)
            initial_phase_offset (float): Initial phase offset in radians (default: 0.0)
            verbose (int): Verbosity level
        """
        super(SolventModel, self).__init__()
        self.device = device
        self.verbose = verbose
        self.float_type = float_type
        self.solvent_radius = radius
        self.erosion_radius = erosion_radius
        self.optimize_phase = optimize_phase
        self._cache = TensorDict()
        
        # Empty initialization
        if model is None:
            self.model = None
            self.max_radius_angstrom = None
            self.transition = transition
            # Register parameters with default values (will be overwritten by load_state_dict)
            self.log_k_solvent = nn.Parameter(torch.log(torch.tensor(k_solvent, dtype=self.float_type, device=self.device)))
            self.b_solvent = nn.Parameter(torch.tensor(b_solvent, dtype=self.float_type, device=self.device))
            if self.optimize_phase:
                self.phase_offset = nn.Parameter(torch.tensor(initial_phase_offset, dtype=self.float_type, device=self.device))
            else:
                self.register_buffer("phase_offset", torch.tensor(0.0, dtype=self.float_type, device=self.device))
            return
        
        # Full initialization with model
        self.model = model
        self.model.get_vdw_radii()  # Ensure VdW radii are available
        assert self.model, 'Model is not initialized'
        if model.real_space_grid == None:
            model.setup_grid()
        
        # Phenix-style parameters
        self.solvent_radius = radius  # For dilation (accessible surface)
        self.erosion_radius = erosion_radius  # For erosion (contact surface)
        
        # For find_relevant_voxels: need to search far enough to capture accessible surface
        # Maximum possible distance is max(VdW) + solvent_radius
        self.max_radius_angstrom = self.model.get_vdw_radii().max() + radius
        
        if not isinstance(k_solvent, torch.Tensor):
            k_solvent = torch.tensor(k_solvent, dtype=self.float_type,device=self.device)
        else:
            k_solvent = k_solvent.to(dtype=self.float_type,device=self.device)
        if not isinstance(b_solvent, torch.Tensor):
            b_solvent = torch.tensor(b_solvent, dtype=self.float_type,device=self.device)
        else:
            b_solvent = b_solvent.to(dtype=self.float_type,device=self.device)
        self.log_k_solvent = nn.Parameter(torch.log(k_solvent))
        self.b_solvent = nn.Parameter(b_solvent)
        
        # Transition width for Gaussian smoothing (in voxels)
        if transition is not None:
            self.transition = transition
        else:
            # Default: use a fraction of solvent_radius converted to voxels
            self.transition = self.model.get_radius(radius) / 4.0
        
        # Phase offset parameter to align solvent phases with protein phases
        # This is critical because FFT of a mask gives arbitrary phases
        self.optimize_phase = optimize_phase
        if self.optimize_phase:
            self.phase_offset = nn.Parameter(torch.tensor(initial_phase_offset, dtype=self.float_type,device=self.device))
        else:
            self.register_buffer("phase_offset", torch.tensor(0.0, dtype=self.float_type,device=self.device))
        self._cache = TensorDict()
    
    def cuda(self, device=None):
        """Move solvent model to CUDA and update device attribute."""
        super().cuda(device)
        self.device = torch.device('cuda') if device is None else torch.device(device)
        if self.verbose > 1:
            print(f"SolventModel moved to device: {self.device}")
        return self
    
    def cpu(self):
        """Move solvent model to CPU and update device attribute."""
        super().cpu()
        self.device = torch.device('cpu')
        if self.verbose > 1:
            print(f"SolventModel moved to cpu")
        return self

    def get_solvent_mask(self):
        """
        Generate solvent mask following Phenix approach with three-step process.
        FULLY VECTORIZED - no loops over atoms or voxels.
        
        Step 1: Create three-valued mask (dilation) - VECTORIZED
            - 0: Inside atomic VdW radii (protein core)
            - -1: Between VdW and VdW+solvent_radius (accessible surface boundary)
            - 1: Beyond VdW+solvent_radius (bulk solvent)
        
        Step 2: Apply symmetry to complete mask with all symmetry mates
        
        Step 3: Erosion - smooth boundary using 3D convolution - VECTORIZED
            - For each boundary point (-1), check if any neighbor within 
              shrink_truncation_radius is solvent (1)
            - If yes: point becomes solvent (1)
            - If no: point becomes protein (0)
        Returns:
            solvent_mask: Boolean mask where True = solvent
        """
        if self.verbose > 1:
            print("\n=== Phenix-Style Bulk Solvent Mask Calculation (VECTORIZED) ===")
            print(f"Solvent radius (dilation): {self.solvent_radius:.2f} Å")
            print(f"Shrink truncation radius (erosion): {self.erosion_radius:.2f} Å")
        
        # Step 1: Create three-valued mask via vectorized dilation
        xyz = self.model.xyz()  # Shape: (N_atoms, 3)
        vdw_radii = self.model.get_vdw_radii()  # Shape: (N_atoms,)
        self.real_space_grid = self.model.real_space_grid
        with torch.no_grad():
            if self.verbose > 2:
                print(f"Processing {xyz.shape[0]} atoms (vectorized)...")
            
            # Find relevant voxels for all atoms (search out to max radius)
            voxel_pos, voxel_coords = find_relevant_voxels(
                self.real_space_grid, xyz, self.max_radius_angstrom,
                inv_frac_matrix=self.model.inv_fractional_matrix
            )
            # Call vectorized function to create three-valued mask
            protein_mask, boundary_mask = add_to_phenix_mask(
                surrounding_coords=voxel_pos,
                voxel_indices=voxel_coords,
                xyz=xyz,
                vdw_radii=vdw_radii,
                solvent_radius=self.solvent_radius,
                inv_frac_matrix=self.model.inv_fractional_matrix,
                frac_matrix=self.model.fractional_matrix,
                grid_shape=self.real_space_grid.shape[:-1],
                device=self.model.device
            )
            
            # Step 2: Apply symmetry operations
            protein_mask = torch.round(self.model.map_symmetry(protein_mask.to(self.model.dtype_float))).to(torch.bool)
            boundary_mask = torch.round(self.model.map_symmetry(boundary_mask.to(self.model.dtype_float))).to(torch.bool)
            boundary_mask = boundary_mask & (~protein_mask)  # Ensure no overlap
            definitely_solvent = ~(protein_mask | boundary_mask)
            if self.verbose > 2:
                n_protein_voxels = torch.sum(protein_mask).item()
                n_boundary_voxels = torch.sum(boundary_mask).item()
                n_solvent_voxels = torch.sum(definitely_solvent).item()
                total_voxels = protein_mask.numel()
                print(f"After symmetry:")
                print(f"  Protein voxels: {n_protein_voxels} / {total_voxels} ({100.0 * n_protein_voxels / total_voxels:.2f}%)")
                print(f"  Boundary voxels: {n_boundary_voxels} / {total_voxels} ({100.0 * n_boundary_voxels / total_voxels:.2f}%)")
                print(f"  Definitely solvent voxels: {n_solvent_voxels} / {total_voxels} ({100.0 * n_solvent_voxels / total_voxels:.2f}%)")

            # Step 3: Erosion via vectorized approach
            boundary_voxel_coords = torch.argwhere(boundary_mask)

            # this can be memory intensive - process in chunks if needed
            target_chunk_size = 1000000  # 1 million voxels at a time
            n_boundary_voxels = boundary_voxel_coords.shape[0]
            chunks = (n_boundary_voxels + target_chunk_size - 1) // target_chunk_size
            chunksize = (n_boundary_voxels + chunks - 1) // chunks
            to_flip = []
            for chunk in range(chunks):
                start = chunk * chunksize
                end = min((chunk + 1) * chunksize, n_boundary_voxels)
                boundary_voxel_coords_chunk = boundary_voxel_coords[start:end]
                roi_coords = excise_angstrom_radius_around_coord(self.real_space_grid, boundary_voxel_coords_chunk, radius_angstrom=self.erosion_radius)
                if self.verbose > 2: print('memory footprint roi', roi_coords.element_size() * roi_coords.nelement() / (1024**2), 'MB')
                roi_definitely_solvent = definitely_solvent[roi_coords[...,0], roi_coords[...,1], roi_coords[...,2]]
                really_should_be_solvent = torch.any(roi_definitely_solvent, dim=1)
                to_flip.append(really_should_be_solvent)
            really_should_be_solvent = torch.cat(to_flip, dim=0)
            voxels_to_flip_sol = boundary_voxel_coords[really_should_be_solvent]
            if self.verbose > 2:
                print(f"Eroding {voxels_to_flip_sol.shape[0]} boundary voxels to solvent...")
            protein_with_boundary = protein_mask | boundary_mask
            protein_with_boundary[voxels_to_flip_sol[:,0], voxels_to_flip_sol[:,1], voxels_to_flip_sol[:,2]] = False
            solvent_mask = ~protein_with_boundary

            self.register_buffer("protein_mask", protein_with_boundary)
            self.register_buffer("solvent_mask", solvent_mask)

            if self.verbose > 1:
                n_solvent_voxels = torch.sum(self.solvent_mask).item()
                total_voxels = self.solvent_mask.numel()
                print(f"Total solvent voxels: {n_solvent_voxels} / {total_voxels} ({100.0 * n_solvent_voxels / total_voxels:.2f}%)")
        return self.solvent_mask

    def update_solvent(self):
        self.get_solvent_mask()
        self.smooth_solvent_mask()        

    def smooth_solvent_mask(self):
        if not hasattr(self, 'solvent_mask'):
            raise ValueError('Solvent mask not computed. Call get_solvent_mask() first.')
        import torch.nn.functional as F
        
        # Convert mask to float for smoothing and ensure it's on the same device
        mask_float = self.solvent_mask.to(dtype=self.log_k_solvent.dtype)
        
        # Smooth the mask using 3D Gaussian convolution
        # This creates soft edges at protein-solvent boundary
        sigma = self.transition 

        # Create 3D Gaussian kernel
        kernel_size = int(4 * sigma + 1)  # Ensure odd size
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Generate 1D Gaussian
        x = torch.arange(kernel_size, dtype=self.log_k_solvent.dtype, device=self.device)
        x = x - kernel_size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        # Create 3D kernel as outer product
        kernel_3d = gauss_1d.view(-1, 1, 1) * gauss_1d.view(1, -1, 1) * gauss_1d.view(1, 1, -1)
        kernel_3d = kernel_3d / kernel_3d.sum()
        
        # Reshape for conv3d: (out_channels, in_channels, D, H, W)
        kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0)
        
        # Reshape mask for conv3d: (batch, channels, D, H, W)
        mask_4d = mask_float.unsqueeze(0).unsqueeze(0)
        
        # Apply circular padding to handle periodic boundaries
        pad = kernel_size // 2
        mask_padded = F.pad(mask_4d, (pad, pad, pad, pad, pad, pad), mode='circular')
        
        # Smooth using 3D convolution
        mask_smoothed = F.conv3d(mask_padded, kernel_3d, padding=0)
        
        # Remove batch and channel dimensions
        mask_smoothed = mask_smoothed.squeeze(0).squeeze(0)

        self.register_buffer("mask_smoothed", mask_smoothed)
        return self.mask_smoothed

    def get_rec_solvent(self, hkl):
        """
        Compute solvent structure factors.
        
        Uses the standard crystallographic approach: compute SFs from the solvent mask.
        The mask represents regions where bulk solvent scattering occurs.
        
        Args:
            hkl: Miller indices
            F_obs: Observed structure factor amplitudes (optional, for future difference-map approach)
            F_calc: Calculated structure factors from protein (optional, for future difference-map approach)
        """

        assert hasattr(self, 'mask_smoothed'), 'Smoothed solvent mask not computed. Call smooth_solvent_mask() first.'

        return extract_structure_factor_from_grid(ifft(self.mask_smoothed), hkl).detach()

    def forward(self, hkl, update_fsol=False, F_protein=None):
        """
        Compute solvent contribution to structure factors at given HKL.
        
        This method is differentiable with respect to k_solvent, b_solvent, and phase_offset parameters.
        
        The solvent model:
        1. Takes the binary solvent mask
        2. Smooths it with Gaussian filter (σ=1.5 voxels) to create soft edges
        3. Computes structure factors via FFT
        4. Applies B-factor damping: exp(-B * s²) where s = sin(θ)/λ
        5. If optimize_phase=True and F_protein provided: blends mask phases with protein phases
           phase_offset controls the blend: 0=use mask phases, ±π=use protein phases
        6. Scales by k_solvent
        
        Args:
            hkl: Miller indices, shape (N, 3)
            F_protein: Protein structure factors (optional), used for phase blending
            
        Returns:
            torch.Tensor: Complex solvent structure factors, shape (N,)
        """
        
        if not update_fsol:
            hkl_hash = hash_tensors([hkl])
            if hkl_hash in self._cache:
                f_sol = self._cache[hkl_hash]
            else:
                f_sol = self.get_rec_solvent(hkl)
                self._cache[hkl_hash] = f_sol
        else:
            f_sol = self.get_rec_solvent(hkl)
            self._cache[hkl_hash] = f_sol


        # Calculate scattering vector magnitude: s = sin(θ)/λ
        # Note: get_scattering_vectors returns h* = (h·a*, k·b*, l·c*)
        # For the Debye-Waller factor, we need s = |h*|/2 = sin(θ)/λ
        scattering_vectors = get_scattering_vectors(hkl, self.model.cell, recB=self.model.recB)
        s = torch.norm(scattering_vectors, dim=1) / 2.0  # This is sin(θ)/λ
        s_squared = s ** 2  # Now s² is correct for B-factor formula
        
        # Apply B-factor damping: exp(-B * s²)
        # The Debye-Waller factor for isotropic displacement
        b_solvent = self.b_solvent
        k_solvent = torch.exp(self.log_k_solvent)
        exp = -b_solvent * s_squared
        b_factor_term = torch.exp(exp.clamp(max=50.0))  # Clamp to avoid overflow
        
        # Phase handling
        if self.optimize_phase and F_protein is not None:
            # Blend between mask phases and protein phases
            # phase_offset = 0: use mask phases entirely
            # phase_offset = π: invert relative to protein (Babinet's principle)
            # This allows the optimizer to find the best phase relationship
            f_mask_amp = torch.abs(f_sol)
            mask_phases = torch.angle(f_sol)
            protein_phases = torch.angle(F_protein)
            
            # Interpolate phases using phase_offset as a blending parameter
            # cos(phase_offset) = 1: use mask phases
            # cos(phase_offset) = -1: use inverted protein phases
            blend_factor = torch.cos(self.phase_offset)
            blended_phase = mask_phases * (1 + blend_factor) / 2 + (protein_phases + torch.pi) * (1 - blend_factor) / 2
            
            phase_adjusted_f_sol = f_mask_amp * torch.exp(1j * blended_phase)
        elif self.optimize_phase:
            # Apply global phase offset
            phase_adjusted_f_sol = f_sol * torch.exp(1j * self.phase_offset)
        else:
            # No phase adjustment - use mask phases as-is4
            phase_adjusted_f_sol = f_sol
        
        # Scale by k_solvent and apply B-factor
        f_solvent = k_solvent * phase_adjusted_f_sol * b_factor_term

        assert torch.isfinite(f_solvent).all(), "Non-finite values in solvent structure factors"
        return f_solvent

    