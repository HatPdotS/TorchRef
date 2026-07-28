"""
A class for modelling solvent contribution to structure factors.
"""

import torch
import torch.nn as nn

from torchref.base import (
    extract_structure_factor_from_grid,
    get_scattering_vectors,
    ifft,
)
from torchref.base.electron_density.main import _get_radius_offsets
from torchref.config import get_float_dtype
from torchref.utils.debug_utils import DebugMixin
from torchref.utils.device_mixin import DeviceMixin
from torchref.utils.device_resolution import resolve_device
from torchref.utils.utils import ModuleReference, TensorDict


class SolventModel(DeviceMixin, DebugMixin, nn.Module):
    """
    SolventModel to compute solvent contribution to structure factors using Phenix-like approach.

    Supports two initialization patterns:

    1. Empty initialization (for state_dict loading)::

        solvent = SolventModel()  # Creates empty shell
        solvent.load_state_dict(torch.load('solvent.pt'))

    2. Full initialization with model::

        # NOTE: 0.35 / 46.0 are the production values that ``Scaler``
        # injects; the bare-constructor defaults are k_solvent=1.1,
        # b_solvent=50.0.
        solvent = SolventModel(model, k_solvent=0.35, b_solvent=46.0)

    Attributes
    ----------
    model : ModelFT or None
        The atomic model for structure factor calculations.
    device : torch.device
        Device for tensor operations.
    verbose : int
        Verbosity level.
    float_type : torch.dtype
        Floating point data type. Defaults to the configured float dtype
        via ``get_float_dtype()`` (e.g. float64 under a float64 config),
        not a hard-wired ``torch.float32``.
    solvent_radius : float
        Probe radius in Angstroms for dilation.
    erosion_radius : float
        Radius in Angstroms for erosion step.
    optimize_phase : bool
        Whether to optimize phase offset parameter.
    log_k_solvent : torch.nn.Parameter
        Log of solvent scattering scale factor.
    b_solvent : torch.nn.Parameter
        Solvent B-factor.
    phase_offset : torch.nn.Parameter or buffer
        Phase offset in radians. It is a trainable ``nn.Parameter`` when
        ``optimize_phase=True``, otherwise a registered buffer fixed at 0.0.
    """

    def __init__(
        self,
        model=None,
        radius=1.1,
        k_solvent=1.1,
        b_solvent=50.0,
        erosion_radius=0.9,
        transition=None,
        optimize_phase=True,
        initial_phase_offset=0.0,
        verbose=1,
        float_type=None,
        device=None,
    ):
        """
        Initialize SolventModel.

        If model is provided, fully initializes the solvent model.
        If not provided (empty init), creates a shell ready for load_state_dict().

        Parameters
        ----------
        model : ModelFT, optional
            The atomic model used for structure factor calculations (optional for empty init).
        radius : float, default 1.1
            Probe radius in Angstroms for dilation (water radius).
        k_solvent : float, default 1.1
            Solvent scattering scale factor.
        b_solvent : float, default 50.0
            Solvent B-factor.
        erosion_radius : float, default 0.9
            Radius in Angstroms for erosion step.
        transition : float, optional
            Gaussian smoothing sigma for mask edges (default: radius/4 in voxels). Avoids ringing artifacts.
        optimize_phase : bool, default True
            Whether to optimize phase offset parameter.
        initial_phase_offset : float, default 0.0
            Initial phase offset in radians.
        verbose : int, default 1
            Verbosity level.
        float_type : torch.dtype, optional
            Floating point data type. If ``None`` (default), resolved at
            runtime to the configured float dtype via ``get_float_dtype()``
            (e.g. float64 under a float64 config), not a hard-wired
            ``torch.float32``.
        device : torch.device, default: configured device.current
            Device for tensor operations.
        """
        super(SolventModel, self).__init__()
        if float_type is None:
            float_type = get_float_dtype()
        # Follow the model when no device is given. ``realspace.py`` constructs
        # a SolventModel with only a model, so falling back to the global
        # default here put the solvent grids on a different device than the
        # structure they are computed from.
        device = resolve_device(model, device=device)
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
            self.log_k_solvent = nn.Parameter(
                torch.log(
                    torch.tensor(k_solvent, dtype=self.float_type, device=self.device)
                )
            )
            self.b_solvent = nn.Parameter(
                torch.tensor(b_solvent, dtype=self.float_type, device=self.device)
            )
            if self.optimize_phase:
                self.phase_offset = nn.Parameter(
                    torch.tensor(
                        initial_phase_offset, dtype=self.float_type, device=self.device
                    )
                )
            else:
                self.register_buffer(
                    "phase_offset",
                    torch.tensor(0.0, dtype=self.float_type, device=self.device),
                )
            return

        # Full initialization with model
        self.model = ModuleReference(model)  # Store reference to model
        self.model.get_vdw_radii()  # Ensure VdW radii are available
        assert self.model, "Model is not initialized"
        if model.real_space_grid == None:
            model.setup_grid()

        # Phenix-style parameters
        self.solvent_radius = radius  # For dilation (accessible surface)
        self.erosion_radius = erosion_radius  # For erosion (contact surface)

        # For find_relevant_voxels: need to search far enough to capture accessible surface
        # Maximum possible distance is max(VdW) + solvent_radius
        self.max_radius_angstrom = self.model.get_vdw_radii().max() + radius

        if not isinstance(k_solvent, torch.Tensor):
            k_solvent = torch.tensor(
                k_solvent, dtype=self.float_type, device=self.device
            )
        else:
            k_solvent = k_solvent.to(dtype=self.float_type, device=self.device)
        if not isinstance(b_solvent, torch.Tensor):
            b_solvent = torch.tensor(
                b_solvent, dtype=self.float_type, device=self.device
            )
        else:
            b_solvent = b_solvent.to(dtype=self.float_type, device=self.device)
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
            self.phase_offset = nn.Parameter(
                torch.tensor(
                    initial_phase_offset, dtype=self.float_type, device=self.device
                )
            )
        else:
            self.register_buffer(
                "phase_offset",
                torch.tensor(0.0, dtype=self.float_type, device=self.device),
            )
        self._cache = TensorDict()

    def get_solvent_mask(self):
        """
        Generate solvent mask following Phenix's three-step process.

        Step 1 (dilation): classify voxels around each atom as protein
            (inside VdW), boundary (between VdW and VdW+solvent_radius), or
            bulk solvent (further out). Built in chunks over atoms so peak
            memory is O(atom_chunk_size × N_box_voxels) rather than
            O(N_atoms × N_box_voxels) — critical because for typical
            macromolecule + grid combinations the dense form is multi-GB.

        Step 2 (symmetry expansion): transform the sparse ASU protein /
            boundary voxel indices through each symop and scatter into the
            P1 grid masks.

        Step 3 (erosion): a boundary voxel becomes solvent if any voxel
            within ``erosion_radius`` of it is bulk solvent. Implemented as
            a single F.conv3d with a precomputed spherical kernel and
            circular padding — replaces the previous Python-loop +
            per-voxel-neighbourhood expansion that itself ran out of memory
            on chunks of 10^6 boundary voxels.

        Returns
        -------
        torch.Tensor
            Solvent mask (boolean) where True = solvent.
        """
        import torch.nn.functional as F

        if self.verbose > 1:
            print("\n=== Phenix-Style Bulk Solvent Mask Calculation ===")
            print(f"Solvent radius (dilation): {self.solvent_radius:.2f} Å")
            print(f"Shrink truncation radius (erosion): {self.erosion_radius:.2f} Å")

        xyz = self.model.xyz()  # (N_atoms, 3)
        vdw_radii = self.model.get_vdw_radii()  # (N_atoms,)
        self.real_space_grid = self.model.real_space_grid
        inv_frac = self.model.inv_fractional_matrix
        frac = self.model.fractional_matrix

        with torch.no_grad():
            spacegroup = self.model.fft.spacegroup
            n_ops = spacegroup.n_ops
            grid_shape = self.real_space_grid.shape[:-1]
            device = self.model.device
            n_atoms = xyz.shape[0]

            # --- Step 1: dilation, chunked over atoms ---
            # Plain-scatter sphere splat (same pattern as the variable-radius
            # CPU splat):
            #   - cached spherical voxel offsets via _get_radius_offsets
            #     (avoids rebuilding the meshgrid each call)
            #   - direct fractional voxel positions from integer indices
            #     (no real_space_grid gather)
            #   - PBC via `diff_frac - round(diff_frac)` (no Cartesian
            #     round-trip via two matmuls)
            #   - r² via metric-tensor einsum (single fused op)
            # ATOM_CHUNK caps working memory at ~chunk * N_voxels_in_sphere
            # times a handful of (float32, 3)-shape transients. 256 keeps
            # peak in the few-hundred-MB range even on the finest grids;
            # the SF code's 1024 OOMs on tight cgroups for 1DAW because
            # our intermediates are denser per atom.
            ATOM_CHUNK = 256

            voxel_size = self.real_space_grid[3, 3, 3] - self.real_space_grid[2, 2, 2]
            local_offsets = _get_radius_offsets(
                voxel_size, self.max_radius_angstrom, device
            )  # (R, 3) int — sphere voxels relative to atom center

            grid_dims = torch.tensor(grid_shape, dtype=torch.long, device=device)
            grid_shape_float = grid_dims.float()
            inv_grid = 1.0 / grid_shape_float
            G = frac.T @ frac  # metric tensor: r²_cart = diff_frac · G · diff_frac

            xyz_frac = xyz @ inv_frac.T  # (N, 3)
            xyz_frac_wrapped = xyz_frac % 1.0
            center_idx = torch.round(
                xyz_frac_wrapped * grid_shape_float
            ).long()  # (N, 3)

            protein_chunks = []
            boundary_chunks = []

            for s in range(0, n_atoms, ATOM_CHUNK):
                e = min(s + ATOM_CHUNK, n_atoms)

                # Wrapped voxel indices: (C, R, 3) int
                vi = (
                    center_idx[s:e].unsqueeze(1) + local_offsets.unsqueeze(0)
                ) % grid_dims

                # Direct fractional voxel positions (skip real_space_grid gather)
                voxel_frac = vi.float() * inv_grid  # (C, R, 3)

                # PBC fractional diff
                diff_frac = voxel_frac - xyz_frac[s:e].unsqueeze(1)
                diff_frac = diff_frac - torch.round(diff_frac)
                del voxel_frac

                # Squared Cartesian distance via metric tensor
                r_sq = torch.einsum("avi,ij,avj->av", diff_frac, G, diff_frac)
                del diff_frac

                vdw_c = vdw_radii[s:e]
                vdw_sq = (vdw_c**2).unsqueeze(1)
                rcut_sq = ((vdw_c + self.solvent_radius) ** 2).unsqueeze(1)

                is_protein = r_sq < vdw_sq
                is_boundary = (~is_protein) & (r_sq < rcut_sq)
                del r_sq

                voxel_flat = vi.reshape(-1, 3)
                del vi

                p_idx = is_protein.flatten().nonzero(as_tuple=True)[0]
                b_idx = is_boundary.flatten().nonzero(as_tuple=True)[0]
                del is_protein, is_boundary

                protein_chunks.append(voxel_flat[p_idx])
                boundary_chunks.append(voxel_flat[b_idx])

            protein_voxels = (
                torch.cat(protein_chunks, dim=0)
                if protein_chunks
                else torch.empty((0, 3), dtype=torch.long, device=device)
            )
            boundary_voxels = (
                torch.cat(boundary_chunks, dim=0)
                if boundary_chunks
                else torch.empty((0, 3), dtype=torch.long, device=device)
            )
            del protein_chunks, boundary_chunks

            # --- Step 2: symmetry expansion via index transform ---
            protein_mask = torch.zeros(grid_shape, dtype=torch.bool, device=device)
            boundary_mask = torch.zeros(grid_shape, dtype=torch.bool, device=device)

            float_dtype = get_float_dtype()
            for op_idx in range(n_ops):
                if op_idx == 0:
                    p_idx = protein_voxels
                    b_idx = boundary_voxels
                else:
                    R = spacegroup.matrices[op_idx].to(device=device, dtype=float_dtype)
                    t = spacegroup.translations[op_idx].to(
                        device=device, dtype=float_dtype
                    )
                    gd = grid_dims.to(float_dtype)

                    p_frac = protein_voxels.to(float_dtype) / gd
                    p_idx = (torch.round((p_frac @ R.T + t) * gd) % grid_dims).long()
                    del p_frac

                    b_frac = boundary_voxels.to(float_dtype) / gd
                    b_idx = (torch.round((b_frac @ R.T + t) * gd) % grid_dims).long()
                    del b_frac

                protein_mask[p_idx[:, 0], p_idx[:, 1], p_idx[:, 2]] = True
                boundary_mask[b_idx[:, 0], b_idx[:, 1], b_idx[:, 2]] = True

            boundary_mask = boundary_mask & (~protein_mask)
            definitely_solvent = ~(protein_mask | boundary_mask)

            if self.verbose > 2:
                total_voxels = protein_mask.numel()
                print(
                    f"After symmetry: protein={protein_mask.sum().item()} "
                    f"boundary={boundary_mask.sum().item()} "
                    f"solvent={definitely_solvent.sum().item()} / {total_voxels}"
                )

            # --- Step 3: erosion ---
            # Semantics: a boundary voxel becomes solvent iff any voxel
            # within `erosion_radius` of it is bulk solvent. Equivalently,
            # dilate `definitely_solvent` by the spherical structuring
            # element, then intersect with `boundary_mask`.
            #
            # Two paths, dispatched on device — the underlying op picks
            # extremely different cost profiles on CPU vs GPU:
            #   - CPU: roll-OR over the ~123 cached sphere offsets is
            #     memory-bandwidth-bound and ~5x faster than F.conv3d's
            #     MKLDNN path on a small (~9³) kernel.
            #   - GPU: F.conv3d fuses to a single launch and beats 123
            #     separate roll launches (each its own kernel) by ~2x.
            # Both produce the exact same mask.
            if torch.device(device).type == "cuda":
                half_k = int(torch.ceil(self.erosion_radius / voxel_size.min()).item())
                K = 2 * half_k + 1
                offs = torch.arange(
                    -half_k, half_k + 1, device=device, dtype=voxel_size.dtype
                )
                dx = offs.view(-1, 1, 1) * voxel_size[0]
                dy = offs.view(1, -1, 1) * voxel_size[1]
                dz = offs.view(1, 1, -1) * voxel_size[2]
                kernel = (
                    ((dx * dx + dy * dy + dz * dz) <= self.erosion_radius**2)
                    .to(self.log_k_solvent.dtype)
                    .view(1, 1, K, K, K)
                )
                solv_float = definitely_solvent.to(self.log_k_solvent.dtype).view(
                    1, 1, *grid_shape
                )
                solv_padded = F.pad(solv_float, (half_k,) * 6, mode="circular")
                neighbour_count = F.conv3d(solv_padded, kernel)
                dilated_solvent = neighbour_count.squeeze(0).squeeze(0) > 0.5
                del solv_float, solv_padded, neighbour_count
            else:
                sphere_offsets = _get_radius_offsets(
                    voxel_size, self.erosion_radius, device
                )  # (R, 3) int
                dilated_solvent = torch.zeros_like(definitely_solvent)
                for i in range(sphere_offsets.shape[0]):
                    dz_, dy_, dx_ = sphere_offsets[i].tolist()
                    dilated_solvent |= torch.roll(
                        definitely_solvent,
                        shifts=(dz_, dy_, dx_),
                        dims=(0, 1, 2),
                    )

            voxels_to_flip = boundary_mask & dilated_solvent
            protein_with_boundary = (protein_mask | boundary_mask) & (~voxels_to_flip)
            solvent_mask = ~protein_with_boundary

            self.register_buffer("protein_mask", protein_with_boundary)
            self.register_buffer("solvent_mask", solvent_mask)

            if self.verbose > 1:
                total_voxels = self.solvent_mask.numel()
                n_solv = self.solvent_mask.sum().item()
                print(
                    f"Total solvent voxels: {n_solv} / {total_voxels} "
                    f"({100.0 * n_solv / total_voxels:.2f}%)"
                )

        assert torch.isfinite(
            self.solvent_mask.float()
        ).all(), "Non-finite values in solvent mask"
        return self.solvent_mask

    def update_solvent(self):
        self.get_solvent_mask()
        self.smooth_solvent_mask()

    def smooth_solvent_mask(self):
        if not hasattr(self, "solvent_mask"):
            raise ValueError(
                "Solvent mask not computed. Call get_solvent_mask() first."
            )
        import torch.nn.functional as F

        mask_float = self.solvent_mask.to(dtype=self.log_k_solvent.dtype)
        sigma = self.transition
        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = torch.arange(
            kernel_size, dtype=self.log_k_solvent.dtype, device=self.device
        )
        x = x - kernel_size // 2
        gauss_1d = torch.exp(-(x**2) / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()

        pad = kernel_size // 2
        mask = mask_float.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        # Separable Gaussian: three sequential 1D conv3d passes (one per
        # spatial axis). For a separable kernel, this is mathematically
        # identical to the full 3D outer-product conv but costs O(3·K·V)
        # instead of O(K³·V). Pad once on all three axes with the circular
        # mode required by periodic crystallographic boundaries; each
        # successive 1D conv shrinks only its own axis back to the original
        # size.
        mask = F.pad(mask, (pad, pad, pad, pad, pad, pad), mode="circular")
        mask = F.conv3d(mask, gauss_1d.view(1, 1, 1, 1, kernel_size))
        mask = F.conv3d(mask, gauss_1d.view(1, 1, 1, kernel_size, 1))
        mask = F.conv3d(mask, gauss_1d.view(1, 1, kernel_size, 1, 1))

        mask_smoothed = mask.squeeze(0).squeeze(0)
        self.register_buffer("mask_smoothed", mask_smoothed)

        assert torch.isfinite(
            self.mask_smoothed
        ).all(), "Non-finite values in solvent mask"
        return self.mask_smoothed

    def get_rec_solvent(self, hkl):
        """
        Compute solvent structure factors.

        Uses the standard crystallographic approach: compute SFs from the solvent mask.
        The mask represents regions where bulk solvent scattering occurs.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices.

        Returns
        -------
        torch.Tensor
            Complex solvent structure factors.
        """

        assert hasattr(
            self, "mask_smoothed"
        ), "Smoothed solvent mask not computed. Call smooth_solvent_mask() first."
        fsol = extract_structure_factor_from_grid(
            ifft(self.mask_smoothed, self.model.cell.volume), hkl
        ).detach()
        assert torch.isfinite(
            fsol
        ).all(), "Non-finite values in solvent structure factors"
        return fsol

    def forward(self, hkl, update_fsol=False, F_protein=None):
        """
        Compute solvent contribution to structure factors at given HKL.

        This method is differentiable with respect to k_solvent, b_solvent,
        and phase_offset parameters.

        The solvent model:

        1. Retrieves the solvent structure factors ``f_sol`` (the FFT of the
           *smoothed* mask) from a per-hkl cache keyed on the hkl fingerprint;
           it is computed only when missing or when ``update_fsol=True``. The
           smoothing itself uses a Gaussian filter with σ = ``self.transition``
           voxels (structure/grid dependent; default ``radius/4``), applied
           upstream when the mask is built.
        2. Applies B-factor damping: exp(-B * s^2) where s = sin(θ)/λ
        3. If optimize_phase=True and F_protein provided: blends mask phases with protein phases
           phase_offset controls the blend: 0=use mask phases, ±π=use protein phases
        4. Scales by k_solvent

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices, shape (N, 3).
        update_fsol : bool, default False
            When ``True``, forces recomputation of the cached solvent
            structure factors for this hkl (FFT of the smoothed mask) and
            refreshes the per-hkl cache entry. When ``False`` (default), a
            cached entry keyed on the hkl fingerprint is reused if present.
        F_protein : torch.Tensor, optional
            Protein structure factors, used for phase blending.

        Returns
        -------
        torch.Tensor
            Complex solvent structure factors, shape (N,).
        """

        # Lightweight fingerprint: (data_ptr, version, numel) — avoids SHA-1
        hkl_key = (hkl.data_ptr(), hkl._version, hkl.numel())

        if not update_fsol and hkl_key in self._cache:
            f_sol = self._cache[hkl_key]
        else:
            f_sol = self.get_rec_solvent(hkl)
            self._cache[hkl_key] = f_sol

        # Calculate scattering vector magnitude: s = sin(θ)/λ
        # Note: get_scattering_vectors returns h* = (h·a*, k·b*, l·c*)
        # For the Debye-Waller factor, we need s = |h*|/2 = sin(θ)/λ
        scattering_vectors = get_scattering_vectors(
            hkl, self.model.cell, recB=self.model.recB
        )
        s = torch.norm(scattering_vectors, dim=1) / 2.0  # This is sin(θ)/λ
        s_squared = s**2  # Now s² is correct for B-factor formula

        # Apply B-factor damping: exp(-B * s²)
        # The Debye-Waller factor for isotropic displacement
        b_solvent = self.b_solvent
        k_solvent = torch.exp(self.log_k_solvent.clamp(min=-10.0, max=10.0))
        exp = -b_solvent.clamp(min=-500.0, max=500.0) * s_squared
        b_factor_term = torch.exp(
            exp.clamp(min=-10.0, max=10.0)
        )  # Clamp to avoid overflow

        # Phase handling
        if self.optimize_phase and F_protein is not None:
            f_mask_amp = torch.abs(f_sol)
            mask_phases = torch.angle(f_sol)
            protein_phases = torch.angle(F_protein)

            # Interpolate phases using phase_offset as a blending parameter
            # cos(phase_offset) = 1: use mask phases
            # cos(phase_offset) = -1: use inverted protein phases
            blend_factor = torch.cos(self.phase_offset)
            blended_phase = (
                mask_phases * (1 + blend_factor) / 2
                + (protein_phases + torch.pi) * (1 - blend_factor) / 2
            )

            phase_adjusted_f_sol = f_mask_amp * torch.exp(1j * blended_phase)
        elif self.optimize_phase:
            # Apply global phase offset
            phase_adjusted_f_sol = f_sol * torch.exp(1j * self.phase_offset)
        else:
            # No phase adjustment - use mask phases as-is
            phase_adjusted_f_sol = f_sol

        # Scale by k_solvent and apply B-factor
        f_solvent = k_solvent * phase_adjusted_f_sol * b_factor_term

        assert torch.isfinite(
            f_solvent
        ).all(), "Non-finite values in solvent structure factors"
        return f_solvent

    def parameters(self):
        return [self.log_k_solvent, self.b_solvent] + (
            [self.phase_offset] if self.optimize_phase else []
        )
