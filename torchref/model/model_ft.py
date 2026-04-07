from typing import Optional, Tuple

import gemmi
import numpy as np
import torch

from torchref.base.fourier import fft, ifft
from torchref.model.sf_fft import SfFFT
from torchref.model.model import Model
from torchref.symmetry import SpaceGroup
from torchref.symmetry.map_symmetry import MapSymmetry
from torchref.config import dtypes
from torchref.utils.caching import CachedForwardMixin


class ModelFT(CachedForwardMixin, Model):
    """
    Model subclass for Fourier Transform-based electron density and structure factor calculations.

    ModelFT extends the base Model class with capabilities for computing electron
    density maps in real space and structure factors via FFT. Uses ITC92
    parametrization for electron density calculations.

    Parameters
    ----------
    max_res : float, optional
        Maximum resolution for grid spacing in Angstroms. Default is 1.0.
    radius_angstrom : float, optional
        Radius in Angstroms for density calculation around each atom. Default is 4.0.
    gridsize : tuple of int, optional
        Explicit grid size (nx, ny, nz). If None, computed from cell and max_res.
    wavelength : float or None, optional
        X-ray wavelength in Angstroms for anomalous scattering correction.
        Default is 1.0 (standard synchrotron, ~12.4 keV). Set to None to
        disable anomalous corrections entirely.
    anomalous_threshold : float, optional
        Significance threshold for anomalous scattering in electrons.
        Atoms with |f'| > threshold or |f''| > threshold will have
        anomalous corrections applied. Default is 0.5.
    *args
        Additional positional arguments passed to parent Model class.
    **kwargs
        Additional keyword arguments passed to parent Model class.

    Attributes
    ----------
    max_res : float
        Maximum resolution for grid spacing.
    radius_angstrom : float
        Radius for density calculation.
    wavelength : float or None
        X-ray wavelength for anomalous scattering corrections.
    anomalous_threshold : float
        Threshold for significant anomalous scattering (electrons).
    gridsize : torch.Tensor
        Grid dimensions (nx, ny, nz).
    real_space_grid : torch.Tensor
        Real-space coordinate grid with shape (nx, ny, nz, 3).
    map : torch.Tensor or None
        Computed electron density map.
    parametrization : dict
        ITC92 parametrization dictionary {element: (A, B, C)}.
    map_symmetry : MapSymmetry
        Symmetry operator for map calculations.

    Examples
    --------
    Empty initialization for state_dict loading::

        model = ModelFT()
        model.load_state_dict(torch.load('model.pt'))

    File-based initialization::

        model = ModelFT(max_res=1.5)
        model.load_pdb('structure.pdb')
    """

    def __init__(
        self,
        *args,
        max_res=1.0,
        radius_angstrom=4.0,
        gridsize: Optional[Tuple[int, int, int]] = None,
        wavelength: Optional[float] = 1.0,
        anomalous_threshold: float = 0.5,
        **kwargs,
    ):
        """
        Initialize an empty ModelFT shell.

        Creates a model shell ready for file loading via load_pdb()/load_cif()
        or state restoration via load_state_dict().

        Parameters
        ----------
        max_res : float, optional
            Maximum resolution for grid spacing in Angstroms. Default is 1.0.
        radius_angstrom : float, optional
            Radius in Angstroms for density calculation. Default is 4.0.
        gridsize : tuple of int, optional
            Explicit grid size tuple (nx, ny, nz). If None, computed automatically.
        wavelength : float or None, optional
            X-ray wavelength in Angstroms for anomalous scattering correction.
            Default is 1.0 (standard synchrotron, ~12.4 keV). Set to None to
            disable anomalous corrections entirely.
        anomalous_threshold : float, optional
            Significance threshold for anomalous scattering in electrons.
            Atoms with |f'| > threshold or |f''| > threshold will have
            anomalous corrections applied. Default is 0.5.
        *args
            Passed to parent Model class.
        **kwargs
            Passed to parent Model class.
        """
        super().__init__(*args, **kwargs)

        # FT-specific configuration
        self.max_res = max_res
        self.radius_angstrom = radius_angstrom
        self._explicit_gridsize = gridsize

        # Anomalous scattering configuration
        self.wavelength = wavelength
        self.anomalous_threshold = anomalous_threshold
        self._anomalous_cache = None  # Will hold (mask, f_prime, f_double_prime)
        self._anomalous_elements_hash = None  # Hash of element list for cache invalidation
        self._fft = None

    @property
    def cell(self):
        """Unit cell object with parameters [a, b, c, alpha, beta, gamma]."""
        return self._cell

    @cell.setter
    def cell(self, value):
        """
        Set the unit cell and initialize FFT if spacegroup is also set.

        Parameters
        ----------
        value : Cell
            The unit cell object to set.
        """
        self._cell = value
        self._maybe_initialize_fft()

    @property
    def spacegroup(self):
        """Space group object."""
        return self._spacegroup

    @spacegroup.setter
    def spacegroup(self, value):
        """
        Set the space group and initialize FFT if cell is also set.

        Parameters
        ----------
        value : SpaceGroup, gemmi.SpaceGroup, str, or int
            The space group to set.
        """
        if value is not None:
            self._spacegroup = SpaceGroup(
                value, dtype=self.dtype_float, device=self.device
            )
        else:
            self._spacegroup = None
        self._maybe_initialize_fft()

    def _maybe_initialize_fft(self):
        """
        Initialize SfFFT module if both cell and spacegroup are set.

        This method is called by the cell and spacegroup setters to ensure
        the SfFFT module is properly configured when both crystallographic
        parameters are available.
        """
        if self._cell is not None and self._spacegroup is not None:
            self._fft = SfFFT(
                cell=self._cell,
                spacegroup=self._spacegroup,
                device=self.device,
                max_res=self.max_res,
                radius_angstrom=self.radius_angstrom,
            )

    def load_pdb(self, filename):
        """
        Load a PDB file and initialize the model with FT-specific setup.

        Parameters
        ----------
        filename : str
            Path to the PDB file.

        Returns
        -------
        ModelFT
            Self, for method chaining.
        """
        super().load_pdb(filename)
        # FFT is now initialized via cell/spacegroup setters in parent load()
        self.setup_grid()
        return self

    def select(self, selection):
        selection = super().select(selection)
        selection._build_parametrization()
        # FFT is initialized via cell/spacegroup setters in parent select()
        selection.setup_grid()
        return selection

    def load_cif(self, filename):
        """
        Load a CIF file and initialize the model with FT-specific setup.

        Parameters
        ----------
        filename : str
            Path to the CIF/mmCIF file.

        Returns
        -------
        ModelFT
            Self, for method chaining.
        """
        super().load_cif(filename)
        self._build_parametrization()
        # FFT is now initialized via cell/spacegroup setters in parent load()
        self.setup_grid()
        return self

    def setup_gridsize(self, max_res=None):
        """
        Compute optimal grid dimensions.

        Delegates to FFT.compute_grid_size().

        Parameters
        ----------
        max_res : float, optional
            Maximum resolution in Angstroms. If None, uses self.max_res.

        Returns
        -------
        torch.Tensor
            Grid dimensions (nx, ny, nz) as int32 tensor.
        """
        if max_res is not None:
            self.max_res = max_res
            self._fft.max_res = max_res

        if self.verbose > 1:
            print(f"Defining grid size for ={self.max_res} Å")

        # Use Cell's compute_grid_size method
        gridsize = self.cell.compute_grid_size(self.max_res)
        return torch.tensor(gridsize, dtype=dtypes.int, device=self.device)

    def _build_parametrization(self):
        """
        Build ITC92 parametrization for all atoms in the model.

        Delegates to parent Model class which handles the actual parametrization
        building. This method exists for API compatibility.
        """
        # Use parent's implementation
        return super()._build_parametrization()

    # =========================================================================
    # Backward-compatible properties for scattering parameters
    # =========================================================================

    @property
    def A(self) -> torch.Tensor:
        """
        ITC92 A parameters (amplitudes) for all atoms.

        Returns
        -------
        torch.Tensor
            A parameters with shape (n_atoms, 5).
        """
        self._build_parametrization()
        return self._A

    @property
    def B(self) -> torch.Tensor:
        """
        ITC92 B parameters (widths) for all atoms.

        Returns
        -------
        torch.Tensor
            B parameters with shape (n_atoms, 5).
        """
        self._build_parametrization()
        return self._B

    # =========================================================================
    # Backward-compatible properties for FFT grid attributes
    # =========================================================================

    @property
    def gridsize(self) -> Optional[torch.Tensor]:
        """Grid dimensions (nx, ny, nz)."""
        return self._fft.gridsize

    @gridsize.setter
    def gridsize(self, value):
        """Set grid size (for backward compatibility)."""
        self._fft.gridsize = value

    @property
    def real_space_grid(self) -> Optional[torch.Tensor]:
        """Real-space coordinate grid with shape (nx, ny, nz, 3)."""
        return self._fft.real_space_grid

    @real_space_grid.setter
    def real_space_grid(self, value):
        """Set real space grid (for backward compatibility)."""
        self._fft.real_space_grid = value

    @property
    def voxel_size(self) -> Optional[torch.Tensor]:
        """Voxel dimensions."""
        return self._fft.voxel_size

    @voxel_size.setter
    def voxel_size(self, value):
        """Set voxel size (for backward compatibility)."""
        self._fft.voxel_size = value

    @property
    def map_symmetry(self) -> Optional[MapSymmetry]:
        """Symmetry operator for map calculations."""
        return self._fft.map_symmetry

    @map_symmetry.setter
    def map_symmetry(self, value):
        """Set map symmetry (for backward compatibility)."""
        self._fft.map_symmetry = value

    def get_iso(self):
        """
        Get isotropic atoms with their ITC92 parameters.

        Returns
        -------
        xyz : torch.Tensor
            Atomic coordinates with shape (n_atoms, 3).
        adp : torch.Tensor
            Atomic displacement parameters (isotropic) with shape (n_atoms,).
        occupancy : torch.Tensor
            Occupancies with shape (n_atoms,).
        A : torch.Tensor
            ITC92 A parameters (amplitudes) with shape (n_atoms, 5).
        B : torch.Tensor
            ITC92 B parameters (widths) with shape (n_atoms, 5).
        """
        # Get base isotropic data from parent
        xyz, adp, occupancy = super().get_iso()

        # Get scattering parameters from parent
        A, B = self.get_scattering_params_iso()

        return xyz, adp, occupancy, A, B

    def get_aniso(self):
        """
        Get anisotropic atoms with their ITC92 parameters.

        Returns
        -------
        xyz : torch.Tensor
            Atomic coordinates with shape (n_atoms, 3).
        u : torch.Tensor
            Anisotropic U parameters with shape (n_atoms, 6).
        occupancy : torch.Tensor
            Occupancies with shape (n_atoms,).
        A : torch.Tensor
            ITC92 A parameters (amplitudes) with shape (n_atoms, 5).
        B : torch.Tensor
            ITC92 B parameters (widths) with shape (n_atoms, 5).
        """
        # Get base anisotropic data from parent
        xyz, u, occupancy = super().get_aniso()

        # Get scattering parameters from parent
        A, B = self.get_scattering_params_aniso()

        return xyz, u, occupancy, A, B

    def setup_grid(self, max_res=None, gridsize=None):
        """
        Setup real-space grid for electron density calculation.

        Delegates to FFT.setup_grid() using the stored cell and spacegroup.

        Parameters
        ----------
        max_res : float, optional
            Maximum resolution for grid spacing in Angstroms.
            If None, uses self.max_res.
        gridsize : tuple of int, optional
            Explicit grid size (nx, ny, nz). If None, computed automatically
            using Cell.compute_grid_size() and SpaceGroup.suggest_grid_size().
        """
        if max_res is not None:
            self.max_res = max_res
            self._fft.max_res = max_res

        if self.verbose > 1:
            print(f"Setting up grids with max_res={self.max_res} Å")

        # Determine grid size to use
        gridsize_to_use = gridsize or self._explicit_gridsize

        # Delegate to FFT submodule (which now uses stored cell/spacegroup)
        self._fft.setup_grid(
            gridsize=gridsize_to_use,
            max_res=self.max_res,
        )

        if self.verbose > 2:
            print(f"Grid shape: {self._fft.real_space_grid.shape[:-1]}")
            print(f"Voxel size: {self._fft.voxel_size}")

    def get_radius(self, min_radius_Angstrom: float = 4.0):
        """
        Get the radius in voxels used for density calculation around each atom.

        Parameters
        ----------
        min_radius_Angstrom : float, optional
            Minimum radius in Angstroms. Default is 4.0.

        Returns
        -------
        int
            Radius in voxels.
        """
        if not hasattr(self, "real_space_grid") or self.real_space_grid is None:
            self.setup_grid()
        voxel_size = self.real_space_grid[1, 1, 1] - self.real_space_grid[0, 0, 0]
        min_radius = (
            torch.ceil(min_radius_Angstrom / torch.min(voxel_size))
            .to(dtypes.int)
            .item()
        )
        if self.verbose > 1:
            print(
                f"Calculated radius for density calculation: {min_radius} voxels (voxel size: {voxel_size}), this corresponds to at least {min_radius_Angstrom} Å"
            )
        return min_radius

    def build_complete_map(self, radius=None, apply_symmetry=True):
        """
        Build electron density map from all atoms.

        Uses get_iso() and get_aniso() to get atom data and constructs
        the complete electron density map.

        Parameters
        ----------
        radius : int, optional
            Radius in voxels around each atom to compute density.
            If None, uses self.radius.
        apply_symmetry : bool, optional
            If True and space group is not P1, apply symmetry operations
            to the map. Default is True.

        Returns
        -------
        torch.Tensor
            Electron density map with symmetry applied if requested.
        """
        self.map = self.build_initial_map(apply_symmetry=apply_symmetry)

        if self.verbose > 2:
            print(
                f"Density map built. Sum: {self.map.sum():.2f}, Max: {self.map.max():.4f}"
            )
        return self.map

    def build_initial_map(self, apply_symmetry=True):
        """
        Build electron density map from atomic parameters.

        Delegates to FFT.build_density_map() using the model's stored parameters.

        Parameters
        ----------
        apply_symmetry : bool, optional
            If True, apply crystallographic symmetry to the map. Default is True.

        Returns
        -------
        torch.Tensor
            Electron density map with shape (nx, ny, nz).
        """
        if self._fft.real_space_grid is None:
            self.setup_grid()

        if self.verbose > 2:
            print(
                f"Building density map with radius={self.radius_angstrom} angstrom..."
            )

        # Get isotropic atoms
        xyz_iso, adp_iso, occ_iso, A_iso, B_iso = self.get_iso()

        if self.verbose > 3:
            assert torch.all(
                torch.isfinite(A_iso)
            ), "Non-finite values found in A_iso during map building."
            assert torch.all(
                torch.isfinite(B_iso)
            ), "Non-finite values found in B_iso during map building."
            assert torch.all(
                torch.isfinite(xyz_iso)
            ), "Non-finite values found in xyz_iso during map building."
            assert torch.all(
                torch.isfinite(adp_iso)
            ), "Non-finite values found in adp_iso during map building."
            assert torch.all(
                torch.isfinite(occ_iso)
            ), "Non-finite values found in occ_iso during map building."

        # Get anisotropic atoms
        xyz_aniso, u_aniso, occ_aniso, A_aniso, B_aniso = self.get_aniso()

        # Delegate to FFT submodule
        self.map = self._fft.build_density_map(
            xyz_iso=xyz_iso,
            adp_iso=adp_iso,
            occ_iso=occ_iso,
            A_iso=A_iso,
            B_iso=B_iso,
            xyz_aniso=xyz_aniso if len(xyz_aniso) > 0 else None,
            u_aniso=u_aniso if len(xyz_aniso) > 0 else None,
            occ_aniso=occ_aniso if len(xyz_aniso) > 0 else None,
            A_aniso=A_aniso if len(xyz_aniso) > 0 else None,
            B_aniso=B_aniso if len(xyz_aniso) > 0 else None,
            apply_symmetry=apply_symmetry,
        )

        if self.verbose > 3:
            assert torch.all(
                torch.isfinite(self.map)
            ), "Non-finite values found in map."

        return self.map

    def save_map(self, filename):
        """
        Save the electron density map to a CCP4 format file.

        Parameters
        ----------
        filename : str
            Output filename for the map.

        Raises
        ------
        ValueError
            If no map has been computed yet.
        """
        if self.map is None:
            raise ValueError("No map to save. Call build_density_map() first.")

        np_map = self.map.detach().cpu().numpy().astype(np.float32)
        cell = self.cell.tolist()
        if self.verbose > 0:
            print(f"Saving map to {filename}")
            print(f"  Map shape: {self.map.shape}")
            print(f"  Map sum: {self.map.sum():.2f}")
            print(f"  Map range: [{self.map.min():.4f}, {self.map.max():.4f}]")

        map_ccp = gemmi.Ccp4Map()
        map_ccp.grid = gemmi.FloatGrid(
            np_map, gemmi.UnitCell(*cell), SpaceGroup("P1")._gemmi
        )
        map_ccp.setup(0.0)
        map_ccp.update_ccp4_header()
        map_ccp.write_ccp4_map(filename)
        if self.verbose > 0:
            print("Map saved successfully")

    def get_map_statistics(self):
        """Get statistics about the current density map."""
        if self.map is None:
            return None

        stats = {
            "shape": self.map.shape,
            "sum": float(self.map.sum()),
            "mean": float(self.map.mean()),
            "std": float(self.map.std()),
            "min": float(self.map.min()),
            "max": float(self.map.max()),
            "n_positive": int((self.map > 0).sum()),
            "n_negative": int((self.map < 0).sum()),
        }
        return stats

    def rebuild_map(self, radius=None):
        """
        Rebuild the density map from scratch.

        Convenience method that clears and rebuilds everything.

        Parameters
        ----------
        radius : int, optional
            Radius in voxels around each atom. If None, uses self.radius.
            If specified, overrides self.radius.

        Returns
        -------
        torch.Tensor
            Rebuilt electron density map.
        """
        if self.verbose > 1:
            print("Rebuilding density map from scratch...")
        return self.build_density_map(radius=radius)

    def to(self, device=None, dtype=None):
        """
        Move model and FT-specific data to specified device and/or dtype.

        Parameters
        ----------
        device : torch.device or str, optional
            Target device.
        dtype : torch.dtype, optional
            Target data type.

        Returns
        -------
        ModelFT
            Self, for method chaining.
        """
        result = super().to(device=device, dtype=dtype)

        # Explicitly move FFT submodule (ensures cell and spacegroup are moved)
        if self._fft is not None:
            self._fft.to(device=device, dtype=dtype)

        return result

    def update_pdb(self):
        """
        Update PDB with current atomic parameters.
        """
        return super().update_pdb()

    def reset_cache(self):
        """Reset SF cache and all wrapper forward caches."""
        self.reset_forward_cache()
        for module in self.children():
            if hasattr(module, "reset_forward_cache"):
                module.reset_forward_cache()

    def invalidate_cache(self):
        """Alias for ``reset_cache()``."""
        self.reset_cache()

    # =========================================================================
    # Anomalous Scattering Correction Methods
    # =========================================================================

    def _get_anomalous_cache(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Lazily compute and cache anomalous correction data.

        Returns mask, f_prime, f_double_prime for significant atoms.
        Recomputes if element list changes.

        Returns
        -------
        mask : torch.Tensor
            Boolean mask of shape (n_atoms,) - True for atoms needing correction
        f_prime : torch.Tensor
            f' values for significant atoms only (n_significant,)
        f_double_prime : torch.Tensor
            f'' values for significant atoms only (n_significant,)
        """
        from torchref.base.scattering.anomalous_table import (
            get_significant_elements,
            get_anomalous_corrections_by_indices,
        )

        # Get element list from PDB
        element_list = self.pdb["element"].tolist()

        # Hash current element list
        elements_hash = hash(tuple(element_list))

        if (
            self._anomalous_cache is None
            or self._anomalous_elements_hash != elements_hash
        ):
            # Find significant elements at current wavelength
            unique_elements = list(set(element_list))
            significant = get_significant_elements(
                unique_elements, self.wavelength, self.anomalous_threshold
            )

            if self.verbose > 1 and significant:
                print(
                    f"Anomalous scatterers at {self.wavelength:.4f} Å: "
                    f"{list(significant.keys())}"
                )

            # Get corrections for all atoms
            mask, f_prime, f_double_prime = get_anomalous_corrections_by_indices(
                element_list, significant, self.device, self.dtype_float
            )

            # Pre-compute integer indices to avoid boolean indexing GPU sync
            has_anomalous = bool(mask.any().item())
            anomalous_indices = mask.nonzero(as_tuple=True)[0] if has_anomalous else None
            self._anomalous_cache = (mask, f_prime, f_double_prime, has_anomalous, anomalous_indices)
            self._anomalous_elements_hash = elements_hash

        return self._anomalous_cache

    def _apply_anomalous_correction(
        self,
        sf: torch.Tensor,
        hkl: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply anomalous scattering correction to structure factors.

        The correction adds the contribution from anomalous scattering:
        ΔF(h) = Σ_significant (f' + if'') × exp(2πi h·r) × occ

        Only computed for atoms where |f'| > threshold or |f''| > threshold.

        Parameters
        ----------
        sf : torch.Tensor
            Complex structure factors from FFT with shape (n_reflections,)
        hkl : torch.Tensor
            Miller indices with shape (n_reflections, 3)

        Returns
        -------
        torch.Tensor
            Corrected complex structure factors with shape (n_reflections,)
        """
        mask, f_prime, f_double_prime, has_anomalous, anomalous_indices = self._get_anomalous_cache()

        if not has_anomalous:
            return sf  # No significant anomalous scatterers

        # Get fractional coordinates and occupancies for significant atoms only
        # Uses pre-computed integer indices to avoid boolean indexing GPU sync
        xyz_frac = self.xyz_fractional()[anomalous_indices]  # (n_significant, 3)
        occ = self.occupancy()[anomalous_indices]  # (n_significant,)

        # Compute phase factors: exp(2πi h·r)
        # h·r is the dot product of hkl with fractional coordinates
        h_dot_r = torch.matmul(
            hkl.to(dtype=self.dtype_float), xyz_frac.T
        )  # (n_refl, n_significant)
        phase = 2 * torch.pi * h_dot_r

        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        # Anomalous contribution weighted by occupancy
        # f' contributes to both real and imaginary parts
        # f'' contributes with a phase shift (it's the imaginary part of f)
        f_prime_occ = f_prime * occ  # (n_significant,)
        f_double_prime_occ = f_double_prime * occ  # (n_significant,)

        # For each reflection:
        # Real part: Σ [f'·cos(φ) - f''·sin(φ)] × occ
        # Imag part: Σ [f'·sin(φ) + f''·cos(φ)] × occ
        delta_real = torch.sum(
            f_prime_occ * cos_phase - f_double_prime_occ * sin_phase, dim=-1
        )
        delta_imag = torch.sum(
            f_prime_occ * sin_phase + f_double_prime_occ * cos_phase, dim=-1
        )

        return sf + torch.complex(delta_real, delta_imag)

    def get_structure_factor(
        self, hkl: torch.Tensor, recalc=False, apply_anomalous: bool = True
    ) -> torch.Tensor:
        """
        Get structure factors for given hkl reflections.

        Uses ``CachedForwardMixin`` to cache the result and auto-invalidate
        when parameters change or a backward pass propagates through.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices with shape (n_reflections, 3).
        recalc : bool, optional
            If True, forces recalculation bypassing the cache.
            Default is False.
        apply_anomalous : bool, optional
            If True and wavelength is set, apply anomalous scattering
            corrections (f' and f'') for heavy atoms. Default is True.

        Returns
        -------
        torch.Tensor
            Complex structure factors with shape (n_reflections,).

        Notes
        -----
        The complete scattering factor is:
            f(s, λ) = f₀(s) + f'(λ) + i·f''(λ)

        where f₀ is the normal (Thomson) scattering factor computed via FFT,
        and f'/f'' are the wavelength-dependent anomalous corrections.

        Anomalous corrections are only computed for atoms where
        |f'| > anomalous_threshold or |f''| > anomalous_threshold.
        """
        return self(hkl, recalc=recalc, apply_anomalous=apply_anomalous)

    @property
    def fft(self):
        """Access the SfFFT submodule."""
        if self._fft is None:
            self._maybe_initialize_fft()

        return self._fft

    def forward(
        self, hkl, apply_anomalous: bool = True
    ) -> torch.Tensor:
        """
        Compute structure factors for given hkl.

        This is called by the mixin's ``__call__`` which handles caching,
        backward-hook registration, and auto-invalidation.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices with shape (n_reflections, 3).
        apply_anomalous : bool, optional
            If True and wavelength is set, apply anomalous scattering
            corrections (f' and f'') for heavy atoms. Default is True.

        Returns
        -------
        torch.Tensor
            Calculated complex structure factors with shape (n_reflections,).
        """
        sf, self.ed = self.fft.compute_structure_factors(
            hkl,
            *self.get_iso(),
            *self.get_aniso(),
            apply_symmetry=True,
        )

        # Apply anomalous correction as post-processing
        if apply_anomalous and self.wavelength is not None:
            sf = self._apply_anomalous_correction(sf, hkl)

        if self.verbose > 2:
            assert torch.all(
                torch.isfinite(sf)
            ), "Non-finite values found while calculating fcalc."

        return sf

    def copy(self, detach: bool = True) -> "ModelFT":
        """
        Create a deep copy of the ModelFT.

        Creates a complete independent copy including all Model base class data,
        FFT submodule state (gridsize, real_space_grid, voxel_size, map_symmetry),
        ITC92 parametrization, and scalar attributes.
        Cache is reset to empty.

        Parameters
        ----------
        detach : bool, optional
            If True, the copy's parameters will be detached from the
            computation graph (default: True).

        Returns
        -------
        ModelFT
            A new ModelFT instance with copied data.

        Examples
        --------
        ::

            model = ModelFT().load_pdb('structure.pdb')
            model_copy = model.copy()
            # model_copy is independent, changes won't affect model
        """
        if not self.initialized:
            raise RuntimeError("Cannot copy an uninitialized ModelFT. Load data first.")

        # Create new ModelFT instance with same configuration
        model_copy = ModelFT(
            dtype_float=self.dtype_float,
            verbose=self.verbose,
            device=self.device,
            strip_H=self.strip_H,
            max_res=self.max_res,
            radius_angstrom=self.radius_angstrom,
            gridsize=self._explicit_gridsize,
            wavelength=self.wavelength,
            anomalous_threshold=self.anomalous_threshold,
        )

        # Deep copy the PDB DataFrame
        model_copy.pdb = self.pdb.copy(deep=True)

        # Copy spacegroup using its copy() method
        if self._spacegroup is not None:
            model_copy._spacegroup = self._spacegroup.copy()
        else:
            model_copy._spacegroup = None

        model_copy.initialized = True

        # Copy Cell object using its clone() method
        if self.cell is not None:
            model_copy.cell = self.cell.clone()

        # Copy all registered buffers using PyTorch's _buffers dict
        # (excluding FFT submodule buffers which are handled separately)
        for buffer_name, buffer_value in self._buffers.items():
            if buffer_value is not None:
                if detach:
                    model_copy.register_buffer(buffer_name, buffer_value.clone().detach())
                else:
                    model_copy.register_buffer(buffer_name, buffer_value.clone())

        # Copy all modules (parameter wrappers) using their .copy() methods
        # Skip _fft and _spacegroup as they are handled separately
        skip_modules = {"_fft", "_spacegroup", "spacegroup", "_symmetry", "symmetry"}
        for module_name, module in self._modules.items():
            if module_name in skip_modules:
                continue
            if module is not None and hasattr(module, "copy"):
                setattr(model_copy, module_name, module.copy())

        # Copy alternative conformation pairs
        if hasattr(self, "altloc_pairs") and self.altloc_pairs:
            model_copy.altloc_pairs = [
                tuple(tensor.clone() for tensor in group) for group in self.altloc_pairs
            ]
        else:
            model_copy.altloc_pairs = []

        # Copy FT-specific attributes: _parametrization dict
        if hasattr(self, "_parametrization") and self._parametrization is not None:
            import copy as copy_module
            model_copy._parametrization = copy_module.deepcopy(self._parametrization)

        # Copy FFT submodule using its copy() method
        if self._fft is not None:
            model_copy._fft = self._fft.copy()
            # Setup grid if it was set up in the original
            if self._fft.real_space_grid is not None:
                model_copy.setup_grid(max_res=self.max_res)

        # Reset cache on the copy (don't share cached structure factors)
        model_copy.reset_cache()

        if self.verbose > 0:
            print(f"✓ ModelFT copied successfully ({len(model_copy.pdb)} atoms)")

        return model_copy

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Return a dictionary containing the complete state of the ModelFT.

        Extends parent Model.state_dict() with FT-specific parameters including
        max_res, radius_angstrom. Grid state is handled by the FFT submodule.

        Parameters
        ----------
        destination : dict, optional
            Optional dict to populate.
        prefix : str, optional
            Prefix for parameter names. Default is ''.
        keep_vars : bool, optional
            Whether to keep variables in computational graph. Default is False.

        Returns
        -------
        dict
            Complete state dictionary.
        """
        # Get parent Model state_dict (includes _A, _B buffers and FFT submodule)
        state = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        # Add ModelFT-specific state
        state[prefix + "max_res"] = self.max_res
        state[prefix + "radius_angstrom"] = self.radius_angstrom
        state[prefix + "wavelength"] = self.wavelength
        state[prefix + "anomalous_threshold"] = self.anomalous_threshold

        # Note: FFT submodule state (gridsize, real_space_grid, voxel_size) is
        # automatically included via PyTorch's module serialization with _fft. prefix
        # _parametrization dict is not saved as it can be rebuilt from _A, _B buffers
        # _cache is not saved as it should be rebuilt
        # _anomalous_cache is not saved as it can be rebuilt from element list

        return state

    @classmethod
    def create_from_state_dict(
        cls,
        state_dict: dict,
        device: torch.device = torch.device("cpu"),
        verbose: int = 1,
        dtype_float: torch.dtype = dtypes.float,
    ) -> "ModelFT":
        """
        Create a fully initialized ModelFT from a state dictionary.

        This is the recommended way to restore a ModelFT from a saved state.
        Creates an instance with properly initialized submodules, then loads the state.

        Parameters
        ----------
        state_dict : dict
            State dictionary from torch.save(model.state_dict(), ...).
        device : torch.device, optional
            Device to place tensors on. Default is torch.device('cpu').
        verbose : int, optional
            Verbosity level. Default is 1.
        dtype_float : torch.dtype, optional
            Float dtype for tensors. Default is dtypes.float.

        Returns
        -------
        ModelFT
            Fully initialized instance with restored state.
        """
        from torchref.symmetry import SpaceGroup

        # Extract ModelFT-specific metadata
        max_res = state_dict.pop("max_res", 1.0)
        radius_angstrom = state_dict.pop("radius_angstrom", 4.0)
        wavelength = state_dict.pop("wavelength", 1.0)
        anomalous_threshold = state_dict.pop("anomalous_threshold", 0.5)

        # Extract Model metadata
        pdb = state_dict.pop("pdb", None)
        spacegroup_str = state_dict.pop("spacegroup", None)
        cell_tensor = state_dict.pop("cell", None)
        initialized = state_dict.pop("initialized", False)
        saved_dtype = state_dict.pop("dtype_float", dtypes.float)
        state_dict.pop("device", None)  # Remove but don't use (use provided device)
        strip_H = state_dict.pop("strip_H", True)
        altloc_pairs = state_dict.pop("altloc_pairs", [])

        # Extract grid info from FFT submodule state
        # Note: FFT buffers are prefixed with "_fft."
        gridsize = state_dict.pop("_fft.gridsize", None)
        # Also try old-style keys for backward compatibility
        if gridsize is None:
            gridsize = state_dict.pop("gridsize", None)

        # Create instance with FT-specific params
        instance = cls(
            dtype_float=saved_dtype,
            verbose=verbose,
            device=device,
            strip_H=strip_H,
            max_res=max_res,
            radius_angstrom=radius_angstrom,
            wavelength=wavelength,
            anomalous_threshold=anomalous_threshold,
        )

        # Set metadata
        instance.pdb = pdb
        instance.initialized = initialized
        instance.altloc_pairs = altloc_pairs

        # Setup spacegroup - setter also sets symmetry automatically
        instance.spacegroup = spacegroup_str

        # Create Cell object - setter will initialize FFT since spacegroup is already set
        from torchref.symmetry import Cell
        if cell_tensor is not None:
            instance.cell = Cell(cell_tensor, dtype=saved_dtype, device=device)

        # If PDB exists, create the parameter wrappers with correct shapes
        if pdb is not None:
            from torchref.model.parameter_wrappers import (
                MixedTensor,
                OccupancyTensor,
                PositiveMixedTensor,
            )

            n_atoms = len(pdb)

            # Create MixedTensors
            xyz_mask = state_dict.get("xyz.refinable_mask")
            b_mask = state_dict.get("b.refinable_mask")
            u_mask = state_dict.get("u.refinable_mask")

            instance.xyz = MixedTensor(
                torch.tensor(pdb[["x", "y", "z"]].values, dtype=saved_dtype),
                refinable_mask=xyz_mask,
                name="xyz",
            )
            instance.b = PositiveMixedTensor(
                torch.tensor(pdb["tempfactor"].values, dtype=saved_dtype),
                refinable_mask=b_mask,
                name="b_factor",
            )
            instance.u = MixedTensor(
                torch.tensor(
                    pdb[["u11", "u22", "u33", "u12", "u13", "u23"]].values,
                    dtype=saved_dtype,
                ),
                refinable_mask=u_mask,
                name="aniso_U",
            )

            # Create OccupancyTensor
            initial_occ = torch.tensor(pdb["occupancy"].values, dtype=saved_dtype)
            sharing_groups, altloc_groups, refinable_mask = (
                instance._create_occupancy_groups(pdb, initial_occ)
            )

            saved_occ_mask = state_dict.get("occupancy.refinable_mask")
            if saved_occ_mask is not None:
                if saved_occ_mask.device != sharing_groups.device:
                    saved_occ_mask = saved_occ_mask.to(sharing_groups.device)
                refinable_mask = saved_occ_mask[sharing_groups]

            instance.occupancy = OccupancyTensor(
                initial_values=initial_occ,
                sharing_groups=sharing_groups,
                altloc_groups=altloc_groups,
                refinable_mask=refinable_mask,
                dtype=saved_dtype,
                device=device,
                name="occupancy",
            )

            # Register aniso_flag buffer
            if "aniso_flag" not in instance._buffers or instance.aniso_flag is None:
                instance.register_buffer(
                    "aniso_flag",
                    torch.tensor(pdb["anisou_flag"].values, dtype=torch.bool),
                )

            # Register mask buffers
            instance.register_buffer(
                "xyz_mask", torch.ones(n_atoms, dtype=torch.bool, device=device)
            )
            instance.register_buffer(
                "b_mask", torch.ones(n_atoms, dtype=torch.bool, device=device)
            )
            instance.register_buffer(
                "u_mask", torch.ones(n_atoms, dtype=torch.bool, device=device)
            )
            instance.register_buffer(
                "occupancy_mask", torch.ones(n_atoms, dtype=torch.bool, device=device)
            )

            # Register vdw_radii if present
            if "vdw_radii" in state_dict and state_dict["vdw_radii"] is not None:
                instance.register_buffer(
                    "vdw_radii", torch.zeros_like(state_dict["vdw_radii"], device=device)
                )

            # Handle _A and _B buffers (scattering parameters)
            # Check both old-style (A, B) and new-style (_A, _B) keys
            a_key = "_A" if "_A" in state_dict else "A" if "A" in state_dict else None
            b_key = "_B" if "_B" in state_dict else "B" if "B" in state_dict else None

            if a_key and state_dict[a_key] is not None:
                instance.register_buffer(
                    "_A", torch.zeros_like(state_dict[a_key], device=device)
                )
            if b_key and state_dict[b_key] is not None:
                instance.register_buffer(
                    "_B", torch.zeros_like(state_dict[b_key], device=device)
                )

        # Setup grid via FFT submodule
        if gridsize is not None and cell_tensor is not None:
            if isinstance(gridsize, torch.Tensor):
                gs_tuple = tuple(int(x) for x in gridsize.tolist())
            else:
                gs_tuple = tuple(int(x) for x in gridsize)

            instance.setup_grid(gridsize=gs_tuple)

        # Filter state_dict and load
        # Remap old-style A/B keys to new _A/_B keys
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if not hasattr(v, 'shape') or v.shape[0] > 0:
                # Remap old keys to new keys
                if k == "A":
                    filtered_state_dict["_A"] = v
                elif k == "B":
                    filtered_state_dict["_B"] = v
                else:
                    filtered_state_dict[k] = v

        instance.load_state_dict(filtered_state_dict, strict=False)

        # Reset cache
        instance.reset_cache()

        if verbose > 0:
            n_atoms = len(instance.pdb) if instance.pdb is not None else 0
            print(f"Created ModelFT from state_dict: {n_atoms} atoms")

        return instance
