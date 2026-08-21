"""ModelFT -- a :class:`~torchref.model.Model` that can compute structure factors.

Adds the electron-density / FFT path (via an :class:`~torchref.model.SfFFT`
submodule created as soon as both cell and space group are set), the ITC92
scattering parametrization, and the anomalous f' / f'' correction.
"""

import math
from typing import Optional, Tuple

import gemmi
import numpy as np
import torch

from torchref.base.fourier import fft, ifft
from torchref.config import dtypes, get_float_dtype, normalize_device
from torchref.model.model import Model
from torchref.model.sf_fft import SfFFT
from torchref.symmetry import SpaceGroup
from torchref.symmetry.map_symmetry import MapSymmetry
from torchref.utils.caching import CachedForwardMixin


class ModelFT(CachedForwardMixin, Model):
    """
    Model subclass for FFT-based electron density and structure factors.

    Extends :class:`Model` with real-space density maps and structure factors via
    FFT, using the ITC92 scattering parametrization. Build it empty
    (``ModelFT()`` then ``load_pdb`` / ``load_state_dict``) or with parameters
    (``ModelFT(max_res=1.5)``).

    Parameters
    ----------
    max_res : float, optional
        Maximum resolution for grid spacing in Angstroms. Default is 1.0.
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
    max_res, wavelength, anomalous_threshold : float
        The constructor arguments above, readable back as attributes.
    gridsize, real_space_grid : torch.Tensor
        Grid dimensions ``(nx, ny, nz)`` and coordinate grid
        ``(nx, ny, nz, 3)``; both live on the ``SfFFT`` submodule.
    map : torch.Tensor or None
        Most recently computed electron density map.
    parametrization : dict
        ITC92 parametrization dictionary {element: (A, B)}.
    map_symmetry : MapSymmetry
        Symmetry operator for map calculations.
    """

    def __init__(
        self,
        *args,
        max_res=1.0,
        gridsize: Optional[Tuple[int, int, int]] = None,
        wavelength: Optional[float] = 1.0,
        anomalous_threshold: float = 0.5,
        apply_bijvoet: bool = False,
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
            (The splat radius is *not* set here: each atom is truncated at its own
            ``torchref.sigma_cutoff_ed * sigma_eff``.)
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
        apply_bijvoet : bool, optional
            Apply the imaginary f'' (Bijvoet) term, which breaks Friedel's law
            (``F(+h) != F(-h)``). Default False, and correct only for
            Friedel-unmerged data -- on merged data f'' cannot affect the
            Friedel-mean amplitude. The dispersive f' is applied whenever a
            wavelength is set. Bound from ``ReflectionData.friedel_merged``.
        *args
            Passed to parent Model class.
        **kwargs
            Passed to parent Model class.
        """
        super().__init__(*args, **kwargs)

        self.max_res = max_res
        self._explicit_gridsize = gridsize

        self.wavelength = wavelength
        self.anomalous_threshold = anomalous_threshold
        # Whether to apply the imaginary f'' (Bijvoet) term. Registered as a buffer
        # so it round-trips through state_dict and follows .to(device). f' is always
        # applied when wavelength is set; f'' only when this is True (unmerged data).
        self.register_buffer(
            "anomalous_bijvoet", torch.tensor(bool(apply_bijvoet)), persistent=True
        )
        self._anomalous_cache = None  # Will hold (mask, f_prime, f_double_prime)
        self._anomalous_elements_hash = (
            None  # Hash of element list for cache invalidation
        )
        self._fft = None

    @property
    def cell(self):
        """Unit cell object with parameters [a, b, c, alpha, beta, gamma]."""
        return self._cell

    @cell.setter
    def cell(self, value):
        """Set the unit cell; also builds the FFT once the spacegroup is set."""
        self._cell = value
        self._maybe_initialize_fft()

    @property
    def spacegroup(self):
        """Space group object."""
        return self._spacegroup

    @spacegroup.setter
    def spacegroup(self, value):
        """Set the space group (SpaceGroup, gemmi.SpaceGroup, name or number);
        also builds the FFT once the cell is set.
        """
        if value is not None:
            self._spacegroup = SpaceGroup(
                value, dtype=self.dtype_float, device=self.device
            )
        else:
            self._spacegroup = None
        self._maybe_initialize_fft()

    def _maybe_initialize_fft(self):
        """(Re)build the SfFFT submodule once both cell and spacegroup are set."""
        if self._cell is not None and self._spacegroup is not None:
            self._fft = SfFFT(
                cell=self._cell,
                spacegroup=self._spacegroup,
                device=self.device,
                max_res=self.max_res,
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
        """
        Return a new ModelFT containing only the selected atoms.

        Extends :meth:`Model.select` with the FT-specific setup: rebuilding
        the ITC92 parametrization and the real-space grid for the reduced
        atom set. The FFT itself is initialized via the cell/spacegroup
        setters during the base ``select``.

        Parameters
        ----------
        selection : array-like or str
            Atom selection forwarded to :meth:`Model.select`.

        Returns
        -------
        ModelFT
            A new model holding the selected atoms.

        Notes
        -----
        The ModelFT-specific constructor arguments -- ``max_res``,
        ``wavelength``, ``anomalous_threshold``, ``gridsize`` -- are **not**
        propagated: :meth:`Model.select` passes only the base kwargs, so the
        returned model silently carries the ModelFT defaults for those.
        """
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
            print(f"Defining grid size for max_res={self.max_res} Å")

        gridsize = self.cell.compute_grid_size(self.max_res)
        return torch.tensor(gridsize, dtype=dtypes.int, device=self.device)

    def _build_parametrization(self):
        """Build the ITC92 parametrization (delegates to :class:`Model`)."""
        return super()._build_parametrization()

    # =========================================================================
    # Backward-compatible properties for scattering parameters
    # =========================================================================

    @property
    def A(self) -> torch.Tensor:
        """ITC92 A parameters (amplitudes), ``(n_atoms, 5)``; builds them if needed."""
        self._build_parametrization()
        return self._A

    @property
    def B(self) -> torch.Tensor:
        """ITC92 B parameters (widths), ``(n_atoms, 5)``; builds them if needed."""
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

        Returns the isotropic subset only (shape ``n_iso``), as produced by
        :meth:`Model.get_iso`, with the per-atom scattering parameters
        appended.

        Returns
        -------
        xyz : torch.Tensor
            Atomic coordinates with shape (n_iso, 3).
        adp : torch.Tensor
            Atomic displacement parameters (isotropic) with shape (n_iso,).
        occupancy : torch.Tensor
            Occupancies with shape (n_iso,).
        A : torch.Tensor
            ITC92 A parameters (amplitudes) with shape (n_iso, 5).
        B : torch.Tensor
            ITC92 B parameters (widths) with shape (n_iso, 5).
        """
        xyz, adp, occupancy = super().get_iso()
        A, B = self.get_scattering_params_iso()

        return xyz, adp, occupancy, A, B

    def get_aniso(self):
        """
        Get anisotropic atoms with their ITC92 parameters.

        Returns the anisotropic subset only (shape ``n_aniso``), as produced
        by :meth:`Model.get_aniso`, with the per-atom scattering parameters
        appended.

        Returns
        -------
        xyz : torch.Tensor
            Atomic coordinates with shape (n_aniso, 3).
        u : torch.Tensor
            Anisotropic U parameters with shape (n_aniso, 6).
        occupancy : torch.Tensor
            Occupancies with shape (n_aniso,).
        A : torch.Tensor
            ITC92 A parameters (amplitudes) with shape (n_aniso, 5).
        B : torch.Tensor
            ITC92 B parameters (widths) with shape (n_aniso, 5).
        """
        xyz, u, occupancy = super().get_aniso()
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

        gridsize_to_use = gridsize or self._explicit_gridsize

        self._fft.setup_grid(
            gridsize=gridsize_to_use,
            max_res=self.max_res,
        )

        if self.verbose > 2:
            print(f"Grid shape: {self._fft.real_space_grid.shape[:-1]}")
            print(f"Voxel size: {self._fft.voxel_size}")

    def get_radius(self, min_radius_Angstrom: float = 4.0):
        """
        Get a single fixed splat radius in voxels for the given minimum.

        Vestigial: the density path truncates each atom at its own
        ``torchref.sigma_cutoff_ed * sigma_eff`` radius and never consults this.

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
            Accepted for backward compatibility but unused; the density splat
            radius is per-atom (``torchref.sigma_cutoff_ed`` sigmas), resolved
            inside the density builder. Default is None.
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
            print("Building density map (per-atom variable radius)...")

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

        xyz_aniso, u_aniso, occ_aniso, A_aniso, B_aniso = self.get_aniso()

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
            raise ValueError(
                "No map to save. Call build_complete_map() (or "
                "build_initial_map()) to compute the density map first."
            )

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

    def update_pdb(self):
        """
        Update PDB with current atomic parameters.
        """
        return super().update_pdb()

    def reset_cache(self):
        """Reset SF cache, anomalous cache, and all wrapper forward caches."""
        self.reset_forward_cache()
        # Drop the anomalous scattering cache; it is recomputed on next use
        # and would otherwise hold tensors on the previous device.
        self._anomalous_cache = None
        self._anomalous_elements_hash = None
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
        """Cached ``(mask, f_prime, f_double_prime, has_anomalous, indices)``.

        ``mask`` is per-atom; ``f_prime`` / ``f_double_prime`` cover only the
        significant scatterers. Recomputed when the element list changes.
        """
        from torchref.base.scattering.anomalous_table import (
            get_anomalous_corrections_by_indices,
            get_significant_elements,
        )

        element_list = self.pdb["element"].tolist()
        elements_hash = hash(tuple(element_list))

        if (
            self._anomalous_cache is None
            or self._anomalous_elements_hash != elements_hash
        ):
            unique_elements = list(set(element_list))
            significant = get_significant_elements(
                unique_elements, self.wavelength, self.anomalous_threshold
            )

            if self.verbose > 1 and significant:
                print(
                    f"Anomalous scatterers at {self.wavelength:.4f} Å: "
                    f"{list(significant.keys())}"
                )

            mask, f_prime, f_double_prime = get_anomalous_corrections_by_indices(
                element_list, significant, self.device, self.dtype_float
            )

            # Pre-compute integer indices to avoid boolean indexing GPU sync
            has_anomalous = bool(mask.any().item())
            anomalous_indices = (
                mask.nonzero(as_tuple=True)[0] if has_anomalous else None
            )
            self._anomalous_cache = (
                mask,
                f_prime,
                f_double_prime,
                has_anomalous,
                anomalous_indices,
            )
            self._anomalous_elements_hash = elements_hash

        return self._anomalous_cache

    def _apply_anomalous_correction(
        self,
        sf: torch.Tensor,
        hkl: torch.Tensor,
        include_fdp: bool = True,
    ) -> torch.Tensor:
        """Add ``ΔF(h) = Σ (f' + i f'') exp(2πi h·r) occ`` to ``sf``.

        Only the significant scatterers (|f'| or |f''| above
        ``anomalous_threshold``) contribute. ``include_fdp=False`` zeroes f'',
        keeping Friedel's law intact -- the correct choice for merged data.
        """
        mask, f_prime, f_double_prime, has_anomalous, anomalous_indices = (
            self._get_anomalous_cache()
        )

        if not has_anomalous:
            return sf  # No significant anomalous scatterers

        # Integer indices, not the boolean mask: boolean indexing forces a GPU sync.
        xyz_frac = self.xyz_fractional()[anomalous_indices]  # (n_significant, 3)
        occ = self.occupancy()[anomalous_indices]  # (n_significant,)

        # Phase factors exp(2πi h·r), h·r over fractional coordinates
        h_dot_r = torch.matmul(
            hkl.to(dtype=self.dtype_float, device=xyz_frac.device), xyz_frac.T
        )  # (n_refl, n_significant)
        phase = 2 * torch.pi * h_dot_r

        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        f_prime_occ = f_prime * occ  # (n_significant,)
        f_double_prime_occ = f_double_prime * occ  # (n_significant,)
        if not include_fdp:
            # Dispersive f' only, so Friedel's law is preserved (merged data).
            f_double_prime_occ = torch.zeros_like(f_double_prime_occ)

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
        The full scattering factor is ``f(s, λ) = f₀(s) + f'(λ) + i f''(λ)``,
        with f₀ from the FFT and the wavelength-dependent f' / f'' applied only
        to atoms above ``anomalous_threshold``.
        """
        return self(hkl, recalc=recalc, apply_anomalous=apply_anomalous)

    @property
    def fft(self):
        """The SfFFT submodule, built on first access (needs cell + spacegroup)."""
        if self._fft is None:
            self._maybe_initialize_fft()

        return self._fft

    def _check_forward_dtype(self, hkl: torch.Tensor) -> None:
        """Fail fast on a model/input float-dtype mismatch, which would otherwise
        surface as a cryptic matmul or Triton-compile error deep in the kernels.

        Integer ``hkl`` always passes (it is cast internally); only *floating*
        ``hkl`` of the wrong dtype, or drifted parameters, are rejected.
        """
        model_dtype = self.dtype_float
        params = self.xyz.refinable_params
        if params is not None and params.numel() and params.dtype != model_dtype:
            raise TypeError(
                f"ModelFT parameters are {params.dtype} but model.dtype_float is "
                f"{model_dtype}. The model is in an inconsistent float dtype; "
                f"rebuild it or call model.to(dtype=...) before computing "
                f"structure factors."
            )
        if hkl.is_floating_point() and hkl.dtype != model_dtype:
            raise TypeError(
                f"hkl has dtype {hkl.dtype} but the model float dtype is "
                f"{model_dtype}. Pass integer Miller indices, or cast with "
                f"hkl.to(model.dtype_float). To run the model in float64, set "
                f"torchref.dtypes.float = torch.float64 before constructing it "
                f"(or TORCHREF_DTYPE_FLOAT=float64)."
            )

    def forward(self, hkl, apply_anomalous: bool = True) -> torch.Tensor:
        """
        Compute structure factors for given hkl.

        This is called by the mixin's ``__call__`` which handles caching,
        backward-hook registration, and auto-invalidation.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices with shape (n_reflections, 3).
        apply_anomalous : bool, optional
            If True and wavelength is set, apply anomalous scattering corrections.
            The dispersive f' term is always applied in that case; the imaginary
            f'' (Bijvoet) term is applied only when ``self.anomalous_bijvoet`` is
            True (i.e. for Friedel-unmerged data). Default is True.

        Returns
        -------
        torch.Tensor
            Calculated complex structure factors with shape (n_reflections,).
        """
        self._check_forward_dtype(hkl)
        sf, self.ed = self.fft.compute_structure_factors(
            hkl,
            *self.get_iso(),
            *self.get_aniso(),
            apply_symmetry=True,
        )

        # Apply anomalous correction as post-processing. f' always applies when a
        # wavelength is set; f'' only for unmerged (Bijvoet) data.
        if apply_anomalous and self.wavelength is not None:
            sf = self._apply_anomalous_correction(
                sf, hkl, include_fdp=bool(self.anomalous_bijvoet)
            )

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
            A new, fully independent ModelFT instance with copied data.
        """
        if not self.initialized:
            raise RuntimeError("Cannot copy an uninitialized ModelFT. Load data first.")

        model_copy = ModelFT(
            dtype_float=self.dtype_float,
            verbose=self.verbose,
            device=self.device,
            strip_H=self.strip_H,
            max_res=self.max_res,
            gridsize=self._explicit_gridsize,
            wavelength=self.wavelength,
            anomalous_threshold=self.anomalous_threshold,
        )

        model_copy.pdb = self.pdb.copy(deep=True)

        if self._spacegroup is not None:
            model_copy._spacegroup = self._spacegroup.copy()
        else:
            model_copy._spacegroup = None

        model_copy.initialized = True

        if self.cell is not None:
            model_copy.cell = self.cell.clone()

        # Own buffers only; the FFT submodule's are handled by its copy() below.
        for buffer_name, buffer_value in self._buffers.items():
            if buffer_value is not None:
                if detach:
                    model_copy.register_buffer(
                        buffer_name, buffer_value.clone().detach()
                    )
                else:
                    model_copy.register_buffer(buffer_name, buffer_value.clone())

        # Parameter wrappers via their own .copy(); _fft / _spacegroup are separate.
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

        if hasattr(self, "_parametrization") and self._parametrization is not None:
            import copy as copy_module

            model_copy._parametrization = copy_module.deepcopy(self._parametrization)

        if self._fft is not None:
            model_copy._fft = self._fft.copy()
            if self._fft.real_space_grid is not None:
                model_copy.setup_grid(max_res=self.max_res)

        # Don't share cached structure factors with the original.
        model_copy.reset_cache()
        # The iso/aniso partition is derived state, not a buffer, so it is not
        # carried by the buffer loop above; get_iso()/get_aniso() read it.
        model_copy._rebuild_sf_indices()

        if self.verbose > 0:
            print(f"✓ ModelFT copied successfully ({len(model_copy.pdb)} atoms)")

        return model_copy

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Return a dictionary containing the complete state of the ModelFT.

        Extends parent Model.state_dict() with FT-specific parameters:
        ``max_res``, ``wavelength``, and ``anomalous_threshold``. Grid state
        is handled by the FFT submodule.

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
        # Parent covers _A/_B and the FFT submodule's buffers.
        state = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        state[prefix + "max_res"] = self.max_res
        state[prefix + "wavelength"] = self.wavelength
        state[prefix + "anomalous_threshold"] = self.anomalous_threshold

        # Deliberately not saved, all rebuildable: _parametrization (from _A/_B),
        # _cache, _anomalous_cache (from the element list).
        return state

    @classmethod
    def create_from_state_dict(
        cls,
        state_dict: dict,
        device: torch.device = None,
        verbose: int = 1,
        dtype_float: torch.dtype = None,
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
            Device to place tensors on. Defaults to the configured device.current.
        verbose : int, optional
            Verbosity level. Default is 1.
        dtype_float : torch.dtype, optional
            Float dtype for tensors. Default is dtypes.float.

        Returns
        -------
        ModelFT
            Fully initialized instance with restored state.

        Notes
        -----
        Legacy state_dicts are accepted: the obsolete ``radius_angstrom`` key is
        ignored and old-style ``A`` / ``B`` buffers are remapped to ``_A`` / ``_B``.
        The anisotropic ``u`` is rebuilt as a :class:`CholeskyMixedTensor`, as in
        :meth:`load`, so the positive-definite parametrization round-trips.
        """
        from torchref.symmetry import SpaceGroup

        # Resolve dtype/device at call time so the fallback below uses the
        # current config rather than an import-time default.
        device = normalize_device(device)
        if dtype_float is None:
            dtype_float = get_float_dtype()

        max_res = state_dict.pop("max_res", 1.0)
        state_dict.pop("radius_angstrom", None)  # legacy key, no longer used
        wavelength = state_dict.pop("wavelength", 1.0)
        anomalous_threshold = state_dict.pop("anomalous_threshold", 0.5)

        pdb = state_dict.pop("pdb", None)
        spacegroup_str = state_dict.pop("spacegroup", None)
        cell_tensor = state_dict.pop("cell", None)
        initialized = state_dict.pop("initialized", False)
        saved_dtype = state_dict.pop("dtype_float", dtype_float)
        state_dict.pop("device", None)  # Remove but don't use (use provided device)
        strip_H = state_dict.pop("strip_H", True)
        altloc_pairs = state_dict.pop("altloc_pairs", [])

        # FFT submodule buffers are prefixed "_fft."; older checkpoints are flat.
        gridsize = state_dict.pop("_fft.gridsize", None)
        if gridsize is None:
            gridsize = state_dict.pop("gridsize", None)

        instance = cls(
            dtype_float=saved_dtype,
            verbose=verbose,
            device=device,
            strip_H=strip_H,
            max_res=max_res,
            wavelength=wavelength,
            anomalous_threshold=anomalous_threshold,
        )

        instance.pdb = pdb
        instance.initialized = initialized
        instance.altloc_pairs = altloc_pairs

        # Setter also sets symmetry; the cell setter below then builds the FFT.
        instance.spacegroup = spacegroup_str

        from torchref.symmetry import Cell

        if cell_tensor is not None:
            instance.cell = Cell(cell_tensor, dtype=saved_dtype, device=device)

        # If PDB exists, create the parameter wrappers with correct shapes
        if pdb is not None:
            from torchref.model.parameter_wrappers import (
                CholeskyMixedTensor,
                MixedTensor,
                OccupancyTensor,
                PositiveMixedTensor,
            )

            n_atoms = len(pdb)

            xyz_mask = state_dict.get("xyz.refinable_mask")
            adp_mask = state_dict.get("adp.refinable_mask")
            u_mask = state_dict.get("u.refinable_mask")

            instance.xyz = MixedTensor(
                torch.tensor(pdb[["x", "y", "z"]].values, dtype=saved_dtype),
                refinable_mask=xyz_mask,
                name="xyz",
            )
            instance.adp = PositiveMixedTensor(
                torch.tensor(pdb["tempfactor"].values, dtype=saved_dtype),
                refinable_mask=adp_mask,
                name="adp",
            )
            instance.u = CholeskyMixedTensor(
                torch.tensor(
                    pdb[["u11", "u22", "u33", "u12", "u13", "u23"]].values,
                    dtype=saved_dtype,
                ),
                refinable_mask=u_mask,
                name="aniso_U",
            )

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
                "adp_mask", torch.ones(n_atoms, dtype=torch.bool, device=device)
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
                    "vdw_radii",
                    torch.zeros_like(state_dict["vdw_radii"], device=device),
                )

            # Scattering buffers: accept both old-style (A, B) and new (_A, _B).
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

        if gridsize is not None and cell_tensor is not None:
            if isinstance(gridsize, torch.Tensor):
                gs_tuple = tuple(int(x) for x in gridsize.tolist())
            else:
                gs_tuple = tuple(int(x) for x in gridsize)

            instance.setup_grid(gridsize=gs_tuple)

        # Drop empty placeholders, remapping old-style A/B keys to _A/_B.
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if not hasattr(v, "shape") or v.numel() > 0:
                if k == "A":
                    filtered_state_dict["_A"] = v
                elif k == "B":
                    filtered_state_dict["_B"] = v
                else:
                    filtered_state_dict[k] = v

        instance.load_state_dict(filtered_state_dict, strict=False)

        instance.reset_cache()

        if verbose > 0:
            n_atoms = len(instance.pdb) if instance.pdb is not None else 0
            print(f"Created ModelFT from state_dict: {n_atoms} atoms")

        return instance
