"""ModelFT -- a :class:`~torchref.model.Model` that can compute structure factors.

Adds the electron-density / FFT path (an :class:`~torchref.model.SfFFT` submodule
that reads the crystal off the model's context and sizes its grid lazily), the
ITC92 scattering parametrization, and the anomalous f' / f'' correction.
"""

import math
from typing import Optional, Tuple

import gemmi
import numpy as np
import torch

from torchref.base.fourier import fft, ifft
from torchref.config import canonical_device, dtypes, get_default_device, get_float_dtype
from torchref.model.model import Model
from torchref.model.sf_fft import SfFFT
from torchref.symmetry import SpaceGroup
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
    gridsize : torch.Tensor or None
        Grid dimensions ``(nx, ny, nz)``, derived by the ``SfFFT`` submodule from
        the cell, space group, ``max_res`` and ``explicit_gridsize`` on first use
        and re-derived when any of them changes. A coordinate grid is not stored;
        :meth:`real_space_grid` builds one on demand for the few callers that want
        the Cartesian positions themselves.
    map : torch.Tensor or None
        Most recently computed electron density map.
    parametrization : dict
        ITC92 parametrization dictionary {element: (A, B)}.
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

        # The engine reads cell and space group off ``self.ctx`` as they are set;
        # its grid is derived on first use and re-derived when the crystal,
        # ``max_res`` or ``explicit_gridsize`` change.
        self._fft = SfFFT(
            ctx=self.ctx,
            max_res=max_res,
            explicit_gridsize=gridsize,
            dtype_float=self.dtype_float,
            device=self.device,
            verbose=self.ctx.verbose,
        )

        self.wavelength = wavelength
        self.anomalous_threshold = anomalous_threshold
        # Whether to apply the imaginary f'' (Bijvoet) term. Registered as a buffer
        # so it round-trips through state_dict and follows .to(device). f' is always
        # applied when wavelength is set; f'' only when this is True (unmerged data).
        self.register_buffer(
            "anomalous_bijvoet",
            torch.tensor(bool(apply_bijvoet), device=self.device),
            persistent=True,
        )
        self._anomalous_cache = None  # Will hold (mask, f_prime, f_double_prime)
        self._anomalous_elements_hash = (
            None  # Hash of element list for cache invalidation
        )

    # =========================================================================
    # Engine binding and grid inputs
    # =========================================================================

    @property
    def fft(self) -> SfFFT:
        """The SfFFT submodule, bound to this model's context.

        ``copy()`` and ``load_state`` replace the context object itself; re-pointing
        the engine here keeps ``fft.ctx is self.ctx`` on every path.
        """
        fft = self._fft
        if fft.ctx is not self.ctx:
            fft.ctx = self.ctx
        return fft

    @property
    def max_res(self) -> Optional[float]:
        """Maximum resolution in Angstroms that sizes the grid; owned by the engine."""
        return self._fft.max_res

    @max_res.setter
    def max_res(self, value) -> None:
        self._fft.max_res = None if value is None else float(value)

    @property
    def explicit_gridsize(self) -> Optional[Tuple[int, int, int]]:
        """Fixed grid dimensions overriding ``max_res``, or None."""
        return self._fft.explicit_gridsize

    @explicit_gridsize.setter
    def explicit_gridsize(self, value) -> None:
        self._fft.explicit_gridsize = value

    @property
    def grid_key(self):
        """What the grid is derived from; see :attr:`SfFFT.grid_key`."""
        return self.fft.grid_key

    def _fingerprint_state(self):
        """Fold the grid key into the forward-cache key.

        Parameters and buffers alone would miss a cell, space-group or resolution
        change that leaves the grid buffers untouched until the next forward.
        """
        return super()._fingerprint_state() + (self.fft.grid_key,)

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
        return self

    def select(self, selection):
        """
        Return a new ModelFT containing only the selected atoms.

        Extends :meth:`Model.select` with the FT-specific setup: rebuilding
        the ITC92 parametrization and carrying ``max_res`` and
        ``explicit_gridsize`` across, so the selection sizes its grid the same way.

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
        ``wavelength`` and ``anomalous_threshold`` are **not** propagated:
        :meth:`Model.select` passes only the base kwargs, so the returned model
        carries the ModelFT defaults for those.
        """
        selection = super().select(selection)
        selection._build_parametrization()
        selection.max_res = self.max_res
        selection.explicit_gridsize = self.explicit_gridsize
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
        return self

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
    # Grid, resolved by the engine
    # =========================================================================

    @property
    def gridsize(self) -> Optional[torch.Tensor]:
        """Grid dimensions (nx, ny, nz), or None until cell and space group are set."""
        return self.fft.gridsize

    def real_space_grid(self) -> torch.Tensor:
        """Build the Cartesian coordinate of every grid point, ``(nx, ny, nz, 3)``.

        Not stored: at ``12 * nx * ny * nz`` bytes it is the largest tensor a model
        would hold, and no structure-factor path reads it -- every splat derives a
        voxel's position from its index. Built here for the callers that genuinely
        want the coordinates, and discarded when they are done with it.
        """
        from torchref.base.fourier import get_real_grid

        return get_real_grid(
            fractional_matrix=self.cell.fractional_matrix,
            gridsize=self.gridsize,
            device=self.device,
        )

    @property
    def grid_shape(self) -> Optional[tuple]:
        """Map dimensions ``(nx, ny, nz)``, or None until cell and space group are set."""
        return self.fft.grid_shape

    @property
    def voxel_size(self) -> Optional[torch.Tensor]:
        """Voxel edge vector sum, or None until cell and space group are set."""
        return self.fft.voxel_size

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

    def setup_grid(self, *, max_res=None, gridsize=None):
        """
        Override the grid's inputs explicitly and resolve the grid now.

        Not needed on the normal path: the engine sizes its grid from the cell,
        space group and ``max_res`` on first use and follows any later change.

        Parameters
        ----------
        max_res : float, optional
            New maximum resolution in Angstroms. None leaves the current value.
        gridsize : tuple of int, optional
            Fixed grid size (nx, ny, nz). None leaves :attr:`explicit_gridsize`
            unchanged.
        """
        self.fft.setup_grid(max_res=max_res, gridsize=gridsize)

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

        if self.ctx.verbose > 2:
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
        if self.ctx.verbose > 2:
            print("Building density map (per-atom variable radius)...")

        xyz_iso, adp_iso, occ_iso, A_iso, B_iso = self.get_iso()

        if self.ctx.verbose > 3:
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

        if self.ctx.verbose > 3:
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
        if self.ctx.verbose > 0:
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
        if self.ctx.verbose > 0:
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

            if self.ctx.verbose > 1 and significant:
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

        if self.ctx.verbose > 2:
            assert torch.all(
                torch.isfinite(sf)
            ), "Non-finite values found while calculating fcalc."

        return sf

    def copy(self, detach: bool = True) -> "ModelFT":
        """
        Create a deep copy of the ModelFT.

        Creates a complete independent copy including all Model base class data,
        the grid inputs (``max_res``, ``explicit_gridsize``; the grid itself is
        re-derived from the copied context), the ITC92 parametrization, and
        scalar attributes.
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
        if not self.ctx.initialized:
            raise RuntimeError("Cannot copy an uninitialized ModelFT. Load data first.")

        model_copy = ModelFT(
            dtype_float=self.dtype_float,
            verbose=self.ctx.verbose,
            device=self.device,
            strip_H=self.ctx.strip_H,
            max_res=self.max_res,
            gridsize=self.explicit_gridsize,
            wavelength=self.wavelength,
            anomalous_threshold=self.anomalous_threshold,
        )

        # Carries the atom table, cell, space group, altloc groups and provenance.
        model_copy.ctx = self.ctx.copy()

        # Own buffers only; the engine's grid buffers are derived, not copied.
        for buffer_name, buffer_value in self._buffers.items():
            if buffer_value is not None:
                if detach:
                    model_copy.register_buffer(
                        buffer_name, buffer_value.clone().detach()
                    )
                else:
                    model_copy.register_buffer(buffer_name, buffer_value.clone())

        # Parameter wrappers via their own .copy(); the engine came from the ctor.
        skip_modules = {"_fft"}
        for module_name, module in self._modules.items():
            if module_name in skip_modules:
                continue
            if module is not None and hasattr(module, "copy"):
                setattr(model_copy, module_name, module.copy())

        if hasattr(self, "_parametrization") and self._parametrization is not None:
            import copy as copy_module

            model_copy._parametrization = copy_module.deepcopy(self._parametrization)

        # Don't share cached structure factors with the original.
        model_copy.reset_cache()

        if self.ctx.verbose > 0:
            print(f"✓ ModelFT copied successfully ({len(model_copy.pdb)} atoms)")

        return model_copy

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Return a dictionary containing the complete state of the ModelFT.

        Extends parent Model.state_dict() with FT-specific parameters:
        ``max_res``, ``explicit_gridsize``, ``wavelength`` and
        ``anomalous_threshold``. The grid is derived from these and the crystal,
        so it is not stored.

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
        # Parent covers _A/_B; the engine's grid buffers are non-persistent.
        state = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        state[prefix + "max_res"] = self.max_res
        state[prefix + "explicit_gridsize"] = self.explicit_gridsize
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
            Move the restored model here once it is built. The restore itself always
            runs on CPU; ``None`` then moves it to the configured default device; see
            :meth:`Model.create_from_state_dict`.
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
        # Build on CPU throughout and move once at the end, as Model does; the grid
        # setup below otherwise sizes an accelerator allocation before the model is
        # placed. The final target is the caller's device, or the configured default
        # when they name none, so a restore lands beside a same-config model.
        target_device = (
            canonical_device(device) if device is not None else get_default_device()
        )
        device = torch.device("cpu")
        if dtype_float is None:
            dtype_float = get_float_dtype()

        max_res = state_dict.pop("max_res", 1.0)
        explicit_gridsize = state_dict.pop("explicit_gridsize", None)
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

        # Checkpoints written while the grid was stored state carry its buffers
        # ("_fft." prefixed, or flat in older ones). The size is adopted below only
        # when it differs from what the crystal and max_res give.
        legacy_gridsize = state_dict.pop("_fft.gridsize", None)
        if legacy_gridsize is None:
            legacy_gridsize = state_dict.pop("gridsize", None)
        state_dict.pop("_fft.voxel_size", None)
        state_dict.pop("voxel_size", None)

        instance = cls(
            dtype_float=saved_dtype,
            verbose=verbose,
            device=device,
            strip_H=strip_H,
            max_res=max_res,
            gridsize=explicit_gridsize,
            wavelength=wavelength,
            anomalous_threshold=anomalous_threshold,
        )

        instance.pdb = pdb
        instance.ctx.initialized = initialized
        instance.ctx.altloc_pairs = altloc_pairs

        # The engine reads both off the context; nothing further to build.
        instance.spacegroup = spacegroup_str

        from torchref.symmetry import Cell

        if cell_tensor is not None:
            instance.cell = Cell(cell_tensor, dtype=saved_dtype, device=device)

        # Wrappers and per-atom buffers: shared with Model so the two restores cannot
        # drift apart again. ModelFT adds only its own scattering buffers below.
        if pdb is not None:
            cls._rebuild_wrappers_from_pdb(
                instance, pdb, state_dict, saved_dtype, device
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

        if (
            legacy_gridsize is not None
            and explicit_gridsize is None
            and instance.ctx.crystal_key is not None
            and instance.max_res is not None
        ):
            if isinstance(legacy_gridsize, torch.Tensor):
                legacy_gridsize = legacy_gridsize.tolist()
            legacy = tuple(int(x) for x in legacy_gridsize)
            if legacy != instance.fft.compute_optimal_gridsize(instance.max_res):
                instance.explicit_gridsize = legacy

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

        # Always placed: target_device is the caller's device or the configured default.
        instance.to(target_device)

        instance.reset_cache()

        if verbose > 0:
            n_atoms = len(instance.pdb) if instance.pdb is not None else 0
            print(f"Created ModelFT from state_dict: {n_atoms} atoms")

        return instance
