"""ModelFT -- a :class:`~torchref.model.Model` that can compute structure factors.

Adds the electron-density / FFT path (an :class:`~torchref.model.SfFFT` submodule
that reads the crystal off the model's context and sizes its grid lazily), the
ITC92 scattering parametrization, and the anomalous f' / f'' terms. Those enter the
same density as f0 rather than a separate sum, so every term of F_calc gets the same
temperature factors and symmetry expansion (see :meth:`ModelFT.forward`).
"""

import math
from typing import NamedTuple, Optional, Tuple

import gemmi
import numpy as np
import torch

from torchref.base.fourier import fft, ifft
from torchref.config import dtypes
from torchref.model.model import Model
from torchref.model.sf_fft import SfFFT
from torchref.symmetry import SpaceGroup
from torchref.utils.caching import CachedForwardMixin


class _AnomalousTerms(NamedTuple):
    """f' and f'' laid out for :meth:`ModelFT._add_anomalous_scattering`.

    ``f_prime_iso`` / ``f_prime_aniso`` are ``(n_iso, 5)`` / ``(n_aniso, 5)`` addends to
    the ITC92 amplitudes of :meth:`ModelFT.get_iso` / :meth:`ModelFT.get_aniso`: f' in
    electrons in the zero-width column (``CONSTANT_TERM``) for atoms above
    ``anomalous_threshold``, zero everywhere else. ``rows_iso`` / ``rows_aniso`` index
    those atoms within the two subsets, and ``f_double_prime_iso`` /
    ``f_double_prime_aniso`` are their f'' amplitudes, ``(len(rows), 5)``, laid out the
    same way.
    """

    f_prime_iso: torch.Tensor
    f_prime_aniso: torch.Tensor
    rows_iso: torch.Tensor
    rows_aniso: torch.Tensor
    f_double_prime_iso: torch.Tensor
    f_double_prime_aniso: torch.Tensor


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
        X-ray wavelength of the data in Angstroms, which sets the anomalous f'
        and f''. Default None: no anomalous scattering, f0 only. f' and f'' are
        strongly wavelength-dependent near an absorption edge, so pass the
        wavelength the data were collected at, not a nominal one.
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
    max_res, wavelength, anomalous_threshold : float or None
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
        wavelength: Optional[float] = None,
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
            X-ray wavelength of the data in Angstroms, which sets the anomalous
            f' and f''. Default None: no anomalous scattering, f0 only.
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
        # (key, partition, _AnomalousTerms or None); see _get_anomalous_cache.
        self._anomalous_cache = None

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
        """Fold the grid key and the anomalous settings into the forward-cache key.

        Parameters and buffers alone would miss a cell, space-group or resolution
        change that leaves the grid buffers untouched until the next forward, and a
        new ``wavelength`` or ``anomalous_threshold``, which are plain attributes.
        """
        return super()._fingerprint_state() + (
            self.wavelength,
            self.anomalous_threshold,
            self.fft.grid_key,
        )

    # =========================================================================
    # ITC92 scattering parameters, built on first use
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

    def build_complete_map(self, apply_symmetry=True):
        """
        Build electron density map from all atoms.

        Uses get_iso() and get_aniso() to get atom data and constructs
        the complete electron density map.

        Parameters
        ----------
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

    def reset_cache(self):
        """Reset SF cache, anomalous cache, and all wrapper forward caches."""
        self.reset_forward_cache()
        # Drop the anomalous scattering cache; it is recomputed on next use
        # and would otherwise hold tensors on the previous device.
        self._anomalous_cache = None
        for module in self.children():
            if hasattr(module, "reset_forward_cache"):
                module.reset_forward_cache()

    # =========================================================================
    # Anomalous scattering
    # =========================================================================

    def _get_anomalous_cache(self) -> Optional[_AnomalousTerms]:
        """f' and f'' of the atoms above ``anomalous_threshold``; None if there are none.

        Rebuilt when the element list, ``wavelength``, ``anomalous_threshold`` or the
        iso/aniso partition changes. Building it costs a device sync; using it, none.

        Raises
        ------
        RuntimeError
            If an anomalous atom's ITC92 column ``CONSTANT_TERM`` has a nonzero width,
            which would spread f' and f'' like an f0 Gaussian.
        """
        from torchref.base.scattering.anomalous_table import (
            get_anomalous_corrections_by_indices,
            get_significant_elements,
        )
        from torchref.base.scattering.scattering_table import CONSTANT_TERM

        elements = self.ctx.topology.atoms.element.tolist()
        key = (hash(tuple(elements)), self.wavelength, self.anomalous_threshold)
        # Compared by identity: ``_sf_partition`` hands back the same tuple until the
        # aniso flags or the hydrogen choice change.
        partition = self._sf_partition()
        cached = self._anomalous_cache
        if cached is not None and cached[0] == key and cached[1] is partition:
            return cached[2]

        terms = None
        significant = get_significant_elements(
            sorted(set(elements)), self.wavelength, self.anomalous_threshold
        )
        if significant:
            if self.ctx.verbose > 1:
                print(
                    f"Anomalous scatterers at {self.wavelength:.4f} Å: "
                    f"{sorted(significant)}"
                )
            mask, f_prime, f_double_prime = get_anomalous_corrections_by_indices(
                elements, significant, self.device, self.dtype_float
            )
            rows = mask.nonzero(as_tuple=True)[0]
            if bool((self.B[rows, CONSTANT_TERM] != 0).any()):
                raise RuntimeError(
                    f"{type(self).__name__}: ITC92 column {CONSTANT_TERM} must be the "
                    "zero-width constant term to carry f' and f'', but an anomalous "
                    "atom has a nonzero width there."
                )
            addend_fp = self.A.new_zeros(len(elements), self.A.shape[1])
            addend_fdp = torch.zeros_like(addend_fp)
            addend_fp[rows, CONSTANT_TERM] = f_prime
            addend_fdp[rows, CONSTANT_TERM] = f_double_prime

            iso_idx, aniso_idx = partition[0], partition[1]
            rows_iso = mask[iso_idx].nonzero(as_tuple=True)[0].to(dtypes.int)
            rows_aniso = mask[aniso_idx].nonzero(as_tuple=True)[0].to(dtypes.int)
            terms = _AnomalousTerms(
                f_prime_iso=addend_fp[iso_idx],
                f_prime_aniso=addend_fp[aniso_idx],
                rows_iso=rows_iso,
                rows_aniso=rows_aniso,
                f_double_prime_iso=addend_fdp[iso_idx][rows_iso],
                f_double_prime_aniso=addend_fdp[aniso_idx][rows_aniso],
            )

        self._anomalous_cache = (key, partition, terms)
        return terms

    def _add_anomalous_scattering(self, iso, aniso, include_fdp: bool):
        """Put f' and f'' into the atoms :meth:`forward` hands to the FFT engine.

        Neither term depends on the scattering angle, so each is a zero-width Gaussian
        in the atom's form factor -- the slot ITC92's constant ``c`` already occupies.
        Placed there, both get exactly what f0 gets: the splat widens the term by the
        atom's own isotropic or anisotropic displacement, and the FFT path expands it
        over the symmetry operators. f' joins the real amplitudes. f'' becomes the only
        amplitude of a copy of the anomalous atoms, which the engine splats into the
        imaginary part of the density; each copy keeps its atom's ITC92 widths so both
        parts are truncated at the same per-atom radius.

        Parameters
        ----------
        iso, aniso : tuple of torch.Tensor
            :meth:`get_iso` and :meth:`get_aniso` -- read from here rather than from the
            parameter wrappers, so a subclass that adjusts those (``EnsembleModel``)
            applies to the anomalous terms too.
        include_fdp : bool
            Build the f'' atoms. False keeps ``F(-h) = F(h)*``, which merged data need.

        Returns
        -------
        iso, aniso : tuple of torch.Tensor
            The inputs with f' added to ``A``; returned as given without significant
            scatterers.
        imaginary : tuple of torch.Tensor or None
            The f'' atoms in the layout of ``(*iso, *aniso)``, or None.
        """
        terms = self._get_anomalous_cache()
        if terms is None:
            return iso, aniso, None
        xyz_i, adp_i, occ_i, A_i, B_i = iso
        xyz_a, u_a, occ_a, A_a, B_a = aniso
        iso = (xyz_i, adp_i, occ_i, A_i + terms.f_prime_iso, B_i)
        aniso = (xyz_a, u_a, occ_a, A_a + terms.f_prime_aniso, B_a)
        if not include_fdp:
            return iso, aniso, None
        ri, ra = terms.rows_iso, terms.rows_aniso
        imaginary = (
            xyz_i[ri],
            adp_i[ri],
            occ_i[ri],
            terms.f_double_prime_iso,
            B_i[ri],
            xyz_a[ra],
            u_a[ra],
            occ_a[ra],
            terms.f_double_prime_aniso,
            B_a[ra],
        )
        return iso, aniso, imaginary

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
        with the wavelength-dependent f' / f'' applied only to atoms above
        ``anomalous_threshold``. All three terms go through the same density and
        FFT, so each carries the atom's temperature factor and symmetry mates.
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

        Notes
        -----
        f' and f'' are not added to F afterwards: they are folded into the atoms'
        form factors before the density is built, so the one splat and FFT apply
        each atom's isotropic or anisotropic temperature factor and the space-group
        symmetry to them exactly as to f0.
        """
        self._check_forward_dtype(hkl)
        iso, aniso, imaginary = self.get_iso(), self.get_aniso(), None
        if apply_anomalous and self.wavelength is not None:
            iso, aniso, imaginary = self._add_anomalous_scattering(
                iso, aniso, include_fdp=bool(self.anomalous_bijvoet)
            )
        sf, _ = self.fft.compute_structure_factors(
            hkl, *iso, *aniso, apply_symmetry=True, imaginary=imaginary
        )

        if self.ctx.verbose > 2:
            assert torch.all(
                torch.isfinite(sf)
            ), "Non-finite values found while calculating fcalc."

        return sf

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

    def _subclass_kwargs(self) -> dict:
        """The grid, wavelength and Bijvoet settings a new instance must share."""
        return {
            "max_res": self.max_res,
            "gridsize": self.explicit_gridsize,
            "wavelength": self.wavelength,
            "anomalous_threshold": self.anomalous_threshold,
            "apply_bijvoet": bool(self.anomalous_bijvoet),
        }

    @classmethod
    def _pop_subclass_state(cls, state_dict: dict) -> dict:
        """Pop the FT settings :meth:`state_dict` wrote, as constructor kwargs.

        The ``radius_angstrom`` key of older checkpoints is dropped unused.
        """
        state_dict.pop("radius_angstrom", None)
        return {
            "max_res": state_dict.pop("max_res", 1.0),
            "gridsize": state_dict.pop("explicit_gridsize", None),
            "wavelength": state_dict.pop("wavelength", None),
            "anomalous_threshold": state_dict.pop("anomalous_threshold", 0.5),
        }

    def _restorable_entries(self, state_dict: dict) -> dict:
        """Register the scattering buffers and adopt a legacy stored grid size.

        Old checkpoints name the scattering buffers ``A`` / ``B`` rather than
        ``_A`` / ``_B``, and those written while the grid was stored state carry its
        size (``_fft.gridsize``, or a flat ``gridsize``). That size is adopted only
        when it differs from what the crystal and ``max_res`` give.
        """
        for old, new in (("A", "_A"), ("B", "_B")):
            if old in state_dict and new not in state_dict:
                state_dict[new] = state_dict.pop(old)
        for name in ("_A", "_B"):
            if state_dict.get(name) is not None and self.ctx.topology is not None:
                self.register_buffer(
                    name, torch.zeros_like(state_dict[name], device=self.device)
                )

        legacy = state_dict.pop("_fft.gridsize", None)
        if legacy is None:
            legacy = state_dict.pop("gridsize", None)
        state_dict.pop("_fft.voxel_size", None)
        state_dict.pop("voxel_size", None)
        if (
            legacy is not None
            and self.explicit_gridsize is None
            and self.ctx.crystal_key is not None
            and self.max_res is not None
        ):
            if isinstance(legacy, torch.Tensor):
                legacy = legacy.tolist()
            legacy = tuple(int(x) for x in legacy)
            if legacy != self.fft.compute_optimal_gridsize(self.max_res):
                self.explicit_gridsize = legacy
        return super()._restorable_entries(state_dict)
