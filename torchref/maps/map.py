"""
Base Map class for crystallographic electron density map computation.

Supports 2Fo-Fc and Fcalc map types. Computes maps via FFT
of map coefficients placed on a reciprocal-space grid. The "2Fo-Fc"
map is a plain 2Fo-Fc map (no figure-of-merit ``m`` and no sigma-A
coefficient ``D``; i.e. ``m=1``, ``D=1``), not a likelihood-weighted
2mFo-DFc map.

FFT convention: ρ(r) = (1/N) * sum_h F(h) * exp(-2πi h·r)
This corresponds to torch.fft.fftn with ``norm="forward"`` (forward DFT
with exp(-2πi) kernel and a 1/N normalization, N = number of grid
points). The map scale is therefore in normalized units. Hermitian
symmetry F(-h) = F*(h) is enforced by place_on_grid to ensure a
real-valued map.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from torchref.base.fourier.coefficients import map_coefficients
from torchref.base.reciprocal.grid_operations import place_on_grid
from torchref.io.cif import write_map
from torchref.scaling.scaler import Scaler
from torchref.symmetry import SpaceGroup
from torchref.utils.device_mixin import DeviceMixin
from torchref.utils.device_resolution import resolve_device


class Map(DeviceMixin):
    """Crystallographic electron density map.

    Parameters
    ----------
    data : ReflectionData
        Observed reflection data with amplitudes, hkl, cell, and spacegroup.
    model : ModelFT
        Model for computing Fcalc (structure factors).
    gridsize : tuple of int, optional
        Grid dimensions (nx, ny, nz). If None, determined automatically
        from cell parameters and resolution.
    map_type : str, optional
        ``"2Fo-Fc"`` (default; unweighted, see :mod:`torchref.maps.map`) or
        ``"Fcalc"``.
    device : torch.device, optional
        Computation device. ``data`` and ``model`` are moved onto it in place; if
        None, ``model`` is moved onto ``data``'s device.
    units : str, optional
        ``"normalized"`` (default) keeps the FFT's ``1/N`` normalisation;
        ``"electrons"`` gives ``(1/V) sum_h F(h) exp(-2 pi i h.x)``, electrons per
        cubic Angstrom, which is meaningful only when ``data.F`` is on the absolute
        scale: every map type is on ``data.F``'s scale.
    scaler : Scaler, optional
        A fitted :class:`~torchref.scaling.scaler.Scaler` for ``(data, model)``, e.g. a
        refinement's ``scaler``. Every map type uses ``scaler(F_calc)``, with bulk
        solvent and anisotropic scale, so F_calc is on ``data.F``'s scale. If None,
        :meth:`calculate` fits the standard one (``initialize()`` then
        ``refine_lbfgs()``) on each call.

    Attributes
    ----------
    map_data : torch.Tensor or None
        The computed 3D real-space map, or ``None`` before ``calculate()``.
    """

    VALID_MAP_TYPES = ("2Fo-Fc", "Fcalc")
    VALID_UNITS = ("normalized", "electrons")

    def __init__(
        self,
        data,
        model,
        gridsize: Optional[Tuple[int, int, int]] = None,
        map_type: str = "2Fo-Fc",
        device: Optional[torch.device] = None,
        units: str = "normalized",
        scaler: Optional[Scaler] = None,
    ):
        if map_type not in self.VALID_MAP_TYPES:
            raise ValueError(
                f"map_type must be one of {self.VALID_MAP_TYPES}, got '{map_type}'"
            )
        if units not in self.VALID_UNITS:
            raise ValueError(f"units must be one of {self.VALID_UNITS}, got '{units}'")
        self.units = units
        self.device = resolve_device(data, model, device=device)
        self.data = data
        self.model = model
        self.gridsize = gridsize
        self.map_type = map_type
        self.scaler = scaler
        self._map: Optional[torch.Tensor] = None

    def reset_cache(self) -> None:
        """Invalidate the cached map tensor; recomputed on next access."""
        self._map = None

    @property
    def map_data(self) -> Optional[torch.Tensor]:
        """The computed 3D real-space map, or None if not yet calculated."""
        return self._map

    def _determine_gridsize(self) -> Tuple[int, int, int]:
        """Determine optimal grid size from cell, resolution, and spacegroup."""
        max_res = float(self.data.resolution.min())
        return self.data.spacegroup.optimal_grid_size(self.data.cell, max_res)

    def _compute_map_coefficients(
        self, fobs: torch.Tensor, fcalc: torch.Tensor
    ) -> torch.Tensor:
        """Compute complex map coefficients.

        Parameters
        ----------
        fobs : torch.Tensor
            Observed amplitudes, shape (N,).
        fcalc : torch.Tensor
            Complex model structure factors on ``fobs``'s scale, shape (N,).

        Returns
        -------
        torch.Tensor
            Complex map coefficients, shape (N,).
        """
        if self.map_type == "Fcalc":
            return fcalc

        return map_coefficients(fobs, fcalc)[0]

    def calculate(self) -> torch.Tensor:
        """Compute the electron density map.

        Returns
        -------
        torch.Tensor
            3D real-space map tensor.
        """
        # One amplitude per reflection: the Hermitian placement would otherwise
        # count every measured Bijvoet pair twice.
        valid = self.data.masks()
        rows = self.data.bijvoet_representatives(valid)
        fobs = self.data.bijvoet_mean(self.data.F, valid)[rows]
        coefficients = self._compute_map_coefficients(fobs, self._scaled_fcalc()[rows])

        # The coefficients exist on the data's own rows, so they are expanded with
        # their phase shifts; place_on_grid adds the Friedel half.
        sg = self.data.spacegroup or SpaceGroup("P1", device=self.data.device)
        hkl_p1, idx, shifts = sg.expand_hkl(self.data.hkl[rows], include_friedel=False)
        coefficients_p1 = coefficients[idx] * torch.exp(1j * shifts)

        # Determine grid size
        if self.gridsize is not None:
            gridsize = self.gridsize
        else:
            gridsize = self._determine_gridsize()

        # Place coefficients on reciprocal-space grid (adds F*(-h) for
        # Hermitian symmetry, ensuring real-valued output)
        grid = place_on_grid(hkl_p1, coefficients_p1, gridsize, enforce_hermitian=True)

        # FFT to real space: ρ(r) = (1/N) * sum_h F(h) * exp(-2πi h·r)
        # (norm="forward" applies the 1/N normalization, N = grid points)
        self._map = torch.fft.fftn(grid, dim=(0, 1, 2), norm="forward").real
        self._map = self._to_units(self._map)

        return self._map

    def _scaled_fcalc(self) -> torch.Tensor:
        """``scaler(F_calc)``, shape (N,), row-aligned with ``data.hkl``.

        The model is evaluated with ``cached=False`` so this no-grad pass leaves
        no detached tensor in its forward cache.
        """
        scaler = self.scaler
        if scaler is None:
            scaler = Scaler(self.model, self.data, verbose=0, device=self.device)
            scaler.initialize()
            scaler.refine_lbfgs(verbose=False)
        with torch.no_grad():
            return scaler(self.data.structure_factors(self.model, cached=False))

    def _to_units(self, real_map: torch.Tensor) -> torch.Tensor:
        """Rescale a ``1/N``-normalised FFT map to the configured units."""
        if self.units == "electrons":
            volume = self.data.cell.volume.to(real_map.dtype)
            return real_map * (real_map.numel() / volume)
        return real_map

    def write(self, filepath: str) -> int:
        """Write the map to a CCP4 file.

        Automatically computes the map if it hasn't been calculated yet.

        Parameters
        ----------
        filepath : str
            Output CCP4 map file path.

        Returns
        -------
        int
            1 on success.
        """
        if self._map is None:
            self.calculate()

        cell = self.data.cell.data
        spacegroup = self.data.spacegroup.name
        return write_map(self._map, cell, filepath, spacegroup=spacegroup)
