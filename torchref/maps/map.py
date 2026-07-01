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

from torchref.base.reciprocal.grid_operations import place_on_grid
from torchref.io.cif import write_map
from torchref.symmetry.grid_utils import calculate_optimal_grid_size
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
        Type of map to compute. One of ``"2Fo-Fc"`` or ``"Fcalc"``.
        Default is ``"2Fo-Fc"``. Note ``"2Fo-Fc"`` is a *plain* 2Fo-Fc map
        (no figure-of-merit ``m`` and no sigma-A coefficient ``D``; i.e.
        ``m=1``, ``D=1``), not a likelihood-weighted 2mFo-DFc map.

    Attributes
    ----------
    data : ReflectionData
        The observed reflection data.
    model : ModelFT
        The model used to compute structure factors.
    gridsize : tuple of int or None
        Requested grid dimensions; ``None`` means auto-determined at
        ``calculate()`` time.
    map_type : str
        The configured map type (one of ``VALID_MAP_TYPES``).
    map_data : torch.Tensor or None
        The computed 3D real-space map, or ``None`` before ``calculate()``.
    device : torch.device
        Computation device.

    Methods
    -------
    calculate()
        Compute and return the 3D real-space map.
    write(filepath)
        Write the map to a CCP4 file (computing it first if needed).
    reset_cache()
        Discard the cached map so it is recomputed on next access.
    """

    VALID_MAP_TYPES = ("2Fo-Fc", "Fcalc")

    def __init__(
        self,
        data,
        model,
        gridsize: Optional[Tuple[int, int, int]] = None,
        map_type: str = "2Fo-Fc",
        device: Optional[torch.device] = None,
    ):
        if map_type not in self.VALID_MAP_TYPES:
            raise ValueError(
                f"map_type must be one of {self.VALID_MAP_TYPES}, got '{map_type}'"
            )
        self.device = resolve_device(data, model, device=device)
        self.data = data
        self.model = model
        self.gridsize = gridsize
        self.map_type = map_type
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
        cell_params = self.data.cell.data
        max_res = float(self.data.resolution.min())
        spacegroup = self.data.spacegroup.name
        return calculate_optimal_grid_size(cell_params, max_res, spacegroup)

    def _compute_map_coefficients(
        self, fobs: torch.Tensor, fcalc: torch.Tensor
    ) -> torch.Tensor:
        """Compute complex map coefficients.

        Parameters
        ----------
        fobs : torch.Tensor
            Observed amplitudes, shape (N,).
        fcalc : torch.Tensor
            Complex structure factors from model, shape (N,).

        Returns
        -------
        torch.Tensor
            Complex map coefficients, shape (N,).
        """
        if self.map_type == "Fcalc":
            return fcalc

        # 2Fo-Fc: (2*Fobs - |Fcalc|) * exp(i * phi_calc). Note this is a plain
        # 2Fo-Fc map: no figure-of-merit ``m`` weights Fobs and no sigma-A
        # coefficient ``D`` scales Fcalc (i.e. m=1, D=1), so it is not a true
        # likelihood-weighted 2mFo-DFc map.
        fcalc_amp = fcalc.abs()
        phi_calc = torch.angle(fcalc)
        return (2.0 * fobs - fcalc_amp) * torch.exp(1j * phi_calc)

    def calculate(self) -> torch.Tensor:
        """Compute the electron density map.

        Returns
        -------
        torch.Tensor
            3D real-space map tensor.
        """
        # Expand to P1 without Friedel mates (place_on_grid handles
        # Hermitian symmetry via enforce_hermitian=True)
        data_p1 = self.data.expand_to_p1(include_friedel=False)
        hkl_p1, fobs_p1, _, _ = data_p1.data_indexed()

        # Compute Fcalc for P1-expanded hkl
        fcalc_p1 = self.model.get_structure_factor(hkl_p1)

        # Compute map coefficients
        coefficients_p1 = self._compute_map_coefficients(fobs_p1, fcalc_p1)

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

        return self._map

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
