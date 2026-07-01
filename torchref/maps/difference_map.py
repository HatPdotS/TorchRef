"""
Isomorphous difference map from two datasets.

Computes a difference Fourier map using DF = F_data - F_reference with
phases from a model, after scaling both datasets to a common reference.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from torchref.base.reciprocal.grid_operations import place_on_grid
from torchref.config import get_default_device
from torchref.io.datasets.collection import DatasetCollection
from torchref.maps.map import Map
from torchref.symmetry.reciprocal_symmetry import expand_hkl
from torchref.utils.device_resolution import resolve_device


class DifferenceMap(Map):
    """Isomorphous difference map between two datasets.

    Scales both datasets to a common reference using ``DatasetCollection``,
    then computes difference Fourier coefficients:
    ``DF * exp(i * phi_calc)`` where ``DF = F_data - F_reference``.

    Parameters
    ----------
    data : ReflectionData
        Reflection data for the perturbed state (e.g., light, derivative).
    data_reference : ReflectionData
        Reflection data for the reference state (e.g., dark, native).
    model : ModelFT
        Model for computing phases.
    gridsize : tuple of int, optional
        Grid dimensions (nx, ny, nz). If None, determined automatically.

    Attributes
    ----------
    data_reference : ReflectionData
        Reflection data for the reference state (e.g., dark, native).
    data_perturbed : ReflectionData
        Reflection data for the perturbed state (e.g., light, derivative).
    map_data : torch.Tensor or None
        The computed 3D real-space difference map, or ``None`` before
        ``calculate()``.
    map_type : str
        Inherited from :class:`Map`; set to ``"Fcalc"`` as a placeholder
        because ``calculate()`` is overridden and does not use it.

    Methods
    -------
    calculate()
        Compute and return the 3D real-space isomorphous difference map.
    write(filepath)
        Inherited from :class:`Map`; write the map to a CCP4 file.
    reset_cache()
        Inherited from :class:`Map`; discard the cached map.

    Notes
    -----
    The FFT uses ``torch.fft.fftn`` with ``norm="forward"`` (a 1/N
    normalization), so the difference-map scale is in normalized units
    (see :mod:`torchref.maps.map`).
    """

    def __init__(self, data, data_reference, model, gridsize=None,
                 device: Optional[torch.device] = None):
        # Pin all three inputs onto one device before constructing the
        # DatasetCollection / super().__init__ — both consume tensors
        # from data.hkl / model and would otherwise inherit whichever
        # device they happened to land on.
        resolved = resolve_device(data, data_reference, model, device=device)
        self.data_reference = data_reference
        self.data_perturbed = data

        # Build collection and scale
        self._collection = DatasetCollection(verbose=0, device=str(resolved))
        self._collection.add_dataset(
            "reference", data_reference, set_as_reference=True
        )
        self._collection.add_dataset("perturbed", data)
        self._collection.scale()

        # Use reference dataset for cell, spacegroup, hkl via super().__init__
        super().__init__(
            data=data_reference,
            model=model,
            gridsize=gridsize,
            map_type="Fcalc",  # placeholder, calculate() is overridden
            device=resolved,
        )

    def calculate(self) -> torch.Tensor:
        """Compute the isomorphous difference map.

        Returns
        -------
        torch.Tensor
            3D real-space difference map tensor.
        """
        # Get scaled amplitudes (applies scale + anisotropy from scale())
        F_ref_scaled, _ = self.data_reference.get_corrected_data()
        F_pert_scaled, _ = self.data_perturbed.get_corrected_data()

        # Combined mask: only use reflections valid in both datasets
        mask_combined = self.data_reference.masks() & self.data_perturbed.masks()
        hkl_asu = self.data_reference.hkl[mask_combined]
        fobs_ref = F_ref_scaled[mask_combined]
        fobs_pert = F_pert_scaled[mask_combined]

        # Expand to P1 without Friedel mates (expand_to_p1() would reset
        # scaling, so expand manually via expand_hkl)
        sg = self.data_reference.spacegroup or "P1"
        hkl_p1, orig_idx, _ = expand_hkl(
            hkl_asu, sg,
            include_friedel=False, remove_absences=True,
            device=hkl_asu.device,
        )

        # Map scaled amplitudes to P1 (amplitudes are invariant under symmetry)
        fobs_ref_p1 = fobs_ref[orig_idx]
        fobs_pert_p1 = fobs_pert[orig_idx]
        delta_f_p1 = fobs_pert_p1 - fobs_ref_p1

        # Compute Fcalc for P1 hkl (for phases)
        fcalc_p1 = self.model.get_structure_factor(hkl_p1)
        phi_calc = torch.angle(fcalc_p1)

        # Difference Fourier coefficients: delta_f * exp(i * phi_calc)
        coefficients_p1 = delta_f_p1 * torch.exp(1j * phi_calc)

        # Determine grid size
        if self.gridsize is not None:
            gridsize = self.gridsize
        else:
            gridsize = self._determine_gridsize()

        # Place on grid with Hermitian enforcement and FFT to real space
        grid = place_on_grid(
            hkl_p1, coefficients_p1, gridsize, enforce_hermitian=True
        )
        self._map = torch.fft.fftn(grid, dim=(0, 1, 2), norm="forward").real

        return self._map
