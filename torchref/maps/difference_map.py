"""
Isomorphous difference map from two datasets.

Computes a difference Fourier map using dF = F_data - F_reference with
phases from a model, after scaling both datasets to a common reference.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from torchref.base.reciprocal.grid_operations import place_on_grid
from torchref.io.datasets.collection import DatasetCollection
from torchref.maps.map import Map
from torchref.symmetry import SpaceGroup
from torchref.utils.device_resolution import resolve_device


class DifferenceMap(Map):
    """Isomorphous difference map between two datasets.

    Scales both datasets to a common reference using ``DatasetCollection``,
    then computes difference Fourier coefficients:
    ``dF * exp(i * phi_calc)`` where ``dF = F_data - F_reference``.

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
    device : torch.device, optional
        Computation device. ``data``, ``data_reference`` and ``model`` are moved onto
        it in place; if None, onto ``data``'s device.
    units : str, optional
        ``"normalized"`` (default) or ``"electrons"``, as for :class:`Map`; electrons
        per cubic Angstrom only when ``scale`` puts the differences on the absolute
        scale.
    scale : torch.Tensor, optional
        Per-reflection observed-to-model scale, shape (N,); the differences are
        divided by it, which puts them in electrons. Row-aligned with the
        collection's reflections, the sorted canonical (ASU) union of both
        datasets' indices that the map exposes as ``data_reference.hkl`` -- the
        reference dataset's own order only when both share one reflection list.

    Raises
    ------
    ValueError
        If ``scale`` does not have one row per union reflection. A scale of the
        right length in another row order is not detected.

    Attributes
    ----------
    data_reference, data_perturbed : ScaledDataset
        The jointly scaled copies of the two inputs on the collection's union
        reflection list; the input datasets themselves stay unscaled.
    map_data : torch.Tensor or None
        The computed 3D real-space difference map, or ``None`` before
        ``calculate()``.
    map_type : str
        Inherited from :class:`Map`; set to ``"Fcalc"`` as a placeholder
        because ``calculate()`` is overridden and does not use it.
    """

    def __init__(
        self,
        data,
        data_reference,
        model,
        gridsize=None,
        device: Optional[torch.device] = None,
        units: str = "normalized",
        scale: Optional[torch.Tensor] = None,
    ):
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
        self.data_reference = self._collection["reference"]
        self.data_perturbed = self._collection["perturbed"]
        n_union = len(self.data_reference.hkl)
        if scale is not None and len(scale) != n_union:
            raise ValueError(
                f"scale has {len(scale)} rows, but it must be row-aligned with the "
                f"{n_union} reflections of the union of both datasets "
                "(data_reference.hkl of the built map)."
            )

        # Use reference dataset for cell, spacegroup, hkl via super().__init__
        super().__init__(
            data=self.data_reference,
            model=model,
            gridsize=gridsize,
            map_type="Fcalc",  # placeholder, calculate() is overridden
            device=resolved,
            units=units,
        )
        self.scale = scale

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
        delta_f = F_pert_scaled - F_ref_scaled
        if self.scale is not None:
            delta_f = delta_f / self.scale.to(delta_f)
        # One difference per reflection (Bijvoet mates averaged): the Hermitian
        # placement below adds each conjugate at -h itself.
        rows = self.data_reference.bijvoet_representatives(mask_combined)
        delta_f = self.data_reference.bijvoet_mean(delta_f, mask_combined)[rows]
        hkl_asu = self.data_reference.hkl[rows]

        # delta_f is derived per reflection, which expand_to_p1 cannot carry: expand
        # the indices.
        sg =self.data_reference.spacegroup or SpaceGroup("P1", device=hkl_asu.device)
        hkl_p1, orig_idx, _ = sg.expand_hkl(
            hkl_asu,
            include_friedel=False, remove_absences=True,
            device=hkl_asu.device,
        )

        # Map scaled amplitudes to P1 (amplitudes are invariant under symmetry)
        delta_f_p1 = delta_f[orig_idx]

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
        self._map = self._to_units(
            torch.fft.fftn(grid, dim=(0, 1, 2), norm="forward").real
        )

        return self._map
