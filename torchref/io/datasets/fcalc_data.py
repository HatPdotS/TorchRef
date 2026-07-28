"""
FcalcDataset - Dataset for storing calculated structure factors.

This module provides a lightweight container for calculated structure factors
(Fcalc) with support for:
- Creation from cell/spacegroup/resolution
- Complex Fcalc with amplitude/phase decomposition
- MTZ file export
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch

from torchref.config import get_default_device, get_float_dtype, normalize_device
from torchref.symmetry import Cell, SpaceGroup, SpaceGroupLike

from .base import CrystalDataset


@dataclass
class FcalcDataset(CrystalDataset):
    """
    Dataset for storing calculated structure factors.

    Provides a lightweight container for Fcalc values with:
    - Cell and spacegroup information (using torchref.symmetry types)
    - HKL indices and resolution
    - Complex Fcalc with amplitude/phase decomposition
    - MTZ export capability

    This class inherits from CrystalDataset and overrides the spacegroup
    field to store torchref.symmetry.SpaceGroup instead of gemmi.SpaceGroup.

    Parameters
    ----------
    hkl : torch.Tensor, optional
        Miller indices of shape (N, 3).
    resolution : torch.Tensor, optional
        Resolution per reflection of shape (N,).
    cell : Cell, optional
        Unit cell object.
    spacegroup : SpaceGroup, optional
        Space group object (torchref.symmetry.SpaceGroup).
    fcalc : torch.Tensor, optional
        Complex structure factors of shape (N,).
    fcalc_amp : torch.Tensor, optional
        Amplitudes |Fcalc| of shape (N,).
    fcalc_phase : torch.Tensor, optional
        Phases in radians of shape (N,).
    device : torch.device
        Device for tensors.

    Examples
    --------
    Create from cell and resolution::

        from torchref.io.datasets import FcalcDataset

        dataset = FcalcDataset.from_cell_and_resolution(
            cell=[50.0, 60.0, 70.0, 90.0, 90.0, 90.0],
            spacegroup='P212121',
            d_min=2.0,
        )

        # Set Fcalc values (complex tensor)
        fcalc = torch.randn(len(dataset), dtype=torch.complex64)
        dataset.set_fcalc(fcalc)

        # Write to MTZ
        dataset.write_mtz('output.mtz')
    """

    # Override spacegroup to use torchref.symmetry.SpaceGroup (not gemmi)
    spacegroup: Optional[SpaceGroup] = None  # type: ignore[assignment]

    # Fcalc-specific fields
    fcalc: Optional[torch.Tensor] = None  # Complex (N,)
    fcalc_amp: Optional[torch.Tensor] = None  # |Fcalc| (N,)
    fcalc_phase: Optional[torch.Tensor] = None  # Phase in radians (N,)

    @staticmethod
    def from_cell_and_resolution(
        cell: Union[torch.Tensor, List[float], Cell],
        spacegroup: SpaceGroupLike,
        d_min: float = 2.0,
        d_max: Optional[float] = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> "FcalcDataset":
        """
        Create FcalcDataset with HKL generated to given resolution.

        Parameters
        ----------
        cell : torch.Tensor, list, or Cell
            Unit cell [a, b, c, alpha, beta, gamma] or Cell object.
        spacegroup : SpaceGroupLike
            Space group (str, int, gemmi.SpaceGroup, or torchref.symmetry.SpaceGroup).
        d_min : float, optional
            High resolution limit in Angstroms. Default is 2.0.
        d_max : float, optional
            Low resolution limit in Angstroms. If provided, reflections
            with d-spacing > d_max are removed.
        device : torch.device, optional
            Target device. If None, defaults to ``get_default_device()``.
        dtype : torch.dtype, optional
            Float dtype for tensors. If None, defaults to ``get_float_dtype()``.

        Returns
        -------
        FcalcDataset
            New dataset with HKL and resolution populated.

        Examples
        --------
        ::

            from torchref.symmetry import Cell, SpaceGroup

            cell = Cell([50.0, 60.0, 70.0, 90.0, 90.0, 90.0])
            sg = SpaceGroup('P212121')
            dataset = FcalcDataset.from_cell_and_resolution(
                cell=cell, spacegroup=sg, d_min=2.0,
            )
            print(f"Generated {len(dataset)} reflections")
        """
        import gemmi

        from torchref.base.reciprocal import get_d_spacing

        # ``Cell`` is the unit-cell carrier: when one is handed in and no device
        # is requested, follow it rather than the global default -- otherwise a
        # CPU-default host silently relocates the caller's cell.
        if device is None and isinstance(cell, Cell):
            device = cell.device
        device = normalize_device(device)
        if dtype is None:
            dtype = get_float_dtype()

        # Handle Cell input - convert to Cell object if needed
        if isinstance(cell, Cell):
            # ``Cell.to`` is in-place, so this moves the *caller's* object when
            # an explicit device disagrees with it. That is the documented
            # resolve_device contract; the branch above avoids it entirely when
            # no device was requested.
            cell_obj = cell.to(device=device)
            cell_tensor = cell_obj.data
        else:
            if not isinstance(cell, torch.Tensor):
                cell_tensor = torch.tensor(cell, dtype=dtype, device=device)
            else:
                cell_tensor = cell.to(device=device, dtype=dtype)
            cell_obj = Cell(cell_tensor, dtype=dtype, device=device)

        # Handle spacegroup - normalize to torchref.symmetry.SpaceGroup
        if isinstance(spacegroup, SpaceGroup):
            sg_obj = spacegroup
        else:
            sg_obj = SpaceGroup(spacegroup)  # Normalize any input

        # Generate HKL indices using gemmi (respects space group symmetry)
        # This generates only unique reflections in the asymmetric unit
        cell_list = cell_tensor.cpu().tolist()
        gemmi_cell = gemmi.UnitCell(
            cell_list[0],
            cell_list[1],
            cell_list[2],
            cell_list[3],
            cell_list[4],
            cell_list[5],
        )
        gemmi_sg = sg_obj._gemmi  # Get underlying gemmi.SpaceGroup

        # make_miller_array returns unique HKL for the asymmetric unit
        hkl_list = gemmi.make_miller_array(gemmi_cell, gemmi_sg, d_min)
        hkl = torch.tensor(hkl_list, dtype=torch.int32, device=device)

        # Calculate resolution
        resolution = get_d_spacing(hkl.float(), cell_tensor)

        # Apply low resolution cutoff if requested
        if d_max is not None:
            mask = resolution <= d_max
            hkl = hkl[mask]
            resolution = resolution[mask]

        print(f"Generated dataset with {len(hkl)} reflections.")

        return FcalcDataset(
            hkl=hkl,
            resolution=resolution,
            cell=cell_obj,
            spacegroup=sg_obj,
            device=device,
        )

    def set_fcalc(self, fcalc: torch.Tensor) -> None:
        """
        Assign complex Fcalc values.

        Automatically computes amplitude and phase from complex values.

        Parameters
        ----------
        fcalc : torch.Tensor
            Complex structure factors with shape (N,).

        Raises
        ------
        ValueError
            If fcalc length doesn't match HKL length.

        Examples
        --------
        ::

            # Create complex Fcalc values
            fcalc = torch.randn(len(dataset), dtype=torch.complex64)
            dataset.set_fcalc(fcalc)

            print(dataset.fcalc_amp[:5])   # Amplitudes
            print(dataset.fcalc_phase[:5]) # Phases in radians
        """
        if self.hkl is None:
            raise ValueError("HKL not set. Cannot assign Fcalc without HKL indices.")

        if fcalc.shape[0] != len(self.hkl):
            raise ValueError(
                f"Fcalc length {fcalc.shape[0]} != HKL length {len(self.hkl)}"
            )

        self.fcalc = fcalc.to(device=self.device)
        self.fcalc_amp = torch.abs(fcalc).to(device=self.device)
        self.fcalc_phase = torch.angle(fcalc).to(device=self.device)

    def write_mtz(self, filepath: str) -> None:
        """
        Write Fcalc to MTZ file.

        Parameters
        ----------
        filepath : str
            Output MTZ filename.

        Raises
        ------
        ValueError
            If no Fcalc values have been set.

        Examples
        --------
        ::

            dataset.set_fcalc(fcalc_values)
            dataset.write_mtz('calculated.mtz')
        """
        from torchref.io import mtz

        if self.fcalc is None:
            raise ValueError("No Fcalc values set. Call set_fcalc() first.")

        if self.hkl is None:
            raise ValueError("No HKL indices set.")

        if self.cell is None:
            raise ValueError("No cell set.")

        if self.spacegroup is None:
            raise ValueError("No spacegroup set.")

        # Build DataFrame
        hkl_np = self.hkl.cpu().numpy()
        df = pd.DataFrame(
            {
                "H": hkl_np[:, 0],
                "K": hkl_np[:, 1],
                "L": hkl_np[:, 2],
                "F-model": self.fcalc_amp.cpu().numpy(),
                "PH-model": torch.rad2deg(self.fcalc_phase).cpu().numpy(),
            }
        )

        # Write using existing mtz.write() - pass SpaceGroup wrapper
        mtz.write(df, self.cell.data, self.spacegroup, filepath)

    def write_mtz_as_fobs(
        self,
        filepath: str,
        sigma_frac: float = 0.05,
        f_column: str = "F-obs",
        sigf_column: str = "SIGF-obs",
        phase_column: str = "PHIF-model",
    ) -> None:
        """
        Write Fcalc to MTZ as if it were observed data (F-obs columns).

        Useful for creating simulated "experimental" MTZ files that can be
        read back by ReflectionData.load_mtz() as observed amplitudes.

        Parameters
        ----------
        filepath : str
            Output MTZ filename.
        sigma_frac : float, optional
            Sigma as a fraction of |F|. Default is 0.05 (5%).
        f_column : str, optional
            Column name for amplitudes. Default is 'F-obs'.
        sigf_column : str, optional
            Column name for sigma. Default is 'SIGF-obs'.
        phase_column : str, optional
            Column name for model phases. Default is 'PHIF-model'.

        Examples
        --------
        ::

            dataset.set_fcalc(fcalc_values)
            dataset.write_mtz_as_fobs('simulated_obs.mtz', sigma_frac=0.05)
        """
        from torchref.io import mtz

        if self.fcalc_amp is None:
            raise ValueError("No Fcalc values set. Call set_fcalc() first.")
        if self.hkl is None:
            raise ValueError("No HKL indices set.")
        if self.cell is None:
            raise ValueError("No cell set.")
        if self.spacegroup is None:
            raise ValueError("No spacegroup set.")

        amp = self.fcalc_amp.cpu().numpy()
        sigma = amp * sigma_frac

        hkl_np = self.hkl.cpu().numpy()
        columns = {
            "H": hkl_np[:, 0],
            "K": hkl_np[:, 1],
            "L": hkl_np[:, 2],
            f_column: amp,
            sigf_column: sigma,
        }
        if self.fcalc_phase is not None:
            columns[phase_column] = torch.rad2deg(self.fcalc_phase).cpu().numpy()

        df = pd.DataFrame(columns)
        mtz.write(df, self.cell.data, self.spacegroup, filepath)

    # ========== SERIALIZATION OVERRIDES ==========

    def _get_state(self) -> Dict[str, Any]:
        """
        Get serializable state, handling SpaceGroup wrapper.

        Returns
        -------
        Dict[str, Any]
            State dictionary with all tensor and metadata fields.
        """
        state = super()._get_state()
        # SpaceGroup is stored as torchref.symmetry.SpaceGroup, serialize as string
        if self.spacegroup is not None:
            state["spacegroup"] = self.spacegroup.hm  # Use hm property
        return state

    @classmethod
    def _from_state(cls, state: Dict[str, Any], device=None) -> "FcalcDataset":
        """
        Reconstruct from state, creating SpaceGroup wrapper.

        Parameters
        ----------
        state : Dict[str, Any]
            State dictionary from _get_state().
        device : str
            Device to load tensors onto.

        Returns
        -------
        FcalcDataset
            Reconstructed dataset.
        """
        from torchref.utils.utils import TensorMasks

        device = normalize_device(device)

        # Extract masks before creating object
        masks_state = state.pop("masks", {})

        # Convert device string back to torch.device
        if "device" in state:
            state["device"] = torch.device(state["device"])

        # Convert spacegroup string to SpaceGroup object
        if "spacegroup" in state and state["spacegroup"] is not None:
            if isinstance(state["spacegroup"], str):
                state["spacegroup"] = SpaceGroup(state["spacegroup"])

        # Convert cell tensor back to Cell object
        if "cell" in state and state["cell"] is not None:
            if isinstance(state["cell"], torch.Tensor):
                state["cell"] = Cell(
                    state["cell"], dtype=get_float_dtype(), device=device
                )

        # Create object with remaining state
        obj = cls(**state)

        # Restore masks
        if masks_state:
            obj.masks = TensorMasks(data=masks_state, device=device)

        return obj.to(device)

    # ========== UTILITY METHODS ==========

    def __repr__(self) -> str:
        """String representation of dataset."""
        n_refl = len(self)
        sg = self.spacegroup.name if self.spacegroup else "unknown"
        has_fcalc = "yes" if self.fcalc is not None else "no"
        return (
            f"{self.__class__.__name__}(n_reflections={n_refl}, "
            f"spacegroup='{sg}', fcalc={has_fcalc}, device={self.device})"
        )

    @property
    def spacegroup_name(self) -> Optional[str]:
        """Get space group name as string (short form, e.g., 'P212121')."""
        if self.spacegroup is None:
            return None
        return self.spacegroup.name

    @property
    def spacegroup_hm(self) -> Optional[str]:
        """Get space group Hermann-Mauguin name with spaces (e.g., 'P 21 21 21')."""
        if self.spacegroup is None:
            return None
        return self.spacegroup.hm

    @property
    def spacegroup_number(self) -> Optional[int]:
        """Get space group number (1-230)."""
        if self.spacegroup is None:
            return None
        return self.spacegroup.number
