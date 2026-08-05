"""
:class:`FcalcDataset` -- container for calculated structure factors.

Generates its own HKL set from cell/spacegroup/resolution, decomposes complex
Fcalc into amplitude and phase, and exports to MTZ either as model columns or
as pseudo-observations.
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

    Unlike :class:`CrystalDataset`, ``spacegroup`` here holds a
    ``torchref.symmetry.SpaceGroup`` object, not a string.

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
    fcalc_amp, fcalc_phase : torch.Tensor, optional
        ``|Fcalc|`` and its phase in radians, shape (N,); normally derived by
        :meth:`set_fcalc` rather than passed.
    device : torch.device
        Device for tensors.
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
            Unit cell [a, b, c, alpha, beta, gamma] or Cell object. A ``Cell``
            is moved IN PLACE if ``device`` disagrees with it.
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
            New dataset with HKL (unique ASU reflections) and resolution set.
        """
        import gemmi

        from torchref.base.reciprocal import get_d_spacing

        # With a Cell in hand and no device requested, follow the Cell rather
        # than the global default, or a CPU-default host silently relocates it.
        if device is None and isinstance(cell, Cell):
            device = cell.device
        device = normalize_device(device)
        if dtype is None:
            dtype = get_float_dtype()

        if isinstance(cell, Cell):
            # ``Cell.to`` is in-place, so an explicit disagreeing device moves
            # the *caller's* object (the resolve_device contract).
            cell_obj = cell.to(device=device)
            cell_tensor = cell_obj.data
        else:
            if not isinstance(cell, torch.Tensor):
                cell_tensor = torch.tensor(cell, dtype=dtype, device=device)
            else:
                cell_tensor = cell.to(device=device, dtype=dtype)
            cell_obj = Cell(cell_tensor, dtype=dtype, device=device)

        if isinstance(spacegroup, SpaceGroup):
            sg_obj = spacegroup
        else:
            sg_obj = SpaceGroup(spacegroup)

        cell_list = cell_tensor.cpu().tolist()
        gemmi_cell = gemmi.UnitCell(
            cell_list[0],
            cell_list[1],
            cell_list[2],
            cell_list[3],
            cell_list[4],
            cell_list[5],
        )
        gemmi_sg = sg_obj._gemmi

        # make_miller_array returns unique HKL for the asymmetric unit only.
        hkl_list = gemmi.make_miller_array(gemmi_cell, gemmi_sg, d_min)
        hkl = torch.tensor(hkl_list, dtype=torch.int32, device=device)

        resolution = get_d_spacing(hkl.float(), cell_tensor)

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
        Assign complex Fcalc, also setting ``fcalc_amp`` and ``fcalc_phase``.

        Parameters
        ----------
        fcalc : torch.Tensor
            Complex structure factors with shape (N,).

        Raises
        ------
        ValueError
            If HKL is unset or ``fcalc`` has a different length.
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
        Write Fcalc to MTZ as ``F-model`` / ``PH-model`` (phase in degrees).

        Parameters
        ----------
        filepath : str
            Output MTZ filename.

        Raises
        ------
        ValueError
            If Fcalc, HKL, cell or spacegroup is unset.
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
        Write Fcalc to MTZ as pseudo-observations, readable back by
        :meth:`ReflectionData.load_mtz` as if measured.

        Sigmas are fabricated as ``sigma_frac * |F|``, not measured.

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

        Raises
        ------
        ValueError
            If Fcalc, HKL, cell or spacegroup is unset.
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
        """As the base, but ``spacegroup`` is flattened via its ``hm`` symbol."""
        state = super()._get_state()
        if self.spacegroup is not None:
            state["spacegroup"] = self.spacegroup.hm
        return state

    @classmethod
    def _from_state(cls, state: Dict[str, Any], device=None) -> "FcalcDataset":
        """Rebuild from a :meth:`_get_state` dict, rewrapping the H-M string as a
        ``SpaceGroup``. Pops ``"masks"``, so ``state`` is mutated.
        """
        from torchref.utils.utils import TensorMasks

        device = normalize_device(device)

        masks_state = state.pop("masks", {})
        state = cls._drop_stale_state_keys(state)

        if "device" in state:
            state["device"] = torch.device(state["device"])

        if "spacegroup" in state and state["spacegroup"] is not None:
            if isinstance(state["spacegroup"], str):
                state["spacegroup"] = SpaceGroup(state["spacegroup"])

        if "cell" in state and state["cell"] is not None:
            if isinstance(state["cell"], torch.Tensor):
                state["cell"] = Cell(
                    state["cell"], dtype=get_float_dtype(), device=device
                )

        obj = cls(**state)

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
