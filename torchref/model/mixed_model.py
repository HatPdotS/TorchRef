"""
Mixed Model for Time-Resolved Crystallography.

This module provides a MixedModel class that combines multiple ModelFT objects
with learnable population fractions, enabling refinement of time-resolved
crystallographic data with multiple conformational states.
"""

from typing import TYPE_CHECKING, List, Optional

import torch
from torch import nn

from torchref.utils.device_mixin import DeviceMovementMixin

if TYPE_CHECKING:
    from torchref.model.model_ft import ModelFT


class MixedModel(DeviceMovementMixin, nn.Module):
    """
    Model wrapper combining N ModelFT objects with learnable fractions.

    Computes: F_mixed = Σ w_i * F_i
    where w_i are learnable weights constrained to sum to 1 via softmax.

    This is useful for time-resolved crystallography where the crystal contains
    a mixture of conformational states (e.g., dark and light states) with
    unknown or refinable population fractions.

    Parameters
    ----------
    models : List[ModelFT]
        List of ModelFT objects to combine. All models must have compatible
        cell parameters and space groups.
    initial_fractions : List[float], optional
        Initial population fractions for each model. Must sum to 1.0.
        If None, equal fractions are used (1/N for each model).
    frozen_fractions : bool, optional
        If True, fractions are not updated during optimization.
        Default is False.
    verbose : int, optional
        Verbosity level. Default is 0.

    Attributes
    ----------
    models : nn.ModuleList
        Constituent ModelFT objects (proper submodule registration).
    fraction_params : nn.Parameter
        Raw parameters for fraction computation (softmax applied).

    Examples
    --------
    Create a mixed model with two states::

        model_dark = ModelFT().load_pdb('dark.pdb')
        model_light = ModelFT().load_pdb('light.pdb')

        # 70% dark, 30% light
        mixed = MixedModel([model_dark, model_light], initial_fractions=[0.7, 0.3])

        # Compute mixed structure factors
        F_mixed = mixed(hkl)

        # Get current fractions
        print(mixed.fractions)  # tensor([0.7, 0.3])
    """

    def __init__(
        self,
        models: List["ModelFT"],
        initial_fractions: Optional[List[float]] = None,
        frozen_fractions: bool = False,
        verbose: int = 0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize MixedModel.

        Parameters
        ----------
        models : List[ModelFT]
            List of ModelFT objects to combine.
        initial_fractions : List[float], optional
            Initial population fractions. Must sum to 1.0.
        frozen_fractions : bool, optional
            If True, fractions are frozen. Default is False.
        verbose : int, optional
            Verbosity level. Default is 0.
        device : torch.device, optional
            Device to place the model and parameters on. If None, infers from
            the first model's device.

        Raises
        ------
        ValueError
            If models list is empty, fractions don't match model count,
            fractions don't sum to 1, or models have incompatible parameters.
        """
        super().__init__()

        if not models:
            raise ValueError("At least one model is required.")

        self.verbose = verbose

        # Infer device from first model if not specified
        if device is None:
            device = models[0].device

        # Store models as ModuleList for proper PyTorch handling
        models = [model.to(device=device) for model in models]
        self.models = nn.ModuleList(models)

        # Validate model compatibility
        self._validate_models()

        # Initialize fractions
        n_models = len(models)
        if initial_fractions is None:
            initial_fractions = [1.0 / n_models] * n_models
        else:
            if len(initial_fractions) != n_models:
                raise ValueError(
                    f"Number of fractions ({len(initial_fractions)}) must match "
                    f"number of models ({n_models})."
                )
            total = sum(initial_fractions)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"Initial fractions must sum to 1.0, got {total:.6f}."
                )

        # Use inverse softmax to initialize parameters
        # softmax(theta) = fractions, so theta = log(fractions)
        fractions_tensor = torch.tensor(initial_fractions, dtype=torch.float32, device=device)
        theta = torch.log(fractions_tensor.clamp(min=1e-6))
        self.fraction_params = nn.Parameter(theta, requires_grad=not frozen_fractions)

        if self.verbose > 0:
            print(f"MixedModel initialized with {n_models} models")
            print(f"  Initial fractions: {initial_fractions}")
            print(f"  Fractions frozen: {frozen_fractions}")

    def _validate_models(self):
        """
        Validate that all models have compatible cell and spacegroup.

        Raises
        ------
        ValueError
            If models have incompatible parameters.
        """
        if len(self.models) < 2:
            return  # Single model always compatible with itself

        ref_model = self.models[0]
        ref_cell = ref_model.cell
        ref_sg = ref_model.spacegroup

        for i, model in enumerate(self.models[1:], start=1):
            # Check cell compatibility (allow small tolerance)
            if ref_cell is not None and model.cell is not None:
                assert torch.allclose( ref_model.cell.data, ref_cell.data,atol=1, rtol=0.01)

            # Check spacegroup compatibility
            if ref_sg is not None and model.spacegroup is not None:
                if ref_sg.number != model.spacegroup.number:
                    raise ValueError(
                        f"Model {i} has incompatible spacegroup. "
                        f"Reference: {ref_sg.number}, Model {i}: {model.spacegroup.number}"
                    )

    @property
    def fractions(self) -> torch.Tensor:
        """
        Get normalized population fractions.

        Returns
        -------
        torch.Tensor
            Population fractions that sum to 1.0, shape (n_models,).
        """
        return torch.softmax(self.fraction_params, dim=0)

    @property
    def cell(self):
        """Unit cell from first model (for compatibility)."""
        return self.models[0].cell

    @property
    def spacegroup(self):
        """Space group from first model (for compatibility)."""
        return self.models[0].spacegroup

    @property
    def device(self):
        """Device from first model (for compatibility)."""
        return self.models[0].device

    @property
    def dtype_float(self):
        """Float dtype from first model (for compatibility)."""
        return self.models[0].dtype_float

    # =========================================================================
    # Grid infrastructure (delegates to constituent models)
    # =========================================================================

    @property
    def real_space_grid(self) -> Optional[torch.Tensor]:
        """Real-space coordinate grid from first model (shared cell → same grid)."""
        return self.models[0].real_space_grid

    @property
    def fft(self):
        """SfFFT submodule from first model (for gridsize access)."""
        return self.models[0].fft

    @property
    def gridsize(self) -> Optional[torch.Tensor]:
        """Grid dimensions (nx, ny, nz) from first model."""
        return self.models[0].gridsize

    @property
    def map_symmetry(self):
        """Map symmetry operator from first model."""
        return self.models[0].map_symmetry

    @property
    def inv_fractional_matrix(self) -> torch.Tensor:
        """Inverse fractionalization (orthogonalization) matrix."""
        return self.cell.inv_fractional_matrix.to(dtype=self.dtype_float)

    @property
    def fractional_matrix(self) -> torch.Tensor:
        """Fractionalization matrix."""
        return self.cell.fractional_matrix.to(dtype=self.dtype_float)

    def setup_grid(self, max_res=None, gridsize=None):
        """
        Setup real-space grid on all constituent models.

        All models share the same cell/spacegroup, so the grid is identical
        across all of them. This ensures each model's SfFFT is ready for
        density map calculations.

        Parameters
        ----------
        max_res : float, optional
            Maximum resolution for grid spacing in Angstroms.
        gridsize : tuple of int, optional
            Explicit grid size (nx, ny, nz).
        """
        for model in self.models:
            model.setup_grid(max_res=max_res, gridsize=gridsize)

    def get_radius(self, min_radius_Angstrom: float = 4.0) -> int:
        """
        Get the radius in voxels for density calculation.

        Delegates to first model (same grid → same voxel size).

        Parameters
        ----------
        min_radius_Angstrom : float, optional
            Minimum radius in Angstroms. Default is 4.0.

        Returns
        -------
        int
            Radius in voxels.
        """
        return self.models[0].get_radius(min_radius_Angstrom)

    def build_complete_map(self) -> torch.Tensor:
        """
        Build the mixed electron density map as the weighted sum of
        constituent model density maps.

        density_mixed = Σ w_i * density_i

        Each constituent model builds its own density map on the shared
        grid, and the results are combined using the current population
        fractions.

        Returns
        -------
        torch.Tensor
            Electron density map with shape (nx, ny, nz).
        """
        fractions = self.fractions

        density = None
        for i, model in enumerate(self.models):
            model_density = model.build_complete_map()
            weighted = fractions[i] * model_density
            if density is None:
                density = weighted
            else:
                density = density + weighted

        return density

    def freeze_fractions(self):
        """
        Exclude fractions from optimization.

        This prevents the population fractions from being updated during
        training while still allowing the constituent models to be refined.
        """
        self.fraction_params.requires_grad = False
        if self.verbose > 0:
            print("Fractions frozen")

    def unfreeze_fractions(self):
        """
        Include fractions in optimization.

        This allows the population fractions to be updated during training.
        """
        self.fraction_params.requires_grad = True
        if self.verbose > 0:
            print("Fractions unfrozen")

    def forward(self, hkl: torch.Tensor, recalc: bool = False) -> torch.Tensor:
        """
        Compute weighted sum of structure factors from all models.

        f_mixed = Σ w_i * f_i

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices with shape (n_reflections, 3).
        recalc : bool, optional
            If True, force recalculation of structure factors.
            Default is False.

        Returns
        -------
        torch.Tensor
            Mixed complex structure factors with shape (n_reflections,).
        """
        fractions = self.fractions

        # Compute structure factors from each model and weight them
        f_mixed = None
        for i, model in enumerate(self.models):
            f_i = model(hkl, recalc=recalc)
            weighted_f = fractions[i] * f_i

            if f_mixed is None:
                f_mixed = weighted_f
            else:
                f_mixed = f_mixed + weighted_f

        if self.verbose > 2:
            print(f"MixedModel forward: fractions = {fractions.detach().tolist()}")

        return f_mixed

    def get_individual_fcalc(
        self, hkl: torch.Tensor, recalc: bool = True
    ) -> List[torch.Tensor]:
        """
        Get structure factors from each model individually.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices with shape (n_reflections, 3).
        recalc : bool, optional
            If True, force recalculation. Default is True.

        Returns
        -------
        List[torch.Tensor]
            List of structure factor tensors, one per model.
        """
        return [model(hkl, recalc=recalc) for model in self.models]

    def copy(self) -> "MixedModel":
        """
        Create a deep copy of the MixedModel.

        Returns
        -------
        MixedModel
            A new MixedModel instance with copied models and parameters.
        """
        # Deep copy each constituent model
        copied_models = [model.copy() for model in self.models]

        # Get current fractions
        with torch.no_grad():
            current_fractions = self.fractions.tolist()

        # Create new MixedModel
        copied = MixedModel(
            models=copied_models,
            initial_fractions=current_fractions,
            frozen_fractions=not self.fraction_params.requires_grad,
            verbose=self.verbose,
        )

        return copied

    def to(self, device=None, dtype=None):
        """
        Move MixedModel to specified device and/or dtype.

        Parameters
        ----------
        device : torch.device or str, optional
            Target device.
        dtype : torch.dtype, optional
            Target data type.

        Returns
        -------
        MixedModel
            Self, for method chaining.
        """
        # Move fraction parameters
        if device is not None or dtype is not None:
            super().to(device=device, dtype=dtype)

        # Move all constituent models
        for model in self.models:
            model.to(device=device, dtype=dtype)

        return self

    def __repr__(self) -> str:
        """String representation."""
        fracs = self.fractions.detach().tolist()
        frac_str = ", ".join([f"{f:.3f}" for f in fracs])
        frozen_str = "frozen" if not self.fraction_params.requires_grad else "learnable"
        return f"MixedModel({len(self.models)} models, fractions=[{frac_str}], {frozen_str})"

    def write_ihm(
        self,
        filepath: str,
        state_names: Optional[List[str]] = None,
        group_name: str = "ensemble",
        datasets: Optional[dict] = None,
    ) -> None:
        """
        Write this MixedModel to an IHM mmCIF file.

        Creates a single model group with the current population fractions
        over the constituent structural states.

        Requires the optional ``python-ihm`` dependency.

        Parameters
        ----------
        filepath : str
            Output file path.
        state_names : list of str, optional
            Names for each constituent model / state.
            Default: ``state_1``, ``state_2``, ...
        group_name : str
            Name for the model group. Default ``"ensemble"``.
        datasets : dict of str -> ReflectionData, optional
            Per-timepoint reflection data to embed in the CIF.
        """
        from torchref.io.ihm import IHMWriter
        from torchref.io.ihm_mapping import (
            IHMEnsembleMapping,
            IHMModelGroupInfo,
            IHMStateInfo,
        )

        n = len(self.models)
        fracs = self.fractions.detach().cpu().tolist()

        # Build states
        states = []
        for i in range(n):
            name = state_names[i] if state_names and i < len(state_names) else f"state_{i + 1}"
            states.append(
                IHMStateInfo(state_id=i + 1, name=name, model_num=i + 1)
            )

        # Build single model group
        state_ids = [s.state_id for s in states]
        state_fractions = dict(zip(state_ids, fracs))
        groups = [
            IHMModelGroupInfo(
                group_id=1,
                name=group_name,
                state_fractions=state_fractions,
            )
        ]

        # Extract cell/spacegroup from first model
        cell = None
        spacegroup = None
        model0 = self.models[0]
        if hasattr(model0, "cell") and model0.cell is not None:
            cell_obj = model0.cell
            if hasattr(cell_obj, "tolist"):
                cell = cell_obj.tolist()
            elif hasattr(cell_obj, "parameters"):
                cell = cell_obj.parameters.tolist()
        if hasattr(model0, "spacegroup") and model0.spacegroup is not None:
            sg = model0.spacegroup
            if hasattr(sg, "hm"):
                spacegroup = sg.hm
            elif hasattr(sg, "xhm"):
                spacegroup = sg.xhm()
            else:
                spacegroup = str(sg)

        mapping = IHMEnsembleMapping(
            states=states,
            model_groups=groups,
            cell=cell,
            spacegroup=spacegroup,
        )

        # Create a temporary ModelCollection-like wrapper for the writer
        writer = IHMWriter._from_mixed_model(self, mapping)
        writer.datasets = datasets
        writer.write(filepath)

    def get_vdw_radii(self) -> torch.Tensor:
        """
        Get van der Waals radii from the first model.

        Returns
        -------
        torch.Tensor
            Van der Waals radii tensor.
        """
        return self.models[0].get_vdw_radii()
    
    def xyz(self) -> torch.Tensor:
        """
        Get atomic coordinates from the first model.

        Returns
        -------
        torch.Tensor
            Atomic coordinates tensor.
        """
        return self.models[0].xyz()