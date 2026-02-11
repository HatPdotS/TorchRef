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