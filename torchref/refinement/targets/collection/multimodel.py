"""Multi-model geometry/ADP restraint targets for a ModelCollection.

Generic (non-kinetic) collection targets: apply the standard geometry / ADP
restraints to the shared base models of a ``ModelCollection`` and sum them.
"""

from typing import TYPE_CHECKING

import torch
from torch import nn

from torchref.refinement.targets.base import Target

if TYPE_CHECKING:
    from torchref.model.model_collection import ModelCollection


class MultiModelGeometryTarget(Target):
    """
    Geometry restraints for the shared base models in a ModelCollection.

    Creates a ``TotalGeometryTarget`` for each base model and sums them.
    Since models are shared across timepoints, restraints only need to be
    computed once per base model (not per timepoint).

    Parameters
    ----------
    model_collection : ModelCollection
    verbose : int
        Verbosity level.
    """

    name: str = "multi_model_geometry"

    def __init__(self, model_collection: "ModelCollection", verbose: int = 0):
        super().__init__(verbose=verbose)
        self._model_collection = model_collection

        from torchref.refinement.targets.combined import TotalGeometryTarget

        self._targets = nn.ModuleList(
            [
                TotalGeometryTarget(model=m, verbose=verbose)
                for m in model_collection.base_models
            ]
        )

    def forward(self) -> torch.Tensor:
        """Geometry loss summed over the collection's base models."""
        total = torch.tensor(0.0, device=self._model_collection.device)
        for target in self._targets:
            total = total + target()
        return total

    def register_to_state(self, state):
        """
        Register each base model's geometry sub-targets into ``state`` individually,
        named ``model_<i>/<sub>``. Returns the state for chaining.
        """
        for i, target in enumerate(self._targets):
            state.register_target("geometry", target, prefix=f"model_{i}")
        return state

    def items(self):
        """Expose sub-targets for LossState auto-expansion."""
        result = {}
        for i, target in enumerate(self._targets):
            for sub_name, sub_target in target.items():
                result[f"model_{i}/{sub_name}"] = sub_target
        return result.items()


class MultiModelADPTarget(Target):
    """
    ADP restraints for the shared base models in a ModelCollection.

    Same pattern as MultiModelGeometryTarget but using TotalADPTarget.

    Parameters
    ----------
    model_collection : ModelCollection
    verbose : int
        Verbosity level.
    """

    name: str = "multi_model_adp"

    def __init__(self, model_collection: "ModelCollection", verbose: int = 0):
        super().__init__(verbose=verbose)
        self._model_collection = model_collection

        from torchref.refinement.targets.combined import TotalADPTarget

        self._targets = nn.ModuleList(
            [
                TotalADPTarget(model=m, verbose=verbose)
                for m in model_collection.base_models
            ]
        )

    def forward(self) -> torch.Tensor:
        """ADP restraint loss summed over the collection's base models."""
        total = torch.tensor(0.0, device=self._model_collection.device)
        for target in self._targets:
            total = total + target()
        return total

    def register_to_state(self, state):
        """Register per-model ADP sub-targets into LossState."""
        for i, target in enumerate(self._targets):
            state.register_target("adp", target, prefix=f"model_{i}")
        return state

    def items(self):
        """Sub-targets as ``("model_<i>/<sub>", target)`` for LossState expansion."""
        result = {}
        for i, target in enumerate(self._targets):
            for sub_name, sub_target in target.items():
                result[f"model_{i}/{sub_name}"] = sub_target
        return result.items()
