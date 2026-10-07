"""Load balancing for a node-field ADP representation."""

import torch
from typing import TYPE_CHECKING, Dict

from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    stat,
)

from .base import ADPTarget

if TYPE_CHECKING:
    from torchref.model.model import Model


class NodeLoadTarget(ADPTarget):
    """Keep every disorder-field node carrying a fair share of atoms.

    A node's load is the total weight it holds across all atoms
    (:meth:`~torchref.model.disorder_field.DisorderFieldTensor.node_load`); the weights
    partition unity, so the mean load is ``n_atoms / K``. The penalty, summed over
    nodes, is ``s * softplus(-log(load / mean_load) / s)``: it grows as a node is
    abandoned and flattens once the node carries its share, one-sided because a broad
    node may legitimately carry more than the mean. It acts through the weights, so its
    gradient reaches node positions and kernel widths but never the node values. Inert
    unless the model is in field mode, so it can be registered unconditionally.

    Parameters
    ----------
    model : Model, optional
        Reference to the Model object.
    sharpness : float, optional
        Softplus temperature ``s`` in log-load units. Smaller is a harder barrier.
        Default 0.5, which leaves a node at the mean load contributing about 0.35 and a
        node at a tenth of the mean about 2.3.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    #: Hierarchical key this target registers under. Required, not cosmetic:
    #: LossState.register_targets takes the key from ``.name``, so without it the
    #: target inherits ``Target.name`` ("model_target"), registers under that,
    #: collides with every other unnamed target, and no ``adp/...`` weight can
    #: reach it -- the term is then built, callable, and never in the loss.
    name: str = "adp/node_load"

    def __init__(
        self,
        model: "Model" = None,
        sharpness: float = 0.5,
        verbose: int = 0,
        device=None,
        **kwargs,
    ):
        super().__init__(model, verbose, device=device, **kwargs)
        self.sharpness = float(sharpness)

    @property
    def _field(self):
        """The disorder field, or ``None`` when the model is not in field mode.

        Reads ``Model.adp_field`` rather than the ``adp`` slot directly: an anisotropic
        payload lives in ``u`` instead, and looking only at ``adp`` would leave this
        target silently inert in exactly the mode with the most node parameters to
        collapse.
        """
        return getattr(self.model, "adp_field", None)

    def _relative_load(self) -> torch.Tensor:
        """Each node's load as a multiple of the mean load, ``(K,)``."""
        field = self._field
        load = field.node_load()
        # Mean load is n_atoms / K exactly, because the weights sum to one per atom.
        return load / (load.sum().detach() / load.shape[0]).clamp(min=1e-12)

    def forward(self) -> torch.Tensor:
        """Summed one-sided load deficit over nodes, or zero outside field mode."""
        field = self._field
        if field is None:
            return torch.zeros((), device=self.device, dtype=self.dtype_float)
        rel = self._relative_load()
        deficit = -torch.log(rel.clamp(min=1e-12)) / self.sharpness
        return torch.nn.functional.softplus(deficit).sum() * self.sharpness

    def stats(self) -> Dict[str, any]:
        """Load distribution across nodes, and how much of it the barrier sees."""
        field = self._field
        if field is None:
            return {"node_load_active": stat(0.0, VERBOSITY_DEBUG)}
        with torch.no_grad():
            rel = self._relative_load()
            loss = self.forward()
        return {
            "node_load_loss": stat(float(loss), VERBOSITY_STANDARD),
            "n_nodes": stat(int(rel.numel()), VERBOSITY_STANDARD),
            "load_min_rel": stat(float(rel.min()), VERBOSITY_STANDARD),
            "load_median_rel": stat(float(rel.median()), VERBOSITY_DETAILED),
            "load_max_rel": stat(float(rel.max()), VERBOSITY_DETAILED),
            # The population the barrier exists for.
            "n_below_quarter_share": stat(
                int((rel < 0.25).sum()), VERBOSITY_STANDARD
            ),
            "load_cv": stat(float(rel.std() / rel.mean().clamp(min=1e-12)),
                            VERBOSITY_DEBUG),
        }
