"""Smoothness prior on the log-B node values of a disorder field."""

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


class NodeSmoothnessTarget(ADPTarget):
    """Penalise a node whose B departs from the nodes around it.

    A distance-weighted mean over node pairs::

        L = sum_{k<j} w_kj (log B_k - log B_j)^2 / sum_{k<j} w_kj,
        w_kj = exp(-d_kj^2 / 2 lambda^2)

    with ``lambda`` the median nearest-neighbour node distance (detached) unless
    ``length_scale`` is given. Only log-B differences enter, so the overall B level is
    left to the scaler and a smooth B gradient across the structure costs little, while
    an isolated spike does not. Inert unless the model is in field mode, so it can be
    registered unconditionally.

    Parameters
    ----------
    model : Model, optional
        Reference to the Model object.
    length_scale : float, optional
        ``lambda`` in Å. By default the median nearest-neighbour node distance,
        recomputed each call so it tracks the node layout.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    #: Hierarchical key this target registers under. Required, not cosmetic:
    #: LossState.register_targets takes the key from ``.name``, so without it the
    #: target inherits ``Target.name`` ("model_target"), registers under that,
    #: collides with every other unnamed target, and no ``adp/...`` weight can
    #: reach it -- the term is then built, callable, and never in the loss.
    name: str = "adp/node_smoothness"

    def __init__(
        self,
        model: "Model" = None,
        length_scale: float = None,
        verbose: int = 0,
        device=None,
        **kwargs,
    ):
        super().__init__(model, verbose, device=device, **kwargs)
        self.length_scale = length_scale

    @property
    def _field(self):
        """The disorder field, or ``None`` when the model is not in field mode.

        Reads ``Model.adp_field`` rather than the ``adp`` slot directly: an anisotropic
        payload lives in ``u`` instead, and looking only at ``adp`` would leave this
        target silently inert in exactly the mode with the most node parameters to
        collapse.
        """
        return getattr(self.model, "adp_field", None)

    def _pair_terms(self):
        """Return ``(w, diff2, lam)``: upper-triangular Gaussian pair weights and
        squared log-B differences, both (K, K), and the length scale lambda in Å.
        """
        field = self._field
        # Through the payload, not by column index: for a tensor payload column 0 is a
        # Cholesky component, not a magnitude.
        log_b = field.log_magnitude()
        pos = field.node_positions()

        # All pairs, not a top-k graph, which would jump as nodes move; K is small.
        d = torch.cdist(pos, pos)
        if self.length_scale is not None:
            lam = float(self.length_scale)
        else:
            # Median nearest-neighbour node distance, detached: the length scale is a
            # property of the layout, not something the optimiser should tune by
            # spreading the nodes out.
            with torch.no_grad():
                masked = d + torch.diag(
                    torch.full((d.shape[0],), float("inf"), device=d.device, dtype=d.dtype)
                )
                lam = float(masked.min(dim=1).values.median()) if d.shape[0] > 1 else 1.0
            lam = max(lam, 1e-3)

        w = torch.exp(-(d**2) / (2.0 * lam * lam))
        w = torch.triu(w, diagonal=1)
        diff2 = (log_b[:, None] - log_b[None, :]) ** 2
        return w, diff2, lam

    def forward(self) -> torch.Tensor:
        """Weighted mean squared log-B difference between nearby nodes."""
        field = self._field
        if field is None or field.n_nodes < 2:
            return torch.zeros((), device=self.device, dtype=self.dtype_float)
        w, diff2, _ = self._pair_terms()
        total = w.sum()
        if float(total.detach()) <= 0.0:
            return total.new_zeros(())
        return (w * diff2).sum() / total

    def stats(self) -> Dict[str, any]:
        """Spread of the node values, and how localised the departures are."""
        field = self._field
        if field is None or field.n_nodes < 2:
            return {"node_smoothness_active": stat(0.0, VERBOSITY_DEBUG)}
        with torch.no_grad():
            w, diff2, lam = self._pair_terms()
            loss = self.forward()
            log_b = field.log_magnitude()
            b = torch.exp(log_b)
        return {
            "node_smoothness_loss": stat(float(loss), VERBOSITY_STANDARD),
            "node_b_median": stat(float(b.median()), VERBOSITY_STANDARD),
            "node_b_max": stat(float(b.max()), VERBOSITY_STANDARD),
            "node_log_b_sd": stat(float(log_b.std()), VERBOSITY_DETAILED),
            "node_pair_length_scale": stat(float(lam), VERBOSITY_DETAILED),
            # How far the worst node sits above its own neighbourhood.
            "node_b_max_over_median": stat(
                float(b.max() / b.median().clamp(min=1e-12)), VERBOSITY_STANDARD
            ),
        }
