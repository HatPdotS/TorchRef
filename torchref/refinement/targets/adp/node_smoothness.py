"""Magnitude prior on the node values of a disorder field."""

import torch
from typing import TYPE_CHECKING, Dict

from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

from .base import ADPTarget

if TYPE_CHECKING:
    from torchref.model.model import Model


class NodeSmoothnessTarget(ADPTarget):
    """Penalise a node whose B departs from the nodes around it.

    The companion to :class:`~torchref.refinement.targets.adp.NodeLoadTarget`, which
    acts on the weights and therefore cannot reach the node *values* at all. Blocking a
    node from narrowing does not stop it taking an extreme B -- measured, it makes the
    consequence broader rather than smaller, because the extreme value can no longer be
    confined to the single atom the node had isolated. So the two terms close different
    halves: one denies the opportunity, this one prices the magnitude.

    The penalty is a distance-weighted sum over node pairs::

        L = sum_{k<j} w_kj (log B_k - log B_j)^2 / sum_{k<j} w_kj,
        w_kj = exp(-d_kj^2 / 2 lambda^2)

    with ``lambda`` set from the median nearest-neighbour node distance, so the scale
    follows the node density rather than needing a tuned length.

    Three properties this form has and a plain magnitude penalty does not:

    * **Level-invariant**, so gauge-safe. It sees only differences of ``log B``, so
      adding a constant to every node costs nothing. The overall B level is largely
      absorbed by the scaler, and a penalty pulling B toward zero would fight it over a
      direction neither owns.
    * **Scale-free**, being in log space, so it treats a factor-of-two departure the
      same at B = 10 and B = 100.
    * **Smooth variation is free.** A structure with a genuinely flexible tail has a
      real B gradient across it; only departures from the *local* level are charged, so
      the gradient passes untouched while an isolated spike does not.

    All node pairs are used rather than a k-nearest-node graph. Node counts are small,
    so the pairwise form costs nothing, and it avoids a discontinuity: a graph built by
    ``topk`` reorders as nodes move, which puts a jump in the loss for no benefit.

    Inert unless the model is in field mode, so it can be registered unconditionally.

    Parameters
    ----------
    model : Model, optional
        Reference to the Model object.
    length_scale : float, optional
        Override for ``lambda`` in Angstrom. By default it is taken from the median
        nearest-neighbour node distance each call, so it tracks the node layout.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

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
        """The disorder field, or ``None`` when the model is not in field mode."""
        adp = getattr(self.model, "adp", None)
        return adp if hasattr(adp, "node_load") else None

    def _pair_terms(self):
        """``(weighted mean squared log-B difference, pair weights)``."""
        field = self._field
        # Through the payload, not by column index: for a tensor payload column 0 is a
        # Cholesky component, not a magnitude.
        log_b = field.log_magnitude()
        pos = field.node_positions()

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
            return torch.zeros((), device=self.device)
        w, diff2, _ = self._pair_terms()
        total = w.sum()
        if float(total) <= 0.0:
            return torch.zeros((), device=self.device)
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
