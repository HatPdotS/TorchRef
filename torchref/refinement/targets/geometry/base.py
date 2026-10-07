from typing import TYPE_CHECKING, Dict, Optional

import torch

from torchref.utils.stats import StatEntry

from ..base import ModelTarget

if TYPE_CHECKING:
    from torchref.model.model import Model


class GeometryTarget(ModelTarget):
    """
    Base class for geometry restraint targets.

    Geometry targets access the model's restraints property (built lazily)
    to compute losses for bonds, angles, torsions, planes, etc.

    Subclasses implement ``stats()`` returning a ``Dict[str, StatEntry]``;
    each :class:`~torchref.utils.stats.StatEntry` carries its value and a
    verbosity level for display-time filtering via ``filter_stats()``.

    Parameters
    ----------
    model : Model, optional
        Reference to the Model object.
    verbose : int, optional
        Verbosity level. Default is 0.

    Notes
    -----
    Extra keyword arguments are discarded, not forwarded, so ``target_value`` or
    ``sigma`` passed here has no effect on the loss; each subclass takes its tuning
    through its own explicit parameters.
    """

    def __init__(
        self,
        model: "Model" = None,
        verbose: int = 0,
        device=None,
        **kwargs,
    ):
        super().__init__(model, verbose, device=device)

    def _restraint_group(
        self, edge_type: str, origin: Optional[str] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Return ``restraints[edge_type][origin]``, or ``restraints[edge_type]`` when
        ``origin`` is None (chirals); None when the group is absent or has no indices.

        A model without a restraint kind lacks the group rather than holding an empty
        one (glycines have no torsion ``all`` group, waters no bonds), and its chiral
        group is ``{}``, so the targets read every group through this.
        """
        group = self.restraints.restraints.get(edge_type, {})
        if origin is not None:
            group = group.get(origin, {})
        indices = group.get("indices")
        if indices is None or len(indices) == 0:
            return None
        return group

    def stats(self) -> Dict[str, StatEntry]:
        """
        Get statistics for this restraint type.

        Returns dict with StatEntry values. Filter with filter_stats() at display time.

        Returns
        -------
        dict
            Statistics dict with StatEntry values containing verbosity levels.
        """
        raise NotImplementedError("Subclasses should implement stats()")
