from typing import TYPE_CHECKING, Dict

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
    ``**kwargs`` accepted by ``__init__`` are forwarded to
    :class:`~torchref.refinement.targets.base.ModelTarget`, which currently
    discards them. Passing ``target_value`` / ``sigma`` here therefore has no
    effect on the loss; per-target tuning is done through each subclass's own
    explicit parameters.
    """

    def __init__(
        self,
        model: "Model" = None,
        verbose: int = 0,
        device=None,
        **kwargs,
    ):
        super().__init__(model, verbose, device=device)

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
