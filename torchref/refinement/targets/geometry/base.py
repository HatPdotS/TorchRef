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

    Parameters
    ----------
    model : Model, optional
        Reference to the Model object.
    verbose : int, optional
        Verbosity level. Default is 0.
    """

    def __init__(
        self,
        model: "Model" = None,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(model, verbose)

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
