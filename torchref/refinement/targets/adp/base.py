from typing import TYPE_CHECKING

from ..base import ModelTarget

if TYPE_CHECKING:
    from torchref.model.model import Model


class ADPTarget(ModelTarget):
    """
    Base class for ADP restraint targets.

    ADP targets access the model's ADP values and restraints for similarity,
    rigid bond, and other ADP-related restraints.

    Subclasses are expected to implement ``stats()`` returning a
    ``Dict[str, StatEntry]`` (the same contract as
    :class:`~torchref.refinement.targets.geometry.base.GeometryTarget`),
    where each :class:`~torchref.utils.stats.StatEntry` carries its value and
    a verbosity level for display-time filtering.

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
