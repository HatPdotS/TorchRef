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
    ``**kwargs`` accepted by ``__init__`` are forwarded to
    :class:`~torchref.refinement.targets.base.ModelTarget`, which currently
    discards them. Passing ``target_value`` / ``sigma`` here therefore has no
    effect on the loss; per-target tuning is done through each subclass's own
    explicit parameters (e.g. ``sigma`` buffers stored on the subclass).
    """

    def __init__(
        self,
        model: "Model" = None,
        verbose: int = 0,
        device=None,
        **kwargs,
    ):
        super().__init__(model, verbose, device=device)
