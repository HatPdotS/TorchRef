from typing import TYPE_CHECKING

from ..base import ModelTarget

if TYPE_CHECKING:
    from torchref.model.model import Model


class ADPTarget(ModelTarget):
    """
    Base class for ADP restraint targets.

    ADP targets access the model's ADP values and restraints for similarity,
    rigid bond, and other ADP-related restraints.

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
