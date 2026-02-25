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
    target_value : float, optional
        Target value for this loss. Default is -1.0.
    sigma : float, optional
        Sigma parameter for weighting. Default is 1.0.
    """

    def __init__(
        self,
        model: "Model" = None,
        verbose: int = 0,
        target_value: float = -1.0,
        sigma: float = 1.0,
    ):
        super().__init__(model, verbose, target_value=target_value, sigma=sigma)
