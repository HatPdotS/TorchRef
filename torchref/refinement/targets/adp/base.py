from typing import TYPE_CHECKING

import torch

from torchref.base.targets.adp import u6_b_eq

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

    def _b_values(self) -> torch.Tensor:
        """Per-atom B in Å², shape ``(n_atoms,)``: B_eq from the unified U6 when any
        atom is anisotropic, else ``model.adp()``.

        Read B through this rather than ``model.adp()``, whose value for an anisotropic
        atom is no longer refined. An all-isotropic model takes the direct path and is
        numerically identical, since ``u6_b_eq`` reduces to B for isotropic atoms.
        """
        if not getattr(self.model, "_aniso_is_empty", True):
            return u6_b_eq(self.model.adp_u6())
        return self.model.adp()
