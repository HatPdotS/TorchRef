"""Build the x-ray target named by ``--xray-mode``. One line, no conditionals."""

from typing import TYPE_CHECKING, Optional

import torch

from torchref.refinement.model_error_estimation.sigma_a import SIGMA_A_MAX, SigmaAConfig

from ._specs import DEFAULT_XRAY_MODE, XRAY_TARGETS
from .base import XrayTarget

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model import Model
    from torchref.scaling.scaler_base import Scaler


def create_xray_target(
    data: "ReflectionData" = None,
    model: "Model" = None,
    scaler: "Scaler" = None,
    mode: str = DEFAULT_XRAY_MODE,
    use_work_set: bool = True,
    sigma_a_max: float = SIGMA_A_MAX,
    shrink: bool = None,
    verbose: int = 0,
    device: Optional[torch.device] = None,
    use_set: str = None,
) -> XrayTarget:
    """Construct the x-ray target for ``mode``.

    Dispatch is ``XRAY_TARGETS.by_name(mode).target_cls(**kwargs)`` -- there is no branching
    left. Until 2026-08 this held three predicates (``spec.family == "least_squares"``,
    ``spec.name == "ls_wunit_k1"``, ``not spec.needs_estimator``) because two classes each
    served several modes; every row now has its own class and the table maps name to class.

    Parameters
    ----------
    data : ReflectionData
        Required for ``forward()``.
    model : Model or ModelFT, optional
        Used for F_calc. If None, ``fcalc`` must be passed to ``forward()``.
    scaler : Scaler, optional
        Owns the overall scale for every mode except ``ls_wunit_k1``, which fits its own.
    mode : str, optional
        Any name in :data:`~torchref.refinement.targets.xray._specs.XRAY_TARGETS`, the single
        source of truth for the taxonomy -- see that module's docstring for what the rows
        mean and which was measured best. Default ``'ml'``.
    use_work_set : bool, optional
        Legacy 2-way selector, superseded by ``use_set``. Default True.
    sigma_a_max, shrink : optional
        Model-error estimator knobs. Packed into one
        :class:`~torchref.refinement.model_error_estimation.sigma_a.SigmaAConfig` and handed
        to **every** row, so no ``needs_estimator`` conditional is needed to decide who gets
        them; rows that do not use them store the config and ignore it. ``shrink=None`` means
        the module default.
    verbose : int, optional
        Verbosity. Default 0.
    device : torch.device, optional
        Pins model/data/scaler before construction.
    use_set : str, optional
        Canonical 3-way subset selector (``"work"``/``"free"``/``"val"``); takes precedence
        over ``use_work_set``.

    Returns
    -------
    XrayTarget
        An instance of the class the table names for ``mode``.
    """
    return XRAY_TARGETS.by_name(mode).target_cls(
        data=data,
        model=model,
        scaler=scaler,
        use_work_set=use_work_set,
        verbose=verbose,
        use_set=use_set,
        device=device,
        sigma_a=SigmaAConfig(sigma_a_max=sigma_a_max, shrink=shrink),
    )
