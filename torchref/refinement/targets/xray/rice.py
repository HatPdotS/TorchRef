from typing import TYPE_CHECKING

import torch

from torchref.base.targets.xray_likelihoods import rice_math

from .base import XrayTarget

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model import Model
    from torchref.scaling.scaler_base import Scaler


class RiceXrayTarget(XrayTarget):
    """Rice at ``Sigma = sigma_obs**2``. **PRIVATE: not a selectable ``--xray-mode``.**

    The ``rice`` mode was removed from the taxonomy and its removal is asserted by
    ``tests/unit/refinement/test_nll_beta.py::test_rice_with_sigma_obs_is_not_offered``.
    Pairing ``sigma_obs`` with a Rice ``Sigma`` asserts an isotropic *complex* error while
    ``sigma_obs`` carries no phase information, so no regime makes it correct -- and
    empirically it was the worst-behaved target measured, destroying geometry (bond RMSZ
    28.0 where every other target sat near 1.3). See ``_specs.py``'s module docstring.

    This class survives for exactly one caller:
    :class:`torchref.experimental.alignment.rigid_body.RigidBodyRefinement`, the FFT-direct
    rigid-body aligner in the molecular-replacement pipeline, which constructs it directly
    rather than through the factory.

    **Why it was not simply repointed at ``nll``** during the 2026-08 target refactor, as
    originally planned: that aligner has **no test coverage whatsoever**
    (``tests/integration/test_rigid_body_refinement.py`` exercises the *other* rigid-body
    module, ``refinement/rigid_body_refinement.py``). Swapping a Rice likelihood for a
    Gaussian there would be an untested numerical change in a live MR path, so the
    likelihood was kept and only its *implementation* was de-duplicated -- the body now
    calls the shared :func:`~torchref.base.targets.xray_likelihoods.rice_math` primitive
    instead of a second copy of the Rice in the deleted ``xray_ml`` module.

    Whoever gives that aligner a test should revisit this: ``nll`` or ``ml_noalpha`` is
    almost certainly the better objective, and then this class can go.
    """

    #: ``epsilon * beta`` was clamped here in the implementation this replaced. Preserved
    #: exactly: it is NOT the same as the primitive's own ``VAR_FLOOR`` (1e-10), and it
    #: fires on genuinely weak data (``sigma < 1e-3``), so dropping it would be a silent
    #: numerical change rather than a refactor.
    _SIGMA_SQ_FLOOR = 1e-6

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """Summed Rice loss on this target's set, at ``Sigma = clamp(sigma**2, 1e-6)``."""
        F_obs, F_calc, sigma, centric_flags, _ = self.get_data(fcalc=fcalc)
        Sigma = torch.clamp(sigma**2, min=self._SIGMA_SQ_FLOOR)
        return rice_math(F_obs, F_calc, Sigma, centric_flags)
