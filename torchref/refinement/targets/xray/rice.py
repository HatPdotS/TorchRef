from typing import TYPE_CHECKING

import torch

from torchref.base.targets.xray_likelihoods import rice_math, rice_per_refl

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

    **It now has no caller at all, and is a deletion candidate.** It survived the
    2026-08 target refactor for exactly one: the FFT-direct rigid-body aligner in the
    molecular-replacement pipeline, which constructed it directly rather than through
    the factory. That aligner had no test coverage
    (``tests/integration/test_rigid_body_refinement.py`` exercises the *other*
    rigid-body module, ``refinement/rigid_body_refinement.py``), so repointing it at
    ``nll`` would have been an untested numerical change in a live path -- the
    likelihood was kept and only its *implementation* de-duplicated onto the shared
    :func:`~torchref.base.targets.xray_likelihoods.rice_math` primitive.

    The MR pipeline no longer polishes placements, so that aligner is gone and the
    constraint with it. What remains is this class, three unit tests of it, and an
    export. Removing all of that is a ``refinement/`` change and belongs in a
    ``refinement/`` commit, not an alignment one.
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

    def _per_refl(self, ctx) -> torch.Tensor:
        """Per-reflection Rice loss at ``Sigma = clamp(sigma**2, 1e-6)``."""
        F_obs, F_calc, sigma, centric_flags, _ = ctx
        Sigma = torch.clamp(sigma**2, min=self._SIGMA_SQ_FLOOR)
        return rice_per_refl(F_obs, F_calc, Sigma, centric_flags)
