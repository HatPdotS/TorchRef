"""Select measured intensity observations for Gaussian X-ray targets.

``IntensityObservableMixin`` reads ``I_obs`` and ``sigma(I)`` and predicts
``|F_calc|**2``, retaining negative intensities that French-Wilson amplitude
conversion reshapes. Likelihoods, subsets and masks come from the target class;
reported R-factors remain in amplitude space. Rice targets describe amplitudes
and therefore have no intensity variant.
"""

from typing import Tuple

import torch

from torchref.base.targets.xray_likelihoods import (
    SIGMA_FLOOR_ABS,
    SIGMA_FLOOR_FRAC,
    gaussian_per_refl,
    intensity_var_from_sigma_obs,
    _masked_sum,
)

from .base import XrayTarget


class IntensityObservableMixin:
    """Read ``I_obs``/``sigma(I)`` and predict ``|F_calc|**2``.

    Mix in **before** an :class:`~.base.XrayTarget` subclass. The only override is
    :meth:`get_data`, which is the single place the observable is chosen -- so a row
    composed with this mixin cannot end up fitting intensities while reporting statistics
    on amplitudes, and no method needs a runtime branch.

    Note there is deliberately no ``_scaled_F_calc_full`` override. That method feeds
    :meth:`XrayTarget.get_rfactor`, and for a ``|F_calc|**2`` model its correct value is
    ``sqrt(I_calc) == |F_calc|`` -- exactly what the inherited implementation returns. So
    **R-factors stay on amplitudes for every row**, comparable across the whole table
    regardless of which observable drove the loss. A row whose intensity model is *not*
    the square of an amplitude (the two-moment model, where it is
    ``|F|**2 + var*|dF|**2``) must override it to report ``sqrt`` of its own model.
    """

    #: Declared for the taxonomy table, and readable off any constructed target.
    observable: str = "intensity"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Checked at construction rather than at the first forward: LossState probes
        # `forward()` once when a target is registered, so a missing column would
        # otherwise surface as a failure deep inside setup with no mention of the cause.
        data = getattr(self, "_data", None)
        if data is not None and getattr(data, "I", None) is None:
            raise ValueError(
                f"{type(self).__name__} fits intensities, but this dataset carries none. "
                "Load an MTZ/mmCIF with an I column (`I-obs`/`intensity_meas`), or select "
                "an amplitude row such as `nll`."
            )

    def get_data(
        self, fcalc: torch.Tensor = None, sub=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, object]:
        """``(I_obs, I_calc, sigma_I, centric, sub)`` -- the intensity twin of
        :meth:`XrayTarget.get_data`, same tuple shape and same subset semantics.

        Both observation columns are the *corrected* views (``sub.I``/``sub.sigI``), in
        which the scale and the anisotropy factor enter squared, so they are on the same
        footing as the squared model amplitude from :meth:`get_I_calc_scaled`.
        """
        if sub is None:
            sub = self._subset()

        I_obs = sub.I
        sigma = sub.sigI
        centric = sub.centric

        if fcalc is not None:
            I_calc_full = self.get_I_calc_scaled(fcalc=fcalc)
        else:
            I_calc_full = self.get_I_calc_scaled(recalc=False)
        I_calc = sub.select(I_calc_full)

        return I_obs, I_calc, sigma, centric, sub

    def _sigma_floor(self) -> torch.Tensor:
        """The intensity-sigma floor, taken from THIS target's own fitted subset.

        Computed here rather than inside the variance builder so it does not depend on
        which reflections a particular call happens to pass. ``forward`` evaluates on the
        target's subset while ``residuals`` evaluates on every reflection; a floor derived
        from the argument therefore differs between them, and the same reflection scores
        differently in the two -- 0.09% on a 1DAW work set, 1.8% on its free set, because
        sigma(I) spans orders of magnitude where sigma(F) does not.

        Detached: it is a numerical safeguard, not a fitted quantity, and letting a
        gradient run back through a median would make the loss depend on the ordering of
        near-equal sigmas.
        """
        sigma = self._subset().sigI
        if sigma is None or sigma.numel() == 0:
            return torch.as_tensor(1e-6)
        return (torch.median(sigma).detach() * SIGMA_FLOOR_FRAC).clamp(min=SIGMA_FLOOR_ABS)


class NLLIntensityXrayTarget(IntensityObservableMixin, XrayTarget):
    """``--xray-mode nll_i``: Gaussian intensity NLL weighted by the experimental sigma.

        NLL = 0.5*(I_obs - |F_calc|**2)**2/sigma_I**2 + log(sigma_I) + 0.5*log(2*pi)

    The intensity counterpart of :class:`~.nll.NLLXrayTarget`, and like it carries no
    model-error term, so it does **not** control overfitting.

    Subclasses :class:`~.base.XrayTarget` directly rather than ``NLLXrayTarget``, because
    that row's ``forward`` calls the fused Triton amplitude kernel
    (``nll_sigma_obs_math``) which has no intensity counterpart. Here ``forward`` is the
    structural ``_masked_sum(_per_refl(...))``, so it and :meth:`residuals` are the same
    expression by construction rather than by test.
    """

    target_value: float = 1.0

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """Summed Gaussian NLL of the observed intensities on this target's set."""
        return _masked_sum(self._per_refl(self._loss_inputs(fcalc=fcalc)))

    def _per_refl(self, ctx) -> torch.Tensor:
        """Per-reflection Gaussian on the intensity. See :func:`gaussian_per_refl`."""
        I_obs, I_calc, sigma, _, _ = ctx
        var = intensity_var_from_sigma_obs(sigma, floor=self._sigma_floor())
        return gaussian_per_refl(I_obs, I_calc, var, var_floor=0.0)
