import torch
from typing import TYPE_CHECKING, Dict

from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    stat,
)

from ..base import Target

if TYPE_CHECKING:
    from torchref.scaling.scaler_base import ScalerBase


class ScalerLogScaleTrendTarget(Target):
    """
    Pin out the Debye-Waller-like trend in the isotropic scale.

    The isotropic scale exists to absorb **localized** data-quality issues —
    an outlier shell, a slightly mis-merged region. It is *not* supposed
    to absorb the global resolution-dependent attenuation envelope; that
    is what atomic B-factors are for. A ``log k_iso(s) ≈ a + b·s²``
    structure with nonzero ``b`` is exactly a B-factor masquerading as a
    scale: it shifts ``B_eff = B_atom − 4·b``, letting atoms drift
    broader or sharper while the overall ``F_scaled`` amplitudes stay
    unchanged.

    The target fits the least-squares slope of the per-reflection log scale
    against ``|s|²`` and penalizes ``slope²``. The intercept ``a`` (the
    overall scale) is free. Curvature off the fit line is also free —
    that encodes the outlier absorption the scale is *supposed* to do.

    Because the log scale is linear in ``c_iso``, so is its least-squares
    slope: precomputing ``w = (x̃ᵀ·design) / (x̃ᵀ·x̃)`` once makes
    ``slope = w·c_iso`` exact and O(n_coeff), with no regression at
    evaluation time.

    Penalty ``slope² · N_ref / nbins`` — the ``/ nbins`` matches how the xray
    gradient on a single scale degree of freedom scales; the ``N_ref`` factor
    brings the total into xray's order of magnitude.
    """

    name: str = "adp/scaler_log_scale"

    def __init__(
        self,
        scaler: "ScalerBase",
        n_reflections: int,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        object.__setattr__(self, "_scaler", scaler)

        with torch.no_grad():
            # |s|**2, NOT the scaler's _s_half_sq: B_equiv = -4*slope is defined against
            # |s|**2, and (sin(theta)/lambda)**2 is a factor of 4 smaller.
            s_sq = (scaler.s**2).sum(dim=1)
            design = scaler._iso_design.to(s_sq.dtype)
            x = s_sq - s_sq.mean()
            x_var = (x * x).sum().clamp(min=1e-12)
            w = (x @ design) / x_var

        self.register_buffer("_slope_weights", w)
        self._nbins = int(scaler.nbins)
        self._scale = float(n_reflections) / max(self._nbins, 1)

    def _slope(self) -> torch.Tensor:
        return self._slope_weights @ self._scaler.c_iso

    def forward(self) -> torch.Tensor:
        """``slope² · N_ref / nbins`` on the log-scale vs ``|s|²`` fit."""
        slope = self._slope()
        return self._scale * slope**2

    def stats(self) -> Dict[str, any]:
        """Loss, the fitted slope, and ``B_equiv = -4·slope`` -- the atomic B shift the
        trend is standing in for.
        """
        with torch.no_grad():
            slope = self._slope().item()
            log_scale = self._scaler.iso_log_scale()
        b_equiv = -4.0 * slope
        return {
            "loss": stat(self._scale * slope**2, VERBOSITY_STANDARD),
            "slope": stat(slope, VERBOSITY_STANDARD),
            "B_equiv": stat(b_equiv, VERBOSITY_STANDARD),
            "log_scale_mean": stat(log_scale.mean().item(), VERBOSITY_DETAILED),
            "log_scale_range": stat(
                (log_scale.max() - log_scale.min()).item(), VERBOSITY_DETAILED
            ),
        }
