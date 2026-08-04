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
    Pin out the Debye-Waller-like trend in the per-bin ``log_scale``.

    Per-bin scales exist to absorb **localized** data-quality issues —
    an outlier bin, a slightly mis-merged shell. They are *not* supposed
    to absorb the global resolution-dependent attenuation envelope; that
    is what atomic B-factors are for. A ``log_scale[i] ≈ a + b·s²[i]``
    structure with nonzero ``b`` is exactly a B-factor masquerading as a
    per-bin scale: it shifts ``B_eff = B_atom − 4·b``, letting atoms drift
    broader or sharper while the overall ``F_scaled`` amplitudes stay
    unchanged.

    The target fits the least-squares slope of ``log_scale`` against the
    bin-mean ``|s|²`` and penalizes ``slope²``. The intercept ``a`` (the
    overall scale) is free. Residuals off the fit line are also free —
    those encode the outlier absorption the scales are *supposed* to do.

    Penalty ``slope² · N_ref / nbins`` — the ``/ nbins`` matches how xray
    gradient on a single ``log_scale[i]`` scales; the ``N_ref`` factor
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
            s_sq = (scaler.s ** 2).sum(dim=1)
            bins = scaler.bins.to(torch.int64)
            nbins = int(scaler.nbins)
            bin_s_sq = torch.zeros(nbins, device=scaler.s.device, dtype=s_sq.dtype)
            bin_counts = torch.zeros_like(bin_s_sq)
            bin_s_sq.scatter_add_(0, bins, s_sq)
            bin_counts.scatter_add_(0, bins, torch.ones_like(s_sq))
            bin_mean_s_sq = bin_s_sq / bin_counts.clamp(min=1.0)
            valid = bin_counts > 0

        s_centers = bin_mean_s_sq[valid]
        x = s_centers - s_centers.mean()
        x_var = (x * x).sum().clamp(min=1e-12)

        self.register_buffer("_valid_mask", valid)
        self.register_buffer("_x_centered", x)
        self.register_buffer("_x_var", x_var)
        self.register_buffer("_s_centers", s_centers)
        self._nbins = int(valid.sum().item())
        self._scale = float(n_reflections) / max(self._nbins, 1)

    def _slope(self) -> torch.Tensor:
        log_scale = self._scaler.log_scale[self._valid_mask]
        y = log_scale - log_scale.mean()
        return (self._x_centered * y).sum() / self._x_var

    def forward(self) -> torch.Tensor:
        """``slope² · N_ref / nbins`` on the ``log_scale`` vs ``|s|²`` fit."""
        slope = self._slope()
        return self._scale * slope ** 2

    def stats(self) -> Dict[str, any]:
        """Loss, the fitted slope, and ``B_equiv = -4·slope`` -- the atomic B shift the
        trend is standing in for.
        """
        with torch.no_grad():
            slope = self._slope().item()
            log_scale = self._scaler.log_scale.detach()
        b_equiv = -4.0 * slope
        return {
            "loss": stat(self._scale * slope ** 2, VERBOSITY_STANDARD),
            "slope": stat(slope, VERBOSITY_STANDARD),
            "B_equiv": stat(b_equiv, VERBOSITY_STANDARD),
            "log_scale_mean": stat(log_scale.mean().item(), VERBOSITY_DETAILED),
            "log_scale_range": stat(
                (log_scale.max() - log_scale.min()).item(), VERBOSITY_DETAILED
            ),
        }
