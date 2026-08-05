"""``--xray-mode nll_beta``: the Read MLF's large-signal Gaussian limit."""

from torchref.base.targets.xray_likelihoods import (
    amplitude_var_from_complex,
    nll_per_refl,
)

from .sigma_a import SigmaAXrayTarget


class NLLBetaXrayTarget(SigmaAXrayTarget):
    """Gaussian amplitude NLL on ``ml``'s variance, centred on ``|F_calc|``.

    Keeps the per-shell model-error variance and throws away the Rice/Bessel structure, so
    comparing it against ``ml_noalpha`` separates "does the *distribution shape* matter" from
    "does the *variance model* matter". Diagnostic, not a competitor.

    Note this is the row whose name is most easily misread: it is a Gaussian, like ``nll``,
    but at ``epsilon*beta`` rather than ``sigma_obs**2`` -- the variance is *inflated* from
    the measurement error to the model error. The conversion from complex to amplitude
    variance (``Sigma/2`` acentric, ``Sigma`` centric) is the large-signal limit and must go
    through the named builder: getting that factor wrong rescales the whole x-ray gradient by
    2, which is indistinguishable from a change of x-ray weight and would confound exactly
    the comparison this row exists for.
    """

    def _per_refl(self, ctx):
        return nll_per_refl(
            ctx.F_obs, ctx.F_calc, amplitude_var_from_complex(ctx.Sigma, ctx.centric)
        )
