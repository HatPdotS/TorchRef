"""``--xray-mode ml_noalpha``: Read MLF with the Luzzati mean coupling fixed at 1."""

from torchref.base.targets.xray_likelihoods import rice_per_refl

from .sigma_a import SigmaAXrayTarget


class MLNoAlphaXrayTarget(SigmaAXrayTarget):
    """Read MLF: Rice / folded normal at ``Sigma = epsilon*beta``, centred on ``|F_calc|``.

    ``ml``'s likelihood with the mean coupling fixed at 1 -- the scaler owns that gauge.
    Its strength is as a **scale-fit** objective, where pinning the coupling is what makes
    it admissible at all.

    Also the row ``ScalerBase.refine_lbfgs`` instantiates for ``scale_target='ml_noalpha'``,
    and the right choice there because ``alpha`` is degenerate with the per-bin scale being
    fitted.

    The mean is the base class default, so there is nothing to override but this docstring.
    """

    def _per_refl(self, ctx):
        return rice_per_refl(ctx.F_obs, ctx.F_calc, ctx.Sigma, ctx.centric)
