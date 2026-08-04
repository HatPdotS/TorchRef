"""``--xray-mode ml_noalpha``: Read MLF with the Luzzati mean coupling fixed at 1."""

from torchref.base.targets.xray_likelihoods import rice_math

from .sigma_a import SigmaAXrayTarget


class MLNoAlphaXrayTarget(SigmaAXrayTarget):
    """Read MLF: Rice / folded normal at ``Sigma = epsilon*beta``, centred on ``|F_calc|``.

    ``ml``'s likelihood with the mean coupling fixed at 1 -- the scaler owns that gauge.
    **Best of the five on R_free** over 766 AF-start structures (0.3244, tied with Phenix);
    ``ml`` is +0.00020 behind and ``ml_full`` +0.00015, both p<=0.0001.

    Also the row ``ScalerBase.refine_lbfgs`` instantiates for ``scale_target='ml_noalpha'``,
    and the right choice there because ``alpha`` is degenerate with the per-bin scale being
    fitted.

    The mean is the base class default, so there is nothing to override but this docstring.
    """

    def _loss(self, ctx):
        return rice_math(ctx.F_obs, ctx.F_calc, ctx.Sigma, ctx.centric)
