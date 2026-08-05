"""``--xray-mode ml`` (the default): Read MLF centred on ``alpha*|F_calc|``."""

from .ml_noalpha import MLNoAlphaXrayTarget
from .sigma_a import AlphaCentredMixin


class MLXrayTarget(AlphaCentredMixin, MLNoAlphaXrayTarget):
    """Read MLF at ``Sigma = epsilon*beta``, conditional mean ``alpha*|F_calc|``.

    The default target. Inherits its likelihood from :class:`MLNoAlphaXrayTarget` and its
    mean from :class:`~torchref.refinement.targets.xray.sigma_a.AlphaCentredMixin`, so this
    class is the *pairing* and nothing else -- which is the whole content of the difference
    between the two rows.

    **On alpha being a net loss.** Over 767 AF-start structures, centring on ``alpha*|F_c|``
    costs R_work +0.00200 and R_free +0.00020 (both p=0.0000) against ``ml_noalpha``, and the
    penalty is concentrated entirely at low resolution. An earlier 50-structure screen read
    the opposite; **do not re-derive the optimistic reading from the small-n numbers.** It
    remains the default because the Luzzati mean is the principled form and the effect is far
    below anything a reviewer would call material -- and because
    ``paper/extended_figures/exF5`` exists to show that.

    The estimator fits ``alpha`` jointly for *every* row regardless (pinning the mean during
    the fit biases ``sigma_A`` +0.035 high), so this row differs only in whether the
    *likelihood* consumes it.
    """
