"""``--xray-mode ml`` (the default): Read MLF centred on ``alpha*|F_calc|``."""

from .ml_noalpha import MLNoAlphaXrayTarget
from .sigma_a import AlphaCentredMixin


class MLXrayTarget(AlphaCentredMixin, MLNoAlphaXrayTarget):
    """Read MLF at ``Sigma = epsilon*beta``, conditional mean ``alpha*|F_calc|``.

    The default target. Inherits its likelihood from :class:`MLNoAlphaXrayTarget` and its
    mean from :class:`~torchref.refinement.targets.xray.sigma_a.AlphaCentredMixin`, so this
    class is the *pairing* and nothing else -- which is the whole content of the difference
    between the two rows. It is the default because the Luzzati mean coupling is the
    principled conditional distribution.

    The estimator fits ``alpha`` jointly for *every* row regardless (pinning the mean during
    the fit would bias ``sigma_A`` high), so this row differs only in whether the
    *likelihood* consumes it.
    """
