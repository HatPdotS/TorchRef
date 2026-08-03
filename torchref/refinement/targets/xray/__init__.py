"""X-ray targets: one class per selectable ``--xray-mode``.

The taxonomy and the name-to-class registry are in :mod:`._specs`; the construction entry
point is :func:`.factory.create_xray_target`. Until 2026-08 four of these rows shared one
parameterised class that branched on a spec row at runtime; they are now five independent
classes over three shared loss primitives.
"""

from .base import XrayTarget
from .factory import create_xray_target
from .least_squares import LeastSquaresXrayTarget, UnitWeightK1XrayTarget
from .ml import MLXrayTarget
from .ml_full import MLFullXrayTarget
from .ml_noalpha import MLNoAlphaXrayTarget
from .nll import NLLXrayTarget
from .nll_beta import NLLBetaXrayTarget
from .rice import RiceXrayTarget
from .sigma_a import AlphaCentredMixin, SigmaALossInputs, SigmaAXrayTarget

__all__ = [
    "XrayTarget",
    # the shared sigma_A machinery
    "SigmaAXrayTarget",
    "SigmaALossInputs",
    "AlphaCentredMixin",
    # the five selectable likelihood rows
    "NLLXrayTarget",
    "NLLBetaXrayTarget",
    "MLXrayTarget",
    "MLNoAlphaXrayTarget",
    "MLFullXrayTarget",
    # least squares
    "LeastSquaresXrayTarget",
    "UnitWeightK1XrayTarget",
    # private, non-selectable: kept for the MR aligner only (see its docstring)
    "RiceXrayTarget",
    "create_xray_target",
]
