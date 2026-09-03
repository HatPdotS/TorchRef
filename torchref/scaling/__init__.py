"""Scaling calculated structure factors onto observed data.

Per-bin overall scale, anisotropic correction and bulk-solvent contribution.
:class:`ScalerBase` is model-independent -- every method that needs ``F_calc``
takes it as an argument; :class:`Scaler` holds a :class:`~torchref.model.Model`
and computes ``F_calc`` itself; :class:`CollectionScaler` fits one shared set of
scales jointly across a dataset/model collection. :class:`SolventModel` supplies
the flat bulk-solvent term (k_sol, B_sol).
"""

from torchref.scaling.scaler import Scaler
from torchref.scaling.scaler_base import ScalerBase
from torchref.scaling.solvent import SolventModel
from torchref.scaling.collection_scaler import CollectionScaler
from torchref.scaling.wilson import WilsonNormaliser

__all__ = [
    "Scaler",
    "ScalerBase",
    "SolventModel",
    "CollectionScaler",
    "WilsonNormaliser",
]
