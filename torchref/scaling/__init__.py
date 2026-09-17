"""Scale observed datasets and calculated structure factors.

Per-bin overall scale, anisotropic correction and bulk-solvent contribution.
:class:`ScalerBase` is model-independent -- every method that needs ``F_calc``
takes it as an argument; :class:`Scaler` holds a :class:`~torchref.model.Model`
and computes ``F_calc`` itself; :class:`CollectionScaler` fits one shared set of
scales jointly across a dataset/model collection. :class:`SolventModel` supplies
the flat bulk-solvent term (k_sol, B_sol). ``DatasetScaler`` independently fits
relative observed-data corrections; ``WilsonNormaliser`` supplies E values.
"""

from torchref.scaling.collection_scaler import CollectionScaler
from torchref.scaling.scaler import Scaler
from torchref.scaling.scaler_base import ScalerBase
from torchref.scaling.solvent import SolventModel
from torchref.scaling.wilson import WilsonNormaliser

__all__ = [
    "Scaler",
    "DatasetScaler",
    "ScalerBase",
    "SolventModel",
    "CollectionScaler",
    "WilsonNormaliser",
]

from torchref.scaling.dataset_scaler import DatasetScaler
