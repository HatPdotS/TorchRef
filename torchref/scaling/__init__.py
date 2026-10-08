"""Scale observed datasets and calculated structure factors.

Chebyshev isotropic scale, anisotropic correction and bulk-solvent contribution.
:class:`~.scaler_base.ScalerBase` is model-independent -- every method that needs
``F_calc`` takes it as an argument; :class:`~.scaler.Scaler` holds a
:class:`~torchref.model.model.Model` and computes ``F_calc`` itself;
:class:`~.collection_scaler.CollectionScaler` fits one shared set of scales jointly
across a dataset/model collection. :class:`~.solvent.SolventModel` supplies the flat
bulk-solvent term (k_sol, ss_half/n_exp falloff). ``DatasetScaler`` independently fits
relative observed-data corrections; ``WilsonNormaliser`` supplies E values.
"""

from torchref.scaling.collection_scaler import CollectionScaler
from torchref.scaling.dataset_scaler import DatasetScaler
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
