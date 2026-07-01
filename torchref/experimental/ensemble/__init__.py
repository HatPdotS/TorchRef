"""
Ensemble refinement for modelling crystallographic disorder.

.. warning::

   Experimental — this entire submodule is fast-moving research code. The
   public classes and their APIs/behaviour may change or be removed without
   notice, and several carry empirically-superseded designs retained only for
   comparison.

This submodule refines a multi-member ensemble of structural models
jointly against a single dataset, capturing static/dynamic disorder as
explicit member spread rather than (or in addition to) B-factors.

Key Classes
-----------
EnsembleRefinement
    Orchestrator that drives LBFGS refinement of an :class:`EnsembleModel`
    with ensemble-aware X-ray, rank-penalty and Wilson-prior targets.
EnsembleModel
    Multi-member atomic model; exposes per-member coordinates and supports
    low-rank / PCA re-parameterisations of the member spread.
LowRankXYZ, PCAEnsembleParam
    Reduced-dimensionality parameterisations of the ensemble displacement
    field (frozen mean + low-rank basis).
EnsembleBhattacharyyaTarget
    Ensemble X-ray likelihood (Bhattacharyya overlap of model/observed
    intensity distributions).
RankPenaltyTarget
    Soft de-overfitting penalty on the rank/magnitude of the ensemble
    displacement matrix.
WilsonPriorTarget
    Per-bin penalty keeping ``<|F_calc|^2>`` on the Wilson curve.
EnsembleAmberTarget, EnsembleAmberKLTarget, QuasiCrystalAmberTarget
    Amber force-field restraints for the ensemble, all subclasses of the
    single-molecule ``AmberTarget``. :class:`QuasiCrystalAmberTarget` (the
    symmetry-expanded PME supercell whose crystal contacts provide the
    anti-collapse regularization) is the **production** Amber path — it is the
    only one wired by :class:`EnsembleRefinement`. The per-member targets are
    **legacy / standalone**, not the default: :class:`EnsembleAmberTarget`
    (per-member mean energy) and :class:`EnsembleAmberKLTarget` (the same plus
    an entropy regularizer, whose KL/entropy anti-collapse approach was
    abandoned for the quasi-crystal layout). Constructing any of them requires
    the optional ``openmm`` dependency.
"""

from torchref.experimental.ensemble.ensemble_model import EnsembleModel
from torchref.experimental.ensemble.low_rank_ensemble import LowRankXYZ
from torchref.experimental.ensemble.pca_model import PCAEnsembleParam
from torchref.experimental.ensemble.ensemble_refinement import EnsembleRefinement
from torchref.experimental.ensemble.ensemble_bhattacharyya import (
    EnsembleBhattacharyyaTarget,
)
from torchref.experimental.ensemble.quasi_crystal_amber import QuasiCrystalAmberTarget
from torchref.experimental.ensemble.rank_penalty import RankPenaltyTarget
from torchref.experimental.ensemble.wilson_prior import WilsonPriorTarget

# Optional: constructing these requires openmm (imported lazily by AmberTarget).
try:
    from torchref.experimental.ensemble.ensemble_amber_kl import (
        EnsembleAmberKLTarget,
        EnsembleAmberTarget,
    )
except ImportError:
    EnsembleAmberKLTarget = None
    EnsembleAmberTarget = None

__all__ = [
    "EnsembleModel",
    "LowRankXYZ",
    "PCAEnsembleParam",
    "EnsembleRefinement",
    "EnsembleBhattacharyyaTarget",
    "QuasiCrystalAmberTarget",
    "RankPenaltyTarget",
    "WilsonPriorTarget",
    "EnsembleAmberTarget",
    "EnsembleAmberKLTarget",
]
