"""
Ensemble Bhattacharyya X-ray target — Fisher information per effective dimension.

The single-model :class:`BhattacharyyaXrayTarget` builds the per-reflection model
uncertainty σ_m from the **data-limited** (Fisher / Cramér-Rao) positional error of
each atom: a position is only as well-determined as the data allow,
``Var(x) = 1/Fisher``. That is the right notion of error. The wrong move (a first
attempt, now discarded) was to feed the *raw ensemble spread* in as σ_m — spread is
a refined model quantity, not a data-limited error, and using its magnitude makes
σ_m circular (spread ↑ → σ_m ↑ → data weight ↓ → spread ↑) so the fit under-converges.

The correct coupling: the data's information budget is **shared across the effective
dimensions the ensemble spends**. So Fisher *per effective dimension* is ``I/k_eff``
and the model variance scales with the effective dimensionality::

    σ_m²(h)  =  sigma_m_scale² · k_eff · R(h)

- ``R(h)`` — the per-reflection **data-limited Fisher response**, i.e. the parent's
  Fisher σ_m² (``Var(x)=hist/g_w`` Cramér-Rao bound, per element-type since B is
  frozen). Detached (depends only on frozen B and fixed data tables).
- ``k_eff`` — the **PCA participation ratio** ``(Σσ²)²/Σσ⁴`` of the centered member
  matrix: a differentiable (value-only SVD), **scale-invariant** count of how many
  independent disorder modes the ensemble uses.

Because k_eff is scale-invariant, σ_m no longer depends on spread *magnitude* — a
large *coherent* (rank-1) motion has k_eff≈1 → small σ_m (data-supported), while a
diffuse many-mode cloud has large k_eff → large σ_m (overfit). And because k_eff is
**differentiable**, the Bhattacharyya log term ``½·log((σ_d²+σ_m²)/(2σ_dσ_m))``
becomes a live penalty on effective dimensionality, balanced against the residual at
the data scale (σ_m vs σ_d) — a Cramér-Rao equilibrium at the data-supported k_eff.

σ_m is on the parent's raw form-factor scale; ``sigma_m_scale`` (default 1.0, no
other scale applied) is calibrated from the σ_m/σ_d diagnostic.
"""

from typing import TYPE_CHECKING

import torch

from torchref.base.targets.xray_bhattacharyya import bhattacharyya_xray_loss_math
from .bhattacharyya import BhattacharyyaXrayTarget

if TYPE_CHECKING:
    from torchref.model.ensemble_model import EnsembleModel


class EnsembleBhattacharyyaTarget(BhattacharyyaXrayTarget):
    """Bhattacharyya target with σ_m = data-limited Fisher error × √(effective dim).

    Inherits the parent's data plumbing and Fisher σ_m machinery (``R(h)``); the
    only change is that σ_m² is multiplied by the differentiable, scale-invariant
    ensemble effective dimensionality ``k_eff`` (PCA participation ratio).
    """

    name: str = "ensemble_bhattacharyya"

    def _k_eff(self) -> torch.Tensor:
        """Differentiable PCA participation ratio of the centered member matrix.

        ``k_eff = (Σσ²)² / Σσ⁴`` over the singular values of the **occupancy-
        weighted, alive-only** centered member matrix — a scale-invariant
        effective mode count (value-only SVD backward; matches
        ``RankPenaltyTarget.spectrum_diagnostics()['participation_ratio']``).

        Weighting: center by the occupancy-weighted mean and scale each row by
        ``√w_m`` so the participation ratio reflects the mixture the data
        actually sees (``F̄ = Σ w_m F_m``). With population refinement off the
        weights are uniform (``1/N``) and alive-all, so this reduces exactly to
        the unweighted participation ratio.
        """
        xyz = self._model.xyz_per_member                  # (N_max, n_atoms, 3)
        w = self._model.member_weights()                  # (N_max,), dead -> 0
        alive = getattr(self._model, "_alive", None)
        if alive is not None:
            xyz = xyz[alive]
            w = w[alive]
        w = w / w.sum().clamp_min(1e-30)
        N = xyz.shape[0]
        X = xyz.reshape(N, -1)                             # (N, D)
        mu = (w.unsqueeze(1) * X).sum(dim=0, keepdim=True)  # weighted mean
        Xc = (X - mu) * torch.sqrt(w).unsqueeze(1)         # √w-scaled rows
        s2 = torch.linalg.svdvals(Xc) ** 2
        return (s2.sum() ** 2) / (s2 ** 2).sum().clamp_min(1e-30)

    def _sigma_m_per_refl(self) -> torch.Tensor:
        """σ_m(h) = √( k_eff · R(h) ), with R the parent's data-limited Fisher σ_m².

        ``R`` is detached (it depends only on frozen B and fixed data tables);
        ``k_eff`` carries the gradient, so in ``forward`` (called outside any
        no_grad) σ_m is differentiable through the effective dimensionality alone.
        """
        with torch.no_grad():
            # Parent R sums over all N·n_atoms ensemble atoms ⇒ ∝ N_members.
            # Normalise to a per-ASU (single mean structure) data-limited Fisher
            # response so σ_m starts physically scaled; sigma_m_scale fine-tunes.
            R = super()._sigma_m_sq_per_refl() / float(self._model.n_members)
        k_eff = self._k_eff()
        return torch.sqrt((k_eff * R).clamp_min(1e-12))

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        if not self._initialized:
            self._initialize_cache()
        F_obs, F_calc, sigma_d, _centric, mask = self.get_data(fcalc=fcalc)
        # σ_m differentiable through k_eff (R detached). NOT wrapped in no_grad —
        # that is the whole point: the log term must penalise effective dimension.
        sigma_m = self._sigma_m_per_refl()
        return bhattacharyya_xray_loss_math(F_obs, F_calc, sigma_d, sigma_m, mask)
