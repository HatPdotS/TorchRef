"""
Wilson prior: per-bin matching of <|F_calc|^2> to the Wilson curve.

.. warning::

   Experimental — part of ``torchref.experimental.ensemble``. The API and
   behaviour may change or be removed without notice.

The Wilson distribution says, in the absence of structural detail, the
expected per-resolution-bin mean intensity of a randomly-placed atomic
ensemble follows::

    <|F|^2>(s) = K * exp(-2 * B_W * s^2)

where ``s = 1/(2*d)`` and ``B_W`` is the overall Wilson B-factor. Real
calculated intensities should track this curve at low-to-mid resolution.
A model that drives the work-set R-factor toward zero by absorbing noise
into extra structural detail (e.g. a B-factor-free ensemble of many
coordinate copies) inflates ``<|F_calc|^2>`` in particular resolution
shells, which this target penalizes.

Loss form
---------
::

    loss = mean_bin( ( log<|F_calc|^2>_bin - log Wilson_expected(s_bin) )^2 )

The reference curve is fit once from the observed data:
``B_W = data.wilson_b`` (already computed by
``ReflectionData._calculate_wilson_b``) and ``K`` from a single
least-squares fit at first ``forward()`` call.

Used as ``'regularization/wilson'`` in the ensemble refinement LossState.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch

from torchref.refinement.targets.base import DataTarget

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model_ft import ModelFT
    from torchref.scaling.scaler_base import Scaler


class WilsonPriorTarget(DataTarget):
    """
    Wilson-curve penalty on calculated intensities.

    .. warning::

       Experimental — API and behaviour may change without notice.

    Three modes:

    - ``"bin_mean"`` (default): penalize the deviation of the *per-bin mean*
      ``log⟨|F_calc|²⟩_bin`` from the Wilson curve. ~20 constraints; a
      reflection at 10× its bin mean is invisible if another sits at 0.1×.

    - ``"per_reflection"``: penalize *each* reflection's
      ``(log|F_calc,h|² − log Σ(s_h))²`` against the Wilson curve evaluated
      at that reflection's resolution. ~N_work constraints — but an ad-hoc
      squared-log form (not a density) that pushes every reflection toward
      the bin *mean* ``z=1``, with a non-zero entropy floor (~2 nats for
      acentric), so its scale relative to the data likelihood is not
      meaningful. Kept for comparison.

    - ``"rice"`` (recommended for ensemble regularization): the Wilson
      distribution *is* the X-ray Rice likelihood evaluated at the
      **zero-structure hypothesis** (centroid ``F=0``) with the width pinned
      to the Wilson mean intensity ``σ²_h = Σ(s_h)``. This reduces to the
      Rayleigh NLL for acentric reflections and the half-normal NLL for
      centrics::

          acentric:  −log(2|F_calc,h|/Σ_h) + |F_calc,h|²/Σ_h
          centric:   −½·log(2/(π·Σ_h))     + |F_calc,h|²/(2·Σ_h)

      It is a proper per-reflection NLL in **nats**, directly commensurate
      with :class:`MaximumLikelihoodXrayTarget`. At weight 1.0 the total is
      the honest joint (data likelihood × Wilson prior): where the data is
      sharp it dominates; where the data is weak (the noise-dominated
      reflections an overparameterized ensemble overfits) the prior's O(1)
      curvature pulls ``|F_calc|²`` toward the zero-structure mode, washing
      out the inflated ``|E|²`` outliers that are the overfitting signature.
      The ``−log|F|`` barrier keeps intensities from collapsing to zero.

    Parameters
    ----------
    data : ReflectionData
        Reflection data. Must have ``wilson_b`` populated (the loader
        already does this).
    model : ModelFT
        Atomic model used to compute F_calc.
    scaler : Scaler
        Scaler used to put F_calc on the F_obs scale.
    nbins : int
        Number of resolution bins for the Wilson-curve fit and the
        ``bin_mean`` reduction (default 20). This binning is **independent**
        of the refinement's own ``nbins`` — :class:`EnsembleRefinement`
        constructs the target without forwarding ``nbins``, so the Wilson
        prior always uses 20 bins regardless of the refinement setting.
    mode : {'bin_mean', 'per_reflection', 'rice'}
        Reduction / loss form (see above).
    eps : float
        Numerical safety added inside ``log``.
    """

    name: str = "wilson_prior"

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "ModelFT" = None,
        scaler: "Scaler" = None,
        nbins: int = 20,
        mode: str = "bin_mean",
        eps: float = 1e-8,
        verbose: int = 0,
    ):
        super().__init__(data=data, model=model, scaler=scaler, verbose=verbose)
        if mode not in ("bin_mean", "per_reflection", "rice"):
            raise ValueError(
                "mode must be 'bin_mean', 'per_reflection', or 'rice'; "
                f"got {mode!r}"
            )
        self.mode = mode
        self.eps = float(eps)
        self.nbins = int(nbins)
        # ``log_K`` is fit lazily on first forward from observed bin intensities.
        self._log_K: Optional[torch.Tensor] = None
        self._B_W: Optional[torch.Tensor] = None
        # Cached resolution-bin assignment for the work-set reflections
        # (filled on first forward).
        self._bin_idx: Optional[torch.Tensor] = None
        self._mean_res: Optional[torch.Tensor] = None
        self._n_active_bins: Optional[int] = None
        self._refl_subset_idx: Optional[torch.Tensor] = None  # parent rows used

    # ------------------------------------------------------------------
    # Wilson curve
    # ------------------------------------------------------------------

    def _wilson_curve(self, mean_res: torch.Tensor) -> torch.Tensor:
        """Expected ``<|F|^2>`` per bin from the Wilson model."""
        # s = 1/(2d) -> s^2 = 1/(4 d^2)
        s_sq = 1.0 / (4.0 * mean_res.clamp(min=1e-3) ** 2)
        # <|F|^2> = K * exp(-2 * B_W * s^2)
        return torch.exp(self._log_K - 2.0 * self._B_W * s_sq)

    def _build_bin_assignment(self) -> None:
        """
        Resolve work-set reflections into ``nbins`` equal-population
        resolution bins. Cached for the lifetime of the target.
        """
        data = self._data
        # Pick the work set (refinement set). Fall back to all reflections
        # if no work subset is available (rare; e.g. tests with synthetic data).
        # The work subset already applies the validity masks, so the indices are
        # restricted to valid reflections (matching the X-ray target's mask).
        work = getattr(data, "work", None)
        if work is not None and work.n > 0:
            refl_idx = work.indices
        else:
            refl_idx = torch.arange(len(data.hkl), device=data.device)
        res = data.resolution.index_select(0, refl_idx)
        # Equal-population bins on the work-set resolution.
        order = torch.argsort(res)
        n = res.numel()
        nbins = min(self.nbins, max(1, n // 50))
        bin_assign = torch.empty(n, dtype=torch.long, device=res.device)
        edges = torch.linspace(0, n, nbins + 1, device=res.device).round().long()
        for b in range(nbins):
            start = int(edges[b].item())
            end = int(edges[b + 1].item())
            bin_assign[order[start:end]] = b
        # Per-bin mean resolution
        mean_res = torch.zeros(nbins, device=res.device, dtype=res.dtype)
        counts = torch.zeros(nbins, device=res.device, dtype=res.dtype)
        mean_res.scatter_add_(0, bin_assign, res)
        counts.scatter_add_(0, bin_assign, torch.ones_like(res))
        mean_res = mean_res / counts.clamp(min=1.0)
        self._bin_idx = bin_assign
        self._mean_res = mean_res
        self._n_active_bins = nbins
        self._refl_subset_idx = refl_idx

    def _fit_K_from_observed(self) -> None:
        """
        Fit the prefactor ``K`` of the Wilson curve from observed binned
        intensities so the prior is centered on the observed scale.
        """
        wilson_b = getattr(self._data, "wilson_b", None)
        if wilson_b is None:
            self._data._calculate_wilson_b()
            wilson_b = self._data.wilson_b
        device = self._data.device
        self._B_W = torch.tensor(float(wilson_b), device=device)

        F_obs = self._data.F.index_select(0, self._refl_subset_idx)
        I_obs = F_obs.float() ** 2
        nbins = self._n_active_bins
        mean_obs = torch.zeros(nbins, device=device, dtype=I_obs.dtype)
        counts = torch.zeros(nbins, device=device, dtype=I_obs.dtype)
        mean_obs.scatter_add_(0, self._bin_idx, I_obs)
        counts.scatter_add_(0, self._bin_idx, torch.ones_like(I_obs))
        mean_obs = mean_obs / counts.clamp(min=1.0)

        s_sq = 1.0 / (4.0 * self._mean_res.clamp(min=1e-3) ** 2)
        # Solve log K from each bin and average for a robust estimate.
        log_K_per_bin = torch.log(mean_obs.clamp(min=self.eps)) + 2.0 * self._B_W * s_sq
        self._log_K = log_K_per_bin.mean().detach()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """
        Squared log-deviation of calculated intensity from the Wilson curve.

        ``bin_mean`` mode reduces to one term per resolution bin;
        ``per_reflection`` mode keeps one term per work-set reflection.
        """
        if fcalc is None:
            fcalc = self._model(self._data.hkl)

        if self._bin_idx is None:
            self._build_bin_assignment()
        if self._log_K is None or self._B_W is None:
            self._fit_K_from_observed()

        # Apply scaler to F_calc so it sits on the F_obs scale, then keep
        # only the work-set rows.
        F_calc_scaled = torch.abs(self._scaler(fcalc))
        F_calc_subset = F_calc_scaled.index_select(0, self._refl_subset_idx)
        I_calc = F_calc_subset ** 2

        if self.mode == "rice":
            # Zero-structure Wilson prior = the ML X-ray Rice likelihood with
            # the centroid pinned to F=0 and the width pinned to the Wilson
            # mean intensity Sigma(s_h). Mirrors ``_ml_xray_loss_math_eager``
            # term-by-term with F_calc=0, eb=Sigma: acentric -> Rayleigh NLL,
            # centric -> half-normal NLL. Per-reflection NLL in nats, so the
            # registered weight is a true prior strength (1.0 = honest joint
            # with the data likelihood). epsilon(h)=1 to match the X-ray
            # target's multiplicity convention.
            res_subset = self._data.resolution.index_select(
                0, self._refl_subset_idx
            )
            Sigma = self._wilson_curve(res_subset).clamp(min=1e-6)  # <|F|^2>(s)
            F = F_calc_subset
            centric_all = self._data.centric
            if centric_all is not None:
                centric = centric_all.index_select(0, self._refl_subset_idx)
            else:
                centric = torch.zeros_like(F, dtype=torch.bool)
            nll_acentric = -torch.log(2.0 * F / Sigma + 1e-12) + I_calc / Sigma
            nll_centric = (
                -0.5 * torch.log(2.0 / (math.pi * Sigma) + 1e-12)
                + I_calc / (2.0 * Sigma)
            )
            nll = torch.where(centric, nll_centric, nll_acentric)
            nll = torch.where(
                torch.isfinite(nll), nll, torch.full_like(nll, 1e6)
            )
            # SUM over work reflections, matching the ML X-ray target's
            # reduction. EnsembleRefinement registers this with the same
            # 1/n_work weight it gives xray/work, so both raw losses are
            # total-nats sums on one scale and the strength dial stays O(1).
            return nll.sum()

        if self.mode == "per_reflection":
            # Each reflection vs the Wilson curve at ITS OWN resolution.
            # No within-bin averaging — this is the strong, high-dimensional
            # ridge that forbids individual reflections from straying.
            res_subset = self._data.resolution.index_select(
                0, self._refl_subset_idx
            )
            wilson_expected = self._wilson_curve(res_subset)
            diff = (
                torch.log(I_calc.clamp(min=self.eps))
                - torch.log(wilson_expected.clamp(min=self.eps))
            )
            return (diff ** 2).mean()

        # bin_mean mode (default): per-bin mean intensity vs the curve.
        nbins = self._n_active_bins
        device = I_calc.device
        mean_calc = torch.zeros(nbins, device=device, dtype=I_calc.dtype)
        counts = torch.zeros(nbins, device=device, dtype=I_calc.dtype)
        mean_calc = mean_calc.scatter_add(0, self._bin_idx, I_calc)
        counts = counts.scatter_add(0, self._bin_idx, torch.ones_like(I_calc))
        mean_calc = mean_calc / counts.clamp(min=1.0)

        wilson_expected = self._wilson_curve(self._mean_res)
        diff = (
            torch.log(mean_calc.clamp(min=self.eps))
            - torch.log(wilson_expected.clamp(min=self.eps))
        )
        return (diff ** 2).mean()
