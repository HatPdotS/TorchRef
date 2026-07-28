from typing import TYPE_CHECKING, Optional, Tuple

import torch

from torchref.base.reciprocal import get_scattering_vectors
from torchref.base.targets.xray_ml_sigmaa import (
    SigmaAEstimator,
    epsilon_from_hkl,
    ml_xray_loss_beta_math,
)

from .base import XrayTarget
from .gaussian import GaussianXrayTarget
from .least_squares import LeastSquaresXrayTarget
from .rice import RiceXrayTarget

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model import Model
    from torchref.scaling.scaler_base import Scaler


class MaximumLikelihoodXrayTarget(XrayTarget):
    """Maximum-likelihood σ_A (Read MLF) target.

    The model mean is ``|F_calc|`` (Luzzati ``alpha`` ≡ 1 — gauge-absorbed by the
    scaler, see :mod:`torchref.base.targets.xray_ml_sigmaa`) and the conditional
    variance is ``epsilon * beta``, where ``beta`` is the per-shell model-error
    variance estimated by maximum likelihood on the **free** set. The variance
    ``beta`` is owned by this target via a :class:`SigmaAEstimator` and refreshed
    once per optimizer-step block through :meth:`maintenance`.

    The simpler raw-σ Rice likelihood (``beta = sigma**2``) is
    :class:`~torchref.refinement.targets.xray.rice.RiceXrayTarget`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The σ_A model-error variance is owned by the target, not the scaler.
        self._sigma_a = SigmaAEstimator()
        # Model-independent per-reflection geometry (multiplicity + d*²),
        # cached per ReflectionData instance (mirrors the bins cache pattern).
        self._eps_cache: torch.Tensor = None
        self._dss_cache: torch.Tensor = None
        self._geom_dataid: int = None

    def _geom(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(epsilon, d_star_sq)`` over the full data HKL, cached per data id.

        Both are model-independent (pure multiplicity / reciprocal geometry), so
        they are computed once per :class:`ReflectionData` and reused.
        """
        dataid = id(self._data)
        if self._eps_cache is None or self._geom_dataid != dataid:
            sg = getattr(self._data, "spacegroup", None)
            eps = epsilon_from_hkl(self._data.hkl, sg)
            s = get_scattering_vectors(self._data.hkl, self._data.cell)
            # d*² = |s|² = 4 * (|s|/2)² (the scaler's _s_half_sq convention).
            dss = (torch.norm(s, dim=1) ** 2).to(eps.dtype)
            self._eps_cache = eps
            self._dss_cache = dss
            self._geom_dataid = dataid
        return self._eps_cache, self._dss_cache

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        """Read-MLF loss with free-set σ_A variance ``epsilon * beta``.

        Parameters
        ----------
        fcalc : torch.Tensor, optional
            Pre-computed structure factors. If provided, uses these instead of
            computing from the model.

        Returns
        -------
        torch.Tensor
            Summed ML loss on this target's set (work or free).
        """
        sub = self._subset()

        # Full-size scaled |F_calc| (aligned to data.hkl). beta is estimated on
        # the full free set, so it needs the full-size arrays.
        if fcalc is not None:
            F_calc_full = self.get_F_calc_scaled(fcalc=fcalc)
        else:
            F_calc_full = self.get_F_calc_scaled(self._data.hkl_for_sf(), recalc=False)

        eps_full, dss_full = self._geom()
        eps_full = eps_full.to(F_calc_full.dtype)
        dss_full = dss_full.to(F_calc_full.dtype)
        F_obs_full = self._data.get_corrected_data()[0].to(F_calc_full.dtype).reshape(-1)
        centric_full = self._data.centric
        free_mask = self._data.free.mask

        # Detached (beta, epsilon); estimated on the free set, cached until
        # maintenance() resets it.
        beta, eps = self._sigma_a.get(
            F_obs_full, F_calc_full, centric_full, eps_full, dss_full, free_mask
        )

        F_obs = sub.F
        F_calc = sub.select(F_calc_full)
        beta_c = sub.select(beta).to(F_obs.dtype)
        eps_c = sub.select(eps).to(F_obs.dtype) if eps is not None else None
        centric = sub.centric
        return ml_xray_loss_beta_math(
            F_obs, F_calc, beta_c, centric, mask=None, epsilon=eps_c
        )

    def maintenance(self) -> None:
        """Invalidate the cached ``beta`` so it re-estimates from the updated
        model on the next forward (``LossState`` calls this after each
        optimizer-step block)."""
        self._sigma_a.reset()


def create_xray_target(
    data: "ReflectionData" = None,
    model: "Model" = None,
    scaler: "Scaler" = None,
    mode: str = "ml",
    use_work_set: bool = True,
    sigma_m_scale: float = 1.0,
    verbose: int = 0,
    device: Optional[torch.device] = None,
    use_set: str = None,
) -> XrayTarget:
    """
    Factory function to create an X-ray target.

    Parameters
    ----------
    data : ReflectionData
        Reference to ReflectionData object. Required for forward().
    model : Model or ModelFT, optional
        Reference to Model object for F_calc computation.
        If None, fcalc must be provided when calling forward().
    scaler : Scaler, optional
        Reference to Scaler object.
    mode : str, optional
        Target mode: ``'gaussian'``, ``'ls'``, ``'ls_wunit_k1'``, ``'rice'``,
        ``'ml'``, ``'bhattacharyya'``, or ``'bhattacharyya_ensemble'`` (experimental).
        Default is ``'ml'`` (maximum-likelihood Read MLF with free-set Luzzati
        σ_A variance ``epsilon*beta``).
        ``'rice'`` is the simpler raw-σ Rice likelihood (``beta = sigma**2``).
        ``'ls_wunit_k1'`` is Phenix-style least squares with unit weights and a
        per-bin closed-form optimal scale recomputed at every gradient call
        (does not use the external scaler). ``'ml_sigmaa'`` is a deprecated alias
        for ``'ml'``.
    use_work_set : bool, optional
        Use work set (True) or test set (False). Default is True.
    sigma_m_scale : float, optional
        Global multiplier applied to σ_m; used only by the ``'bhattacharyya'``
        and ``'bhattacharyya_ensemble'`` modes. Default is 1.0.
    verbose : int, optional
        Verbosity level. Default is 0.
    device : torch.device, optional
        Device to pin model/data/scaler onto before constructing the target.
    use_set : str, optional
        Canonical 3-way subset selector (``"work"``/``"free"``/``"val"``);
        takes precedence over ``use_work_set``. Default is None.

    Returns
    -------
    XrayTarget
        Appropriate XrayTarget instance.
    """
    import warnings

    # Legacy spelling — ``ml_sigmaa`` is the same as ``ml`` now.
    if mode == "ml_sigmaa":
        warnings.warn(
            "Target mode 'ml_sigmaa' is deprecated; use 'ml' instead. "
            "It now resolves to the same MaximumLikelihoodXrayTarget.",
            DeprecationWarning,
            stacklevel=2,
        )
        mode = "ml"

    # Device reconciliation is ``DataTarget.__init__``'s job now (it calls
    # ``_adopt_device(model, data, scaler)`` before allocating anything), so
    # doing it again here would be a second copy of the same policy, free to
    # drift. ``device`` is forwarded instead.
    kwargs = dict(
        data=data,
        model=model,
        scaler=scaler,
        use_work_set=use_work_set,
        verbose=verbose,
        use_set=use_set,
        device=device,
    )
    if mode == "gaussian":
        return GaussianXrayTarget(**kwargs)
    elif mode == "ls":
        return LeastSquaresXrayTarget(**kwargs)
    elif mode == "ls_wunit_k1":
        # Phenix-style: unit weights, SINGLE GLOBAL K recomputed every
        # gradient call. The "k1" in the target name means "K_one" — one
        # K, not per-bin (Phenix's update_all_scales fits a single
        # k_overall, aniso off at d>3Å). n_bins=1 collapses the
        # closed-form c[bins] to a single scalar — matches Phenix exactly.
        # Gradient w.r.t. θ treats c as constant via .detach() in forward.
        return LeastSquaresXrayTarget(
            weighting="unit",
            scale_mode="binwise_optimal",
            n_bins=1,
            **kwargs,
        )
    elif mode == "rice":
        return RiceXrayTarget(**kwargs)
    elif mode == "ml":
        return MaximumLikelihoodXrayTarget(**kwargs)
    elif mode == "bhattacharyya":
        from .bhattacharyya import BhattacharyyaXrayTarget

        return BhattacharyyaXrayTarget(
            sigma_m_scale=sigma_m_scale,
            **kwargs,
        )
    elif mode == "bhattacharyya_ensemble":
        from torchref.experimental.ensemble.ensemble_bhattacharyya import (
            EnsembleBhattacharyyaTarget,
        )

        return EnsembleBhattacharyyaTarget(
            sigma_m_scale=sigma_m_scale,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown X-ray target mode: {mode}")
