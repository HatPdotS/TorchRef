"""Two-moment intensity target for time-resolved collections.

Merged Bragg intensities see the crystal-to-crystal activation distribution only through
its first two moments. With the branching among excited components conserved, the mixture
is exactly linear in the activation fraction, so the intensity is exactly quadratic and::

    <I> = |F(alpha_mean)|^2  +  sigma_alpha^2 |dF/dalpha|^2

holds for *any* activation distribution and any number of components -- an identity, not a
truncation. The first term is what every existing target models; the second is the variance
the coherent model discards, and it is strictly positive, phase-blind, and largest exactly
where the difference signal is.

The target works in **intensities** rather than amplitudes on purpose: the French-Wilson
conversion reshapes precisely the quadratic information the second moment lives in, so an
amplitude formulation would fit a distorted version of the quantity it is trying to measure.
Members must therefore carry ``I``/``SIGI``; there is no ``F**2`` fallback, because that
would silently reintroduce the distortion.
"""

from typing import TYPE_CHECKING, Dict, List

import torch

from torchref.base.metrics.rfactor import rfactor_work_free
from torchref.base.targets.xray_likelihoods import (
    gaussian_per_refl,
    intensity_var_from_sigma_obs,
)
from torchref.utils.stats import (
    VERBOSITY_DEBUG,
    VERBOSITY_ESSENTIAL,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

from .base import CollectionXrayTarget

if TYPE_CHECKING:
    from torchref.io.datasets.collection import DatasetCollection
    from torchref.model.model_collection import ModelCollection
    from torchref.scaling.scaler_base import ScalerBase


class CollectionTwoMomentIntensityTarget(CollectionXrayTarget):
    """
    Gaussian intensity likelihood at the two-moment forward model.

    Inherits the subset selector, the cache-reset discipline and the stats shape from
    :class:`~torchref.refinement.targets.collection.base.CollectionXrayTarget`, so it
    cannot disagree with the amplitude targets about which reflections it fits.

    The forward model is built from **one** set of per-component structure factors,
    contracted twice: once with the fractions to get the mean, once with the activation
    Jacobian to get its derivative. Both contractions go through the shared scaler, which
    is affine in ``F_calc`` and mixes the bulk solvent linearly in the weights -- so the
    second contraction returns the correctly scaled derivative rather than needing a
    separate differentiation path.

    With ``lambda_twin`` fixed at zero the variance branch is not built at all. That makes
    the coherent limit identical rather than merely equal: multiplying a live ``dF`` branch
    by exactly zero would still propagate a non-finite ``F_calc`` into the loss.

    Parameters
    ----------
    dataset_collection : DatasetCollection
        Members must all carry intensities.
    model_collection : ModelCollection
        Supplies the components, the fractions and the activation moments.
    scaler : ScalerBase, optional
        Shared scaler. Needs ``forward_batched`` to scale the batch in one pass; without
        one the unscaled mixture is used.
    use_work_set : bool, optional
        Legacy bool; superseded by ``use_set``.
    use_set : str, optional
        Canonical 3-way subset selector ``"work"``/``"free"``/``"val"``.
    verbose : int, optional
        Verbosity level.
    base_weight : float, optional
        Multiplies the summed loss on the work set. Intensities are squared amplitudes,
        so this target's loss and gradient are on a completely different scale from the
        amplitude targets it sits beside -- left at 1.0 it swamps them and the geometry
        restraints with it. Default 1.0; use :meth:`calibrate_base_weight` to set it
        from the data rather than by hand.

    Raises
    ------
    ValueError
        On construction, if any fitted dataset carries no intensities.
    """

    name: str = "collection_two_moment_intensity"

    def __init__(
        self,
        dataset_collection: "DatasetCollection",
        model_collection: "ModelCollection",
        scaler: "ScalerBase" = None,
        use_work_set: bool = True,
        use_set: str = None,
        verbose: int = 0,
        base_weight: float = 1.0,
    ):
        super().__init__(
            dataset_collection,
            model_collection,
            scaler=scaler,
            use_work_set=use_work_set,
            use_set=use_set,
            verbose=verbose,
        )
        self.base_weight = float(base_weight)
        # Fail here rather than inside the first loss evaluation: LossState probes a
        # target's forward at registration, and a traceback from there is much harder to
        # trace back to "this MTZ had no intensity columns".
        missing = [
            key for key in self._keys() if dataset_collection[key].I is None
        ]
        if missing:
            raise ValueError(
                f"Datasets {missing} carry no intensities. The two-moment target fits "
                f"merged intensities directly -- converting amplitudes back with F**2 "
                f"would reintroduce the French-Wilson distortion it exists to avoid. "
                f"Supply reflection files with I/SIGI columns."
            )

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------

    def _row_indices(self, keys: List[str]) -> List[int]:
        """Rows of the collection's fraction matrix corresponding to ``keys``."""
        order = self._model_collection.keys()
        return [order.index(k) for k in keys]

    def _scale_batch(self, fcalc_batch, weights):
        """Apply the shared scaler to a ``[T, R]`` batch with per-row weights."""
        scaler = self._scaler
        if scaler is None:
            return fcalc_batch
        return scaler.forward_batched(fcalc_batch, weights)

    def intensity_model(self, recalc: bool = False) -> torch.Tensor:
        """The two-moment predicted intensities, shape ``(n_datasets, n_reflections)``.

        Parameters
        ----------
        recalc : bool, optional
            Force recomputation of the component structure factors.

        Returns
        -------
        torch.Tensor
            Predicted intensities, rows aligned with :meth:`_keys`.
        """
        dc, mc = self._dataset_collection, self._model_collection
        keys = self._keys()
        rows = self._row_indices(keys)

        components = dc.component_structure_factors(mc, recalc=recalc)
        weights = mc.fractions_matrix()[rows]

        mean = self._scale_batch(
            mc.mix_component_fcalcs(components, weights), weights
        )
        intensity = mean.abs() ** 2

        sigma_alpha_sq = mc.sigma_alpha_sq
        if self._variance_is_live(sigma_alpha_sq):
            jacobian = mc.activation_jacobian()[rows]
            derivative = self._scale_batch(
                mc.mix_component_fcalcs(components, jacobian), jacobian
            )
            intensity = intensity + sigma_alpha_sq * derivative.abs() ** 2
        return intensity

    def _variance_is_live(self, sigma_alpha_sq) -> bool:
        """Whether the second moment contributes.

        False only when the dispersion is *exactly* zero and not refinable, in which case
        the derivative branch is skipped entirely rather than multiplied by zero.
        """
        mc = self._model_collection
        if mc._lambda_fixed is None:
            return True
        return bool(sigma_alpha_sq.detach().ne(0).any())

    def forward(self) -> torch.Tensor:
        """Summed Gaussian NLL of the observed intensities under the two-moment model."""
        dc = self._dataset_collection
        keys = self._keys()
        if not keys:
            return torch.zeros((), device=dc.hkl.device)

        # Clear cached forwards so a preceding no-grad stats()/get_rfactor() call cannot
        # leave a detached tensor that breaks the loss backward.
        self._reset_model_caches()

        model = self.intensity_model(recalc=False)
        obs = dc.stack_I_obs(keys).to(model.dtype)
        sigma = dc.stack_I_sigma(keys).to(model.dtype)
        mask = dc.stack_masks(keys, use_set=self.use_set)

        # Real reflection files carry non-finite intensities (excluded rows, and rows
        # French-Wilson rejected). Masking the *loss* is not enough: a NaN observation
        # makes the residual NaN, and torch.where selects the finite branch for the
        # value while still backpropagating NaN through the branch it discarded. So the
        # observations are sanitised into the mask BEFORE they reach the graph.
        valid = torch.isfinite(obs) & torch.isfinite(sigma)
        mask = mask & valid
        obs = torch.where(valid, obs, torch.zeros_like(obs))
        sigma = torch.where(valid, sigma, torch.ones_like(sigma))

        # The residual is formed and masked BEFORE the Gaussian, so a masked-out row
        # contributes an exact zero rather than a value that merely gets multiplied by
        # zero. That matters if the model is ever non-finite on an unfitted row: here the
        # `where` discards it, whereas `nll * mask` would propagate NaN into the sum.
        # Hence the Gaussian is evaluated at (residual, 0) rather than (obs, model).
        residual = torch.where(mask, obs - model, torch.zeros_like(obs))
        nll = gaussian_per_refl(
            residual,
            torch.zeros_like(residual),
            intensity_var_from_sigma_obs(sigma, mask),
            var_floor=0.0,
        )
        total = (nll * mask).sum()
        # Applied on the work set only, matching CollectionMLTarget: the free-set value
        # is a diagnostic and must stay comparable across weightings.
        if self.use_work_set:
            total = self.base_weight * total
        return total

    # ------------------------------------------------------------------
    # Weight calibration
    # ------------------------------------------------------------------

    def calibrate_base_weight(
        self, reference, parameters, ratio: float = 1.0, floor: float = 1e-12
    ) -> float:
        """Set ``base_weight`` so this target pushes as hard as ``reference``.

        Matched on the **gradient norm** with respect to the refined parameters, not on
        the loss value. A large loss with a flat gradient moves nothing, so loss
        magnitude is the wrong thing to equalise; what competes with the geometry and
        similarity restraints is the size of the step this term asks for.

        This is the per-cycle gradient-ratio weighting the collection targets' base
        weights were a stopgap for, applied once at setup rather than every cycle --
        enough to put an intensity target and an amplitude target on the same footing,
        which is otherwise a several-orders-of-magnitude mismatch.

        Parameters
        ----------
        reference : Target
            The target to match, normally the difference target already driving the
            refinement.
        parameters : iterable of torch.nn.Parameter
            The parameters actually being refined; only those with ``requires_grad``
            are used.
        ratio : float, optional
            Desired ratio of this target's gradient norm to the reference's. Default
            1.0 (equal footing); below 1 makes this target the junior partner.
        floor : float, optional
            Guard for a vanishing reference gradient.

        Returns
        -------
        float
            The ``base_weight`` that was set.
        """
        params = [p for p in parameters if p.requires_grad]
        if not params:
            raise ValueError("No refinable parameters given; cannot calibrate.")

        def _grad_norm(target, scale_out=1.0):
            grads = torch.autograd.grad(
                target.forward(), params, retain_graph=False, allow_unused=True
            )
            total = sum(
                float((g.detach() ** 2).sum()) for g in grads if g is not None
            )
            return (total**0.5) / scale_out

        saved = self.base_weight
        self.base_weight = 1.0
        try:
            own = _grad_norm(self)
        finally:
            self.base_weight = saved

        ref = _grad_norm(reference)
        if own <= floor:
            if self.verbose:
                print(
                    "  two-moment calibration: own gradient is ~0, leaving "
                    f"base_weight at {self.base_weight:.4g}"
                )
            return self.base_weight

        self.base_weight = float(ratio * max(ref, floor) / own)
        if self.verbose:
            print(
                f"  two-moment weight calibration: |grad_ref|={ref:.4g}, "
                f"|grad_self|={own:.4g}  ->  base_weight={self.base_weight:.4g}"
            )
        return self.base_weight

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_rfactor(self) -> Dict[str, object]:
        """Per-dataset R-work / R-free against the two-moment amplitudes.

        The reported amplitude is ``sqrt(I_model)``, the RMS amplitude the two-moment
        model actually predicts -- not ``|F(alpha_mean)|``, which is only its first term
        and would not correspond to the loss being minimised.

        Overrides the base implementation, which is per-pair and would recompute the
        component stack once per dataset.
        """
        dc = self._dataset_collection
        keys = self._keys()
        per_dataset: Dict[str, tuple] = {}
        rworks: List[float] = []
        rfrees: List[float] = []

        with torch.no_grad():
            model = self.intensity_model(recalc=True)
            amplitudes = model.clamp(min=0.0).sqrt()
            for row, key in enumerate(keys):
                rwork, rfree = rfactor_work_free(dc[key], amplitudes[row])
                per_dataset[key] = (rwork, rfree)
                rworks.append(rwork)
                rfrees.append(rfree)

        return {
            "per_dataset": per_dataset,
            "rwork_pct": self._percentiles(rworks),
            "rfree_pct": self._percentiles(rfrees),
        }

    def stats(self) -> Dict[str, StatEntry]:
        """Base collection X-ray stats plus the activation moments.

        ``dI_frac`` is the mean fraction of the predicted intensity carried by the second
        moment. It is what separates "the dispersion refined to zero" from "the dispersion
        was never refined", which are otherwise indistinguishable in the summary.
        """
        out = super().stats()
        mc = self._model_collection

        with torch.no_grad():
            alpha = float(mc.alpha_mean)
            lam = float(mc.lambda_twin)
            sigma_sq = float(mc.sigma_alpha_sq)

            out["base_weight"] = stat(self.base_weight, VERBOSITY_STANDARD)
            out["alpha_mean"] = stat(alpha, VERBOSITY_ESSENTIAL)
            out["lambda_twin"] = stat(lam, VERBOSITY_ESSENTIAL)
            out["sigma_alpha_sq"] = stat(sigma_sq, VERBOSITY_STANDARD)
            out["alpha_sd"] = stat(sigma_sq**0.5, VERBOSITY_STANDARD)

            keys = self._keys()
            if keys and self._variance_is_live(mc.sigma_alpha_sq):
                rows = self._row_indices(keys)
                components = self._dataset_collection.component_structure_factors(
                    mc, recalc=True
                )
                jacobian = mc.activation_jacobian()[rows]
                derivative = self._scale_batch(
                    mc.mix_component_fcalcs(components, jacobian), jacobian
                )
                variance_term = mc.sigma_alpha_sq * derivative.abs() ** 2
                total = self.intensity_model(recalc=False)
                mask = self._dataset_collection.stack_masks(
                    keys, use_set=self.use_set
                )
                denom = total[mask].abs().clamp(min=1e-12)
                out["dI_frac"] = stat(
                    float((variance_term[mask] / denom).mean()), VERBOSITY_STANDARD
                )
            else:
                out["dI_frac"] = stat(0.0, VERBOSITY_STANDARD)

            branching = mc.branching()
            for name, row in mc._branching_rows.items():
                for k in range(branching.shape[1]):
                    out[f"q_{name}_{k + 1}"] = stat(
                        float(branching[row, k]), VERBOSITY_DEBUG
                    )
        return out
