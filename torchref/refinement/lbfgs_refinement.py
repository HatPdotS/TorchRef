"""
LBFGS-based refinement framework for crystallographic structure refinement.

This module provides an LBFGS optimizer-based refinement approach which has been
shown to converge much faster than first-order optimizers (Adam, SGD, etc.).
LBFGS typically reaches near-convergence in just 1-2 macro cycles.

The refinement composes three pieces:

- A persistent :class:`~torchref.refinement.loss_state.LossState` built once via
  :meth:`~torchref.refinement.base_refinement.Refinement.complete_loss_state`.
- Persistent LBFGS optimizers (one per parameter group — xyz, adp+u+occupancy,
  and the joint set). These are created lazily on first use and reused across
  macro cycles so the construction cost is paid once.
- Scaler refinement, which runs its own local LossState + LBFGS step via
  :meth:`~torchref.scaling.scaler_base.ScalerBase.refine_lbfgs` and is invoked
  independently between body-parameter refinements.

Each body step clears the LBFGS curvature history for its own optimizer before
running. This is necessary because (a) the Hessian approximation does not
transfer across mode transitions (xyz → adp) and (b) scaler updates between
refine_xyz and refine_adp bump the parameters that feed the xray target, so
prior curvature information is stale.
"""

from typing import Optional

import numpy as np
import torch

from torchref.refinement.base_refinement import Refinement


class LBFGSRefinement(Refinement):
    """
    LBFGS-based refinement subclass using the L-BFGS optimizer for fast convergence.

    L-BFGS (Limited-memory BFGS) is a quasi-Newton optimization method that
    approximates the Hessian matrix, leading to much faster convergence than
    first-order methods.

    Key advantages:

    - Converges in 1-2 macro cycles (vs 5+ for Adam)
    - Better final R-factors
    - More stable convergence
    - Automatically handles step size via line search

    Parameters
    ----------
    target_mode : str, optional
        X-ray target mode ('gaussian', 'ls', 'rice', 'ml', 'bhattacharyya'). Default is 'ml'.
    *args
        Passed to parent Refinement class.
    **kwargs
        Passed to parent Refinement class.

    Attributes
    ----------
    target_mode : str
        Current X-ray target mode.

    Examples
    --------
    Basic usage::

        from torchref.refinement import LBFGSRefinement

        refinement = LBFGSRefinement(
            data_file='data.mtz',
            pdb='model.pdb',
            target_mode='ml'
        )
        refinement.refine(macro_cycles=2)
    """

    LBFGS_DEFAULTS = dict(
        lr=1.0,
        max_iter=20,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def __init__(
        self,
        *args,
        target_mode: str = "ml",
        sigma_m_scale: float = 1.0,
        corefine_scaler: bool = False,
        use_lossstate_scaler: bool = True,
        **kwargs,
    ):
        """
        Initialize LBFGS refinement.

        Parameters
        ----------
        target_mode : str, optional
            X-ray target mode ('gaussian', 'ls', 'rice', 'ml', 'bhattacharyya').
            Default is 'ml' (maximum-likelihood Read MLF with Luzzati σ_A).
        sigma_m_scale : float, optional
            Global multiplier for σ_m in the Bhattacharyya target only.
            Ignored for other target modes. Default 1.0.
        use_lossstate_scaler : bool, optional
            If True (default), :meth:`refine_scaler` uses the full
            :class:`LossState` with the body's x-ray target — so scaler and
            body steps share one consistent loss. If False, falls back to
            ``Scaler.refine_lbfgs`` which minimises a standalone
            ``nll_xray`` and can pull scales in a different direction than
            the body optimization.
        *args
            Passed to parent Refinement class.
        **kwargs
            Passed to parent Refinement class.
        """
        super().__init__(*args, **kwargs)

        self.sigma_m_scale = sigma_m_scale
        # Default False: hold the scaler fixed during the xyz/adp body steps and
        # only update it via the separate refine_scaler() step. Co-refining the
        # few high-leverage scaler params (scale / aniso U / bulk solvent) in the
        # same LBFGS as thousands of body params is ill-conditioned and drags
        # R-free up (validated on the AF-start benchmark).
        self.corefine_scaler = corefine_scaler
        # Set the X-ray target mode (uses the new target system from base class)
        self.set_xray_target_mode(target_mode)
        self.target_mode = target_mode
        self.use_lossstate_scaler = use_lossstate_scaler

        # Lazy persistent optimizers. Built on first access by
        # _lbfgs_for_types so that LBFGSRefinement instances without a
        # loaded model can still be constructed.
        self._persistent_optimizers: dict = {}

    def xray_loss(self):
        """
        Compute X-ray loss using the instantiated target.

        Returns
        -------
        torch.Tensor
            X-ray loss on work set.
        """
        return self.xray_loss_work()

    # =========================================================================
    # Persistent optimizer machinery
    # =========================================================================

    def _lbfgs_for_types(self, types: tuple) -> torch.optim.LBFGS:
        """Return a persistent LBFGS optimizer over the given parameter types.

        Optimizers are cached by the tuple of type names (e.g. ``("xyz",)``
        or ``("adp", "u", "occupancy")``) and reused across refinement
        calls. Curvature history must be cleared by the caller before each
        use via :meth:`_reset_lbfgs_history`.

        Parameters
        ----------
        types : tuple of str
            Parameter type names to include in the optimizer. Any of
            ``"xyz"``, ``"adp"``, ``"u"``, ``"occupancy"``.

        Returns
        -------
        torch.optim.LBFGS
            The cached optimizer, constructed on first call for this key.
        """
        key = tuple(types)
        opt = self._persistent_optimizers.get(key)
        if opt is None:
            params = self.model.parameters_of_types(types)
            if not params:
                raise RuntimeError(
                    f"No parameters found for types={types}; cannot build LBFGS."
                )
            opt = torch.optim.LBFGS(params, **self.LBFGS_DEFAULTS)
            self._persistent_optimizers[key] = opt
        return opt

    @staticmethod
    def _reset_lbfgs_history(optimizer: torch.optim.Optimizer) -> None:
        """Drop LBFGS curvature state so the next step starts from scratch.

        The LBFGS two-loop recursion depends on recent (s, y) pairs sampled
        under the *same* loss landscape. Between a refine_xyz and refine_adp
        call the active parameter set changes; between any two body calls
        the scaler's separate LBFGS has updated parameters the xray target
        reads from. Either way the stored curvature is stale and can produce
        bad search directions. Clearing state forces a fresh steepest-descent
        direction on the first inner iteration.
        """
        optimizer.state.clear()

    # =========================================================================
    # Refinement Methods
    # =========================================================================

    def refine_scaler(self):
        """Refine scaler parameters against the full refinement loss.

        Builds the body :class:`LossState` via
        :meth:`complete_loss_state`, constructs a fresh LBFGS optimizer
        over ``list(self.scaler.parameters())``, and delegates to
        :meth:`LossState.step`. Because ``state.step`` disables
        ``requires_grad`` on every loss leaf outside the optimizer's
        intent set, xyz / adp / u / occupancy are pinned for the duration
        — only scaler parameters move.

        The critical property is that the x-ray target used here is the
        same one the body :meth:`refine_xyz` and :meth:`refine_adp` see.
        The legacy :meth:`Scaler.refine_lbfgs` minimises a standalone
        ``nll_xray`` + ``U^2`` penalty, which can pull scales in a
        different direction than a ``bhattacharyya`` or ``ml`` body loss
        and leaves the body to chase a scaler that disagrees with its own
        objective.

        When ``use_lossstate_scaler`` is False, fall back to the legacy
        :meth:`Scaler.refine_lbfgs` path.

        Returns
        -------
        LossState or dict
            ``LossState`` with history if ``use_lossstate_scaler`` is
            True, otherwise the metrics dict from
            :meth:`Scaler.refine_lbfgs`.
        """
        if not self.use_lossstate_scaler:
            return self.scaler.refine_lbfgs()

        state = self.complete_loss_state()
        scaler_params = list(self.scaler.parameters())
        if not scaler_params:
            return state
        optimizer = torch.optim.LBFGS(scaler_params, **self.LBFGS_DEFAULTS)
        state.step(optimizer, context="lbfgs_refinement.refine_scaler")
        return state

    def refine_rigid_body(
        self,
        cutoffs=None,
        iterations_per_step: int = 30,
        commit: bool = True,
    ):
        """Multi-resolution per-chain rigid-body refinement.

        Swaps the model for a :class:`RigidModelFT` whose ``xyz`` exposes
        only per-chain ZYZ-Euler rotations and translations, then runs an
        LBFGS step at each cutoff in a coarse → fine schedule. Only the
        xray target and ``geometry/nonbonded`` (vdW) are active.

        Parameters
        ----------
        cutoffs : list of float, optional
            High-resolution cutoffs (Å), coarse → fine. Defaults to an
            auto-generated schedule from the native data resolution.
        iterations_per_step : int, optional
            ``max_iter`` for each per-cutoff LBFGS step. Default 30.
        commit : bool, optional
            If True (default), bakes the final coordinates back into a
            regular ``ModelFT`` so subsequent refinement uses per-atom xyz.

        Returns
        -------
        list of (d_min, LossState)
            Per-cutoff state.
        """
        from torchref.refinement.rigid_body_refinement import (
            RigidBodyRefinementStep,
        )

        step = RigidBodyRefinementStep(
            self,
            cutoffs=cutoffs,
            iterations_per_step=iterations_per_step,
            commit=commit,
        )
        return step.run()

    def refine_xyz(self):
        """Refine Cartesian coordinates jointly with scaler parameters.

        Scaler parameters (``log_scale``, ``U``, solvent terms) are
        included in the same LBFGS call as ``xyz``. The joint curvature
        lets xyz steps see the scaler as an anchor — residuals the scaler
        can absorb do not have to be chased by atomic motion — and the
        ``adp/scaler_U`` and ``adp/scaler_log_scale`` priors bite on every
        step, so nothing in the scaler drifts between refine_xyz and
        refine_adp calls.

        Returns
        -------
        LossState
            State with history containing before/after loss values.
        """
        state = self.complete_loss_state()
        body = self.model.parameters_of_types(("xyz",))
        params = body + self._scaler_body_params()
        optimizer = torch.optim.LBFGS(params, **self.LBFGS_DEFAULTS)
        state.step(optimizer, context="lbfgs_refinement.refine_xyz")
        return state

    def refine_adp(self):
        """Refine ADP / U / occupancy jointly with scaler parameters.

        Scaler parameters (``log_scale``, ``U``, solvent terms) are
        included in the same LBFGS call as the ADP-block body parameters
        so the joint curvature can slide along the atomic-B / scaler-U
        degeneracy ridge together with the ``adp/scaler_U`` regularizer.
        XYZ is left frozen.

        Returns
        -------
        LossState
            State with history containing before/after loss values.
        """
        state = self.complete_loss_state()
        body = self.model.parameters_of_types(("adp", "u", "occupancy"))
        params = body + self._scaler_body_params()
        optimizer = torch.optim.LBFGS(params, **self.LBFGS_DEFAULTS)
        state.step(optimizer, context="lbfgs_refinement.refine_adp")
        return state

    def _scaler_body_params(self):
        """Scaler parameters to co-refine inside the body (xyz/adp) steps.

        Returns the scaler parameter list when ``corefine_scaler`` is True
        (default, historical behaviour), else an empty list so the scaler
        is held fixed during xyz/adp and only updated by the separate
        :meth:`refine_scaler` step at each macro-cycle end. Co-refining the
        few high-leverage scaler params (scale, anisotropic U, bulk solvent)
        in the same LBFGS as thousands of xyz params is ill-conditioned and
        can drive the ML-NLL down while R goes up.
        """
        if getattr(self, "corefine_scaler", True):
            return list(self.scaler.parameters())
        return []

    def refine_joint(self):
        """Joint LBFGS over every refinable parameter in one step.

        Optimizes ``xyz``, ``adp``, ``u``, ``occupancy``, and every
        scaler parameter (``log_scale``, anisotropic ``U``, solvent
        terms) in a single LBFGS call. The joint curvature couples all
        of them through the same x-ray target and through the
        ``adp/scaler_U`` / ``adp/scaler_log_scale`` priors — unlike
        alternating refine_xyz → refine_adp, there's no "frozen partner"
        in either half that could lock the step into a locally bad
        direction.

        Returns
        -------
        LossState
            State with history containing before/after loss values.
        """
        state = self.complete_loss_state()
        body = self.model.parameters_of_types(("xyz", "adp", "u", "occupancy"))
        params = body + self._scaler_body_params()
        optimizer = torch.optim.LBFGS(params, **self.LBFGS_DEFAULTS)
        state.step(optimizer, context="lbfgs_refinement.refine_joint")
        return state

    def _refine_everything_lbfgs_single_cycle(self, nsteps: int = 1):
        """Joint LBFGS over xyz + adp + u + occupancy for one macro cycle.

        Used by :meth:`refine_everything`. Scaler is refined separately
        before the body step.
        """
        self.scaler.refine_lbfgs()
        state = self.complete_loss_state()
        optimizer = self._lbfgs_for_types(("xyz", "adp", "u", "occupancy"))
        self._reset_lbfgs_history(optimizer)
        state.run(
            optimizer,
            nsteps=nsteps,
            context="lbfgs_refinement._refine_everything_lbfgs_single_cycle",
        )
        return state

    def refine(self, macro_cycles=5):
        """
        Run full LBFGS refinement cycle (ADP + XYZ).

        Parameters
        ----------
        macro_cycles : int, optional
            Number of refinement cycles to perform. Default is 5.

        Returns
        -------
        dict
            History dictionary with all metrics per cycle (hierarchical structure).
        """
        i = 0

        while True:
            i += 1
            master_key = f"refinement_{i}"
            if master_key not in self.history:
                break

        self.history[master_key] = []

        # Clear logger history for fresh refinement
        self.logger.clear()

        for cycle in range(macro_cycles):
            cycle_dict = {
                "cycle": cycle + 1,
                "before_scaling": {},
                "after_scaling": {},
                "xyz": {"before": {}, "after": {}, "weights": {}},
                "adp": {"before": {}, "after": {}, "weights": {}},
            }

            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"LBFGS Refinement - Cycle {cycle+1}/{macro_cycles}")
                print(f"{'='*60}")

            with torch.no_grad():
                before_scaling = self.collect_metrics()
                cycle_dict["before_scaling"] = before_scaling

            if getattr(self.scaler, "solvent", None) is not None:
                self.scaler.solvent.update_solvent()
            self.reflection_data.find_outliers(self.model, self.scaler, z_threshold=5.0)

            with torch.no_grad():
                after_scaling = self.collect_metrics()
                cycle_dict["after_scaling"] = after_scaling
                if self.verbose > 0:
                    print(
                        f"After scaling: Rwork={after_scaling['rwork']:.4f}, "
                        f"Rfree={after_scaling['rfree']:.4f}"
                    )

            self.logger.record(label="before_xyz")
            cycle_dict["xyz"]["before"] = self.collect_metrics()

            self.refine_xyz()

            self.logger.record(label="after_xyz")
            cycle_dict["xyz"]["after"] = self.collect_metrics()
            if self.verbose > 0:
                self.logger.compare(
                    label_before="before_xyz",
                    label_after="after_xyz",
                    title="XYZ Refinement",
                )

            self.logger.record(label="before_adp")
            cycle_dict["adp"]["before"] = self.collect_metrics()

            self.refine_adp()

            self.logger.record(label="after_adp")
            cycle_dict["adp"]["after"] = self.collect_metrics()
            if self.verbose > 0:
                self.logger.compare(
                    label_before="before_adp",
                    label_after="after_adp",
                    title="ADP Refinement",
                )

            self.refine_scaler()

            self.history[master_key].append(cycle_dict)

        return self.history

    def refine_everything(self, macro_cycles=5):
        """
        Run full LBFGS refinement cycle (ADP + XYZ) without weight screening.

        Parameters
        ----------
        macro_cycles : int, optional
            Number of refinement cycles to perform. Default is 5.

        Returns
        -------
        dict
            History dictionary with all metrics per cycle (hierarchical structure).
        """
        self.model.unfreeze_all()
        i = 0

        while True:
            i += 1
            master_key = f"refinement_everything_{i}"
            if master_key not in self.history:
                break

        self.history[master_key] = []
        self.history["initial"] = self.collect_metrics()

        self.logger.clear()

        for cycle in range(macro_cycles):
            cycle_dict = {
                "cycle": cycle + 1,
                "before_scaling": {},
                "after_scaling": {},
                "after_refinement": {},
            }
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"LBFGS Refinement Everything - Cycle {cycle+1}/{macro_cycles}")
                print(f"{'='*60}")

            self.get_scales()

            self.logger.record(label="after_scaling")
            with torch.no_grad():
                after_scaling = self.collect_metrics()
                cycle_dict["after_scaling"] = after_scaling
                if self.verbose > 0:
                    print(
                        f"After scaling: Rwork={after_scaling['rwork']:.4f}, "
                        f"Rfree={after_scaling['rfree']:.4f}"
                    )

            self._refine_everything_lbfgs_single_cycle()

            self.logger.record(label="after_refinement")
            with torch.no_grad():
                after_refinement = self.collect_metrics()
                cycle_dict["after_refinement"] = after_refinement
                if self.verbose > 0:
                    print(
                        f"After refinement: Rwork={after_refinement['rwork']:.4f}, "
                        f"Rfree={after_refinement['rfree']:.4f}"
                    )
                    self.logger.compare(
                        label_before="after_scaling",
                        label_after="after_refinement",
                        title="Joint XYZ+ADP Refinement",
                    )

            self.history[master_key].append(cycle_dict)

        return self.history
