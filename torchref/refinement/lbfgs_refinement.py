"""LBFGS-based refinement framework for crystallographic structure refinement.

As a quasi-Newton method LBFGS converges in far fewer macro cycles than first-order
optimizers; the production default is ``macro_cycles=5``. The refinement composes a
persistent :class:`~torchref.refinement.loss_state.LossState`, persistent per-group
LBFGS optimizers (xyz, adp+u+occupancy, joint) created lazily and reused, and scaler
refinement which runs its own local LossState + LBFGS step between body refinements.

**Each body step clears its optimizer's LBFGS curvature history first.** The Hessian
approximation does not transfer across a mode transition (xyz -> adp), and scaler
updates between body steps move parameters the xray target reads, so retained curvature
is stale.
"""

from typing import Optional

import numpy as np
import torch

from torchref.refinement.base_refinement import Refinement


class LBFGSRefinement(Refinement):
    """Refinement driven by L-BFGS: fewer macro cycles, better final R-factors, and
    step size handled by the line search.

    Parameters
    ----------
    target_mode : str, optional
        X-ray target mode, default ``'ml'``; see
        :mod:`torchref.refinement.targets.xray._specs` for the taxonomy.
    *args, **kwargs
        Passed to :class:`~torchref.refinement.base_refinement.Refinement`.

    Examples
    --------
    ::

        refinement = LBFGSRefinement(data_file='data.mtz', pdb='model.pdb')
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
        corefine_scaler: bool = False,
        **kwargs,
    ):
        """Initialize LBFGS refinement.

        Parameters
        ----------
        target_mode : str, optional
            X-ray target mode. Default ``'ml'`` (Read MLF with Luzzati σ_A, centred on
            ``alpha*|F_calc|``); see :mod:`torchref.refinement.targets.xray._specs`.
        corefine_scaler : bool, optional
            Co-refine the scaler parameters in the same optimizer step as the body rather
            than in a separate scaler step. Default False.
        *args, **kwargs
            Passed to :class:`~torchref.refinement.base_refinement.Refinement`.
        """
        # Hand the mode to the base so it builds the targets ONCE, with the full
        # configuration -- a second build here reverts whatever it fails to forward
        # (see Refinement._xray_target_kwargs).
        kwargs.setdefault("xray_mode", target_mode)
        super().__init__(*args, **kwargs)

        # Default False: hold the scaler fixed during the xyz/adp body steps and only
        # update it via refine_scaler(). Co-refining a few high-leverage scaler params
        # in the same LBFGS as thousands of body params is ill-conditioned.
        self.corefine_scaler = corefine_scaler
        # Targets are already built for this mode by super().__init__(); no rebuild.
        self.target_mode = target_mode

        # Lazy persistent optimizers. Built on first access by
        # _lbfgs_for_types so that LBFGSRefinement instances without a
        # loaded model can still be constructed.
        self._persistent_optimizers: dict = {}

    def xray_loss(self):
        """X-ray loss on the work set, from the instantiated target."""
        return self.xray_loss_work()

    # =========================================================================
    # Persistent optimizer machinery
    # =========================================================================

    def _lbfgs_for_types(self, types: tuple) -> torch.optim.LBFGS:
        """The persistent LBFGS over ``types`` (any of ``"xyz"``, ``"adp"``, ``"u"``,
        ``"occupancy"``), cached by that tuple and reused across calls.

        **Callers must clear curvature via :meth:`_reset_lbfgs_history` before each use.**
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
        """Drop LBFGS curvature state so the next step starts from steepest descent.

        The two-loop recursion needs ``(s, y)`` pairs from the *same* landscape; between
        refine_xyz and refine_adp the active parameter set changes, and between any two body
        calls the scaler has moved parameters the xray target reads.
        """
        optimizer.state.clear()

    # =========================================================================
    # Refinement Methods
    # =========================================================================

    def refine_rigid_body(
        self,
        cutoffs=None,
        iterations_per_step: int = 30,
        commit: bool = True,
    ):
        """Multi-resolution per-chain rigid-body refinement.

        Swaps the model for a :class:`RigidModelFT` whose ``xyz`` exposes only per-chain
        XYZ-Euler rotations and translations, then runs an LBFGS step at each cutoff,
        coarse to
        fine. Only the xray target is active.

        Parameters
        ----------
        cutoffs : list of float, optional
            High-resolution cutoffs (Å), coarse to fine. Defaults to a schedule generated from
            the native data resolution.
        iterations_per_step : int, optional
            ``max_iter`` per cutoff. The default 30 **under-converges** in practice (9RTS needs
            >= 100); raise it for production. Under the solvent-only (``ls_wunit_k1``)
            inner-cycle path this is per *inner* cycle, so the total is
            ``n_inner * iterations_per_step``.
        commit : bool, optional
            If True (default), bake the final coordinates back into a regular ``ModelFT`` so
            subsequent refinement uses per-atom xyz.

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
        """LBFGS over the ``xyz`` body parameters; returns the LossState with history.

        Scaler parameters (``log_scale``, ``U``, solvent) join this call only when
        ``corefine_scaler`` is True; by default they are fixed here and updated by
        :meth:`refine_scaler`.
        """
        state = self.complete_loss_state()
        body = self.model.parameters_of_types(("xyz",))
        params = body + self._scaler_body_params()
        optimizer = torch.optim.LBFGS(params, **self.LBFGS_DEFAULTS)
        state.step(optimizer, context="lbfgs_refinement.refine_xyz")
        return state

    def refine_adp(self):
        """LBFGS over ``adp``, ``u`` and ``occupancy``, xyz frozen; returns the LossState.

        Scaler parameters join this call only when ``corefine_scaler`` is True; by default
        they are fixed here and updated by :meth:`refine_scaler`.
        """
        state = self.complete_loss_state()
        body = self.model.parameters_of_types(("adp", "u", "occupancy"))
        params = body + self._scaler_body_params()
        optimizer = torch.optim.LBFGS(params, **self.LBFGS_DEFAULTS)
        state.step(optimizer, context="lbfgs_refinement.refine_adp")
        return state

    def _scaler_body_params(self):
        """Scaler parameters to co-refine inside the body steps, or ``[]``.

        Non-empty only with ``corefine_scaler`` (opt-in, default False): co-refining a few
        high-leverage scaler params in the same LBFGS as thousands of xyz params is
        ill-conditioned and can drive the ML-NLL down while R goes up. The ``getattr``
        fallback
        matches the default so an instance built without ``__init__``
        (``create_from_state_dict``)
        behaves the same.
        """
        if getattr(self, "corefine_scaler", False):
            return list(self.scaler.parameters())
        return []

    def refine_joint(self):
        """Joint LBFGS over ``xyz``, ``adp``, ``u`` and ``occupancy`` in one step.

        The joint curvature couples them through the same x-ray target, so unlike
        alternating
        refine_xyz -> refine_adp there is no frozen partner to lock the step into a
        locally bad
        direction. Scaler parameters join only when ``corefine_scaler`` is True.
        """
        state = self.complete_loss_state()
        body = self.model.parameters_of_types(("xyz", "adp", "u", "occupancy"))
        params = body + self._scaler_body_params()
        optimizer = torch.optim.LBFGS(params, **self.LBFGS_DEFAULTS)
        state.step(optimizer, context="lbfgs_refinement.refine_joint")
        return state

    def _refine_everything_lbfgs_single_cycle(self, nsteps: int = 1):
        """Joint LBFGS over xyz + adp + u + occupancy for one macro cycle.

        Used by :meth:`refine_everything`, which fits the scaler via ``get_scales()``
        immediately beforehand; this method therefore touches only body parameters.
        """
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
        """Run ``macro_cycles`` cycles of ``refine_scaler`` -> ``refine_xyz`` ->
        ``refine_adp``.

        Contrast :meth:`refine_everything`, which optimizes xyz, ADP, U and occupancy
        jointly.
        Returns the hierarchical per-cycle history dict.
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
            # Before the `after_scaling` metrics below, so that label describes this cycle's
            # scaler rather than the previous one's.
            self.refine_scaler()

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

            self.history[master_key].append(cycle_dict)

        return self.history

    def refine_everything(self, macro_cycles=5):
        """Run ``macro_cycles`` cycles of one joint step over xyz, ADP, U and occupancy.

        Calls ``unfreeze_all`` first. Contrast :meth:`refine`, which alternates
        ``refine_scaler`` -> ``refine_xyz`` -> ``refine_adp``. Returns the hierarchical
        per-cycle history dict.
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
