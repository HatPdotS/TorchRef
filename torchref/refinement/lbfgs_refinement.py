"""LBFGS-based refinement framework for crystallographic structure refinement.

As a quasi-Newton method LBFGS converges in far fewer macro cycles than first-order
optimizers; the production default is ``macro_cycles=5``. The refinement composes a
persistent :class:`~torchref.refinement.loss_state.LossState`, body steps over the
xyz, adp+u+occupancy and joint parameter groups, and scaler refinement which runs its
own local LossState + LBFGS step between body refinements.

**Each body step builds a fresh LBFGS over its parameter group, so curvature never
carries across steps.** The Hessian approximation does not transfer across a mode
transition (xyz -> adp), and scaler updates between body steps move parameters the
xray target reads, so retained curvature would be stale.
"""

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

        Swaps the model's ``xyz`` in place for a
        :class:`~torchref.model.rigid_xyz.RigidXYZTensor` that exposes only per-chain
        XYZ-Euler rotations and translations, then runs an LBFGS step at each cutoff,
        coarse to fine. Only the xray target is active.

        Parameters
        ----------
        cutoffs : list of float, optional
            High-resolution cutoffs (Å), coarse to fine. Defaults to a schedule generated from
            the native data resolution.
        iterations_per_step : int, optional
            ``max_iter`` per cutoff. The default 30 **under-converges** in practice; raise
            it for production.
        commit : bool, optional
            If True (default), bake the final coordinates into a per-atom xyz container
            on the same model so subsequent refinement uses per-atom xyz. False leaves
            the rigid xyz installed and ``adp``, ``u`` and ``occupancy`` frozen until
            :meth:`~torchref.model.model.Model.restore_xyz_from_rigid` is called
            (``commit=True`` there keeps the transform).

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

        Scaler parameters (``c_iso``, ``U``, solvent) join this call only when
        ``corefine_scaler`` is True; by default they are fixed here and updated by
        :meth:`~torchref.refinement.base_refinement.Refinement.refine_scaler`.
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
        they are fixed here and updated by
        :meth:`~torchref.refinement.base_refinement.Refinement.refine_scaler`.
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
        fallback matches the default so an instance built without ``__init__`` behaves
        the same.
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

        Used by :meth:`refine_everything`, which refits the scaler warm via
        :meth:`_refresh_scales` immediately beforehand; this method therefore touches
        only body parameters.
        """
        state = self.complete_loss_state()
        body = self.model.parameters_of_types(("xyz", "adp", "u", "occupancy"))
        optimizer = torch.optim.LBFGS(body, **self.LBFGS_DEFAULTS)
        state.run(
            optimizer,
            nsteps=nsteps,
            context="lbfgs_refinement._refine_everything_lbfgs_single_cycle",
        )
        return state

    def _refresh_scales(self):
        """Rebuild the solvent mask at the current coordinates, then refit the scaler.

        The per-cycle scale update of :meth:`refine` and :meth:`refine_everything`. Warm:
        the scale, anisotropy and solvent refined in the previous cycle are the starting
        point, where :meth:`~torchref.refinement.base_refinement.Refinement.get_scales`
        would reseed them.
        """
        if self.scaler is not None:
            self.scaler.update_solvent()
        return self.refine_scaler()

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

            # Before the `after_scaling` metrics below, so that label describes this cycle's
            # scaler rather than the previous one's.
            self._refresh_scales()

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

            self._refresh_scales()

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
