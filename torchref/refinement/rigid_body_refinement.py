"""
Multi-resolution rigid-body refinement strategy.

Wraps an existing :class:`~torchref.refinement.lbfgs_refinement.LBFGSRefinement`,
swaps the model's ``xyz`` container for a
:class:`~torchref.model.rigid_xyz.RigidXYZTensor` via
:meth:`~torchref.model.model.Model.use_rigid_xyz`, and runs an LBFGS (or Adam)
step at each cutoff in a coarse → fine resolution schedule.

At coarse resolutions (d > 6 Å) the xray target is Phenix-style
``ls_wunit_k1`` — least squares with unit weights and a per-bin optimal
scale recomputed at every gradient call (no external scaler). This avoids
the frozen-scale bias that pinned the L2 minimum at the starting positions.
At high resolutions (d ≤ 6 Å) the target switches to ``ml_sigmaa`` with the
normal Scaler module.
"""

from typing import List, Optional

import torch

from torchref.scaling.scaler import Scaler


class RigidBodyRefinementStep:
    """
    Multi-resolution rigid-body refinement.

    Parameters
    ----------
    refinement : LBFGSRefinement
        Refinement object whose ``model`` will be swapped for a rigid model
        for the duration of the run.
    cutoffs : list of float, optional
        High-resolution cutoffs (Å) to step through, coarse → fine. If
        ``None``, a default schedule is generated from the native data
        resolution via :meth:`default_cutoffs`.
    iterations_per_step : int, optional
        ``max_iter`` for the per-cutoff LBFGS step. Default 30.
    commit : bool, optional
        If ``True`` (default), the final rotated/translated coordinates are
        baked back into a plain ``ModelFT`` so subsequent regular refinement
        sees a normal per-atom xyz. If ``False``, the rigid model is left
        installed on the refinement.
    optimizer : str, optional
        ``"lbfgs"`` (default) or ``"adam"``.
    adam_lr : float, optional
        Learning rate when ``optimizer="adam"``. Default 0.01.
    """

    DEFAULT_LBFGS_KWARGS = dict(
        lr=1.0,
        history_size=5,  # scitbx LBFGS default m=5 (vs PyTorch's typical 100)
        line_search_fn="strong_wolfe",
    )

    # Phenix's mmtbx.refinement.rigid_body target_auto_switch_resolution.
    TARGET_SWITCH_RES = 6.0

    def __init__(
        self,
        refinement,
        cutoffs: Optional[List[float]] = None,
        iterations_per_step: int = 30,
        commit: bool = True,
        optimizer: str = "lbfgs",
        adam_lr: float = 0.01,
        with_solvent: bool = True,
    ):
        self.refinement = refinement
        self.cutoffs = cutoffs
        self.iterations_per_step = int(iterations_per_step)
        self.commit = bool(commit)
        if optimizer not in ("lbfgs", "adam"):
            raise ValueError(f"optimizer must be 'lbfgs' or 'adam', got {optimizer!r}")
        self.optimizer = optimizer
        self.adam_lr = float(adam_lr)
        self.with_solvent = bool(with_solvent)

    # -----------------------------------------------------------------------
    # Schedule
    # -----------------------------------------------------------------------
    @staticmethod
    def default_cutoffs(native_dmin: float) -> List[float]:
        """Geometric schedule from a coarse start down to ``native_dmin``."""
        native = float(native_dmin)
        if native <= 0:
            raise ValueError(f"native_dmin must be > 0, got {native}")
        if native >= 6.0:
            cuts = [native * 1.5, native * 1.2, native]
        else:
            coarse = max(6.0, native * 2.0)
            cuts = [coarse, (coarse * native) ** 0.5, native]
        # Enforce strictly decreasing and >= native.
        out: List[float] = []
        for c in cuts:
            c = max(float(c), native)
            if not out or c < out[-1] - 1e-6:
                out.append(c)
        if out[-1] > native + 1e-6:
            out.append(native)
        return out

    @classmethod
    def _xray_mode_for_cutoff(cls, d_min: float) -> str:
        """Phenix-style target auto-switch."""
        return "ls_wunit_k1" if d_min > cls.TARGET_SWITCH_RES else "ml_sigmaa"

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------
    def run(self):
        ref = self.refinement
        original_data = ref.reflection_data

        native_dmin = float(original_data.get_max_res())
        cutoffs = (
            self.cutoffs
            if self.cutoffs is not None
            else self.default_cutoffs(native_dmin)
        )

        # Swap the model's xyz container in place for a RigidXYZTensor.
        ref.model.use_rigid_xyz()

        history = []
        try:
            for d_min in cutoffs:
                xray_mode = self._xray_mode_for_cutoff(d_min)
                # Phenix's "update_all_scales BEFORE rigid body, frozen
                # during" only fires for the LS path (coarse cutoffs);
                # for ml_sigmaa we fall back to the original code that
                # co-refines the scaler with rigid params each cutoff.
                if xray_mode == "ls_wunit_k1":
                    self._fit_and_freeze_scaler(original_data)
                else:
                    self._fitted_scaler = None
                self._rebind_for_data(
                    original_data.cut_res(highres=float(d_min)),
                    xray_mode=xray_mode,
                )
                step_state = self._run_one_cutoff(d_min)
                history.append((float(d_min), step_state))
        finally:
            # Restore full-resolution data view.
            self._rebind_for_data(original_data)

        if self.commit:
            # Bake rigid coords into a fresh MixedTensor and re-bind
            # scaler/targets/loss-state against the per-atom xyz.
            ref.model.restore_xyz_from_rigid(commit=True)
            self._rebind_for_data(original_data)

        return history

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _fit_and_freeze_scaler(self, full_data):
        """Re-fit the scaler on FULL data with current atoms; freeze params.

        Mirrors Phenix's ``update_all_scales`` step: fits ``K_overall``,
        anisotropic ``U``, and bulk-solvent ``k_sol``/``B_sol`` against
        the entire reflection set at the current atomic positions, then
        sets ``requires_grad=False`` on every scaler leaf so the
        subsequent rigid-body LBFGS sees a fixed scale + solvent layer.

        ``nbins=1`` keeps the per-bin scale collapsed to a single global
        K (matching Phenix's k_overall semantics). A fresh ``Scaler`` is
        built every call because the bin-edge cache + solvent mask are
        tied to the data instance; this also forces a fresh
        ``update_solvent()`` from the current atoms.
        """
        ref = self.refinement
        ref.reflection_data = full_data
        ref.scaler = Scaler(
            ref.model,
            full_data,
            nbins=1,
            verbose=ref.verbose,
            device=ref.device,
        )
        ref.scaler.initialize()
        # Internal LBFGS refines log_scale + U + solvent params against
        # the full data with current atoms.
        ref.scaler.refine_lbfgs()
        # Freeze every scaler leaf so the rigid LBFGS can't touch them.
        for p in ref.scaler.parameters():
            p.requires_grad = False
        # Cache the fitted scaler so _rebind_for_data can reuse it.
        self._fitted_scaler = ref.scaler

    @staticmethod
    def _copy_scaler_params(src, dst):
        """Copy K_overall (log_scale), aniso U, and bulk solvent params
        from ``src`` scaler to ``dst`` scaler. ``src``/``dst`` must use
        the same ``nbins`` (per-bin shape compatibility)."""
        with torch.no_grad():
            if (hasattr(src, "log_scale") and hasattr(dst, "log_scale")
                    and src.log_scale.shape == dst.log_scale.shape):
                dst.log_scale.data.copy_(src.log_scale.data)
            if hasattr(src, "U") and hasattr(dst, "U"):
                dst.U.data.copy_(src.U.data)
            sv_src = getattr(src, "solvent", None)
            sv_dst = getattr(dst, "solvent", None)
            if sv_src is not None and sv_dst is not None:
                sv_dst.log_k_solvent.data.copy_(sv_src.log_k_solvent.data)
                sv_dst.b_solvent.data.copy_(sv_src.b_solvent.data)
                if (isinstance(getattr(sv_dst, "phase_offset", None),
                               torch.nn.Parameter)
                        and isinstance(getattr(sv_src, "phase_offset", None),
                                       torch.nn.Parameter)):
                    sv_dst.phase_offset.data.copy_(sv_src.phase_offset.data)
        # Refresh the cached additive bulk-solvent SFs since k_sol/B_sol
        # have new values.
        dst._f_sol_raw = None

    def _init_targets_no_refit(self, xray_mode: str):
        """Like ``Refinement._init_targets`` but skips ``get_scales``,
        which would re-run scaler LBFGS and wipe the frozen params."""
        from torchref.refinement.targets.xray import create_xray_target
        from torchref.refinement.targets.combined import (
            TotalGeometryTarget, TotalADPTarget,
        )

        ref = self.refinement
        ref.xray_target_work = create_xray_target(
            model=ref.model,
            data=ref.reflection_data,
            scaler=ref.scaler,
            mode=xray_mode,
            use_work_set=True,
            verbose=ref.verbose,
        )
        ref.xray_target_test = create_xray_target(
            model=ref.model,
            data=ref.reflection_data,
            scaler=ref.scaler,
            mode=xray_mode,
            use_work_set=False,
            verbose=ref.verbose,
        )
        ref.geometry_target = TotalGeometryTarget(ref.model, verbose=ref.verbose)
        ref.adp_target = TotalADPTarget(ref.model, verbose=ref.verbose)

    def _rebind_for_data(self, data, model=None, xray_mode=None):
        """Point scaler + targets + loss-state at the given ReflectionData.

        For ``ls_wunit_k1`` cutoffs we build a **solvent-only** Scaler: the
        bulk-solvent term (mask-based, Phenix-style) is added to F_calc, but
        the per-bin scale ``log_scale`` is fixed at 0 (so ``K_overall = 1``)
        and the anisotropy ``U`` is removed (so ``aniso = 1``). The
        closed-form per-bin scale ``c[bins]`` in the LS target keeps owning
        the overall scaling — the scaler's contribution is purely additive
        (F_calc + k_sol·exp(-B_sol·s²/4)·F_mask).

        For ``ml_sigmaa`` and other modes a fresh full Scaler is built.
        """
        ref = self.refinement
        if model is None:
            model = ref.model
        ref.reflection_data = data

        if xray_mode is None:
            xray_mode = getattr(ref, "target_mode", "ml_sigmaa")

        # If _fit_and_freeze_scaler has been called this cycle, build a
        # cut-data scaler that REUSES the frozen full-data fit (params
        # are independent of the reflection set; only the per-reflection
        # internal state — scattering vectors, bin edges — needs to be
        # rebuilt for the cut). Skip the per-cutoff LBFGS refit so the
        # frozen full-data params actually stick through the rigid LBFGS.
        fitted = getattr(self, "_fitted_scaler", None)
        if fitted is not None and xray_mode == "ls_wunit_k1":
            ref.scaler = Scaler(
                model, data,
                nbins=1,
                verbose=ref.verbose,
                device=ref.device,
            )
            ref.scaler.initialize()        # builds s, bins, mask (atoms)
            self._copy_scaler_params(src=fitted, dst=ref.scaler)
            for p in ref.scaler.parameters():
                p.requires_grad = False
            # NB: _init_targets calls get_scales which would refit the
            # scaler via LBFGS, wiping our copied params. Build targets
            # directly without get_scales.
            self._init_targets_no_refit(xray_mode=xray_mode)
        else:
            ref.scaler = Scaler(
                model, data,
                nbins=1 if xray_mode == "ls_wunit_k1" else getattr(ref, "nbins", 20),
                verbose=ref.verbose,
                device=ref.device,
            )
            ref.scaler.initialize()
            ref._init_targets(xray_mode=xray_mode)

        ref.reset_loss_state()
        # Clear cached LBFGS optimizers (they were built over the old model's
        # parameters and would now point at stale leaves).
        if hasattr(ref, "_persistent_optimizers"):
            ref._persistent_optimizers.clear()
        model.reset_cache()

    def _run_one_cutoff(self, d_min: float):
        ref = self.refinement
        rigid_model = ref.model

        state = ref.complete_loss_state()

        # Snapshot weights so we can restore after the step.
        original_weights = dict(state.weights)
        try:
            # Active targets during rigid-body refinement: xray only. Internal
            # bonded geometry is rigid by construction; ADP / occupancy are
            # frozen. Inter-chain vdW is intentionally off — Phenix runs
            # rigid-body without atomistic restraints and we've observed vdW
            # adds no signal here and destabilizes the coarsest cutoff.
            #
            # We DROP non-xray targets from state entirely (not just zero
            # their weight) so their maintenance() hooks don't fire during
            # the rigid-body LBFGS. In particular ``NonBondedTarget``
            # rebuilds its VDW pair list whenever atoms drift >1 Å, a
            # multi-second recomputation that's pure waste when the
            # target weight is 0. State is rebuilt fresh on the next
            # cutoff via _rebind_for_data → reset_loss_state →
            # _init_targets, so this drop is local to this cutoff.
            keep_names = {"xray"}
            for name in list(state.targets.keys()):
                if name not in keep_names:
                    state.targets.pop(name, None)
                    original_weights.pop(name, None)

            rigid_params = [
                rigid_model.xyz.euler_angles,
                rigid_model.xyz.translations,
            ]

            # Decide whether to use the inner-cycle (mask-refresh) loop.
            # Triggered when the scaler has a bulk-solvent component whose
            # mask depends on atom positions (ls_wunit_k1 path here, with
            # log_scale frozen + no anisotropy). For the ml_sigmaa path the
            # scaler is fully refit between cutoffs and co-optimized with
            # rigid params in a single LBFGS — leave that branch unchanged.
            use_inner_cycles = (
                self.optimizer == "lbfgs"
                and ref.scaler is not None
                and getattr(ref.scaler, "solvent", None) is not None
                and ref.scaler.log_scale.requires_grad is False
            )

            if use_inner_cycles:
                self._run_inner_cycles(
                    d_min, state, rigid_params, n_inner=5
                )
            else:
                # Single-shot path: rigid params + scaler params co-optimized.
                if ref.scaler is not None:
                    opt_params = rigid_params + list(ref.scaler.parameters())
                else:
                    opt_params = rigid_params

                rigid_model.reset_cache()
                if self.optimizer == "adam":
                    opt = torch.optim.Adam(opt_params, lr=self.adam_lr)
                    for _ in range(self.iterations_per_step):
                        opt.zero_grad()
                        loss = state.aggregate()
                        loss.backward()
                        opt.step()
                else:
                    opt = torch.optim.LBFGS(
                        opt_params,
                        max_iter=self.iterations_per_step,
                        **self.DEFAULT_LBFGS_KWARGS,
                    )
                    state.step(
                        opt,
                        context=f"rigid_body[d_min={d_min:.2f}]",
                    )
            if ref.verbose > 0:
                try:
                    rwork, rfree = ref.get_rfactor()
                    print(
                        f"  rigid-body d_min={d_min:.2f} "
                        f"({self.optimizer}, iters={self.iterations_per_step}): "
                        f"Rwork={rwork:.4f} Rfree={rfree:.4f}"
                    )
                except Exception:
                    pass
        finally:
            # Restore weights.
            for name, w in original_weights.items():
                state.set_weight(name, w)
        return state

    def _run_inner_cycles(self, d_min, state, rigid_params, n_inner: int = 5):
        """Phenix-style inner cycle: refresh mask + refit solvent + rigid LBFGS.

        Used at coarse cutoffs with a solvent-only scaler. Each inner cycle:

        1. Rebuild the bulk-solvent mask FFT at current atom positions
           (``SolventModel.update_solvent`` + invalidate ``_f_sol_raw``).
        2. Run a short LBFGS over the solvent parameters (``log_k_solvent``,
           ``b_solvent``, and ``phase_offset`` if it's a Parameter) to refit
           the solvent contribution against the current F_calc.
        3. Run the rigid-body LBFGS for ``iterations_per_step // n_inner``
           iterations over rigid params only (mask frozen here, matching
           Phenix's per-inner-cycle locality assumption).
        """
        ref = self.refinement
        rigid_model = ref.model
        solvent = ref.scaler.solvent

        # Build the solvent parameter list — only the ones that are
        # actually refinable. If ``_fit_and_freeze_scaler`` ran upstream
        # these will all be False and the list is empty → solvent refit
        # is skipped (parameters are already at their full-data fit).
        solvent_params = []
        for name in ("log_k_solvent", "b_solvent"):
            p = getattr(solvent, name, None)
            if isinstance(p, torch.nn.Parameter) and p.requires_grad:
                solvent_params.append(p)
        phase = getattr(solvent, "phase_offset", None)
        if isinstance(phase, torch.nn.Parameter) and phase.requires_grad:
            solvent_params.append(phase)

        # In the inner-cycle path, iterations_per_step is the per-INNER
        # LBFGS max_iter (Phenix uses 25). Total rigid LBFGS work per
        # cutoff = n_inner × iterations_per_step.
        iters_per_inner = self.iterations_per_step

        for inner in range(n_inner):
            # (a) Refresh the bulk-solvent mask at current positions.
            solvent.update_solvent()
            ref.scaler._f_sol_raw = None
            rigid_model.reset_cache()

            # (b) Refit solvent params (k_sol, B_sol[, phase]) against F_obs
            # — only if any solvent params are actually refinable.
            if solvent_params:
                sol_opt = torch.optim.LBFGS(
                    solvent_params,
                    max_iter=20,
                    **self.DEFAULT_LBFGS_KWARGS,
                )
                state.step(
                    sol_opt,
                    context=f"rigid_body[d_min={d_min:.2f},inner={inner + 1},solvent]",
                )

            # (c) Rigid LBFGS — mask + solvent params frozen here.
            # lr=5 in DEFAULT_LBFGS_KWARGS bumps the iter-1 trial step
            # closer to scitbx's ``stp1 = 1/||g||₂`` initialization (PyTorch
            # scales by ``1/||g||₁`` instead, which is ~√24× smaller for
            # our 24-dim params and gives a more conservative first step).
            rigid_model.reset_cache()
            rb_opt = torch.optim.LBFGS(
                rigid_params,
                max_iter=iters_per_inner,
                **self.DEFAULT_LBFGS_KWARGS,
            )
            state.step(
                rb_opt,
                context=f"rigid_body[d_min={d_min:.2f},inner={inner + 1},rigid]",
            )

            # (d) Bake the current rigid transform into original_xyz and
            # zero euler+translation, matching Phenix's per-macro-cycle
            # reset (mmtbx/refinement/rigid_body.py:344-370). Keeps the
            # next solve in the small-angle regime with a clean Hessian.
            rigid_model.xyz.bake()
            rigid_model.reset_cache()

    def _newton_lm_steps(
        self,
        state,
        rigid_params,
        n_steps: int,
        context: str = "",
        lambda_init: float = 1e-3,
        lambda_min: float = 1e-8,
        lambda_max: float = 1e6,
    ):
        """Damped Newton (Levenberg-Marquardt) steps on rigid params.

        Computes the full Hessian under the EAGER kernel engine (which
        supports second derivatives) at the current point, then takes
        ``delta = -(H + lambda * diag(|H|.clamp_min(eps)))^-1 * g``.
        Adapts ``lambda`` per step: shrink ×0.5 on accepted step,
        expand ×4 on rejected step. Total per-cycle cost ~ n_steps full
        Hessians at small parameter dim (~24 for 4 chains).
        """
        from torch.autograd.functional import hessian
        from torchref.utils import Engine, use_engine

        ref = self.refinement
        rigid_model = ref.model
        device = rigid_params[0].device
        dtype = rigid_params[0].dtype
        n_total = sum(p.numel() for p in rigid_params)

        def get_flat():
            return torch.cat([p.detach().flatten() for p in rigid_params]).clone()

        def set_flat(x_flat):
            offset = 0
            for p in rigid_params:
                n = p.numel()
                with torch.no_grad():
                    p.copy_(x_flat[offset:offset + n].view_as(p))
                offset += n
            rigid_model.reset_cache()

        def loss_flat(x_flat):
            # IMPORTANT: this MUST attach to x_flat so we get gradients/
            # Hessian w.r.t. it. Re-package as views into params.
            offset = 0
            for p in rigid_params:
                n = p.numel()
                # Assign via .data so that the autograd graph follows
                # x_flat -> p -> model -> loss.
                p.data = x_flat[offset:offset + n].view_as(p).to(dtype=p.dtype)
                offset += n
            rigid_model.reset_cache()
            return state.aggregate()

        lam = float(lambda_init)
        x = get_flat()
        prev_loss = float(loss_flat(x).detach())
        # Restore x (loss_flat assigned in place).
        set_flat(x)

        for it in range(n_steps):
            # Gradient via standard autograd over rigid_params.
            rigid_model.reset_cache()
            loss = state.aggregate()
            grads = torch.autograd.grad(loss, rigid_params, retain_graph=False)
            g = torch.cat([gr.flatten() for gr in grads]).detach()

            # Hessian via functional.hessian — requires eager kernels.
            x_now = get_flat()
            try:
                with use_engine(Engine.EAGER):
                    H = hessian(loss_flat, x_now, create_graph=False, vectorize=True)
            except Exception:
                # Fallback without vectorize (older PyTorch) or on engine error.
                with use_engine(Engine.EAGER):
                    H = hessian(loss_flat, x_now, create_graph=False)
            set_flat(x_now)
            H = 0.5 * (H + H.T)  # symmetrize

            # LM damping using |diag(H)| so units match per-parameter.
            diag_abs = H.diag().abs().clamp_min(1e-8)
            n = H.shape[0]
            I_d = torch.diag(diag_abs)

            accepted = False
            tried = 0
            while not accepted and lam <= lambda_max and tried < 8:
                A = H + lam * I_d
                try:
                    delta = -torch.linalg.solve(A, g)
                except RuntimeError:
                    lam = min(lam * 4.0, lambda_max)
                    tried += 1
                    continue

                x_new = x_now + delta
                new_loss = float(loss_flat(x_new).detach())
                if new_loss < prev_loss - 1e-12:
                    # Accept.
                    set_flat(x_new)
                    prev_loss = new_loss
                    lam = max(lam * 0.5, lambda_min)
                    accepted = True
                else:
                    # Reject — increase damping, retry from x_now.
                    set_flat(x_now)
                    lam = min(lam * 4.0, lambda_max)
                    tried += 1

            if not accepted:
                # Couldn't make progress at this point; stop early.
                if ref.verbose > 1:
                    print(f"  [{context}] Newton step {it + 1}: no progress, "
                          f"lambda={lam:.3e}, stopping")
                break
