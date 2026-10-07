"""Multi-resolution rigid-body refinement strategy.

Wraps an :class:`~torchref.refinement.lbfgs_refinement.LBFGSRefinement`, swaps the
model's ``xyz`` for a :class:`~torchref.model.rigid_xyz.RigidXYZTensor` via
:meth:`~torchref.model.model.Model.use_rigid_xyz`, and runs one LBFGS step per cutoff
coarse to fine. Above 6 Å the xray target is Phenix-style ``ls_wunit_k1`` (unit weights,
one global K refit every gradient call) over a Scaler cut to one isotropic coefficient,
anisotropy and bulk solvent; at 6 Å and below it is ``ml`` with a full Scaler.
"""

import copy
from typing import List, Optional

import torch

from torchref.refinement.loss_state import LossState
from torchref.scaling.scaler import Scaler


class RigidBodyRefinementStep:
    """Multi-resolution rigid-body refinement.

    Parameters
    ----------
    refinement : LBFGSRefinement
        Refinement whose model is refined in place, its ``xyz`` swapped for a
        :class:`~torchref.model.rigid_xyz.RigidXYZTensor` (see ``commit``). Its data,
        scaler, targets and weights are left as they were.
    cutoffs : list of float, optional
        High-resolution cutoffs (Å), coarse to fine. ``None`` generates a schedule from the
        native data resolution via :meth:`default_cutoffs`.
    iterations_per_step : int, optional
        ``max_iter`` per cutoff. The default 30 **under-converges** in practice; raise it
        for production.
    commit : bool, optional
        If True (default), bake the final coordinates into a per-atom xyz container on
        the same model so later refinement sees normal per-atom xyz. False leaves the
        rigid xyz installed and ``adp``, ``u`` and ``occupancy`` frozen until
        :meth:`~torchref.model.model.Model.restore_xyz_from_rigid` is called
        (``commit=True`` there keeps the transform).
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
    ):
        self.refinement = refinement
        self.cutoffs = cutoffs
        self.iterations_per_step = int(iterations_per_step)
        self.commit = bool(commit)

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
        return "ls_wunit_k1" if d_min > cls.TARGET_SWITCH_RES else "ml"

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------
    @staticmethod
    def _sandbox(ref):
        """Shallow clone of ``ref`` whose attribute assignments do not reach ``ref``.

        ``__dict__``, ``_modules``, ``_parameters`` and ``_buffers`` are private copies,
        so the ``reflection_data``, ``scaler`` and x-ray targets that
        :meth:`_rebind_for_data` assigns per cutoff stay on the clone. The objects they
        hold are shared: the model, so refined xyz reaches the caller by identity, and
        the caller's ``ReflectionData``, whose resolution mask :meth:`_run` restores.
        """
        sandbox = copy.copy(ref)
        sandbox.__dict__ = dict(ref.__dict__)
        for slot in ("_modules", "_parameters", "_buffers"):
            if slot in sandbox.__dict__:
                sandbox.__dict__[slot] = dict(ref.__dict__[slot])
        return sandbox

    def run(self):
        """Step through every cutoff coarse to fine and return
        ``[(d_min, LossState), ...]``.
        """
        real = self.refinement
        self.refinement = self._sandbox(real)
        try:
            return self._run()
        finally:
            self.refinement = real

    def _run(self):
        ref = self.refinement
        original_data = ref.reflection_data

        native_dmin = float(original_data.d_min)
        cutoffs = (
            self.cutoffs
            if self.cutoffs is not None
            else self.default_cutoffs(native_dmin)
        )

        # ``filter_by_resolution`` masks in place and returns ``self``, so each
        # cutoff below stamps ``masks["resolution"]`` on the caller's own object and
        # rebinding restores nothing. Snapshot it (or its absence) to put back.
        had_resolution_mask = "resolution" in original_data.masks
        saved_resolution_mask = (
            original_data.masks["resolution"].clone() if had_resolution_mask else None
        )

        def restore_resolution_mask():
            if had_resolution_mask:
                original_data.masks["resolution"] = saved_resolution_mask
            else:
                original_data.masks.pop("resolution", None)

        # Swap the model's xyz container in place for a RigidXYZTensor.
        ref.model.use_rigid_xyz()

        history = []
        try:
            for d_min in cutoffs:
                xray_mode = self._xray_mode_for_cutoff(d_min)
                self._rebind_for_data(
                    original_data.filter_by_resolution(d_min=float(d_min)),
                    xray_mode=xray_mode,
                )
                step_state = self._run_one_cutoff(d_min)
                history.append((float(d_min), step_state))
        finally:
            restore_resolution_mask()

        if self.commit:
            ref.model.restore_xyz_from_rigid(commit=True)

        return history

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _rebind_for_data(self, data, xray_mode, model=None):
        """Point a new scaler and the ``xray_mode`` targets at ``data``.

        For ``ls_wunit_k1`` cutoffs the Scaler is built with ``nbins=1`` -- the mask-based
        bulk-solvent term is added to F_calc and the LS target's closed-form ``c[bins]``
        owns
        the overall scaling. Other modes get a fresh full Scaler with ``ref.nbins`` bins.
        """
        ref = self.refinement
        if model is None:
            model = ref.model
        ref.reflection_data = data

        ref.scaler = Scaler(
            model, data,
            nbins=1 if xray_mode == "ls_wunit_k1" else getattr(ref, "nbins", 20),
            # T_0 == 1, so a single coefficient IS one global K. ``ls_wunit_k1`` owns its
            # own closed-form scale, and any further isotropic freedom here double-scales
            # against it -- silently, since both fits succeed.
            n_iso_coeff=1 if xray_mode == "ls_wunit_k1" else getattr(
                ref, "n_iso_coeff", 6),
            verbose=ref.verbose,
            device=ref.device,
        )
        # Only the x-ray half of _init_targets: the step optimizes against x-ray data
        # alone, and TotalGeometryTarget / TotalADPTarget would be constructed here
        # purely to be left unused -- NonBondedTarget's pair list among them.
        # get_scales() still runs: it cold-starts the new scaler, whose parameters the
        # x-ray target reads.
        ref._build_xray_targets(xray_mode)
        ref.get_scales()
        model.reset_cache()

    def _run_one_cutoff(self, d_min: float):
        ref = self.refinement
        rigid_model = ref.model

        # Active targets during rigid-body refinement: x-ray only. Internal bonded
        # geometry is rigid by construction; ADP / occupancy are frozen. Inter-chain vdW
        # is intentionally off -- Phenix runs rigid-body without atomistic restraints,
        # and we have measured that vdW adds no signal here and destabilizes the
        # coarsest cutoff.
        #
        # A state of its own rather than the refinement's: nothing here has to be undone
        # afterwards, no maintenance() hook of a target we are not using can fire (in
        # particular NonBondedTarget rebuilds its VDW pair list whenever atoms drift
        # >1 A, seconds of work for a term that is not in the sum), and the caller's
        # state keeps its targets and any weights registered on it.
        #
        # Weight 1.0: with one term the weight is a scalar on the whole objective, and
        # 1.0 is what DEFAULT_GROUP_WEIGHTS gives x-ray anyway.
        state = LossState(device=ref.device)
        state.register_target("xray", ref.xray_target_work)
        state.set_weight("xray", 1.0)
        state.cache_losses()

        # Rigid parameters only. The body target centres on
        # ``alpha*|F_calc|`` and ``alpha`` absorbs a rescaling of ``F_calc``
        # exactly, so the scale has a flat direction here -- the rule
        # ``SCALE_TARGETS`` states for the scale fit. ``refine_scaler``
        # (objective ``ls``) owns the scale, between cutoffs. Omitting them
        # is enough: ``LossState.run`` freezes leaves the optimizer lacks.
        rigid_params = [
            rigid_model.xyz.euler_angles,
            rigid_model.xyz.translations,
        ]

        rigid_model.reset_cache()
        opt = torch.optim.LBFGS(
            rigid_params,
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
                    f"(lbfgs, iters={self.iterations_per_step}): "
                    f"Rwork={rwork:.4f} Rfree={rfree:.4f}"
                )
            except Exception:
                pass
        return state
