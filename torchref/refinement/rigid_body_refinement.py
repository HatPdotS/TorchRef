"""
Multi-resolution rigid-body refinement strategy.

Wraps an existing :class:`~torchref.refinement.lbfgs_refinement.LBFGSRefinement`,
swaps the model's ``xyz`` container for a
:class:`~torchref.model.rigid_xyz.RigidXYZTensor` via
:meth:`~torchref.model.model.Model.use_rigid_xyz`, and runs an LBFGS step at
each cutoff in a coarse → fine resolution schedule. Only the xray target and
the non-bonded (vdW) geometry term are active — chains are internally rigid
by construction but can still clash against each other.
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
    """

    DEFAULT_LBFGS_KWARGS = dict(
        lr=1.0,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

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
                self._rebind_for_data(original_data.cut_res(highres=float(d_min)))
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
    def _rebind_for_data(self, data, model=None):
        """Point scaler + targets + loss-state at the given ReflectionData."""
        ref = self.refinement
        if model is None:
            model = ref.model
        ref.reflection_data = data

        # Fresh scaler with new bin structure for the new hkl set.
        ref.scaler = Scaler(
            model,
            data,
            nbins=getattr(ref, "nbins", 20),
            verbose=ref.verbose,
            device=ref.device,
        )
        ref.scaler.initialize()

        xray_mode = getattr(ref, "target_mode", "bhattacharyya")
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
            # Active targets during rigid-body refinement: xray + non-bonded
            # vdW. Internal bonded geometry is rigid by construction; ADP /
            # occupancy are frozen. Zero every other registered target.
            keep_names = {"xray", "geometry/nonbonded"}
            for name in list(state.targets.keys()):
                if name not in keep_names:
                    state.set_weight(name, 0.0)

            # Build LBFGS over the rigid leaves + scaler parameters.
            rigid_params = [
                rigid_model.xyz.euler_angles,
                rigid_model.xyz.translations,
            ]
            params = rigid_params + list(ref.scaler.parameters())
            optimizer = torch.optim.LBFGS(
                params,
                max_iter=self.iterations_per_step,
                **self.DEFAULT_LBFGS_KWARGS,
            )
            rigid_model.reset_cache()
            state.step(
                optimizer,
                context=f"rigid_body[d_min={d_min:.2f}]",
            )
            if ref.verbose > 0:
                try:
                    rwork, rfree = ref.get_rfactor()
                    print(
                        f"  rigid-body d_min={d_min:.2f}: "
                        f"Rwork={rwork:.4f} Rfree={rfree:.4f}"
                    )
                except Exception:
                    pass
        finally:
            # Restore weights.
            for name, w in original_weights.items():
                state.set_weight(name, w)
        return state
