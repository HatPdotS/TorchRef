"""Monolithic, macrocycle-free refinement using a differentiable model-error variance.

EXPERIMENTAL. :class:`MonolithicRefinement` is an :class:`LBFGSRefinement` that
drives the X-ray target with
:class:`~torchref.experimental.monolithic_refinement.targets.RiceSigmaMXrayTarget`
— the Read-MLF Rice likelihood whose conditional variance ``Sigma = epsilon * c *
sigma_m**2`` is built from the differentiable Fisher model-error variance, with the
calibration ``c`` co-refined.

Unlike the default flow (free-set ``beta`` re-estimated between optimizer blocks via
``maintenance()`` — a macrocycle in disguise), everything here lives in one autograd
graph: a single :meth:`refine_joint` co-refines ``xyz``, ``adp``, ``u``,
``occupancy``, the scaler body params (when ``corefine_scaler``), **and** the
error-model calibration ``c``. :meth:`refine_monolithic` simply iterates that one
joint step.

Example
-------
::

    from torchref.experimental.monolithic_refinement import MonolithicRefinement

    ref = MonolithicRefinement(data_file="data.mtz", pdb="model.pdb")
    ref.refine_monolithic(n_steps=5)
    rwork, rfree = ref.get_rfactor()
"""

import torch

from torchref.experimental.monolithic_refinement.targets import RiceSigmaMXrayTarget
from torchref.refinement.lbfgs_refinement import LBFGSRefinement


class MonolithicRefinement(LBFGSRefinement):
    """LBFGS refinement with a co-refined differentiable model-error variance.

    Parameters
    ----------
    sigma_m_calib_bins : int, optional
        DOF of the co-refined ``sigma_m`` calibration (1 = global scalar, default;
        ``> 1`` = per scaler resolution bin). Passed to
        :class:`RiceSigmaMXrayTarget`.

    Other parameters are inherited from :class:`LBFGSRefinement`. ``target_mode``
    is forced to ``"rice_sigma_m"``.
    """

    def __init__(
        self,
        *args,
        sigma_m_calib_bins: int = 1,
        use_density_solvent: bool = True,
        **kwargs,
    ):
        # Set before super().__init__ — set_xray_target_mode and setup_scaler
        # (both called inside) read these. Plain attribute assignment is safe
        # before nn.Module.__init__.
        self._sigma_m_calib_bins = int(sigma_m_calib_bins)
        # Select the differentiable density-derived bulk solvent (default for
        # this experimental, fully-differentiable refinement). Set False to use
        # the standard vdW-mask Scaler. setup_scaler() reads ``_scaler_class``.
        if use_density_solvent:
            from torchref.experimental.monolithic_refinement.density_scaler import (
                DensitySolventScaler,
            )

            self._scaler_class = DensitySolventScaler
        kwargs["target_mode"] = "rice_sigma_m"
        super().__init__(*args, **kwargs)

    def set_xray_target_mode(self, mode: str):
        """Build the work/test RiceSigmaMXrayTargets sharing one calibration.

        Falls back to the base implementation for any non-monolithic mode.
        """
        if mode != "rice_sigma_m":
            return super().set_xray_target_mode(mode)

        sigma_m_scale = getattr(self, "sigma_m_scale", 1.0)
        calib_bins = getattr(self, "_sigma_m_calib_bins", 1)
        common = dict(
            model=self.model,
            data=self.reflection_data,
            scaler=self.scaler,
            sigma_m_scale=sigma_m_scale,
            sigma_m_calib_bins=calib_bins,
            verbose=self.verbose,
        )
        work = RiceSigmaMXrayTarget(use_work_set=True, **common)
        # Test target shares the work target's calibration so R-free statistics
        # reflect the same c the work loss refines.
        test = RiceSigmaMXrayTarget(
            use_work_set=False,
            shared_log_sigma_m_scale=work.log_sigma_m_scale,
            **common,
        )
        self.xray_target_work = work
        self.xray_target_test = test
        # Drop any cached LossState so the new targets are picked up.
        if hasattr(self, "reset_loss_state"):
            self.reset_loss_state()

    def _error_model_params(self):
        """The co-refined model-error calibration (always in the joint optimizer)."""
        calib = getattr(self.xray_target_work, "log_sigma_m_scale", None)
        return [calib] if calib is not None else []

    def refine_joint(self):
        """One monolithic LBFGS step over body + scaler-body + error-model calib.

        Like :meth:`LBFGSRefinement.refine_joint` but also co-refines the
        differentiable ``sigma_m`` calibration ``c`` — the single error-model
        parameter — regardless of ``corefine_scaler`` (which gates only the
        scaler scale/U/solvent).
        """
        state = self.complete_loss_state()
        body = self.model.parameters_of_types(("xyz", "adp", "u", "occupancy"))
        params = body + self._scaler_body_params() + self._error_model_params()
        optimizer = torch.optim.LBFGS(params, **self.LBFGS_DEFAULTS)
        state.step(optimizer, context="monolithic_refinement.refine_joint")
        return state

    def refine_monolithic(self, n_steps: int = 5):
        """Run ``n_steps`` monolithic joint steps (no separate scaler/xyz/adp macrocycles).

        Returns
        -------
        tuple
            ``(rwork, rfree)`` after the final step.
        """
        for _ in range(n_steps):
            self.refine_joint()
        return self.get_rfactor()
