"""
LBFGS-based refinement framework for crystallographic structure refinement.

This module provides an LBFGS optimizer-based refinement approach which has been
shown to converge much faster than first-order optimizers (Adam, SGD, etc.).
LBFGS typically reaches near-convergence in just 1-2 macro cycles.

Weight optimization follows the Phenix approach:
- Screen multiple weights on a log scale
- Run complete LBFGS optimization for each weight
- Select best weight based on Rfree while respecting gap constraints
"""

from typing import Optional

import numpy as np
import torch

from torchref.refinement.base_refinement import Refinement
from torchref.refinement.loss_state import LossState


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
        X-ray target mode ('gaussian', 'ls', or 'ml'). Default is 'ml'.
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

    def __init__(self, *args, target_mode: str = "ml", **kwargs):
        """
        Initialize LBFGS refinement.

        Parameters
        ----------
        target_mode : str, optional
            X-ray target mode ('gaussian', 'ls', or 'ml'). Default is 'ml'.
        *args
            Passed to parent Refinement class.
        **kwargs
            Passed to parent Refinement class.
        """
        super().__init__(*args, **kwargs)

        # Set the X-ray target mode (uses the new target system from base class)
        self.set_xray_target_mode(target_mode)
        self.target_mode = target_mode

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
    # Core Optimizer Functions
    # =========================================================================

    def _optimize_lbfgs(
        self,
        state: LossState,
        params=None,
        lr: float = 1.0,
        max_iter: int = 20,
        nsteps: int = 1,
    ) -> LossState:
        """
        Run LBFGS optimization on a LossState.

        Logs initial state, runs optimization, logs final state.

        Parameters
        ----------
        state : LossState
            Configured loss state with targets and weights.
        params : iterable, optional
            Parameters to optimize. Defaults to self.parameters().
        lr : float, optional
            Learning rate. Default is 1.0.
        max_iter : int, optional
            Maximum iterations per LBFGS step. Default is 20.
        nsteps : int, optional
            Number of LBFGS steps. Default is 1.

        Returns
        -------
        LossState
            State with history containing before/after loss values.
        """
        if params is None:
            params = self.parameters()

        # Log initial state
        state.aggregate(log_values=True)

        optimizer = torch.optim.LBFGS(
            params,
            lr=lr,
            max_iter=max_iter,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = state.aggregate()
            loss.backward()
            return loss

        for _ in range(nsteps):
            optimizer.step(closure)

        # Log final state
        state.new_entry()
        state.aggregate(log_values=True)

        return state

    def _optimize_adamw(
        self,
        state: LossState,
        params=None,
        lr: float = 1e-3,
        steps: int = 100,
    ) -> LossState:
        """
        Run AdamW optimization on a LossState.

        Logs initial state, runs optimization, logs final state.

        Parameters
        ----------
        state : LossState
            Configured loss state with targets and weights.
        params : iterable, optional
            Parameters to optimize. Defaults to self.parameters().
        lr : float, optional
            Learning rate. Default is 1e-3.
        steps : int, optional
            Number of optimization steps. Default is 100.

        Returns
        -------
        LossState
            State with history containing before/after loss values.
        """
        if params is None:
            params = self.parameters()

        # Log initial state
        state.aggregate(log_values=True)

        optimizer = torch.optim.AdamW(params, lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            loss = state.aggregate()
            loss.backward()
            if self.verbose > 1 and step % 10 == 0:
                print(f"Step {step+1}/{steps}, Loss: {loss.item():.4f}")
            optimizer.step()

        # Log final state
        state.new_entry()
        state.aggregate(log_values=True)

        return state

    # =========================================================================
    # Refinement Methods
    # =========================================================================

    def refine_adp(self):
        """
        Refine B-factors (ADP) using LBFGS optimizer.

        Freezes all parameters except B-factors and runs LBFGS optimization
        with a combined ADP and X-ray loss.

        Returns
        -------
        LossState
            State with history containing before/after loss values.
        """
        self.model.freeze_all()
        self.model.unfreeze("b")

        self.scaler.refine_lbfgs()
        state = self.complete_loss_state()
        self._optimize_lbfgs(state)

        self.model.unfreeze_all()
        return state

    def refine_xyz(self):
        """
        Refine coordinates (XYZ) using LBFGS optimizer.

        Freezes all parameters except coordinates and runs LBFGS optimization
        with a combined restraints and X-ray loss.

        Returns
        -------
        LossState
            State with history containing before/after loss values.
        """
        self.model.freeze_all()
        self.scaler.freeze()
        self.model.unfreeze("xyz")

        self.scaler.refine_lbfgs()
        state = self.complete_loss_state()
        self._optimize_lbfgs(state)

        self.model.unfreeze_all()
        return state

    def regularize_adp(self, lr=0.1):
        """
        Apply regularization to B-factors (ADP) using LBFGS optimizer.

        Parameters
        ----------
        lr : float, optional
            Learning rate for LBFGS optimizer. Default is 0.1.

        Returns
        -------
        LossState
            State with history containing before/after loss values.
        """
        self.model.freeze_all()
        self.model.unfreeze("b")

        self.scaler.refine_lbfgs()
        state = LossState(self.device)
        state.register_target('adp_target', self.adp_target)

        self._optimize_lbfgs(state, lr=lr)

        self.model.unfreeze_all()
        return state

    def refine_xyz_adamW(self, lr=1e-3, steps=100):
        """
        Refine coordinates (XYZ) using AdamW optimizer as an alternative.

        Parameters
        ----------
        lr : float, optional
            Learning rate for AdamW optimizer. Default is 1e-3.
        steps : int, optional
            Number of AdamW optimization steps. Default is 100.

        Returns
        -------
        LossState
            State with history containing before/after loss values.
        """
        self.model.freeze_all()
        self.scaler.freeze()
        self.model.unfreeze("xyz")

        self.scaler.refine_lbfgs()
        state = self.complete_loss_state()
        self._optimize_adamw(state, lr=lr, steps=steps)

        self.model.unfreeze_all()
        return state

    def refine_b_adamW(self, lr=1e-3, steps=100):
        """
        Refine B-factors (ADP) using AdamW optimizer as an alternative.

        Parameters
        ----------
        lr : float, optional
            Learning rate for AdamW optimizer. Default is 1e-3.
        steps : int, optional
            Number of AdamW optimization steps. Default is 100.

        Returns
        -------
        LossState
            State with history containing before/after loss values.
        """
        self.model.freeze_all()
        self.model.unfreeze("b")

        self.scaler.refine_lbfgs()
        state = self.complete_loss_state()
        self._optimize_adamw(state, lr=lr, steps=steps)

        self.model.unfreeze_all()
        return state

    def refine_everything_adamW(self, lr=1e-3, steps=100):
        """
        Refine both coordinates (XYZ) and B-factors (ADP) using AdamW optimizer.

        Parameters
        ----------
        lr : float, optional
            Learning rate for AdamW optimizer. Default is 1e-3.
        steps : int, optional
            Number of AdamW optimization steps. Default is 100.

        Returns
        -------
        LossState
            State with history containing before/after loss values.
        """
        self.model.freeze_all()
        self.scaler.unfreeze()
        self.model.unfreeze("xyz")
        self.model.unfreeze("b")

        self.scaler.refine_lbfgs()
        state = self.complete_loss_state()
        self._optimize_adamw(state, lr=lr, steps=steps)

        self.model.unfreeze_all()
        return state

    def _refine_everything_lbfgs_single_cycle(self, nsteps=1):
        """
        Refine both coordinates (XYZ) and B-factors (ADP) using LBFGS optimizer.

        Jointly optimizes all parameters with the combined restraints, ADP,
        and X-ray loss.

        Returns
        -------
        LossState
            State with history containing before/after loss values.
        """
        self.scaler.refine_lbfgs()
        state = self.complete_loss_state()
        self._optimize_lbfgs(state, nsteps=nsteps)

        self.model.unfreeze_all()
        return state

    # =========================================================================
    # Training Loop for Policy Learning
    # =========================================================================

    def run_training_trajectory(
        self,
        policy_weighting,
        n_steps: int = 10,
        pdb_id: str = "",
        structure_path: str = "",
        sf_path: str = "",
        seed: Optional[int] = None,
        policy_version: Optional[str] = None,
    ):
        """
        Run a training trajectory with policy-guided refinement.

        This method runs a sequence of refinement steps using a policy
        to select component weights. It records state-action-reward tuples
        for training the policy with AWR or similar algorithms.

        Parameters
        ----------
        policy_weighting : PolicyComponentWeighting
            Policy weighting scheme (should be in training mode with sampling).
        n_steps : int, optional
            Number of refinement steps in the trajectory (default: 10).
        pdb_id : str, optional
            PDB identifier for recording.
        structure_path : str, optional
            Path to structure file for recording.
        sf_path : str, optional
            Path to structure factors file for recording.
        seed : int, optional
            Random seed for reproducibility.
        policy_version : str, optional
            Version identifier of the policy being used.

        Returns
        -------
        TrajectoryData
            Complete trajectory with state-action-reward tuples.

        Example
        -------
        ::

            from torchref.refinement.weighting import PolicyComponentWeighting

            # Create policy in training mode (sampling enabled)
            policy = PolicyComponentWeighting(
                refinement, policy_path='policy.pt',
                sample=True, temperature=1.0
            )

            # Run trajectory
            trajectory = refinement.run_training_trajectory(
                policy, n_steps=10, pdb_id='3GR5'
            )

            # Save trajectory for training
            import json
            from torchref.refinement.weighting import trajectory_to_dict
            with open('trajectory.json', 'w') as f:
                json.dump(trajectory_to_dict(trajectory), f)
        """
        import time

        start_time = time.time()

        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Start recording
        policy_weighting.start_recording(
            pdb_id=pdb_id,
            structure_path=structure_path,
            sf_path=sf_path,
            seed=seed,
            policy_version=policy_version,
        )

        try:
            # Initial scaling
            self.scaler.refine_lbfgs()

            for step in range(n_steps):
                if self.verbose > 1:
                    print(f"Step {step + 1}/{n_steps}")

                # Create LossState and apply policy weights
                state = self.complete_loss_state()

                # Evaluate once to populate loss cache (needed for feature extraction)
                with torch.no_grad():
                    state.aggregate()

                # Apply policy weights (this also records the step)
                policy_weighting.apply_to_state(state)

                # Run LBFGS optimization with policy weights
                self._optimize_lbfgs(state, nsteps=1)

                # Increment step counter
                policy_weighting.increment_step()

            # Stop recording and get trajectory
            trajectory = policy_weighting.stop_recording()
            trajectory.total_time = time.time() - start_time
            trajectory.success = True

        except Exception as e:
            # Record failure
            trajectory = policy_weighting.stop_recording()
            if trajectory is not None:
                trajectory.success = False
                trajectory.error_message = str(e)
                trajectory.total_time = time.time() - start_time
            raise

        return trajectory

    def run_training_trajectory_joint(
        self,
        policy_weighting,
        n_steps: int = 10,
        pdb_id: str = "",
        structure_path: str = "",
        sf_path: str = "",
        seed: Optional[int] = None,
        policy_version: Optional[str] = None,
    ):
        """
        Run a training trajectory with joint XYZ+ADP refinement.

        Similar to run_training_trajectory but refines both XYZ and ADP
        together in each step, which may be more efficient.

        Parameters
        ----------
        policy_weighting : PolicyComponentWeighting
            Policy weighting scheme (should be in training mode).
        n_steps : int, optional
            Number of refinement steps (default: 10).
        pdb_id, structure_path, sf_path : str, optional
            Identifiers for trajectory recording.
        seed : int, optional
            Random seed for reproducibility.
        policy_version : str, optional
            Policy version identifier.

        Returns
        -------
        TrajectoryData
            Complete trajectory with state-action-reward tuples.
        """
        import time

        start_time = time.time()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        policy_weighting.start_recording(
            pdb_id=pdb_id,
            structure_path=structure_path,
            sf_path=sf_path,
            seed=seed,
            policy_version=policy_version,
        )

        try:
            # Initial scaling
            self.scaler.refine_lbfgs()

            # Unfreeze all parameters for joint refinement
            self.model.unfreeze("xyz")
            self.model.unfreeze("b")

            for step in range(n_steps):
                if self.verbose > 1:
                    print(f"Step {step + 1}/{n_steps}")

                # Create LossState and evaluate to populate cache
                state = self.complete_loss_state()
                with torch.no_grad():
                    state.aggregate()

                # Apply policy weights (records the step)
                policy_weighting.apply_to_state(state)

                # Run LBFGS optimization
                self._optimize_lbfgs(state, nsteps=1)

                # Increment step counter
                policy_weighting.increment_step()

            # Freeze everything back
            self.model.freeze_all()

            trajectory = policy_weighting.stop_recording()
            trajectory.total_time = time.time() - start_time
            trajectory.success = True

        except Exception as e:
            self.model.freeze_all()
            trajectory = policy_weighting.stop_recording()
            if trajectory is not None:
                trajectory.success = False
                trajectory.error_message = str(e)
                trajectory.total_time = time.time() - start_time
            raise

        return trajectory

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
        self.scaler.freeze()
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
            # Hierarchical cycle dict structure
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

            # Collect metrics before scaling
            with torch.no_grad():
                before_scaling = self.collect_metrics()
                cycle_dict["before_scaling"] = before_scaling

            self.get_scales()

            # Collect metrics after scaling
            with torch.no_grad():
                after_scaling = self.collect_metrics()
                cycle_dict["after_scaling"] = after_scaling
                if self.verbose > 0:
                    print(
                        f"After scaling: Rwork={after_scaling['rwork']:.4f}, Rfree={after_scaling['rfree']:.4f}"
                    )

            # Record state before XYZ
            self.logger.record(label="before_xyz")
            cycle_dict["xyz"]["before"] = self.collect_metrics()

            # XYZ refinement with cycle-aware weighting
            self.refine_xyz()

            # Record state after XYZ and log comparison
            self.logger.record(label="after_xyz")
            cycle_dict["xyz"]["after"] = self.collect_metrics()
            if self.verbose > 0:
                self.logger.compare(
                    label_before="before_xyz",
                    label_after="after_xyz",
                    title="XYZ Refinement",
                )

            # Record state before ADP
            self.logger.record(label="before_adp")
            cycle_dict["adp"]["before"] = self.collect_metrics()

            # B-factor refinement with cycle-aware weighting
            self.refine_adp()

            # Record state after ADP and log comparison
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
        self.scaler.unfreeze()
        self.model.unfreeze_all()
        i = 0

        while True:
            i += 1
            master_key = f"refinement_everything_{i}"
            if master_key not in self.history:
                break

        self.history[master_key] = []
        self.history["initial"] = self.collect_metrics()

        # Clear logger history for fresh refinement
        self.logger.clear()

        for cycle in range(macro_cycles):
            # Hierarchical cycle dict structure
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

            # Record state after scaling (before refinement)
            self.logger.record(label="after_scaling")
            with torch.no_grad():
                after_scaling = self.collect_metrics()
                cycle_dict["after_scaling"] = after_scaling
                if self.verbose > 0:
                    print(
                        f"After scaling: Rwork={after_scaling['rwork']:.4f}, Rfree={after_scaling['rfree']:.4f}"
                    )

            # Full refinement
            self._refine_everything_lbfgs_single_cycle()

            # Record state after refinement and log comparison
            self.logger.record(label="after_refinement")
            with torch.no_grad():
                after_refinement = self.collect_metrics()
                cycle_dict["after_refinement"] = after_refinement
                if self.verbose > 0:
                    print(
                        f"After refinement: Rwork={after_refinement['rwork']:.4f}, Rfree={after_refinement['rfree']:.4f}"
                    )
                    self.logger.compare(
                        label_before="after_scaling",
                        label_after="after_refinement",
                        title="Joint XYZ+ADP Refinement",
                    )

            self.history[master_key].append(cycle_dict)

        return self.history