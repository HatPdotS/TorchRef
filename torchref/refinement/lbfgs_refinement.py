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

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
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
    >>> from torchref.refinement.loss_weighting import ResolutionDependentWeighting
    >>> weighter = ResolutionDependentWeighting()
    >>> refinement = LBFGSRefinement(mtz_file, pdb_file, weighter=weighter, target_mode='ml')
    >>> refinement.refine(macro_cycles=2)
    """

    def __init__(self, *args, target_mode: str = 'ml', **kwargs):
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
            line_search_fn="strong_wolfe"
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

        optimizer = torch.optim.AdamW(
            params,
            lr=lr
        )

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
        self.model.unfreeze('b')

        self.scaler.refine_lbfgs()
        state = self.create_loss_state()
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
        self.model.unfreeze('xyz')

        self.scaler.refine_lbfgs()
        state = self.create_loss_state()
        self._optimize_lbfgs(state)

        self.model.unfreeze_all()
        return state
    
    def _run_xyz_with_weight(self, restraint_weight: float, max_iter: int = 20) -> Dict:
        """
        Run XYZ refinement with a fixed restraint weight and return metrics.

        This is used internally for weight screening. Saves and restores model state.

        Parameters
        ----------
        restraint_weight : float
            Weight for restraints relative to X-ray target.
        max_iter : int, optional
            Maximum LBFGS iterations. Default is 20.

        Returns
        -------
        dict
            Dictionary with rwork, rfree, rmsd_bonds, rmsd_angles, state, etc.
        """
        with torch.no_grad():
            rwork_start, rfree_start = self.get_rfactor()

        self.scaler.refine_lbfgs()
        state = self.create_loss_state()
        self._optimize_lbfgs(state, params=self.model.parameters(), max_iter=max_iter)

        # Collect metrics
        with torch.no_grad():
            rwork, rfree = self.get_rfactor()
            xray_target = self.xray_loss().item()
            restraints_target = self.restraints_loss().item()
            bond_devs, _ = self.restraints.bond_deviations()
            rmsd_bonds = torch.sqrt((bond_devs ** 2).mean()).item()
            angle_devs, _ = self.restraints.angle_deviations()
            rmsd_angles = torch.sqrt((angle_devs ** 2).mean()).item()

        return {
            'weight': restraint_weight,
            'rwork': rwork,
            'rfree': rfree,
            'rwork_start': rwork_start,
            'rfree_start': rfree_start,
            'gap': rfree - rwork,
            'xray_target': xray_target,
            'restraints_target': restraints_target,
            'rmsd_bonds': rmsd_bonds,
            'rmsd_angles': rmsd_angles,
            'state': state,
        }

    def _run_adp_with_weight(self, adp_weight: float, max_iter: int = 20) -> Dict:
        """
        Run ADP refinement with a fixed ADP weight and return metrics.

        This is used internally for weight screening. Saves and restores model state.

        Parameters
        ----------
        adp_weight : float
            Weight for ADP restraints relative to X-ray target.
        max_iter : int, optional
            Maximum LBFGS iterations. Default is 20.

        Returns
        -------
        dict
            Dictionary with rwork, rfree, mean_b, state, etc.
        """
        with torch.no_grad():
            rwork_start, rfree_start = self.get_rfactor()

        self.scaler.refine_lbfgs()
        state = self.create_loss_state()
        self._optimize_lbfgs(state, params=self.model.parameters(), max_iter=max_iter)

        # Collect metrics
        with torch.no_grad():
            rwork, rfree = self.get_rfactor()
            xray_target = self.xray_loss().item()
            adp_target = self.adp_loss().item()
            b_factors = self.model.b()
            mean_b = b_factors.mean().item()
            bi_bj = self._compute_mean_bi_bj()

        return {
            'weight': adp_weight,
            'rwork': rwork,
            'rfree': rfree,
            'gap': rfree - rwork,
            'xray_target': xray_target,
            'adp_target': adp_target,
            'mean_b': mean_b,
            'bi_bj': bi_bj,
            'rwork_start': rwork_start,
            'rfree_start': rfree_start,
            'state': state,
        }

    def _compute_mean_bi_bj(self) -> float:
        """
        Compute mean |Bi - Bj| for bonded atom pairs.

        Returns
        -------
        float
            Mean absolute B-factor difference for bonded atoms.
        """
        b = self.model.b()
        
        # Make sure 'all' indices are available
        if 'all' not in self.restraints.restraints['bond']:
            self.restraints.cat_dict()
        
        bonds = self.restraints.restraints['bond']['all']['indices']
        if bonds is None or len(bonds) == 0:
            return 0.0
        bi = b[bonds[:, 0]]
        bj = b[bonds[:, 1]]
        return torch.abs(bi - bj).mean().item()
    
    def screen_xyz_weights(
        self,
        weights: Optional[List[float]] = None,
        n_weights: int = 10,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
        max_gap: float = 0.06,
        max_iter: int = 20,
    ) -> Tuple[float, List[Dict]]:
        """
        Screen XYZ refinement weights (Phenix-style approach).

        For each weight, runs a complete LBFGS optimization and records metrics.
        Selects the best weight based on lowest Rfree while respecting gap constraint.

        Parameters
        ----------
        weights : list of float, optional
            Explicit list of weights to try. If None, generates log-spaced weights.
        n_weights : int, optional
            Number of weights to screen (if weights is None). Default is 10.
        min_weight : float, optional
            Minimum weight value (if weights is None). Default is 0.1.
        max_weight : float, optional
            Maximum weight value (if weights is None). Default is 10.0.
        max_gap : float, optional
            Maximum allowed Rfree-Rwork gap. Default is 0.06.
        max_iter : int, optional
            Maximum LBFGS iterations per weight trial. Default is 20.

        Returns
        -------
        tuple
            Tuple of (best_weight, results_list).
        """
        from copy import deepcopy
        if weights is None:
            weights = np.logspace(np.log10(min_weight), np.log10(max_weight), n_weights).tolist()

        self.scaler.freeze()
        self.model.freeze_all()
        self.model.unfreeze('xyz')
        print(self.model.parameters())
        
        # Deep copy initial state - critical for proper restoration
        model_initial = deepcopy(self.model.xyz.state_dict())
        
        results = []
        
        # Get starting metrics (before refinement)
        with torch.no_grad():
            rwork_start, rfree_start = self.get_rfactor()
        
        if self.verbose > 0:
            print(f"\n{'='*80}")
            print(f"XYZ Weight Screening (Phenix-style)")
            print(f"{'='*80}")
            print(f"{'WEIGHT':>10} {'Rwork':>8} {'Rfree':>8} {'Gap':>8} {'RMSD_b':>8} {'RMSD_a':>8} {'X-ray':>10} {'Restr':>10}")
            print(f"{'start':>10} {rwork_start*100:>7.2f}% {rfree_start*100:>7.2f}% {(rfree_start-rwork_start)*100:>7.2f}%")
        
        for weight in weights:
            result = self._run_xyz_with_weight(weight, max_iter=max_iter)
            # Deep copy the state dict with tensor cloning
            result['state_dict'] = deepcopy(self.model.xyz.state_dict())
            results.append(result)
            
            # Restore initial state for next trial
            self.model.xyz.load_state_dict(model_initial)
            
            if self.verbose > 0:
                print(f"{weight:>10.3f} {result['rwork']*100:>7.2f}% {result['rfree']*100:>7.2f}% "
                      f"{result['gap']*100:>7.2f}% {result['rmsd_bonds']:>8.4f} {result['rmsd_angles']:>8.2f} "
                      f"{result['xray_target']:>10.4f} {result['restraints_target']:>10.4f} {result['rwork_start']*100:>7.2f}% {result['rfree_start']*100:>7.2f}%")
        
        # Multi-metric weight selection
        best = self._select_best_xyz_weight(results, max_gap, rwork_start, rfree_start)
        
        best_weight = best['weight']

        self.model.xyz.load_state_dict(best['state_dict'])
        
        if self.verbose > 0:
            print(f"\nBest XYZ weight: {best_weight:.3f}")
            print(f"  Rwork={best['rwork']*100:.2f}%, Rfree={best['rfree']*100:.2f}%, Gap={best['gap']*100:.2f}%")
            print(f"  Selection score: {best.get('score', 'N/A')}")

        self.model.unfreeze_all()
        self.scaler.unfreeze()
        
        return best_weight, results
    
    def _select_best_xyz_weight(self, results: List[Dict], max_gap: float, 
                                 rwork_start: float, rfree_start: float) -> Dict:
        """
        Select best XYZ weight using multi-metric ranking.

        Scoring criteria (in priority order):

        1. Rfree must improve from starting value
        2. Gap should be reasonable (penalize large gaps)
        3. Lower Rfree is better
        4. Reasonable geometry (RMSD bonds/angles)

        Parameters
        ----------
        results : list of dict
            List of result dicts from weight screening.
        max_gap : float
            Maximum desired gap (soft constraint).
        rwork_start : float
            Starting Rwork before refinement.
        rfree_start : float
            Starting Rfree before refinement.

        Returns
        -------
        dict
            Best result dict with 'score' field added.
        """
        # Compute normalized scores for each metric
        rfree_values = [r['rfree'] for r in results]
        gap_values = [r['gap'] for r in results]
        rmsd_bonds = [r['rmsd_bonds'] for r in results]
        
        rfree_min, rfree_max = min(rfree_values), max(rfree_values)
        gap_min, gap_max = min(gap_values), max(gap_values)
        rmsd_min, rmsd_max = min(rmsd_bonds), max(rmsd_bonds)
        
        for r in results:
            # Normalize each metric to [0, 1] where lower is better
            if rfree_max > rfree_min:
                rfree_norm = (r['rfree'] - rfree_min) / (rfree_max - rfree_min)
            else:
                rfree_norm = 0.0
                
            if gap_max > gap_min:
                gap_norm = (r['gap'] - gap_min) / (gap_max - gap_min)
            else:
                gap_norm = 0.0
                
            if rmsd_max > rmsd_min:
                rmsd_norm = (r['rmsd_bonds'] - rmsd_min) / (rmsd_max - rmsd_min)
            else:
                rmsd_norm = 0.0
            
            # Penalty if Rfree didn't improve
            rfree_improvement_penalty = 0.0 if r['rfree'] < rfree_start else 0.5
            
            # Penalty for excessive gap (soft constraint)
            gap_penalty = max(0, (r['gap'] - max_gap) / max_gap) if r['gap'] > max_gap else 0.0
            
            # Composite score: weighted sum (lower is better)
            # Prioritize: Rfree (40%), Gap (35%), Geometry (15%), Improvement (10%)
            r['score'] = (
                0.40 * rfree_norm +
                0.35 * gap_norm +
                0.15 * rmsd_norm +
                0.10 * rfree_improvement_penalty +
                0.50 * gap_penalty  # Strong penalty for excessive gap
            )
        
        # Print ranking if verbose
        if self.verbose > 1:
            print("\nWeight ranking (top 5):")
            sorted_results = sorted(results, key=lambda x: x['score'])
            for i, r in enumerate(sorted_results[:5]):
                print(f"  {i+1}. w={r['weight']:.3f}: Rfree={r['rfree']*100:.2f}%, "
                      f"Gap={r['gap']*100:.2f}%, Score={r['score']:.4f}")
        
        # Return best (lowest score)
        return min(results, key=lambda x: x['score'])
    
    def screen_adp_weights(
        self,
        weights: Optional[List[float]] = None,
        n_weights: int = 20,
        min_weight: float = 1,
        max_weight: float = 100.0,
        max_gap: float = 0.06,
        max_bi_bj: float = 10.0,
        max_iter: int = 20,
    ) -> Tuple[float, List[Dict]]:
        """
        Screen ADP refinement weights (Phenix-style approach).

        For each weight, runs a complete LBFGS optimization and records metrics.
        Selects the best weight based on lowest Rfree while respecting constraints.

        Parameters
        ----------
        weights : list of float, optional
            Explicit list of weights to try. If None, generates log-spaced weights.
        n_weights : int, optional
            Number of weights to screen (if weights is None). Default is 20.
        min_weight : float, optional
            Minimum weight value (if weights is None). Default is 1.
        max_weight : float, optional
            Maximum weight value (if weights is None). Default is 100.0.
        max_gap : float, optional
            Maximum allowed Rfree-Rwork gap. Default is 0.06.
        max_bi_bj : float, optional
            Maximum allowed mean |Bi-Bj| for bonded atoms. Default is 10.0.
        max_iter : int, optional
            Maximum LBFGS iterations per weight trial. Default is 20.

        Returns
        -------
        tuple
            Tuple of (best_weight, results_list).
        """
        from copy import deepcopy

        if weights is None:
            weights = np.logspace(np.log10(min_weight), np.log10(max_weight), n_weights).tolist()
        self.scaler.freeze()
        self.model.freeze_all()
        self.model.unfreeze('b')
        print(self.model.parameters())
        # Deep copy initial state - critical for proper restoration
        b_initial = deepcopy(self.model.b.state_dict())
        
        results = []
        
        # Get starting metrics
        with torch.no_grad():
            rwork_start, rfree_start = self.get_rfactor()
            mean_b_start = self.model.b().mean().item()
            bi_bj_start = self._compute_mean_bi_bj()
        
        if self.verbose > 0:
            print(f"\n{'='*80}")
            print(f"ADP Weight Screening (Phenix-style)")
            print(f"{'='*80}")
            print(f"{'WEIGHT':>10} {'Rwork':>8} {'Rfree':>8} {'Gap':>8} {'<Bi-Bj>':>8} {'<B>':>8} {'X-ray':>10} {'ADP':>10}")
            print(f"{'start':>10} {rwork_start*100:>7.2f}% {rfree_start*100:>7.2f}% {(rfree_start-rwork_start)*100:>7.2f}% "
                  f"{bi_bj_start:>8.2f} {mean_b_start:>8.1f}")
        
        for weight in weights:
            result = self._run_adp_with_weight(weight, max_iter=max_iter)
            # Deep copy the state dict with tensor cloning
            result['state_dict'] = deepcopy(self.model.b.state_dict())
            results.append(result)
            
            # Restore initial state for next trial
            self.model.b.load_state_dict(b_initial)
            
            if self.verbose > 0:
                print(f"{weight:>10.3f} {result['rwork']*100:>7.2f}% {result['rfree']*100:>7.2f}% "
                      f"{result['gap']*100:>7.2f}% {result['bi_bj']:>8.2f} {result['mean_b']:>8.1f} "
                      f"{result['xray_target']:>10.4f} {result['adp_target']:>10.4f} {result['rwork_start']*100:>7.2f}% {result['rfree_start']*100:>7.2f}%")
        
        # Multi-metric weight selection
        best = self._select_best_adp_weight(results, max_gap, max_bi_bj, rwork_start, rfree_start)
        
        best_weight = best['weight']
        
        if self.verbose > 0:
            print(f"\nBest ADP weight: {best_weight:.3f}")
            print(f"  Rwork={best['rwork']*100:.2f}%, Rfree={best['rfree']*100:.2f}%, "
                  f"Gap={best['gap']*100:.2f}%, <Bi-Bj>={best['bi_bj']:.2f}")
            print(f"  Selection score: {best.get('score', 'N/A')}")
        
        self.model.b.load_state_dict(best['state_dict'])
        self.model.unfreeze_all()
        self.scaler.unfreeze()  
        return best_weight, results
    
    def _select_best_adp_weight(self, results: List[Dict], max_gap: float, max_bi_bj: float,
                                 rwork_start: float, rfree_start: float) -> Dict:
        """
        Select best ADP weight using multi-metric ranking.

        Scoring criteria:

        1. Rfree must improve from starting value
        2. Gap should be reasonable (penalize large gaps)
        3. <Bi-Bj> should be reasonable (penalize too large or too small)
        4. Lower Rfree is better

        Parameters
        ----------
        results : list of dict
            List of result dicts from weight screening.
        max_gap : float
            Maximum desired gap (soft constraint).
        max_bi_bj : float
            Maximum desired <Bi-Bj> (soft constraint).
        rwork_start : float
            Starting Rwork before refinement.
        rfree_start : float
            Starting Rfree before refinement.

        Returns
        -------
        dict
            Best result dict with 'score' field added.
        """
        # Compute normalized scores for each metric
        rfree_values = [r['rfree'] for r in results]
        gap_values = [r['gap'] for r in results]
        bi_bj_values = [r['bi_bj'] for r in results]
        
        rfree_min, rfree_max = min(rfree_values), max(rfree_values)
        gap_min, gap_max = min(gap_values), max(gap_values)
        bi_bj_min, bi_bj_max = min(bi_bj_values), max(bi_bj_values)
        
        # Ideal <Bi-Bj> is around 2-4 Å² (similar to Phenix target)
        ideal_bi_bj = 3.0
        
        for r in results:
            # Normalize each metric to [0, 1] where lower is better
            if rfree_max > rfree_min:
                rfree_norm = (r['rfree'] - rfree_min) / (rfree_max - rfree_min)
            else:
                rfree_norm = 0.0
                
            if gap_max > gap_min:
                gap_norm = (r['gap'] - gap_min) / (gap_max - gap_min)
            else:
                gap_norm = 0.0
            
            # <Bi-Bj> score: penalize deviation from ideal
            bi_bj_deviation = abs(r['bi_bj'] - ideal_bi_bj) / max(max_bi_bj, 1.0)
            bi_bj_norm = min(bi_bj_deviation, 1.0)
            
            # Penalty if Rfree didn't improve
            rfree_improvement_penalty = 0.0 if r['rfree'] < rfree_start else 0.5
            
            # Penalty for excessive gap
            gap_penalty = max(0, (r['gap'] - max_gap) / max_gap) if r['gap'] > max_gap else 0.0
            
            # Penalty for excessive <Bi-Bj>
            bi_bj_penalty = max(0, (r['bi_bj'] - max_bi_bj) / max_bi_bj) if r['bi_bj'] > max_bi_bj else 0.0
            
            # Composite score: weighted sum (lower is better)
            # Prioritize: Rfree (30%), Gap (30%), Bi-Bj (20%), Improvement (10%), Penalties (10%)
            r['score'] = (
                0.30 * rfree_norm +
                0.30 * gap_norm +
                0.20 * bi_bj_norm +
                0.10 * rfree_improvement_penalty +
                0.50 * gap_penalty +
                0.30 * bi_bj_penalty
            )
        
        # Print ranking if verbose
        if self.verbose > 1:
            print("\nWeight ranking (top 5):")
            sorted_results = sorted(results, key=lambda x: x['score'])
            for i, r in enumerate(sorted_results[:5]):
                print(f"  {i+1}. w={r['weight']:.3f}: Rfree={r['rfree']*100:.2f}%, "
                      f"Gap={r['gap']*100:.2f}%, <Bi-Bj>={r['bi_bj']:.2f}, Score={r['score']:.4f}")
        
        # Return best (lowest score)
        return min(results, key=lambda x: x['score'])
    
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
        self.model.unfreeze('b')

        self.scaler.refine_lbfgs()
        state = self.create_loss_state()
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
        self.model.unfreeze('xyz')

        self.scaler.refine_lbfgs()
        state = self.create_loss_state()
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
        self.model.unfreeze('b')

        self.scaler.refine_lbfgs()
        state = self.create_loss_state()
        self._optimize_adamw(state, lr=lr, steps=steps)

        self.model.unfreeze_all()
        return state

    def regularize_xyz_adp_to_rfactor_gap(self, lr=1e-1, max_steps=100, target_rfactor_gap=0.05):
        """
        Apply regularization to coordinates (XYZ) and B-factors (ADP) until target gap.

        Uses AdamW optimizer and stops when the Rfree-Rwork gap reaches the target.
        Note: This method has custom early stopping logic and cannot use _optimize_adamw.

        Parameters
        ----------
        lr : float, optional
            Learning rate for AdamW optimizer. Default is 0.1.
        max_steps : int, optional
            Maximum number of optimization steps. Default is 100.
        target_rfactor_gap : float, optional
            Target Rfree-Rwork gap to achieve. Default is 0.05.

        Returns
        -------
        LossState
            State with history containing before/after loss values.
        """
        self.model.freeze_all()
        self.scaler.freeze()
        self.model.unfreeze('xyz')
        self.model.unfreeze('b')

        self.scaler.refine_lbfgs()
        state = self.create_loss_state()

        # Log initial state
        state.aggregate(log_values=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        def closure():
            optimizer.zero_grad()
            loss = state.aggregate()
            loss.backward()
            return loss

        for step in range(max_steps):
            optimizer.step(closure)
            with torch.no_grad():
                rwork, rfree = self.get_rfactor()
                rfactor_gap = rfree - rwork
                if self.verbose > 1:
                    print(f"Step {step+1}/{max_steps}, Rwork: {rwork:.4f}, Rfree: {rfree:.4f}, Rfactor gap: {rfactor_gap:.4f}")
                if rfactor_gap <= target_rfactor_gap:
                    if self.verbose > 0:
                        print(f"Target R-factor gap of {target_rfactor_gap} reached at step {step+1}. Stopping regularization.")
                    break

        # Log final state
        state.new_entry()
        state.aggregate(log_values=True)

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
        self.model.unfreeze('xyz')
        self.model.unfreeze('b')

        self.scaler.refine_lbfgs()
        state = self.create_loss_state()
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
        state = self.create_loss_state()
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
        >>> from torchref.refinement.weighting import PolicyComponentWeighting
        >>>
        >>> # Create policy in training mode (sampling enabled)
        >>> policy = PolicyComponentWeighting(
        ...     refinement, policy_path='policy.pt',
        ...     sample=True, temperature=1.0
        ... )
        >>>
        >>> # Run trajectory
        >>> trajectory = refinement.run_training_trajectory(
        ...     policy, n_steps=10, pdb_id='3GR5'
        ... )
        >>>
        >>> # Save trajectory for training
        >>> import json
        >>> from torchref.refinement.weighting import trajectory_to_dict
        >>> with open('trajectory.json', 'w') as f:
        ...     json.dump(trajectory_to_dict(trajectory), f)
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
                state = self.create_loss_state()

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
            self.model.unfreeze('xyz')
            self.model.unfreeze('b')

            for step in range(n_steps):
                if self.verbose > 1:
                    print(f"Step {step + 1}/{n_steps}")

                # Create LossState and evaluate to populate cache
                state = self.create_loss_state()
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
            master_key = f'refinement_{i}'
            if not master_key in self.history:
                break

        self.history[master_key] = []
        for cycle in range(macro_cycles):
            # Hierarchical cycle dict structure
            cycle_dict = {
                'cycle': cycle + 1,
                'before_scaling': {},
                'after_scaling': {},
                'xyz': {
                    'before': {},
                    'after': {},
                    'weights': {}
                },
                'adp': {
                    'before': {},
                    'after': {},
                    'weights': {}
                }
            }
            self.component_weighting.update_weights()
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"LBFGS Refinement - Cycle {cycle+1}/{macro_cycles}")
                print(f"{'='*60}")
            
            # Collect metrics before scaling
            with torch.no_grad():
                before_scaling = self.collect_metrics()
                cycle_dict['before_scaling'] = before_scaling
                
            self.get_scales()
            
            # Collect metrics after scaling
            with torch.no_grad():
                after_scaling = self.collect_metrics()
                cycle_dict['after_scaling'] = after_scaling
                if self.verbose > 0:
                    print(f"After scaling: Rwork={after_scaling['rwork']:.4f}, Rfree={after_scaling['rfree']:.4f}")

            # Store metrics before XYZ
            with torch.no_grad():
                before_xyz = self.collect_metrics()
                cycle_dict['xyz']['before'] = before_xyz
                cycle_dict['xyz']['weights'] = self.component_weighting.weights.copy()
            
            # XYZ refinement with cycle-aware weighting
            self.refine_xyz()
            
            # Collect metrics after XYZ
            with torch.no_grad():
                after_xyz = self.collect_metrics()
                cycle_dict['xyz']['after'] = after_xyz
                if self.verbose > 0:
                    self.log_xyz_comparison(before_xyz, after_xyz)

            # Store metrics before ADP
            with torch.no_grad():
                before_adp = self.collect_metrics()
                cycle_dict['adp']['before'] = before_adp
                cycle_dict['adp']['weights'] = self.component_weighting.weights.copy()
            
            # B-factor refinement with cycle-aware weighting
            self.refine_adp()
            
            # Collect metrics after ADP (final for this cycle)
            with torch.no_grad():
                after_adp = self.collect_metrics()
                cycle_dict['adp']['after'] = after_adp
                if self.verbose > 0:
                    self.log_adp_comparison(before_adp, after_adp)

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
            master_key = f'refinement_everything_{i}'
            if not master_key in self.history:
                break

        self.history[master_key] = []
        self.history['initial'] = self.collect_metrics()
        for cycle in range(macro_cycles):
            # Hierarchical cycle dict structure
            cycle_dict = {
                'cycle': cycle + 1,
                'before_scaling': {},
                'after_scaling': {},
                'after_refinement': {}
            }
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"LBFGS Refinement Everything - Cycle {cycle+1}/{macro_cycles}")
                print(f"{'='*60}")
                
            self.component_weighting.update_weights()
                
            self.get_scales()
            
            # Collect metrics after scaling
            with torch.no_grad():
                after_scaling = self.collect_metrics()
                cycle_dict['after_scaling'] = after_scaling
                if self.verbose > 0:
                    print(f"After scaling: Rwork={after_scaling['rwork']:.4f}, Rfree={after_scaling['rfree']:.4f}")

            # Full refinement
            self._refine_everything_lbfgs_single_cycle()
            
            # Collect metrics after refinement
            with torch.no_grad():
                after_refinement = self.collect_metrics()
                cycle_dict['after_refinement'] = after_refinement
                if self.verbose > 0:
                    print(f"After refinement: Rwork={after_refinement['rwork']:.4f}, Rfree={after_refinement['rfree']:.4f}")
                self.log_xyz_comparison(after_scaling, after_refinement)
                self.log_adp_comparison(after_scaling, after_refinement)

            self.history[master_key].append(cycle_dict)

        return self.history

    def refine_screened(
        self,
        macro_cycles: int = 5,
        xyz_weights: Optional[List[float]] = None,
        adp_weights: Optional[List[float]] = None,
        n_xyz_weights: int = 20,
        n_adp_weights: int = 20,
        xyz_min_weight: float = 1,
        xyz_max_weight: float = 100.0,
        adp_min_weight: float = 1,
        adp_max_weight: float = 100.0,
        max_gap: float = 0.06,
        max_bi_bj: float = 10.0,
        max_iter: int = 20,
    ):
        """
        Run full LBFGS refinement with Phenix-style weight screening.

        This approach screens multiple weights for each refinement step,
        selects the best weight based on Rfree (respecting gap constraints),
        and applies the refinement with that weight.

        This is fundamentally different from GradNorm-based adaptive weighting:

        - GradNorm: Adjusts weights dynamically during optimization
        - Screening: Runs multiple complete optimizations with fixed weights

        Parameters
        ----------
        macro_cycles : int, optional
            Number of refinement macro cycles. Default is 5.
        xyz_weights : list of float, optional
            Explicit XYZ weight list (or auto-generate).
        adp_weights : list of float, optional
            Explicit ADP weight list (or auto-generate).
        n_xyz_weights : int, optional
            Number of XYZ weights to screen. Default is 20.
        n_adp_weights : int, optional
            Number of ADP weights to screen. Default is 20.
        xyz_min_weight : float, optional
            Minimum XYZ weight. Default is 1.
        xyz_max_weight : float, optional
            Maximum XYZ weight. Default is 100.0.
        adp_min_weight : float, optional
            Minimum ADP weight. Default is 1.
        adp_max_weight : float, optional
            Maximum ADP weight. Default is 100.0.
        max_gap : float, optional
            Maximum allowed Rfree-Rwork gap. Default is 0.06.
        max_bi_bj : float, optional
            Maximum allowed mean |Bi-Bj|. Default is 10.0.
        max_iter : int, optional
            Maximum LBFGS iterations per weight trial. Default is 20.

        Returns
        -------
        dict
            History dictionary with refinement metrics (hierarchical structure).
        """
        self.scaler.freeze()
        
        # Find unique history key
        i = 0
        while True:
            i += 1
            master_key = f'refinement_screened_{i}'
            if master_key not in self.history:
                break
        
        self.history[master_key] = []
        
        for cycle in range(macro_cycles):
            # Hierarchical cycle dict structure
            cycle_dict = {
                'cycle': cycle + 1,
                'before_scaling': {},
                'after_scaling': {},
                'xyz': {
                    'before': {},
                    'after': {},
                    'weight': None,
                    'weight_screens': None
                },
                'adp': {
                    'before': {},
                    'after': {},
                    'weight': None,
                    'weight_screens': None
                }
            }
            
            if self.verbose > 0:
                print(f"\n{'='*80}")
                print(f"LBFGS Refinement with Weight Screening - Cycle {cycle+1}/{macro_cycles}")
                print(f"{'='*80}")
            
            # Collect metrics before scaling
            with torch.no_grad():
                before_scaling = self.collect_metrics()
                cycle_dict['before_scaling'] = before_scaling
            
            # Scaling
            self.get_scales()
            
            # Collect metrics after scaling
            with torch.no_grad():
                after_scaling = self.collect_metrics()
                cycle_dict['after_scaling'] = after_scaling
                if self.verbose > 0:
                    print(f"\nAfter scaling: Rwork={after_scaling['rwork']:.4f}, Rfree={after_scaling['rfree']:.4f}")
            
            # Store metrics before XYZ
            with torch.no_grad():
                before_xyz = self.collect_metrics()
                cycle_dict['xyz']['before'] = before_xyz
            
            # XYZ refinement with weight screening
            best_xyz_weight, xyz_screens = self.screen_xyz_weights(
                weights=xyz_weights,
                n_weights=n_xyz_weights,
                min_weight=xyz_min_weight,
                max_weight=xyz_max_weight,
                max_gap=max_gap,
                max_iter=max_iter,
            )
            cycle_dict['xyz']['weight_screens'] = xyz_screens
            cycle_dict['xyz']['weight'] = best_xyz_weight

            xyz_min_weight = best_xyz_weight / 10
            xyz_max_weight = best_xyz_weight * 10
            
            # Collect metrics after XYZ
            with torch.no_grad():
                after_xyz = self.collect_metrics()
                cycle_dict['xyz']['after'] = after_xyz
                if self.verbose > 0:
                    self.log_xyz_comparison(before_xyz, after_xyz, weight=best_xyz_weight)
            
            # Store metrics before ADP
            with torch.no_grad():
                before_adp = self.collect_metrics()
                cycle_dict['adp']['before'] = before_adp
            
            # ADP refinement with weight screening
            best_adp_weight, adp_screens = self.screen_adp_weights(
                weights=adp_weights,
                n_weights=n_adp_weights,
                min_weight=adp_min_weight,
                max_weight=adp_max_weight,
                max_gap=max_gap,
                max_bi_bj=max_bi_bj,
                max_iter=max_iter,
            )
            cycle_dict['adp']['weight_screens'] = adp_screens
            cycle_dict['adp']['weight'] = best_adp_weight

            adp_min_weight = best_adp_weight / 10
            adp_max_weight = best_adp_weight * 10
            
            # Collect final metrics after ADP
            with torch.no_grad():
                after_adp = self.collect_metrics()
                cycle_dict['adp']['after'] = after_adp
                if self.verbose > 0:
                    self.log_adp_comparison(before_adp, after_adp, weight=best_adp_weight)
            
            # Summary
            if self.verbose > 0:
                print(f"\n--- Cycle {cycle+1} Summary ---")
                print(f"  Best XYZ weight: {best_xyz_weight:.3f}")
                print(f"  Best ADP weight: {best_adp_weight:.3f}")
                print(f"  Final: Rwork={after_adp['rwork']:.4f}, Rfree={after_adp['rfree']:.4f}, "
                      f"Gap={after_adp['rfree_gap']:.4f}")
                print(f"  Bond RMSD: {after_adp.get('geom_bond_rmsd', 0):.4f} Å, "
                      f"Angle RMSD: {after_adp.get('geom_angle_rmsd', 0):.2f}°")
                print(f"  <B>: {after_adp.get('adp_mean_b', 0):.1f} Å², "
                      f"<Bi-Bj>: {after_adp.get('adp_mean_bi_bj', 0):.2f} Å²")
            
            self.history[master_key].append(cycle_dict)
        
        return self.history