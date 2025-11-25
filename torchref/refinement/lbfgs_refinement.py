"""
LBFGS-based refinement framework for crystallographic structure refinement.

This module provides an LBFGS optimizer-based refinement approach which has been
shown to converge much faster than first-order optimizers (Adam, SGD, etc.).
LBFGS typically reaches near-convergence in just 1-2 macro cycles.
"""

import torch
from typing import Optional, Dict, List, Tuple
from torchref.refinement.base_refinement import Refinement


class LBFGSRefinement(Refinement):
    """
    LBFGS-based refinement subclass that uses the L-BFGS optimizer for fast convergence.
    
    L-BFGS (Limited-memory BFGS) is a quasi-Newton optimization method that approximates
    the Hessian matrix, leading to much faster convergence than first-order methods.
    
    Key advantages:
    - Converges in 1-2 macro cycles (vs 5+ for Adam)
    - Better final R-factors
    - More stable convergence
    - Automatically handles step size via line search
    
    Usage:
        from torchref.refinement.loss_weighting import ResolutionDependentWeighting
        
        weighter = ResolutionDependentWeighting()
        refinement = LBFGSRefinement(mtz_file, pdb_file, weighter=weighter, target_mode='ml')
        refinement.refine(macro_cycles=2)
    """

    def __init__(self, *args, target_mode: str = 'gaussian', **kwargs):
        """
        Initialize LBFGS refinement.
        
        Args:
            target_mode: X-ray target mode ('gaussian', 'ls', or 'ml')
            *args, **kwargs: Passed to parent Refinement class
        """
        super().__init__(*args, **kwargs)
        
        # Set the X-ray target mode (uses the new target system from base class)
        self.set_xray_target_mode(target_mode)
        self.target_mode = target_mode

    def xray_loss(self):
        """Compute X-ray loss using the instantiated target."""
        return self.xray_loss_work()

    def refine_adp(self):
        """
        Refine B-factors (ADP).
        
        Args:
            cycle (int): Current refinement cycle for weighting module
        """
        self.model.freeze_all()
        self.model.unfreeze('b')
    

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=1.0,
            max_iter=20,
            history_size=100,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            loss = self.adp_loss() * self.effective_weights['adp'] + self.xray_loss() * self.effective_weights['xray']
            loss.backward()
            return loss

        optimizer.step(closure)
        self.model.unfreeze_all()

    def refine_xyz(self):
        """
        Refine coordinates (XYZ).
        
        Args:
            cycle (int): Current refinement cycle for weighting module
        """
        self.model.freeze_all()
        self.scaler.freeze()
        self.model.unfreeze('xyz')

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=1.0,
            max_iter=20,
            history_size=100,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            loss = self.restraints_loss() * self.effective_weights['restraints'] + self.xray_loss() * self.effective_weights['xray']
            loss.backward()
            return loss

        optimizer.step(closure)
        self.model.unfreeze_all()
    
    def regularize_xyz(self,lr=0.1):
        """
        Apply regularization to coordinates (XYZ).
        """
        self.model.freeze_all()
        self.scaler.freeze()
        self.model.unfreeze('xyz')

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=lr,
            max_iter=20,
            history_size=100,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            loss = self.restraints_loss()
            loss.backward()
            return loss

        optimizer.step(closure)
        self.model.unfreeze_all()
    
    def regularize_adp(self,lr=0.1):
        """
        Apply regularization to B-factors (ADP).
        """
        self.model.freeze_all()
        self.model.unfreeze('b')

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=0.1,
            max_iter=20,
            history_size=100,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            loss = self.adp_loss()
            loss.backward()
            return loss

        optimizer.step(closure)
        self.model.unfreeze_all()

    def refine_xyz_adamW(self, lr=1e-3, steps=100):
        """
        Refine coordinates (XYZ) using Adam optimizer as an alternative.
        
        Args:
            lr (float): Learning rate for Adam optimizer
            steps (int): Number of Adam optimization steps
        """
        self.model.freeze_all()
        self.scaler.freeze()
        self.model.unfreeze('xyz')

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr
        )

        for step in range(steps):
            optimizer.zero_grad()
            loss = self.restraints_loss() * self.effective_weights['restraints'] + self.xray_loss() * self.effective_weights['xray']
            loss.backward()
            if self.verbose > 1 and step % 10 == 0:
                print(f"Step {step+1}/{steps}, Loss: {loss.item():.4f}")
            optimizer.step()

        self.model.unfreeze_all()
    
    def refine_b_adamW(self, lr=1e-3, steps=100):
        """
        Refine B-factors (ADP) using Adam optimizer as an alternative.
        
        Args:
            lr (float): Learning rate for Adam optimizer
            steps (int): Number of Adam optimization steps
        """
        self.model.freeze_all()
        self.model.unfreeze('b')

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr
        )

        for step in range(steps):
            optimizer.zero_grad()
            loss = self.adp_loss() * self.effective_weights['adp'] + self.xray_loss() * self.effective_weights['xray']
            loss.backward()
            if self.verbose > 1 and step % 10 == 0:
                print(f"Step {step+1}/{steps}, Loss: {loss.item():.4f}")
            optimizer.step()
        self.model.unfreeze_all()
    
    def regularize_xyz_adp_to_rfactor_gap(self, lr=1e-1, max_steps=100, target_rfactor_gap=0.05):

        """
        Apply regularization to both coordinates (XYZ) and B-factors (ADP) using Adam optimizer as an alternative.
        
        Args:
            lr (float): Learning rate for Adam optimizer
            steps (int): Number of Adam optimization steps
        """
        self.model.freeze_all()
        self.scaler.freeze()
        self.model.unfreeze('xyz')
        self.model.unfreeze('b')

        def loss_fn():
            return (self.restraints_loss() +
                    self.adp_loss())
        def closure():
            optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr
        )
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


    def refine_everything_adamW(self, lr=1e-3, steps=100):
        """
        Refine both coordinates (XYZ) and B-factors (ADP) using Adam optimizer as an alternative.
        
        Args:
            lr (float): Learning rate for Adam optimizer
            steps (int): Number of Adam optimization steps
        """
        self.model.freeze_all()
        self.scaler.unfreeze()
        self.model.unfreeze('xyz')
        self.model.unfreeze('b')

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr
        )

        for step in range(steps):
            optimizer.zero_grad()
            loss = (self.restraints_loss() * self.effective_weights['restraints'] +
                    self.adp_loss() * self.effective_weights['adp'] +
                    self.xray_loss() * self.effective_weights['xray'])
            loss.backward()
            if self.verbose > 1 and step % 10 == 0:
                print(f"Step {step+1}/{steps}, Loss: {loss.item():.4f}")
            optimizer.step()

    def refine_everything_lbfgs(self):
        
        """
        Refine both coordinates (XYZ) and B-factors (ADP) using LBFGS optimizer as an alternative.
        """

        self.model.freeze_all()
        self.scaler.unfreeze()
        self.model.unfreeze_all()

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=1.0,
            max_iter=20,
            history_size=100,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            loss = (self.restraints_loss() * self.effective_weights['restraints'] +
                    self.adp_loss() * self.effective_weights['adp'] +
                    self.xray_loss() * self.effective_weights['xray'])
            loss.backward()
            return loss

        optimizer.step(closure)
        self.model.unfreeze_all()


    def refine(self, macro_cycles=5):
        """
        Run full LBFGS refinement cycle (ADP + XYZ).
        
        Args:
            macro_cycles (int): Number of refinement cycles to perform
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
            cycle_dict = {}
            cycle_dict['cycle'] = cycle + 1
            self.update_effective_weights(cycle=cycle)
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"LBFGS Refinement - Cycle {cycle+1}/{macro_cycles}")
                print(f"{'='*60}")
            
            with torch.no_grad():
                rwork, rfree = self.get_rfactor()
                cycle_dict['rwork_before_not_scaled'] = rwork
                cycle_dict['rfree_before_not_scaled'] = rfree
                
            self.get_scales()
            
            with torch.no_grad():
                rwork, rfree = self.get_rfactor()
                cycle_dict['rwork_before_scaled'] = rwork
                cycle_dict['rfree_before_scaled'] = rfree
                if self.verbose > 0:
                    print(f"After scaling: Rwork={rwork:.4f}, Rfree={rfree:.4f}")

            # XYZ refinement with cycle-aware weighting
            self.refine_xyz()
            # self.regularize_xyz(lr=0.2)
            
            with torch.no_grad():
                cycle_dict['rwork_after_xyz_scaled'], cycle_dict['rfree_after_xyz_scaled'] = self.get_rfactor()
                cycle_dict['restraints_weight'] = self.effective_weights.get('restraints', 0.0)
                cycle_dict['nll_work_after_xyz'], cycle_dict['nll_free_after_xyz'] = self.nll_xray()
                cycle_dict['nll_bonds'] = self.restraints.nll_bonds().mean().item()
                cycle_dict['nll_angles'] = self.restraints.nll_angles().mean().item()
                cycle_dict['nll_torsion'] = self.restraints.nll_torsions().mean().item()
                cycle_dict['nll_planes'] = self.restraints.nll_planes().mean().item()
                cycle_dict['nll_vdw'] = self.restraints.nll_vdw().mean().item()
                if self.verbose > 0:
                    rwork = cycle_dict['rwork_after_xyz_scaled']
                    rfree = cycle_dict['rfree_after_xyz_scaled']
                    rw = cycle_dict['restraints_weight']
                    if isinstance(rw, torch.Tensor): rw = rw.item()
                    print(f"After XYZ: Rwork={rwork:.4f}, Rfree={rfree:.4f}, "
                          f"restraint_weight={rw:.3f}")
            if cycle_dict['rwork_after_xyz_scaled'] - cycle_dict['rfree_after_xyz_scaled'] > 0.05:
                if self.verbose > 0:
                    print("Large R-factor gap detected after XYZ refinement. Applying additional regularization.")
                self.regularize_xyz(lr=0.6)
            with torch.no_grad():
                rwork, rfree = self.get_rfactor()
                cycle_dict['rwork_after_xyz_reg_scaled'] = rwork
                cycle_dict['rfree_after_xyz_reg_scaled'] = rfree
                if self.verbose > 0:
                    print(f"After XYZ Regularization: Rwork={rwork:.4f}, Rfree={rfree:.4f}")

            # B-factor refinement with cycle-aware weighting
            self.refine_adp()
            # self.regularize_adp(lr=0.2)
            
            with torch.no_grad():
                cycle_dict['rwork_after_adp_scaled'], cycle_dict['rfree_after_adp_scaled'] = self.get_rfactor()
                cycle_dict['adp_weight'] = self.effective_weights.get('adp', 0.0)
                cycle_dict['nll_work_after_adp'], cycle_dict['nll_free_after_adp'] = self.nll_xray()
                if self.verbose > 0:
                    rwork = cycle_dict['rwork_after_adp_scaled']
                    rfree = cycle_dict['rfree_after_adp_scaled']
                    aw = cycle_dict['adp_weight']
                    if isinstance(aw, torch.Tensor): aw = aw.item()
                    print(f"After ADP: Rwork={rwork:.4f}, Rfree={rfree:.4f}, "
                          f"adp_weight={aw:.3f}")
                    print(f"Effective weights: {self.effective_weights}")

            if cycle_dict['rwork_after_adp_scaled'] - cycle_dict['rfree_after_adp_scaled'] > 0.05:
                if self.verbose > 0:
                    print("Large R-factor gap detected after ADP refinement. Applying additional regularization.")
                self.regularize_adp(lr=0.6)

            with torch.no_grad():
                rwork, rfree = self.get_rfactor()
                cycle_dict['rwork_after_adp_reg_scaled'] = rwork
                cycle_dict['rfree_after_adp_reg_scaled'] = rfree
                if self.verbose > 0:
                    print(f"After ADP Regularization: Rwork={rwork:.4f}, Rfree={rfree:.4f}")

            # Convert tensors to scalars
            for key in cycle_dict:
                if isinstance(cycle_dict[key], torch.Tensor):
                    cycle_dict[key] = cycle_dict[key].mean().item()

            self.history[master_key].append(cycle_dict)

        return self.history