"""
Rigid Body Refinement for Molecular Replacement.

Implements rigid body refinement where rotation and translation parameters
are optimized to minimize the difference between F_calc and F_obs.
This follows the rotation search (FRF) and translation search stages.

The refinement optimizes 6 parameters:
- 3 rotation angles (alpha, beta, gamma) as small perturbations
- 3 translation components (fractional coordinates)

Key design: Bypasses Model/MixedTensor to maintain gradient flow.
Stores all required tensors and uses FFT.compute_structure_factors() directly.

Gradient flow:
    d_alpha → rotation_matrix → xyz_transformed → FFT.compute_structure_factors() → loss

Uses ScalerBase for proper crystallographic scaling during optimization.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from torchref.scaling import Scaler
from torchref.model import SfFFT
from torchref.symmetry import spacegroup
from torchref.refinement.targets import MaximumLikelihoodXrayTarget
from torchref.base import rotation_matrix_euler_zyz

@dataclass
class RigidBodyResult:
    """
    Results from rigid body refinement.

    Attributes
    ----------
    final_rotation : torch.Tensor
        Final Euler angles (alpha, beta, gamma) in radians.
    final_translation_frac : torch.Tensor
        Final translation in fractional coordinates.
    initial_r_factor : float
        R-factor before refinement.
    final_r_factor : float
        R-factor after refinement.
    final_ml_loss : float
        Final ML loss value.
    n_steps : int
        Number of optimization steps performed.
    converged : bool
        Whether the refinement converged.
    """
    final_rotation: torch.Tensor
    final_translation_frac: torch.Tensor
    initial_r_factor: float
    final_r_factor: float
    final_ml_loss: float
    n_steps: int
    LBFGS_iterations: int
    LBFGS_function_evaluations: int
    converged: bool





class RigidBodyRefinement(nn.Module):
    """
    Rigid body refinement using FFT directly (bypasses Model/MixedTensor).

    Optimizes 6 parameters (3 rotation + 3 translation) to maximize
    agreement between calculated and observed structure factors using
    Maximum Likelihood target.

    Key design: Extracts all tensors from Model once at init, then uses
    FFT.compute_structure_factors() directly. This maintains gradient flow:
        d_alpha → rotation_matrix → xyz_transformed → FFT → loss

    Parameters
    ----------
    model : ModelFT
        Model with atomic coordinates (tensors extracted, not stored).
    data : ReflectionData
        Observed reflection data.
    initial_rotation : torch.Tensor, optional
        Initial Euler angles (alpha, beta, gamma) in radians.
        Default is (0, 0, 0).
    initial_translation : torch.Tensor, optional
        Initial fractional translation vector (3,).
        Default is [0, 0, 0].
    device : torch.device, optional
        Computation device. Default is CPU.

    Attributes
    ----------
    d_alpha, d_beta, d_gamma : nn.Parameter
        Refinable rotation perturbations.
    translation_frac : nn.Parameter
        Refinable fractional translation.
    scaler : ScalerBase
        Scaler for crystallographic scaling (jointly optimized).
    """

    def __init__(
        self,
        model,  # ModelFT
        data,  # ReflectionData
        expected_rotational_error: float = 0.1,
        initial_rotation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        initial_translation: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
        rfactor_converged_threshold: float = 0.45,
        max_res: float = 4.0,
        verbose: int = 1,
    ):
        super().__init__()
        self.device = device
        self.data = data

        xyz_iso, adp_iso, occ_iso, A_iso, B_iso = model.get_iso()

        self.register_buffer("xyz_initial", xyz_iso.detach().clone().to(device))
        self.register_buffer("adp_iso", adp_iso.detach().clone().to(device))
        self.register_buffer("occ_iso", occ_iso.detach().clone().to(device))
        self.register_buffer("A_iso", A_iso.detach().clone().to(device))
        self.register_buffer("B_iso", B_iso.detach().clone().to(device))
        centroid = torch.mean(self.xyz_initial, dim=0)
        self.register_buffer("centroid", centroid)

        # Move Cell (which holds fractional_matrix etc.) to the target device
        # so RigidBodyRefinement.get_transformed_xyz can mm a GPU tensor
        # against fractional_matrix.T without "mat2 on cpu" crashes.
        self.cell = data.cell.to(device=device) if hasattr(data.cell, "to") else data.cell
        self.spacegroup = data.spacegroup

        # Forward `device` to SfFFT — its `setup_grid` reads `self.device`
        # to allocate the real-space grid; without this, the grid lands on
        # CPU even when the surrounding RigidBodyRefinement is on cuda, and
        # the joint refine crashes with "mat2 on cuda, others on cpu" at
        # the first compute_structure_factors call.
        self.fft = SfFFT(self.cell, self.spacegroup, max_res=max_res,
                         device=device)


        self.verbose = verbose
        self.rfactor_converged_threshold = rfactor_converged_threshold
        # Get anisotropic atoms if any
        xyz_aniso, u_aniso, occ_aniso, A_aniso, B_aniso = model.get_aniso()
        self.has_aniso = len(xyz_aniso) > 0
        if self.has_aniso:
            self.xyz_aniso_original = xyz_aniso.detach().clone().to(device)
            self.u_aniso = u_aniso.detach().clone().to(device)
            self.occ_aniso = occ_aniso.detach().clone().to(device)
            self.A_aniso = A_aniso.detach().clone().to(device)
            self.B_aniso = B_aniso.detach().clone().to(device)
        else:
            self.xyz_aniso_original = None
            self.u_aniso = None
            self.occ_aniso = None
            self.A_aniso = None
            self.B_aniso = None




        # Store initial rotation
        self.register_buffer("initial_rotation", initial_rotation.to(device=device).clone())

        self.rotation_parameters = nn.Parameter(torch.zeros(3, device=device))
        self.expected_rotational_error = expected_rotational_error

        # Refinable translation (fractional coordinates)
        if initial_translation is None:
            initial_translation = torch.zeros(3, device=device)
        else:
            initial_translation = initial_translation.to(device=device).clone()
        self.translation_frac = nn.Parameter(initial_translation)

        # Use `Scaler` for per-bin scales + anisotropy correction, but skip
        # the bulk-solvent setup. The solvent mask is computed once from
        # `model.xyz()` and goes stale as the joint refine moves atoms; the
        # mismatch then biases the LBFGS gradient. Better to leave solvent
        # out of the joint refine — `fit_to_data` does a fresh
        # solvent-aware Scaler refit on the final polished model for the
        # user-facing R-work.
        self.scaler = Scaler(model=model, data=data, nbins=20,
                              verbose=0, device=device)
        fcalc_initial = self()
        self.scaler.calc_initial_scale(fcalc_initial)
        self.scaler.setup_anisotropy_correction()
        self.scaler.refine_lbfgs(fcalc=fcalc_initial)

        self.xray_target = MaximumLikelihoodXrayTarget(data=self.data, scaler=self.scaler)

    def get_rotation_matrix(self) -> torch.Tensor:
        """
        Compute current rotation matrix from Euler angles (differentiable).

        Returns
        -------
        torch.Tensor
            3x3 rotation matrix combining initial and perturbation rotations.
        """

        angles = self.initial_rotation + self.rotation
        return rotation_matrix_euler_zyz(angles)

    @property
    def rotation(self) -> torch.Tensor:
        """
        Get current rotation perturbation angles.

        Returns
        -------
        torch.Tensor
            Current (d_alpha, d_beta, d_gamma) in radians.
        """
        return (2 * torch.sigmoid(self.rotation_parameters) - 1) * self.expected_rotational_error

    def get_current_rotation_angles(self) -> torch.Tensor:
        """
        Get current rotation angles (initial + perturbation).

        Returns
        -------
        torch.Tensor
            Current (alpha, beta, gamma) in radians.
        """
        angles = self.initial_rotation + self.rotation
        return angles

    def get_transformed_xyz(self) -> torch.Tensor:
        """
        Transform coordinates - maintains gradient flow.

        Applies rotation around centroid, then translation.

        Returns
        -------
        torch.Tensor
            Transformed atomic coordinates with shape (n_atoms, 3).
        """
        R = self.get_rotation_matrix()

        # Rotate around centroid
        xyz_centered = self.xyz_initial - self.centroid
        xyz_rotated = xyz_centered @ R.T + self.centroid

        # Apply translation (fractional → Cartesian via cell.fractional_matrix.T;
        # see Cell.fractional_to_cartesian for the canonical convention).
        t_cart = self.translation_frac @ self.cell.fractional_matrix.T
        return xyz_rotated + t_cart

    def get_scale(self) -> float:
        """Get current scale factor from scaler."""
        return self.scaler.get_scale()

    def forward(self, debug: bool = False) -> torch.Tensor:
        """
        Compute unscaled structure factors using FFT directly.

        Gradient flows: d_alpha/d_beta/d_gamma → R → xyz → density → SF

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices with shape (n_reflections, 3).
        debug : bool
            If True, print gradient tracking info.

        Returns
        -------
        torch.Tensor
            Unscaled calculated structure factors (scaling done by scaler).
        """
        # Get transformed coordinates (has gradient to rotation params)
        xyz_transformed = self.get_transformed_xyz()

        hkl = self.data.hkl

        if debug:
            print(f"      xyz_transformed.requires_grad: {xyz_transformed.requires_grad}")
            print(f"      xyz_transformed.grad_fn: {xyz_transformed.grad_fn}")

        # Transform anisotropic atoms if present
        xyz_aniso = None
        if self.has_aniso:
            R = self.get_rotation_matrix()
            xyz_aniso_centered = self.xyz_aniso_original - self.centroid
            xyz_aniso_rotated = xyz_aniso_centered @ R.T + self.centroid
            t_cart = self.translation_frac @ self.cell.fractional_matrix.T
            xyz_aniso = xyz_aniso_rotated + t_cart

        # Compute structure factors via FFT (bypasses MixedTensor!)
        # Note: fractional matrices are now obtained from FFT's internal Cell object
        sf, _ = self.fft.compute_structure_factors(
            hkl=hkl,
            xyz_iso=xyz_transformed,
            adp_iso=self.adp_iso,
            occ_iso=self.occ_iso,
            A_iso=self.A_iso,
            B_iso=self.B_iso,
            xyz_aniso=xyz_aniso,
            u_aniso=self.u_aniso if self.has_aniso else None,
            occ_aniso=self.occ_aniso if self.has_aniso else None,
            A_aniso=self.A_aniso if self.has_aniso else None,
            B_aniso=self.B_aniso if self.has_aniso else None,
        )

        if debug:
            print(f"      sf.requires_grad: {sf.requires_grad}")
            print(f"      sf.grad_fn: {sf.grad_fn}")

        return sf

    def refine(
        self,
        n_tries: int = 1,
        n_iter: int = 100, 
    ) -> RigidBodyResult:
        """
        Run rigid body refinement using least-squares loss with Adam optimizer.

        Uses analytical scale fitting at each step rather than jointly optimizing
        the scale parameter with rotation/translation.

        Parameters
        ----------
        n_steps : int, optional
            Maximum number of optimization steps. Default is 100.
        lr : float, optional
            Learning rate for Adam optimizer. Default is 0.01.
        convergence_threshold : float, optional
            Stop if loss change is below this threshold. Default is 1e-6.
        print_interval : int, optional
            Print progress every N steps. Default is 5.
        verbose : bool, optional
            Print progress information. Default is True.
        loss_type : str, optional
            Loss function to use: "ls" for least-squares, "ml" for ML.
            Default is "ls".

        Returns
        -------
        RigidBodyResult
            Refinement results including final parameters and R-factors.
        """
        import sys

        if self.verbose > 2:
            print(f"    Setting up LBFGS optimizer niter = {n_iter} and max tries = {n_tries}")
            sys.stdout.flush()
        parameters = [self.rotation_parameters, self.translation_frac, *self.scaler.parameters()]

        self.optimizer = torch.optim.LBFGS(
            parameters,
            lr=1, max_iter=100, line_search_fn='strong_wolfe'
        )

        def loss():
            fcalc = self()
            return self.xray_target(fcalc)

        noise = 0


        def closure():
            self.optimizer.zero_grad()
            current_loss = loss()
            current_loss.backward()
            gradnorm = self.optimizer.param_groups[0]['params'][0].norm().item()
            if noise > 0:
                self.optimizer.param_groups[0]['params'][0].grad  += noise * torch.randn_like(self.optimizer.param_groups[0]['params'][0].grad) * gradnorm
            return current_loss
        
        rwork_initial, rfree_initial = self.scaler.rfactor(self())

        initial_loss = closure().item()
        if self.verbose > 0:
            print(f"Initial R-work: {rwork_initial:.4f}, R-free: {rfree_initial:.4f}, ML loss: {initial_loss:.4f}")

        from time import time
        start_time = time()

        tries_needed = 0

        while True:
            tries_needed += 1

            self.optimizer.step(closure)
            with torch.no_grad():
                current_loss = loss().item()
                if self.verbose > 1:
                    print(f"Iter {tries_needed}   Current ML loss: {current_loss:.4f}")
            final_loss = closure().item()
            final_rwork, final_rfree = self.scaler.rfactor(self())
            converged = final_rwork < self.rfactor_converged_threshold

            if converged or tries_needed >= n_tries:
                if noise > 0:
                    noise = 0
                    self.optimizer.step(closure) 
                if self.verbose > 1:
                    print(f"Converged at iteration {tries_needed} with R-work: {final_rwork:.4f}")
                break

            else:
                noise += 1e-2
                self.optimizer = torch.optim.LBFGS(
                    parameters,
                    lr=1, max_iter=100, line_search_fn='strong_wolfe'
                )

        end_time = time()

        if self.verbose > 0:
            print(f"\nRefinement complete after {tries_needed} steps in {end_time - start_time:.2f} seconds.")
            print(f"  Final R-work: {final_rwork:.4f} (improved by {rwork_initial - final_rwork:.4f})")
            print(f"  Final R-free: {final_rfree:.4f} (improved by {rfree_initial - final_rfree:.4f})")
            print(f"  Final rotation angles (deg):", self.get_current_rotation_angles().rad2deg().detach().cpu().numpy())
            print(f"  Final translation: {self.translation_frac.detach().cpu().numpy()}")

        return RigidBodyResult(
            final_rotation=self.get_current_rotation_angles().detach().cpu().numpy().tolist(),
            final_translation_frac=self.translation_frac.detach().cpu().numpy().tolist(),
            initial_r_factor=rwork_initial,
            final_r_factor=final_rwork,
            final_ml_loss=final_loss,
            LBFGS_iterations=self.optimizer.state['n_iter'],
            LBFGS_function_evaluations=self.optimizer.state['func_evals'],
            n_steps=tries_needed,
            converged=converged
        )

    def get_final_parameters(self) -> dict:
        """
        Get final refined parameters.

        Returns
        -------
        dict
            Dictionary with rotation angles (degrees), translation (fractional),
            and scale factor.
        """
        angles = self.get_current_rotation_angles().detach().cpu()
        rotation_perturbation = self.rotation.detach().cpu()
        return {
            'alpha_deg': np.degrees(angles[0].item()),
            'beta_deg': np.degrees(angles[1].item()),
            'gamma_deg': np.degrees(angles[2].item()),
            'translation_frac': self.translation_frac.detach().cpu().numpy(),
            'scale': self.get_scale(),
            'd_alpha_deg': np.degrees(rotation_perturbation[0].item()),
            'd_beta_deg': np.degrees(rotation_perturbation[1].item()),
            'd_gamma_deg': np.degrees(rotation_perturbation[2].item()),
        }
