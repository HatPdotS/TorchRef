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

from torchref.scaling import ScalerBase


def rotation_matrix_from_euler_zyz_torch(
    alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    """
    Create rotation matrix from ZYZ Euler angles (differentiable PyTorch version).

    R = Rz(alpha) @ Ry(beta) @ Rz(gamma)

    Parameters
    ----------
    alpha : torch.Tensor
        First rotation angle (around Z) in radians.
    beta : torch.Tensor
        Second rotation angle (around Y) in radians.
    gamma : torch.Tensor
        Third rotation angle (around Z) in radians.

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix.
    """
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)

    # Build rotation matrix element by element
    R = torch.stack([
        torch.stack([ca*cb*cg - sa*sg, -ca*cb*sg - sa*cg, ca*sb]),
        torch.stack([sa*cb*cg + ca*sg, -sa*cb*sg + ca*cg, sa*sb]),
        torch.stack([-sb*cg,            sb*sg,             cb])
    ])

    return R


def ml_xray_loss(
    fobs: torch.Tensor,
    fcalc: torch.Tensor,
    sigma: torch.Tensor,
    centric_flags: torch.Tensor,
) -> torch.Tensor:
    """
    Maximum Likelihood X-ray target with proper centric/acentric handling.

    Based on the MaximumLikelihoodXrayTarget implementation in
    torchref/refinement/targets/targets.py.

    Parameters
    ----------
    fobs : torch.Tensor
        Observed structure factor amplitudes.
    fcalc : torch.Tensor
        Calculated structure factors (complex or amplitude).
    sigma : torch.Tensor
        Standard deviations of observed amplitudes.
    centric_flags : torch.Tensor
        Boolean tensor indicating centric reflections.

    Returns
    -------
    torch.Tensor
        Mean ML loss across all reflections.
    """
    # Default model parameters (simplified for rigid body refinement)
    alpha = torch.ones_like(fobs)  # Figure of merit parameter
    beta = sigma ** 2  # Variance estimate
    epsilon = torch.ones_like(fobs)  # Symmetry factor

    fcalc_amp = torch.abs(fcalc)

    # Precompute common terms
    eb = (epsilon * beta).clamp(min=1e-6)

    # Acentric term (Rice distribution)
    term1 = -torch.log(2 * fobs / eb + 1e-12)
    term2 = (fobs ** 2) / eb
    term3 = (alpha * fcalc_amp) ** 2 / eb

    arg_bessel = 2 * alpha * fobs * fcalc_amp / eb
    # Use numerically stable I0 computation: log(I0(x)) approx x for large x
    term4 = -(
        torch.log(torch.special.i0e(arg_bessel) + 1e-12) + torch.abs(arg_bessel)
    )

    loss_acentric = term1 + term2 + term3 + term4

    # Centric term (Woolfson distribution)
    term1_c = -0.5 * torch.log(2 / (np.pi * eb) + 1e-12)
    term2_c = (fobs ** 2) / (2 * eb)
    term3_c = (alpha * fcalc_amp) ** 2 / (2 * eb)
    term4_c = -(alpha * fobs * fcalc_amp) / eb

    arg_exp = -2 * alpha * fobs * fcalc_amp / eb
    term5_c = -torch.log((1 + torch.exp(arg_exp.clamp(max=10))) / 2 + 1e-12)

    loss_centric = term1_c + term2_c + term3_c + term4_c + term5_c

    # Combine based on centric flags
    loss = torch.where(centric_flags, loss_centric, loss_acentric)

    return loss.mean()


def compute_r_factor(
    fobs: torch.Tensor, fcalc: torch.Tensor
) -> float:
    """
    Compute R-factor between observed and calculated structure factors.

    R = sum(||Fobs| - |Fcalc||) / sum(|Fobs|)

    Parameters
    ----------
    fobs : torch.Tensor
        Observed structure factor amplitudes.
    fcalc : torch.Tensor
        Calculated structure factors (complex or amplitude).

    Returns
    -------
    float
        R-factor value.
    """
    fobs_amp = torch.abs(fobs)
    fcalc_amp = torch.abs(fcalc)

    r = torch.sum(torch.abs(fobs_amp - fcalc_amp)) / torch.sum(fobs_amp)
    return r.item()


@dataclass
class RigidBodyResult:
    """
    Results from rigid body refinement.

    Attributes
    ----------
    final_rotation : Tuple[float, float, float]
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
    final_rotation: Tuple[float, float, float]
    final_translation_frac: torch.Tensor
    initial_r_factor: float
    final_r_factor: float
    final_ml_loss: float
    n_steps: int
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
    initial_rotation : Tuple[float, float, float], optional
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
        initial_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        initial_translation: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.data = data

        # Extract and store all tensors from model ONCE
        # These are detached - no connection to Model's MixedTensor
        self.original_xyz = model.xyz().detach().clone().to(device)
        self.centroid = self.original_xyz.mean(dim=0)

        # Get isotropic atomic parameters (B-factors, occupancy, scattering params)
        xyz_iso, b_iso, occ_iso, A_iso, B_iso = model.get_iso()
        self.b_iso = b_iso.detach().clone().to(device)
        self.occ_iso = occ_iso.detach().clone().to(device)
        self.A_iso = A_iso.detach().clone().to(device)
        self.B_iso = B_iso.detach().clone().to(device)

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

        # Store fractional matrices
        self.inv_fractional_matrix = model.inv_fractional_matrix.detach().clone().to(device)
        self.fractional_matrix = model.fractional_matrix.detach().clone().to(device)

        # Store FFT module reference (already initialized with cell/spacegroup/grid)
        self.fft = model._fft

        # Refinable rotation parameters (small perturbations around initial)
        self.d_alpha = nn.Parameter(torch.tensor(0.0, device=device))
        self.d_beta = nn.Parameter(torch.tensor(0.0, device=device))
        self.d_gamma = nn.Parameter(torch.tensor(0.0, device=device))

        # Store initial rotation
        self.alpha0, self.beta0, self.gamma0 = initial_rotation

        # Refinable translation (fractional coordinates)
        if initial_translation is None:
            initial_translation = torch.zeros(3, device=device)
        else:
            initial_translation = initial_translation.to(device=device).clone()
        self.translation_frac = nn.Parameter(initial_translation)

        # Create ScalerBase for proper crystallographic scaling
        # Will be initialized with fcalc in refine()
        self.scaler = ScalerBase(data=data, nbins=20, verbose=0, device=device)

    def get_rotation_matrix(self) -> torch.Tensor:
        """
        Compute current rotation matrix from Euler angles (differentiable).

        Returns
        -------
        torch.Tensor
            3x3 rotation matrix combining initial and perturbation rotations.
        """
        alpha = self.alpha0 + self.d_alpha
        beta = self.beta0 + self.d_beta
        gamma = self.gamma0 + self.d_gamma
        return rotation_matrix_from_euler_zyz_torch(alpha, beta, gamma)

    def get_current_rotation_angles(self) -> Tuple[float, float, float]:
        """
        Get current rotation angles (initial + perturbation).

        Returns
        -------
        Tuple[float, float, float]
            Current (alpha, beta, gamma) in radians.
        """
        alpha = self.alpha0 + self.d_alpha.item()
        beta = self.beta0 + self.d_beta.item()
        gamma = self.gamma0 + self.d_gamma.item()
        return (alpha, beta, gamma)

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
        xyz_centered = self.original_xyz - self.centroid
        xyz_rotated = xyz_centered @ R.T + self.centroid

        # Apply translation (fractional -> Cartesian)
        t_cart = self.translation_frac @ self.fractional_matrix
        return xyz_rotated + t_cart

    def get_scale(self) -> float:
        """Get current scale factor from scaler."""
        return self.scaler.get_scale()

    def forward(self, hkl: torch.Tensor, debug: bool = False) -> torch.Tensor:
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

        if debug:
            print(f"      xyz_transformed.requires_grad: {xyz_transformed.requires_grad}")
            print(f"      xyz_transformed.grad_fn: {xyz_transformed.grad_fn}")

        # Transform anisotropic atoms if present
        xyz_aniso = None
        if self.has_aniso:
            R = self.get_rotation_matrix()
            xyz_aniso_centered = self.xyz_aniso_original - self.centroid
            xyz_aniso_rotated = xyz_aniso_centered @ R.T + self.centroid
            t_cart = self.translation_frac @ self.fractional_matrix
            xyz_aniso = xyz_aniso_rotated + t_cart

        # Compute structure factors via FFT (bypasses MixedTensor!)
        sf, _ = self.fft.compute_structure_factors(
            hkl=hkl,
            xyz_iso=xyz_transformed,
            b_iso=self.b_iso,
            occ_iso=self.occ_iso,
            A_iso=self.A_iso,
            B_iso=self.B_iso,
            inv_fractional_matrix=self.inv_fractional_matrix,
            fractional_matrix=self.fractional_matrix,
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
        n_steps: int = 100,
        lr: float = 1.0,
        convergence_threshold: float = 1e-5,
        print_interval: int = 5,
        verbose: bool = True,
        line_search_fn: str = "strong_wolfe",
    ) -> RigidBodyResult:
        """
        Run rigid body refinement using ML target with LBFGS optimizer.

        Optimizes rigid body parameters (rotation, translation) AND
        scale parameter jointly for best Fcalc-Fobs agreement.

        Parameters
        ----------
        n_steps : int, optional
            Maximum number of optimization steps. Default is 100.
        lr : float, optional
            Learning rate for LBFGS. Default is 1.0.
        convergence_threshold : float, optional
            Stop if loss change is below this threshold. Default is 1e-5.
        print_interval : int, optional
            Print progress every N steps. Default is 5.
        verbose : bool, optional
            Print progress information. Default is True.
        line_search_fn : str, optional
            Line search function for LBFGS. Default is "strong_wolfe".

        Returns
        -------
        RigidBodyResult
            Refinement results including final parameters and R-factors.
        """
        import sys

        if verbose:
            print("    Setting up LBFGS optimizer...")
            sys.stdout.flush()

        # Get reflection data
        hkl, fobs, sigma, rfree_mask = self.data()

        if verbose:
            print(f"    Got {len(hkl)} reflections, moving to device...")
            sys.stdout.flush()

        hkl = hkl.to(self.device)
        fobs = fobs.to(self.device)
        sigma = sigma.to(self.device)
        rfree_mask = rfree_mask.to(self.device)

        # Handle MaskedTensor: get underlying data
        if hasattr(fobs, 'get_data'):
            fobs_data = fobs.get_data()
        else:
            fobs_data = fobs

        if hasattr(sigma, 'get_data'):
            sigma_data = sigma.get_data()
        else:
            sigma_data = sigma

        # Get centric flags if available
        if hasattr(self.data, 'centric') and self.data.centric is not None:
            centric_flags = self.data.centric.to(self.device)
        else:
            centric_flags = torch.zeros_like(fobs_data, dtype=torch.bool)

        # Extract work set indices for training (need this before initial R-factor)
        work_indices = rfree_mask.nonzero(as_tuple=True)[0]
        fobs_work = fobs_data[work_indices]
        sigma_work = sigma_data[work_indices]
        centric_work = centric_flags[work_indices]

        if verbose:
            print("    Computing initial structure factors and initializing scaler...")
            sys.stdout.flush()

        # Compute initial Fcalc and initialize scaler
        with torch.no_grad():
            fcalc_initial = self.forward(hkl)
            # Initialize scaler with initial Fcalc
            self.scaler.initialize(fcalc_initial)
            # Compute initial R-factor with analytical scale
            fcalc_work_init = fcalc_initial[work_indices]
            fcalc_amp = torch.abs(fcalc_work_init)
            scale_init = (fobs_work * fcalc_amp).sum() / (fcalc_amp**2).sum().clamp(min=1e-10)
            initial_r_factor = compute_r_factor(fobs_work, scale_init * fcalc_work_init)

        if verbose:
            print(f"Rigid body refinement starting (LBFGS)")
            print(f"  Initial R-work: {initial_r_factor:.4f}")
            print(f"  Initial rotation: ({np.degrees(self.alpha0):.2f}, "
                  f"{np.degrees(self.beta0):.2f}, {np.degrees(self.gamma0):.2f}) deg")
            print(f"  Initial translation: {self.translation_frac.detach().cpu().numpy()}")

        # Setup LBFGS optimizer with rigid body params + scaler params
        rigid_params = [self.d_alpha, self.d_beta, self.d_gamma, self.translation_frac]
        self.scaler.unfreeze()
        all_params = rigid_params + list(self.scaler.parameters())
        optimizer = torch.optim.LBFGS(
            all_params,
            lr=lr,
            max_iter=20,
            history_size=10,
            line_search_fn=line_search_fn,
        )

        prev_loss = float('inf')
        converged = False
        final_step = 0

        def closure():
            optimizer.zero_grad()
            # Compute unscaled Fcalc
            fcalc_raw = self.forward(hkl)
            # Apply scaling via scaler
            fcalc_scaled = self.scaler(fcalc_raw)
            fcalc_work = fcalc_scaled[work_indices]
            loss = ml_xray_loss(fobs_work, fcalc_work, sigma_work, centric_work)
            loss.backward()
            return loss

        for step in range(n_steps):
            final_step = step

            # LBFGS step
            loss = optimizer.step(closure)
            loss_val = loss.item()

            # Debug: check gradients on first step
            if step == 0 and verbose:
                print(f"    Gradient check (after first step):")
                print(f"      d_alpha.grad: {self.d_alpha.grad}")
                print(f"      d_beta.grad: {self.d_beta.grad}")
                print(f"      d_gamma.grad: {self.d_gamma.grad}")
                print(f"      translation_frac.grad: {self.translation_frac.grad}")
                print(f"      scaler log_scale.grad: {self.scaler.log_scale.grad.mean() if self.scaler.log_scale.grad is not None else None}")
                print(f"      loss value: {loss_val}")
                sys.stdout.flush()

            # Check convergence
            if abs(prev_loss - loss_val) < convergence_threshold:
                converged = True
                if verbose:
                    print(f"Converged at step {step}")
                break
            prev_loss = loss_val

            # Print progress with analytical scale fit for accurate R-factor
            if verbose and (step % print_interval == 0 or step == n_steps - 1):
                with torch.no_grad():
                    fcalc_raw = self.forward(hkl)
                    fcalc_work_eval = fcalc_raw[work_indices]
                    # Analytical scale for R-factor evaluation
                    fcalc_amp = torch.abs(fcalc_work_eval)
                    scale_fit = (fobs_work * fcalc_amp).sum() / (fcalc_amp**2).sum().clamp(min=1e-10)
                    rwork = compute_r_factor(fobs_work, scale_fit * fcalc_work_eval)
                    print(f"  Step {step:3d}: R-work = {rwork:.4f}, "
                          f"ML-loss = {loss_val:.4f}, scale = {scale_fit.item():.4f}")

        # Final evaluation with analytical scale fit
        with torch.no_grad():
            fcalc_final = self.forward(hkl)
            fcalc_work_final = fcalc_final[work_indices]
            # Analytical scale for final R-factor
            fcalc_amp = torch.abs(fcalc_work_final)
            scale_final = (fobs_work * fcalc_amp).sum() / (fcalc_amp**2).sum().clamp(min=1e-10)
            final_r_factor = compute_r_factor(fobs_work, scale_final * fcalc_work_final)
            # Compute final loss with analytical scale
            final_loss = ml_xray_loss(
                fobs_work,
                scale_final * fcalc_work_final,
                sigma_work,
                centric_work
            ).item()

        if verbose:
            print(f"\nRefinement complete after {final_step + 1} steps")
            print(f"  Final R-work: {final_r_factor:.4f} (improved by {initial_r_factor - final_r_factor:.4f})")
            print(f"  Final rotation perturbation: ({np.degrees(self.d_alpha.item()):.4f}, "
                  f"{np.degrees(self.d_beta.item()):.4f}, {np.degrees(self.d_gamma.item()):.4f}) deg")
            print(f"  Final translation: {self.translation_frac.detach().cpu().numpy()}")

        return RigidBodyResult(
            final_rotation=self.get_current_rotation_angles(),
            final_translation_frac=self.translation_frac.detach().clone(),
            initial_r_factor=initial_r_factor,
            final_r_factor=final_r_factor,
            final_ml_loss=final_loss,
            n_steps=final_step + 1,
            converged=converged,
        )

    def apply_to_model(self, model):
        """
        Apply current transformation to a model.

        Parameters
        ----------
        model : Model
            Model to transform.

        Returns
        -------
        Model
            Transformed model.
        """
        # Get final transformation
        R = self.get_rotation_matrix().detach()
        t_frac = self.translation_frac.detach()

        # Apply rotation and translation
        model.rotate(R.cpu(), center=self.centroid.cpu())
        model.translate(t_frac.cpu(), fractional=True)

        return model

    def get_final_parameters(self) -> dict:
        """
        Get final refined parameters.

        Returns
        -------
        dict
            Dictionary with rotation angles (degrees), translation (fractional),
            and scale factor.
        """
        alpha, beta, gamma = self.get_current_rotation_angles()
        return {
            'alpha_deg': np.degrees(alpha),
            'beta_deg': np.degrees(beta),
            'gamma_deg': np.degrees(gamma),
            'translation_frac': self.translation_frac.detach().cpu().numpy(),
            'scale': self.get_scale(),
            'd_alpha_deg': np.degrees(self.d_alpha.item()),
            'd_beta_deg': np.degrees(self.d_beta.item()),
            'd_gamma_deg': np.degrees(self.d_gamma.item()),
        }
