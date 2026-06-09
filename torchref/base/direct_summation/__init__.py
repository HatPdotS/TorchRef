"""
Structure factor calculation functions.

This submodule provides functions for computing structure factors
from atomic models:
- Isotropic structure factor calculations
- Anisotropic structure factor calculations
- Correction terms (anharmonic, core deformation)
"""

import torch


def compute_scattering_factors_batch(
    s_batch: torch.Tensor, A: torch.Tensor, B_coeff: torch.Tensor
) -> torch.Tensor:
    """
    Compute scattering factors for a batch of reflections.

    Uses the ITC92 (International Tables of Crystallography) 5-Gaussian
    approximation for atomic scattering factors.

    Parameters
    ----------
    s_batch : torch.Tensor
        Scattering vector magnitudes (batch_size,).
    A : torch.Tensor
        ITC92 A coefficients (N_atoms, 5).
    B_coeff : torch.Tensor
        ITC92 B coefficients (N_atoms, 5).

    Returns
    -------
    torch.Tensor
        Scattering factors (batch_size, N_atoms).
    """
    # s: (batch,) -> (batch, 1, 1)
    s_sq = (s_batch.reshape(-1, 1, 1) ** 2) / 4
    # A, B: (N_atoms, 5) -> (1, N_atoms, 5)
    A_exp = A.unsqueeze(0)
    B_exp = B_coeff.unsqueeze(0)
    # Compute: (batch, N_atoms, 5)
    exp_terms = torch.exp(-B_exp * s_sq)
    # Sum over Gaussians: (batch, N_atoms)
    return torch.sum(A_exp * exp_terms, dim=-1)


from .isotropic import (
    iso_structure_factor_torched,
    iso_structure_factor_torched_no_complex,
)

from .anisotropic import (
    aniso_structure_factor_torched,
    aniso_structure_factor_torched_no_complex,
)

from .corrections import (
    anharmonic_correction,
    anharmonic_correction_no_complex,
    core_deformation,
    multiplication_quasi_complex_tensor,
)

# Capability-based backend dispatch (Triton on CUDA+fp32, else checkpointed
# eager). Keep ``triton_ds`` itself lazy (loaded inside dispatch) so a broken
# Triton install never breaks ``import torchref``.
from .dispatch import Engine, ds_aniso, ds_iso

__all__ = [
    # Isotropic
    "iso_structure_factor_torched",
    "iso_structure_factor_torched_no_complex",
    # Anisotropic
    "aniso_structure_factor_torched",
    "aniso_structure_factor_torched_no_complex",
    # Corrections
    "anharmonic_correction",
    "anharmonic_correction_no_complex",
    "core_deformation",
    "multiplication_quasi_complex_tensor",
    # Dispatch
    "Engine",
    "ds_iso",
    "ds_aniso",
]
