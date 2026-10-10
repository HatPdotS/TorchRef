"""
Direct-summation P1 structure factors.

``ds_iso`` and ``ds_aniso`` run the backend that ``_backends.DS_BACKENDS`` selects (the
Triton kernel or the checkpointed reference), and ``compute_scattering_factors_batch`` is
the ITC92 f(s) sum the reference evaluates. Symmetry is applied by the caller.
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


# Capability-based backend dispatch (Triton on CUDA+fp32, else checkpointed
# eager). Keep ``triton_ds`` itself lazy (loaded inside dispatch) so a broken
# Triton install never breaks ``import torchref``.
from .dispatch import ds_aniso, ds_iso

__all__ = [
    # Scattering factor batch helper
    "compute_scattering_factors_batch",
    # Dispatch
    "ds_iso",
    "ds_aniso",
]
