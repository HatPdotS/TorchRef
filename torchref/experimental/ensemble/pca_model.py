"""
Fully-refinable PCA-space parameterization of an ensemble's coordinates.

.. warning::

   Experimental — part of ``torchref.experimental.ensemble``. The API and
   behaviour may change or be removed without notice.

Where :class:`~torchref.experimental.ensemble.low_rank_ensemble.LowRankXYZ` freezes the mean and
basis (only amplitudes refine), this module refines **all three** factors of the
low-rank decomposition::

    xyz_i = μ + Σ_k a_{ik} v_k          (Xc = A Vᵀ, rank K)

with ``μ`` (mean structure), ``V`` (K basis modes) and ``A`` (per-member
amplitudes) all ``nn.Parameter``. Seeded by an SVD of an existing (overfit)
ensemble — in the de-overfit workflow that ensemble must come from a saved
checkpoint (seed via ``--branch-from ckpt``, *not* ``--init-pdb``), since the
basis is only meaningful once real disorder has developed. Hypothesis:
refining in collective-coordinate space is an easier
landscape than raw Cartesian, and the explicit spectrum is the natural place for
the maxent (shrink + diversity) regularizer to act.

The loss depends only on the product ``μ + A Vᵀ``, which is gauge-invariant
under ``A → A R``, ``V → R⁻ᵀ V`` — so the A↔V rotational redundancy is harmless
to the optimizer (no gauge-fixing needed); a post-hoc SVD of the reconstruction
recovers a clean orthonormal PCA. ``K = N-1`` is a complete reparameterization
(same expressiveness as full Cartesian); smaller K is a hard rank cap.

Drop-in for ``model.xyz`` (called as ``self.xyz()`` everywhere), exposing the
same MixedTensor-compat shims as ``LowRankXYZ`` plus ``parameters()`` over
μ/A/V so the optimizer can collect all three.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class PCAEnsembleParam(nn.Module):
    """Refinable low-rank PCA parameterization ``xyz = μ + A Vᵀ``.

    .. warning::

       Experimental — API and behaviour may change without notice.

    Parameters
    ----------
    mu : torch.Tensor
        Mean coordinate vector, shape ``(D,)``, ``D = n_atoms * 3``.
    V : torch.Tensor
        Basis modes, shape ``(K, D)`` (seeded orthonormal; not constrained).
    A : torch.Tensor
        Per-member amplitudes, shape ``(N, K)``.
    n_members, n_atoms : int
        Ensemble shape; ``D == n_atoms * 3``.
    explained_variance : float
        Cumulative variance fraction captured at seed time (logging).
    """

    def __init__(
        self,
        mu: torch.Tensor,
        V: torch.Tensor,
        A: torch.Tensor,
        n_members: int,
        n_atoms: int,
        explained_variance: float = float("nan"),
    ) -> None:
        super().__init__()
        self.n_members = int(n_members)
        self.n_atoms = int(n_atoms)
        self.K = int(V.shape[0])
        self.explained_variance = float(explained_variance)
        # All three factors refine.
        self.mu = nn.Parameter(mu.detach().clone())          # (D,)
        self.V = nn.Parameter(V.detach().clone())            # (K, D)
        self.A = nn.Parameter(A.detach().clone())            # (N, K)

    @classmethod
    def from_ensemble(
        cls,
        xyz_flat: torch.Tensor,
        n_members: int,
        n_atoms: int,
        K: Optional[int] = None,
    ) -> "PCAEnsembleParam":
        """Seed from a flat ``(N*n_atoms, 3)`` ensemble via SVD of the centered
        member matrix. ``K`` defaults to ``N-1`` (complete reparameterization)."""
        N = int(n_members)
        with torch.no_grad():
            X = xyz_flat.detach().reshape(N, n_atoms * 3).to(torch.float64)
            mu = X.mean(dim=0)
            Xc = X - mu.unsqueeze(0)
            U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
            max_rank = max(1, N - 1)
            K = max_rank if K is None else min(int(K), max_rank)
            Vk = Vt[:K]                                      # (K, D)
            A0 = Xc @ Vk.T                                   # (N, K) = U S
            total = (S ** 2).sum().clamp_min(1e-30)
            explained = float((S[:K] ** 2).sum() / total)
        dtype = xyz_flat.dtype
        return cls(
            mu.to(dtype), Vk.to(dtype), A0.to(dtype),
            n_members=N, n_atoms=n_atoms, explained_variance=explained,
        ).to(xyz_flat.device)

    def forward(self) -> torch.Tensor:
        """Reconstruct the flat ``(N*n_atoms, 3)`` coordinate tensor."""
        recon = self.mu.unsqueeze(0) + self.A @ self.V       # (N, D)
        return recon.reshape(self.n_members * self.n_atoms, 3)

    # --- optimizer / wrapper-compat surface ---------------------------
    @property
    def refinable_params(self) -> nn.Parameter:
        """Primary leaf (amplitudes) — for code that expects a single tensor
        (e.g. diagnostics). The optimizer collects all of ``parameters()``."""
        return self.A

    @property
    def fixed_values(self) -> torch.Tensor:
        """Reference tensor (dtype/device) for ``get_aniso``'s empty path."""
        return self.mu

    @property
    def refinable_mask(self):
        return None

    def update_refinable_mask(self, mask) -> None:
        return None

    def reset_forward_cache(self) -> None:
        return None

    @torch.no_grad()
    def prune_modes(self, var_threshold: float = 1e-6) -> int:
        """Drop modes whose amplitude variance falls below ``var_threshold`` of
        the total (K↓ → cheaper). Returns the new K. Call between cycles."""
        col_var = (self.A ** 2).mean(dim=0)                  # (K,)
        keep = col_var > (var_threshold * col_var.sum().clamp_min(1e-30))
        if bool(keep.all()):
            return self.K
        idx = torch.nonzero(keep, as_tuple=False).flatten()
        self.A = nn.Parameter(self.A.data[:, idx].contiguous())
        self.V = nn.Parameter(self.V.data[idx, :].contiguous())
        self.K = int(idx.numel())
        return self.K

    def extra_repr(self) -> str:
        return (
            f"N={self.n_members}, n_atoms={self.n_atoms}, K={self.K}, "
            f"refine[μ,A,V], explained_var={self.explained_variance:.4f}"
        )
