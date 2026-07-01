"""
Low-rank (frozen-basis PCA) reparameterization of an ensemble's coordinates.

.. warning::

   Experimental — part of ``torchref.experimental.ensemble``. The API and
   behaviour may change or be removed without notice.

The full ensemble stores ``N * n_atoms * 3`` independent coordinates — the
overfitting capacity that produces the work-free gap. This module locks the
ensemble to a ``K``-dimensional affine subspace::

    xyz_member[i] = mu + amplitudes[i] @ V

where ``mu`` (the mean structure) and ``V`` (the top-K principal coordinate
modes) are **frozen** buffers computed once by
:meth:`EnsembleModel.enable_low_rank`, and only the per-member amplitudes
``A`` (shape ``(N, K)``) refine. Degrees of freedom collapse from
``N·n_atoms·3`` to ``N·K``. The basis is computed from the *current* ensemble,
so the de-overfit workflow seeds it from a saved checkpoint
(``--branch-from ckpt``, *not* ``--init-pdb``) — a fresh replicate-and-perturb
ensemble has only near-degenerate spread to decompose.

Drop-in for the ``MixedTensor`` that normally lives at ``model.xyz``: it is
called as ``self.xyz()`` everywhere in the model, returns the flat
``(N·n_atoms, 3)`` coordinate tensor, and exposes ``.refinable_params`` so
``Model.parameters_of_types(("xyz",))`` collects the amplitudes for the
optimizer with no other changes. Mirrors the existing
``self.xyz = SegmentedInternalCoordinateTensor(...)`` swap pattern in
``model.py``.
"""

from __future__ import annotations

import torch
from torch import nn


class LowRankXYZ(nn.Module):
    """Frozen-basis low-rank coordinate parameterization for an ensemble.

    .. warning::

       Experimental — API and behaviour may change without notice.

    Parameters
    ----------
    mu : torch.Tensor
        Mean coordinate vector, shape ``(D,)`` with ``D = n_atoms * 3``
        (row-major over atoms then xyz). Frozen buffer.
    V : torch.Tensor
        Orthonormal principal modes, shape ``(K, D)``. Frozen buffer.
    amplitudes : torch.Tensor
        Per-member amplitudes, shape ``(N, K)``. The sole refinable leaf.
    n_members : int
        Number of ensemble members ``N``.
    n_atoms : int
        Atoms per member (so ``D == n_atoms * 3``).
    explained_variance : float
        Cumulative fraction of ensemble coordinate variance captured by the
        retained ``K`` modes — stored for logging / result metadata.

    Notes
    -----
    No forward caching: ``forward`` is a single ``(N, K) @ (K, D)`` matmul
    (a few million flops at ``K ~ 8``), recomputed each call. The model's
    structure-factor cache (``CachedForwardMixin``) still invalidates
    correctly because ``amplitudes`` is a registered ``nn.Parameter`` whose
    ``_version`` bumps on each optimizer step.
    """

    def __init__(
        self,
        mu: torch.Tensor,
        V: torch.Tensor,
        amplitudes: torch.Tensor,
        n_members: int,
        n_atoms: int,
        explained_variance: float = float("nan"),
    ) -> None:
        super().__init__()
        self.n_members = int(n_members)
        self.n_atoms = int(n_atoms)
        self.K = int(V.shape[0])
        self.explained_variance = float(explained_variance)

        self.register_buffer("mu", mu.detach().clone())          # (D,)
        self.register_buffer("V", V.detach().clone())            # (K, D)
        self.amplitudes = nn.Parameter(amplitudes.detach().clone())  # (N, K)

    def forward(self) -> torch.Tensor:
        """Reconstruct the flat ``(N·n_atoms, 3)`` coordinate tensor."""
        recon = self.mu.unsqueeze(0) + self.amplitudes @ self.V  # (N, D)
        return recon.reshape(self.n_members * self.n_atoms, 3)

    @property
    def refinable_params(self) -> nn.Parameter:
        """The amplitudes leaf — picked up by ``parameters_of_types``."""
        return self.amplitudes

    # --- MixedTensor-compat shims -------------------------------------
    # The model's ``get_aniso`` reads ``self.xyz.fixed_values`` purely as a
    # dtype/device reference for empty placeholders (the ensemble has no
    # anisotropic atoms); ``mu`` serves that role. The mask hooks are no-ops
    # because the amplitudes are always fully refinable.

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
        """No-op: the reconstruction is recomputed on every call."""
        return None

    def extra_repr(self) -> str:
        return (
            f"N={self.n_members}, n_atoms={self.n_atoms}, K={self.K}, "
            f"dof={self.n_members * self.K}, "
            f"explained_var={self.explained_variance:.4f}"
        )
