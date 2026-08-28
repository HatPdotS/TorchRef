"""Reciprocal-space ("late") symmetry for structure factor calculation.

The alternative to symmetrizing the density map before the FFT ("early" symmetry, via
:meth:`~torchref.symmetry.symmetry.Symmetry.symmetrize_map`): here symmetry is applied
to the P1 transform afterwards, avoiding the map symmetrization entirely. Per operation
{R|t},

    F_sym(h) = sum_ops exp(2 pi i h.t) * F_P1(R^T h)

and because crystallographic R is integer-valued, ``R^T h`` lands exactly on grid
points and needs no interpolation. **Every grid or map argument here is the P1 one** --
feeding in an already-symmetrized grid double-counts.

Both halves of that sum come from :class:`~torchref.symmetry.symmetry.Symmetry`:
``R^T h`` from :meth:`~torchref.symmetry.symmetry.Symmetry.expand_reciprocal` and the
phases from :meth:`~torchref.symmetry.symmetry.Symmetry.phase_factors`. Reach this
class through
:meth:`~torchref.symmetry.symmetry.Symmetry.reciprocal_extractor`, which caches it.
"""

from typing import TYPE_CHECKING, Optional

import torch

from torchref.config import canonical_device
from torchref.utils.autograd_ops import gather_with_index_add
from torchref.utils.device_mixin import DeviceMixin

if TYPE_CHECKING:
    from torchref.symmetry.symmetry import Symmetry


def _equiv_hkls_to_flat_indices(
    equiv_hkls: torch.Tensor, Nx: int, Ny: int, Nz: int
) -> torch.Tensor:
    """Flatten symmetry-equivalent Miller indices into linear grid indices.

    Parameters
    ----------
    equiv_hkls : torch.Tensor
        Equivalent indices, shape ``(n_ops, N, 3)``.
    Nx, Ny, Nz : int
        Reciprocal grid dimensions.

    Returns
    -------
    torch.Tensor
        Flat indices, shape ``(n_ops * N,)``, dtype ``int64``, wrapped modulo the grid.
    """
    all_hkl = equiv_hkls.reshape(-1, 3)
    hi = torch.remainder(all_hkl[:, 0], Nx)
    ki = torch.remainder(all_hkl[:, 1], Ny)
    li = torch.remainder(all_hkl[:, 2], Nz)
    return (hi * (Ny * Nz) + ki * Nz + li).to(torch.int64)


class ReciprocalSymmetryExtractor(DeviceMixin):
    """Precomputed symmetrized structure-factor extraction at fixed hkl and grid.

    For repeated evaluation during refinement: the equivalent indices, phases and flat
    gather indices are computed once here, so each call is one gather, multiply and
    sum. The precomputation binds both ``hkl`` and ``grid_shape`` -- either changing
    needs a new extractor, which is what
    :meth:`~torchref.symmetry.symmetry.Symmetry.reciprocal_extractor` tracks.

    Parameters
    ----------
    hkl : torch.Tensor
        Target Miller indices, shape ``(N, 3)``.
    symmetry : Symmetry
        The group supplying the operations.
    grid_shape : tuple of int
        Reciprocal grid dimensions ``(Nx, Ny, Nz)``.
    device : torch.device, optional
        Device for the precomputed tensors. Defaults to ``hkl``'s.
    """

    def __init__(
        self,
        hkl: torch.Tensor,
        symmetry: "Symmetry",
        grid_shape: tuple,
        device: Optional[torch.device] = None,
    ):
        # ``is not None``, not ``or``: ``device=0`` means cuda:0/mps:0 and is falsy, so
        # ``or`` silently discarded it.
        self.device = canonical_device(
            device if device is not None else hkl.device
        )
        self.hkl = hkl.to(device=self.device)
        self.symmetry = symmetry
        self.n_ops = symmetry.n_ops
        self.N = len(hkl)
        self.grid_shape = grid_shape

        self.equiv_hkls = symmetry.expand_reciprocal(self.hkl).to(device=self.device)
        self.phases = symmetry.phase_factors(self.hkl).to(device=self.device)

        Nx, Ny, Nz = grid_shape
        self._flat_indices = _equiv_hkls_to_flat_indices(
            self.equiv_hkls, Nx, Ny, Nz
        )

    def __call__(self, density_map: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`extract`."""
        return self.extract(density_map)

    def extract(self, density_map: torch.Tensor) -> torch.Tensor:
        """Transform a P1 density map and extract symmetrized structure factors.

        Parameters
        ----------
        density_map : torch.Tensor
            **P1** electron density, shape ``(Nx, Ny, Nz)``. A symmetrized map
            double-counts.

        Returns
        -------
        torch.Tensor
            Complex structure factors, shape ``(N,)``.
        """
        from torchref.base.fourier.fft import ifft

        return self.extract_from_grid(ifft(density_map))

    def extract_from_grid(self, reciprocal_grid: torch.Tensor) -> torch.Tensor:
        """Extract structure factors from an already-transformed P1 grid.

        Parameters
        ----------
        reciprocal_grid : torch.Tensor
            Complex grid from the FFT of the **P1** map, shape ``(Nx, Ny, Nz)``;
            its shape must match the ``grid_shape`` this extractor was built for.

        Returns
        -------
        torch.Tensor
            Complex structure factors, shape ``(N,)``.
        """
        # gather_with_index_add keeps the backward a single ``index_add_``
        # (atomic scatter, no radix sort + dedup).
        f_all = gather_with_index_add(
            reciprocal_grid.reshape(-1), self._flat_indices
        )  # (n_ops * N,)
        return (f_all.view(self.n_ops, self.N) * self.phases).sum(dim=0)


__all__ = ["ReciprocalSymmetryExtractor"]
