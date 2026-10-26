"""The batch contract shared by every density splat kernel.

A batch is a stack of models that share one crystal -- cell, space group and grid -- and
differ only in their atoms: ensemble members, timepoints, conformers, multistarts. Their
maps are stacked as ``(n_members, nx, ny, nz)``, and member ``b``'s atoms land in slab
``b``, at flat offset ``b * nx * ny * nz``.

Atoms reach the kernels flat, grouped by member, with CSR-style ``batch_offsets``:
member ``b`` owns atoms ``[offsets[b], offsets[b + 1])``. That form lets members carry
different atom counts (an empty member is allowed) without padding. Callers may also pass
the dense form, ``xyz (B, N, 3)``, which :func:`normalize_atoms` flattens.

No batch is the default and changes nothing: ``(N, 3)`` atoms into an ``(nx, ny, nz)``
map is the unbatched path, and every kernel runs it as one member. The kernels decide
which form they were given from the map's rank and ``batch_offsets`` alone, through
:func:`check_batch`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = ["normalize_atoms", "check_batch", "member_index", "dense_offsets"]


def dense_offsets(n_members: int, n_atoms: int, device) -> torch.Tensor:
    """Offsets of ``n_members`` members of ``n_atoms`` atoms each, shape ``(B + 1,)``."""
    # dtype-ok: atom offsets index flat buffers; int64 so no atom count can overflow them
    return torch.arange(n_members + 1, dtype=torch.int64, device=device) * n_atoms


def _check_offsets(offsets: torch.Tensor, n_atoms: int) -> torch.Tensor:
    """Validate ragged ``batch_offsets`` against ``n_atoms`` and return them as int64."""
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError(
            f"batch_offsets must be 1-D with at least two entries (B + 1), got shape "
            f"{tuple(offsets.shape)}"
        )
    if offsets.is_floating_point() or offsets.is_complex():
        raise TypeError(f"batch_offsets must be integer, got {offsets.dtype}")
    # B + 1 values: one host read is cheaper than any device-side check.
    values = offsets.detach().cpu().tolist()
    if values[0] != 0 or values[-1] != n_atoms:
        raise ValueError(
            f"batch_offsets must run from 0 to the atom count {n_atoms}, got "
            f"{values[0]} .. {values[-1]}"
        )
    if any(b < a for a, b in zip(values, values[1:])):
        raise ValueError("batch_offsets must be non-decreasing")
    # dtype-ok: atom offsets index flat buffers; int64 so no atom count can overflow them
    return offsets.to(dtype=torch.int64)


def normalize_atoms(
    xyz: torch.Tensor,
    adp_or_u: torch.Tensor,
    occ: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    batch_offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """Bring one set of atoms into the flat form the kernels take.

    Parameters
    ----------
    xyz : torch.Tensor
        ``(N, 3)`` unbatched or ragged, or ``(B, N, 3)`` dense.
    adp_or_u : torch.Tensor
        ``(N,)`` B-factors or ``(N, 6)`` U components, with a leading ``B`` when dense.
    occ : torch.Tensor
        ``(N,)``, with a leading ``B`` when dense.
    A, B : torch.Tensor
        ITC92 coefficients, ``(N, 5)``. When dense they may also be ``(B, N, 5)``;
        ``(N, 5)`` is then shared by every member, as it is for an ensemble of one
        structure.
    batch_offsets : torch.Tensor, optional
        Ragged form only: ``(B + 1,)`` integer offsets into the flat atoms. Not allowed
        with dense input, whose offsets are implied.

    Returns
    -------
    tuple
        ``(xyz, adp_or_u, occ, A, B, offsets)`` with every atom tensor flat, and
        ``offsets`` the int64 ``(B + 1,)`` offsets, or None when unbatched.
    """
    if xyz.ndim == 2:
        if batch_offsets is None:
            return xyz, adp_or_u, occ, A, B, None
        return xyz, adp_or_u, occ, A, B, _check_offsets(batch_offsets, xyz.shape[0])
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(
            f"xyz must be (N, 3) or (B, N, 3), got shape {tuple(xyz.shape)}"
        )
    if batch_offsets is not None:
        raise ValueError(
            "batch_offsets is for flat (ragged) atoms; dense (B, N, 3) input implies them"
        )
    n_members, n_atoms = int(xyz.shape[0]), int(xyz.shape[1])

    def flat(t: torch.Tensor, name: str, shareable: bool = False) -> torch.Tensor:
        if shareable and t.ndim == 2:
            t = t.unsqueeze(0).expand(n_members, *t.shape)
        if tuple(t.shape[:2]) != (n_members, n_atoms):
            raise ValueError(
                f"{name} must lead with (B, N) = ({n_members}, {n_atoms}) to match xyz, "
                f"got shape {tuple(t.shape)}"
            )
        return t.reshape(n_members * n_atoms, *t.shape[2:])

    return (
        flat(xyz, "xyz"),
        flat(adp_or_u, "adp/u"),
        flat(occ, "occ"),
        flat(A, "A", shareable=True),
        flat(B, "B", shareable=True),
        dense_offsets(n_members, n_atoms, xyz.device),
    )


def check_batch(
    density_map: torch.Tensor,
    n_atoms: int,
    batch_offsets: Optional[torch.Tensor],
) -> Tuple[Tuple[int, int, int], Optional[torch.Tensor]]:
    """Read the grid and the batch off a kernel call, refusing an inconsistent pair.

    Parameters
    ----------
    density_map : torch.Tensor
        ``(nx, ny, nz)`` or ``(B, nx, ny, nz)``.
    n_atoms : int
        Number of flat atoms in the call.
    batch_offsets : torch.Tensor, optional
        ``(B + 1,)`` offsets; required with a 4-D map, and with a 3-D map allowed only
        for one member.

    Returns
    -------
    grid_dims : tuple of int
        ``(nx, ny, nz)``.
    offsets : torch.Tensor or None
        Validated int64 offsets, or None for an unbatched call.
    """
    if density_map.ndim not in (3, 4):
        raise ValueError(
            f"density_map must be (nx, ny, nz) or (B, nx, ny, nz), got shape "
            f"{tuple(density_map.shape)}"
        )
    grid_dims = tuple(int(s) for s in density_map.shape[-3:])
    if batch_offsets is None:
        if density_map.ndim == 4:
            raise ValueError(
                "a batched (B, nx, ny, nz) density map needs batch_offsets to say "
                "which atoms belong to which member"
            )
        return grid_dims, None
    offsets = _check_offsets(batch_offsets, n_atoms)
    n_members = offsets.numel() - 1
    map_members = int(density_map.shape[0]) if density_map.ndim == 4 else 1
    if n_members != map_members:
        raise ValueError(
            f"batch_offsets describe {n_members} members but the density map holds "
            f"{map_members}"
        )
    return grid_dims, offsets


def member_index(offsets: torch.Tensor, n_atoms: int, device=None) -> torch.Tensor:
    """Per-atom member id ``(N,)``, int32, for kernels that work one atom at a time."""
    counts = offsets[1:] - offsets[:-1]
    members = torch.arange(
        # dtype-ok: member ids are kernel indices; int32 is what the GPU kernels read
        counts.numel(), dtype=torch.int32, device=offsets.device
    )
    out = torch.repeat_interleave(members, counts, output_size=n_atoms)
    return out if device is None else out.to(device=device)
