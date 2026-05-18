"""
Lattman-Love (1970) structure-factor interpolation for the alignment module.

Compute F_calc once for the search model in a large cubic P1 box (densely sampled
in reciprocal space), then interpolate the dense grid at rotated reciprocal-space
positions of the *real* crystal cell to obtain F_calc for any candidate rotation.

This is the standard Phaser MR setup (Phaser paper §2.2.2): F_calc is generated
"by structure-factor interpolation (Lattman & Love, 1970) from a model in a large
P1 unit cell". It removes the sphere-sampling bias that arises if one instead
rotates atom coordinates and recomputes F_calc on the (non-uniform) real-cell
HKL grid — which is what the bare ball-search hits on real, non-cubic data.

Convention (matches `torchref/base/reciprocal/interpolation.py::interpolate_for_rotation`):
    For a model whose atom coordinates have been rotated by R (column-vector
    convention: xyz_new = R · xyz_old), the structure factor at real-cell HKL h
    equals the un-rotated model's structure factor at the rotated reciprocal-
    space point R^T · s_real, where s_real = h · rec_basis(real_cell).

Only amplitudes are needed for the rotation function and Sim MLRF rescoring.
Translation search (downstream) needs the phase too — this class returns complex
F so callers can use either.
"""

from __future__ import annotations

from typing import Optional

import torch

from torchref.base.fourier.fft import ifft
from torchref.base.reciprocal.interpolation import (
    interpolate_complex_from_grid,
    interpolate_structure_factor_from_grid,
)
from torchref.model.sf_fft import SfFFT
from torchref.symmetry import SpaceGroup
from torchref.symmetry.cell import Cell


class LattmanLoveInterpolator:
    """
    Compute F_calc on a dense P1 reciprocal grid once; interpolate at arbitrary
    rotated reciprocal positions per query.

    Parameters
    ----------
    model : ModelFT
        Search model. Its current atom coordinates are used and the FT is built
        for those positions (un-rotated; rotations are applied later in
        `evaluate(R, ...)`). The model's own cell/spacegroup are NOT used.
    padding_factor : float, default 2.0
        Cubic P1 box side = padding_factor * molecule_bounding_box_diameter.
        Phaser uses ~2.0. Larger → finer reciprocal grid spacing, more memory.
    min_cell_size_A : float, optional
        Lower bound on the cubic side (Å). Useful for very small molecules where
        2·diameter would give a tiny FFT grid.
    max_res_A : float, optional
        Resolution limit (Å). The dense grid will resolve features down to this.
        Default: 2.0 Å (suitable for proteins up to that resolution).
    radius_angstrom : float, default 3.0
        Atomic-density support radius for SfFFT.
    device : torch.device, optional
        Target device. Defaults to the model's device.
    """

    def __init__(
        self,
        model,
        padding_factor: float = 2.0,
        min_cell_size_A: Optional[float] = None,
        max_res_A: float = 2.0,
        radius_angstrom: float = 3.0,
        device: Optional[torch.device] = None,
        verbose: int = 0,
    ):
        if device is None:
            device = model.xyz().device

        # Bounding-box diameter of the un-rotated atomic coordinates.
        xyz = model.xyz().detach().to(device)
        bbox_max = xyz.max(dim=0).values
        bbox_min = xyz.min(dim=0).values
        diameter_A = (bbox_max - bbox_min).norm().item()
        cubic_side = padding_factor * diameter_A
        if min_cell_size_A is not None:
            cubic_side = max(cubic_side, float(min_cell_size_A))

        # Cubic P1 cell. Keep dtype float32 to match Cell defaults / SfFFT grid math.
        self.cubic_cell = Cell(
            [cubic_side, cubic_side, cubic_side, 90.0, 90.0, 90.0],
            dtype=torch.float32, device=device,
        )

        # Shift atoms so the molecule centroid sits at the centre of the cubic box.
        centroid = xyz.mean(dim=0)
        target = torch.tensor(
            [cubic_side / 2, cubic_side / 2, cubic_side / 2],
            dtype=xyz.dtype, device=device,
        )
        self.shift_vec = target - centroid  # (3,) — applied to atom coords

        # Extract atomic parameters (use the model's own helper).
        xyz_iso, adp_iso, occ_iso, A_iso, B_iso = model.get_iso()
        xyz_iso = (xyz_iso.detach().to(device) + self.shift_vec).to(xyz.dtype)
        adp_iso = adp_iso.detach().to(device)
        occ_iso = occ_iso.detach().to(device)
        A_iso = A_iso.detach().to(device)
        B_iso = B_iso.detach().to(device)

        # Handle the (uncommon) anisotropic atoms by leaving them aside for the
        # search model — Phaser-style MR also approximates with isotropic ADP.
        # Caller is free to call evaluate after adding aniso atoms in a subclass.

        # Build the dense F_calc on a P1 cubic grid.
        sf = SfFFT(
            cell=self.cubic_cell,
            spacegroup=SpaceGroup("P 1"),
            max_res=max_res_A,
            radius_angstrom=radius_angstrom,
            dtype_float=torch.float32,
            device=device,
            verbose=verbose,
        )
        sf.setup_grid()
        density_map = sf.build_density_map(
            xyz_iso=xyz_iso,
            adp_iso=adp_iso,
            occ_iso=occ_iso,
            A_iso=A_iso,
            B_iso=B_iso,
            apply_symmetry=False,  # already P1
        )
        # IFFT to reciprocal space; gives a complex (Nx, Ny, Nz) tensor with
        # crystallographic normalization. Layout: DC at index (0, 0, 0); negative
        # HKL wraps to high indices. This matches the convention expected by
        # `interpolate_structure_factor_from_grid`.
        self.reciprocal_grid = ifft(density_map, self.cubic_cell.volume.item())
        self.cubic_cell_volume = float(self.cubic_cell.volume.item())
        self.device = device
        self.cubic_side = cubic_side
        self.gridsize = tuple(int(x) for x in self.reciprocal_grid.shape)
        self.max_res_A = max_res_A

        if verbose:
            print(f"LattmanLove: molecule diameter {diameter_A:.1f} Å, "
                  f"cubic side {cubic_side:.1f} Å, grid {self.gridsize}, "
                  f"max_res {max_res_A:.2f} Å")

    @staticmethod
    def _real_hkl_to_cubic_hkl(
        hkl_real: torch.Tensor, real_cell: Cell, cubic_cell: Cell,
    ) -> torch.Tensor:
        """Convert HKL of `real_cell` to (float) HKL of `cubic_cell`."""
        # s = h @ rec_basis  (Å^-1, Cartesian)
        rec_real = real_cell.reciprocal_basis_matrix.to(hkl_real.device)
        s = hkl_real.to(rec_real.dtype) @ rec_real
        rec_cubic = cubic_cell.reciprocal_basis_matrix.to(hkl_real.device)
        # cubic_hkl = s @ rec_cubic^{-1}
        return s @ torch.linalg.inv(rec_cubic.to(rec_real.dtype))

    def evaluate(
        self,
        R: torch.Tensor,
        hkl_real: torch.Tensor,
        real_cell: Cell,
        return_amplitude: bool = True,
    ) -> torch.Tensor:
        """
        Interpolate F_calc at the real-cell HKL set after rotating the model
        by R (column-vector convention).

        Parameters
        ----------
        R : torch.Tensor
            Rotation matrix, shape (3, 3) or (B, 3, 3).
        hkl_real : torch.Tensor
            Miller indices in the real crystal cell, shape (N, 3).
        real_cell : Cell
            The real crystal cell (provides `reciprocal_basis_matrix`).
        return_amplitude : bool, default True
            If True, return |F_calc| (real, no phase ambiguity from trilinear
            interpolation). If False, return complex F — only safe if the dense
            grid is well-oversampled (small `max_res_A`).

        Returns
        -------
        torch.Tensor
            Interpolated structure factors, shape (N,) or (B, N).
        """
        batched = R.dim() == 3
        if not batched:
            R = R.unsqueeze(0)
        R = R.to(self.device).to(torch.float32)

        # Real HKL → Cartesian s (Å^-1, real cell)
        rec_real = real_cell.reciprocal_basis_matrix.to(self.device).to(torch.float32)
        s_real = hkl_real.to(self.device).to(torch.float32) @ rec_real  # (N, 3)

        # Rotated reciprocal point: F_rotated_model(s) = F_orig(R^T s)  =>
        # use R^T · s to look up the un-rotated grid.
        # For batched R, einsum:
        s_rot = torch.einsum("bij,nj->bni", R.transpose(-1, -2), s_real)  # (B, N, 3)

        # Cartesian s → cubic-cell float HKL
        rec_cubic = self.cubic_cell.reciprocal_basis_matrix.to(self.device).to(torch.float32)
        inv_rec_cubic = torch.linalg.inv(rec_cubic)
        hkl_cubic = s_rot @ inv_rec_cubic  # (B, N, 3)

        B, N, _ = hkl_cubic.shape
        flat = hkl_cubic.reshape(B * N, 3)
        if return_amplitude:
            interp = interpolate_structure_factor_from_grid(
                self.reciprocal_grid, flat, interpolate_amplitude=True,
            )
        else:
            interp = interpolate_complex_from_grid(self.reciprocal_grid, flat)
        out = interp.reshape(B, N)
        return out if batched else out.squeeze(0)
