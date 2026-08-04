"""Reciprocal space symmetry operations for structure factor grids.

The reciprocal-space counterpart to ``map_symmetry.py``: :func:`ReciprocalSymmetry`
(grid operator), :func:`expand_hkl` / :func:`expand_reflections` /
:func:`expand_reciprocal_grid` (ASU -> P1), :func:`reduce_hkl` (P1 -> ASU),
:func:`complete_hkl` (find reflections missing from a dataset, same space group)
and :func:`canonicalize_hkl` (CCP4 ASU representative). Space groups accept
strings, ints 1-230 or ``gemmi.SpaceGroup``.

Miller indices transform as h' = h @ R = R^T @ h with R the *real-space* rotation;
``reciprocal_matrices`` already holds the transpose, so do not transpose again.
Translations become phase shifts of -2π h·t; the sign is load-bearing and wrong
signs are invisible in P21/P212121/C2 (see :func:`expand_hkl`).
"""

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from torchref.config import get_float_dtype, normalize_device
from torchref.symmetry.spacegroup import SpaceGroup, SpaceGroupLike
from torchref.utils.device_mixin import DeviceMixin

if TYPE_CHECKING:
    from torchref.io.datasets.reflection_data import ReflectionData


def ReciprocalSymmetry(
    space_group: SpaceGroupLike,
    grid_shape,
    dtype_float=None,
    verbose=1,
    device=None,
):
    """
    Factory function to create the appropriate ReciprocalSymmetry implementation.

    Parameters
    ----------
    space_group : str, int, or gemmi.SpaceGroup
        Space group specification (e.g., 'P21', 4, gemmi.SpaceGroup('P 21')).
    grid_shape : tuple of int
        Shape of the reciprocal space grid (nh, nk, nl).
        The grid spans from -n//2 to n//2 for each dimension.
    dtype_float : torch.dtype, default: configured dtypes.float
        Floating point precision to use.
    verbose : int, default 1
        Verbosity level (0=silent, 1=info, 2=debug).
    device : torch.device, default: configured device.current
        Device to use for computation.

    Returns
    -------
    ReciprocalSymmetryGrid
        Implementation for reciprocal space grid symmetry operations.
    """
    return ReciprocalSymmetryGrid(space_group, grid_shape, dtype_float, verbose, device)


class ReciprocalSymmetryGrid(DeviceMixin, nn.Module):
    """
    Reciprocal space symmetry operations for Miller index grids.

    Covers Miller-index transformation, systematic absences, centric reflections,
    Friedel pairs and symmetry expansion/averaging of structure factors. The
    per-operation index maps, phase shifts, absence mask and centric mask are all
    precomputed in ``__init__`` for the fixed ``grid_shape``.

    Attributes
    ----------
    space_group : str
        Space group name.
    grid_shape : tuple of int
        Shape of the reciprocal space grid (nh, nk, nl).
    symmetry : SpaceGroup
        Base symmetry operations handler.
    n_ops : int
        Number of symmetry operations.

    Examples
    --------
    ::

        recip_sym = ReciprocalSymmetry('P21', grid_shape=(64, 64, 64))
        F_expanded = recip_sym(F_asym)  # Expand from asymmetric unit
        F_avg = recip_sym.symmetry_average(F_full)  # Average symmetry-related reflections
    """

    def __init__(
        self,
        space_group,
        grid_shape,
        dtype_float=None,
        verbose=1,
        device=None,
    ):
        """
        Initialize reciprocal space symmetry operator.

        Parameters
        ----------
        space_group : str
            Space group name.
        grid_shape : tuple of int
            Shape of the reciprocal space grid (nh, nk, nl).
        dtype_float : torch.dtype, default: configured dtypes.float
            Floating point precision.
        verbose : int, default 1
            Verbosity level.
        device : torch.device, default: configured device.current
            Computation device.
        """
        super().__init__()
        if dtype_float is None:
            dtype_float = get_float_dtype()
        device = normalize_device(device)
        self.dtype_float = dtype_float
        self.space_group = space_group
        self.grid_shape = tuple(grid_shape)
        self.verbose = verbose
        self.device = device

        self.symmetry = SpaceGroup(space_group, dtype=dtype_float, device=device)
        self.n_ops = self.symmetry.matrices.shape[0]

        self._setup_reciprocal_matrices()

        if self.verbose > 0:
            print(f"ReciprocalSymmetryGrid initialized for {space_group}")
            print(f"  Number of symmetry operations: {self.n_ops}")
            print(f"  Grid shape: {self.grid_shape}")

        self._setup_hkl_grid()
        self._setup_symmetry_index_grids()
        self._setup_systematic_absences()
        self._setup_centric_reflections()

        if self.verbose > 0:
            n_absent = self.systematic_absences.sum().item()
            n_centric = self.centric_mask.sum().item()
            n_total = np.prod(self.grid_shape)
            print(f"  Systematic absences: {n_absent} ({100*n_absent/n_total:.2f}%)")
            print(f"  Centric reflections: {n_centric} ({100*n_centric/n_total:.2f}%)")

    def _setup_reciprocal_matrices(self):
        """Cache ``reciprocal_matrices`` = R^T, for h' = R^T @ h."""
        # Translations cause phase shifts, not index changes, so they are not folded in.
        recip_matrices = self.symmetry.matrices.transpose(-2, -1).contiguous()
        self.register_buffer("reciprocal_matrices", recip_matrices)

    def _setup_hkl_grid(self):
        """Build ``hkl_grid`` in FFT index order (0...n//2, -n//2+1...-1)."""
        nh, nk, nl = self.grid_shape

        h = torch.fft.fftfreq(nh, d=1.0) * nh  # gives 0,1,2,...,n//2,-n//2+1,...,-1
        k = torch.fft.fftfreq(nk, d=1.0) * nk
        l = torch.fft.fftfreq(nl, d=1.0) * nl

        h = h.to(dtype=torch.int64, device=self.device)
        k = k.to(dtype=torch.int64, device=self.device)
        l = l.to(dtype=torch.int64, device=self.device)

        grid_h, grid_k, grid_l = torch.meshgrid(h, k, l, indexing="ij")
        hkl_grid = torch.stack([grid_h, grid_k, grid_l], dim=-1)

        self.register_buffer("hkl_grid", hkl_grid)

        # Float copy for the matmuls; the int copy stays authoritative for indexing.
        hkl_grid_float = hkl_grid.to(dtype=self.dtype_float)
        self.register_buffer("hkl_grid_float", hkl_grid_float)

    def _setup_symmetry_index_grids(self):
        """Precompute ``index_grids`` (n_ops, nh, nk, nl, 3) and ``phase_shifts``.

        The grid path uses the +2π h·t convention, unlike :func:`expand_hkl`.
        """
        nh, nk, nl = self.grid_shape
        hkl_flat = self.hkl_grid_float.reshape(-1, 3)  # (N, 3)

        index_grids_list = []
        phase_shift_grids_list = []

        for i in range(self.n_ops):
            transformed = torch.matmul(hkl_flat, self.reciprocal_matrices[i].T)
            # Exact for valid symmetry ops; round only mops up float error.
            transformed_int = torch.round(transformed).to(torch.int64)

            # F(h') = F(h) * exp(2πi h·t)
            translation = self.symmetry.translations[i]
            phase_shift = 2.0 * np.pi * torch.matmul(hkl_flat, translation)
            phase_shift = phase_shift.reshape(nh, nk, nl)
            phase_shift_grids_list.append(phase_shift)

            # Wrap with periodic boundary to get grid indices.
            idx_h = transformed_int[:, 0] % nh
            idx_k = transformed_int[:, 1] % nk
            idx_l = transformed_int[:, 2] % nl

            index_grid = torch.stack([idx_h, idx_k, idx_l], dim=-1)
            index_grid = index_grid.reshape(nh, nk, nl, 3)
            index_grids_list.append(index_grid)

        index_grids = torch.stack(index_grids_list, dim=0)
        self.register_buffer("index_grids", index_grids)

        phase_shifts = torch.stack(phase_shift_grids_list, dim=0)
        self.register_buffer("phase_shifts", phase_shifts)

    def _setup_systematic_absences(self):
        """Mask reflections with some op mapping h -> h at h·t not integral.

        Such a reflection is destroyed by interference from the translation.
        """
        nh, nk, nl = self.grid_shape
        absences = torch.zeros(self.grid_shape, dtype=torch.bool, device=self.device)

        hkl_flat = self.hkl_grid_float.reshape(-1, 3)

        for i in range(self.n_ops):
            transformed = torch.matmul(hkl_flat, self.reciprocal_matrices[i].T)
            transformed_int = torch.round(transformed).to(torch.int64)

            hkl_int = self.hkl_grid.reshape(-1, 3)
            same_reflection = (transformed_int == hkl_int).all(dim=-1)

            translation = self.symmetry.translations[i]
            h_dot_t = torch.matmul(hkl_flat, translation)

            phase_mod = torch.abs(h_dot_t - torch.round(h_dot_t))
            non_zero_phase = phase_mod > 1e-6

            absent_mask = (same_reflection & non_zero_phase).reshape(self.grid_shape)
            absences = absences | absent_mask

        self.register_buffer("systematic_absences", absences)

    def _setup_centric_reflections(self):
        """Mask reflections with some op mapping h -> -h (phase restricted to 0/π)."""
        nh, nk, nl = self.grid_shape
        centric = torch.zeros(self.grid_shape, dtype=torch.bool, device=self.device)

        hkl_flat = self.hkl_grid_float.reshape(-1, 3)

        for i in range(self.n_ops):
            transformed = torch.matmul(hkl_flat, self.reciprocal_matrices[i].T)
            transformed_int = torch.round(transformed).to(torch.int64)

            hkl_int = self.hkl_grid.reshape(-1, 3)
            maps_to_minus_h = (transformed_int == -hkl_int).all(dim=-1)

            centric_mask = maps_to_minus_h.reshape(self.grid_shape)
            centric = centric | centric_mask

        self.register_buffer("centric_mask", centric)

    def apply_to_indices(self, hkl, operation_index=None):
        """
        Apply symmetry operation(s) to Miller indices.

        Parameters
        ----------
        hkl : torch.Tensor, shape (..., 3)
            Miller indices (h, k, l).
        operation_index : int, optional
            If specified, apply only this operation.
            If None, apply all operations.

        Returns
        -------
        torch.Tensor
            Transformed Miller indices.
            If operation_index is None: shape (n_ops, ..., 3)
            Otherwise: shape (..., 3)
        """
        hkl = hkl.to(dtype=self.dtype_float, device=self.device)
        original_shape = hkl.shape[:-1]

        if operation_index is not None:
            # Apply single operation
            R = self.reciprocal_matrices[operation_index]
            transformed = torch.matmul(hkl, R.T)
            return torch.round(transformed).to(torch.int64)
        else:
            # Apply all operations
            hkl_flat = hkl.reshape(-1, 3)  # (N, 3)
            results = []
            for i in range(self.n_ops):
                R = self.reciprocal_matrices[i]
                transformed = torch.matmul(hkl_flat, R.T)
                results.append(torch.round(transformed).to(torch.int64))

            # Stack: (n_ops, N, 3)
            stacked = torch.stack(results, dim=0)
            return stacked.reshape(self.n_ops, *original_shape, 3)

    def get_phase_shift(self, hkl, operation_index):
        """
        Get phase shift for a symmetry operation on given Miller indices.

        The phase shift is exp(2πi h·t) where t is the translation.

        Parameters
        ----------
        hkl : torch.Tensor, shape (..., 3)
            Miller indices.
        operation_index : int
            Symmetry operation index.

        Returns
        -------
        torch.Tensor
            Phase shift in radians, shape (...).
        """
        hkl = hkl.to(dtype=self.dtype_float, device=self.device)
        translation = self.symmetry.translations[operation_index]
        phase = 2.0 * np.pi * torch.matmul(hkl, translation)
        return phase

    def get_symmetry_mate(self, F_grid, operation_index):
        """
        Apply a single symmetry operation to a structure factor grid.

        Parameters
        ----------
        F_grid : torch.Tensor, shape (nh, nk, nl)
            Complex structure factor grid.
        operation_index : int
            Index of the symmetry operation (0 to n_ops-1).

        Returns
        -------
        torch.Tensor, shape (nh, nk, nl)
            Structure factors after applying symmetry operation.
            Includes phase shift from translation component.
        """
        if operation_index < 0 or operation_index >= self.n_ops:
            raise ValueError(
                f"Operation index {operation_index} out of range [0, {self.n_ops-1}]"
            )

        if F_grid.shape != self.grid_shape:
            raise ValueError(
                f"Grid shape {F_grid.shape} doesn't match expected {self.grid_shape}"
            )

        # Get precomputed index grid for this operation
        idx_grid = self.index_grids[operation_index]  # (nh, nk, nl, 3)

        # Gather structure factors from transformed positions
        F_transformed = F_grid[idx_grid[..., 0], idx_grid[..., 1], idx_grid[..., 2]]

        # Apply phase shift from translation: F(h') = F(h) * exp(2πi h·t)
        phase = self.phase_shifts[operation_index]
        if F_grid.is_complex():
            phase_factor = torch.exp(1j * phase.to(F_grid.dtype))
            F_transformed = F_transformed * phase_factor
        # For real-valued grids (amplitudes), no phase shift needed

        return F_transformed

    def get_all_symmetry_mates(self, F_grid):
        """
        Get all symmetry-related structure factor grids.

        Parameters
        ----------
        F_grid : torch.Tensor, shape (nh, nk, nl)
            Complex structure factor grid.

        Returns
        -------
        list of torch.Tensor
            List of symmetry-related grids.
        """
        return [self.get_symmetry_mate(F_grid, i) for i in range(self.n_ops)]

    def symmetry_average(self, F_grid, weights=None):
        """
        Average structure factors over all symmetry equivalents.

        This is useful for enforcing symmetry constraints on structure factors.

        Parameters
        ----------
        F_grid : torch.Tensor, shape (nh, nk, nl)
            Complex structure factor grid.
        weights : torch.Tensor, optional
            Weights for averaging, shape (nh, nk, nl).
            If None, equal weights are used.

        Returns
        -------
        torch.Tensor, shape (nh, nk, nl)
            Symmetry-averaged structure factors.
        """
        mates = self.get_all_symmetry_mates(F_grid)
        stacked = torch.stack(mates, dim=0)  # (n_ops, nh, nk, nl)

        if weights is not None:
            weights = weights.unsqueeze(0)  # (1, nh, nk, nl)
            stacked = stacked * weights
            return stacked.sum(dim=0) / (weights.sum() * self.n_ops)
        else:
            return stacked.mean(dim=0)

    def expand_to_p1(self, F_asym, asym_mask=None):
        """
        Expand structure factors from asymmetric unit to full P1.

        Takes structure factors defined on the asymmetric unit and
        generates the full reciprocal space by applying all symmetry
        operations.

        Parameters
        ----------
        F_asym : torch.Tensor, shape (nh, nk, nl)
            Structure factors on asymmetric unit (other positions can be zero).
        asym_mask : torch.Tensor, optional
            Currently a no-op: this parameter is accepted but ignored by the
            implementation. Regardless of its value, positions are filled using
            a non-zero heuristic (entries with ``|F| < 1e-10`` are treated as
            unset and filled from symmetry mates).

        Returns
        -------
        torch.Tensor, shape (nh, nk, nl)
            Full structure factor grid with all symmetry equivalents filled.
        """
        F_full = F_asym.clone()

        for i in range(1, self.n_ops):  # Skip identity
            F_mate = self.get_symmetry_mate(F_asym, i)

            # Only fill in positions that are zero (not yet set)
            if F_full.is_complex():
                mask = F_full.abs() < 1e-10
            else:
                mask = F_full.abs() < 1e-10

            F_full = torch.where(mask, F_mate, F_full)

        return F_full

    def apply_friedel(self, F_grid):
        """
        Apply Friedel's law: F(-h,-k,-l) = F*(h,k,l).

        For normal (non-anomalous) scattering, the structure factor
        at -h is the complex conjugate of F(h).

        Parameters
        ----------
        F_grid : torch.Tensor, shape (nh, nk, nl)
            Complex structure factor grid.

        Returns
        -------
        torch.Tensor, shape (nh, nk, nl)
            Structure factors averaged toward Friedel symmetry. The result is
            ``0.5 * (F(h) + F*(-h))``, i.e. an average of each reflection with
            its conjugated Friedel mate, not a hard replacement/enforcement.
        """
        # Flip all indices: F(-h,-k,-l)
        F_friedel = torch.flip(F_grid, dims=[0, 1, 2])

        # Roll to handle the asymmetry at 0 index
        nh, nk, nl = self.grid_shape
        F_friedel = torch.roll(F_friedel, shifts=(1, 1, 1), dims=(0, 1, 2))

        if F_grid.is_complex():
            F_friedel = F_friedel.conj()

        # Average F(h) and F*(-h)
        return 0.5 * (F_grid + F_friedel)

    def is_systematic_absence(self, h, k, l):
        """
        Check if a reflection is systematically absent.

        Parameters
        ----------
        h, k, l : int
            Miller indices.

        Returns
        -------
        bool
            True if the reflection is systematically absent.
        """
        nh, nk, nl = self.grid_shape
        idx_h = h % nh
        idx_k = k % nk
        idx_l = l % nl
        return self.systematic_absences[idx_h, idx_k, idx_l].item()

    def is_centric(self, h, k, l):
        """
        Check if a reflection is centric.

        Parameters
        ----------
        h, k, l : int
            Miller indices.

        Returns
        -------
        bool
            True if the reflection is centric (phase restricted to 0 or π).
        """
        nh, nk, nl = self.grid_shape
        idx_h = h % nh
        idx_k = k % nk
        idx_l = l % nl
        return self.centric_mask[idx_h, idx_k, idx_l].item()

    def get_epsilon(self):
        """
        Compute epsilon (multiplicity) factors for each reflection.

        Epsilon is the number of symmetry operations that map h to itself
        (h -> h) or to its Friedel mate (h -> -h). This count is taken
        unconditionally for every space group; there is no centric/acentric
        branch. Note that folding in Friedel mates inflates the count relative
        to the conventional ε (pure rotational multiplicity, h -> h only).

        Returns
        -------
        torch.Tensor, shape (nh, nk, nl)
            Epsilon factors for each reflection.
        """
        epsilon = torch.zeros(self.grid_shape, dtype=torch.int32, device=self.device)

        hkl_flat = self.hkl_grid.reshape(-1, 3)

        for i in range(self.n_ops):
            # Get transformed indices
            idx_grid = self.index_grids[i]
            transformed_flat = idx_grid.reshape(-1, 3)

            # Check if h' ≡ h or h' ≡ -h (same reflection or Friedel)
            same = (transformed_flat == hkl_flat).all(dim=-1)

            nh, nk, nl = self.grid_shape
            # Also check Friedel (-h, -k, -l)
            hkl_neg = (-self.hkl_grid).reshape(-1, 3)
            hkl_neg[:, 0] = hkl_neg[:, 0] % nh
            hkl_neg[:, 1] = hkl_neg[:, 1] % nk
            hkl_neg[:, 2] = hkl_neg[:, 2] % nl

            friedel = (transformed_flat == hkl_neg).all(dim=-1)

            contributes = (same | friedel).reshape(self.grid_shape)
            epsilon += contributes.to(torch.int32)

        return epsilon

    def forward(self, F_grid, mode="average"):
        """
        Apply symmetry to structure factor grid.

        Parameters
        ----------
        F_grid : torch.Tensor, shape (nh, nk, nl)
            Complex structure factor grid.
        mode : str, default 'average'
            Operation mode:
            - 'average': Average over all symmetry equivalents
            - 'expand': Expand from asymmetric unit to full grid
            - 'sum': Sum all symmetry mates (for accumulation)

        Returns
        -------
        torch.Tensor, shape (nh, nk, nl)
            Processed structure factor grid.
        """
        if mode == "average":
            return self.symmetry_average(F_grid)
        elif mode == "expand":
            return self.expand_to_p1(F_grid)
        elif mode == "sum":
            mates = self.get_all_symmetry_mates(F_grid)
            return torch.stack(mates, dim=0).sum(dim=0)
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'average', 'expand', or 'sum'."
            )

    def __call__(self, F_grid, mode="average"):
        """Make the class callable."""
        return self.forward(F_grid, mode=mode)

    def get_symmetry_info(self):
        """
        Get information about reciprocal space symmetry.

        Returns
        -------
        dict
            Dictionary with symmetry information.
        """
        return {
            "space_group": self.space_group,
            "n_operations": self.n_ops,
            "reciprocal_matrices": self.reciprocal_matrices,
            "translations": self.symmetry.translations,
            "n_systematic_absences": self.systematic_absences.sum().item(),
            "n_centric": self.centric_mask.sum().item(),
            "grid_shape": self.grid_shape,
        }

    def __repr__(self):
        return (
            f"ReciprocalSymmetryGrid(space_group='{self.space_group}', "
            f"n_ops={self.n_ops}, grid_shape={self.grid_shape})"
        )


# =============================================================================
# Standalone functions for symmetry expansion
# =============================================================================


def expand_hkl(
    hkl: torch.Tensor,
    spacegroup: SpaceGroupLike,
    include_friedel: bool = True,
    remove_absences: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand Miller indices under crystallographic symmetry (ASU -> P1).

    The low-level primitive: returns the expanded indices plus the index map and
    phase offsets needed to expand any associated per-reflection data.

    Parameters
    ----------
    hkl : torch.Tensor, shape (N, 3)
        Input Miller indices (asymmetric unit).
    spacegroup : str, int, or gemmi.SpaceGroup
        Space group specification.
    include_friedel : bool, default True
        Include Friedel mates (-h, -k, -l).
    remove_absences : bool, default True
        Remove systematically absent reflections.
    device : torch.device, optional
        Computation device. If None, uses hkl's device.

    Returns
    -------
    expanded_hkl : torch.Tensor, shape (M, 3), dtype=int32
        All unique expanded Miller indices.
    orig_indices : torch.Tensor, shape (M,), dtype=int64
        Index mapping expanded → original: ``F_expanded = F_orig[orig_indices]``.
    phase_shifts : torch.Tensor, shape (M,), dtype=float32
        Translation phase offsets in radians:
        ``phase_expanded = phase_orig[orig_indices] + phase_shifts``.
    """
    if device is None:
        device = hkl.device

    # Get symmetry operations
    symmetry = SpaceGroup(spacegroup, dtype=get_float_dtype(), device=device)
    n_ops = symmetry.matrices.shape[0]

    # Reciprocal space matrices (transpose of real space)
    recip_matrices = symmetry.matrices.transpose(-2, -1)
    translations = symmetry.translations

    # Convert hkl to float for matrix operations
    hkl_float = hkl.to(dtype=get_float_dtype(), device=device)
    n_orig = len(hkl_float)

    # Apply all symmetry operations
    all_hkl = []
    all_phases = []

    for i in range(n_ops):
        # h' = h @ R^T
        hkl_transformed = torch.round(torch.matmul(hkl_float, recip_matrices[i].T)).to(
            torch.int32
        )
        # Phase shift from translation: -2π h·t, for h' = hR under the convention
        # F(h) = Σ_j f_j exp(+2πi h·x_j). Do NOT "simplify" the sign: the wrong sign
        # costs 4π h·t mod 2π, which is exactly zero for 2₁ screws and centring, so
        # P21/P212121/C2 cannot see it. tests/unit/symmetry/test_phase_convention.py.
        phase_shift = -2.0 * np.pi * torch.matmul(hkl_float, translations[i])

        all_hkl.append(hkl_transformed)
        all_phases.append(phase_shift)

    # Add Friedel mates if requested
    if include_friedel:
        for i in range(n_ops):
            all_hkl.append(-all_hkl[i])
            all_phases.append(-all_phases[i])

    # Stack all transformed hkl and phases
    hkl_expanded = torch.cat(all_hkl, dim=0)
    phases_expanded = torch.cat(all_phases, dim=0)

    # Remove duplicates - keep unique (h,k,l) tuples with index mapping
    hkl_np = hkl_expanded.cpu().numpy()
    phase_np = phases_expanded.cpu().numpy()

    # Build dictionary: key=(h,k,l), value=(first_occurrence_idx, phase)
    unique_dict = {}
    for idx, (h, phase) in enumerate(zip(hkl_np, phase_np)):
        key = tuple(h)
        if key not in unique_dict:
            unique_dict[key] = (idx, phase)

    # Extract unique data
    unique_indices = [v[0] for v in unique_dict.values()]
    unique_phases = [v[1] for v in unique_dict.values()]

    # Map back to original reflection index
    n_total_ops = n_ops * (2 if include_friedel else 1)
    orig_indices = [idx % n_orig for idx in unique_indices]

    # Build output tensors
    expanded_hkl = torch.tensor(
        [list(k) for k in unique_dict.keys()], dtype=torch.int32, device=device
    )
    phase_shifts = torch.tensor(unique_phases, dtype=get_float_dtype(), device=device)
    orig_idx_tensor = torch.tensor(orig_indices, dtype=torch.int64, device=device)

    # Remove systematic absences if requested
    sg = SpaceGroup(spacegroup)
    is_p1 = sg.number == 1

    if remove_absences and not is_p1:
        absence_mask = _check_systematic_absences(
            expanded_hkl, symmetry.matrices, translations, device
        )
        keep_mask = ~absence_mask

        expanded_hkl = expanded_hkl[keep_mask]
        phase_shifts = phase_shifts[keep_mask]
        orig_idx_tensor = orig_idx_tensor[keep_mask]

    return expanded_hkl, orig_idx_tensor, phase_shifts


def complete_hkl(
    input_hkl: torch.Tensor,
    cell: torch.Tensor,
    spacegroup: SpaceGroupLike,
    d_min: float,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Complete a set of Miller indices by identifying missing reflections.

    Generates every reflection within ``d_min`` for ``spacegroup`` (minus
    systematic absences), then maps the input onto that complete set. This does
    *not* expand symmetry -- the output stays in the input space group.

    Parameters
    ----------
    input_hkl : torch.Tensor, shape (N, 3)
        Input Miller indices (may be incomplete).
    cell : torch.Tensor, shape (6,)
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    spacegroup : str, int, or gemmi.SpaceGroup
        Space group specification.
    d_min : float
        High resolution limit in Angstroms.
    device : torch.device, optional
        Computation device. If None, uses input_hkl's device.

    Returns
    -------
    complete_hkl : torch.Tensor, shape (M, 3), dtype int32
        All possible Miller indices within resolution (minus systematic absences).
    input_indices : torch.Tensor, shape (M,), dtype int64
        Index mapping complete → input, or -1 where missing. Use as
        ``F_complete[~missing] = F_input[input_indices[~missing]]``.
    missing_mask : torch.Tensor, shape (M,), dtype bool
        True where reflection is missing from input.
    """
    from torchref.base.reciprocal import generate_possible_hkl

    if device is None:
        device = input_hkl.device

    # Generate all possible HKL within resolution
    all_hkl = generate_possible_hkl(cell, d_min, device=device)

    # Get symmetry operations for absence check
    symmetry = SpaceGroup(spacegroup, dtype=get_float_dtype(), device=device)
    translations = symmetry.translations

    # Remove systematic absences
    sg = SpaceGroup(spacegroup)
    is_p1 = sg.number == 1

    if not is_p1:
        absence_mask = _check_systematic_absences(
            all_hkl, symmetry.matrices, translations, device
        )
        all_hkl = all_hkl[~absence_mask]

    # Build lookup dictionary from input hkl to indices
    input_hkl_np = input_hkl.cpu().numpy()
    input_lookup = {}
    for idx, hkl in enumerate(input_hkl_np):
        key = tuple(hkl)
        input_lookup[key] = idx

    # Match complete set to input
    all_hkl_np = all_hkl.cpu().numpy()
    n_complete = len(all_hkl)

    input_indices = torch.full((n_complete,), -1, dtype=torch.int64, device=device)
    missing_mask = torch.ones(n_complete, dtype=torch.bool, device=device)

    for i, hkl in enumerate(all_hkl_np):
        key = tuple(hkl)
        if key in input_lookup:
            input_indices[i] = input_lookup[key]
            missing_mask[i] = False

    return all_hkl, input_indices, missing_mask


def expand_reflections(
    reflection_data: "ReflectionData",
    include_friedel: bool = True,
    remove_absences: bool = True,
    verbose: int = 1,
) -> "ReflectionData":
    """Expand ReflectionData from the asymmetric unit to P1.

    A :func:`expand_hkl` wrapper that carries every ReflectionData field (F,
    sigmas, I, phases, FOM, R-free flags) through the expansion. Phases receive
    the translation shift; resolution is recomputed and ``bin_indices`` cleared,
    since expansion invalidates them.

    Parameters
    ----------
    reflection_data : ReflectionData
        Input reflection data with hkl, F, F_sigma, etc.
    include_friedel : bool, default True
        If True, also include Friedel mates (-h, -k, -l).
    remove_absences : bool, default True
        If True, remove systematically absent reflections from output.
    verbose : int, default 1
        Verbosity level.

    Returns
    -------
    ReflectionData
        New object holding the expanded reflections, spacegroup set to 'P1'.

    See Also
    --------
    expand_hkl : Low-level function for HKL expansion without ReflectionData.
    """
    from torchref.io.datasets.reflection_data import ReflectionData as RefData

    if reflection_data.hkl is None:
        raise ValueError("ReflectionData has no Miller indices loaded")

    space_group = reflection_data.spacegroup or "P1"
    device = reflection_data.device
    n_orig = len(reflection_data.hkl)

    if verbose > 0:
        symmetry = SpaceGroup(space_group, dtype=get_float_dtype(), device=device)
        print(f"Expanding reflections for {space_group}")
        print(f"  Original reflections: {n_orig}")
        print(f"  Symmetry operations: {symmetry.n_ops}")

    hkl_expanded, orig_idx_tensor, phase_shifts = expand_hkl(
        reflection_data.hkl,
        space_group,
        include_friedel=include_friedel,
        remove_absences=remove_absences,
        device=device,
    )

    if verbose > 0:
        print(f"  After expansion: {len(hkl_expanded)} unique reflections")

    expanded = RefData(verbose=reflection_data.verbose, device=device)

    expanded.hkl = hkl_expanded
    expanded.cell = (
        reflection_data.cell.clone() if reflection_data.cell is not None else None
    )
    expanded.spacegroup = SpaceGroup("P1")  # Now in P1 since symmetry is expanded

    if reflection_data.F is not None:
        expanded.F = reflection_data.F[orig_idx_tensor]

    if reflection_data.F_sigma is not None:
        expanded.F_sigma = reflection_data.F_sigma[orig_idx_tensor]

    if reflection_data.I is not None:
        expanded.I = reflection_data.I[orig_idx_tensor]

    if hasattr(reflection_data, "I_sigma") and reflection_data.I_sigma is not None:
        expanded.I_sigma = reflection_data.I_sigma[orig_idx_tensor]

    # Phases must absorb the translation shift; with no phases present, stash the
    # shifts so a later phase assignment can still apply them.
    if hasattr(reflection_data, "phase") and reflection_data.phase is not None:
        expanded.phase = reflection_data.phase[orig_idx_tensor] + phase_shifts
    else:
        expanded._expansion_phase_shifts = phase_shifts

    if hasattr(reflection_data, "fom") and reflection_data.fom is not None:
        expanded.fom = reflection_data.fom[orig_idx_tensor]

    if reflection_data.rfree_flags is not None:
        expanded.rfree_flags = reflection_data.rfree_flags[orig_idx_tensor]

    if expanded.cell is not None:
        expanded._calculate_resolution()

    expanded.bin_indices = None  # invalidated by expansion

    expanded.amplitude_source = reflection_data.amplitude_source
    expanded.intensity_source = reflection_data.intensity_source
    expanded.phase_source = reflection_data.phase_source
    expanded.rfree_source = reflection_data.rfree_source

    expanded.source = reflection_data
    expanded.last_op = f"expand_to_p1(include_friedel={include_friedel})"

    return expanded


def _check_systematic_absences(
    hkl: torch.Tensor,
    matrices: torch.Tensor,
    translations: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Bool mask over ``hkl``, True where some op maps h -> h at non-integer h·t.

    ``matrices`` are the *real-space* rotations (n_ops, 3, 3); the transpose is
    taken here.
    """
    n_refl = len(hkl)
    n_ops = matrices.shape[0]
    absent = torch.zeros(n_refl, dtype=torch.bool, device=device)

    hkl_float = hkl.to(dtype=get_float_dtype(), device=device)
    recip_matrices = matrices.transpose(-2, -1).to(
        dtype=get_float_dtype(), device=device
    )
    translations = translations.to(dtype=get_float_dtype(), device=device)

    for i in range(n_ops):
        R = recip_matrices[i]
        t = translations[i]

        hkl_transformed = torch.matmul(hkl_float, R.T)
        hkl_transformed_int = torch.round(hkl_transformed).to(torch.int32)

        same_reflection = (hkl_transformed_int == hkl).all(dim=-1)

        h_dot_t = torch.matmul(hkl_float, t)

        phase_mod = torch.abs(h_dot_t - torch.round(h_dot_t))
        non_integer_phase = phase_mod > 1e-6

        absent = absent | (same_reflection & non_integer_phase)

    return absent


def reduce_hkl(
    hkl_p1: torch.Tensor,
    spacegroup: SpaceGroupLike,
    include_friedel: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce P1 Miller indices to the asymmetric unit of a target space group.

    The inverse of :func:`expand_hkl`: symmetry-equivalent P1 reflections merge
    into one ASU reflection. The index map has *constant multiplicity* -- its
    second dimension is always ``n_equiv = n_ops * (2 if include_friedel else 1)``
    however many equivalents actually exist in ``hkl_p1`` -- so aggregation needs
    no variable-length ops.

    Parameters
    ----------
    hkl_p1 : torch.Tensor, shape (N, 3)
        Input Miller indices in P1 (complete hemisphere).
    spacegroup : str, int, or gemmi.SpaceGroup
        Target space group specification.
    include_friedel : bool, default True
        If True, also consider Friedel mates when finding ASU representative.
    device : torch.device, optional
        Computation device. If None, uses hkl_p1's device.

    Returns
    -------
    hkl_asu : torch.Tensor, shape (M, 3), dtype int32
        Unique Miller indices in the asymmetric unit.
    reduction_indices : torch.Tensor, shape (M, n_equiv), dtype int64
        Indices into ``hkl_p1`` for each ASU reflection's equivalents, **-1 where
        no P1 reflection exists** -- mask or clamp before gathering, or a -1 will
        silently read the last row: ``F_asu = aggregate(F_p1[reduction_indices], dim=1)``.
    phase_shifts : torch.Tensor, shape (M, n_equiv), dtype float32
        Phase shifts to apply before aggregation; negated for Friedel mates.
    """
    if device is None:
        device = hkl_p1.device

    # Get symmetry operations
    symmetry = SpaceGroup(spacegroup, dtype=get_float_dtype(), device=device)
    n_ops = symmetry.matrices.shape[0]

    # Reciprocal space matrices (transpose of real space)
    recip_matrices = symmetry.matrices.transpose(-2, -1)
    translations = symmetry.translations

    # Total number of equivalent positions per ASU reflection
    n_equiv = n_ops * (2 if include_friedel else 1)

    # Convert hkl to float for matrix operations
    hkl_float = hkl_p1.to(dtype=get_float_dtype(), device=device)
    n_p1 = len(hkl_float)

    # Build lookup from hkl tuple to index in P1 array
    hkl_p1_np = hkl_p1.cpu().numpy()
    p1_lookup = {tuple(h): idx for idx, h in enumerate(hkl_p1_np)}

    # For each P1 reflection, find its "canonical" ASU representative
    # The canonical form is the lexicographically smallest (h, k, l) among all equivalents
    def get_canonical_hkl(hkl_single):
        """Find canonical ASU representative for a single reflection."""
        equivalents = []

        for i in range(n_ops):
            # h' = h @ R^T
            hkl_trans = torch.round(torch.matmul(hkl_single, recip_matrices[i].T)).to(
                torch.int32
            )
            equivalents.append(hkl_trans)

            if include_friedel:
                equivalents.append(-hkl_trans)

        # Stack and find lexicographically smallest
        equiv_stack = torch.stack(equivalents, dim=0)

        # Sort by (h, k, l) lexicographically
        # Convert to tuple for comparison
        equiv_np = equiv_stack.cpu().numpy()
        equiv_tuples = [tuple(e) for e in equiv_np]
        canonical = min(equiv_tuples)

        return canonical

    # Map each P1 reflection to its canonical ASU representative
    p1_to_asu = {}  # maps P1 index to canonical ASU tuple
    asu_reflections = (
        {}
    )  # maps canonical ASU tuple to list of (P1_idx, phase_shift, equiv_idx)

    for p1_idx, hkl_single in enumerate(hkl_float):
        canonical = get_canonical_hkl(hkl_single)

        p1_to_asu[p1_idx] = canonical

        if canonical not in asu_reflections:
            asu_reflections[canonical] = []

        # Find which equivalent this P1 reflection corresponds to
        for equiv_idx in range(n_ops):
            R = recip_matrices[equiv_idx]
            t = translations[equiv_idx]

            hkl_trans = torch.round(torch.matmul(hkl_single, R.T)).to(torch.int32)
            # -2π h·t, same convention as expand_hkl (see the derivation there).
            phase_shift = -2.0 * np.pi * torch.matmul(hkl_single, t)

            if tuple(hkl_trans.cpu().numpy()) == canonical:
                asu_reflections[canonical].append(
                    (p1_idx, phase_shift.item(), equiv_idx)
                )
                break

            if include_friedel:
                if tuple((-hkl_trans).cpu().numpy()) == canonical:
                    # Friedel mate: phase is negated
                    asu_reflections[canonical].append(
                        (p1_idx, -phase_shift.item(), equiv_idx + n_ops)
                    )
                    break

    # Build output tensors
    asu_list = sorted(asu_reflections.keys())
    n_asu = len(asu_list)

    hkl_asu = torch.tensor(asu_list, dtype=torch.int32, device=device)
    reduction_indices = torch.full(
        (n_asu, n_equiv), -1, dtype=torch.int64, device=device
    )
    phase_shifts = torch.zeros((n_asu, n_equiv), dtype=get_float_dtype(), device=device)

    # Fill in the indices and phase shifts
    for asu_idx, asu_hkl in enumerate(asu_list):
        for p1_idx, phase, equiv_idx in asu_reflections[asu_hkl]:
            reduction_indices[asu_idx, equiv_idx] = p1_idx
            phase_shifts[asu_idx, equiv_idx] = phase

    return hkl_asu, reduction_indices, phase_shifts


def _asu_condition_vectorized(h, k, l, condition_key):
    """Vectorized CCP4 ASU membership test over numpy ``h``/``k``/``l`` arrays.

    ``condition_key`` is a condition string from
    ``gemmi.ReciprocalAsu.condition_str()``; an unrecognised one raises
    ``ValueError``, which callers catch to fall back on ``gemmi``'s own scalar check.
    """
    # Map the 10 distinct CCP4 ASU conditions (covers all 230 space groups).
    _conditions = {
        # Laue -1 (triclinic)
        "l>0 or (l=0 and (h>0 or (h=0 and k>=0)))": lambda h, k, l: (l > 0)
        | ((l == 0) & ((h > 0) | ((h == 0) & (k >= 0)))),
        # Laue 2/m (monoclinic)
        "k>=0 and (l>0 or (l=0 and h>=0))": lambda h, k, l: (k >= 0)
        & ((l > 0) | ((l == 0) & (h >= 0))),
        # Laue mmm (orthorhombic)
        "h>=0 and k>=0 and l>=0": lambda h, k, l: (h >= 0) & (k >= 0) & (l >= 0),
        # Laue 4/m, 6/m (tetragonal, hexagonal)
        "l>=0 and ((h>=0 and k>0) or (h=0 and k=0))": lambda h, k, l: (l >= 0)
        & (((h >= 0) & (k > 0)) | ((h == 0) & (k == 0))),
        # Laue 4/mmm, 6/mmm
        "h>=k and k>=0 and l>=0": lambda h, k, l: (h >= k) & (k >= 0) & (l >= 0),
        # Laue -3 (trigonal, no mirror)
        "(h>=0 and k>0) or (h=0 and k=0 and l>=0)": lambda h, k, l: ((h >= 0) & (k > 0))
        | ((h == 0) & (k == 0) & (l >= 0)),
        # Laue -3m, P312 variant
        "h>=k and k>=0 and (k>0 or l>=0)": lambda h, k, l: (h >= k)
        & (k >= 0)
        & ((k > 0) | (l >= 0)),
        # Laue -3m, P321 variant
        "h>=k and k>=0 and (h>k or l>=0)": lambda h, k, l: (h >= k)
        & (k >= 0)
        & ((h > k) | (l >= 0)),
        # Laue m-3 (cubic)
        "h>=0 and ((l>=h and k>h) or (l=h and k=h))": lambda h, k, l: (h >= 0)
        & (((l >= h) & (k > h)) | ((l == h) & (k == h))),
        # Laue m-3m (cubic, full symmetry)
        "k>=l and l>=h and h>=0": lambda h, k, l: (k >= l) & (l >= h) & (h >= 0),
    }

    fn = _conditions.get(condition_key)
    if fn is None:
        raise ValueError(f"Unknown ASU condition: {condition_key}")
    return fn(h, k, l)


def canonicalize_hkl(
    hkl: torch.Tensor,
    spacegroup: SpaceGroupLike,
    include_friedel: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map Miller indices to canonical CCP4 ASU representatives.

    Selects one representative per reflection under the standard CCP4 asymmetric
    unit convention. Runs on CPU/numpy regardless of ``device`` (the ASU lookup
    tables are numpy-backed), returning tensors on ``device``. Raises
    ``ValueError`` if any reflection has no ASU representative, which happens for
    the Friedel half of reciprocal space when ``include_friedel=False``.

    Parameters
    ----------
    hkl : torch.Tensor, shape (N, 3), dtype int32
        Input Miller indices.
    spacegroup : str, int, or gemmi.SpaceGroup
        Space group specification.
    include_friedel : bool, default True
        Whether Friedel mates are considered equivalent.
    device : torch.device, optional
        Computation device. If None, uses hkl's device.

    Returns
    -------
    canonical_hkl : torch.Tensor, shape (N, 3), dtype int32
        Remapped indices, sorted lexicographically by (h, k, l).
    phase_shifts : torch.Tensor, shape (N,), dtype float32
        Additive phase correction in radians.
    friedel_flags : torch.Tensor, shape (N,), dtype bool
        True where Friedel conjugation was applied.
    sort_indices : torch.Tensor, shape (N,), dtype int64
        Permutation from original to sorted order.

    Notes
    -----
    ``phase_shifts`` assumes the caller conjugates first — the contract is
    ``phi_new = torch.where(friedel_flags, -phi_old, phi_old) + phase_shifts``.
    """
    import gemmi

    if device is None:
        device = hkl.device
    hkl_dtype = hkl.dtype

    n_refl = len(hkl)
    if n_refl == 0:
        empty_hkl = torch.empty((0, 3), dtype=hkl_dtype, device=device)
        empty_f = torch.empty(0, dtype=get_float_dtype(), device=device)
        empty_b = torch.empty(0, dtype=torch.bool, device=device)
        empty_i = torch.empty(0, dtype=torch.int64, device=device)
        return empty_hkl, empty_f, empty_b, empty_i

    # Normalize spacegroup (CPU-only: this branch builds numpy-backed lookup tables)
    sg_obj = SpaceGroup(spacegroup, dtype=get_float_dtype(), device=torch.device("cpu"))
    sg_gemmi = sg_obj._gemmi
    asu = gemmi.ReciprocalAsu(sg_gemmi)
    condition_key = asu.condition_str()
    recip_mats = sg_obj.matrices.transpose(-2, -1).numpy()  # (n_ops, 3, 3)
    translations_np = sg_obj.translations.numpy()  # (n_ops, 3)
    n_ops = len(recip_mats)

    hkl_np = hkl.cpu().numpy().astype(np.int32)  # (N, 3)
    # Reciprocal-space rotation matrices are always integer-valued (0, ±1).
    recip_mats_i = np.round(recip_mats).astype(np.int32)

    # One op (+ its Friedel mate) at a time, so high-symmetry groups exit early:
    # most reflections are resolved by the first few operators.
    canonical_np = np.empty_like(hkl_np)
    op_idx = np.empty(n_refl, dtype=np.int32)
    friedel_np = np.zeros(n_refl, dtype=bool)
    remaining = np.ones(n_refl, dtype=bool)

    for i_op in range(n_ops):
        if not remaining.any():
            break
        idx = np.where(remaining)[0]
        hkl_sub = hkl_np[idx]  # (M, 3)
        R = recip_mats_i[i_op]  # (3, 3)
        equiv_sub = hkl_sub @ R.T  # (M, 3), int32 matmul — no rounding needed

        # Check non-Friedel
        h, k, l = equiv_sub[:, 0], equiv_sub[:, 1], equiv_sub[:, 2]
        try:
            in_asu_pos = _asu_condition_vectorized(h, k, l, condition_key)
        except ValueError:
            in_asu_pos = np.array(
                [asu.is_in(row.tolist()) for row in equiv_sub], dtype=bool
            )

        hit_pos = np.where(in_asu_pos)[0]
        if len(hit_pos) > 0:
            global_idx = idx[hit_pos]
            canonical_np[global_idx] = equiv_sub[hit_pos]
            op_idx[global_idx] = i_op
            remaining[global_idx] = False

        # Check Friedel mate
        if include_friedel and remaining.any():
            # Recompute idx for remaining after non-Friedel hits
            idx_f = np.where(remaining)[0]
            hkl_sub_f = hkl_np[idx_f]
            equiv_neg = -(hkl_sub_f @ R.T)

            h_n, k_n, l_n = equiv_neg[:, 0], equiv_neg[:, 1], equiv_neg[:, 2]
            try:
                in_asu_neg = _asu_condition_vectorized(h_n, k_n, l_n, condition_key)
            except ValueError:
                in_asu_neg = np.array(
                    [asu.is_in(row.tolist()) for row in equiv_neg], dtype=bool
                )

            hit_neg = np.where(in_asu_neg)[0]
            if len(hit_neg) > 0:
                global_idx_f = idx_f[hit_neg]
                canonical_np[global_idx_f] = equiv_neg[hit_neg]
                op_idx[global_idx_f] = i_op
                friedel_np[global_idx_f] = True
                remaining[global_idx_f] = False

    # ``canonical_np``/``op_idx`` are uninitialized ``np.empty`` buffers, so an
    # unmapped row would propagate garbage indices and phases. Fail loudly instead.
    if remaining.any():
        n_unmapped = int(remaining.sum())
        example = hkl_np[np.where(remaining)[0][0]].tolist()
        raise ValueError(
            f"canonicalize_hkl could not map {n_unmapped} reflection(s) to the "
            f"reciprocal ASU of space group {sg_obj} "
            f"(include_friedel={include_friedel}); e.g. hkl={example}. With "
            f"include_friedel=False the Friedel half of reciprocal space has no "
            f"pure-rotation representative in the Laue-based CCP4 ASU."
        )

    # Sign depends on whether the row was Friedel-flipped, because the consumer
    # already negated phi for those rows: -2π h·t normally, +2π h·t for Friedel.
    # A single uniform sign is wrong for one half and invisible in P21/P212121/C2,
    # where every shift is 0 or π. tests/unit/symmetry/test_phase_convention.py.
    t_selected = translations_np[op_idx]  # (N, 3)
    friedel_sign = np.where(friedel_np, 1.0, -1.0).astype(np.float32)
    phase_shifts_np = (
        friedel_sign
        * 2.0
        * np.pi
        * np.sum(hkl_np.astype(np.float32) * t_selected, axis=1)
    ).astype(np.float32)

    # --- Convert to tensors and sort ---
    canonical_hkl = torch.tensor(canonical_np, dtype=hkl_dtype, device=device)
    phase_shifts = torch.tensor(phase_shifts_np, dtype=get_float_dtype(), device=device)
    friedel_flags = torch.tensor(friedel_np, dtype=torch.bool, device=device)

    # Lexicographic sort by (h, k, l) via composite key
    h_max = int(canonical_hkl.abs().max().item()) + 1
    base = 2 * h_max + 1
    sort_key = (
        canonical_hkl[:, 0].to(torch.int64) * base * base
        + canonical_hkl[:, 1].to(torch.int64) * base
        + canonical_hkl[:, 2].to(torch.int64)
    )
    sort_indices = torch.argsort(sort_key)

    return (
        canonical_hkl[sort_indices],
        phase_shifts[sort_indices],
        friedel_flags[sort_indices],
        sort_indices,
    )


def expand_reciprocal_grid(
    F_grid: torch.Tensor,
    space_group: str,
    mode: str = "average",
    include_friedel: bool = True,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Expand or symmetrize a reciprocal space grid using crystallographic symmetry.

    Convenience wrapper that builds a :class:`ReciprocalSymmetryGrid` for
    ``F_grid.shape`` and applies it, so it re-pays the whole precompute on every
    call -- hold the class yourself inside a loop.

    Parameters
    ----------
    F_grid : torch.Tensor, shape (nh, nk, nl)
        Input structure factor grid (can be complex or real).
    space_group : str, int, or gemmi.SpaceGroup
        Space group specification (e.g., 'P21', 'P212121'). The type hint is
        narrowed to ``str``, but any ``SpaceGroupLike`` value is accepted.
    mode : {'average', 'expand', 'sum'}, default 'average'
        Average over all symmetry equivalents (symmetrize), expand from the
        asymmetric unit to the full grid, or sum all symmetry mates.
    include_friedel : bool, default True
        If True, also apply Friedel symmetry after space group symmetry.
    device : torch.device, optional
        Device for computation. If None, uses F_grid's device.

    Returns
    -------
    torch.Tensor, shape (nh, nk, nl)
        Symmetrized or expanded structure factor grid.
    """
    if device is None:
        device = F_grid.device

    grid_shape = F_grid.shape
    dtype = get_float_dtype() if not F_grid.is_complex() else F_grid.real.dtype

    recip_sym = ReciprocalSymmetryGrid(
        space_group=space_group,
        grid_shape=grid_shape,
        dtype_float=dtype,
        verbose=0,
        device=device,
    )

    F_result = recip_sym(F_grid, mode=mode)

    if include_friedel:
        F_result = recip_sym.apply_friedel(F_result)

    return F_result
