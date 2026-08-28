"""SfDS -- structure factors by direct summation.

An nn.Module alternative to :class:`~torchref.model.SfFFT` that needs no grid:
it sums atomic contributions (isotropic and anisotropic) directly and applies
crystallographic symmetry in reciprocal space.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from torchref.base.direct_summation import (
    ds_aniso,
    ds_iso,
)
from torchref.base.reciprocal import (
    get_scattering_vectors,
    reciprocal_basis_matrix,
)
from torchref.config import dtypes, get_complex_dtype, get_default_device
from torchref.symmetry import Cell, SpaceGroup
from torchref.symmetry.spacegroup import SpaceGroupLike
from torchref.utils.device_mixin import DeviceMovementMixin
from torchref.utils.device_resolution import require_cell_dtype, resolve_device


class SfDS(DeviceMovementMixin, nn.Module):
    """
    Structure Factor calculator using Direct Summation.

    Sums atomic contributions without building an intermediate electron-density
    map, batching automatically to stay inside ``max_memory_gb``. Unlike
    :class:`SfFFT` it needs no grid setup, computes scattering factors from the
    ITC92 A/B coefficients internally, and returns ``(sf, None)`` where SfFFT
    returns ``(sf, density_map)``.

    Parameters
    ----------
    cell : Cell, optional
        Unit cell object containing cell parameters.
    spacegroup : SpaceGroupLike, optional
        Space group specification (string, int, or gemmi.SpaceGroup).
        If None, defaults to P1.
    dtype_float : torch.dtype, optional
        Data type for floating point tensors. Default is dtypes.float.
    device : torch.device, optional
        Computation device. Defaults to the configured default device
        (``get_default_device()``).
    verbose : int, optional
        Verbosity level for logging. Default is 0.
    max_memory_gb : float, optional
        Maximum memory to use for intermediate tensors in GB. Default is 2.0.
        Set to None to disable batching.

    Attributes
    ----------
    cell, spacegroup : Cell, SpaceGroup
        The unit cell and the space group as an nn.Module carrying its symmetry
        matrices and translations; setting ``cell`` drops the cached
        reciprocal basis.

    Examples
    --------
    Standalone usage::

        from torchref.symmetry import Cell
        cell = Cell([50, 60, 70, 90, 90, 90])
        sf_ds = SfDS(cell, spacegroup='P212121')
        sf, _ = sf_ds.compute_structure_factors(
            hkl, xyz_iso, adp_iso, occ_iso, A_iso, B_iso
        )
    """

    def __init__(
        self,
        cell: Optional[Cell] = None,
        spacegroup: SpaceGroupLike = None,
        dtype_float: torch.dtype = None,
        device: torch.device = None,
        verbose: int = 0,
        max_memory_gb: float = 2.0,
        force_portable: Optional[bool] = None,
    ):
        """
        Initialize the SfDS module with cell and spacegroup.

        Parameters
        ----------
        cell : Cell, optional
            Unit cell object. If None, must be set later.
        spacegroup : SpaceGroupLike, optional
            Space group specification. If None, defaults to P1.
        dtype_float : torch.dtype, optional
            Data type for floating point tensors. Default is dtypes.float.
        device : torch.device, optional
            Computation device. Defaults to the configured device.current.
        verbose : int, optional
            Verbosity level for logging. Default is 0.
        max_memory_gb : float, optional
            Maximum memory for intermediate tensors in GB. Default is 2.0.
        force_portable : bool, optional
            Pin the portable reference path instead of the fastest usable backend,
            per instance. ``None`` (default) defers to the process-wide setting,
            so ``with use_portable():`` steers an unconfigured instance.
        """
        super().__init__()
        if dtype_float is None:
            dtype_float = dtypes.float
        self.dtype_float = dtype_float
        # Derive from ``cell`` when no device is given, instead of jumping to
        # the global default and leaving a caller-supplied cell behind on
        # another device. An explicit ``device`` still wins and moves the cell.
        self.device = resolve_device(cell, device=device)
        self.verbose = verbose
        self.max_memory_gb = max_memory_gb
        self.force_portable = force_portable

        # Store cell and spacegroup
        self._cell = cell
        self._spacegroup = None

        if spacegroup is not None or cell is not None:
            self._spacegroup = SpaceGroup(
                spacegroup, dtype=dtype_float, device=self.device
            )

        # Cache reciprocal basis matrix
        self._recB: Optional[torch.Tensor] = None

    # =========================================================================
    # Cell and SpaceGroup properties
    # =========================================================================

    @property
    def cell(self) -> Optional[Cell]:
        """Unit cell object."""
        return self._cell

    @cell.setter
    def cell(self, value: Cell):
        """Set unit cell and invalidate cached reciprocal basis matrix."""
        self._cell = value
        self._recB = None  # Invalidate cache

    @property
    def spacegroup(self) -> Optional[SpaceGroup]:
        """Space group object (SpaceGroup nn.Module)."""
        return self._spacegroup

    @spacegroup.setter
    def spacegroup(self, value: SpaceGroupLike):
        """Set space group."""
        if value is not None:
            self._spacegroup = SpaceGroup(
                value, dtype=self.dtype_float, device=self.device
            )
        else:
            self._spacegroup = None

    @property
    def fractional_matrix(self) -> Optional[torch.Tensor]:
        """Get fractionalization matrix from cell."""
        if self._cell is not None:
            return self._cell.fractional_matrix
        return None

    @property
    def inv_fractional_matrix(self) -> Optional[torch.Tensor]:
        """Get orthogonalization matrix from cell."""
        if self._cell is not None:
            return self._cell.inv_fractional_matrix
        return None

    def set_cell_and_spacegroup(self, cell: Cell, spacegroup: SpaceGroupLike = None):
        """
        Set cell and spacegroup for this SfDS instance.

        Parameters
        ----------
        cell : Cell
            Unit cell object.
        spacegroup : SpaceGroupLike, optional
            Space group specification.

        Notes
        -----
        Receiver wins: an incoming cell on another device is moved to match
        this module rather than the other way round.
        """
        self.device = resolve_device(self, cell)
        self._cell = cell
        self._recB = None  # Invalidate cache
        self.spacegroup = spacegroup

    # =========================================================================
    # Internal helper methods
    # =========================================================================

    def _get_reciprocal_basis_matrix(self) -> torch.Tensor:
        """Cached ``(3, 3)`` reciprocal basis (a*, b*, c* as rows).

        Raises ``RuntimeError`` if no cell is set, and refuses a cell whose dtype
        differs from ``self.dtype_float``.
        """
        if self._cell is None:
            raise RuntimeError("Cell not set. Call set_cell_and_spacegroup() first.")
        require_cell_dtype(self._cell, self.dtype_float, type(self).__name__)

        if self._recB is None:
            self._recB = reciprocal_basis_matrix(self._cell.data)

        return self._recB

    def _compute_scattering_factors(
        self, s: torch.Tensor, A: torch.Tensor, B: torch.Tensor
    ) -> torch.Tensor:
        """``f(s) = sum_i A_i exp(-B_i s^2 / 4)``, shape ``(N_refl, N_atoms)``.

        Unused by the production path: the active backends
        (``ds_iso``/``ds_aniso``, dispatched from :meth:`_compute_p1_sf`) work
        from raw A/B and never materialize this ``(N_refl, N_atoms)`` array.
        """
        s_sq = (s.reshape(-1, 1, 1) ** 2) / 4  # (N_refl, 1, 1)
        B_expanded = B.unsqueeze(0)  # (1, N_atoms, 5)
        A_expanded = A.unsqueeze(0)  # (1, N_atoms, 5)

        exp_terms = torch.exp(-B_expanded * s_sq)  # (N_refl, N_atoms, 5)

        # Sum over Gaussian components: (N_refl, N_atoms)
        f = torch.sum(A_expanded * exp_terms, dim=-1)

        return f

    def _cartesian_to_fractional(self, xyz_cartesian: torch.Tensor) -> torch.Tensor:
        """``(N, 3)`` Cartesian coordinates to fractional; needs a cell whose
        dtype matches ``self.dtype_float``.
        """
        if self._cell is None:
            raise RuntimeError("Cell not set. Call set_cell_and_spacegroup() first.")
        require_cell_dtype(self._cell, self.dtype_float, type(self).__name__)

        # fractional = cartesian @ inv_frac_matrix.T
        return torch.matmul(xyz_cartesian, self.inv_fractional_matrix.T)

    # =========================================================================
    # Structure Factor Computation
    # =========================================================================

    def compute_structure_factors(
        self,
        hkl: torch.Tensor,
        xyz_iso: torch.Tensor,
        adp_iso: torch.Tensor,
        occ_iso: torch.Tensor,
        A_iso: torch.Tensor,
        B_iso: torch.Tensor,
        xyz_aniso: Optional[torch.Tensor] = None,
        u_aniso: Optional[torch.Tensor] = None,
        occ_aniso: Optional[torch.Tensor] = None,
        A_aniso: Optional[torch.Tensor] = None,
        B_aniso: Optional[torch.Tensor] = None,
        apply_symmetry: bool = True,
    ) -> Tuple[torch.Tensor, None]:
        """
        Compute structure factors from atomic parameters using direct summation.

        Late symmetry, as in :class:`SfFFT`: P1 structure factors are computed at
        the symmetry-equivalent HKLs and combined as
        ``F_sym(h) = Σ_ops exp(2πi h.t) * F_P1(R^T @ h)``.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices with shape (n_reflections, 3).
        xyz_iso, adp_iso, occ_iso : torch.Tensor
            Isotropic atoms: Cartesian coordinates ``(n_iso, 3)``, isotropic ADPs
            ``(n_iso,)``, occupancies ``(n_iso,)``.
        A_iso, B_iso : torch.Tensor
            ITC92 amplitudes / widths for the isotropic atoms, ``(n_iso, 5)``.
        xyz_aniso, u_aniso, occ_aniso : torch.Tensor, optional
            Anisotropic atoms: coordinates ``(n_aniso, 3)``, U components
            ``(n_aniso, 6)``, occupancies ``(n_aniso,)``.
        A_aniso, B_aniso : torch.Tensor, optional
            ITC92 amplitudes / widths for the anisotropic atoms, ``(n_aniso, 5)``.
        apply_symmetry : bool, optional
            If True, apply crystallographic symmetry. Default is True.

        Returns
        -------
        sf : torch.Tensor
            Complex structure factors with shape (n_reflections,).
        None
            Second return value is None (for API compatibility with SfFFT).
        """
        if self._cell is None:
            raise RuntimeError("Cell not set. Call set_cell_and_spacegroup() first.")
        # Refused, not reconciled: unlike the device normalization just below, a dtype cast
        # is lossy, so the cell is the caller's to fix. See ``require_cell_dtype``.
        require_cell_dtype(self._cell, self.dtype_float, type(self).__name__)

        # Normalize the input hkl onto this module's device. The symmetry
        # helpers derive equiv_hkls/phases from hkl.device while sf_total is
        # allocated on self.device, so a caller passing hkl on another device
        # would hit a cross-device add. resolve_device gives the module's
        # canonical device; hkl is moved explicitly (resolve_device only moves
        # modules in place, not plain tensors).
        device = resolve_device(self)
        hkl = hkl.to(device)

        # Cache atomic parameters for reuse
        xyz_frac_iso = (
            self._cartesian_to_fractional(xyz_iso) if len(xyz_iso) > 0 else None
        )
        xyz_frac_aniso = (
            self._cartesian_to_fractional(xyz_aniso)
            if xyz_aniso is not None and len(xyz_aniso) > 0
            else None
        )

        # No symmetry: compute F_P1 directly
        if not apply_symmetry or self._spacegroup is None:
            sf_p1 = self._compute_p1_sf(
                hkl,
                xyz_frac_iso,
                adp_iso,
                occ_iso,
                A_iso,
                B_iso,
                xyz_frac_aniso,
                u_aniso,
                occ_aniso,
                A_aniso,
                B_aniso,
            )
            return sf_p1, None

        # Apply late symmetry: F_sym(h) = Σ_ops exp(2πi h.t) * F_P1(R^T @ h)
        n_ops = self._spacegroup.n_ops
        equiv_hkls = self._spacegroup.expand_reciprocal(hkl)  # (n_ops, N, 3)
        phases = self._spacegroup.phase_factors(hkl)  # (n_ops, N)

        # Compute F_P1 at each equivalent HKL and combine
        sf_total = torch.zeros(
            hkl.shape[0], dtype=get_complex_dtype(), device=self.device
        )

        for i in range(n_ops):
            equiv_hkl_i = equiv_hkls[i].to(dtype=self.dtype_float)  # (N, 3)

            sf_p1_i = self._compute_p1_sf(
                equiv_hkl_i,
                xyz_frac_iso,
                adp_iso,
                occ_iso,
                A_iso,
                B_iso,
                xyz_frac_aniso,
                u_aniso,
                occ_aniso,
                A_aniso,
                B_aniso,
            )

            # Apply phase and accumulate
            sf_total = sf_total + phases[i] * sf_p1_i

        return sf_total, None

    def _compute_p1_sf(
        self,
        hkl: torch.Tensor,
        xyz_frac_iso: Optional[torch.Tensor],
        adp_iso: torch.Tensor,
        occ_iso: torch.Tensor,
        A_iso: torch.Tensor,
        B_iso: torch.Tensor,
        xyz_frac_aniso: Optional[torch.Tensor],
        u_aniso: Optional[torch.Tensor],
        occ_aniso: Optional[torch.Tensor],
        A_aniso: Optional[torch.Tensor],
        B_aniso: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute P1 structure factors (no symmetry expansion).

        Dispatches to the capability-selected backend (Triton on CUDA+float32,
        else checkpointed eager). The backend computes scattering factors
        internally from A/B and never materializes a large (N_refl, N_atoms)
        intermediate.
        """
        # Get reciprocal basis matrix and compute scattering vectors
        recB = self._get_reciprocal_basis_matrix()
        s_vectors = get_scattering_vectors(hkl, self._cell.data, recB)
        s = torch.norm(s_vectors, dim=1)

        sf_total = torch.zeros(
            hkl.shape[0], dtype=get_complex_dtype(), device=self.device
        )

        # Compute isotropic contribution
        if xyz_frac_iso is not None and len(xyz_frac_iso) > 0:
            sf_iso = ds_iso(
                hkl,
                s,
                xyz_frac_iso,
                occ_iso,
                adp_iso,
                A_iso,
                B_iso,
                force_portable=self.force_portable,
                max_memory_gb=self.max_memory_gb,
            )
            sf_total = sf_total + sf_iso.to(sf_total.dtype)

        # Compute anisotropic contribution
        if xyz_frac_aniso is not None and len(xyz_frac_aniso) > 0:
            sf_aniso = ds_aniso(
                hkl,
                s_vectors,
                xyz_frac_aniso,
                occ_aniso,
                u_aniso,
                A_aniso,
                B_aniso,
                force_portable=self.force_portable,
                max_memory_gb=self.max_memory_gb,
            )
            sf_total = sf_total + sf_aniso.to(sf_total.dtype)

        return sf_total

    # =========================================================================
    # Device Movement
    # =========================================================================

    def reset_cache(self) -> None:
        """Drop the cached reciprocal-basis matrix; recomputed on next use."""
        self._recB = None

    def copy(self) -> "SfDS":
        """Create a deep copy of this SfDS module.

        Returns
        -------
        SfDS
            A new SfDS instance with cloned cell and spacegroup.
        """
        # Clone the cell
        new_cell = self._cell.clone() if self._cell is not None else None

        # Copy the spacegroup
        new_spacegroup = (
            self._spacegroup.copy() if self._spacegroup is not None else None
        )

        # Create new SfDS with copied components
        new_ds = SfDS(
            cell=new_cell,
            spacegroup=new_spacegroup,
            dtype_float=self.dtype_float,
            device=self.device,
            verbose=self.verbose,
            max_memory_gb=self.max_memory_gb,
            force_portable=self.force_portable,
        )

        return new_ds
