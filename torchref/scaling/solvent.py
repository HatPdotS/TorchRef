"""
A class for modelling solvent contribution to structure factors.
"""

import torch
import torch.nn as nn

from torchref.base import (
    extract_structure_factor_from_grid,
    get_scattering_vectors,
    ifft,
)
from torchref.config import get_float_dtype
from torchref.utils.debug_utils import DebugMixin
from torchref.utils.device_mixin import DeviceMixin
from torchref.utils.device_resolution import resolve_device
from torchref.utils.utils import ModuleReference, TensorDict

#: ``ln 2``, so ``s_half_sq = ss_half`` halves the solvent term by construction.
_LN2 = 0.6931471805599453

#: Bounds on the solvent falloff, applied as clamps inside :meth:`SolventModel.damping`.
#: ``ss_half`` is the half-point in ``(sin(theta)/lambda)**2``, quoted here as the
#: resolution ``d_half = 1 / (2 sqrt(ss_half))``; the range spans well beyond the observed
#: spread while excluding the degenerate slow-power-law fits the unbounded form can reach.
#: ``n = 1`` is exactly a Debye-Waller factor with ``B = ln2 / ss_half``, so the shipped
#: exponential is nested at the lower bound rather than merely approximated.
SS_HALF_BOUNDS = (0.0025, 0.04)   # d_half 10.0 .. 2.5 A
N_EXP_BOUNDS = (1.0, 20.0)

#: Cache for :func:`_voxel_offsets_within`, keyed on the arguments that determine the
#: result. Solvent masks are rebuilt every time coordinates move, always on the same grid
#: and cell, so the offsets are computed once per refinement rather than once per call.
_OFFSET_CACHE = {}


def _voxel_offsets_within(radius, grid_dims, frac, device, strict=False):
    """Integer voxel offsets whose Cartesian displacement is within ``radius``.

    Offset ``o`` displaces a point by the Cartesian vector ``frac @ (o / grid_dims)``, so
    its length follows from the cell's metric tensor and the enumerated set is a true
    Cartesian ball in **any** unit cell, not only orthogonal ones. The per-axis search box
    comes from the reciprocal basis: ``|o_i| <= grid_dims_i * |a*_i| * radius``.

    Parameters
    ----------
    radius : float
        Cutoff in Angstrom.
    grid_dims : torch.Tensor
        Grid dimensions ``(N0, N1, N2)``, integer.
    frac : torch.Tensor
        Fractional-to-Cartesian matrix, shape ``(3, 3)``.
    device : torch.device
        Device the offsets are built on.
    strict : bool, default False
        Use ``<`` rather than ``<=`` against ``radius``.

    Returns
    -------
    torch.Tensor
        Offsets, shape ``(R, 3)``, integer.
    """
    key = (
        float(radius),
        tuple(int(v) for v in grid_dims.tolist()),
        tuple(round(float(v), 10) for v in frac.flatten().tolist()),
        str(device),
        bool(strict),
    )
    cached = _OFFSET_CACHE.get(key)
    if cached is not None:
        return cached

    dtype = get_float_dtype()
    frac = frac.to(device=device, dtype=dtype)
    N = grid_dims.to(device=device, dtype=dtype)
    # Rows of the Cartesian-to-fractional matrix are the reciprocal basis vectors.
    recip_norms = torch.linalg.inv(frac).norm(dim=1)
    bounds = torch.ceil(N * recip_norms * radius).long()

    ranges = [
        torch.arange(-int(b), int(b) + 1, device=device) for b in bounds.tolist()
    ]
    offsets = torch.stack(torch.meshgrid(*ranges, indexing="ij"), dim=-1).reshape(-1, 3)

    disp = (offsets.to(dtype) / N) @ frac.T
    dist_sq = (disp**2).sum(-1)
    r_sq = radius**2
    keep = dist_sq < r_sq if strict else dist_sq <= r_sq
    local_offsets = offsets[keep]

    _OFFSET_CACHE[key] = local_offsets
    return local_offsets


class SolventModel(DeviceMixin, DebugMixin, nn.Module):
    """
    Bulk-solvent contribution to structure factors, Phenix-style.

    Constructed either with a model (``SolventModel(model, k_solvent=0.35)`` -- the value
    ``Scaler`` injects; the bare-constructor default is 1.1) or empty, as a shell for
    ``load_state_dict``.

    The solvent falls off as ``k_sol * exp(-ln2 * (ss / ss_half)**n)`` in
    ``ss = (sin(theta)/lambda)**2``. ``ss_half`` is where the term is halved and ``n``
    how sharply it switches off; ``n = 1`` is exactly ``exp(-B ss)`` with
    ``B = ln2 / ss_half``, so a Debye-Waller solvent is a special case rather than a
    different model. Both are clamped to :data:`SS_HALF_BOUNDS` / :data:`N_EXP_BOUNDS`.

    Attributes
    ----------
    model : ModelFT or None
        The atomic model the solvent mask is built from.
    device : torch.device
        Device for tensor operations.
    verbose : int
        Verbosity level.
    float_type : torch.dtype
        Float dtype; defaults to the configured ``get_float_dtype()``, not a
        hard-wired ``torch.float32``.
    solvent_radius, erosion_radius : float
        Probe radius for dilation and radius for the erosion step (Å).
    optimize_phase : bool
        Whether the phase offset is refined.
    log_k_solvent, log_ss_half, log_n_exp : torch.nn.Parameter
        Log solvent scattering scale, and the logs of the falloff half-point and
        exponent. Refined in log space so each stays positive.
    phase_offset : torch.nn.Parameter or buffer
        Phase offset in radians: a trainable parameter when
        ``optimize_phase=True``, otherwise a buffer fixed at 0.0.
    """

    def __init__(
        self,
        model=None,
        radius=1.1,
        k_solvent=1.1,
        d_half=3.59,
        n_exp=5.0,
        erosion_radius=0.9,
        optimize_phase=True,
        initial_phase_offset=0.0,
        verbose=1,
        float_type=None,
        device=None,
    ):
        """
        Initialize SolventModel.

        If model is provided, fully initializes the solvent model.
        If not provided (empty init), creates a shell ready for load_state_dict().

        Parameters
        ----------
        model : ModelFT, optional
            The atomic model used for structure factor calculations (optional for empty init).
        radius : float, default 1.1
            Probe radius in Angstroms for dilation (water radius).
        k_solvent : float, default 1.1
            Solvent scattering scale factor.
        d_half : float, default 3.59
            Resolution (A) at which the solvent term is halved; stored as
            ``ss_half = 1 / (4 d_half**2)``.
        n_exp : float, default 5.0
            Falloff exponent. ``1.0`` reduces the form to ``exp(-B ss)``.
        erosion_radius : float, default 0.9
            Radius in Angstroms for erosion step.
        optimize_phase : bool, default True
            Whether to optimize phase offset parameter.
        initial_phase_offset : float, default 0.0
            Initial phase offset in radians.
        verbose : int, default 1
            Verbosity level.
        float_type : torch.dtype, optional
            Float dtype. ``None`` (default) resolves at runtime to
            ``get_float_dtype()``, not a hard-wired ``torch.float32``.
        device : torch.device, default: configured device.current
            Device for tensor operations.
        """
        super(SolventModel, self).__init__()
        if float_type is None:
            float_type = get_float_dtype()
        # Follow the model when no device is given: the global default would put
        # the solvent grids on a different device than the structure they are
        # computed from (``realspace.py`` passes only a model).
        device = resolve_device(model, device=device)
        self.device = device
        self.verbose = verbose
        self.float_type = float_type
        self.solvent_radius = radius
        self.erosion_radius = erosion_radius
        self.optimize_phase = optimize_phase
        self._cache = TensorDict()

        # Empty initialization
        if model is None:
            self.model = None
            self.max_radius_angstrom = None
            # Register parameters with default values (will be overwritten by load_state_dict)
            self.log_k_solvent = nn.Parameter(
                torch.log(
                    torch.tensor(k_solvent, dtype=self.float_type, device=self.device)
                )
            )
            self._init_falloff(d_half, n_exp)
            if self.optimize_phase:
                self.phase_offset = nn.Parameter(
                    torch.tensor(
                        initial_phase_offset, dtype=self.float_type, device=self.device
                    )
                )
            else:
                self.register_buffer(
                    "phase_offset",
                    torch.tensor(0.0, dtype=self.float_type, device=self.device),
                )
            return

        # Full initialization with model
        self.model = ModuleReference(model)  # Store reference to model
        self.model.get_vdw_radii()  # Ensure VdW radii are available
        assert self.model, "Model is not initialized"
        if model.real_space_grid == None:
            model.setup_grid()

        # Phenix-style parameters
        self.solvent_radius = radius  # For dilation (accessible surface)
        self.erosion_radius = erosion_radius  # For erosion (contact surface)

        # For find_relevant_voxels: need to search far enough to capture accessible surface
        # Maximum possible distance is max(VdW) + solvent_radius
        self.max_radius_angstrom = self.model.get_vdw_radii().max() + radius

        if not isinstance(k_solvent, torch.Tensor):
            k_solvent = torch.tensor(
                k_solvent, dtype=self.float_type, device=self.device
            )
        else:
            k_solvent = k_solvent.to(dtype=self.float_type, device=self.device)
        self.log_k_solvent = nn.Parameter(torch.log(k_solvent))
        self._init_falloff(d_half, n_exp)

        # Phase offset parameter to align solvent phases with protein phases
        # This is critical because FFT of a mask gives arbitrary phases
        self.optimize_phase = optimize_phase
        if self.optimize_phase:
            self.phase_offset = nn.Parameter(
                torch.tensor(
                    initial_phase_offset, dtype=self.float_type, device=self.device
                )
            )
        else:
            self.register_buffer(
                "phase_offset",
                torch.tensor(0.0, dtype=self.float_type, device=self.device),
            )
        self._cache = TensorDict()

    def _init_falloff(self, d_half, n_exp):
        """Register ``log_ss_half`` / ``log_n_exp`` from a resolution and an exponent."""
        ss_half = 1.0 / (4.0 * float(d_half) ** 2)
        for name, value in (("log_ss_half", ss_half), ("log_n_exp", float(n_exp))):
            setattr(
                self,
                name,
                nn.Parameter(
                    torch.log(
                        torch.tensor(
                            value, dtype=self.float_type, device=self.device
                        )
                    )
                ),
            )

    def ss_half(self) -> torch.Tensor:
        """Half-point of the solvent falloff in ``(sin(theta)/lambda)**2``, clamped."""
        return torch.exp(self.log_ss_half).clamp(*SS_HALF_BOUNDS)

    def n_exp(self) -> torch.Tensor:
        """Falloff exponent, clamped. ``1`` is a Debye-Waller factor."""
        return torch.exp(self.log_n_exp).clamp(*N_EXP_BOUNDS)

    def k_solvent(self) -> torch.Tensor:
        """Solvent scattering scale."""
        return torch.exp(self.log_k_solvent.clamp(min=-10.0, max=10.0))

    def damping(self, s_half_sq: torch.Tensor) -> torch.Tensor:
        """``exp(-ln2 * (ss / ss_half)**n)`` at ``ss = (sin(theta)/lambda)**2``.

        Parameters
        ----------
        s_half_sq : torch.Tensor
            ``(sin(theta)/lambda)**2`` per reflection.

        Returns
        -------
        torch.Tensor
            Falloff factor in ``[0, 1]``, same shape as the input.
        """
        ratio = (s_half_sq / self.ss_half()).clamp(min=1e-12)
        return torch.exp((-_LN2 * ratio.pow(self.n_exp())).clamp(min=-30.0))

    def b_solvent_equivalent(self, s_half_sq: torch.Tensor) -> float:
        """The single ``B`` whose ``exp(-B ss)`` best matches this falloff.

        A reporting quantity: PDB ``REMARK 3`` and mmCIF have a field for a solvent
        B-factor and the fitted form has none, so it is back-fitted by least squares on
        ``log(damping)`` over the reflections actually present, weighted by the damping
        itself so the fit follows the range where the solvent contributes.
        """
        with torch.no_grad():
            ss = s_half_sq.detach().flatten()
            ss = ss[ss > 0]
            if ss.numel() == 0:
                return 0.0
            d = self.damping(ss)
            w = d.clamp(min=1e-6)
            # log d = -B ss, through the origin: B = -sum(w ss log d) / sum(w ss^2)
            num = (w * ss * torch.log(d.clamp(min=1e-30))).sum()
            den = (w * ss * ss).sum().clamp(min=1e-30)
            return float(-num / den)

    def get_solvent_mask(self):
        """
        Generate solvent mask following Phenix's three-step process.

        Step 1 (dilation): classify voxels around each atom as protein
            (inside VdW), boundary (between VdW and VdW+solvent_radius), or
            bulk solvent (further out). Built in chunks over atoms so peak
            memory is O(atom_chunk_size × N_box_voxels) rather than
            O(N_atoms × N_box_voxels) — critical because for typical
            macromolecule + grid combinations the dense form is multi-GB.

        Step 2 (symmetry expansion): transform the sparse ASU protein /
            boundary voxel indices through each symop and scatter into the
            P1 grid masks.

        Step 3 (erosion): a boundary voxel becomes solvent if any voxel
            within ``erosion_radius`` of it is bulk solvent, computed with a
            precomputed spherical structuring element under circular padding.

        Returns
        -------
        torch.Tensor
            Solvent mask (boolean) where True = solvent.
        """
        import torch.nn.functional as F

        if self.verbose > 1:
            print("\n=== Phenix-Style Bulk Solvent Mask Calculation ===")
            print(f"Solvent radius (dilation): {self.solvent_radius:.2f} Å")
            print(f"Shrink truncation radius (erosion): {self.erosion_radius:.2f} Å")

        xyz = self.model.xyz()  # (N_atoms, 3)
        vdw_radii = self.model.get_vdw_radii()  # (N_atoms,)
        self.real_space_grid = self.model.real_space_grid
        inv_frac = self.model.inv_fractional_matrix
        frac = self.model.fractional_matrix

        with torch.no_grad():
            spacegroup = self.model.fft.spacegroup
            n_ops = spacegroup.n_ops
            grid_shape = self.real_space_grid.shape[:-1]
            device = self.model.device
            n_atoms = xyz.shape[0]

            # --- Step 1: dilation, chunked over atoms ---
            # Plain-scatter sphere splat (same pattern as the variable-radius CPU
            # splat): cached spherical voxel offsets, fractional voxel positions
            # straight from the integer indices, PBC via
            # `diff_frac - round(diff_frac)`, and r² from a metric-tensor einsum.
            # ATOM_CHUNK caps working memory at ~chunk * N_voxels_in_sphere; 256
            # keeps the peak in the few-hundred-MB range even on the finest
            # grids, where the SF code's 1024 would OOM (denser intermediates).
            ATOM_CHUNK = 256

            grid_dims = torch.tensor(grid_shape, dtype=torch.long, device=device)
            grid_shape_float = grid_dims.float()
            inv_grid = 1.0 / grid_shape_float
            G = frac.T @ frac  # metric tensor: r²_cart = diff_frac · G · diff_frac

            # The offsets are enumerated around the atom's nearest grid NODE, but every
            # distance is then measured from the atom's true position, which sits up to
            # half a voxel diagonal away. The search radius carries that slack so the
            # enumerated set is a guaranteed superset of the voxels the classification
            # below can accept; without it, voxels genuinely inside
            # `vdw + solvent_radius` of the atom fall outside the ball around the node,
            # are never tested, and default to bulk solvent.
            signs = torch.tensor(
                [[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]],
                dtype=frac.dtype,
                device=device,
            )
            half_voxel_diagonal = 0.5 * float(
                ((signs * inv_grid.to(frac.dtype)) @ frac.T).norm(dim=1).max()
            )
            local_offsets = _voxel_offsets_within(
                self.max_radius_angstrom + half_voxel_diagonal,
                grid_dims,
                frac,
                device,
            )  # (R, 3) int — candidate voxels relative to the atom's grid node

            xyz_frac = xyz @ inv_frac.T  # (N, 3)
            xyz_frac_wrapped = xyz_frac % 1.0
            center_idx = torch.round(
                xyz_frac_wrapped * grid_shape_float
            ).long()  # (N, 3)

            protein_chunks = []
            boundary_chunks = []

            for s in range(0, n_atoms, ATOM_CHUNK):
                e = min(s + ATOM_CHUNK, n_atoms)

                # Wrapped voxel indices: (C, R, 3) int
                vi = (
                    center_idx[s:e].unsqueeze(1) + local_offsets.unsqueeze(0)
                ) % grid_dims

                # Direct fractional voxel positions (skip real_space_grid gather)
                voxel_frac = vi.float() * inv_grid  # (C, R, 3)

                # PBC fractional diff
                diff_frac = voxel_frac - xyz_frac[s:e].unsqueeze(1)
                diff_frac = diff_frac - torch.round(diff_frac)
                del voxel_frac

                # Squared Cartesian distance via metric tensor
                r_sq = torch.einsum("avi,ij,avj->av", diff_frac, G, diff_frac)
                del diff_frac

                vdw_c = vdw_radii[s:e]
                vdw_sq = (vdw_c**2).unsqueeze(1)
                rcut_sq = ((vdw_c + self.solvent_radius) ** 2).unsqueeze(1)

                is_protein = r_sq < vdw_sq
                is_boundary = (~is_protein) & (r_sq < rcut_sq)
                del r_sq

                voxel_flat = vi.reshape(-1, 3)
                del vi

                p_idx = is_protein.flatten().nonzero(as_tuple=True)[0]
                b_idx = is_boundary.flatten().nonzero(as_tuple=True)[0]
                del is_protein, is_boundary

                protein_chunks.append(voxel_flat[p_idx])
                boundary_chunks.append(voxel_flat[b_idx])

            protein_voxels = (
                torch.cat(protein_chunks, dim=0)
                if protein_chunks
                else torch.empty((0, 3), dtype=torch.long, device=device)
            )
            boundary_voxels = (
                torch.cat(boundary_chunks, dim=0)
                if boundary_chunks
                else torch.empty((0, 3), dtype=torch.long, device=device)
            )
            del protein_chunks, boundary_chunks

            # --- Step 2: symmetry expansion via index transform ---
            protein_mask = torch.zeros(grid_shape, dtype=torch.bool, device=device)
            boundary_mask = torch.zeros(grid_shape, dtype=torch.bool, device=device)

            float_dtype = get_float_dtype()
            for op_idx in range(n_ops):
                if op_idx == 0:
                    p_idx = protein_voxels
                    b_idx = boundary_voxels
                else:
                    R = spacegroup.matrices[op_idx].to(device=device, dtype=float_dtype)
                    t = spacegroup.translations[op_idx].to(
                        device=device, dtype=float_dtype
                    )
                    gd = grid_dims.to(float_dtype)

                    p_frac = protein_voxels.to(float_dtype) / gd
                    p_idx = (torch.round((p_frac @ R.T + t) * gd) % grid_dims).long()
                    del p_frac

                    b_frac = boundary_voxels.to(float_dtype) / gd
                    b_idx = (torch.round((b_frac @ R.T + t) * gd) % grid_dims).long()
                    del b_frac

                protein_mask[p_idx[:, 0], p_idx[:, 1], p_idx[:, 2]] = True
                boundary_mask[b_idx[:, 0], b_idx[:, 1], b_idx[:, 2]] = True

            boundary_mask = boundary_mask & (~protein_mask)
            definitely_solvent = ~(protein_mask | boundary_mask)

            if self.verbose > 2:
                total_voxels = protein_mask.numel()
                print(
                    f"After symmetry: protein={protein_mask.sum().item()} "
                    f"boundary={boundary_mask.sum().item()} "
                    f"solvent={definitely_solvent.sum().item()} / {total_voxels}"
                )

            # --- Step 3: erosion ---
            # A boundary voxel becomes solvent iff any voxel strictly within
            # `erosion_radius` is bulk solvent: dilate `definitely_solvent` by the
            # spherical structuring element, then intersect with `boundary_mask`.
            # Two paths, dispatched on device, producing the exact same mask because
            # both are built from the same offset list: CPU roll-OR is bandwidth-bound
            # and beats conv3d on a small kernel, while on GPU conv3d fuses to one
            # launch instead of one kernel per offset.
            sphere_offsets = _voxel_offsets_within(
                self.erosion_radius, grid_dims, frac, device, strict=True
            )  # (R, 3) int
            if torch.device(device).type == "cuda":
                half_k = int(sphere_offsets.abs().max().item())
                K = 2 * half_k + 1
                kernel = torch.zeros(
                    (K, K, K), dtype=self.log_k_solvent.dtype, device=device
                )
                ki = sphere_offsets + half_k
                kernel[ki[:, 0], ki[:, 1], ki[:, 2]] = 1.0
                kernel = kernel.view(1, 1, K, K, K)
                solv_float = definitely_solvent.to(self.log_k_solvent.dtype).view(
                    1, 1, *grid_shape
                )
                solv_padded = F.pad(solv_float, (half_k,) * 6, mode="circular")
                neighbour_count = F.conv3d(solv_padded, kernel)
                dilated_solvent = neighbour_count.squeeze(0).squeeze(0) > 0.5
                del solv_float, solv_padded, neighbour_count
            else:
                dilated_solvent = torch.zeros_like(definitely_solvent)
                for i in range(sphere_offsets.shape[0]):
                    dz_, dy_, dx_ = sphere_offsets[i].tolist()
                    dilated_solvent |= torch.roll(
                        definitely_solvent,
                        shifts=(dz_, dy_, dx_),
                        dims=(0, 1, 2),
                    )

            voxels_to_flip = boundary_mask & dilated_solvent
            protein_with_boundary = (protein_mask | boundary_mask) & (~voxels_to_flip)
            solvent_mask = ~protein_with_boundary

            self.register_buffer("solvent_mask", solvent_mask)

            if self.verbose > 1:
                total_voxels = self.solvent_mask.numel()
                n_solv = self.solvent_mask.sum().item()
                print(
                    f"Total solvent voxels: {n_solv} / {total_voxels} "
                    f"({100.0 * n_solv / total_voxels:.2f}%)"
                )

        assert torch.isfinite(
            self.solvent_mask.float()
        ).all(), "Non-finite values in solvent mask"
        return self.solvent_mask

    def update_solvent(self):
        """Rebuild the solvent mask from current coordinates and drop the mask-derived cache.

        Prefer :meth:`~torchref.scaling.scaler_base.ScalerBase.update_solvent`, which also
        clears the scaler's own ``_f_sol_raw``; that one is what ``F_calc`` reads. Calling
        this directly refreshes the mask but leaves the scaler on the old ``F_sol``.
        """
        self.get_solvent_mask()
        # The per-hkl cache is the FFT of the mask, so a new mask invalidates all of it.
        self._cache = TensorDict()

    def get_rec_solvent(self, hkl):
        """
        Compute solvent structure factors.

        Uses the standard crystallographic approach: compute SFs from the solvent mask.
        The mask represents regions where bulk solvent scattering occurs.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices.

        Returns
        -------
        torch.Tensor
            Complex solvent structure factors.
        """

        assert hasattr(
            self, "solvent_mask"
        ), "Solvent mask not computed. Call get_solvent_mask() first."
        mask = self.solvent_mask.to(dtype=self.log_k_solvent.dtype)
        fsol = extract_structure_factor_from_grid(
            ifft(mask, self.model.cell.volume), hkl
        ).detach()
        assert torch.isfinite(
            fsol
        ).all(), "Non-finite values in solvent structure factors"
        return fsol

    def forward(self, hkl, update_fsol=False, F_protein=None):
        """
        Compute solvent contribution to structure factors at given HKL.

        Differentiable w.r.t. ``log_k_solvent``, ``log_ss_half``, ``log_n_exp`` and
        ``phase_offset``. Takes ``f_sol`` (the FFT of the binary mask) from a per-hkl
        cache, applies :meth:`damping` at ``ss = (sin(θ)/λ)**2``,
        blends mask phases toward the protein phases when ``optimize_phase`` and
        ``F_protein`` are both given (``phase_offset`` 0 = mask phases,
        ±π = protein phases), and scales by ``k_solvent``.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices, shape (N, 3).
        update_fsol : bool, default False
            Force recomputation of the cached solvent structure factors for this
            hkl and refresh the cache entry, instead of reusing a cached entry
            keyed on the hkl fingerprint.
        F_protein : torch.Tensor, optional
            Protein structure factors, used for phase blending.

        Returns
        -------
        torch.Tensor
            Complex solvent structure factors, shape (N,).
        """

        # Lightweight fingerprint: (data_ptr, version, numel) — avoids SHA-1
        hkl_key = (hkl.data_ptr(), hkl._version, hkl.numel())

        if not update_fsol and hkl_key in self._cache:
            f_sol = self._cache[hkl_key]
        else:
            f_sol = self.get_rec_solvent(hkl)
            self._cache[hkl_key] = f_sol

        # Calculate scattering vector magnitude: s = sin(θ)/λ
        # Note: get_scattering_vectors returns h* = (h·a*, k·b*, l·c*)
        # For the Debye-Waller factor, we need s = |h*|/2 = sin(θ)/λ
        scattering_vectors = get_scattering_vectors(
            hkl, self.model.cell, recB=self.model.recB
        )
        s = torch.norm(scattering_vectors, dim=1) / 2.0  # This is sin(θ)/λ
        s_squared = s**2  # Now s² is correct for B-factor formula

        falloff = self.damping(s_squared)
        k_solvent = self.k_solvent()

        # Phase handling
        if self.optimize_phase and F_protein is not None:
            f_mask_amp = torch.abs(f_sol)
            mask_phases = torch.angle(f_sol)
            protein_phases = torch.angle(F_protein)

            # Interpolate phases using phase_offset as a blending parameter
            # cos(phase_offset) = 1: use mask phases
            # cos(phase_offset) = -1: use inverted protein phases
            blend_factor = torch.cos(self.phase_offset)
            blended_phase = (
                mask_phases * (1 + blend_factor) / 2
                + (protein_phases + torch.pi) * (1 - blend_factor) / 2
            )

            phase_adjusted_f_sol = f_mask_amp * torch.exp(1j * blended_phase)
        elif self.optimize_phase:
            # Apply global phase offset
            phase_adjusted_f_sol = f_sol * torch.exp(1j * self.phase_offset)
        else:
            # No phase adjustment - use mask phases as-is
            phase_adjusted_f_sol = f_sol

        f_solvent = k_solvent * phase_adjusted_f_sol * falloff

        assert torch.isfinite(
            f_solvent
        ).all(), "Non-finite values in solvent structure factors"
        return f_solvent

    def parameters(self):
        """Refinable solvent parameters as a list (phase offset only if refined)."""
        return [self.log_k_solvent, self.log_ss_half, self.log_n_exp] + (
            [self.phase_offset] if self.optimize_phase else []
        )
