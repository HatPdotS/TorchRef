"""
Density-derived bulk-solvent model.

A standalone, fully differentiable alternative to the vdW-sphere
:class:`~torchref.scaling.solvent.SolventModel`. Instead of building a binary
mask from atomic spheres, this module derives the solvent occupancy directly from
the model's real-space electron density ``rho(x)`` and returns a solvent structure
factor on the same absolute (electron) scale as the protein structure factors.

With the natural exponential occupancy the solvent contribution is::

    M(x)       = exp(-rho(x) / rho0)          # solvent occupancy in (0, 1]
    rho_sol(x) = rho_s * M(x)                 # solvent density field, e/A^3
    F_sol(h)   = rho_s * ifft(M, V_cell)|_hkl # same FFT path as F_protein

The ``rho0`` saturation level interpolates the Babinet/exponential limit
(``rho0 -> inf``) and the flat-mask limit (``rho0 -> 0``). An optional residual
``B_sol`` (default off) applies ``exp(-B_sol * s^2)`` damping to F_sol.

The occupancy is nonlinear, so the full-cell density is assembled in real space
before the mask. Bulk solvent is low-resolution, so the module owns a coarse
``SfFFT`` grid sized to ``solvent_res`` (~4 A); for hkl beyond the coarse Nyquist
the solvent contribution is returned as exactly 0.
"""

import torch
import torch.nn as nn

from torchref.base import (
    extract_structure_factor_from_grid,
    get_scattering_vectors,
    ifft,
)
from torchref.config import get_default_device, get_float_dtype
from torchref.model.context import ModelContext
from torchref.model.sf_fft import SfFFT
from torchref.utils.debug_utils import DebugMixin
from torchref.utils.device_mixin import DeviceMixin
from torchref.utils.utils import TensorDict, ModuleReference


class DensitySolventModel(DeviceMixin, DebugMixin, nn.Module):
    """
    Differentiable bulk-solvent model derived from the model electron density.

    Supports the same two initialization patterns as
    :class:`~torchref.scaling.solvent.SolventModel`:

    1. Empty initialization (for ``state_dict`` loading)::

        solvent = DensitySolventModel()
        solvent.load_state_dict(torch.load("density_solvent.pt"))

    2. Full initialization with a model::

        solvent = DensitySolventModel(model, rho_s=0.34, rho0=2.0)
        f_sol = solvent(hkl)

    Parameters
    ----------
    model : ModelFT, optional
        Atomic model providing the atoms, cell and spacegroup. Optional for
        empty init.
    rho_s : float, default 0.34
        Initial bulk solvent electron density (e/A^3). Note: the
        :class:`~torchref.experimental.monolithic_refinement.density_scaler.DensityDerivedSolvent`
        wrapper freezes ``rho_s`` at 1.0 and uses ``rho0=0.016`` instead.
    rho0 : float, default 2.0
        Initial protein-density saturation level (e/A^3). Interpolates Babinet
        (large) <-> flat-mask (small). Note: the
        :class:`~torchref.experimental.monolithic_refinement.density_scaler.DensityDerivedSolvent`
        wrapper instead uses ``rho0=0.016`` (the sharp-mask regime).
    occupancy : {"exp", "sigmoid", "shell"}, default "exp"
        Occupancy function mapping density -> solvent fraction. ``"exp"`` uses
        ``M = exp(-rho/rho0)``. ``"sigmoid"`` uses ``M = sigmoid((rho0 - rho)/w)``
        with fixed edge width ``sigmoid_width``. ``"shell"`` smooths the density
        (Gaussian width ``shell_sigma``), standardizes it, and thresholds with a
        sigmoid at z-cutoff ``shell_tau``: ``M = 1 - sigmoid((z - tau)/w)``.
    sigmoid_width : float, default 0.5
        Edge width ``w`` (e/A^3) for the ``"sigmoid"`` occupancy; ignored for
        ``"exp"``/``"shell"``. Not refined.
    shell_sigma : float, default 1.5
        Initial smoothing width (A) for the ``"shell"`` occupancy; refinable
        (clamped to [0, 5] A). Ignored for the other modes.
    shell_tau : float, optional
        z-cutoff for the ``"shell"`` threshold (refinable). If None (default), it
        is lazily initialised from the ``shell_quantile`` quantile of the
        standardized smoothed density on the first build.
    shell_quantile : float, default 0.5
        Quantile used to initialise ``shell_tau`` (~ the starting solvent voxel
        fraction). Ignored if ``shell_tau`` is given.
    shell_edge : float, default 0.5
        Sigmoid edge width ``w`` (in standardized-density / z units) for the
        ``"shell"`` threshold. Not refined.
    residual_bsol : bool, default False
        If True, add a refinable scalar ``b_solvent`` (init 0) applying an
        ``exp(-B_sol * s^2)`` residual damping to F_sol.
    solvent_res : float, optional
        Resolution (A) sizing the coarse solvent grid. Lower = finer/slower.
        Defaults to 4.0 for ``"exp"``/``"sigmoid"`` and 2.0 for ``"shell"`` (the
        finer grid resolves the depletion shell). Pass an explicit value to override.
    verbose : int, default 1
        Verbosity level.
    float_type : torch.dtype, default: configured float dtype
        Floating point data type.
    device : torch.device, default: configured device
        Device for tensor operations.
    """

    def __init__(
        self,
        model=None,
        rho_s=0.34,
        rho0=2.0,
        occupancy="exp",
        sigmoid_width=0.5,
        shell_sigma=1.5,
        shell_tau=None,
        shell_quantile=0.5,
        shell_edge=0.5,
        residual_bsol=False,
        solvent_res=None,
        verbose=1,
        float_type=get_float_dtype(),
        device=get_default_device(),
    ):
        super(DensitySolventModel, self).__init__()
        if occupancy not in ("exp", "sigmoid", "shell"):
            raise ValueError(
                f"occupancy must be 'exp', 'sigmoid' or 'shell', got {occupancy!r}"
            )
        self.device = device
        self.verbose = verbose
        self.float_type = float_type
        self.occupancy = occupancy
        self.sigmoid_width = float(sigmoid_width)
        self.shell_quantile = float(shell_quantile)
        self.shell_edge = float(shell_edge)
        self.residual_bsol = residual_bsol
        # "shell" needs a finer grid than "exp"/"sigmoid" to resolve the ~1-2 A
        # depletion shell; fall back to the coarse default for the other modes.
        if solvent_res is None:
            solvent_res = 2.0 if occupancy == "shell" else 4.0
        self.solvent_res = float(solvent_res)
        self._cache = TensorDict()

        # Refinable parameters live in log-space for positivity, mirroring
        # SolventModel.log_k_solvent.
        self.log_rho_s = nn.Parameter(
            torch.log(torch.tensor(rho_s, dtype=float_type, device=device))
        )
        self.log_rho0 = nn.Parameter(
            torch.log(torch.tensor(rho0, dtype=float_type, device=device))
        )
        if residual_bsol:
            self.b_solvent = nn.Parameter(
                torch.tensor(0.0, dtype=float_type, device=device)
            )
        else:
            self.register_buffer(
                "b_solvent", torch.tensor(0.0, dtype=float_type, device=device)
            )

        # "shell" mode is a morphological operator: smooth the density (width
        # sigma, Angstroms; stored linearly so sigma=0 = no smoothing) then
        # threshold the standardized smoothed density with a sigmoid at z-cutoff
        # tau (the single erosion/dilation knob). tau is lazily initialised from
        # a quantile of the standardized smoothed density on the first build,
        # unless an explicit shell_tau is given.
        if occupancy == "shell":
            self.sigma_shell = nn.Parameter(
                torch.tensor(float(shell_sigma), dtype=float_type, device=device)
            )
            self.tau_shell = nn.Parameter(
                torch.tensor(
                    0.0 if shell_tau is None else float(shell_tau),
                    dtype=float_type,
                    device=device,
                )
            )
            self.register_buffer(
                "_tau_initialized",
                torch.tensor(shell_tau is not None, device=device),
            )

        # Empty init: shell ready for load_state_dict().
        if model is None:
            self.model = None
            self.solvent_fft = None
            return

        self.model = ModuleReference(model)
        assert self.model, "Model is not initialized"

        # Dedicated COARSE grid for the solvent. Early (real-space) symmetry,
        # because the nonlinear occupancy needs the full-cell density assembled
        # before the mask. The per-atom splat radius is governed by
        # torchref.sigma_cutoff_ed inside the density builder.
        # Its own context: the cell is shared with the model, but the space group
        # is copied because it memoises operators per grid shape and this engine's
        # coarse grid must not evict the model's.
        solvent_ctx = ModelContext(
            cell=model.cell, spacegroup=model.spacegroup.copy()
        )
        self.solvent_fft = SfFFT(
            ctx=solvent_ctx,
            max_res=self.solvent_res,
            dtype_float=float_type,
            device=device,
            verbose=max(0, verbose - 1),
            use_late_symmetry=False,
        )

    # ------------------------------------------------------------------
    # Density -> occupancy -> structure factor
    # ------------------------------------------------------------------
    def get_density(self):
        """
        Build the symmetry-expanded, full-cell density on the coarse solvent grid,
        in the autograd graph (differentiable w.r.t. atomic xyz / B / occupancy).

        Returns
        -------
        torch.Tensor
            Real-space density of shape (ncx, ncy, ncz).
        """
        m = self.model
        xyz_iso, adp_iso, occ_iso, A_iso, B_iso = m.get_iso()
        xyz_aniso, u_aniso, occ_aniso, A_aniso, B_aniso = m.get_aniso()
        has_aniso = len(xyz_aniso) > 0
        return self.solvent_fft.build_density_map(
            xyz_iso=xyz_iso,
            adp_iso=adp_iso,
            occ_iso=occ_iso,
            A_iso=A_iso,
            B_iso=B_iso,
            xyz_aniso=xyz_aniso if has_aniso else None,
            u_aniso=u_aniso if has_aniso else None,
            occ_aniso=occ_aniso if has_aniso else None,
            A_aniso=A_aniso if has_aniso else None,
            B_aniso=B_aniso if has_aniso else None,
            apply_symmetry=True,
        )

    def mask_from_density(self, rho):
        """
        Map electron density to solvent occupancy ``M(x) in [0, 1]``.

        ``"exp"``: ``M = exp(-rho / rho0)`` -- 1 in bulk (rho=0), ->0 in cores.
        ``"sigmoid"``: ``M = sigmoid((rho0 - rho) / w)``.
        ``"shell"``: smooth the density in reciprocal space (Gaussian width
            ``sigma_shell`` A), standardize it, and threshold with a sigmoid at
            z-cutoff ``tau_shell``: ``M = 1 - sigmoid((z - tau) / w)`` with
            ``z = (rho - mu) / sd`` of the smoothed density. ``tau_shell`` moves the
            boundary, ``sigma_shell`` sets the reach, and ``w`` (``shell_edge``) the
            edge sharpness.
        """
        rho0 = torch.exp(self.log_rho0.clamp(min=-20.0, max=20.0))
        if self.occupancy == "exp":
            # rho >= 0 from real-space Gaussian accumulation; clamp guards ripples.
            return torch.exp(-(rho.clamp(min=0.0)) / rho0)
        if self.occupancy == "shell":
            rho_smooth = self._smooth(rho.clamp(min=0.0))
            # Standardize (mu/sd detached: scale carried by rho_s / the scaler;
            # gradients to coords flow through the numerator). z-cutoff makes tau
            # and the edge width portable across structures (SFcalculator-style).
            mu = rho_smooth.mean().detach()
            sd = rho_smooth.std().detach().clamp(min=1e-6)
            z = (rho_smooth - mu) / sd
            tau = self._get_tau(z)
            w = max(self.shell_edge, 1e-3)
            return (1.0 - torch.sigmoid((z - tau) / w)).clamp(min=0.0, max=1.0)
        w = self.sigmoid_width
        return torch.sigmoid((rho0 - rho) / w)

    def _smooth(self, field):
        """
        Smooth a real-space ``field`` with a normalized Gaussian of width
        ``sigma_shell`` (Angstroms), via reciprocal-space convolution.

        ``out = real(ifftn(fftn(field) * Khat))`` with
        ``Khat(f) = exp(-2*pi^2 * sigma^2 * |f|^2)`` on the full fft grid using
        physical frequencies ``fftfreq(N, d=voxel_spacing)`` per axis. ``Khat(0)
        = 1`` so the smoothing preserves the DC component (a free "B" multiply).
        At ``sigma=0`` the kernel is the identity (no smoothing). Differentiable
        w.r.t. ``sigma_shell`` and -- through ``field`` -- w.r.t. atomic xyz/B.
        """
        # Axis spacings: column j of the fractional (frac->cart) matrix is cell edge
        # vector j, so its norm over the sampling count along that axis is the step
        # between grid points one index apart -- what differencing the coordinate grid
        # used to measure, without building the grid.
        frac = self.solvent_fft.cell.fractional_matrix
        n = self.solvent_fft.grid_shape
        dx, dy, dz = (float(frac[:, j].norm()) / n[j] for j in range(3))
        sigma = self.sigma_shell.clamp(min=0.0, max=5.0)
        nx, ny, nz = field.shape
        two_pi2 = 2.0 * torch.pi**2
        fx = torch.fft.fftfreq(nx, d=dx, device=field.device, dtype=field.dtype)
        fy = torch.fft.fftfreq(ny, d=dy, device=field.device, dtype=field.dtype)
        fz = torch.fft.fftfreq(nz, d=dz, device=field.device, dtype=field.dtype)
        gx = torch.exp(-two_pi2 * sigma**2 * fx**2)
        gy = torch.exp(-two_pi2 * sigma**2 * fy**2)
        gz = torch.exp(-two_pi2 * sigma**2 * fz**2)
        G = gx[:, None, None] * gy[None, :, None] * gz[None, None, :]
        F = torch.fft.fftn(field)
        return torch.real(torch.fft.ifftn(F * G.to(F.dtype)))

    def _get_tau(self, z):
        """
        Return the z-cutoff ``tau_shell``, lazily initialising it on first use
        from the ``shell_quantile`` quantile of the standardized smoothed density
        ``z`` (so a fraction ~``shell_quantile`` of voxels start as solvent).
        The init is a no-grad data write guarded by a persistent buffer, so a
        value loaded via ``load_state_dict`` is never overwritten.
        """
        if not bool(self._tau_initialized):
            with torch.no_grad():
                flat = z.flatten()
                stride = max(1, flat.numel() // 1_000_000)
                q = torch.quantile(flat[::stride], self.shell_quantile)
                self.tau_shell.copy_(q)
                self._tau_initialized.fill_(True)
        return self.tau_shell.clamp(min=-10.0, max=10.0)

    def _nyquist_mask(self, hkl):
        """True where |h|,|k|,|l| are within the coarse grid Nyquist limit."""
        nc = self.solvent_fft.grid_shape  # (ncx, ncy, ncz)
        nyq = torch.tensor(
            [n // 2 for n in nc], device=hkl.device, dtype=hkl.dtype
        )
        return (hkl.abs() <= nyq).all(dim=1)

    def forward(self, hkl, rho=None):
        """
        Compute solvent structure factors at the given Miller indices.

        Differentiable w.r.t. ``log_rho_s``, ``log_rho0`` (and ``b_solvent`` if
        enabled), and -- through ``rho`` -- w.r.t. atomic coordinates and
        B-factors.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices, shape (N, 3).
        rho : torch.Tensor, optional
            Precomputed coarse-grid density. If None, built via
            :meth:`get_density`.

        Returns
        -------
        torch.Tensor
            Complex solvent structure factors, shape (N,), on the same absolute
            scale as the protein structure factors. Exactly 0 for hkl beyond the
            coarse-grid Nyquist (negligible solvent contribution there).
        """
        if rho is None:
            rho = self.get_density()

        M = self.mask_from_density(rho)
        rho_s = torch.exp(self.log_rho_s.clamp(min=-20.0, max=20.0))

        # Same FFT normalization + extraction as the protein SF path and as
        # SolventModel.get_rec_solvent -> identical absolute scale.
        f_mask = extract_structure_factor_from_grid(
            ifft(M, self.model.cell.volume), hkl
        )
        f_sol = rho_s * f_mask

        # Zero the aliased high-res tail (solvent ~0 beyond coarse Nyquist).
        valid = self._nyquist_mask(hkl)
        f_sol = torch.where(valid, f_sol, torch.zeros_like(f_sol))

        # Optional residual B_sol damping: exp(-B_sol * s^2), s = sin(theta)/lambda.
        if self.residual_bsol:
            scattering_vectors = get_scattering_vectors(
                hkl, self.model.cell, recB=self.model.recB
            )
            s_sq = (torch.norm(scattering_vectors, dim=1) / 2.0) ** 2
            exp = (-self.b_solvent.clamp(min=-500.0, max=500.0) * s_sq).clamp(
                min=-50.0, max=50.0
            )
            f_sol = f_sol * torch.exp(exp)

        assert torch.isfinite(
            f_sol
        ).all(), "Non-finite values in solvent structure factors"
        return f_sol

    @property
    def rho_s(self):
        """Current bulk solvent electron density (e/A^3)."""
        return torch.exp(self.log_rho_s).detach()

    @property
    def rho0(self):
        """Current protein-density saturation level (e/A^3)."""
        return torch.exp(self.log_rho0).detach()

    def parameters(self, recurse=True):
        """Owned refinable leaves, so a host optimizer can co-refine them."""
        leaves = [self.log_rho_s, self.log_rho0]
        if self.occupancy == "shell":
            leaves.append(self.sigma_shell)
            leaves.append(self.tau_shell)
        if self.residual_bsol:
            leaves.append(self.b_solvent)
        return iter(leaves)
