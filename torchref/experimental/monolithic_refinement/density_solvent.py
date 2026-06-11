"""
Density-derived bulk-solvent model.

A standalone, fully differentiable alternative to the vdW-sphere
:class:`~torchref.scaling.solvent.SolventModel`. Instead of building a binary
mask from atomic spheres (under ``torch.no_grad()`` and then detaching the
result), this module derives the solvent occupancy directly from the model's
real-space electron density ``rho(x)`` and returns a solvent structure factor on
the same absolute (electron) scale as the protein structure factors.

Physical premise
----------------
Crystallographic B-factors mostly model *inter-copy disorder*, not thermal
vibration, so the model density ``rho(x)`` is an ensemble-averaged density. The
bulk solvent fills the complement of where the (disordered) protein is, and its
boundary softness is *inherited* from ``rho`` itself -- fuzzy where the protein is
disordered (high B), sharp where it is ordered. A single global B_sol cannot
represent that; deriving the mask from ``rho`` does it for free, with spatially
varying edge softness and no separate sharpness knob.

Formulation
-----------
With the natural exponential occupancy the solvent mask is a one-liner::

    M(x)      = exp(-rho(x) / rho0)          # solvent occupancy in (0, 1]
    rho_sol(x)= rho_s * M(x)                 # solvent density field, e/A^3
    F_sol(h)  = rho_s * ifft(M, V_cell)|_hkl # same FFT path as F_protein

Two owned, physical parameters:

``rho_s``
    Bulk solvent electron density / protein-solvent contrast (init ~0.34 e/A^3).
    Scales solvent *relative* to protein, so it is identifiable and distinct from
    the scaler's overall scale K (which stays shared between protein and solvent).

``rho0``
    Protein-density saturation level. A single master knob that continuously
    interpolates the two classic bulk-solvent models:

    * ``rho0 -> inf`` (``M ~ 1 - rho/rho0``):
      ``F_sol = -(rho_s/rho0) * F_protein`` -- the Babinet / exponential model.
    * ``rho0 -> 0`` (``M -> indicator of rho ~ 0``):
      ``F_sol = -rho_s * FFT(envelope)`` -- the flat-mask model.

An optional scalar residual ``B_sol`` (default off) damps F_sol with
``exp(-B_sol * s^2)`` to absorb first-hydration-shell smearing the protein
B-factors do not capture; it is now identifiable because the edge is otherwise
pinned by ``rho``.

Cost
----
The occupancy is nonlinear, so the full-cell density must be assembled in real
space before the mask (reciprocal-space symmetry expansion does not commute with
``exp``). But bulk solvent is a *low-resolution* object, so it does not need the
atomic ``d_min/3`` grid: the module owns its own **coarse** ``SfFFT`` grid sized
to ``solvent_res`` (~4 A) and builds the symmetry-expanded density there. Both the
symmetry expansion and the FFT then cost ~(spacing ratio)^3 less than on the
atomic grid. For hkl beyond the coarse Nyquist the solvent contribution is ~0 and
is returned as exactly 0 (avoids modulo-aliasing in the SF extraction).

This module is purely additive: it does not touch ``SolventModel``, ``Scaler`` or
the refinement loop. It is intended to be A/B'd against the current solvent first
(see ``paper/figure2_alphafold_start/analysis/solvent_density_probe.py``).
"""

import torch
import torch.nn as nn

from torchref.base import (
    extract_structure_factor_from_grid,
    get_scattering_vectors,
    ifft,
)
from torchref.config import get_default_device, get_float_dtype
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
        Initial bulk solvent electron density (e/A^3).
    rho0 : float, default 2.0
        Initial protein-density saturation level (e/A^3). Interpolates Babinet
        (large) <-> flat-mask (small).
    occupancy : {"exp", "sigmoid"}, default "exp"
        Occupancy function mapping density -> solvent fraction. ``"exp"`` uses
        ``M = exp(-rho/rho0)`` (single knob, clean Babinet/mask limits).
        ``"sigmoid"`` uses ``M = sigmoid((rho0 - rho)/w)`` (contour at ``rho0``,
        fixed edge width ``sigmoid_width``).
    sigmoid_width : float, default 0.5
        Edge width ``w`` (e/A^3) for the ``"sigmoid"`` occupancy; ignored for
        ``"exp"``. Not refined (the edge is meant to come from ``rho``).
    residual_bsol : bool, default False
        If True, add a refinable scalar ``b_solvent`` (init 0) applying an
        ``exp(-B_sol * s^2)`` residual damping to F_sol.
    solvent_res : float, default 4.0
        Resolution (A) sizing the coarse solvent grid. Lower = finer/slower.
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
        residual_bsol=False,
        solvent_res=4.0,
        verbose=1,
        float_type=get_float_dtype(),
        device=get_default_device(),
    ):
        super(DensitySolventModel, self).__init__()
        if occupancy not in ("exp", "sigmoid"):
            raise ValueError(
                f"occupancy must be 'exp' or 'sigmoid', got {occupancy!r}"
            )
        self.device = device
        self.verbose = verbose
        self.float_type = float_type
        self.occupancy = occupancy
        self.sigmoid_width = float(sigmoid_width)
        self.residual_bsol = residual_bsol
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

        # Empty init: shell ready for load_state_dict().
        if model is None:
            self.model = None
            self.solvent_fft = None
            return

        self.model = ModuleReference(model)
        assert self.model, "Model is not initialized"

        # Dedicated COARSE grid for the solvent. Early (real-space) symmetry,
        # because the nonlinear occupancy needs the full-cell density assembled
        # before the mask. radius_angstrom matches the model's density build.
        radius = getattr(model, "radius_angstrom", 3.0)
        self.solvent_fft = SfFFT(
            cell=model.cell,
            spacegroup=model.fft.spacegroup,
            max_res=self.solvent_res,
            radius_angstrom=radius,
            dtype_float=float_type,
            device=device,
            verbose=max(0, verbose - 1),
            use_late_symmetry=False,
        )
        self.solvent_fft.setup_grid()

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
        """
        rho0 = torch.exp(self.log_rho0.clamp(min=-20.0, max=20.0))
        if self.occupancy == "exp":
            # rho >= 0 from real-space Gaussian accumulation; clamp guards ripples.
            return torch.exp(-(rho.clamp(min=0.0)) / rho0)
        w = self.sigmoid_width
        return torch.sigmoid((rho0 - rho) / w)

    def _nyquist_mask(self, hkl):
        """True where |h|,|k|,|l| are within the coarse grid Nyquist limit."""
        nc = self.solvent_fft.real_space_grid.shape[:-1]  # (ncx, ncy, ncz)
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
        if self.residual_bsol:
            leaves.append(self.b_solvent)
        return iter(leaves)
