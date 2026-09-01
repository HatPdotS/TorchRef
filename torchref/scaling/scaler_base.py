"""
Base scaler class for crystallographic scaling without model dependency.

:class:`ScalerBase` holds no reference to a ``Model``: every method needing calculated
structure factors takes ``F_calc`` as an argument, so any source will do (molecular
replacement, precomputed factors, a custom model).
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from torchref.base.math_torch import U_to_matrix
from torchref.base.metrics import (
    binwise_scale,
    nll_xray,
    nll_xray_lognormal,
    nll_xray_mean,
    rfactor_work_free,
)
from torchref.base.reciprocal import get_scattering_vectors
from torchref.config import get_complex_dtype, get_float_dtype
from torchref.utils.autograd_ops import gather_with_index_add
from torchref.utils.debug_utils import DebugMixin
from torchref.utils.device_mixin import DeviceMixin
from torchref.utils.device_resolution import resolve_device
from torchref.scaling.solvent import SS_HALF_BOUNDS
from torchref.utils.utils import ModuleReference

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.scaling.solvent import SolventModel


#: Selectable objectives for the scaler's own L-BFGS scale fit. **Both are rows of
#: :data:`~torchref.refinement.targets.xray._specs.XRAY_TARGETS`**, not a private enum: the
#: scale fit and the body refinement evaluate the same likelihood code, and differ only in
#: which row they pick. Neither row may centre on ``alpha`` -- see
#: :meth:`ScalerBase.refine_lbfgs`.
SCALE_TARGETS = ("nll", "ml_noalpha", "ls")

#: The default scale-fit objective, defined ONCE: the constructors, the
#: ``getattr`` fallbacks and the CLI's ``default=``/``choices=`` must all read it
#: from here rather than repeating the string.
#:
#: ``ls`` weights every reflection equally, matching how R itself weights them; ``nll``
#: divides each residual by ``sigma**2`` and so up-weights weak reflections relative to
#: their contribution to R.
DEFAULT_SCALE_TARGET = "ls"


class ScalerBase(DeviceMixin, DebugMixin, nn.Module):
    """
    Base scaler class for crystallographic scaling without model dependency.

    Construct either fully (``ScalerBase(data=..., nbins=20)`` then ``initialize(fcalc)``)
    or empty (``ScalerBase()`` then ``load_state_dict``). Note that ``c_iso``, ``U``
    and ``solvent`` do **not** exist until ``initialize()`` / ``set_solvent_model()`` runs
    -- :meth:`forward` tests for each with ``hasattr`` and silently skips the missing ones,
    so an un-initialized scaler is an identity transform rather than an error.

    The isotropic scale is a Chebyshev polynomial in ``sqrt(s_half_sq)``, evaluated per
    reflection: ``k_iso = exp(sum_i c_iso[i] T_i(u))``. Every reflection contributes to
    every coefficient with a continuous weight, so there are no bin boundaries and nothing
    changes discontinuously when a reflection moves between shells. Bins survive only as
    the device that seeds ``c_iso`` (:meth:`calc_initial_scale`) and as the sizing oracle
    for :meth:`forward`'s masking test.

    Parameters
    ----------
    data : ReflectionData, optional
        ReflectionData object with observed data.
    nbins : int, default 20
        Number of resolution bins used to seed the scale and to size the per-reflection
        masking test. Overwritten by the binner's actual count.
    n_iso_coeff : int, default 6
        Number of Chebyshev terms in the isotropic scale. ``1`` collapses it to a single
        global constant, since ``T_0 = 1``.
    verbose : int, default 1
        Verbosity level.
    device : torch.device, optional
        Computation device; defaults to ``data``'s, else the configured default. An
        explicit value moves ``data``.

    Attributes
    ----------
    c_iso : torch.nn.Parameter
        Chebyshev coefficients of the isotropic log scale.
    U : torch.nn.Parameter
        Anisotropic scaling parameters, as the 6 unique components.
    solvent : SolventModel
        Bulk-solvent model.
    cell, bins, s
        Cell, per-reflection bin indices and scattering vectors, from the data.
    _iso_design : torch.Tensor
        ``(N_reflections, n_iso_coeff)`` Chebyshev design matrix for ``c_iso``. Derived
        from the reflection set, so it is rebuilt rather than serialised.
    """

    def __init__(
        self,
        data: Optional["ReflectionData"] = None,
        nbins: int = 20,
        n_iso_coeff: int = 6,
        verbose: int = 1,
        device: Optional[torch.device] = None,
    ):
        """See the class docstring. ``data=None`` builds an empty shell for
        ``load_state_dict``; an explicit ``device`` forces ``data`` onto it."""
        super(ScalerBase, self).__init__()
        self.device = resolve_device(data, device=device)
        self.verbose = verbose
        self.nbins = nbins
        self.n_iso_coeff = n_iso_coeff

        # Empty shell: configuration only, ready for load_state_dict().
        if data is None:
            self._data = None
            self.cell = None
            self.register_buffer("s", None)
            self.register_buffer("bins", None)
            self.register_buffer("_s_half_sq", None)
            # Non-persistent: derived from ``_s_half_sq`` and valid only for THIS
            # reflection set, so a checkpoint rebuilds it rather than carrying a design
            # matrix belonging to different data.
            self.register_buffer("_iso_design", None, persistent=False)
            self._f_sol_raw = None
            return

        self.to(self.device)
        self._data = ModuleReference(data)

        self.cell = data.cell
        s = get_scattering_vectors(data.hkl, self.cell)
        self.register_buffer("s", s)
        # Precompute (sin(θ)/λ)^2 for B-factor damping — avoids recomputing per call
        self.register_buffer("_s_half_sq", (torch.norm(s, dim=1) / 2.0) ** 2)
        self.register_buffer("_iso_design", self._build_iso_design(), persistent=False)
        self._f_sol_raw = None
        bins, self.nbins = self._data.get_bins(self.nbins)
        self.register_buffer("bins", bins)
        if self.verbose > 0:
            print(f"Initialized ScalerBase with {self.nbins} bins.")

    def _build_iso_design(self) -> torch.Tensor:
        """``(N, n_iso_coeff)`` Chebyshev design matrix for the isotropic scale.

        The abscissa is ``sqrt(s_half_sq)``, i.e. ``sin(theta)/lambda``, mapped onto
        ``[-1, 1]``. That coordinate rather than ``s**2`` because the modulation is gentle
        through the bulk of the resolution range but has real structure in the first few
        percent of ``s**2``; a basis uniform in ``s**2`` spends nearly all its resolution
        where nothing happens.
        """
        x = torch.sqrt(self._s_half_sq.clamp(min=0))
        lo, hi = x.min(), x.max()
        u = (2 * (x - lo) / (hi - lo).clamp(min=1e-12) - 1).clamp(-1.0, 1.0)
        cols = [torch.ones_like(u), u]
        for _ in range(2, self.n_iso_coeff):
            cols.append(2 * u * cols[-1] - cols[-2])  # Chebyshev recurrence
        return torch.stack(cols[: self.n_iso_coeff], dim=1)

    def iso_log_scale(self, design: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-reflection isotropic log scale ``design @ c_iso``, clamped to ``[-10, 10]``.

        The clamp is per reflection, not on the coefficients: a polynomial is unbounded at
        the ends of its range, so without it a single extreme reflection can carry an
        arbitrarily large scale.
        """
        if design is None:
            design = self._iso_design
        return (design @ self.c_iso).clamp(min=-10.0, max=10.0)

    def set_data(self, data: "ReflectionData"):
        """
        Reconnect a data object after an empty init or a ``load_state_dict``.

        Receiver wins: ``data`` is moved onto *this scaler's* device, since the scaler may
        already hold buffers. Buffers that already exist are left alone -- only ``s``,
        ``_s_half_sq`` and ``bins`` that are still ``None`` get built.

        Parameters
        ----------
        data : ReflectionData
            ReflectionData object with observed data.
        """
        self.device = resolve_device(self, data)
        self._data = ModuleReference(data)
        if data.cell is not None:
            self.cell = data.cell
        if self.s is None and data.hkl is not None and data.cell is not None:
            s = get_scattering_vectors(data.hkl, data.cell)
            self.register_buffer("s", s)
            self.register_buffer("_s_half_sq", (torch.norm(s, dim=1) / 2.0) ** 2)
        if getattr(self, "_iso_design", None) is None and self._s_half_sq is not None:
            self.register_buffer(
                "_iso_design", self._build_iso_design(), persistent=False
            )
        self._f_sol_raw = None
        if self.bins is None and data.hkl is not None:
            bins, self.nbins = self._data.get_bins(self.nbins)
            self.register_buffer("bins", bins)

    def initialize(self, fcalc: torch.Tensor):
        """Create ``c_iso`` and ``U`` from the complex ``fcalc``."""
        self.calc_initial_scale(fcalc)
        self.setup_anisotropy_correction()

    @property
    def hkl(self):
        """Get HKL indices from data."""
        return self._data.hkl

    def calc_initial_scale(self, fcalc: torch.Tensor):
        """
        Seed ``c_iso`` from the closed-form observed/calculated amplitude ratio.

        The ratio is evaluated per bin, then projected onto the Chebyshev basis by least
        squares, so the fit starts from the same scale curve the binned model would have
        started from. Fitted on the **work set only**, excluding negative-intensity
        reflections whose French-Wilson F values are biased.

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors (complex). Asserted finite.

        Returns
        -------
        torch.nn.Parameter
            The Chebyshev coefficient parameter ``c_iso``.
        """
        fobs = self._data.get_corrected_data()[0]
        if self.verbose > 0:
            print(f"Calculating initial scale factors using {self.nbins} bins.")
        assert torch.all(
            torch.isfinite(fcalc)
        ), "Non-finite values found in fcalc during initial scale calculation."

        fcalc_amp = torch.abs(fcalc).to(fobs.dtype)

        # Negative intensities carry French-Wilson-biased F values.
        if hasattr(self._data, "I") and self._data.I is not None:
            positive_mask = self._data.I > 0
            if self.verbose > 1:
                n_excluded = (~positive_mask).sum().item()
                print(
                    f"Excluding {n_excluded} negative intensity reflections from scale calculation"
                )
        else:
            positive_mask = torch.ones_like(fobs, dtype=torch.bool)

        # Fit on the work set, positive-intensity reflections only.
        mask = (self._data.work.mask & positive_mask).to(torch.bool)
        c = binwise_scale(
            fcalc_amp,
            fobs,
            self.bins,
            valid=mask,
            nbins=self.nbins,
        ).to(self.device)
        initial_log_scale = torch.log(c.clamp(min=1e-6))
        if self.verbose > 1:
            print(
                "Initial scale factors per bin:",
                initial_log_scale.detach().cpu().numpy(),
            )
        with torch.no_grad():
            target = initial_log_scale.detach().to(self.device)[self.bins.to(torch.int64)]  # dtype-ok: bin indices for advanced indexing; PyTorch requires int64
            design = self._iso_design.to(target.dtype)
            coeff = torch.linalg.lstsq(design, target.unsqueeze(1)).solution.squeeze(1)
        self.c_iso = nn.Parameter(coeff.detach().to(self.device))
        return self.c_iso

    def setup_anisotropy_correction(self):
        """Initialize anisotropic correction parameters."""
        self.U = nn.Parameter(
            torch.normal(0, 0.001, (6,), dtype=get_float_dtype(), device=self.device)
        )

    def anisotropy_correction(self):
        """Per-reflection anisotropic correction ``exp(-2 pi^2 s^T U s)``, clamped."""
        U = U_to_matrix(self.U)
        # matmul + multiply + sum, not einsum: much faster for this bilinear form on CPU.
        sU = torch.matmul(self.s, U)  # (N, 3)
        exp = -2 * torch.pi**2 * (sU * self.s).sum(dim=1)
        return torch.exp(exp.clamp(max=10.0, min=-10.0))

    def set_solvent_model(self, solvent_model: "SolventModel") -> None:
        """
        Attach a pre-configured :class:`SolventModel` and invalidate the ``F_sol`` cache.

        The solvent model must be built externally (it needs a ``Model``).

        Parameters
        ----------
        solvent_model : SolventModel
            Solvent model that can compute solvent structure factors.
        """
        self.solvent = solvent_model
        self._f_sol_raw = None  # Invalidate cached raw solvent SFs

    def update_solvent(self) -> None:
        """Rebuild the solvent mask at the current coordinates and drop the cached ``F_sol``.

        Both halves belong to one operation. :meth:`forward` recomputes ``_f_sol_raw`` only
        when it is ``None``, so rebuilding the mask without clearing the cache leaves
        ``F_calc`` on the mask from whenever the cache was last filled -- on a macrocycle
        loop, the starting coordinates.

        No-op when there is no solvent model, so callers do not need to guard.
        """
        if getattr(self, "solvent", None) is None:
            return
        self.solvent.update_solvent()
        self._f_sol_raw = None

    def setup_binwise_solvent_scale(self):
        """
        Create ``log_kmask``, a per-bin solvent scale (Phenix-style kmask).

        Once this exists, :meth:`forward` uses it *instead of* the solvent model's global
        ``k_sol``/``B_sol``, which then stop affecting the result.
        """
        mean_res = self._data.mean_res_per_bin()

        # Seeded from k_sol * exp(-B s^2) with Phenix-like k=0.35, B=46.
        s_per_bin = 1.0 / (2.0 * mean_res + 1e-6)  # sin(theta)/lambda
        initial_kmask = 0.35 * torch.exp(-46.0 * s_per_bin**2)

        # Zero the high-resolution tail.
        initial_kmask = torch.where(
            initial_kmask < 0.05, torch.zeros_like(initial_kmask), initial_kmask
        )

        self.log_kmask = nn.Parameter(
            torch.log(initial_kmask.clamp(min=1e-6) + 1e-6).to(self.device)
        )

    def get_scale(self) -> float:
        """``exp`` of the reflection-mean isotropic log scale, or 1.0 if unscaled.

        One number summarising a curve, for callers that want an overall magnitude; it is
        not the scale applied to any particular reflection.
        """
        if hasattr(self, "c_iso") and self.c_iso is not None:
            with torch.no_grad():
                return torch.exp(self.iso_log_scale().mean()).item()
        return 1.0

    def setup_bin_wise_bfactor(self):
        """Initialize bin-wise B-factor correction parameters."""
        self.bin_wise_bfactor = nn.Parameter(
            torch.zeros(self.nbins, dtype=get_float_dtype(), device=self.device)
        )

    def bin_wise_bfactor_correction(self):
        """Per-reflection ``exp(-B_bin s^2 / 4)`` from the per-bin B parameter."""
        # Index-add-backward gather: the parameter is O(nbins) while the default ``[bins]``
        # backward radix-sorts all N_refl indices before scattering. Same pattern is used
        # for ``log_scale`` and ``log_kmask`` in ``forward``.
        b_expanded = gather_with_index_add(self.bin_wise_bfactor, self.bins)
        s = torch.norm(self.s, dim=1)
        s_squared = s**2
        exp = -b_expanded * s_squared / 4
        return torch.exp(exp.clamp(max=10.0, min=-10.0))

    def get_binwise_mean_intensity(self, fcalc: torch.Tensor):
        """
        Per-bin mean observed and scaled-calculated intensities, plus mean resolution.

        Computed over valid **work-set** reflections only.

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors (complex); scaled internally.

        Returns
        -------
        tuple
            ``(mean_I_obs, mean_I_calc, mean_resolution)``, each per bin.
        """
        F_calc = torch.abs(self(fcalc))
        fobs = self._data.get_corrected_data()[0]
        valid = self._data.masks().to(torch.bool)
        # ``rfree_flags != 0`` is the WORK set (1=work, 0=test); despite the
        # ``rfree`` name this boolean mask selects work reflections.
        rfree = self._data.rfree_flags.to(torch.bool)
        sel = valid & rfree  # valid work-set reflections
        intensities = fobs ** 2
        calc_intensities = F_calc ** 2
        # Accumulators must match the scatter source dtype: scatter_add raises on a
        # mismatch under a float64 config.
        mean_obs_intensity = torch.zeros(self.nbins, device=self.device, dtype=fobs.dtype)
        mean_calc_intensity = torch.zeros(self.nbins, device=self.device, dtype=fobs.dtype)
        counts = torch.zeros(self.nbins, device=self.device, dtype=fobs.dtype)
        counts_vals = torch.ones_like(F_calc, device=self.device, dtype=fobs.dtype)
        bins_sel = self.bins.to(torch.int64)[sel]  # dtype-ok: bin indices for advanced indexing; PyTorch requires int64
        mean_obs_intensity = torch.scatter_add(
            mean_obs_intensity, 0, bins_sel, intensities[sel]
        )
        mean_calc_intensity = torch.scatter_add(
            mean_calc_intensity, 0, bins_sel, calc_intensities[sel]
        )
        counts = torch.scatter_add(counts, 0, bins_sel, counts_vals[sel])
        mean_obs_intensity = mean_obs_intensity / (counts + 1e-6)
        mean_calc_intensity = mean_calc_intensity / (counts + 1e-6)
        return mean_obs_intensity, mean_calc_intensity, self._data.mean_res_per_bin()

    def screen_solvent_params(
        self,
        fcalc: torch.Tensor,
        steps: int = 15,
        use_low_res_weighting: bool = True,
        low_res_cutoff: float = 5.0,
        fit_on_low_res_only: bool = True,
        low_res_limit: float = 3.5,
    ):
        """
        Grid-search ``(k_sol, ss_half)`` and write the best pair into the solvent model.

        Mutates ``self.solvent`` in place (``.data`` assignment, so no gradient history) and
        leaves the winning values behind; there is no restore. The falloff exponent ``n``
        is left at its current value -- it trades off against ``ss_half`` and a
        three-dimensional grid costs ``steps**3`` forward passes for a parameter the
        subsequent L-BFGS fit refines anyway. Restricting the fit to low resolution keeps
        high-resolution reflections, where the solvent has already switched off, from
        driving the falloff. Falls back to all work reflections if fewer than 100 pass
        ``low_res_limit``.

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors (complex).
        steps : int, default 15
            Grid points per parameter, so ``steps**2`` forward passes.
        use_low_res_weighting : bool, default True
            Weight reflections by ``exp(-s * low_res_cutoff)``.
        low_res_cutoff : float, default 5.0
            Weighting scale, in Angstroms.
        fit_on_low_res_only : bool, default True
            Restrict the fit to reflections beyond ``low_res_limit``.
        low_res_limit : float, default 3.5
            Resolution limit for low-res-only fitting, in Angstroms.

        Raises
        ------
        RuntimeError
            If no solvent model has been set.
        """
        if not hasattr(self, "solvent") or self.solvent is None:
            raise RuntimeError("No solvent model set. Call set_solvent_model() first.")

        fobs, sigma = self._data.get_corrected_data()
        fobs = fobs.to(get_float_dtype()).detach()
        # Note: ``rfree_flags != 0`` is the WORK set (1=work, 0=test), so this
        # boolean mask (despite the ``rfree`` name) selects work reflections.
        rfree = self._data.rfree_flags.to(torch.bool)
        fcalc = fcalc.detach()

        s = torch.norm(get_scattering_vectors(self._data.hkl, self.cell), dim=1)
        resolution = 1.0 / (s + 1e-6)

        if fit_on_low_res_only:
            low_res_mask = (resolution > low_res_limit) & rfree
            n_low_res = low_res_mask.sum().item()
            if self.verbose > 1:
                print(
                    f"Solvent screening using {n_low_res} low-res reflections (>{low_res_limit}Å)"
                )

            if n_low_res < 100:
                print(
                    f"Warning: Only {n_low_res} low-res reflections, using all reflections instead"
                )
                fit_on_low_res_only = False

        if not fit_on_low_res_only:
            low_res_mask = rfree

        if use_low_res_weighting:
            weights = torch.exp(-s * low_res_cutoff).detach()
            weights = weights / weights[low_res_mask].sum()
            if self.verbose > 1:
                low_res_frac = (resolution > low_res_cutoff).float().mean()
                print(
                    f"Low-resolution weighting: {low_res_frac*100:.1f}% reflections above {low_res_cutoff}Å"
                )
        else:
            weights = torch.ones_like(fobs)
            weights = weights / weights[low_res_mask].sum()

        best_log_k_solvent = self.solvent.log_k_solvent.clone()
        best_log_ss_half = self.solvent.log_ss_half.clone()
        best_loss = float("inf")

        ksol_start = torch.log(torch.tensor(0.1, device=self.device))
        ksol_end = torch.log(torch.tensor(0.6, device=self.device))
        ss_lo, ss_hi = SS_HALF_BOUNDS

        for log_k_solvent in torch.linspace(
            ksol_start, ksol_end, steps=steps, device=self.device
        ):
            for log_ss_half in torch.linspace(
                float(torch.log(torch.tensor(ss_lo))),
                float(torch.log(torch.tensor(ss_hi))),
                steps=steps,
                device=self.device,
            ):
                self.solvent.log_k_solvent.data = log_k_solvent.to(
                    dtype=self.solvent.log_k_solvent.dtype
                )
                self.solvent.log_ss_half.data = log_ss_half.to(
                    dtype=self.solvent.log_ss_half.dtype
                )

                scaled_fcalc = self.forward(fcalc)

                diff = fobs[low_res_mask] - torch.abs(scaled_fcalc[low_res_mask])
                sigma_subset = sigma[low_res_mask]
                if hasattr(sigma_subset, "get_mask"):
                    sigma_data = sigma_subset.get_data()[sigma_subset.get_mask()]
                    eps = torch.median(sigma_data).item() * 1e-1
                else:
                    eps = torch.median(sigma_subset).item() * 1e-1
                sigma_safe = torch.clamp(sigma_subset, min=eps)
                nll_per_refl = 0.5 * (diff**2) / (sigma_safe**2)

                if use_low_res_weighting:
                    nll_loss = (nll_per_refl * weights[low_res_mask]).sum()
                else:
                    nll_loss = nll_per_refl.mean()

                if nll_loss.item() < best_loss:
                    best_loss = nll_loss.item()
                    best_log_k_solvent = log_k_solvent.clone()
                    best_log_ss_half = log_ss_half.clone()

        self.solvent.log_k_solvent.data = best_log_k_solvent.to(
            dtype=self.solvent.log_k_solvent.dtype
        )
        self.solvent.log_ss_half.data = best_log_ss_half.to(
            dtype=self.solvent.log_ss_half.dtype
        )

        if self.verbose > 0:
            k_sol = torch.exp(best_log_k_solvent).item()
            d_half = 1.0 / (2.0 * torch.exp(best_log_ss_half).sqrt().item())
            print(
                f"Optimal solvent parameters found: k_sol={k_sol:.4f}, "
                f"d_half={d_half:.2f} A, NLL Loss={best_loss:.4f}"
            )

    def refine_lbfgs(
        self,
        fcalc: torch.Tensor,
        nsteps: int = 3,
        lr: float = 1.0,
        max_iter: int = 200,
        history_size: int = 10,
        verbose: bool = True,
        scale_target: str = DEFAULT_SCALE_TARGET,
    ):
        """
        Refine every scaler parameter with L-BFGS on the work set.

        ``fcalc`` is detached, so the only leaves in the graph are the scaler's own
        parameters. Adds an ``U**2`` penalty to whichever objective is chosen.

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors (complex).
        nsteps : int, default 3
            Number of LBFGS steps.
        lr : float, default 1.0
            Learning rate (typically 1.0 for LBFGS).
        max_iter : int, default 200
            Maximum iterations per line search.
        history_size : int, default 10
            Number of previous gradients kept for the Hessian approximation.
        verbose : bool, default True
            Print progress; gated by ``self.verbose`` as well.
        scale_target : str, default :data:`DEFAULT_SCALE_TARGET`
            Which :data:`~torchref.refinement.targets.xray._specs.XRAY_TARGETS` row to
            minimise, restricted to :data:`SCALE_TARGETS`. ``'ls'`` is unit-weight least
            squares; ``'nll'`` is the sigma_obs-weighted Gaussian; ``'ml_noalpha'`` is the
            Read-MLF sigma_A likelihood.

            Only rows whose likelihood centres on ``|F_calc|`` are admissible: ``alpha`` is
            degenerate with the scale being fitted, so an ``alpha*|F_calc|``-centred row
            (``ml``, ``ml_full``) drives the scale to absorb ``1/alpha`` and inflates every
            R-factor computed from ``k*|F_calc|``.

            A least-squares fit can drive the per-bin scale toward 0 in shells where
            ``F_obs`` is noise-dominated and uncorrelated with ``F_calc``, which blows up
            R. The diagnostic is ``min(k)/median(k)`` over the per-bin scales;
            ``'ml_noalpha'`` absorbs such a mismatch into ``beta`` instead.

        Returns
        -------
        dict
            Per-step ``steps``, ``xray_work``, ``xray_test`` (per-reflection *means*, so
            work and free are comparable despite the ~20x size difference), ``rwork``,
            ``rfree``.

        Raises
        ------
        ValueError
            If ``scale_target`` is not in :data:`SCALE_TARGETS`.
        """
        if scale_target not in SCALE_TARGETS:
            raise ValueError(
                f"scale_target must be one of {SCALE_TARGETS}, got {scale_target!r}"
            )
        # Local import, deliberately: `torchref.refinement` imports
        # `torchref.scaling` at module scope, so hoisting these to module level
        # closes an import cycle. Do not "tidy" them up.
        from torchref.refinement.loss_state import LossState
        from torchref.refinement.targets.xray import create_xray_target

        fcalc = fcalc.detach()

        # A registry row, built exactly as the body target is, so both evaluate the same
        # likelihood code. `model=None`: `fcalc` is passed to `forward()` per closure call, so
        # the fit never recomputes structure factors and the only leaves in the graph are the
        # scaler's.
        scale_xray = create_xray_target(
            data=self._data,
            model=None,
            scaler=self,
            mode=scale_target,
            use_set="work",
            verbose=0,
            device=self.device,
        )
        scaler_self = self

        # One constant applied to EVERY term below, so the objective is an exact
        # rescaling: same minimiser, same gradient direction, same relative weight
        # between the likelihood and the U penalty.
        #
        # ``torch.optim.LBFGS`` converges on ABSOLUTE tolerances (``tolerance_grad``,
        # ``tolerance_change``), so the objective has to be O(1) for them to mean
        # anything. Unnormalised it carries the data's own magnitude -- a large work set
        # under unit weights reaches ~1e9, where the float32 ulp of the loss exceeds the
        # decrease the line search is trying to resolve, and the walk leaves ``U`` far
        # enough out that ``sum(U**2)`` overflows. Dividing by ``sum(F_obs**2)`` makes the
        # loss dimensionless and O(R**2) whatever the structure size or data scale.
        #
        # Computed in the configured dtype. Because the constant rescales every term
        # identically, its own precision does not enter the result: it only has to be
        # finite, positive and of the right magnitude. float32 gives it to ~5e-8 relative
        # (``sum`` reduces pairwise, so error grows like log(N)*eps, not N*eps), which
        # cancels out of the minimiser and the gradient direction alike. Do not cast to
        # float64 -- MPS has none and raises on ``.double()``.
        _f_obs_work = self._data.work.F.detach()
        _norm = float(1.0 / _f_obs_work.pow(2).sum().clamp(min=1e-30))

        class _ScalerXrayTarget(nn.Module):
            """The registry row, closed over the detached ``fcalc``."""

            name = "scaler/xray"

            def forward(self):
                return scale_xray(fcalc=fcalc) * _norm

            def maintenance(self):
                """Forward the hook so sigma_A rows drop their ``beta`` cache after a step
                block, as they do under the body refinement."""
                maint = getattr(scale_xray, "maintenance", None)
                if maint is not None:
                    maint()

        class _ScalerUPenalty(nn.Module):
            """``sum(U**2)`` on the anisotropic scale tensor.

            A scaler regularizer rather than part of any likelihood, registered as its own
            target so it stays visible in the loss breakdown and leaves the x-ray term
            directly comparable to the body target's.
            """

            name = "scaler/u_penalty"

            def forward(self):
                return torch.sum(scaler_self.U**2) * _norm

        state = LossState(device=self.device)
        state.register_target("scaler/xray", _ScalerXrayTarget())
        state.register_target("scaler/u_penalty", _ScalerUPenalty())

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=lr,
            max_iter=max_iter,
            history_size=history_size,
            line_search_fn="strong_wolfe",
        )

        metrics = {
            "target": "scales",
            "steps": [],
            "xray_work": [],
            "xray_test": [],
            "rwork": [],
            "rfree": [],
        }

        if verbose and self.verbose > 0:
            print("Refining scales with LBFGS...")

        if self.verbose > 2:
            assert torch.all(
                torch.isfinite(fcalc)
            ), "Non-finite values found in fcalc during scale optimization."

        for step in range(nsteps):
            state.step(optimizer, context="scaler.refine_lbfgs")

            with torch.no_grad():
                fcalc_scaled = self.forward(fcalc)
                work, free = self._data.work, self._data.free

                # Means, not sums: work and free differ in size by ~20x.
                xray_work = nll_xray_mean(
                    work.F, work.select(fcalc_scaled), work.sigF
                )
                xray_test = nll_xray_mean(
                    free.F, free.select(fcalc_scaled), free.sigF
                )
                # Same work/free partition as XrayTarget.get_rfactor, so any difference
                # between the two reported R values is model state, not method.
                rwork, rfree_val = rfactor_work_free(
                    self._data, torch.abs(fcalc_scaled)
                )

                metrics["steps"].append(step + 1)
                metrics["xray_work"].append(xray_work.item())
                metrics["xray_test"].append(xray_test.item())
                metrics["rwork"].append(rwork)
                metrics["rfree"].append(rfree_val)

                if verbose and self.verbose > 2:
                    print(
                        f"  Step {step+1}/{nsteps}: "
                        f"Rwork={rwork:.4f}, Rfree={rfree_val:.4f}, "
                        f"NLL_work={xray_work.item():.2f}, NLL_test={xray_test.item():.2f}"
                    )

        if verbose and self.verbose > 0:
            with torch.no_grad():
                print(
                    f"Scale refinement complete. rwork: {rwork:.4f}, rfree: {rfree_val:.4f}\n"
                )
                print("Final Scale Parameters: ")
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        print(f"  {name}: {param.data}")

        return metrics

    def forward(
        self,
        fcalc: torch.Tensor,
        use_mask: bool = True,
        f_sol_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply per-bin scale, B, anisotropy and bulk solvent to ``fcalc``.

        Every component is optional: each is applied only if the corresponding attribute
        exists, so an un-initialized scaler returns its input unchanged.

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors, shape ``(N,)`` or ``(B, N)``. ``N`` matching the
            full HKL size means no internal masking; anything else is taken to be the
            already-masked subset and the scaler masks its own per-reflection terms to match.
        use_mask : bool, default True
            Deprecated and inert -- never read. Masking follows the input shape, so
            ``use_mask=False`` does *not* disable it.
        f_sol_override : torch.Tensor, optional
            Raw solvent structure factors replacing the cached ``_f_sol_raw`` (k_sol / B_sol
            / phase damping still applied). **Overwrites the cache**, so it persists into
            later calls until invalidated. Used by ``CollectionScaler``.

        Returns
        -------
        torch.Tensor
            Scaled structure factors, same shape as the input.
        """
        batched = True

        if fcalc.ndim == 1:
            fcalc = fcalc.unsqueeze(0)
            batched = False

        n_full = len(self.bins)
        n_fcalc = fcalc.shape[1]

        if n_fcalc == n_full:
            apply_internal_mask = False
        else:
            apply_internal_mask = True
            mask = self._data.masks().to(torch.bool)

        if hasattr(self, "U"):
            anisotropy_factors = self.anisotropy_correction()
            aniso_correction = (
                anisotropy_factors[mask] if apply_internal_mask else anisotropy_factors
            )
        else:
            aniso_correction = torch.tensor(1.0, device=self.device, dtype=fcalc.dtype)

        if f_sol_override is not None:
            self._f_sol_raw = f_sol_override

        if hasattr(self, "solvent") and self.solvent is not None:
            # Lazily cache raw solvent SFs (FFT of mask) — only recomputed
            # when invalidated via _f_sol_raw = None (e.g. after update_solvent)
            if self._f_sol_raw is None:
                # The solvent mask is real density with no anomalous term, so
                # F_sol(-h) is exactly conj(F_sol(h)) and evaluating on the
                # canonical index already matches the canonical fcalc below.
                self._f_sol_raw = self.solvent.get_rec_solvent(self.hkl)

            f_sol_raw = (
                self._f_sol_raw[mask] if apply_internal_mask else self._f_sol_raw
            )

            if hasattr(self, "log_kmask"):
                # Per-bin kmask REPLACES the model's global k_sol/B_sol below.
                kmask = torch.exp(self.log_kmask.clamp(min=-10.0, max=10.0))
                kmask = torch.clamp(kmask, min=0.0, max=10.0)
                bins_to_use = self.bins[mask] if apply_internal_mask else self.bins
                kmask_per_refl = gather_with_index_add(kmask, bins_to_use)
                f_sol = kmask_per_refl * f_sol_raw
            else:
                # k_sol * exp(i*phase) * falloff(ss) * f_mask, with the falloff taken
                # from the solvent model itself so this path and ``SolventModel.forward``
                # cannot drift apart.
                sol = self.solvent
                k_sol = sol.k_solvent()
                s_half_sq = (
                    self._s_half_sq[mask] if apply_internal_mask else self._s_half_sq
                )
                b_factor = sol.damping(s_half_sq)
                if sol.optimize_phase:
                    # A bare ``1j`` would promote the product to complex128.
                    j = torch.tensor(1j, dtype=get_complex_dtype(), device=self.device)
                    f_sol_raw = f_sol_raw * torch.exp(j * sol.phase_offset)
                f_sol = k_sol * f_sol_raw * b_factor
        else:
            f_sol = torch.tensor(0.0, device=self.device, dtype=fcalc.dtype)

        if hasattr(self, "c_iso") and self.c_iso is not None:
            design = self._iso_design[mask] if apply_internal_mask else self._iso_design
            K_overall = torch.exp(self.iso_log_scale(design))
        else:
            K_overall = torch.tensor(1.0, device=self.device, dtype=fcalc.dtype)

        if hasattr(self, "bin_wise_bfactor") and self.bin_wise_bfactor is not None:
            bfactor_factors = self.bin_wise_bfactor_correction()
            b_overall = (
                bfactor_factors[mask] if apply_internal_mask else bfactor_factors
            )
        else:
            b_overall = torch.tensor(1.0, device=self.device, dtype=fcalc.dtype)

        fcalc = (
            K_overall.unsqueeze(0)
            * b_overall.unsqueeze(0)
            * (aniso_correction.unsqueeze(0) * fcalc + f_sol.unsqueeze(0))
        )

        if not batched:
            fcalc = fcalc.squeeze(0)

        return fcalc

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Buffers and parameters, plus ``nbins``/``n_iso_coeff``/``verbose`` and the solvent
        sub-state.

        The **data reference is not saved** -- reattach it with :meth:`set_data` after
        loading.

        Parameters
        ----------
        destination : dict, optional
            Optional dict to populate.
        prefix : str, default ''
            Prefix for parameter names.
        keep_vars : bool, default False
            Whether to keep variables in the computational graph.
        """
        state = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        state[prefix + "nbins"] = self.nbins
        state[prefix + "n_iso_coeff"] = self.n_iso_coeff
        state[prefix + "verbose"] = self.verbose

        if hasattr(self, "solvent") and self.solvent is not None:
            state[prefix + "solvent"] = self.solvent.state_dict()

        return state

    def load_state_dict(self, state_dict, strict=True):
        """
        Load scaler state; assumes data is already set via ``__init__`` or ``set_data``.

        **Mutates ``state_dict``**: the metadata and solvent keys are ``pop``-ed out of the
        caller's dict before delegating, so it cannot be reused for a second load.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing scaler state.
        strict : bool, default True
            Whether to strictly enforce that keys match.
        """
        self.nbins = state_dict.pop("nbins", 20)
        self.n_iso_coeff = state_dict.pop("n_iso_coeff", 6)
        self.verbose = state_dict.pop("verbose", 1)
        # Legacy state dicts may contain a "frozen" entry; drop it silently.
        state_dict.pop("frozen", None)

        solvent_state = state_dict.pop("solvent", None)

        result = super().load_state_dict(state_dict, strict=strict)

        if (
            solvent_state is not None
            and hasattr(self, "solvent")
            and self.solvent is not None
        ):
            self.solvent.load_state_dict(solvent_state)

        return result

    def save_state(self, path: str):
        """``torch.save`` the state dict to ``path``."""
        torch.save(self.state_dict(), path)
        if self.verbose > 0:
            print(f"Saved scaler state to {path}")

    def load_state(self, path: str, strict: bool = True):
        """Load a state dict from ``path`` onto this scaler's device.

        ``strict`` is forwarded to :meth:`load_state_dict`. Uses
        ``weights_only=False``, so only load files you trust.
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state_dict, strict=strict)
        if self.verbose > 0:
            print(f"Loaded scaler state from {path}")
