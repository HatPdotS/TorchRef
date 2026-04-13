"""
Base scaler class for crystallographic scaling without model dependency.

This module provides `ScalerBase`, a class that implements all scaling functionality
but does NOT maintain a reference to a Model object. All methods that require
calculated structure factors (F_calc) take them as input arguments.

This allows the scaler to be used independently of any specific model implementation,
making it suitable for use cases like:
- Molecular replacement where F_calc comes from external sources
- Testing and validation with precomputed structure factors
- Custom model implementations
"""

from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

from torchref.base.math_torch import U_to_matrix
from torchref.base.metrics import bin_wise_rfactors, get_rfactors, nll_xray, nll_xray_lognormal
from torchref.base.reciprocal import get_scattering_vectors
from torchref.utils.debug_utils import DebugMixin
from torchref.utils.utils import ModuleReference

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.scaling.solvent import SolventModel


class ScalerBase(DebugMixin, nn.Module):
    """
    Base scaler class for crystallographic scaling without model dependency.

    All methods that require calculated structure factors (F_calc) take them
    as input arguments. This allows the scaler to be used independently of
    any specific model implementation.

    Supports two initialization patterns:

    1. Empty initialization (for state_dict loading)::

        scaler = ScalerBase()  # Creates empty shell
        scaler.load_state_dict(torch.load('scaler.pt'))

    2. Full initialization with data::

        scaler = ScalerBase(data=reflection_data, nbins=20)
        scaler.initialize(fcalc)

    Parameters
    ----------
    data : ReflectionData, optional
        ReflectionData object with observed data.
    nbins : int, default 20
        Number of resolution bins.
    verbose : int, default 1
        Verbosity level.
    device : torch.device, default torch.device('cpu')
        Computation device.

    Attributes
    ----------
    device : torch.device
        Current computation device.
    nbins : int
        Number of resolution bins.
    frozen : bool
        Whether the scaler parameters are frozen.
    """

    def __init__(
        self,
        data: Optional["ReflectionData"] = None,
        nbins: int = 20,
        verbose: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize ScalerBase.

        If data is provided, fully initializes the scaler.
        If not provided (empty init), creates a shell ready for load_state_dict().

        Parameters
        ----------
        data : ReflectionData, optional
            ReflectionData object with observed data.
        nbins : int, default 20
            Number of resolution bins.
        verbose : int, default 1
            Verbosity level.
        device : torch.device, default torch.device('cpu')
            Computation device.
        """
        super(ScalerBase, self).__init__()
        self.device = device
        self.verbose = verbose
        self.nbins = nbins
        self.frozen = False

        # Empty initialization - just set up configuration
        if data is None:
            self._data = None
            self.cell = None
            self.register_buffer("s", None)
            self.register_buffer("bins", None)
            self.register_buffer("_s_half_sq", None)
            self.register_buffer("sigma_eff", None)
            self.register_buffer("sigma_eff_per_bin", None)
            self._f_sol_raw = None
            return

        # Full initialization with data
        self.to(self.device)
        self._data = ModuleReference(data)

        self.cell = data.cell
        s = get_scattering_vectors(data.hkl, self.cell)
        self.register_buffer("s", s)
        # Precompute (sin(θ)/λ)² for B-factor damping — avoids recomputing per call
        self.register_buffer("_s_half_sq", (torch.norm(s, dim=1) / 2.0) ** 2)
        self._f_sol_raw = None
        bins, self.nbins = self._data.get_bins(self.nbins)
        self.register_buffer("bins", bins)
        # Effective sigma buffers (populated by estimate_sigma_eff after scaling)
        # sigma_eff: per-reflection effective sigma, shape (N,)
        # sigma_eff_per_bin: per-bin effective sigma, shape (nbins,)
        # Initialized to raw sigmas; will be updated after scaling.
        _, _, sigma_raw, _ = self._data(mask=False)
        self.register_buffer("sigma_eff", sigma_raw.clone().to(self.device))
        self.register_buffer(
            "sigma_eff_per_bin",
            torch.zeros(self.nbins, device=self.device, dtype=sigma_raw.dtype),
        )
        if self.verbose > 0:
            print(f"Initialized ScalerBase with {self.nbins} bins.")

    def set_data(self, data: "ReflectionData"):
        """
        Set data reference after empty initialization.

        This is useful when loading from state_dict and then needing
        to reconnect to a data object.

        Parameters
        ----------
        data : ReflectionData
            ReflectionData object with observed data.
        """
        self._data = ModuleReference(data)
        if data.cell is not None:
            self.cell = data.cell
        if self.s is None and data.hkl is not None and data.cell is not None:
            s = get_scattering_vectors(data.hkl, data.cell)
            self.register_buffer("s", s)
            self.register_buffer("_s_half_sq", (torch.norm(s, dim=1) / 2.0) ** 2)
        self._f_sol_raw = None
        if self.bins is None and data.hkl is not None:
            bins, self.nbins = self._data.get_bins(self.nbins)
            self.register_buffer("bins", bins)

    def initialize(self, fcalc: torch.Tensor):
        """
        Initialize scaling parameters using provided F_calc.

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors (complex).
        """
        self.calc_initial_scale(fcalc)
        self.setup_anisotropy_correction()

    @property
    def hkl(self):
        """Get HKL indices from data."""
        return self._data.hkl

    def freeze(self):
        """Freeze scaler parameters (exclude from optimization)."""
        self.frozen = True

    def unfreeze(self):
        """Unfreeze scaler parameters (include in optimization)."""
        self.frozen = False

    def calc_initial_scale(self, fcalc: torch.Tensor):
        """
        Calculate the initial scale factor based on the ratio of observed to calculated structure factors.

        Excludes reflections with negative intensities to avoid bias from French-Wilson conversion.

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors (complex).

        Returns
        -------
        torch.nn.Parameter
            The log scale parameter for each resolution bin.
        """
        hkl, fobs, sigma, rfree = self._data(mask=False)
        if self.verbose > 0:
            print(f"Calculating initial scale factors using {self.nbins} bins.")
        assert torch.all(
            torch.isfinite(fcalc)
        ), "Non-finite values found in fcalc during initial scale calculation."

        scales = torch.zeros(self.nbins, device=self.device, dtype=fobs.dtype)
        counts = torch.zeros(self.nbins, device=self.device, dtype=fobs.dtype)
        fcalc_amp = torch.abs(fcalc).to(fobs.dtype)

        # Exclude reflections with negative intensities from scale calculation
        # These have biased F values from French-Wilson conversion
        if hasattr(self._data, "I") and self._data.I is not None:
            positive_mask = self._data.I > 0
            if self.verbose > 1:
                n_excluded = (~positive_mask).sum().item()
                print(
                    f"Excluding {n_excluded} negative intensity reflections from scale calculation"
                )
        else:
            positive_mask = torch.ones_like(fobs, dtype=torch.bool)

        # Calculate ratios only for positive intensity reflections
        mask = (self._data.masks() & rfree & positive_mask).to(torch.bool)
        bins = self.bins[mask].to(torch.int64)

        fobs = fobs.clamp(min=1e-3)[mask]
        fcalc_amp = fcalc_amp.clamp(min=1e-3)[mask]

        log_ratios = torch.log(fobs) - torch.log(fcalc_amp)
        assert torch.all(
            torch.isfinite(log_ratios)
        ), f"Non-finite log ratios encountered in initial scale calculation {torch.sum(~torch.isfinite(log_ratios)).item()}"

        # Ensure all tensors are on the same device for scatter_add
        log_ratios = log_ratios.to(self.device)
        bins = bins.to(self.device)
        counts_vals = torch.ones_like(self.bins, device=self.device, dtype=fobs.dtype)
        sum_log_scales = torch.scatter_add(scales, 0, bins, log_ratios)
        counts = torch.scatter_add(counts, 0, bins, counts_vals)
        log_scale = sum_log_scales / (counts + 1e-6)
        initial_log_scale = log_scale
        if self.verbose > 1:
            print(
                "Initial scale factors per bin:",
                initial_log_scale.detach().cpu().numpy(),
            )
        self.log_scale = nn.Parameter(initial_log_scale.detach().to(self.device))
        return self.log_scale

    def setup_anisotropy_correction(self):
        """Initialize anisotropic correction parameters."""
        self.U = nn.Parameter(
            torch.normal(0, 0.001, (6,), dtype=torch.float32, device=self.device)
        )

    def anisotropy_correction(self):
        """
        Compute anisotropic correction factors.

        Returns
        -------
        torch.Tensor
            Anisotropic correction factors for each reflection.
        """
        U = U_to_matrix(self.U)
        # matmul + element-wise multiply + sum is much faster than einsum
        # for this bilinear form s^T U s on CPU (avoids einsum dispatch overhead)
        sU = torch.matmul(self.s, U)  # (N, 3)
        exp = -2 * torch.pi**2 * (sU * self.s).sum(dim=1)
        return torch.exp(exp.clamp(max=10.0, min=-10.0))

    def fit_anisotropy(self, fcalc: torch.Tensor, nsteps: int = 100):
        """
        Fit anisotropic correction using provided F_calc.

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors (complex).
        nsteps : int, default 100
            Number of optimization steps.
        """
        if not hasattr(self, "U"):
            self.U = nn.Parameter(
                torch.normal(0, 0.01, (6,), dtype=torch.float32, device=self.device)
            )
        hkl, fobs, sigma, rfree = self._data()

        fobs = fobs.to(torch.float32).detach()
        fcalc = torch.abs(fcalc).to(torch.float32).detach()

        optimizer = torch.optim.Adam([self.U, self.log_scale], lr=1e-1)
        for i in range(nsteps):
            optimizer.zero_grad()
            scaled_fcalc = self.forward(fcalc)
            loss = nll_xray(fobs[rfree], scaled_fcalc[rfree], sigma[rfree])

            loss.backward()
            optimizer.step()
            if self.verbose > 0 and (i % 10 == 0 or i == nsteps - 1):
                print(f"Anisotropy fit iteration {i+1}/{nsteps}, Loss: {loss.item():.4f}")

    def set_solvent_model(self, solvent_model: "SolventModel") -> None:
        """
        Set a pre-configured SolventModel for solvent contribution.

        The SolventModel must be initialized externally (requires a Model object).

        Parameters
        ----------
        solvent_model : SolventModel
            Pre-configured solvent model that can compute solvent structure factors.
        """
        self.solvent = solvent_model
        self._f_sol_raw = None  # Invalidate cached raw solvent SFs

    def setup_binwise_solvent_scale(self):
        """
        Setup bin-wise solvent scaling (Phenix-style kmask per bin).

        This allows finer control over solvent contribution per resolution bin,
        which is more flexible than a single global B_sol parameter.
        """
        mean_res = self._data.mean_res_per_bin()

        # Initialize with exponential decay: k_mask = k_sol * exp(-B * s^2)
        # Use B=46 as starting point (Phenix-like)
        s_per_bin = 1.0 / (2.0 * mean_res + 1e-6)  # sin(theta)/lambda
        initial_kmask = 0.35 * torch.exp(-46.0 * s_per_bin**2)

        # Set high-res bins to 0 (where kmask < 0.05)
        initial_kmask = torch.where(
            initial_kmask < 0.05, torch.zeros_like(initial_kmask), initial_kmask
        )

        self.log_kmask = nn.Parameter(
            torch.log(initial_kmask.clamp(min=1e-6) + 1e-6).to(self.device)
        )

    def fit_all_scales(self, fcalc: torch.Tensor):
        """
        Fit all scale parameters using provided F_calc.

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors (complex).
        """
        hkl, fobs, sigma, rfree = self._data()
        fobs = fobs.to(torch.float32).detach()
        fcalc = fcalc.detach()

        for lr in [1e-1, 5e-2, 1e-2]:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            for i in range(20):
                optimizer.zero_grad()
                scaled_fcalc = self.forward(fcalc)
                nll_loss = nll_xray(fobs[rfree], scaled_fcalc[rfree], sigma[rfree])
                if torch.isnan(nll_loss):
                    raise ValueError(
                        "NaN encountered in NLL loss during scale fitting."
                    )
                nll_log_loss_xray = nll_xray_lognormal(
                    fobs[rfree], scaled_fcalc[rfree], sigma[rfree]
                )
                loss = nll_loss
                loss.backward()
                optimizer.step()
            if self.verbose > 1:
                print(
                    f"Solvent fit after step, Loss: {loss.item():.4f}, NLL: {nll_loss.item():.4f}, LogLoss: {nll_log_loss_xray.item():.4f}"
                )

    def fit_simple(self, fobs: torch.Tensor, fcalc: torch.Tensor) -> None:
        """
        Fit a single global scale factor analytically (least-squares).

        This is the simple scaling approach:
            k = sum(|F_obs||F_calc|) / sum(|F_calc|²)

        Useful for rigid body refinement where only an overall scale is needed.

        Parameters
        ----------
        fobs : torch.Tensor
            Observed structure factor amplitudes.
        fcalc : torch.Tensor
            Calculated structure factors (complex).
        """
        fcalc_amp = torch.abs(fcalc)
        fobs_amp = torch.abs(fobs)

        # Analytical least-squares solution
        numerator = torch.sum(fobs_amp * fcalc_amp)
        denominator = torch.sum(fcalc_amp**2)

        # Avoid division by zero
        scale = numerator / denominator.clamp(min=1e-10)

        # Initialize log_scale if not present
        if not hasattr(self, "log_scale") or self.log_scale is None:
            self.log_scale = nn.Parameter(
                torch.zeros(self.nbins, dtype=fobs.dtype, device=self.device)
            )

        # Store as log_scale (broadcast single value to all bins)
        with torch.no_grad():
            self.log_scale.fill_(torch.log(scale.clamp(min=1e-6)))

    def get_scale(self) -> float:
        """
        Get the current overall scale factor value.

        Returns the mean scale factor across all bins.

        Returns
        -------
        float
            Current scale factor (not log).
        """
        if hasattr(self, "log_scale") and self.log_scale is not None:
            return torch.exp(self.log_scale.mean().clamp(-10, 10)).item()
        return 1.0

    def cuda(self, device=None):
        """
        Move the ScalerBase module to GPU.

        Parameters
        ----------
        device : torch.device, optional
            The target device. If None, uses the default CUDA device.
        """
        super().cuda(device)
        if hasattr(self, "solvent") and self.solvent is not None:
            self.solvent.cuda(device)
        self.device = torch.device("cuda" if device is None else device)
        if self.verbose > 1:
            print(f"ScalerBase moved to device: {self.device}")

    def cpu(self):
        """
        Move the ScalerBase module to CPU.
        """
        super().cpu()
        if hasattr(self, "solvent") and self.solvent is not None:
            self.solvent.cpu()
        self.device = next(self.parameters()).device
        if self.verbose > 1:
            print("ScalerBase moved to CPU")

    def rfactor(self, fcalc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the R-factor between observed and calculated structure factors.

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors (complex).

        Returns
        -------
        tuple
            R-work and R-free values.
        """
        hkl, fobs, _, rfree = self._data()
        fcalc_scaled = self.forward(fcalc)
        return get_rfactors(torch.abs(fobs), torch.abs(fcalc_scaled), rfree)

    def bin_wise_rfactor(self, fcalc: torch.Tensor):
        """
        Calculate the bin-wise R-factor between observed and calculated structure factors.

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors (complex).

        Returns
        -------
        mean_res_per_bin : torch.Tensor
            Mean resolution per bin.
        rwork_per_bin : torch.Tensor
            R-work per bin.
        rfree_per_bin : torch.Tensor
            R-free per bin.
        """
        hkl, fobs, _, rfree = self._data()
        fcalc_scaled = self.forward(fcalc)
        mean_res_per_bin = self._data.mean_res_per_bin()
        return mean_res_per_bin, *bin_wise_rfactors(
            torch.abs(fobs),
            torch.abs(fcalc_scaled),
            rfree,
            self.bins[self._data.masks()],
        )

    def setup_bin_wise_bfactor(self):
        """Initialize bin-wise B-factor correction parameters."""
        self.bin_wise_bfactor = nn.Parameter(
            torch.zeros(self.nbins, dtype=torch.float32, device=self.device)
        )

    def bin_wise_bfactor_correction(self):
        """
        Compute bin-wise B-factor correction factors.

        Returns
        -------
        torch.Tensor
            B-factor correction factors for each reflection.
        """
        b_expanded = self.bin_wise_bfactor[self.bins]
        s = torch.norm(self.s, dim=1)
        s_squared = s**2
        exp = -b_expanded * s_squared / 4
        return torch.exp(exp.clamp(max=10.0, min=-10.0))

    def get_binwise_mean_intensity(self, fcalc: torch.Tensor):
        """
        Get bin-wise mean intensities for observed and calculated structure factors.

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors (complex).

        Returns
        -------
        tuple
            Mean observed intensity, mean calculated intensity, and mean resolution per bin.
        """
        hkl, fobs, _, rfree = self._data()
        F_calc = torch.abs(self(fcalc))
        intensities = torch.abs(fobs) ** 2
        calc_intensities = torch.abs(F_calc) ** 2
        mean_obs_intensity = torch.zeros(self.nbins, device=self.device)
        mean_calc_intensity = torch.zeros(self.nbins, device=self.device)
        counts = torch.zeros(self.nbins, device=self.device)
        counts_vals = torch.ones_like(F_calc, device=self.device, dtype=fobs.dtype)
        mask = self._data.get_mask()
        mean_obs_intensity = torch.scatter_add(
            mean_obs_intensity,
            0,
            self.bins.to(torch.int64)[mask][rfree],
            intensities[rfree],
        )
        mean_calc_intensity = torch.scatter_add(
            mean_calc_intensity,
            0,
            self.bins.to(torch.int64)[mask][rfree],
            calc_intensities[rfree],
        )
        counts = torch.scatter_add(
            counts, 0, self.bins.to(torch.int64)[mask][rfree], counts_vals[rfree]
        )
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
        Screen solvent parameters (k_sol, B_sol) using grid search.

        The bulk solvent contributes primarily at low resolution. Fitting on low-resolution
        reflections only (fit_on_low_res_only=True) prevents high-resolution reflections
        from dominating the optimization and pushing B_sol too low.

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors (complex).
        steps : int, default 15
            Number of grid points for each parameter.
        use_low_res_weighting : bool, default True
            If True, weight low-resolution reflections more heavily
            since solvent primarily contributes at low resolution.
        low_res_cutoff : float, default 5.0
            Resolution cutoff for weighting in Angstroms.
        fit_on_low_res_only : bool, default True
            If True, fit using only low-resolution reflections.
        low_res_limit : float, default 3.5
            Resolution limit for low-res only fitting in Angstroms.
        """
        if not hasattr(self, "solvent") or self.solvent is None:
            raise RuntimeError("No solvent model set. Call set_solvent_model() first.")

        hkl, fobs, sigma, rfree = self._data()
        fobs = fobs.to(torch.float32).detach()
        fcalc = fcalc.detach()

        # Calculate resolution for weighting/filtering
        s = torch.norm(get_scattering_vectors(hkl, self.cell), dim=1)
        resolution = 1.0 / (s + 1e-6)

        # Create mask for low-resolution reflections
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

        # Create weights for low-resolution preference
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
        best_b_solvent = self.solvent.b_solvent.clone()
        best_loss = float("inf")

        ksol_start = torch.log(torch.tensor(0.1, device=self.device))
        ksol_end = torch.log(torch.tensor(0.6, device=self.device))

        for log_k_solvent in torch.linspace(
            ksol_start, ksol_end, steps=steps, device=self.device
        ):
            for b_solvent in torch.linspace(
                30.0, 100.0, steps=steps, device=self.device
            ):
                self.solvent.log_k_solvent.data = log_k_solvent.to(
                    dtype=self.solvent.log_k_solvent.dtype
                )
                self.solvent.b_solvent.data = b_solvent.to(
                    dtype=self.solvent.b_solvent.dtype
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
                    best_b_solvent = b_solvent.clone()

        self.solvent.log_k_solvent.data = best_log_k_solvent.to(
            dtype=self.solvent.log_k_solvent.dtype
        )
        self.solvent.b_solvent.data = best_b_solvent.to(
            dtype=self.solvent.b_solvent.dtype
        )

        if self.verbose > 0:
            k_sol = torch.exp(best_log_k_solvent).item()
            print(
                f"Optimal solvent parameters found: k_sol={k_sol:.4f}, B_sol={best_b_solvent.item():.1f}, "
                f"NLL Loss={best_loss:.4f}"
            )

    def refine_lbfgs(
        self,
        fcalc: torch.Tensor,
        nsteps: int = 3,
        lr: float = 1.0,
        max_iter: int = 200,
        history_size: int = 10,
        verbose: bool = True,
    ):
        """
        Refine scale parameters using LBFGS optimizer.

        This method optimizes the anisotropic scaling and B-factor parameters
        that relate calculated structure factors to observed structure factors.
        Uses the L-BFGS quasi-Newton optimization method for fast convergence.

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
            Number of previous gradients to store for Hessian approximation.
        verbose : bool, default True
            Print progress information.

        Returns
        -------
        dict
            Dictionary with refinement metrics including steps, xray_work,
            xray_test, rwork, rfree.

        Examples
        --------
        ::

            scaler.unfreeze()
            metrics = scaler.refine_lbfgs(fcalc, nsteps=5, verbose=True)
            scaler.freeze()
        """
        from torchref.refinement.loss_state import LossState

        # Ensure scaler is unfrozen
        was_frozen = self.frozen
        if was_frozen:
            self.unfreeze()

        hkl, fobs, sigma, rfree_mask = self._data()
        fcalc = fcalc.detach()

        # Wrap the scaler loss as a LossState target so this path uses the
        # same closure/NaN-validation/auto-freeze infrastructure as the main
        # refinement loop. Because fcalc is detached, the only leaves the
        # loss touches are the scaler's own parameters — LossState's probe
        # picks those up at register time. validate_loss inside state.step
        # handles NaN/Inf rejection so no per-target try/except is needed.
        scaler_self = self

        class _ScalerXrayTarget(nn.Module):
            name = "scaler/xray"

            def forward(self):
                fcalc_scaled = scaler_self.forward(fcalc)
                u_penalty = torch.sum(scaler_self.U**2)
                return nll_xray(fobs, fcalc_scaled, sigma) + u_penalty

        state = LossState(device=self.device)
        state.register_target("scaler/xray", _ScalerXrayTarget())

        # Create LBFGS optimizer for scaler parameters only
        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=lr,
            max_iter=max_iter,
            history_size=history_size,
            line_search_fn="strong_wolfe",
        )

        # Track metrics
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

        # Run optimization
        for step in range(nsteps):
            state.step(optimizer, context="scaler.refine_lbfgs")

            # Evaluate metrics
            with torch.no_grad():
                hkl, fobs, sigma, rfree_mask = self._data()
                fcalc_scaled = self.forward(fcalc)

                xray_work = nll_xray(fobs[rfree_mask], fcalc_scaled[rfree_mask], sigma[rfree_mask])
                xray_test = nll_xray(fobs[~rfree_mask], fcalc_scaled[~rfree_mask], sigma[~rfree_mask])
                rwork, rfree_val = get_rfactors(
                    torch.abs(fobs), torch.abs(fcalc_scaled), rfree_mask
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

        # Restore frozen state
        if was_frozen:
            self.freeze()

        # Estimate per-shell effective sigmas from residuals (SIGMAA-style)
        # This makes the X-ray likelihood robust to miscalibrated experimental sigmas.
        self.estimate_sigma_eff(fcalc)

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

    def estimate_sigma_eff(
        self,
        fcalc: torch.Tensor,
        max_inflation: float = 2.0,
    ) -> torch.Tensor:
        """
        Estimate per-resolution-shell effective sigmas from current residuals.

        Pannu & Read / SIGMAA-style correction: detects miscalibrated
        experimental sigmas by comparing residual variance to the claimed
        variance, per resolution bin.

        For each resolution bin:

            D_bin         = < (F_obs - k * |F_calc|)^2 >      (using work set)
            ratio_bin     = sqrt(D_bin / <sigma_F^2>)
            ratio_capped  = clamp(ratio_bin, 1.0, max_inflation)
            sigma_eff     = sigma_F * ratio_capped

        **Why the cap?** At the start of refinement the model is bad, so
        residuals are dominated by *model error* (which is fixable by
        refining), not noise. Uncapped inflation creates a vicious cycle:
        bad model -> huge sigma_eff -> weak data gradient -> bad model.
        Capping at ``max_inflation`` (default 2.0, i.e. sigmas can grow
        at most 2x) prevents runaway while still correcting genuinely
        under-estimated sigmas.

        As the model improves, residuals shrink and the ratio drops toward
        1, so sigma_eff converges to the raw sigma (good calibration).

        Uses the work set only so the test set doesn't leak into
        sigma estimation.

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors (complex, unscaled).
        max_inflation : float, optional
            Maximum allowed ratio sigma_eff / sigma_raw. Default 2.0.

        Returns
        -------
        torch.Tensor
            Per-reflection effective sigmas, shape (N,).
        """
        with torch.no_grad():
            hkl, fobs_raw, sigma_raw, rfree_mask = self._data(mask=False)
            # Apply scaling to F_calc
            fcalc_scaled = self.forward(fcalc).squeeze(0) if fcalc.ndim == 1 else self.forward(fcalc)
            if fcalc_scaled.ndim > 1:
                fcalc_scaled = fcalc_scaled.squeeze(0)
            fcalc_amp = torch.abs(fcalc_scaled).to(fobs_raw.dtype)

            # Work set only (rfree=True = work in this codebase convention)
            validity = self._data.masks().to(torch.bool)
            work_mask = validity & rfree_mask.bool()

            bins_work = self.bins[work_mask].to(torch.int64)
            fobs_work = fobs_raw[work_mask]
            fcalc_work = fcalc_amp[work_mask]
            sigma_work = sigma_raw[work_mask]

            # Residuals using F_calc directly (alpha=1 assumed post-scaling)
            residuals_sq = (fobs_work - fcalc_work) ** 2

            # Per-bin sums
            sum_d = torch.zeros(self.nbins, device=self.device, dtype=fobs_raw.dtype)
            sum_s2 = torch.zeros(self.nbins, device=self.device, dtype=fobs_raw.dtype)
            counts = torch.zeros(self.nbins, device=self.device, dtype=fobs_raw.dtype)

            sum_d = torch.scatter_add(sum_d, 0, bins_work, residuals_sq)
            sum_s2 = torch.scatter_add(sum_s2, 0, bins_work, sigma_work ** 2)
            counts = torch.scatter_add(
                counts, 0, bins_work, torch.ones_like(fobs_work)
            )

            # Per-bin empirical residual variance and mean raw variance
            d_per_bin = sum_d / counts.clamp(min=1.0)
            mean_sigma2_per_bin = sum_s2 / counts.clamp(min=1.0)

            # Ratio sigma_eff / sigma_raw, capped to [1, max_inflation]
            ratio_per_bin = torch.sqrt(
                (d_per_bin / mean_sigma2_per_bin.clamp(min=1e-12)).clamp(min=1e-12)
            )
            ratio_per_bin = torch.clamp(ratio_per_bin, 1.0, float(max_inflation))

            # sigma_eff per reflection = sigma_raw * ratio_for_its_bin
            all_bins = self.bins.to(torch.int64)
            ratio_per_refl = ratio_per_bin[all_bins]
            sigma_eff_all = sigma_raw * ratio_per_refl

            # Store per-bin representative sigma_eff (using mean raw sigma in bin)
            sigma_eff_per_bin = torch.sqrt(mean_sigma2_per_bin.clamp(min=1e-12)) * ratio_per_bin
            self.sigma_eff_per_bin.copy_(sigma_eff_per_bin)
            self.sigma_eff.copy_(sigma_eff_all)

            if self.verbose > 1:
                mean_raw = sigma_raw[work_mask].mean().item()
                mean_eff = sigma_eff_all[work_mask].mean().item()
                print(
                    f"  sigma_eff estimation: mean raw={mean_raw:.3f}, "
                    f"mean effective={mean_eff:.3f}, "
                    f"ratio={mean_eff/max(mean_raw,1e-6):.2f}, "
                    f"per-bin ratios={ratio_per_bin.cpu().tolist()}"
                )

            return sigma_eff_all

    def parameters(self, recurse=True):
        """Return parameters, respecting frozen state."""
        if self.frozen:
            return []
        return super().parameters(recurse)

    def forward(
        self,
        fcalc: torch.Tensor,
        use_mask: bool = True,
        f_sol_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the ScalerBase module.

        Parameters
        ----------
        fcalc : torch.Tensor
            Calculated structure factors. Expected shape (N,), an additional
            dimension for batch is possible. N should match the full HKL size.
        use_mask : bool, default True
            Deprecated parameter, kept for backward compatibility.
        f_sol_override : torch.Tensor, optional
            Pre-computed raw solvent structure factors.  When provided, these
            replace the internally-cached ``_f_sol_raw``.  The scaler's
            k_sol / B_sol / phase damping is still applied.  This is used by
            ``CollectionScaler`` to supply mixed (fraction-weighted) solvent
            contributions.

        Returns
        -------
        torch.Tensor
            Scaled structure factors of same shape as input.
        """
        batched = True

        if fcalc.ndim == 1:
            fcalc = fcalc.unsqueeze(0)
            batched = False

        # Determine if we should mask internally or work with full arrays
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
                self._f_sol_raw = self.solvent.get_rec_solvent(self.hkl)

            f_sol_raw = self._f_sol_raw[mask] if apply_internal_mask else self._f_sol_raw

            if hasattr(self, "log_kmask"):
                kmask = torch.exp(self.log_kmask.clamp(min=-10.0, max=10.0))
                kmask = torch.clamp(kmask, min=0.0, max=10.0)
                bins_to_use = self.bins[mask] if apply_internal_mask else self.bins
                kmask_per_refl = kmask[bins_to_use]
                f_sol = kmask_per_refl * f_sol_raw
            else:
                # Inline solvent scaling: k_sol * exp(i*phase) * exp(-B*s²) * f_mask
                # Uses precomputed self._s_half_sq instead of recomputing scattering vectors
                sol = self.solvent
                k_sol = torch.exp(sol.log_k_solvent.clamp(min=-10.0, max=10.0))
                s_half_sq = self._s_half_sq[mask] if apply_internal_mask else self._s_half_sq
                b_factor = torch.exp(
                    (-sol.b_solvent.clamp(min=-500.0, max=500.0) * s_half_sq).clamp(min=-10.0, max=10.0)
                )
                if sol.optimize_phase:
                    f_sol_raw = f_sol_raw * torch.exp(1j * sol.phase_offset)
                f_sol = k_sol * f_sol_raw * b_factor
        else:
            f_sol = torch.tensor(0.0, device=self.device, dtype=fcalc.dtype)

        if hasattr(self, "log_scale") and self.log_scale is not None:
            bins_to_use = self.bins[mask] if apply_internal_mask else self.bins
            K_overall = torch.exp(
                self.log_scale[bins_to_use].clamp(min=-10.0, max=10.0)
            )
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
        Return a dictionary containing the complete state of the ScalerBase.

        This includes:

        - All registered buffers and parameters (via parent class)
        - Scaler-specific metadata (nbins, frozen state, etc.)
        - Solvent model state (if set)

        Note: Data reference is NOT saved (managed separately).

        Parameters
        ----------
        destination : dict, optional
            Optional dict to populate.
        prefix : str, default ''
            Prefix for parameter names.
        keep_vars : bool, default False
            Whether to keep variables in computational graph.

        Returns
        -------
        dict
            Complete state dictionary.
        """
        state = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        state[prefix + "nbins"] = self.nbins
        state[prefix + "verbose"] = self.verbose
        state[prefix + "frozen"] = self.frozen

        if hasattr(self, "solvent") and self.solvent is not None:
            state[prefix + "solvent"] = self.solvent.state_dict()

        return state

    def load_state_dict(self, state_dict, strict=True):
        """
        Load the ScalerBase state from a dictionary.

        Note: This assumes data is already set via __init__ or set_data().

        Parameters
        ----------
        state_dict : dict
            Dictionary containing scaler state.
        strict : bool, default True
            Whether to strictly enforce that keys match.
        """
        self.nbins = state_dict.pop("nbins", 20)
        self.verbose = state_dict.pop("verbose", 1)
        self.frozen = state_dict.pop("frozen", False)

        solvent_state = state_dict.pop("solvent", None)

        result = super().load_state_dict(state_dict, strict=strict)

        if solvent_state is not None and hasattr(self, "solvent") and self.solvent is not None:
            self.solvent.load_state_dict(solvent_state)

        return result

    def save_state(self, path: str):
        """
        Save the complete state of the scaler to a file.

        Parameters
        ----------
        path : str
            Path to save the state dictionary to.
        """
        torch.save(self.state_dict(), path)
        if self.verbose > 0:
            print(f"Saved scaler state to {path}")

    def load_state(self, path: str, strict: bool = True):
        """
        Load the complete state of the scaler from a file.

        Parameters
        ----------
        path : str
            Path to load the state dictionary from.
        strict : bool, default True
            Whether to strictly enforce that keys match.
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state_dict, strict=strict)
        if self.verbose > 0:
            print(f"Loaded scaler state from {path}")
