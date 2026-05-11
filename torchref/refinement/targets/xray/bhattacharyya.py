"""
Bhattacharyya overlap X-ray target with first-principles model uncertainty.

Loss
----
For each reflection h:

    L_h = (F_obs - |F_calc|)² / (4 · (σ_d² + σ_m²))
        + 0.5 · log( (σ_d² + σ_m²) / (2 σ_d σ_m) )

Total: L = Σ_h L_h. 

sigma_m derivation
------------------
σ_m is derived from per-atom positional and B-factor uncertainty under the
diagonal data-only Fisher information. Atoms are binned on a 2-D grid of
(element type k, B-factor b). Within each element type the ITC92 scattering
factor f_k(s) is a 1-D function of resolution.

Per-element Fisher-info sums (static — depend only on data and scattering):

    f_k²(s_h)  = ( Σ_m A_km · exp(-B_km · s_half_sq) )²
    g_w(k, b)  = Σ_h (|s|² · f_k²(s_h) / <σ_d²>) · exp(-2 b · s_half_sq)
    g_4(k, b)  = Σ_h (|s|⁴ · f_k²(s_h) / <σ_d²>) · exp(-2 b · s_half_sq)

Each refinement cycle, the current atomic B-factors are soft-histogrammed by
element into ``hist[k, b]``. Then:

    σ_m²(h) = scale² · Σ_k f_k²(s_h) · [
        3 · |s|² · (hist[k] / g_w(k)) @ exp_table_shared[:, h]    (position)
      +     s⁴  · (hist[k] / g_4(k)) @ exp_table_shared[:, h]    (B-factor)
    ]

The outer f_k²(s_h) comes from forward propagation of Var(x_j, B_j) into
F_calc. The f² inside g_w/g_4 cancels one of the two factors that arise from
Var(x_j) ∝ 1/(f² · g), leaving the outer f_k² factor.

"""

import math
import torch
from typing import TYPE_CHECKING, Dict

from torchref.base.targets._dispatch import use_triton
from torchref.base.targets.xray_bhattacharyya import bhattacharyya_xray_loss_math
from torchref.utils.stats import (
    VERBOSITY_DETAILED,
    VERBOSITY_STANDARD,
    StatEntry,
    stat,
)

from .base import XrayTarget

if TYPE_CHECKING:
    from torchref.io import ReflectionData
    from torchref.model.model import Model
    from torchref.scaling.scaler_base import Scaler


class BhattacharyyaXrayTarget(XrayTarget):
    """
    X-ray target based on the Bhattacharyya overlap between data and
    model Gaussians.

    Parameters
    ----------
    data : ReflectionData, optional
    model : Model, optional
    scaler : Scaler, optional
    use_work_set : bool, optional
        Use work set (default) or test set for loss.
    sigma_m_scale : float, optional
        Global multiplier applied to σ_m. Default 1.0.
    b_grid_min, b_grid_max, b_grid_n : float, int, optional
        Log-spaced B-factor grid for σ_m computation.
        Default 1–200 Å², 100 points.
    verbose : int, optional
    """

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "Model" = None,
        scaler: "Scaler" = None,
        use_work_set: bool = True,
        sigma_m_scale: float = 1.0,
        b_grid_min: float = 1.0,
        b_grid_max: float = 200.0,
        b_grid_n: int = 100,
        verbose: int = 0,
        **kwargs,
    ):
        kwargs.pop("sigma_mode", None)
        kwargs.pop("n_bins", None)  # legacy kwarg ignored
        super().__init__(
            data=data,
            model=model,
            scaler=scaler,
            use_work_set=use_work_set,
            sigma_mode="raw",
            verbose=verbose,
        )
        # log-spaced B grid
        log_b_grid = torch.linspace(
            math.log(b_grid_min), math.log(b_grid_max), b_grid_n
        )
        self.register_buffer("log_b_grid", log_b_grid)
        self.register_buffer("b_grid", torch.exp(log_b_grid))
        self._log_b_min = float(log_b_grid[0].item())
        self._log_b_max = float(log_b_grid[-1].item())
        self._log_b_step = (self._log_b_max - self._log_b_min) / (b_grid_n - 1)
        # Global σ_m scale (tunable, non-learnable)
        self.register_buffer(
            "sigma_m_scale", torch.tensor(float(sigma_m_scale))
        )
        # Populated by _initialize_cache() on first forward()
        self.register_buffer("exp_table", torch.empty(0))      # (b_grid_n, N_refl)
        self.register_buffer("s_sq_per_refl", torch.empty(0))  # (N_refl,)
        self.register_buffer("s_4_per_refl", torch.empty(0))   # (N_refl,)
        self.register_buffer("f_sq_kh", torch.empty(0))        # (K, N_refl)
        self.register_buffer("g_w_table", torch.empty(0))      # (K, b_grid_n)
        self.register_buffer("g_4_table", torch.empty(0))      # (K, b_grid_n)
        self.register_buffer("atom_to_element", torch.empty(0, dtype=torch.long))
        self.register_buffer("sigma_d_mean", torch.tensor(0.0))
        self._initialized = False

    # ------------------------------------------------------------------
    # Cache initialisation (called once, on first forward)
    # ------------------------------------------------------------------

    def _initialize_cache(self) -> None:
        """Precompute f_sq_kh, exp_table, g_w_table, g_4_table, atom_to_element.

        Reflection axis is kept fully resolved; B-factor axis is discretised
        on the log-spaced b_grid. The element-type axis groups iso atoms by
        unique (A, B) ITC92 rows.
        """
        if self._data is None or self._scaler is None or self._model is None:
            raise RuntimeError(
                "BhattacharyyaXrayTarget requires data, model and scaler "
                "to be set before forward()."
            )

        device = self._scaler._s_half_sq.device
        dtype = self._scaler._s_half_sq.dtype

        s_half_sq = self._scaler._s_half_sq.to(device=device, dtype=dtype)
        s_sq = 4.0 * s_half_sq

        _, _, sigma_raw, _ = self._data(mask=False)
        sigma_data = sigma_raw
        if hasattr(sigma_data, "get_data"):
            sigma_data = sigma_data.get_data()
        sigma_data = sigma_data.to(device=device, dtype=dtype)
        validity = self._data.masks().to(torch.bool).to(device)
        valid_f = validity.to(dtype)

        self.s_sq_per_refl = s_sq
        self.s_4_per_refl = s_sq * s_sq
        self.sigma_d_mean = (sigma_data * valid_f).sum() / valid_f.sum().clamp(
            min=1.0
        )
        mean_sigma_sq = (
            (sigma_data ** 2 * valid_f).sum() / valid_f.sum().clamp(min=1.0)
        ).clamp(min=1e-12)

        # --- element-type grid -----------------------------------------
        A_iso, B_iso = self._model.get_scattering_params_iso()  # (n_iso, 5)
        A_iso = A_iso.to(device=device, dtype=dtype)
        B_iso = B_iso.to(device=device, dtype=dtype)
        # Row-hash to identify unique element types (one row per atom).
        ab_rows = torch.cat([A_iso, B_iso], dim=-1)              # (n_iso, 10)
        unique_rows, atom_to_element = torch.unique(
            ab_rows, dim=0, return_inverse=True
        )                                                        # (K, 10), (n_iso,)
        K = unique_rows.shape[0]
        element_A = unique_rows[:, :5]                           # (K, 5)
        element_B = unique_rows[:, 5:]                           # (K, 5)
        self.atom_to_element = atom_to_element.to(device=device)

        # f_k(s_h) = Σ_m A_km · exp(-B_km · s_half_sq)            (K, N_refl)
        # Vectorised over (K, 5, N_refl)
        expon_f = (
            -element_B.unsqueeze(-1) * s_half_sq.view(1, 1, -1)
        ).clamp(min=-80.0, max=80.0)
        f_kh = (element_A.unsqueeze(-1) * torch.exp(expon_f)).sum(dim=1)  # (K, N_refl)
        self.f_sq_kh = f_kh * f_kh

        # --- shared exp_table (b_grid_n, N_refl) -----------------------
        b_grid = self.b_grid.to(device=device, dtype=dtype)
        b_grid_n = b_grid.shape[0]
        n_refl = s_sq.shape[0]
        exp_table = torch.empty(b_grid_n, n_refl, device=device, dtype=dtype)

        chunk = 32
        for start in range(0, b_grid_n, chunk):
            end = min(start + chunk, b_grid_n)
            b_chunk = b_grid[start:end]
            expon = (
                -2.0 * b_chunk.unsqueeze(-1) * s_half_sq.unsqueeze(0)
            ).clamp(min=-80.0, max=80.0)
            exp_table[start:end] = torch.exp(expon)

        invalid = (~validity).nonzero(as_tuple=True)[0]
        if invalid.numel() > 0:
            exp_table[:, invalid] = 0.0
        self.exp_table = exp_table

        # --- per-element Fisher-info tables: (K, b_grid_n) -------------
        # w_w_k[h] = |s|²·f_k²(s_h)/<σ²> · valid_f
        # w_4_k[h] = |s|⁴·f_k²(s_h)/<σ²> · valid_f
        inv_sig_sq_valid = valid_f / mean_sigma_sq
        w_w = s_sq.unsqueeze(0) * self.f_sq_kh * inv_sig_sq_valid.unsqueeze(0)   # (K, N_refl)
        w_4 = (s_sq * s_sq).unsqueeze(0) * self.f_sq_kh * inv_sig_sq_valid.unsqueeze(0)
        # g_?_table[k, b] = Σ_h w_?[k, h] · exp_table[b, h]
        # Use matmul: (K, N_refl) @ (N_refl, b_grid_n) → (K, b_grid_n)
        exp_table_T = exp_table.transpose(0, 1)                  # (N_refl, b_grid_n)
        self.g_w_table = torch.matmul(w_w, exp_table_T)          # (K, b_grid_n)
        self.g_4_table = torch.matmul(w_4, exp_table_T)          # (K, b_grid_n)

        self._initialized = True

        if self.verbose > 1:
            n_valid = valid_f.sum().item()
            print(
                f"  BhattacharyyaXrayTarget cache: n_refl={int(n_valid)}, "
                f"K_elements={K}, b_grid_n={b_grid_n}, "
                f"sigma_d_mean={self.sigma_d_mean.item():.3f}"
            )

    # ------------------------------------------------------------------
    # B-histogram and sigma_m computation
    # ------------------------------------------------------------------

    def _log_b_index(self, b: torch.Tensor):
        """Return (idx_lo, frac) for linear interpolation in LOG-B space."""
        log_b = torch.log(b.clamp(min=1e-6))
        log_b_clamped = torch.clamp(log_b, self._log_b_min, self._log_b_max)
        idx_f = (log_b_clamped - self._log_b_min) / self._log_b_step
        n_b = self.b_grid.shape[0]
        idx_lo = idx_f.floor().long().clamp(0, n_b - 2)
        frac = (idx_f - idx_lo.to(idx_f.dtype)).clamp(0.0, 1.0)
        return idx_lo, frac

    def _build_element_b_histogram(self, b: torch.Tensor) -> torch.Tensor:
        """
        Soft 2-D histogram of iso atoms over (element_type, log_B).

        Parameters
        ----------
        b : torch.Tensor
            (n_iso,) B-factors of the iso atoms.

        Returns
        -------
        hist : torch.Tensor
            (K, b_grid_n) soft histogram. Each atom contributes (1-frac) and
            (frac) to its two log-B neighbours.
        """
        K = self.f_sq_kh.shape[0]
        n_b = self.b_grid.shape[0]
        idx_lo, frac = self._log_b_index(b)
        elem = self.atom_to_element
        # Flat index into (K, n_b) grid: elem * n_b + b_idx.
        flat_lo = elem * n_b + idx_lo
        flat_hi = elem * n_b + (idx_lo + 1)
        hist = torch.zeros(K * n_b, device=b.device, dtype=b.dtype)
        hist.scatter_add_(0, flat_lo, 1.0 - frac)
        hist.scatter_add_(0, flat_hi, frac)
        return hist.view(K, n_b)

    def _sigma_m_sq_per_refl(self) -> torch.Tensor:
        """
        Per-reflection σ_m² — see module docstring for derivation.

        Returns
        -------
        torch.Tensor
            (N_refl,) σ_m² with ``sigma_m_scale`` applied.
        """
        b_iso = self._model.adp()[self._model._iso_indices]       # (n_iso,)
        hist = self._build_element_b_histogram(b_iso)             # (K, b_grid_n)

        weighted_w = hist / self.g_w_table.clamp(min=1e-30)       # (K, b_grid_n)
        weighted_4 = hist / self.g_4_table.clamp(min=1e-30)
        atom_factor_w = torch.matmul(weighted_w, self.exp_table)  # (K, N_refl)
        atom_factor_4 = torch.matmul(weighted_4, self.exp_table)  # (K, N_refl)

        per_type = self.f_sq_kh * (
            3.0 * self.s_sq_per_refl.unsqueeze(0) * atom_factor_w
            + self.s_4_per_refl.unsqueeze(0) * atom_factor_4
        )                                                         # (K, N_refl)
        sigma_m_sq = per_type.sum(dim=0).clamp(min=1e-12)         # (N_refl,)
        return (self.sigma_m_scale ** 2) * sigma_m_sq

    def _sigma_m_per_refl(self) -> torch.Tensor:
        return torch.sqrt(self._sigma_m_sq_per_refl())

    # ------------------------------------------------------------------
    # Forward: Bhattacharyya overlap loss
    # ------------------------------------------------------------------

    def forward(self, fcalc: torch.Tensor = None) -> torch.Tensor:
        if not self._initialized:
            self._initialize_cache()

        F_obs, F_calc, sigma_d, _centric, mask = self.get_data(fcalc=fcalc)

        # σ_m is non-differentiable — refreshed each call from the current
        # atomic B distribution.
        with torch.no_grad():
            sigma_m = self._sigma_m_per_refl()

        if use_triton(F_calc, F_obs, sigma_d, sigma_m):
            from torchref.base.targets.triton.xray_bhattacharyya import (
                bhattacharyya_xray_loss_math_triton,
            )
            return bhattacharyya_xray_loss_math_triton(
                F_obs, F_calc, sigma_d, sigma_m, mask
            )
        return bhattacharyya_xray_loss_math(
            F_obs, F_calc, sigma_d, sigma_m, mask
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self, fcalc: torch.Tensor = None) -> Dict[str, StatEntry]:
        """Add σ_m/σ_d diagnostics (mean and max ratio over all reflections)."""
        base = super().stats(fcalc=fcalc)
        if not self._initialized:
            return base

        with torch.no_grad():
            sigma_m = self._sigma_m_per_refl()
            _, _, sigma_raw_all, _ = self._data()
            if hasattr(sigma_raw_all, "get_data"):
                sigma_d_all = sigma_raw_all.get_data()
            else:
                sigma_d_all = sigma_raw_all
            ratio = sigma_m / sigma_d_all.clamp(min=1e-6)

        base["sigma_m_over_d_mean"] = stat(ratio.mean().item(), VERBOSITY_STANDARD)
        base["sigma_m_over_d_max"] = stat(ratio.max().item(), VERBOSITY_DETAILED)
        return base
