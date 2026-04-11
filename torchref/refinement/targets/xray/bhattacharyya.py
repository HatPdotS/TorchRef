"""
Bhattacharyya overlap X-ray target with first-principles model uncertainty.

Implementation notes:
- Reflection axis is kept fully resolved.
- B-factors are histogrammed onto a LOG-SPACED grid (B follows a lognormal-
  ish distribution per Wilson statistics, so log-space binning captures the
  B distribution accurately at all magnitudes).
- A precomputed (b_grid, N_refl) exp table lets every per-cycle operation
  reduce to a matmul against ``exp_table``. No N_atoms × N_refl tensor is
  ever materialised.

Computes the Bhattacharyya distance between two Gaussian distributions:
- data:  N(F_obs, sigma_d^2)
- model: N(|F_calc|, sigma_m^2)

sigma_m derivation
------------------
Per-atom variances from the diagonal data-only Fisher info, with the
phase-averaging 1/2 factor included:

    Var(x_j,k) = 6  / ( (2π)^2 * f_j^2 * g_w(B_j) )
    Var(B_j)   = 32 / ( f_j^2 * g_4(B_j) )
    g_w(B)     = Σ_h (|s_h|^2 / σ_d^2(h)) * exp(-2 B s_h^2/4)
    g_4(B)     = Σ_h ( s_h^4  / σ_d^2(h)) * exp(-2 B s_h^2/4)

Propagating both into F_calc (the f_j^2 cancels between numerator and
denominator in each term), one gets two additive contributions:

    σ_m²(h) = 3 · |s_h|² * Σ_j exp(-2 B_j s_h²/4) / g_w(B_j)   [position]
            +     s_h⁴  * Σ_j exp(-2 B_j s_h²/4) / g_4(B_j)   [B-factor]

The B-term has an extra s² weighting (s⁴ vs |s|²), so it is negligible
at low resolution and dominant at high resolution — consistent with the
Gaussian-fit intuition that the error on the width is ~1/√2 times the
error on the mean, but with resolution-dependent weighting.

A global multiplier ``sigma_m_scale`` is applied to σ_m for empirical
tuning of the overall magnitude without changing the per-reflection
structure:

    σ_m(h) → sigma_m_scale · σ_m(h)

See paper/design_doc_overlap_loss.md for the Bhattacharyya formulation.
"""

import math
import torch
from typing import TYPE_CHECKING, Dict

from torchref.utils.stats import (
    VERBOSITY_DEBUG,
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

    For each reflection h:

        L_h = (F_obs - |F_calc|)^2 / (4 * (sigma_d^2 + sigma_m^2))
              + 0.5 * log( (sigma_d^2 + sigma_m^2) / (2 * sigma_d * sigma_m) )

    Total loss: sum over reflections.

    **Model variance (diagonal Fisher + global scale)**

    Per-atom positional variance from the diagonal data-only LS:

        Var(x_j) = 3 / ( (2π)^2 * g_w(B_j) )
        g_w(B)   = Σ_h (|s_h|^2 / sigma_d^2(h)) * exp(-2 B s_h^2/4)

    Propagated into F_calc (unit scatterer):

        sigma_m^2(h) = scale^2 * 3 * |s_h|^2 *
                       Σ_j exp(-2 B_j s_h^2/4) / g_w(B_j)

    The diagonal formula under-estimates the absolute scale because the
    true CRLB ignores parameter correlations. A global multiplier
    ``sigma_m_scale`` is applied to σ_m (not σ_m²) so the per-reflection
    shape is preserved and only the overall magnitude is tuned.

    **Update schedule**

    - exp_table[b, h]: precomputed once.
    - g_w(B): precomputed once (static — depends only on data).
    - sigma_m(h): recomputed each forward() call from the current B
      histogram. Gradients flow only through F_calc.

    Parameters
    ----------
    data : ReflectionData, optional
    model : Model, optional
    scaler : Scaler, optional
    use_work_set : bool, optional
        Use work set (default) or test set for loss.
    sigma_m_scale : float, optional
        Global multiplier applied to σ_m. Default 1.0 (bare formula).
    b_grid_min, b_grid_max, b_grid_n : float, int, optional
        Grid for g_w(B) lookup (A^2). Default 1-200 A^2, 100 points.
    verbose : int, optional

    Notes
    -----
    Currently uses isotropic atoms only; aniso atoms are ignored.
    Hydrogens follow the model's ``exclude_H_from_sf`` setting via
    ``model._iso_indices``.
    """

    def __init__(
        self,
        data: "ReflectionData" = None,
        model: "Model" = None,
        scaler: "Scaler" = None,
        use_work_set: bool = True,
        sigma_m_scale: float = 1.0,
        sigma_weighting: str = "per_refl",
        info_sum_mode: str = "g_w",
        scatterer_profile: str = "unit",
        b_grid_min: float = 1.0,
        b_grid_max: float = 200.0,
        b_grid_n: int = 100,
        verbose: int = 0,
        **kwargs,
    ):
        if sigma_weighting not in ("per_refl", "const"):
            raise ValueError(
                f"sigma_weighting must be 'per_refl' or 'const', got "
                f"{sigma_weighting!r}"
            )
        if info_sum_mode not in ("g_w", "n_eff"):
            raise ValueError(
                f"info_sum_mode must be 'g_w' or 'n_eff', got {info_sum_mode!r}"
            )
        if scatterer_profile not in ("unit", "protein_rep"):
            raise ValueError(
                f"scatterer_profile must be 'unit' or 'protein_rep', got "
                f"{scatterer_profile!r}"
            )
        self._sigma_weighting = sigma_weighting
        self._info_sum_mode = info_sum_mode
        self._scatterer_profile = scatterer_profile
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
        # log-spaced B grid (B ~ lognormal per Wilson statistics)
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
        self.register_buffer("s_4_per_refl", torch.empty(0))   # (N_refl,) |s|^4
        # Per-reflection f²(s) for the scatterer profile (ones if 'unit').
        self.register_buffer("f_sq_per_refl", torch.empty(0))  # (N_refl,)
        # Static Fisher-info denominator tables:
        #   g_w_table[b] = Σ_h (f²(s)·|s|²/σ²) · exp(-2 b s²/4)   (position term)
        #   g_4_table[b] = Σ_h (f²(s)·|s|⁴/σ²) · exp(-2 b s²/4)   (B-factor term)
        self.register_buffer("g_w_table", torch.empty(0))      # (b_grid_n,)
        self.register_buffer("g_4_table", torch.empty(0))      # (b_grid_n,)
        self.register_buffer("sigma_d_mean", torch.tensor(0.0))
        self._initialized = False

    # ------------------------------------------------------------------
    # Cache initialisation (called once, on first forward)
    # ------------------------------------------------------------------

    def _initialize_cache(self) -> None:
        """Precompute the exp_table[b, h] and refl_info_weight[h].

        Reflection axis is kept fully resolved (no resolution binning);
        only the B-factor axis is discretised onto the log-spaced b_grid
        (default 100 points). Memory cost: exp_table is (b_grid_n × N_refl).
        For 100 b values and 100 k reflections that's 40 MB float32.
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

        # f²(s) per reflection from the scatterer profile. 'unit' gives 1;
        # 'protein_rep' uses carbon ITC92 coefficients as a representative.
        if self._scatterer_profile == "unit":
            f_sq = torch.ones_like(s_sq)
        else:  # "protein_rep": ITC92 for C
            a_c = torch.tensor(
                [2.31, 1.02, 1.5886, 0.865], device=device, dtype=dtype
            )
            b_c = torch.tensor(
                [20.8439, 10.2075, 0.5687, 51.6512], device=device, dtype=dtype
            )
            c_c = 0.2156
            # f(s) = Σ a_i · exp(-b_i · s_half_sq) + c, where s_half_sq = sin²θ/λ²
            f_per_refl = c_c + (
                a_c.view(-1, 1)
                * torch.exp(-b_c.view(-1, 1) * s_half_sq.view(1, -1))
            ).sum(dim=0)
            f_sq = f_per_refl ** 2
        self.f_sq_per_refl = f_sq

        # Per-reflection Fisher weights with invalids zeroed out.
        if self._sigma_weighting == "per_refl":
            inv_sd2 = (1.0 / (sigma_data ** 2).clamp(min=1e-12)) * valid_f
        else:  # "const": use constant 1/<σ_d>² for all valid reflections
            mean_sigma_sq = (
                (sigma_data ** 2 * valid_f).sum() / valid_f.sum().clamp(min=1.0)
            )
            inv_sd2 = valid_f / mean_sigma_sq.clamp(min=1e-12)
        refl_info_weight_w = s_sq * f_sq * inv_sd2         # (N_refl,) — g_w
        refl_info_weight_4 = s_sq * s_sq * f_sq * inv_sd2  # (N_refl,) — g_4

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

        # Zero invalid reflections so they contribute 0 in every matmul.
        invalid = (~validity).nonzero(as_tuple=True)[0]
        if invalid.numel() > 0:
            exp_table[:, invalid] = 0.0
        self.exp_table = exp_table

        if self._info_sum_mode == "g_w":
            # Fisher-info sums: g_w(B) = Σ_h (|s|²/σ²)·exp(-2Bs²/4),
            #                   g_4(B) = Σ_h (|s|⁴/σ²)·exp(-2Bs²/4).
            self.g_w_table = torch.matmul(exp_table, refl_info_weight_w)
            self.g_4_table = torch.matmul(exp_table, refl_info_weight_4)
        else:  # "n_eff": Kish participation ratio × mean Fisher weight
            # N_eff(B) = (Σ_h exp(-2Bs²/4))² / Σ_h exp(-4Bs²/4)
            exp_sum = exp_table.sum(dim=-1)                      # (b_grid_n,)
            exp_sq_sum = (exp_table ** 2).sum(dim=-1)            # (b_grid_n,)
            n_eff = (exp_sum ** 2) / exp_sq_sum.clamp(min=1e-30) # (b_grid_n,)
            # Scale N_eff by the mean per-reflection Fisher weight so σ_m
            # has the same dimensional prefactor as in the g_w mode.
            n_valid = valid_f.sum().clamp(min=1.0)
            mean_w = refl_info_weight_w.sum() / n_valid
            mean_4 = refl_info_weight_4.sum() / n_valid
            self.g_w_table = n_eff * mean_w
            self.g_4_table = n_eff * mean_4

        self._initialized = True

        if self.verbose > 1:
            n_valid = valid_f.sum().item()
            print(
                f"  BhattacharyyaXrayTarget cache: n_refl={int(n_valid)}, "
                f"b_grid_n={b_grid_n}, "
                f"sigma_d_mean={self.sigma_d_mean.item():.3f}"
            )

    # ------------------------------------------------------------------
    # B-histogram and sigma_m computation (Wilson-share / option 3)
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

    def _build_b_histogram(self, b: torch.Tensor) -> torch.Tensor:
        """
        Soft histogram of atomic B-factors over the LOG-spaced b_grid.

        Each atom contributes (1-frac) and (frac) to its two log-B
        neighbours, so Σ_b hist[b]·f(b) reproduces Σ_atoms f(B_atom) with
        sub-grid accuracy for smooth f.
        """
        n_b = self.b_grid.shape[0]
        idx_lo, frac = self._log_b_index(b)
        hist = torch.zeros(n_b, device=b.device, dtype=b.dtype)
        hist.scatter_add_(0, idx_lo, 1.0 - frac)
        hist.scatter_add_(0, idx_lo + 1, frac)
        return hist

    def _sigma_m_sq_per_refl(self) -> torch.Tensor:
        """
        Per-reflection σ_m² with both position and B-factor error:

            σ_m²(h) = scale² · f²(s_h) · [
                3 · |s_h|² · Σ_j exp(-2 B_j s_h²/4) / g_w(B_j)    (position)
              +     s_h⁴  · Σ_j exp(-2 B_j s_h²/4) / g_4(B_j)    (B-factor)
            ]

        f²(s_h) is 1 for the 'unit' scatterer profile and the squared C
        scattering factor for 'protein_rep'. g_w(B) and g_4(B) are static
        precomputed 1-D tables whose sums include the f²(s) weight.
        """
        model = self._model
        iso_idx = model._iso_indices
        b_iso = model.adp()[iso_idx]                              # (N_iso,)
        hist = self._build_b_histogram(b_iso)                     # (b_grid_n,)

        weighted_w = hist / self.g_w_table.clamp(min=1e-30)
        weighted_4 = hist / self.g_4_table.clamp(min=1e-30)
        atom_factor_w = torch.matmul(weighted_w, self.exp_table)  # (N_refl,)
        atom_factor_4 = torch.matmul(weighted_4, self.exp_table)  # (N_refl,)

        sigma_m_sq = self.f_sq_per_refl * (
            3.0 * self.s_sq_per_refl * atom_factor_w
            + self.s_4_per_refl * atom_factor_4
        ).clamp(min=1e-12)
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
        F_calc_amp = torch.abs(F_calc)

        # sigma_m is non-differentiable: refreshed per call from the
        # current atomic B distribution via the Wilson-share coupling.
        with torch.no_grad():
            sigma_m = self._sigma_m_per_refl()

        eps = 1e-6
        sigma_d_safe = torch.clamp(sigma_d, min=eps)
        sigma_m_safe = torch.clamp(sigma_m, min=eps)

        var_d = sigma_d_safe ** 2
        var_m = sigma_m_safe ** 2
        var_sum = var_d + var_m

        diff = F_obs - F_calc_amp
        l_mean = (diff ** 2) / (4.0 * var_sum)
        l_var = 0.5 * torch.log(var_sum / (2.0 * sigma_d_safe * sigma_m_safe))

        return ((l_mean + l_var) * mask).sum()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self, fcalc: torch.Tensor = None) -> Dict[str, StatEntry]:
        """Add sigma_m/sigma_d diagnostics (per-reflection mean & max ratio)."""
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
