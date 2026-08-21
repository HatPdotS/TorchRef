"""The fast rotation function engine: obs-side preprocessing, then scoring.

Pipeline (mirrors Phaser ``run_FRF()``):
  1. Resolution mask (both sides).
  2. Wilson normalisation, optionally + French-Wilson + DFAC on obs.
  3. Build LERF1 obs intensity.
  4. Optional per-shell variance reweight on obs intensity.
  5. σA Eterm on calc intensity.
  6. Detect ZSYMM → m-symmetry filter on obs SH coefficients.
  7. Bessel-SH expand both sides.
  8. Cross-correlate on the radial axis → ξ_{lmn}.
  9. Per-β fixed-shape FFT + adaptive sample list bilinear interp →
     adaptive rotation function.
  10. Greedy SO(3) NMS peak finding.
"""
from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import torch

from .data_mr import bessel_sh_expand, cross_correlate_xi
from .peak_finder import find_rotation_peaks
from .preprocessing import (
    apply_shell_variance_weights,
    build_lerf1_intensity,
    detect_zsymm,
    eterm_sigma_a,
    french_wilson_preprocess,
    wilson_normalise,
)
from .sitelist_ang import evaluate_rotation_function
from .types import AdaptiveRotationFunction, RotationPeak

__all__ = ["FastRotationFunction", "phaser_lmax_resolution"]


def phaser_lmax_resolution(
    model_radius_A: float,
    d_min_data: float,
    lmax_cap: int = 48,
):
    """Phaser's coupling of SH bandwidth to rotation-function resolution.

    Phaser source: ``runMR_FRF.cc:407-419``::

        sphereOuter = 2 * mean_radius
        LMAX = ceil(2*pi*sphereOuter / HIRES)      # HIRES = data d_min
        if LMAX odd: LMAX++
        LMAX = min(LMAX, DEF_CLMN_LMAX=100)
        LMAX_RESO = (LMAX capped) ? 2*pi*sphereOuter/LMAX : HIRES

    The point: including data finer than the bandwidth can represent only
    adds aliasing background (the discrete Y_lm are not orthogonal over
    scattered reflections), which buries the symmetry-diluted true peak on
    large / high-symmetry structures. So Phaser either raises LMAX to match
    the resolution, or — when LMAX hits the cap — coarsens the resolution to
    ``LMAX_RESO`` and drops finer reflections (DataMR.cc:984).

    **In practice this is a per-structure high-resolution cutoff.** For real
    protein search models ``LMAX_ideal = ceil(2*pi*2r/d_min)`` is ~100-170, so
    it always hits ``lmax_cap`` and the function reduces to: use ``L=lmax_cap``
    and keep only data coarser than ``d_min_eff = 2*pi*(2r)/lmax_cap`` — the
    finest resolution that bandwidth can faithfully represent for a molecule of
    radius ``r``. Bigger molecule -> coarser cutoff. The variable-L branch only
    matters for tiny models / low-res data (``LMAX_ideal < lmax_cap``).

    **lmax_cap default is 48, NOT Phaser's 100 — for a different reason than
    before.** The contraction now uses ``frf_separate.wigner_d.small_d_stable``
    (J_y eigendecomposition = π/2 / SOFT basis), validated stable+correct to
    l=128, so there is no longer a *numerical* ceiling (the old Edmonds-sum
    ``small_d_packed`` exploded to |d|~1e11 at l>=50). BUT raising L empirically
    makes high-symmetry cases WORSE in this pipeline: at L=100/4.4Å on 4BX9 the
    truth went from #taller=158 (cap=48) to 39084, and σA weighting did not
    suppress it. Cause: at high L the SH modes (~L² per shell) are
    under-determined by the obs sampled on the sparse crystal lattice (~10⁴
    reflections), so high-l coefficients are noise. Phaser avoids this by
    computing the *model* transform on a dense P1-box FFT grid; we sample the
    calc at the crystal lattice. Until that is changed, lmax_cap≈48 is the sweet
    spot. small_d_stable is kept regardless (correct + removes the ceiling, and
    is the prerequisite for any future dense-sampling high-L work).

    Parameters
    ----------
    model_radius_A : float
        The search model's mean atomic radius from its centroid (Å).
        ``sphereOuter = 2 * model_radius_A``.
    d_min_data : float
        High-resolution limit of the data (Å).
    lmax_cap : int
        Hard bandwidth cap. Default 100 (Phaser's DEF_CLMN_LMAX). The stable
        small_d_stable Wigner-d makes higher caps safe at increasing compute cost
        (~L^3); for very large assemblies one may even exceed Phaser's 100.

    Returns
    -------
    (L, d_min_eff) : Tuple[int, float]
        ``L`` is the frf_separate bandwidth (lmax = L-1, even), ``d_min_eff``
        is the resolution to actually use for the expansion.
    """
    sphere_outer = 2.0 * float(model_radius_A)
    lmax = int(math.ceil(2.0 * math.pi * sphere_outer / float(d_min_data)))
    if lmax % 2 != 0:
        lmax += 1
    lmax = min(lmax, int(lmax_cap))
    if lmax >= int(lmax_cap):
        d_min_eff = 2.0 * math.pi * sphere_outer / lmax     # coarsen to match cap
    else:
        d_min_eff = float(d_min_data)
    if lmax >= 256:
        warnings.warn(
            f"phaser_lmax_resolution chose lmax={lmax} (>=256): small_d_stable "
            "is stable here but the contraction cost grows ~l^3 and memory ~l^2; "
            "consider a tighter lmax_cap if this is slow.",
            RuntimeWarning, stacklevel=2,
        )
    return lmax + 1, d_min_eff           # L = lmax + 1 (our bandwidth convention)


def _resolution_mask(
    s_vec: torch.Tensor,
    extra: Tuple[torch.Tensor, ...],
    d_min: Optional[float],
    d_max: Optional[float],
):
    smag = s_vec.norm(dim=-1)
    lo = 1.0 / d_max if d_max is not None else 0.0
    hi = 1.0 / d_min if d_min is not None else float("inf")
    keep = (smag >= lo) & (smag <= hi)
    return s_vec[keep], tuple(e[keep] for e in extra), smag[keep]


class FastRotationFunction:
    """Reusable obs-side preprocessor + SH expansion.

    Instantiate once per (obs reflections, sym, config) tuple, then
    call ``score_model(s_calc, F_calc)`` for each candidate model.
    """

    def __init__(
        self,
        s_obs: torch.Tensor,
        F_obs: torch.Tensor,
        centric_obs: torch.Tensor,
        sym_mats: Optional[torch.Tensor],
        *,
        L: int = 24,
        d_min: Optional[float] = None,
        d_max: Optional[float] = None,
        delta_vrms_A: float = 1.0,
        n_wilson_shells: int = 20,
        sig_F_obs: Optional[torch.Tensor] = None,
        grid_sampling_deg: float = 2.0,
        model_radius_A: Optional[float] = None,
        auto_lmax: bool = False,
        lmax_cap: int = 64,
        compute_dtype: Optional[torch.dtype] = None,
    ):
        self.device = s_obs.device
        self.real_dtype = s_obs.dtype
        self.compute_dtype = compute_dtype

        # Phaser-faithful coupling of bandwidth to resolution (runMR_FRF.cc:408).
        # Overrides L and d_min so the SH expansion is not flooded with data
        # finer than L can represent (the high-symmetry failure mode).
        if auto_lmax:
            if model_radius_A is None or d_min is None:
                raise ValueError(
                    "auto_lmax=True requires model_radius_A and d_min (data res)."
                )
            L, d_min = phaser_lmax_resolution(model_radius_A, d_min, lmax_cap=lmax_cap)

        self.L = L
        self.d_min = d_min
        self.d_max = d_max
        self.delta_vrms_A = delta_vrms_A
        self.n_wilson_shells = n_wilson_shells
        self.grid_sampling_deg = grid_sampling_deg

        # 1. Resolution mask on obs.
        extras = (F_obs, centric_obs)
        if sig_F_obs is not None:
            extras = extras + (sig_F_obs,)
        s_obs, extras, smag_obs = _resolution_mask(s_obs, extras, d_min, d_max)
        F_obs, centric_obs = extras[0], extras[1]
        if sig_F_obs is not None:
            sig_F_obs = extras[2]

        if s_obs.shape[0] < n_wilson_shells * 5:
            raise ValueError(
                f"Too few obs reflections ({s_obs.shape[0]}) for "
                f"{n_wilson_shells} Wilson shells in [{d_min}, {d_max}] Å."
            )

        # 2. Bessel scaling — Phaser's lmax · d_min (DataMR.cc:1107).
        #    `bessel_h_scale = 2 pi R_patt` is the Patterson integration radius
        #    (the chi_Omega sphere): the Bessel argument is
        #    `h = bessel_h_scale |s|`, so the radial basis represents the
        #    Patterson out to `R_patt = bessel_h_scale / (2 pi)`. Under
        #    `auto_lmax` that comes to `R_patt ~ sphereOuter = 2 x mean radius`.
        if d_min is None:
            raise ValueError("d_min is required to set the Bessel scaling")
        lmax = L - 1
        lmax_even = lmax if lmax % 2 == 0 else lmax - 1
        self.bessel_h_scale = float(lmax_even) * float(d_min)

        # 3. Wilson normalisation. With sigmas, through the French-Wilson
        #    posterior, which does its own per-shell normalisation and handles
        #    the axial reflections; without them, plain per-shell Wilson.
        if sig_F_obs is not None:
            fw = french_wilson_preprocess(
                F_obs, sig_F_obs, smag_obs, centric_obs,
                n_wilson_shells=n_wilson_shells,
            )
            eEobs, dfac = fw["eEobs"], fw["DFAC"]
        else:
            eEobs, _ = wilson_normalise(F_obs, smag_obs, n_wilson_shells)
            dfac = torch.ones_like(eEobs)

        # 4. LERF1 obs intensity, and the per-shell variance reweight.
        intensity_obs = build_lerf1_intensity(
            eEobs, centric_obs, dfac=dfac, use_centric_weight=True,
        )
        intensity_obs = apply_shell_variance_weights(
            intensity_obs, smag_obs, n_var_shells=n_wilson_shells,
        )

        # 5. ZSYMM m-filter on the obs SH coefficients. The calc side is never
        #    filtered -- see score_model.
        zsymm = detect_zsymm(sym_mats)

        # 6. Bessel-SH expand the obs side.
        self._c_obs = bessel_sh_expand(
            s_obs, intensity_obs.to(self.real_dtype),
            L=L, bessel_h_scale=self.bessel_h_scale,
            zsymm=zsymm, enforce_friedel=True,
            compute_dtype=self.compute_dtype,
        )


    def score_model(
        self,
        s_calc: torch.Tensor,
        F_calc: torch.Tensor,
        *,
        n_peaks: int = 500,
        sigma_threshold: float = -5.0,
        apply_bulk_solvent: bool = False,
        solvent_fsol: float = 0.95,
        solvent_bsol: float = 300.0,
    ) -> Tuple[AdaptiveRotationFunction, List[RotationPeak]]:
        # Resolution mask + Wilson + Eterm on calc.
        s_calc, (F_calc,), smag_calc = _resolution_mask(
            s_calc, (F_calc,), self.d_min, self.d_max,
        )
        E_calc, _ = wilson_normalise(F_calc, smag_calc, self.n_wilson_shells)
        eterm = eterm_sigma_a(smag_calc, self.delta_vrms_A)
        # Optional Babinet bulk-solvent factor: Phaser folds it into σ_A as
        # `σ_A_eff = solTerm(s²) · Luzzati(s², vrms)` (EnsemblePDB.cc:96-100).
        # Default OFF; flip after v25 validates.
        if apply_bulk_solvent:
            from .preprocessing import bulk_solvent_factor
            sol = bulk_solvent_factor(
                smag_calc, fsol=solvent_fsol, bsol=solvent_bsol,
            ).to(eterm.dtype)
            eterm = eterm * sol
        intensity_calc = (eterm * eterm) * (E_calc * E_calc - 1.0)

        # Bessel-SH expand calc. The calc is NEVER m-filtered (zsymm=1): the
        # model carries no crystal symmetry, and projecting it onto the obs's
        # invariant-m subspace destroys the orientation information that
        # discriminates truth — a calc-side m-filter was tested and is strongly
        # harmful on high-symmetry cases (3K7M rank 8→92), so the knob was removed.
        c_calc = bessel_sh_expand(
            s_calc, intensity_calc.to(self.real_dtype),
            L=self.L, bessel_h_scale=self.bessel_h_scale,
            zsymm=1, enforce_friedel=True,
            compute_dtype=self.compute_dtype,
        )

        # 8. Cross-correlate over the radial axis.
        xi = cross_correlate_xi(self._c_obs, c_calc)

        # 9. FRF on the adaptive SO(3) grid.
        arf = evaluate_rotation_function(xi, grid_sampling_deg=self.grid_sampling_deg)

        # 10. Peak finding (greedy SO(3) NMS).
        peaks = find_rotation_peaks(
            arf,
            n_peaks=n_peaks,
            sigma_threshold=sigma_threshold,
            nms_radius_deg=max(2.0 * self.grid_sampling_deg, 6.0),
        )
        return arf, peaks
