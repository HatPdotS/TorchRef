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
    lmax_cap: int = 64,
):
    """Couple the spherical-harmonic bandwidth to the rotation-function
    resolution, as Phaser does (``runMR_FRF.cc:427-448``)::

        sphereOuter = 2 * mean_radius
        LMAX = ceil(2 pi sphereOuter / HIRES)      # HIRES = the data's d_min
        if LMAX is odd: LMAX += 1
        LMAX = min(LMAX, lmax_cap)
        LMAX_RESO = (LMAX hit the cap) ? 2 pi sphereOuter / LMAX : HIRES

    Data finer than the bandwidth can represent contributes aliasing rather than
    signal -- the discrete ``Y_lm`` are not orthogonal over scattered
    reflections -- and that background buries the symmetry-diluted true peak on
    large or high-symmetry structures. So either the bandwidth rises to meet the
    resolution, or, once it is capped, the resolution is coarsened to
    ``LMAX_RESO`` and the finer reflections are dropped (``DataMR.cc:984``).

    For real protein search models ``ceil(2 pi 2r / d_min)`` is 100 to 170, so
    the cap binds and this reduces to: use ``L = lmax_cap``, and keep only data
    coarser than ``2 pi (2r) / lmax_cap`` -- the finest resolution that
    bandwidth can carry for a molecule of that radius. Bigger molecule, coarser
    cutoff. The variable-``L`` branch matters only for small models or
    low-resolution data.

    Parameters
    ----------
    model_radius_A : float
        Mean distance of the model's atoms from its centroid (Angstrom).
        ``sphereOuter = 2 * model_radius_A``.
    d_min_data : float
        High-resolution limit of the data (Angstrom).
    lmax_cap : int
        Bandwidth ceiling. The production value lives in
        :data:`torchref.experimental.alignment.rotation_search.LMAX_CAP`, which
        carries the measurement behind it; this default exists only for direct
        callers. Cost grows about as ``l^3`` in time and ``l^2`` in memory.

    Returns
    -------
    (L, d_min_eff) : Tuple[int, float]
        ``L`` is the bandwidth in this package's convention (``lmax = L - 1``,
        even); ``d_min_eff`` is the resolution to expand at.
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
            f"phaser_lmax_resolution chose lmax={lmax} (>=256): the Wigner "
            f"contraction is numerically fine here, but its cost grows about as "
            f"l^3 in time and l^2 in memory. Consider a tighter lmax_cap.",
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
        asu_idx: Optional[torch.Tensor] = None,
        s_mag_asu: Optional[torch.Tensor] = None,
    ):
        self.device = s_obs.device

        # `L` and `d_min` arrive already coupled: the caller runs
        # `phaser_lmax_resolution` because it needs the same pair to size the
        # dense calc box, so re-deriving them here would only rediscover what it
        # computed a line earlier. `d_min` is therefore the *coarsened* limit --
        # the resolution this bandwidth can represent -- not the data's own.
        self.L = L
        self.d_min = d_min
        self.d_max = d_max
        self.delta_vrms_A = delta_vrms_A
        self.n_wilson_shells = n_wilson_shells
        self.grid_sampling_deg = grid_sampling_deg

        # 1. Resolution window.
        #
        # With `asu_idx` the caller has already masked, and `F_obs` / `sig_F_obs`
        # / `centric_obs` are ONE ROW PER UNIQUE REFLECTION while `s_obs` carries
        # the full symmetry-unrolled geometry. Everything from here to the
        # expansion is a per-reflection function of (F, sigma_F, |s|, centric),
        # all four of which are symmetry-invariant, so the chain runs on the
        # unique set and is broadcast at the end. That is exact -- not an
        # approximation -- and it is the difference between doing the
        # French-Wilson posterior once and doing it n_ops times.
        #
        # Without `asu_idx` every array is per-row of `s_obs` and the window is
        # applied here, which is the path direct callers and the synthetic tests
        # take.
        if asu_idx is None:
            extras = (F_obs, centric_obs)
            if sig_F_obs is not None:
                extras = extras + (sig_F_obs,)
            s_obs, extras, smag_src = _resolution_mask(s_obs, extras, d_min, d_max)
            F_obs, centric_obs = extras[0], extras[1]
            if sig_F_obs is not None:
                sig_F_obs = extras[2]
        else:
            if s_mag_asu is None:
                raise ValueError("asu_idx requires s_mag_asu (|s| per unique row)")
            if int(asu_idx.shape[0]) != int(s_obs.shape[0]):
                raise ValueError(
                    f"asu_idx has {int(asu_idx.shape[0])} entries for "
                    f"{int(s_obs.shape[0])} unrolled reflections"
                )
            smag_src = s_mag_asu

        if F_obs.shape[0] < n_wilson_shells * 5:
            raise ValueError(
                f"Too few obs reflections ({F_obs.shape[0]}) for "
                f"{n_wilson_shells} Wilson shells in [{d_min}, {d_max}] Å."
            )

        # 2. Bessel scaling — Phaser's lmax · d_min (DataMR.cc:1107).
        #    `bessel_h_scale = 2 pi R_patt` is the Patterson integration radius
        #    (the chi_Omega sphere): the Bessel argument is
        #    `h = bessel_h_scale |s|`, so the radial basis represents the
        #    Patterson out to `R_patt = bessel_h_scale / (2 pi)`. With the
        #    bandwidth-coupled `d_min` the caller passes, that comes to
        #    `R_patt ~ sphereOuter = 2 x mean radius`.
        if d_min is None:
            raise ValueError("d_min is required to set the Bessel scaling")
        lmax = L - 1
        lmax_even = lmax if lmax % 2 == 0 else lmax - 1
        self.bessel_h_scale = float(lmax_even) * float(d_min)

        # 3. ONE shell assignment, shared by everything below.
        #
        # The French-Wilson posterior, the LERF1 build and the variance reweight
        # all normalise per resolution shell, and each used to derive its own
        # equal-count edges from the same |s| -- one in numpy, one in torch, with
        # different quantile-rank rounding. That put a handful of boundary
        # reflections in different shells depending on which consumer asked,
        # which is a difference of ~2e-4 relative on their normalisation for no
        # reason. Assign once, pass it down.
        from ..sh import assign_shells, equal_count_shell_edges

        shell_edges, _ = equal_count_shell_edges(smag_src, n_wilson_shells)
        obs_shell_idx = assign_shells(smag_src, shell_edges)

        # 3b. Wilson normalisation. With sigmas, through the French-Wilson
        #    posterior, which handles the axial reflections; without them, plain
        #    per-shell Wilson.
        if sig_F_obs is not None:
            fw = french_wilson_preprocess(
                F_obs, sig_F_obs, smag_src, centric_obs,
                n_wilson_shells=n_wilson_shells, shell_idx=obs_shell_idx,
            )
            eEobs, dfac = fw["eEobs"], fw["DFAC"]
        else:
            eEobs, _ = wilson_normalise(F_obs, smag_src, n_wilson_shells)
            dfac = torch.ones_like(eEobs)

        # 4. LERF1 obs intensity, and the per-shell variance reweight.
        intensity_obs = build_lerf1_intensity(
            eEobs, centric_obs, dfac=dfac, use_centric_weight=True,
        )
        intensity_obs = apply_shell_variance_weights(
            intensity_obs, smag_src, n_var_shells=n_wilson_shells,
            shell_idx=obs_shell_idx,
        )
        if asu_idx is not None:
            # One value per unique reflection -> one per unrolled reflection.
            intensity_obs = intensity_obs[asu_idx]

        # 5. ZSYMM m-filter on the obs SH coefficients. The calc side is never
        #    filtered -- see score_model.
        zsymm = detect_zsymm(sym_mats)

        # 6. Bessel-SH expand the obs side.
        # `s_obs` stays at the caller's (wider) dtype so the expansion's
        # clustering keys keep their resolution; the intensity does not need to,
        # and the expansion casts it to the working precision anyway.
        self._c_obs = bessel_sh_expand(
            s_obs, intensity_obs,
            L=L, bessel_h_scale=self.bessel_h_scale,
            zsymm=zsymm, enforce_friedel=True,
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
        """Score one model's transform against the prepared observations.

        ``s_calc`` / ``F_calc`` must already lie inside ``[d_max, d_min]`` --
        this method does not re-mask them. The window is the caller's because
        the caller had to know it to build the calc set in the first place.
        """
        # The caller owns the calc-side resolution window: `dense_calc_via_box`
        # samples the box over `[d_max, d_min]` already, and re-masking here was
        # measured to drop 0 of 339040 reflections on 3K7M and 0 of 271630 on
        # 1DAW. So take `s_calc` as given and only derive |s| from it.
        smag_calc = s_calc.norm(dim=-1)
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
            s_calc, intensity_calc,
            L=self.L, bessel_h_scale=self.bessel_h_scale,
            zsymm=1, enforce_friedel=True,
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
