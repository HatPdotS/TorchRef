#!/usr/bin/env python3 -u

"""
Collection-based difference refinement: joint scaling with bulk solvent.

Uses ModelCollection / DatasetCollection / CollectionScaler so ONE set of scale
parameters (overall scale, anisotropy, bulk solvent k_sol/B_sol) is shared across the
dark and light datasets.

Writes refined dark and light models (PDB/CIF), a JSON summary, and a difference MTZ:

Observed        ``Fo_dark``, ``SIGFo_dark``, ``Fo_light``, ``SIGFo_light``
Differences     ``DF`` = F_light - F_dark, ``SIGDF`` (propagated), ``WDF`` (sigma-weighted)
Calculated      ``Fc_dark``, ``Fc_light`` (mixed model), ``DFc`` (scalar), ``DFc_complex``
DED map coeffs  ``2mDFop-DFc``, ``mDFop-DFc`` (phase-aware, figure-of-merit weighted)
Phases          ``PHIC_dark``, ``PHIC_mixed``, ``PHIC_diff``, ``PHIC_light``
Extrapolations  ``Fextp`` (phase-aware), ``Fextc`` (amplitude-only, no phases),
                ``Fextb`` (empirical-Bayes shrinkage toward Fo_dark -- see
                :func:`compute_bayes_extrapolated_amplitudes`), each with its sigma
                and ``2F-Fc`` / ``F-Fc`` map coefficients

Examples
--------
::

    torchref.difference-refine \\
        -dm dark.pdb -lm light.pdb \\
        -dsf dark.mtz -lsf light.mtz \\
        --fraction 0.37 -o output/
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

from torchref.cli._common import (
    add_dual_model_args,
    add_dmin_arg,
    add_general_args,
    add_metadata_args,
    add_outdir_arg,
    add_output_format_args,
    add_weights_arg,
    build_dual_column_names,
    configure_unbuffered_output,
    load_model,
    load_reflection_data,
    parse_weights,
    register_timing,
    parse_device_str,
    validate_cif_files,
    validate_files,
)
from torchref.utils.serialization import convert_to_serializable

configure_unbuffered_output()

# ---------------------------------------------------------------------------
# Default target weights
# ---------------------------------------------------------------------------

DEFAULT_TARGET_WEIGHTS = {
    "xray/difference": 1.0,
    "xray/rice": 0.0,
    # "geometry/bond": 1.0, # geometry restraint should never require tuning, so leave at 1.0
    # "geometry/angle": 1.0,
    # "geometry/torsion": 1.0,
    # "geometry/planarity": 1.0,
    # "geometry/chiral": 1.0,
    # "geometry/nonbonded": 1.0,
    # "geometry/ramachandran": 1.0,
    # "adp/simu": 1.0,
    # "adp/locality": 1.0,
    # "adp/sigd": 1.0,
    "similarity": 1.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_model_collection(pdb_dark, pdb_light, fractions, cif, d_min,
                           device, verbose, hydrogenate=False):
    """Load models and create a ModelCollection.

    Parameters
    ----------
    hydrogenate : bool
        If True, add explicit hydrogens to both models.  H atoms
        participate in geometry/VDW restraints (preventing clashes)
        but are excluded from structure factor calculations.
    """
    from torchref.model.model_collection import ModelCollection

    model_dark = load_model(
        pdb_dark, max_res=d_min, device=device, verbose=verbose, cif=cif,
    )
    model_light = load_model(
        pdb_light, max_res=d_min, device=device, verbose=verbose, cif=cif,
    )

    if hydrogenate:
        if verbose > 0:
            print("Adding hydrogens for VDW clash prevention...")
            sys.stdout.flush()
        model_dark = model_dark.hydrogenate(verbose=max(0, verbose - 1))
        model_light = model_light.hydrogenate(verbose=max(0, verbose - 1))
        model_dark.exclude_H_from_sf = True
        model_light.exclude_H_from_sf = True

    mc = ModelCollection([model_dark, model_light], dark_key="dark")
    mc.add_dark()
    mc.add_timepoint("light", fractions=fractions)
    return mc


def setup_dataset_collection(sf_dark, sf_light, d_min, device,
                             column_names_dark=None, column_names_light=None):
    """Load reflection data and create a DatasetCollection."""
    from torchref import DatasetCollection

    data_dark = load_reflection_data(
        sf_dark, device=device, column_names=column_names_dark,
    )
    data_light = load_reflection_data(
        sf_light, device=device, column_names=column_names_light,
    )
    if d_min is not None:
        data_dark.cut_res(highres=d_min)
        data_light.cut_res(highres=d_min)

    dc = DatasetCollection(device=device)
    dc.add_dataset("dark", data_dark)
    dc.add_dataset("light", data_light)
    dc.scale()
    return dc


def setup_scaler(dataset_collection, model_collection, device, verbose=1):
    """Create a CollectionScaler with per-component solvent models."""
    from torchref.scaling import CollectionScaler

    scaler = CollectionScaler(
        dataset_collection=dataset_collection,
        model_collection=model_collection,
        device=device,
        verbose=verbose,
    )
    scaler.initialize()
    scaler.screen_solvent_params_joint()
    scaler.refine_lbfgs_joint()
    return scaler


def compute_rfactors(model, data, scaler):
    """Compute R-work/R-free using forward_mixed for proper solvent.

    Routes through ``rfactor_work_free`` — the shared source of truth used by the
    refinement targets — so the validity mask is applied and the validation set is
    excluded from both work and free, matching every other reported R-factor.
    """
    from torchref.base.metrics.rfactor import rfactor_work_free

    with torch.no_grad():
        hkl = data.hkl
        fcalc = model(hkl)
        fcalc_scaled = scaler.forward_mixed(fcalc, model.fractions)
        return rfactor_work_free(data, torch.abs(fcalc_scaled))


def setup_loss_state(dataset_collection, model_collection, scaler,
                     target_weights, device, similarity_alpha=2.0):
    """Build LossState with collection-aware targets.

    Geometry and ADP restraints are applied only to the light base model
    (the dark model is a frozen reference).
    """
    from torchref.refinement import LossState
    from torchref.experimental.kinetic.targets import (
        CollectionDifferenceTarget,
        CollectionRiceTarget,
    )
    from torchref.refinement.targets import TotalADPTarget, TotalGeometryTarget
    from torchref.refinement.targets.similarity import CoordinateSimilarityTarget

    state = LossState(device=device)

    model_light = model_collection.base_models[1]
    model_dark = model_collection.base_models[0]

    diff_target = CollectionDifferenceTarget(
        dataset_collection, model_collection, scaler=scaler,
    )
    rice_target = CollectionRiceTarget(
        dataset_collection, model_collection, scaler=scaler,
    )
    geom_target = TotalGeometryTarget(model_light)
    adp_target = TotalADPTarget(model_light)

    similarity_target = CoordinateSimilarityTarget(
        model_dark=model_dark, model_light=model_light, alpha=similarity_alpha,
    )

    state.register_target("xray/difference", diff_target)
    state.register_target("xray/rice", rice_target)
    state.register_target("geometry", geom_target)
    state.register_target("adp", adp_target)
    state.register_target("similarity", similarity_target)

    state.set_weights(target_weights)

    return state


def compute_bayes_extrapolated_amplitudes(
    Fobs_dark, Fobs_light, sig_dark, sig_light, phi_dark, phi_mixed, f,
    *, tau_sq_floor=1e-4,
):
    """Empirical Bayes shrinkage estimator for extrapolated SF amplitudes.

    Estimates per-reflection shrinkage weights from the propagated variance of the
    extrapolation, then shrinks the phase-aware extrapolated amplitude toward
    Fo_dark, regularising noisy high-resolution and weakly-measured reflections::

        F_ext     = |F_dark*e^(iφ_d) + ΔF/f|         (phase-aware amplitude)
        σ_ext²    = (σ_light² + σ_dark²) / f²
        τ²        = max(<(F_ext - Fo_dark)²> - <σ_ext²>, floor)
        w(h)      = τ² / (τ² + σ_ext²(h))
        F_extb    = w(h)·F_ext + (1-w(h))·Fo_dark    (amplitude shrinkage)

    Parameters
    ----------
    Fobs_dark, Fobs_light : Tensor (N,)
        Observed amplitudes.
    sig_dark, sig_light : Tensor (N,)
        Measurement uncertainties.
    phi_dark, phi_mixed : Tensor (N,)
        Calculated phases (radians) for the dark and mixed models.
    f : float or Tensor
        Excited-state population fraction.
    tau_sq_floor : float
        Floor on the estimated signal variance τ².

    Returns
    -------
    tuple
        ``(F_ext_bayes, var_ext_bayes, w_shrinkage, tau_sq)`` -- extrapolated
        amplitudes **before** shrinkage, posterior variance and shrinkage weight per
        reflection, and the global τ² as a float.
    """
    F_dark_phased = Fobs_dark * torch.exp(1j * phi_dark)
    F_light_phased = Fobs_light * torch.exp(1j * phi_mixed)
    delta_F = F_light_phased - F_dark_phased

    # Propagated variance
    sig_sq_dF = sig_light**2 + sig_dark**2
    sig_sq_ext = sig_sq_dF / f**2

    # Phase-aware extrapolated amplitude
    F_ext_complex = F_dark_phased + delta_F / f
    F_ext = torch.abs(F_ext_complex)

    # Estimate signal variance τ²
    residuals_sq = (F_ext - Fobs_dark) ** 2
    tau_sq = max((residuals_sq.mean() - sig_sq_ext.mean()).item(), tau_sq_floor)

    # Per-reflection shrinkage weight (in [0, 1])
    w = tau_sq / (tau_sq + sig_sq_ext)

    # Posterior variance
    var_ext_bayes = (tau_sq * sig_sq_ext) / (tau_sq + sig_sq_ext)

    return F_ext, var_ext_bayes, w, tau_sq


def write_results_mtz(dc, mc, scaler, filename):
    """Write difference / extrapolated map coefficients to an MTZ file.

    Parameters
    ----------
    dc : DatasetCollection
    mc : ModelCollection
    scaler : CollectionScaler
    filename : str
        Output MTZ path.
    """
    import numpy as np
    import reciprocalspaceship as rs
    from torchref import ReflectionData, Scaler

    data_dark = dc[mc.dark_key]
    data_light = dc["light"]
    dark_model = mc.dark_model
    mixed_model = mc["light"]
    model_light = mc.base_models[1]

    hkl_all = data_dark.hkl
    Fobs_dark_full, sig_dark_full = data_dark.get_corrected_data()
    Fobs_light_full, sig_light_full = data_light.get_corrected_data()

    mask = data_dark.masks().to(torch.bool) & data_light.masks().to(torch.bool)
    hkl = hkl_all[mask]
    Fobs_dark_vals = Fobs_dark_full[mask]
    Fobs_light_vals = Fobs_light_full[mask]
    sig_dark_vals = sig_dark_full[mask]
    sig_light_vals = sig_light_full[mask]
    rfree_flags_masked = (
        data_light.rfree_flags[mask] if data_light.rfree_flags is not None else None
    )

    fractions = mixed_model.fractions.detach()
    w_dark = fractions[0]
    w_light = fractions[1]

    # Compute Fcalc on full HKL then mask (scalers fitted on full datasets)
    with torch.no_grad():
        fcalc_dark = scaler.forward_mixed(dark_model(hkl_all), dark_model.fractions)[mask]
        fcalc_mixed = scaler.forward_mixed(mixed_model(hkl_all), mixed_model.fractions)[mask]
    fcalc_diff = fcalc_mixed - fcalc_dark

    phi_dark = torch.angle(fcalc_dark)
    phi_mixed = torch.angle(fcalc_mixed)

    F_obs_dark_phased = Fobs_dark_vals * torch.exp(1j * phi_dark)
    F_obs_light_phased = Fobs_light_vals * torch.exp(1j * phi_mixed)

    # --- Phase-aware extrapolation ---
    F_light_extra = (F_obs_light_phased - w_dark * F_obs_dark_phased) / w_light
    amp_light_extra = torch.abs(F_light_extra)
    sig_light_extra = torch.sqrt(sig_light_vals**2 + w_dark**2 * sig_dark_vals**2) / w_light

    data_light_extra = ReflectionData.from_tensors(
        hkl=hkl, F=amp_light_extra, F_sigma=sig_light_extra,
        cell=data_light.cell, spacegroup=data_light.spacegroup,
        rfree_flags=rfree_flags_masked, device=str(hkl.device), verbose=0,
    )
    scaler_extra = Scaler(model_light, data_light_extra, device=hkl.device, verbose=-1)
    scaler_extra.initialize().refine_lbfgs()
    F_calc_extra = scaler_extra(model_light(hkl))

    amp_extra = torch.abs(F_light_extra)
    amp_light_calc = torch.abs(F_calc_extra)
    phi_light_calc = torch.angle(F_calc_extra)
    amp_2fofc_light = 2 * amp_extra - amp_light_calc
    amp_fextfc = amp_extra - amp_light_calc

    # --- Classic (amplitude-only) extrapolation ---
    amp_extra_classic = (Fobs_light_vals - w_dark * Fobs_dark_vals) / w_light
    sig_extra_classic = sig_light_extra

    data_extra_classic = ReflectionData.from_tensors(
        hkl=hkl, F=amp_extra_classic, F_sigma=sig_extra_classic,
        cell=data_light.cell, spacegroup=data_light.spacegroup,
        rfree_flags=rfree_flags_masked, device=str(hkl.device), verbose=0,
    )
    scaler_classic = Scaler(model_light, data_extra_classic, device=hkl.device, verbose=-1)
    scaler_classic.initialize().refine_lbfgs()
    F_calc_classic = scaler_classic(model_light(hkl))

    amp_light_calc_classic = torch.abs(F_calc_classic)
    amp_2fofc_classic = 2 * amp_extra_classic - amp_light_calc_classic
    amp_fofc_classic = amp_extra_classic - amp_light_calc_classic

    # --- Empirical Bayes extrapolation (amplitude-only shrinkage) ---
    F_ext_bayes, var_ext_bayes, w_shrinkage, tau_sq = (
        compute_bayes_extrapolated_amplitudes(
            Fobs_dark_vals, Fobs_light_vals,
            sig_dark_vals, sig_light_vals,
            phi_dark, phi_mixed, w_light,
        )
    )
    sig_ext_bayes = torch.sqrt(var_ext_bayes)

    # Shrink |F_ext| towards |Fo_dark| — scalar operation, no phase interference
    F_ext_amp_only = torch.abs(F_light_extra)
    F_ext_bayes_amp = w_shrinkage * F_ext_amp_only + (1 - w_shrinkage) * Fobs_dark_vals

    data_extra_bayes = ReflectionData.from_tensors(
        hkl=hkl, F=F_ext_bayes_amp, F_sigma=sig_ext_bayes,
        cell=data_light.cell, spacegroup=data_light.spacegroup,
        rfree_flags=rfree_flags_masked, device=str(hkl.device), verbose=0,
    )
    scaler_bayes = Scaler(model_light, data_extra_bayes, device=hkl.device, verbose=-1)
    scaler_bayes.initialize().refine_lbfgs()
    F_calc_bayes = scaler_bayes(model_light(hkl))

    amp_calc_bayes = torch.abs(F_calc_bayes)
    amp_2fofc_bayes = 2 * F_ext_bayes_amp - amp_calc_bayes
    amp_fofc_bayes = F_ext_bayes_amp - amp_calc_bayes

    from torchref.base.metrics.rfactor import rfactor_work_free

    def _extrapolation_rfactors(data, fcalc_scaled):
        return rfactor_work_free(data, torch.abs(fcalc_scaled))

    print("Phase-aware extrapolation rfactors:",
          _extrapolation_rfactors(data_light_extra, F_calc_extra))
    print("Classic extrapolation rfactors:",
          _extrapolation_rfactors(data_extra_classic, F_calc_classic))
    print("Bayes extrapolation rfactors:",
          _extrapolation_rfactors(data_extra_bayes, F_calc_bayes))
    print(f"  Bayes: tau^2 = {tau_sq:.4f}, mean w(h) = {w_shrinkage.mean().item():.3f}")

    # --- Build MTZ ---
    hkl_np = hkl.cpu().numpy()
    Fobs_dark = Fobs_dark_vals.cpu().numpy()
    Fobs_light = Fobs_light_vals.cpu().numpy()
    sig_dark = sig_dark_vals.cpu().numpy()
    sig_light = sig_light_vals.cpu().numpy()

    Fcalc_dark = torch.abs(fcalc_dark).detach().cpu().numpy()
    Fcalc_light = torch.abs(fcalc_mixed).detach().cpu().numpy()
    phases_dark = torch.angle(fcalc_dark).detach().rad2deg().cpu().numpy()
    phases_mixed = torch.angle(fcalc_mixed).detach().rad2deg().cpu().numpy()

    Fcalc_diff_amp = torch.abs(fcalc_diff).detach().cpu().numpy()
    Fcalc_diff_scalar = Fcalc_light - Fcalc_dark
    phases_diff = torch.angle(fcalc_diff).detach().rad2deg().cpu().numpy()

    Fobs_diff_phased = torch.abs(
        F_obs_light_phased - F_obs_dark_phased
    ).detach().cpu().numpy()

    diff_Fobs = Fobs_light - Fobs_dark
    sig_diff = (sig_dark**2 + sig_light**2) ** 0.5
    weights = 1 / sig_diff**2
    weights = weights / weights.mean()
    weighted_diff_Fobs = diff_Fobs * weights

    amp_2DFoDFc = (2 * Fobs_diff_phased - Fcalc_diff_amp) * weights
    amp_DFoDFc = (Fobs_diff_phased - Fcalc_diff_amp) * weights

    df = rs.DataSet(
        {
            "H": hkl_np[:, 0], "K": hkl_np[:, 1], "L": hkl_np[:, 2],
            # Observed
            "Fo_dark": Fobs_dark, "SIGFo_dark": sig_dark,
            "Fo_light": Fobs_light, "SIGFo_light": sig_light,
            # Differences
            "DF": diff_Fobs, "SIGDF": sig_diff, "WDF": weighted_diff_Fobs,
            # Calculated
            "Fc_dark": Fcalc_dark, "Fc_light": Fcalc_light,
            "DFc": Fcalc_diff_scalar, "DFc_complex": Fcalc_diff_amp,
            # DED map coefficients (phase-aware)
            "2mDFop-DFc": amp_2DFoDFc, "mDFop-DFc": amp_DFoDFc,
            # Phases
            "PHIC_dark": phases_dark, "PHIC_mixed": phases_mixed,
            "PHIC_diff": phases_diff,
            "PHIC_light": phi_light_calc.detach().rad2deg().cpu().numpy(),
            # Phase-aware extrapolation
            "Fextp": amp_extra.detach().cpu().numpy(),
            "2Fextp-Fc": amp_2fofc_light.detach().cpu().numpy(),
            "Fextp-Fc": amp_fextfc.detach().cpu().numpy(),
            # Classic extrapolation
            "Fextc": amp_extra_classic.detach().cpu().numpy(),
            "SIGFextc": sig_extra_classic.detach().cpu().numpy(),
            "2Fextc-Fc": amp_2fofc_classic.detach().cpu().numpy(),
            "Fextc-Fc": amp_fofc_classic.detach().cpu().numpy(),
            # Bayes extrapolation (amplitude-only shrinkage)
            "Fextb": F_ext_bayes_amp.detach().cpu().numpy(),
            "SIGFextb": sig_ext_bayes.detach().cpu().numpy(),
            "2Fextb-Fc": amp_2fofc_bayes.detach().cpu().numpy(),
            "Fextb-Fc": amp_fofc_bayes.detach().cpu().numpy(),
            # R-free flags (1=work, 0=free)
            "FreeR_flag_dark": (
                data_dark.rfree_flags[mask].cpu().numpy().astype(int)
                if data_dark.rfree_flags is not None
                else np.ones(len(hkl_np), dtype=int)
            ),
            "FreeR_flag_light": (
                data_light.rfree_flags[mask].cpu().numpy().astype(int)
                if data_light.rfree_flags is not None
                else np.ones(len(hkl_np), dtype=int)
            ),
        },
        cell=data_dark.cell.data.cpu().tolist(),
        spacegroup=data_dark.spacegroup.hm,
    )

    df[["H", "K", "L"]] = df[["H", "K", "L"]].astype("H")
    f_cols = [
        "Fo_dark", "Fo_light", "DF", "WDF",
        "Fc_dark", "Fc_light", "DFc", "DFc_complex",
        "2mDFop-DFc", "mDFop-DFc",
        "Fextp", "2Fextp-Fc", "Fextp-Fc",
        "Fextc", "2Fextc-Fc", "Fextc-Fc",
        "Fextb", "2Fextb-Fc", "Fextb-Fc",
    ]
    df[f_cols] = df[f_cols].astype("F")
    sig_cols = ["SIGFo_dark", "SIGFo_light", "SIGDF", "SIGFextc", "SIGFextb"]
    df[sig_cols] = df[sig_cols].astype("Q")
    phase_cols = ["PHIC_dark", "PHIC_mixed", "PHIC_diff", "PHIC_light"]
    df[phase_cols] = df[phase_cols].astype("P")
    df["FreeR_flag_dark"] = df["FreeR_flag_dark"].astype("I")
    df["FreeR_flag_light"] = df["FreeR_flag_light"].astype("I")
    df.set_index(["H", "K", "L"], inplace=True)
    df.write_mtz(filename)
    print(f"  Results MTZ written to {filename}")
    print(f"  w_dark={w_dark.item():.3f}, w_light={w_light.item():.3f}")


def optimize_lbfgs(state, parameters, max_iter, nsteps, n_clean, verbose):
    """Run a block of LBFGS optimisation steps via :meth:`LossState.step`.

    ``state.step`` handles the closure, NaN validation, and automatically
    disables ``requires_grad`` on any loss-relevant leaves outside
    ``parameters`` — in particular the dark model's leaves, which appear in
    the difference target's autograd graph but are intentionally not in the
    optimizer's intent. The dark model effectively becomes a frozen
    reference at the autograd level for the duration of each step.
    """
    parameters = list(parameters)

    def _make_optimizer():
        return torch.optim.LBFGS(
            parameters, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

    optimizer = _make_optimizer()
    for i in range(nsteps):
        if i > 0 and i % n_clean == 0:
            # Periodic LBFGS curvature-history reset.
            optimizer = _make_optimizer()
        state.step(
            optimizer, context="collection_difference_refine.optimize_lbfgs"
        )
        if verbose > 0:
            with torch.no_grad():
                loss = state.aggregate()
                print(f"    LBFGS step {i + 1}/{nsteps}, loss: {loss.item():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Entry point for ``torchref.difference-refine``; returns the exit code."""
    parser = argparse.ArgumentParser(
        prog="torchref.difference-refine",
        description="Collection-based difference refinement with joint "
                    "scaling and bulk solvent correction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  torchref.difference-refine \\
      -dm dark.pdb -lm light.pdb \\
      -dsf dark.mtz -lsf light.mtz \\
      --fraction 0.37 -o output/
        """,
    )

    add_dual_model_args(parser)

    output = parser.add_argument_group("Output")
    add_outdir_arg(output, help="Output directory for refined structures and maps")
    add_output_format_args(output)
    add_metadata_args(output)

    refine = parser.add_argument_group("Refinement")
    refine.add_argument(
        "--weight-schedule", type=str, default="5,3,2",
        help="Comma-separated difference-target weights applied in "
             "sequence each macro-cycle (default: '5,3,2')",
    )
    refine.add_argument(
        "--n-cycles", type=int, default=3,
        help="Number of macro-cycles (default: 3)",
    )
    refine.add_argument(
        "--n-steps", type=int, default=2,
        help="LBFGS optimisation rounds per weight step (default: 2)",
    )
    refine.add_argument(
        "--max-iter", type=int, default=100,
        help="Max line-search iterations per LBFGS step (default: 100)",
    )
    refine.add_argument(
        "--n-clean", type=int, default=2,
        help="Reset LBFGS history every N steps (default: 2)",
    )
    add_weights_arg(refine, default_weights=DEFAULT_TARGET_WEIGHTS)
    refine.add_argument(
        "--refine-fractions", action="store_true", default=False,
        help="Refine population fractions during optimisation (default: frozen)",
    )
    refine.add_argument(
        "--similarity-weight", type=float, default=1.0,
        help="Weight for dark/light coordinate similarity restraint "
             "(0 to disable, default: 1.0)",
    )
    refine.add_argument(
        "--similarity-alpha", type=float, default=2.0,
        help="Log prior odds for spike-and-slab similarity restraint. "
             "Higher = stronger denoising (default: 2.0)",
    )

    res = parser.add_argument_group("Resolution")
    add_dmin_arg(res)

    add_general_args(parser)

    args = parser.parse_args()
    register_timing()

    # --- Parse fractions ---
    if not (0.0 < args.fraction < 1.0):
        print(f"Error: --fraction must be between 0 and 1 (got {args.fraction})",
              file=sys.stderr)
        return 1
    fractions = [1.0 - args.fraction, args.fraction]

    # --- Parse weight schedule ---
    try:
        weight_schedule = [float(x) for x in args.weight_schedule.split(",")]
        if not weight_schedule:
            raise ValueError
    except ValueError:
        print("Error: --weight-schedule must be comma-separated floats",
              file=sys.stderr)
        return 1

    # --- Parse and merge target weights ---
    target_weights = dict(DEFAULT_TARGET_WEIGHTS)
    target_weights["xray/difference"] = weight_schedule[0]
    target_weights["similarity"] = args.similarity_weight
    target_weights, err = parse_weights(args.weights, defaults=target_weights)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    # --- Validate input files ---
    rc = validate_files([
        (args.dark_model, "Dark model"),
        (args.light_model, "Light model"),
        (args.dark_structure_factor, "Dark structure factor"),
        (args.light_structure_factor, "Light structure factor"),
    ])
    if rc:
        return rc
    rc = validate_cif_files(args.cif)
    if rc:
        return rc

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = parse_device_str(args.device)
    d_min = args.dmin if args.dmin is not None else 1.0

    # --- Header ---
    if args.verbose > 0:
        print("=" * 72)
        print("TorchRef Collection Difference Refinement")
        print("=" * 72)
        print(f"Dark model:        {args.dark_model}")
        print(f"Light model:       {args.light_model}")
        print(f"Dark data:         {args.dark_structure_factor}")
        print(f"Light data:        {args.light_structure_factor}")
        frac_mode = "refinable" if args.refine_fractions else "frozen"
        print(f"Fractions:         dark={fractions[0]}, light={fractions[1]} ({frac_mode})")
        print(f"Output:            {outdir}")
        print(f"Device:            {device}")
        if args.dmin:
            print(f"Resolution cutoff: {args.dmin:.2f} A")
        if args.cif:
            print(f"CIF restraints:    {', '.join(args.cif)}")
        print(f"Weight schedule:   {weight_schedule} x {args.n_cycles} cycles")
        print(f"LBFGS steps/weight: {args.n_steps}  (max_iter={args.max_iter})")
        print()
        print("Target weights:")
        for wk, wv in sorted(target_weights.items()):
            print(f"  {wk}: {wv}")
        print("=" * 72)
        print()
        sys.stdout.flush()

    # --- Setup models ---
    if args.verbose > 0:
        print("Setting up models...")
        sys.stdout.flush()

    mc = setup_model_collection(
        args.dark_model, args.light_model, fractions,
        args.cif, d_min, device, args.verbose,
    )
    dark = mc.dark_model            # _SharedMixedModel (fractions=[1, 0])
    mixed = mc["light"]             # _SharedMixedModel (fractions=[1-f, f] from --fraction)
    model_dark = mc.base_models[0]  # ModelFT (for output)
    model_light = mc.base_models[1] # ModelFT (for output)

    if args.refine_fractions:
        mixed.unfreeze_fractions()
    else:
        mixed.freeze_fractions()

    # --- Setup data ---
    if args.verbose > 0:
        print("Loading reflection data...")
        sys.stdout.flush()

    col_dark, col_light = build_dual_column_names(args)

    dc = setup_dataset_collection(
        args.dark_structure_factor, args.light_structure_factor,
        args.dmin, device,
        column_names_dark=col_dark, column_names_light=col_light,
    )
    data_dark = dc["dark"]
    data_light = dc["light"]

    if args.verbose > 0:
        # Compare R-free flags between datasets
        if data_dark.rfree_flags is not None and data_light.rfree_flags is not None:
            rfree_d = data_dark.rfree_flags.bool()
            rfree_l = data_light.rfree_flags.bool()
            n_agree = (rfree_d == rfree_l).sum().item()
            n_total = len(rfree_d)
            n_free_d = (~rfree_d).sum().item()
            n_free_l = (~rfree_l).sum().item()
            print(f"  R-free flags: dark={n_free_d} free, light={n_free_l} free, "
                  f"agreement={n_agree}/{n_total} ({100*n_agree/n_total:.1f}%)")
        sys.stdout.flush()

    # --- Setup scaler ---
    if args.verbose > 0:
        print("Setting up joint scaler...")
        sys.stdout.flush()

    scaler = setup_scaler(dc, mc, device, args.verbose)

    if args.verbose > 0:
        r_work_d, r_free_d = compute_rfactors(dark, data_dark, scaler)
        r_work_l, r_free_l = compute_rfactors(mixed, data_light, scaler)
        r_work_dl, r_free_dl = compute_rfactors(dark, data_light, scaler)
        print(f"  Initial R-factor (dark  vs dark data):  R_work={r_work_d:.4f}  R_free={r_free_d:.4f}")
        print(f"  Initial R-factor (mixed vs light data): R_work={r_work_l:.4f}  R_free={r_free_l:.4f}")
        print(f"  Initial R-factor (dark  vs light data): R_work={r_work_dl:.4f}  R_free={r_free_dl:.4f}")
        print()
        sys.stdout.flush()

    # --- Setup targets ---
    state = setup_loss_state(dc, mc, scaler, target_weights, device,
                             similarity_alpha=args.similarity_alpha)

    if args.verbose > 0:
        print("Initial loss breakdown:")
        state.summary()
        print()
        sys.stdout.flush()

    total_rounds = args.n_cycles * len(weight_schedule)
    round_idx = 0

    if args.refine_fractions:
        params = list(itertools.chain(
            model_light.parameters(), [mixed.fraction_params]
        ))
    else:
        params = list(model_light.parameters())

    fraction_history = []
    if args.refine_fractions:
        fraction_history.append(mixed.fractions[1].detach().cpu().item())

    for cycle in range(args.n_cycles):
        for t_weight in weight_schedule:
            round_idx += 1
            if args.verbose > 0:
                frac_str = (
                    f", fractions={mixed.fractions.detach().cpu().tolist()}"
                    if args.refine_fractions else ""
                )
                print(
                    f"[{round_idx}/{total_rounds}] cycle {cycle + 1}/"
                    f"{args.n_cycles}, diff_weight={t_weight}{frac_str}"
                )
                sys.stdout.flush()

            state.set_weights({"xray/difference": t_weight})
            optimize_lbfgs(
                state, params,
                max_iter=args.max_iter,
                nsteps=args.n_steps,
                n_clean=args.n_clean,
                verbose=args.verbose,
            )

            # Update solvent masks and re-refine scaler jointly
            scaler.update_solvent()
            scaler.refine_lbfgs_joint(verbose=(args.verbose > 1))

            if args.verbose > 0:
                rw_d, rf_d = compute_rfactors(dark, data_dark, scaler)
                rw_l, rf_l = compute_rfactors(mixed, data_light, scaler)
                print(f"  R-factor (dark):  Rwork={rw_d:.4f}, Rfree={rf_d:.4f}")
                print(f"  R-factor (light): Rwork={rw_l:.4f}, Rfree={rf_l:.4f}")

            # Reset model caches after no_grad rfactor computation
            # to ensure gradients flow in the next optimisation round.
            model_light.reset_cache()
            model_dark.reset_cache()

            if args.refine_fractions:
                fraction_history.append(mixed.fractions[1].detach().cpu().item())

            if args.verbose > 1:
                state.summary()
                sys.stdout.flush()

    # --- Final statistics ---
    if args.verbose > 0:
        print()
        print("=" * 72)
        print("Refinement complete")
        print("=" * 72)
        r_work_d, r_free_d = compute_rfactors(dark, data_dark, scaler)
        r_work_l, r_free_l = compute_rfactors(mixed, data_light, scaler)
        r_work_dl, r_free_dl = compute_rfactors(dark, data_light, scaler)
        print(f"  Final R-factor (dark  vs dark data):  R_work={r_work_d:.4f}  R_free={r_free_d:.4f}")
        print(f"  Final R-factor (mixed vs light data): R_work={r_work_l:.4f}  R_free={r_free_l:.4f}")
        print(f"  Final R-factor (dark  vs light data): R_work={r_work_dl:.4f}  R_free={r_free_dl:.4f}")
        print(f"  Refined fractions:      {mixed.fractions.detach().cpu().numpy()}")
        print()
        sys.stdout.flush()

    # --- Save outputs ---
    prefix = f"fractions_{int(fractions[0]*100)}_{int(fractions[1]*100)}"

    dark_pdb_out = str(outdir / f"{prefix}_dark.pdb")
    light_pdb_out = str(outdir / f"{prefix}_light.pdb")
    diff_mtz_out = str(outdir / f"{prefix}_difference_data.mtz")
    summary_path = str(outdir / f"{prefix}_summary.json")

    # Strip hydrogens for output (H were only needed for VDW restraints).
    # strip_hydrogens() returns new models with consistent pdb + tensors.
    model_dark = model_dark.strip_hydrogens()
    model_light = model_light.strip_hydrogens()

    no_header = getattr(args, "no_header", False)
    output_format = getattr(args, "output_format", "both")
    dark_meta = light_meta = None

    if not no_header:
        from torchref import __version__
        from torchref.io.metadata import RefinementMetadata

        def _build_metadata(model, data, r_work, r_free):
            """Build RefinementMetadata for a model/data pair."""
            meta = RefinementMetadata(
                program_version=__version__,
                refinement_method="difference-refine",
                r_work=float(r_work), r_free=float(r_free),
            )
            # Resolution (from masks, respects cutoff)
            if data.resolution is not None:
                valid = data.masks().to(torch.bool)
                res_valid = data.resolution[valid]
                if len(res_valid) > 0:
                    meta.resolution_high = float(res_valid.min())
                    meta.resolution_low = float(res_valid.max())

            # Reflection counts (standard work/free subset accessors: validity
            # masked, validation carved out of both).
            with torch.no_grad():
                if data.rfree_flags is not None:
                    n_work = data.work.n
                    n_test = data.free.n
                    n_all = n_work + n_test
                    meta.n_reflections_work = n_work
                    meta.n_reflections_test = n_test
                    meta.n_reflections_all = n_all
                    meta.percent_free = 100.0 * n_test / n_all if n_all > 0 else None

            # B-factor statistics
            pdb = model.pdb
            bvals = pdb["tempfactor"]
            meta.b_mean_overall = float(bvals.mean())
            meta.b_min = float(bvals.min())
            meta.b_max = float(bvals.max())

            # Atom counts
            meta.n_atoms_total = len(pdb)
            meta.n_atoms_protein = int((pdb["ATOM"] == "ATOM").sum())
            meta.n_atoms_solvent = int((pdb["ATOM"] == "HETATM").sum())

            # Geometry deviations
            if model.initialized and model._restraints is not None:
                restraints = model.restraints
                with torch.no_grad():
                    if hasattr(restraints, "bond_deviations"):
                        bond_devs, _ = restraints.bond_deviations()
                        meta.rmsd_bond_lengths = float(torch.sqrt((bond_devs**2).mean()))
                    if hasattr(restraints, "angle_deviations"):
                        angle_devs, _ = restraints.angle_deviations()
                        meta.rmsd_bond_angles = float(torch.sqrt((angle_devs**2).mean()))

            # Solvent model from CollectionScaler
            if hasattr(scaler, "solvent") and scaler.solvent is not None:
                sm = scaler.solvent
                meta.solvent_model_ksol = float(sm.k_solvent().detach())
                meta.solvent_model_bsol = sm.b_solvent_equivalent(scaler._s_half_sq)

            # Cell and spacegroup
            if model.cell is not None:
                meta.cell = [float(x) for x in model.cell.data.tolist()]
            if model.spacegroup is not None:
                meta.spacegroup = model.spacegroup.hm

            # CLI overrides / defaults
            if getattr(args, "title", None):
                meta.title = args.title
            meta.authors = getattr(args, "authors", None) or ["AUTHOR NAME"]

            return meta

        dark_meta = _build_metadata(model_dark, data_dark, r_work_d, r_free_d)
        light_meta = _build_metadata(model_light, data_light, r_work_l, r_free_l)

    if output_format in ("pdb", "both"):
        model_dark.write_pdb(dark_pdb_out, metadata=dark_meta)
        model_light.write_pdb(light_pdb_out, metadata=light_meta)
    if output_format in ("cif", "both"):
        dark_cif_out = str(outdir / f"{prefix}_dark.cif")
        light_cif_out = str(outdir / f"{prefix}_light.cif")
        model_dark.write_cif(dark_cif_out, metadata=dark_meta)
        model_light.write_cif(light_cif_out, metadata=light_meta)

    # --- Write merged deposition CIF (if no altlocs) ---
    merged_cif_out = str(outdir / f"{prefix}_merged.cif")
    has_altloc_dark = (model_dark.pdb["altloc"].astype(str).str.strip() != "").any()
    has_altloc_light = (model_light.pdb["altloc"].astype(str).str.strip() != "").any()

    if not has_altloc_dark and not has_altloc_light:
        import pandas as pd
        from torchref import __version__
        from torchref.io.metadata import RefinementMetadata

        dark_df = model_dark.pdb.copy()
        dark_df["altloc"] = "A"
        dark_df["occupancy"] = fractions[0]

        light_df = model_light.pdb.copy()
        light_df["altloc"] = "B"
        light_df["occupancy"] = fractions[1]

        merged_df = pd.concat([dark_df, light_df], ignore_index=True)
        merged_df = merged_df.sort_values(
            ["chainid", "resseq", "icode", "name", "altloc"]
        ).reset_index(drop=True)
        merged_df["serial"] = range(1, len(merged_df) + 1)
        merged_df.attrs["cell"] = model_dark.cell.data.tolist()
        merged_df.attrs["spacegroup"] = (
            model_dark.spacegroup.hm if model_dark.spacegroup else "P 1"
        )

        merged_meta = RefinementMetadata(
            program_version=__version__,
            refinement_method="difference-refine",
            r_work=float(r_work_l),
            r_free=float(r_free_l),
            authors=getattr(args, "authors", None) or ["AUTHOR NAME"],
        )
        if data_light.resolution is not None:
            valid = data_light.masks().to(torch.bool)
            res_valid = data_light.resolution[valid]
            if len(res_valid) > 0:
                merged_meta.resolution_high = float(res_valid.min())
                merged_meta.resolution_low = float(res_valid.max())
        merged_meta.b_mean_overall = float(merged_df["tempfactor"].mean())
        merged_meta.n_atoms_total = len(merged_df)

        if hasattr(scaler, "solvent") and scaler.solvent is not None:
            sm = scaler.solvent
            merged_meta.solvent_model_ksol = float(sm.k_solvent().detach())
            merged_meta.solvent_model_bsol = sm.b_solvent_equivalent(scaler._s_half_sq)

        ensemble_note = (
            f"Mixed-state ensemble from TorchRef difference refinement. "
            f"Conformer A (occupancy {fractions[0]:.2f}): dark/ground state. "
            f"Conformer B (occupancy {fractions[1]:.2f}): light/excited state. "
            f"R-factors: mixed vs light data Rwork={r_work_l:.4f} Rfree={r_free_l:.4f}; "
            f"dark vs dark data Rwork={r_work_d:.4f} Rfree={r_free_d:.4f}."
        )
        merged_meta.title = ensemble_note

        from torchref.io import cif as cif_io
        cif_io.write_model(merged_df, merged_cif_out, metadata=merged_meta)

        if args.verbose > 0:
            print(f"  Merged deposition CIF written to {merged_cif_out}")
    else:
        if args.verbose > 0:
            print("  Skipping merged CIF: input models contain altlocs")
        merged_cif_out = None

    # --- Write per-dataset structure factor files (MTZ + CIF) ---
    dark_sf_mtz = str(outdir / f"{prefix}_dark-sf.mtz")
    light_sf_mtz = str(outdir / f"{prefix}_light-sf.mtz")
    dark_sf_cif = str(outdir / f"{prefix}_dark-sf.cif")
    light_sf_cif = str(outdir / f"{prefix}_light-sf.cif")

    with torch.no_grad():
        fcalc_dark_full = scaler.forward_mixed(
            dark.models[0](data_dark.hkl), dark.fractions
        )
        fcalc_light_full = scaler.forward_mixed(
            mixed(data_light.hkl), mixed.fractions
        )
    data_dark.write_mtz(dark_sf_mtz, fcalc=fcalc_dark_full)
    data_light.write_mtz(light_sf_mtz, fcalc=fcalc_light_full)

    def _mtz_to_cif(mtz_path, cif_path):
        """Convert MTZ to mmCIF structure factor file via gemmi."""
        import gemmi
        mtz = gemmi.read_mtz_file(mtz_path)
        m2c = gemmi.MtzToCif()
        cif_str = m2c.write_cif_to_string(mtz)
        with open(cif_path, "w") as f:
            f.write(cif_str)

    _mtz_to_cif(dark_sf_mtz, dark_sf_cif)
    _mtz_to_cif(light_sf_mtz, light_sf_cif)

    if args.verbose > 0:
        print(f"  Dark SF written to {dark_sf_mtz}, {dark_sf_cif}")
        print(f"  Light SF written to {light_sf_mtz}, {light_sf_cif}")

    write_results_mtz(dc, mc, scaler, diff_mtz_out)

    # --- JSON summary ---
    summary = {
        "input": {
            "dark_model": args.dark_model,
            "light_model": args.light_model,
            "dark_structure_factor": args.dark_structure_factor,
            "light_structure_factor": args.light_structure_factor,
            "fractions": fractions,
            "cif": args.cif,
            "dmin": args.dmin,
        },
        "parameters": {
            "weight_schedule": weight_schedule,
            "n_cycles": args.n_cycles,
            "n_steps": args.n_steps,
            "max_iter": args.max_iter,
            "weights": target_weights,
        },
        "results": {
            "r_factor_dark": dict(zip(
                ["r_work", "r_free"],
                compute_rfactors(dark, data_dark, scaler),
            )),
            "r_factor_light": dict(zip(
                ["r_work", "r_free"],
                compute_rfactors(mixed, data_light, scaler),
            )),
            "fractions": mixed.fractions.detach().cpu().tolist(),
        },
        "output_files": {
            "dark_pdb": dark_pdb_out,
            "light_pdb": light_pdb_out,
            "dark_sf_mtz": dark_sf_mtz,
            "dark_sf_cif": dark_sf_cif,
            "light_sf_mtz": light_sf_mtz,
            "light_sf_cif": light_sf_cif,
            "difference_mtz": diff_mtz_out,
            "merged_cif": merged_cif_out,
            "summary": summary_path,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(convert_to_serializable(summary), f, indent=2)

    if args.verbose > 0:
        print("Output files:")
        print(f"  - {dark_pdb_out}")
        print(f"  - {light_pdb_out}")
        if merged_cif_out:
            print(f"  - {merged_cif_out}")
        print(f"  - {dark_sf_mtz} / {dark_sf_cif}")
        print(f"  - {light_sf_mtz} / {light_sf_cif}")
        print(f"  - {diff_mtz_out}")
        print(f"  - {summary_path}")
        print()
        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
