#!/usr/bin/env python3 -u

"""
Command-line script for difference refinement of time-resolved
crystallographic data using torchref.

Refines a mixed model (dark + light state) against observed dark and
light reflection data using an amplitude-only difference target
together with geometry, ADP, and maximum-likelihood restraints.

The weight schedule controls how the difference-target weight is annealed
over the course of refinement.  By default the schedule ``5,3,2`` is
repeated for 3 macro-cycles with 2 LBFGS optimisation rounds per weight
step.

Rfree-flags from both datasets are respected. In the difference target
the rfree sets are combined.

Examples
--------
::

    # Basic difference refinement
    torchref.difference-refine \\
        --dark-pdb dark.pdb --light-pdb light.pdb \\
        --dark-mtz dark.mtz --light-mtz light.mtz \\
        --fraction 0.37 -o output/

    # Custom weight schedule
    torchref.difference-refine \\
        --dark-pdb dark.pdb --light-pdb light.pdb \\
        --dark-mtz dark.mtz --light-mtz light.mtz \\
        --fraction 0.37 --weight-schedule 10,5,3,1 -o output/
"""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import torch

from torchref.utils.serialization import convert_to_serializable

# Force unbuffered output for batch systems like SLURM
(
    sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stdout, "reconfigure")
    else None
)
(
    sys.stderr.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure")
    else None
)
os.environ["PYTHONUNBUFFERED"] = "1"


# ---------------------------------------------------------------------------
# Default target weights
# ---------------------------------------------------------------------------
# Keys use the full path as registered in LossState (prefix/target/component).
# The difference target weight is controlled separately via --weight-schedule.

DEFAULT_TARGET_WEIGHTS = {
    # ML target
    "light/maximum_likelihood_xray": 1.0,
    # Geometry (7 components)
    "light/model_target/geometry/bond": 1.0,
    "light/model_target/geometry/angle": 0.3,
    "light/model_target/geometry/torsion": 0.3,
    "light/model_target/geometry/planarity": 1.0,
    "light/model_target/geometry/chiral": 2.0,
    "light/model_target/geometry/nonbonded": 0.5,
    "light/model_target/geometry/ramachandran": 0.5,
    # ADP (3 components)
    "light/model_target/adp/simu": 0.5,
    "light/model_target/adp/locality": 0.2,
    "light/model_target/adp/KL": 0.2,
    # Real-space targets
    "light/realspace/correlation": 0,
    "light/realspace_extrapolated": 0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------

def setup_mixed_model(pdb_dark, pdb_light, fractions, restraints_cif,
                      d_min, device, verbose):
    """Load dark and light ModelFT instances and combine into a MixedModel."""
    from torchref import ModelFT
    from torchref.model import MixedModel

    model_dark = (
        ModelFT(max_res=d_min, device=device, verbose=verbose)
        .load_pdb(pdb_dark)
    )
    model_light = (
        ModelFT(max_res=d_min, device=device, verbose=verbose)
        .load_pdb(pdb_light)
    )
    if restraints_cif:
        model_dark.set_restraints_cif(restraints_cif)
        model_light.set_restraints_cif(restraints_cif)

    mixed = MixedModel(
        [model_dark, model_light], initial_fractions=fractions
    )
    return mixed, model_dark, model_light


def setup_data(mtz_dark, mtz_light, d_min, device):
    """Load dark and light MTZ files into ReflectionData, with optional resolution cut."""
    from torchref import ReflectionData

    data_dark = ReflectionData(device=device).load_mtz(mtz_dark)
    data_light = ReflectionData(device=device).load_mtz(mtz_light)
    if d_min is not None:
        data_dark.cut_res(highres=d_min)
        data_light.cut_res(highres=d_min)
    return data_dark, data_light


def setup_collection(data_dark, data_light, device):
    """Bundle dark and light datasets into a scaled DatasetCollection."""
    from torchref import DatasetCollection

    collection = DatasetCollection(device=device)
    collection.add_dataset("dark", data_dark)
    collection.add_dataset("light", data_light)
    collection.scale()
    return collection


def setup_scaler(model, dataset, device):
    """Create a Scaler, compute initial scale and anisotropy, then refine."""
    from torchref import Scaler

    scaler = Scaler(model, dataset, device=device)
    scaler.calc_initial_scale()
    scaler.setup_anisotropy_correction()
    scaler.refine_lbfgs()
    return scaler


_DIFF_TARGET_CHOICES = {
    "amplitude": "DifferenceXrayTarget",
    "phase_informed": "PhaseInformedDifferenceTarget",
    "taylor": "TaylorCorrectedDifferenceTarget",
    "rice": "RiceDifferenceTarget",
}


def setup_loss_state(mixed_model, model_dark, model_light,
                     dataset_collection, scaler_dark, scaler_mixed,
                     target_weights, device, diff_target_type="amplitude"):
    """Build LossState with all targets and apply *target_weights*.

    Parameters
    ----------
    target_weights : dict
        Merged dictionary of target weights (DEFAULT_TARGET_WEIGHTS updated
        with any user overrides).  The difference-target weight is included
        as a starting value but will be overridden each schedule step.
    diff_target_type : str
        Which difference target to use. One of 'amplitude', 'phase_informed',
        'taylor', or 'rice'.
    """
    from torchref.refinement import LossState
    from torchref.refinement.targets import (
        DifferenceXrayTarget,
        PhaseInformedDifferenceTarget,
        TaylorCorrectedDifferenceTarget,
        RiceDifferenceTarget,
        TotalADPTarget,
        TotalGeometryTarget,
        MaximumLikelihoodXrayTarget,
        RealSpaceCorrelationTarget,
        RealSpaceExtrapolatedTarget,
    )

    diff_target_classes = {
        "amplitude": DifferenceXrayTarget,
        "phase_informed": PhaseInformedDifferenceTarget,
        "taylor": TaylorCorrectedDifferenceTarget,
        "rice": RiceDifferenceTarget,
    }

    DiffClass = diff_target_classes[diff_target_type]

    state = LossState(device=device)

    diff_target = DiffClass(
        dataset_collection=dataset_collection,
        model_dark=model_dark,
        model_light=mixed_model,
        scaler_dark=scaler_dark,
        scaler_light=scaler_mixed,
    )
    geom_target_light = TotalGeometryTarget(model_light)
    adp_target_light = TotalADPTarget(model_light)
    ml_target_light = MaximumLikelihoodXrayTarget(
        dataset_collection["light"], mixed_model, scaler=scaler_mixed
    )
    rs_mixed_target = RealSpaceCorrelationTarget(
        data=dataset_collection["light"],
        model=mixed_model,
        scaler=scaler_mixed,
    )
    rs_extra_target = RealSpaceExtrapolatedTarget(
        dataset_collection,
        model_dark=model_dark,
        model_light=model_light,
        model_mixed=mixed_model,
        scaler_dark=scaler_dark,
        scaler_mixed=scaler_mixed,
    )

    state.register_target(diff_target.name, diff_target)
    state.register_target(
        geom_target_light.name, geom_target_light, prefix="light"
    )
    state.register_target(
        adp_target_light.name, adp_target_light, prefix="light"
    )
    state.register_target(
        ml_target_light.name, ml_target_light, prefix="light"
    )
    state.register_target(
        rs_mixed_target.name, rs_mixed_target, prefix="light"
    )
    state.register_target(
        rs_extra_target.name, rs_extra_target, prefix="light"
    )

    state.set_weights(target_weights)

    return state


def optimize_lbfgs(state, parameters, max_iter, nsteps, n_clean, verbose):
    """Run a block of LBFGS optimisation steps."""
    parameters = list(parameters)
    optimizer = torch.optim.LBFGS(
        parameters, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = state.aggregate()
        loss.backward()
        return loss

    for i in range(nsteps):
        if i > 0 and i % n_clean == 0:
            optimizer = torch.optim.LBFGS(
                parameters, max_iter=max_iter, line_search_fn="strong_wolfe"
            )
        optimizer.step(closure)
        if verbose > 0:
            with torch.no_grad():
                loss = state.aggregate()
                print(f"    LBFGS step {i + 1}/{nsteps}, loss: {loss.item():.4f}")


def compute_bayes_extrapolated_amplitudes(
    Fobs_dark, Fobs_light, sig_dark, sig_light, phi_dark, phi_mixed, f,
    *, tau_sq_floor=1e-4,
):
    """Empirical Bayes shrinkage estimator for extrapolated SF amplitudes.

    Given dark- and mixed-state observed amplitudes with measurement
    uncertainties, compute posterior-mean extrapolated amplitudes for the
    pure excited state by shrinking noisy extrapolated values toward the
    dark-state amplitudes.  The shrinkage is per-reflection and driven
    entirely by the propagated variance sigma_ext^2(h) = sigma_dF^2(h)/f^2,
    so high-resolution / weakly-measured reflections are automatically
    damped without resolution binning.

    Parameters
    ----------
    Fobs_dark, Fobs_light : Tensor, shape (N,)
        Observed amplitudes (real, positive).
    sig_dark, sig_light : Tensor, shape (N,)
        Measurement standard deviations.
    phi_dark, phi_mixed : Tensor, shape (N,)
        Calculated phases (radians) for dark and mixed models.
    f : float or Tensor (scalar)
        Excited-state population fraction.
    tau_sq_floor : float
        Minimum value for the estimated signal variance tau^2.

    Returns
    -------
    F_ext_bayes : Tensor (N,)
        Posterior-mean extrapolated amplitudes.
    var_ext_bayes : Tensor (N,)
        Posterior variance per reflection.
    w_shrinkage : Tensor (N,)
        Shrinkage weight w(h) = tau^2 / (tau^2 + sigma_ext^2(h)).
    tau_sq : float
        Estimated global signal variance.
    """
    # --- Step 1: complex difference structure factors ---
    F_dark_phased = Fobs_dark * torch.exp(1j * phi_dark)
    F_light_phased = Fobs_light * torch.exp(1j * phi_mixed)
    delta_F = F_light_phased - F_dark_phased

    # Propagated variance on the complex difference (phases exact)
    sig_sq_dF = sig_light**2 + sig_dark**2

    # --- Step 2: noisy complex extrapolation ---
    F_ext_complex = F_dark_phased + delta_F / f
    F_ext_obs = torch.abs(F_ext_complex)

    # Propagated variance on extrapolated amplitude
    sig_sq_ext = sig_sq_dF / f**2

    # --- Step 3: estimate tau^2 (global signal variance) ---
    residuals_sq = (F_ext_obs - Fobs_dark) ** 2
    tau_sq = (residuals_sq.mean() - sig_sq_ext.mean()).item()
    tau_sq = max(tau_sq, tau_sq_floor)

    # --- Step 4: posterior mean (shrinkage) ---
    w = tau_sq / (tau_sq + sig_sq_ext)           # per-reflection weight
    w = w / w.mean()                             # normalise to mean 1
    F_ext_bayes = w * F_ext_obs + (1 - w) * Fobs_dark

    # --- Step 5: posterior variance ---
    var_ext_bayes = (tau_sq * sig_sq_ext) / (tau_sq + sig_sq_ext)

    return F_ext_bayes, var_ext_bayes, w, tau_sq


def write_results_mtz(data_dark, data_light, mixed_model, model_dark,
                      model_light, scaler_dark, scaler_mixed, filename):
    """Write difference / extrapolated map coefficients to an MTZ file."""
    import reciprocalspaceship as rs
    from torchref import ReflectionData, Scaler
    hkl_all, Fobs_dark_mt, sig_Fobs_dark_mt, _ = data_dark()
    _, Fobs_light_mt, sig_Fobs_light_mt, _ = data_light()

    mask = Fobs_dark_mt.get_mask() & Fobs_light_mt.get_mask()
    hkl = hkl_all[mask]
    Fobs_dark_vals = Fobs_dark_mt.get_data()[mask]
    Fobs_light_vals = Fobs_light_mt.get_data()[mask]
    sig_dark_vals = sig_Fobs_dark_mt.get_data()[mask]
    sig_light_vals = sig_Fobs_light_mt.get_data()[mask]
    rfree_flags_masked = data_light.rfree_flags[mask] if data_light.rfree_flags is not None else None

    fractions = mixed_model.fractions.detach()
    w_dark = fractions[0]
    w_light = fractions[1]

    # Calculate different types of extrapolated amplitudes for comparison, all using the same phases from the mixed model to ensure a fair comparison of amplitude estimates and resulting maps.
    # Compute on full HKL set then mask, because scalers were fitted on full datasets
    # and their internal bins/masks won't match an externally-masked HKL subset.
    fcalc_dark = scaler_dark(model_dark(hkl_all))[mask]
    fcalc_mixed = scaler_mixed(mixed_model(hkl_all))[mask]
    fcalc_diff = fcalc_mixed - fcalc_dark

    phi_dark = torch.angle(fcalc_dark)
    phi_mixed = torch.angle(fcalc_mixed)

    F_obs_dark_phased = Fobs_dark_vals * torch.exp(1j * phi_dark)
    F_obs_light_phased = Fobs_light_vals * torch.exp(1j * phi_mixed)

    F_light_extra = (                                                   # Phase aware extrapolation of the difference, scaled by the light fraction
        (F_obs_light_phased - w_dark * F_obs_dark_phased) / w_light
    )

    amp_light_extra = torch.abs(F_light_extra)
    sig_light_extra = torch.sqrt(sig_light_vals**2 + w_dark**2 * sig_dark_vals**2) / w_light

    data_light_extra = ReflectionData.from_tensors(
        hkl=hkl,
        F=amp_light_extra,
        F_sigma=sig_light_extra,
        cell=data_light.cell,
        spacegroup=data_light.spacegroup,
        rfree_flags=rfree_flags_masked,
        device=str(hkl.device),
        verbose=0,
    )

    scaler_light_extra = Scaler(model_light, data_light_extra, device=hkl.device, verbose=-1)
    scaler_light_extra.initialize().refine_lbfgs()

    F_light_calc_scaled_F_extra = scaler_light_extra(model_light(hkl))

    amp_extra = torch.abs(F_light_extra)
    amp_light_calc = torch.abs(F_light_calc_scaled_F_extra)

    phi_light_calc = torch.angle(F_light_calc_scaled_F_extra)
    amp_2fofc_light = 2 * amp_extra - amp_light_calc
    amp_fextfc = amp_extra - amp_light_calc

    # Classic (amplitude-only) extrapolation: F_ext = (|F_l| - w_d·|F_d|) / w_l
    amp_extra_classic = (Fobs_light_vals - w_dark * Fobs_dark_vals) / w_light
    
    sig_extra_classic = sig_light_extra  # same error propagation

    data_light_extra_classic = ReflectionData.from_tensors(
        hkl=hkl,
        F=amp_extra_classic,
        F_sigma=sig_extra_classic,
        cell=data_light.cell,
        spacegroup=data_light.spacegroup,
        rfree_flags=rfree_flags_masked,
        device=str(hkl.device),
        verbose=0,
    )

    scaler_light_extra_classic = Scaler(model_light, data_light_extra_classic, device=hkl.device, verbose=-1)
    scaler_light_extra_classic.initialize().refine_lbfgs()  

    F_light_calc_scaled_F_extra_classic = scaler_light_extra_classic(model_light(hkl))
    amp_light_calc_classic = torch.abs(F_light_calc_scaled_F_extra_classic)

    amp_2fofc_classic = 2 * amp_extra_classic - amp_light_calc_classic
    amp_fofc_classic = amp_extra_classic - amp_light_calc_classic



    # --- Empirical Bayes extrapolation ---
    F_ext_bayes, var_ext_bayes, w_shrinkage, tau_sq = (
        compute_bayes_extrapolated_amplitudes(
            Fobs_dark_vals, Fobs_light_vals,
            sig_dark_vals, sig_light_vals,
            phi_dark, phi_mixed, w_light,
        )
    )
    sig_ext_bayes = torch.sqrt(var_ext_bayes)

    data_light_extra_bayes = ReflectionData.from_tensors(
        hkl=hkl,
        F=F_ext_bayes,
        F_sigma=sig_ext_bayes,
        cell=data_light.cell,
        spacegroup=data_light.spacegroup,
        rfree_flags=rfree_flags_masked,
        device=str(hkl.device),
        verbose=0,
    )
    scaler_light_extra_bayes = Scaler(model_light, data_light_extra_bayes, device=hkl.device, verbose=-1)
    scaler_light_extra_bayes.initialize().refine_lbfgs()

    F_light_calc_scaled_F_extra_bayes = scaler_light_extra_bayes(model_light(hkl))

    # Map coefficients: use light-model Fcalc (already scaled above)
    amp_calc_bayes = torch.abs(F_light_calc_scaled_F_extra_bayes)
    amp_2fofc_bayes = 2 * F_ext_bayes - amp_calc_bayes
    amp_fofc_bayes = F_ext_bayes - amp_calc_bayes

    print("Phase-aware extrapolation rfactors:", scaler_light_extra.rfactor())
    print("Classic extrapolation rfactors:", scaler_light_extra_classic.rfactor())
    print("Bayes extrapolation rfactors:", scaler_light_extra_bayes.rfactor())
    print(f"  Bayes extrapolation: tau^2 = {tau_sq:.4f}, "
          f"mean w(h) = {w_shrinkage.mean().item():.3f}")

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

    # Complex observed difference: |F_obs_light·exp(iφ_mixed) - F_obs_dark·exp(iφ_dark)|
    # Pairs with phases_diff for a proper DED map
    Fobs_diff_phased = torch.abs(
        F_obs_light_phased - F_obs_dark_phased
    ).detach().cpu().numpy()

    diff_Fobs = Fobs_light - Fobs_dark
    sig_diff = (sig_dark**2 + sig_light**2) ** 0.5
    weights = 1 / sig_diff**2
    weights = weights / weights.mean()
    weighted_diff_Fobs = diff_Fobs * weights

    # 2Fo-Fc style DED: sigma-weighted, reduces phase bias
    amp_2DFoDFc = (2 * Fobs_diff_phased - Fcalc_diff_amp) * weights
    amp_DFoDFc = (Fobs_diff_phased - Fcalc_diff_amp) * weights

    df = rs.DataSet(
        {
            "H": hkl_np[:, 0],
            "K": hkl_np[:, 1],
            "L": hkl_np[:, 2],
            "Fobs_dark": Fobs_dark,
            "sig_Fobs_dark": sig_dark,
            "Fobs_light": Fobs_light,
            "sig_Fobs_light": sig_light,
            "DFo": diff_Fobs,
            "WDFo": weighted_diff_Fobs,
            "sig_DFo": sig_diff,
            "Fcalc_dark": Fcalc_dark,
            "Fcalc_light": Fcalc_light,
            "Fcalc_diff": Fcalc_diff_scalar,
            "Fcalc_diff_w_phases": Fcalc_diff_amp,
            "W2DFoDFc": amp_2DFoDFc,
            "WDFoDFc": amp_DFoDFc,
            "phase_dark": phases_dark,
            "phase_mixed": phases_mixed,
            "phase_diff": phases_diff,
            "phase_light": phi_light_calc.detach().rad2deg().cpu().numpy(),
            "2FextFc_light": amp_2fofc_light.detach().cpu().numpy(),
            "FextFc": amp_fextfc.detach().cpu().numpy(),
            "Fext_classic": amp_extra_classic.detach().cpu().numpy(),
            "sig_Fext_classic": sig_extra_classic.detach().cpu().numpy(),
            "2Fext_classic_Fc": amp_2fofc_classic.detach().cpu().numpy(),
            "Fext_classic_Fc": amp_fofc_classic.detach().cpu().numpy(),
            "Fext_bayes": F_ext_bayes.detach().cpu().numpy(),
            "sig_Fext_bayes": sig_ext_bayes.detach().cpu().numpy(),
            "2Fext_bayes_Fc": amp_2fofc_bayes.detach().cpu().numpy(),
            "Fext_bayes_Fc": amp_fofc_bayes.detach().cpu().numpy(),
        },
        cell=data_dark.cell.data.cpu().tolist(),
        spacegroup=data_dark.spacegroup.hm,
    )

    df[["H", "K", "L"]] = df[["H", "K", "L"]].astype("H")
    df[
        [
            "Fobs_dark", "Fobs_light", "DFo", "WDFo", "sig_DFo",
            "Fcalc_dark", "Fcalc_light", "Fcalc_diff", "Fcalc_diff_w_phases",
            "W2DFoDFc", "WDFoDFc",
            "2FextFc_light", "FextFc",
            "Fext_classic", "2Fext_classic_Fc", "Fext_classic_Fc",
            "Fext_bayes", "2Fext_bayes_Fc", "Fext_bayes_Fc"
        ]
    ] = df[
        [
            "Fobs_dark", "Fobs_light", "DFo", "WDFo", "sig_DFo",
            "Fcalc_dark", "Fcalc_light", "Fcalc_diff", "Fcalc_diff_w_phases",
            "W2DFoDFc", "WDFoDFc",
            "2FextFc_light", "FextFc",
            "Fext_classic", "2Fext_classic_Fc", "Fext_classic_Fc",
            "Fext_bayes", "2Fext_bayes_Fc", "Fext_bayes_Fc",
        ]
    ].astype("F")
    df[["sig_Fobs_dark", "sig_Fobs_light", "sig_DFo", "sig_Fext_classic", "sig_Fext_bayes"]] = df[
        ["sig_Fobs_dark", "sig_Fobs_light", "sig_DFo", "sig_Fext_classic", "sig_Fext_bayes"]
    ].astype("Q")
    df[["phase_dark", "phase_light", "phase_diff", "phase_mixed"]] = df[
        ["phase_dark", "phase_light", "phase_diff", "phase_mixed"]
    ].astype("P")
    df.set_index(["H", "K", "L"], inplace=True)
    df.write_mtz(filename)
    print(f"  Results MTZ written to {filename}")
    print(
        f"  w_dark={w_dark.item():.3f}, w_light={w_light.item():.3f}, "
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run difference refinement of a mixed model against "
                    "dark and light reflection data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic difference refinement
  torchref.difference-refine \\
      --dark-pdb dark.pdb --light-pdb light.pdb \\
      --dark-mtz dark.mtz --light-mtz light.mtz \\
      --fraction 0.37 -o output/

  # Custom weight schedule (anneal from 10 down to 1)
  torchref.difference-refine \\
      --dark-pdb dark.pdb --light-pdb light.pdb \\
      --dark-mtz dark.mtz --light-mtz light.mtz \\
      --fraction 0.37 \\
      --weight-schedule 10,5,3,1 --n-cycles 2 -o output/

  # With custom restraints and resolution cutoff
  torchref.difference-refine \\
      --dark-pdb dark.pdb --light-pdb light.pdb \\
      --dark-mtz dark.mtz --light-mtz light.mtz \\
      --fraction 0.37 \\
      --restraints-cif ligand.cif --max-res 1.7 -o output/

  # Override specific regularisation weights
  torchref.difference-refine \\
      --dark-pdb dark.pdb --light-pdb light.pdb \\
      --dark-mtz dark.mtz --light-mtz light.mtz \\
      --fraction 0.37 \\
      --target-weights '{"light/model_target/geometry/chiral": 5,
                         "light/model_target/adp/KL": 0.1}' -o output/
        """,
    )

    # --- Required arguments ---
    parser.add_argument(
        "--dark-pdb",
        required=True,
        type=str,
        help="PDB file for the dark / reference state",
    )
    parser.add_argument(
        "--light-pdb",
        required=True,
        type=str,
        help="PDB file for the light / triggered state",
    )
    parser.add_argument(
        "--dark-mtz",
        required=True,
        type=str,
        help="MTZ file with dark / reference reflection data",
    )
    parser.add_argument(
        "--light-mtz",
        required=True,
        type=str,
        help="MTZ file with light / triggered reflection data",
    )
    parser.add_argument(
        "--fraction",
        required=True,
        type=float,
        help="Occupancy fraction of the light/excited state "
             "(e.g. 0.37). Dark fraction is computed as 1 - fraction.",
    )
    parser.add_argument(
        "-o", "--outdir",
        required=True,
        type=str,
        help="Output directory for refined structures and maps",
    )

    # --- Difference target ---
    parser.add_argument(
        "--diff-target",
        type=str,
        default="amplitude",
        choices=list(_DIFF_TARGET_CHOICES.keys()),
        help="Difference target function: 'amplitude' (Gaussian NLL on "
             "|F_light|-|F_dark|), 'phase_informed' (complex with grafted "
             "phases, MSE), 'taylor' (Taylor-corrected complex, MSE), "
             "'rice' (Rice distribution NLL on complex difference "
             "amplitudes) (default: amplitude)",
    )

    # --- Weight schedule ---
    parser.add_argument(
        "--weight-schedule",
        type=str,
        default="5,3,2",
        help="Comma-separated difference-target weights applied in "
             "sequence each macro-cycle (default: '5,3,2')",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=3,
        help="Number of macro-cycles (repeats of the weight schedule) "
             "(default: 3)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2,
        help="LBFGS optimisation rounds per weight step (default: 2)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Max line-search iterations per LBFGS step (default: 100)",
    )
    parser.add_argument(
        "--n-clean",
        type=int,
        default=2,
        help="Reset LBFGS history every N steps (default: 2)",
    )

    # --- Optional arguments ---
    parser.add_argument(
        "--restraints-cif",
        type=str,
        nargs="+",
        default=None,
        help="One or more CIF restraint dictionaries",
    )
    parser.add_argument(
        "--max-res",
        type=float,
        default=None,
        help="High-resolution cutoff in Angstroms (uses data limit if not set)",
    )
    parser.add_argument(
        "--target-weights",
        type=str,
        default=None,
        help="JSON dictionary to override default regularisation weights. "
             "Keys are the full LossState paths.  Only the keys you supply "
             "are changed; the rest keep their defaults.  Defaults: "
             + json.dumps(DEFAULT_TARGET_WEIGHTS, indent=None),
    )
    parser.add_argument(
        "--refine-fractions",
        action="store_true",
        default=False,
        help="Refine population fractions during optimisation "
             "(default: fractions are frozen at initial values)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device (default: auto, uses CUDA if available)",
    )
    parser.add_argument(
        "-v", "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level: 0=quiet, 1=normal, 2=detailed (default: 1)",
    )

    args = parser.parse_args()

    from torchref.utils.timing import register_timing

    register_timing()

    # --- Parse fractions ---
    if not (0.0 < args.fraction < 1.0):
        print(
            "Error: --fraction must be between 0 and 1 "
            f"(got {args.fraction})",
            file=sys.stderr,
        )
        return 1
    fractions = [1.0 - args.fraction, args.fraction]

    # --- Parse weight schedule ---
    try:
        weight_schedule = [float(x) for x in args.weight_schedule.split(",")]
        if not weight_schedule:
            raise ValueError
    except ValueError:
        print(
            "Error: --weight-schedule must be comma-separated floats, "
            "e.g. '5,3,2'",
            file=sys.stderr,
        )
        return 1

    # --- Resolve difference target name ---
    # Import the target classes to get the .name attribute
    from torchref.refinement.targets import (
        DifferenceXrayTarget,
        PhaseInformedDifferenceTarget,
        TaylorCorrectedDifferenceTarget,
        RiceDifferenceTarget,
    )
    _diff_name_map = {
        "amplitude": DifferenceXrayTarget.name,
        "phase_informed": PhaseInformedDifferenceTarget.name,
        "taylor": TaylorCorrectedDifferenceTarget.name,
        "rice": RiceDifferenceTarget.name,
    }
    diff_target_name = _diff_name_map[args.diff_target]

    # --- Parse and merge target weights ---
    target_weights = dict(DEFAULT_TARGET_WEIGHTS)
    # Include the difference target with an initial value (will be
    # overridden each schedule step, but needs a key in the dict).
    target_weights[diff_target_name] = weight_schedule[0]

    if args.target_weights is not None:
        try:
            # Accept either a JSON file path or an inline JSON string
            if Path(args.target_weights).is_file():
                with open(args.target_weights) as f:
                    user_weights = json.load(f)
            else:
                user_weights = json.loads(args.target_weights)
            if not isinstance(user_weights, dict):
                raise ValueError("must be a JSON object")
            target_weights.update(user_weights)
        except (json.JSONDecodeError, ValueError) as e:
            print(
                f"Error: Invalid JSON for --target-weights: {e}",
                file=sys.stderr,
            )
            return 1

    # --- Validate input files ---
    for label, path in [
        ("dark PDB", args.dark_pdb),
        ("light PDB", args.light_pdb),
        ("dark MTZ", args.dark_mtz),
        ("light MTZ", args.light_mtz),
    ]:
        if not Path(path).exists():
            print(f"Error: {label} file not found: {path}", file=sys.stderr)
            return 1

    if args.restraints_cif:
        for cif_path in args.restraints_cif:
            if not Path(cif_path).exists():
                print(
                    f"Error: Restraints CIF not found: {cif_path}",
                    file=sys.stderr,
                )
                return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if args.device == "cuda" and not torch.cuda.is_available():
            print(
                "Warning: CUDA requested but not available, falling back to CPU",
                file=sys.stderr,
            )
            device = torch.device("cpu")

    # --- Header ---
    if args.verbose > 0:
        print("=" * 72)
        print("TorchRef Difference Refinement")
        print("=" * 72)
        print(f"Dark PDB:          {args.dark_pdb}")
        print(f"Light PDB:         {args.light_pdb}")
        print(f"Dark MTZ:          {args.dark_mtz}")
        print(f"Light MTZ:         {args.light_mtz}")
        frac_mode = "refinable" if args.refine_fractions else "frozen"
        print(f"Fractions:         dark={fractions[0]}, light={fractions[1]} ({frac_mode})")
        print(f"Output:            {outdir}")
        print(f"Device:            {device}")
        if args.max_res:
            print(f"Resolution cutoff: {args.max_res:.2f} A")
        if args.restraints_cif:
            print(f"Restraints CIF:    {', '.join(args.restraints_cif)}")
        print(f"Diff target:       {args.diff_target} ({diff_target_name})")
        print(f"Weight schedule:   {weight_schedule} x {args.n_cycles} cycles")
        print(f"LBFGS steps/weight: {args.n_steps}  (max_iter={args.max_iter})")
        print()
        print("Target weights:")
        for wk, wv in sorted(target_weights.items()):
            print(f"  {wk}: {wv}")
        print("=" * 72)
        print()
        sys.stdout.flush()

    # --- Resolution ---
    d_min = args.max_res if args.max_res is not None else 1.0

    # --- Setup ---
    if args.verbose > 0:
        print("Setting up models...")
        sys.stdout.flush()

    mixed, dark, light = setup_mixed_model(
        args.dark_pdb, args.light_pdb, fractions,
        args.restraints_cif, d_min, device, args.verbose,
    )

    if args.refine_fractions:
        mixed.unfreeze_fractions()
    else:
        mixed.freeze_fractions()

    if args.verbose > 0:
        print("Loading reflection data...")
        sys.stdout.flush()

    data_dark, data_light = setup_data(
        args.dark_mtz, args.light_mtz, args.max_res, device,
    )

    if args.verbose > 0:
        print("Scaling datasets...")
        sys.stdout.flush()

    collection = setup_collection(data_dark, data_light, device)

    if args.verbose > 0:
        print("Setting up scalers...")
        sys.stdout.flush()

    scaler_dark = setup_scaler(dark, data_dark, device)
    scaler_mixed = setup_scaler(mixed, data_light, device)

    if args.verbose > 0:
        r_work_d, r_free_d = scaler_dark.rfactor()
        r_work_l, r_free_l = scaler_mixed.rfactor()
        print(f"  Initial R-factor (dark):  R_work={r_work_d:.4f}  R_free={r_free_d:.4f}")
        print(f"  Initial R-factor (light): R_work={r_work_l:.4f}  R_free={r_free_l:.4f}")
        print()
        sys.stdout.flush()

    state = setup_loss_state(
        mixed, dark, light, collection,
        scaler_dark, scaler_mixed, target_weights, device,
        diff_target_type=args.diff_target,
    )

    if args.verbose > 0:
        print("Initial loss breakdown:")
        state.summary()
        print()
        sys.stdout.flush()

    # --- Refinement loop ---
    total_rounds = args.n_cycles * len(weight_schedule)
    round_idx = 0

    # Collect parameters for optimisation
    if args.refine_fractions:
        params = itertools.chain(light.parameters(), [mixed.fraction_params])
    else:
        params = light.parameters()
    params = list(params)

    # Track fraction history for plotting
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

            state.set_weights({diff_target_name: t_weight})
            optimize_lbfgs(
                state, params,
                max_iter=args.max_iter,
                nsteps=args.n_steps,
                n_clean=args.n_clean,
                verbose=args.verbose,
            )
            scaler_mixed.refine_lbfgs()

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
        r_work_d, r_free_d = scaler_dark.rfactor()
        r_work_l, r_free_l = scaler_mixed.rfactor()
        print(f"  Final R-factor (dark):  R_work={r_work_d:.4f}  R_free={r_free_d:.4f}")
        print(f"  Final R-factor (light): R_work={r_work_l:.4f}  R_free={r_free_l:.4f}")
        print(
            f"  Refined fractions:      "
            f"{mixed.fractions.detach().cpu().numpy()}"
        )
        print()
        sys.stdout.flush()

    # --- Save outputs ---
    prefix = (
        f"fractions_{int(fractions[0]*100)}_{int(fractions[1]*100)}"
    )

    dark_pdb_out = str(outdir / f"{prefix}_dark.pdb")
    light_pdb_out = str(outdir / f"{prefix}_light.pdb")
    light_mtz_out = str(outdir / f"{prefix}_refined_light.mtz")
    diff_mtz_out = str(outdir / f"{prefix}_difference_data.mtz")

    dark.write_pdb(dark_pdb_out)
    light.write_pdb(light_pdb_out)

    write_results_mtz(
        data_dark, data_light, mixed, dark, light,
        scaler_dark, scaler_mixed, diff_mtz_out,
    )

    # --- JSON summary ---
    summary = {
        "input": {
            "dark_pdb": args.dark_pdb,
            "light_pdb": args.light_pdb,
            "dark_mtz": args.dark_mtz,
            "light_mtz": args.light_mtz,
            "fractions": fractions,
            "restraints_cif": args.restraints_cif,
            "max_res": args.max_res,
        },
        "parameters": {
            "diff_target": args.diff_target,
            "weight_schedule": weight_schedule,
            "n_cycles": args.n_cycles,
            "n_steps": args.n_steps,
            "max_iter": args.max_iter,
            "target_weights": target_weights,
        },
        "results": {
            "r_factor_dark": dict(zip(["r_work", "r_free"], scaler_dark.rfactor())),
            "r_factor_light": dict(zip(["r_work", "r_free"], scaler_mixed.rfactor())),
            "fractions": mixed.fractions.detach().cpu().tolist(),
        },
        "output_files": {
            "dark_pdb": dark_pdb_out,
            "light_pdb": light_pdb_out,
            "light_mtz": light_mtz_out,
            "difference_mtz": diff_mtz_out,
        },
    }

    summary_path = outdir / f"{prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(convert_to_serializable(summary), f, indent=2)

    # --- Plot light fraction vs cycle ---
    if args.refine_fractions and fraction_history:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(len(fraction_history)), fraction_history, "o-", color="C0")
        ax.set_xlabel("Refinement step")
        ax.set_ylabel("Light fraction")
        ax.set_title("Light fraction vs refinement step")
        ax.axhline(fractions[1], ls="--", color="gray", label=f"initial ({fractions[1]:.3f})")
        ax.legend()
        fig.tight_layout()
        frac_plot_path = str(outdir / f"{prefix}_fraction_vs_cycle.png")
        fig.savefig(frac_plot_path, dpi=150)
        plt.close(fig)
        if args.verbose > 0:
            print(f"  Fraction plot: {frac_plot_path}")

    if args.verbose > 0:
        print()
        print("Output files:")
        print(f"  - {dark_pdb_out}")
        print(f"  - {light_pdb_out}")
        print(f"  - {light_mtz_out}")
        print(f"  - {diff_mtz_out}")
        print(f"  - {summary_path}")
        if args.refine_fractions and fraction_history:
            print(f"  - {frac_plot_path}")
        print()
        print("Done.")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
