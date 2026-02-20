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
step, matching the protocol in Seidel et al.

"""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import torch

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
    "light/model_target/geometry/angle": 0.5,
    "light/model_target/geometry/torsion": 0.5,
    "light/model_target/geometry/planarity": 2.0,
    "light/model_target/geometry/chiral": 3.0,
    "light/model_target/geometry/nonbonded": 0.5,
    "light/model_target/geometry/ramachandran": 1.0,
    # ADP (3 components)
    "light/model_target/adp/simu": 3.0,
    "light/model_target/adp/locality": 0.5,
    "light/model_target/adp/KL": 0.3,
    # Real-space targets
    "light/realspace/correlation": 5,
    "light/realspace_extrapolated": 10,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_device():
    """Pick CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def convert_to_serializable(obj):
    """Convert tensors and numpy arrays to JSON-serializable types."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.numel() > 1 else obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
        except ImportError:
            pass
        return obj


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------

def setup_mixed_model(pdb_dark, pdb_light, fractions, restraints_cif,
                      d_min, device, verbose):
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
    from torchref import ReflectionData

    data_dark = ReflectionData(device=device).load_mtz(mtz_dark)
    data_light = ReflectionData(device=device).load_mtz(mtz_light)
    if d_min is not None:
        data_dark.cut_res(highres=d_min)
        data_light.cut_res(highres=d_min)
    return data_dark, data_light


def setup_collection(data_dark, data_light, device):
    from torchref import DatasetCollection

    collection = DatasetCollection(device=device)
    collection.add_dataset("dark", data_dark)
    collection.add_dataset("light", data_light)
    collection.scale()
    return collection


def setup_scaler(model, dataset, device):
    from torchref import Scaler

    scaler = Scaler(model, dataset, device=device)
    scaler.calc_initial_scale()
    scaler.setup_anisotropy_correction()
    scaler.refine_lbfgs()
    return scaler


def setup_loss_state(mixed_model, model_dark, model_light,
                     dataset_collection, scaler_dark, scaler_mixed,
                     target_weights, device):
    """Build LossState with all targets and apply *target_weights*.

    Parameters
    ----------
    target_weights : dict
        Merged dictionary of target weights (DEFAULT_TARGET_WEIGHTS updated
        with any user overrides).  The difference-target weight is included
        as a starting value but will be overridden each schedule step.
    """
    from torchref.refinement import LossState
    from torchref.refinement.targets import (
        DifferenceXrayTarget,
        TotalADPTarget,
        TotalGeometryTarget,
        MaximumLikelihoodXrayTarget,
        RealSpaceCorrelationTarget,
        RealSpaceExtrapolatedTarget,
    )

    state = LossState(device=device)

    diff_target = DifferenceXrayTarget(
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


def write_results_mtz(data_dark, data_light, mixed_model, model_dark,
                      model_light, scaler_dark, scaler_mixed, filename):
    """Write difference / extrapolated map coefficients to an MTZ file."""
    import reciprocalspaceship as rs
    from torchref import ReflectionData, Scaler
    hkl, Fobs_dark_mt, sig_Fobs_dark_mt, _ = data_dark()
    _, Fobs_light_mt, sig_Fobs_light_mt, _ = data_light()

    mask = Fobs_dark_mt.get_mask() & Fobs_light_mt.get_mask()
    Fobs_dark_vals = Fobs_dark_mt.get_data()
    Fobs_light_vals = Fobs_light_mt.get_data()
    sig_dark_vals = sig_Fobs_dark_mt.get_data()
    sig_light_vals = sig_Fobs_light_mt.get_data()

    fractions = mixed_model.fractions.detach()
    w_dark = fractions[0]
    w_light = fractions[1]

    with torch.no_grad():
        fcalc_dark = scaler_dark(model_dark(hkl))
        fcalc_mixed = scaler_mixed(mixed_model(hkl))
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
            rfree_flags=data_light.rfree_flags,
            device=str(hkl.device),
            verbose=0,
        )

        scaler_light_extra = Scaler(model_light, data_light_extra, device=hkl.device)
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
            rfree_flags=data_light.rfree_flags,
            device=str(hkl.device),
            verbose=0,
        )

        scaler_light_extra_classic = Scaler(model_light, data_light_extra_classic, device=hkl.device)
        scaler_light_extra_classic.initialize().refine_lbfgs()  

        F_light_calc_scaled_F_extra_classic = scaler_light_extra_classic(model_light(hkl))
        amp_light_calc_classic = torch.abs(F_light_calc_scaled_F_extra_classic)

        amp_2fofc_classic = 2 * amp_extra_classic - amp_light_calc_classic
        amp_fofc_classic = amp_extra_classic - amp_light_calc_classic

    m = mask
    hkl_np = hkl[m].cpu().numpy()
    Fobs_dark = Fobs_dark_vals[m].cpu().numpy()
    Fobs_light = Fobs_light_vals[m].cpu().numpy()
    sig_dark = sig_dark_vals[m].cpu().numpy()
    sig_light = sig_light_vals[m].cpu().numpy()

    Fcalc_dark = torch.abs(fcalc_dark[m]).cpu().numpy()
    Fcalc_light = torch.abs(fcalc_mixed[m]).cpu().numpy()
    phases_dark = torch.angle(fcalc_dark[m]).rad2deg().cpu().numpy()
    phases_mixed = torch.angle(fcalc_mixed[m]).rad2deg().cpu().numpy()

    Fcalc_diff_amp = torch.abs(fcalc_diff[m]).cpu().numpy()
    Fcalc_diff_scalar = Fcalc_light - Fcalc_dark
    phases_diff = torch.angle(fcalc_diff[m]).rad2deg().cpu().numpy()

    # Complex observed difference: |F_obs_light·exp(iφ_mixed) - F_obs_dark·exp(iφ_dark)|
    # Pairs with phases_diff for a proper DED map
    Fobs_diff_phased = torch.abs(
        F_obs_light_phased[m] - F_obs_dark_phased[m]
    ).cpu().numpy()

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
            "phase_light": phi_light_calc[m].rad2deg().cpu().numpy(),
            "2FextFc_light": amp_2fofc_light[m].cpu().numpy(),
            "FextFc": amp_fextfc[m].cpu().numpy(),
            "Fext_classic": amp_extra_classic[m].cpu().numpy(),
            "sig_Fext_classic": sig_extra_classic[m].cpu().numpy(),
            "2Fext_classic_Fc": amp_2fofc_classic[m].cpu().numpy(),
            "Fext_classic_Fc": amp_fofc_classic[m].cpu().numpy(),
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
        ]
    ] = df[
        [
            "Fobs_dark", "Fobs_light", "DFo", "WDFo", "sig_DFo",
            "Fcalc_dark", "Fcalc_light", "Fcalc_diff", "Fcalc_diff_w_phases",
            "W2DFoDFc", "WDFoDFc",
            "2FextFc_light", "FextFc",
            "Fext_classic", "2Fext_classic_Fc", "Fext_classic_Fc",
        ]
    ].astype("F")
    df[["sig_Fobs_dark", "sig_Fobs_light", "sig_Fobs_diff", "sig_Fext_classic"]] = df[
        ["sig_Fobs_dark", "sig_Fobs_light", "sig_Fobs_diff", "sig_Fext_classic"]
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
  torchref-difference-refine \\
      --dark-pdb dark.pdb --light-pdb light.pdb \\
      --dark-mtz dark.mtz --light-mtz light.mtz \\
      --fractions 0.63,0.37 -o output/

  # Custom weight schedule (anneal from 10 down to 1)
  torchref-difference-refine \\
      --dark-pdb dark.pdb --light-pdb light.pdb \\
      --dark-mtz dark.mtz --light-mtz light.mtz \\
      --fractions 0.63,0.37 \\
      --weight-schedule 10,5,3,1 --n-cycles 2 -o output/

  # With custom restraints and resolution cutoff
  torchref-difference-refine \\
      --dark-pdb dark.pdb --light-pdb light.pdb \\
      --dark-mtz dark.mtz --light-mtz light.mtz \\
      --fractions 0.63,0.37 \\
      --restraints-cif ligand.cif --max-res 1.7 -o output/

  # Override specific regularisation weights
  torchref-difference-refine \\
      --dark-pdb dark.pdb --light-pdb light.pdb \\
      --dark-mtz dark.mtz --light-mtz light.mtz \\
      --fractions 0.63,0.37 \\
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
        "--fractions",
        required=True,
        type=str,
        help="Comma-separated dark,light occupancy fractions "
             "(e.g. '0.63,0.37')",
    )
    parser.add_argument(
        "-o", "--outdir",
        required=True,
        type=str,
        help="Output directory for refined structures and maps",
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
        "-v", "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level: 0=quiet, 1=normal, 2=detailed (default: 1)",
    )

    args = parser.parse_args()

    # --- Parse fractions ---
    try:
        fractions = [float(x) for x in args.fractions.split(",")]
        if len(fractions) != 2:
            raise ValueError
    except ValueError:
        print(
            "Error: --fractions must be two comma-separated floats, "
            "e.g. '0.63,0.37'",
            file=sys.stderr,
        )
        return 1

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

    # --- Parse and merge target weights ---
    target_weights = dict(DEFAULT_TARGET_WEIGHTS)
    # Include the difference target with an initial value (will be
    # overridden each schedule step, but needs a key in the dict).
    target_weights["difference_xray"] = weight_schedule[0]

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
    device = _auto_device()

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

            state.set_weights({"difference_xray": t_weight})
            optimize_lbfgs(
                state, params,
                max_iter=args.max_iter,
                nsteps=args.n_steps,
                n_clean=args.n_clean,
                verbose=args.verbose,
            )
            scaler_mixed.refine_lbfgs()

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

    if args.verbose > 0:
        print()
        print("Output files:")
        print(f"  - {dark_pdb_out}")
        print(f"  - {light_pdb_out}")
        print(f"  - {light_mtz_out}")
        print(f"  - {diff_mtz_out}")
        print(f"  - {summary_path}")
        print()
        print("Done.")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
