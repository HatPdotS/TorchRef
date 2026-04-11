#!/usr/bin/env python3 -u

"""
Command-line script for LBFGS refinement with HYPERPARAMETER-TUNED weighting.

This script uses ComponentWeighting (ResolutionWeighting + OverfittingWeighting)
with hyperparameters loaded from file.

The hyperparameters were optimized to achieve the best R-free across a diverse
set of protein structures at various resolutions.

Options for ``--hyperparameters``:

- ``"default"``: Load optimized hyperparameters from package data.
- Custom path: Load from user-specified JSON file.
- ``"none"``: Skip hyperparameter loading (same as static weighting).

Examples
--------
::

    torchref.refine-hyper -m model.pdb -sf reflections.mtz -o output_dir/
    torchref.refine-hyper -m model.pdb -sf reflections.mtz -o output/ --hyperparameters my_params.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from torchref.cli._common import (
    add_dmin_arg,
    add_general_args,
    add_metadata_args,
    add_n_cycles_arg,
    add_outdir_arg,
    add_output_format_args,
    add_single_model_args,
    build_column_names,
    configure_unbuffered_output,
    register_timing,
    resolve_device,
    validate_files,
    write_refinement_outputs,
)

configure_unbuffered_output()

# Import stats module early to patch json with StatEntry encoder
import torchref.utils.stats  # noqa: F401


def main():
    parser = argparse.ArgumentParser(
        description="Run LBFGS refinement with Optuna-optimized hyperparameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Refinement with default optimized hyperparameters
  torchref.refine-hyper -m model.pdb -sf reflections.mtz -o output_dir/

  # With custom hyperparameters file
  torchref.refine-hyper -m model.pdb -sf reflections.mtz -o output/ --hyperparameters my_params.json

  # Skip hyperparameters (equivalent to static weighting)
  torchref.refine-hyper -m model.pdb -sf reflections.mtz -o output/ --hyperparameters none
        """,
    )

    add_single_model_args(parser)

    output = parser.add_argument_group("Output")
    add_outdir_arg(output)
    add_output_format_args(output)
    add_metadata_args(output)

    refine = parser.add_argument_group("Refinement")
    add_n_cycles_arg(refine)
    refine.add_argument(
        "--hyperparameters",
        type=str,
        default="default",
        help='Path to hyperparameters JSON file, or "default" to use optimized defaults, '
        'or "none" to skip. The JSON file can be edited to customize refinement behavior. '
        '(default: "default" uses Optuna-optimized hyperparameters)',
    )
    refine.add_argument(
        "--mode",
        type=str,
        default="everything",
        choices=["everything", "refine"],
        help='Refinement mode: "everything" for joint XYZ+ADP+scaler LBFGS, '
        '"refine" for separated XYZ then ADP cycles (default: "everything")',
    )
    refine.add_argument(
        "--xray-mode",
        type=str,
        default="ml",
        choices=["gaussian", "ls", "ml", "bhattacharyya"],
        help="X-ray target function. 'bhattacharyya' uses the Bhattacharyya "
        "overlap loss with first-principles model error estimation and does "
        "not need manual weight tuning (default: 'ml')",
    )
    refine.add_argument(
        "--sigma-m-scale",
        type=float,
        default=1.0,
        help="Global multiplier applied to σ_m for the Bhattacharyya target. "
        "Ignored for other targets. Default 1.0.",
    )
    refine.add_argument(
        "--sigma-weighting",
        type=str,
        default="per_refl",
        choices=["per_refl", "const"],
        help="Bhattacharyya-only: Fisher-info weighting scheme. 'per_refl' "
        "(default) weights by 1/σ²(h); 'const' weights uniformly by "
        "1/<σ>² across valid reflections (v1-like, robust to σ_d variance).",
    )
    refine.add_argument(
        "--info-sum-mode",
        type=str,
        default="g_w",
        choices=["g_w", "n_eff"],
        help="Bhattacharyya-only: how per-atom Fisher info is summed. "
        "'g_w' (default) uses Σ_h (|s|²/σ²) exp(-2Bs²/4); 'n_eff' uses "
        "Kish participation ratio (Σ exp)²/(Σ exp²) scaled by mean Fisher "
        "weight — the v1-style.",
    )
    refine.add_argument(
        "--scatterer-profile",
        type=str,
        default="unit",
        choices=["unit", "protein_rep"],
        help="Bhattacharyya-only: atomic scattering factor used in σ_m. "
        "'unit' (default) assumes all atoms are unit scatterers (f=1); "
        "'protein_rep' uses carbon ITC92 scattering factor as a "
        "representative for proteins.",
    )

    res = parser.add_argument_group("Resolution")
    add_dmin_arg(res)

    add_general_args(parser)

    args = parser.parse_args()

    register_timing()

    # Validate inputs
    model_path = Path(args.model)
    sf_path = Path(args.structure_factor)
    outdir = Path(args.outdir)

    validate_files(
        [(str(model_path), "Model"), (str(sf_path), "Structure factors")],
        exit_on_error=True,
    )

    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)

    # Import here to avoid slow startup for --help
    try:
        from torchref.refinement.lbfgs_refinement import LBFGSRefinement
    except ImportError as e:
        print(f"Error: Failed to import torchref modules: {e}", file=sys.stderr)
        print("Please ensure torchref is properly installed.", file=sys.stderr)
        sys.exit(1)

    # Print header
    if args.verbose > 0:
        print("=" * 80)
        print("TorchRef LBFGS Refinement - HYPERPARAMETER-TUNED")
        print("=" * 80)
        print("Weighting scheme: ComponentWeighting with optimized hyperparameters")
        print(f"Hyperparameters:  {args.hyperparameters}")
        print(f"Model:            {model_path}")
        print(f"Structure factor: {sf_path}")
        print(f"Output directory: {outdir}")
        print(f"Refinement mode:  {args.mode}")
        print(f"Refinement cycles: {args.n_cycles}")
        print(f"Device:           {args.device}")
        if args.dmin:
            print(f"Resolution cutoff: {args.dmin:.2f} A")
        print("=" * 80)
        print()
        sys.stdout.flush()

    # Setup device
    device = resolve_device(args.device)

    if args.verbose > 0:
        print("Initializing refinement...")
        sys.stdout.flush()

    # Build column_names for MTZ loading
    column_names = build_column_names(args.column_structure_factor, args.column_sigma)

    # Initialize refinement
    refinement = LBFGSRefinement(
        data_file=str(sf_path),
        pdb=str(model_path),
        cif=args.cif,
        verbose=args.verbose,
        max_res=args.dmin,
        device=device,
        column_names=column_names,
        target_mode=args.xray_mode,
        sigma_m_scale=args.sigma_m_scale,
        sigma_weighting=args.sigma_weighting,
        info_sum_mode=args.info_sum_mode,
        scatterer_profile=args.scatterer_profile,
    )

    if args.verbose > 0:
        print("Refinement initialized successfully.\n")
        sys.stdout.flush()

    # Load and apply hyperparameters
    hyperparams_source = None
    n_hyperparams = 0

    if args.hyperparameters.lower() != "none":
        try:
            from torchref.utils.utils import json_to_state_dicts_separate

            if args.hyperparameters.lower() == "default":
                # Load default hyperparameters from package data
                from torchref import PATH_TORCHREF_DATA

                hyperparams_path = os.path.join(
                    PATH_TORCHREF_DATA, "default_hyperparameters.json"
                )

                hyperparams_source = "package default (Optuna-optimized)"
                if args.verbose > 0:
                    print("Loading optimized default hyperparameters...")
                    sys.stdout.flush()
            else:
                # Load from user-specified file
                hyperparams_path = Path(args.hyperparameters)
                if not hyperparams_path.exists():
                    print(
                        f"Error: Hyperparameters file not found: {hyperparams_path}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                hyperparams_source = str(hyperparams_path)
                if args.verbose > 0:
                    print(f"Loading hyperparameters from: {hyperparams_path}")
                    sys.stdout.flush()

            # Convert to state dict and apply
            (
                component_weighting_state,
                geometry_target_state,
                adp_target_state,
                unassigned_keys,
            ) = json_to_state_dicts_separate(hyperparams_path)

            refinement.component_weighting.load_state_dict(
                component_weighting_state, strict=False
            )
            refinement.geometry_target.load_state_dict(
                geometry_target_state, strict=False
            )
            refinement.adp_target.load_state_dict(adp_target_state, strict=False)

            n_hyperparams = (
                len(component_weighting_state)
                + len(geometry_target_state)
                + len(adp_target_state)
            )

            if unassigned_keys and args.verbose > 1:
                print("Warning: Unassigned hyperparameter keys in JSON:")
                for key in unassigned_keys:
                    print(f"  - {key}")
                print()
                sys.stdout.flush()

            if args.verbose > 0:
                print(f"Applied {n_hyperparams} hyperparameters.\n")
                sys.stdout.flush()

        except Exception as e:
            print(f"Warning: Could not load hyperparameters: {e}", file=sys.stderr)
            if args.verbose > 1:
                import traceback

                traceback.print_exc()
    else:
        if args.verbose > 0:
            print("Hyperparameters skipped - using default values.\n")
            sys.stdout.flush()

    # Run refinement
    try:
        if args.verbose > 0:
            print(f"Starting refinement with {args.n_cycles} macro cycles...\n")
            sys.stdout.flush()

        if args.mode == "everything":
            refinement.refine_everything(macro_cycles=args.n_cycles)
        else:
            refinement.refine(macro_cycles=args.n_cycles)

        refinement.get_scales()

        if args.verbose > 0:
            print("\nRefinement completed successfully.")
            sys.stdout.flush()

    except Exception as e:
        refinement.debug_on_error(e)
        raise e

    if args.verbose > 0:
        print(f"\nSaving results to {outdir}...")
        sys.stdout.flush()

    # Save refined structure(s) with metadata
    outputs = write_refinement_outputs(refinement, outdir, args, verbose=args.verbose)

    # Save refined structure factors
    output_mtz = outdir / "refined.mtz"
    hkl, fobs, sigma, rfree = refinement.reflection_data()
    fcalc = refinement.get_F_calc_scaled(hkl, recalc=True)
    refinement.write_out_mtz(str(output_mtz))

    if args.verbose > 0:
        print(f"  Refined structure factors: {output_mtz}")
        sys.stdout.flush()

    # Save refinement history as JSON
    output_json = outdir / "refinement_history.json"

    # Prepare history data
    history_data = {
        "weighting_scheme": "hyperparameters",
        "input_files": {
            "model": str(model_path),
            "structure_factor": str(sf_path),
            "cif": args.cif,
        },
        "parameters": {
            "n_cycles": args.n_cycles,
            "dmin": args.dmin,
            "device": str(device),
            "hyperparameters_source": hyperparams_source,
            "n_hyperparameters": n_hyperparams,
        },
        "history": refinement.history if hasattr(refinement, "history") else {},
        "final_statistics": {},
    }

    # Add final R-factors if available
    try:
        work_nll, test_nll = refinement.nll_xray()
        hkl, fobs, sigma, rfree = refinement.reflection_data()
        fcalc = refinement.get_F_calc_scaled(hkl, recalc=True)

        # Calculate R-factors
        work_mask = rfree
        test_mask = ~rfree

        r_work = torch.sum(torch.abs(fobs[work_mask] - fcalc[work_mask])) / torch.sum(
            fobs[work_mask]
        )
        r_free = torch.sum(torch.abs(fobs[test_mask] - fcalc[test_mask])) / torch.sum(
            fobs[test_mask]
        )

        history_data["final_statistics"] = {
            "R_work": float(r_work.item()),
            "R_free": float(r_free.item()),
            "NLL_work": float(work_nll.item()),
            "NLL_test": float(test_nll.item()),
            "n_reflections_work": int(work_mask.sum().item()),
            "n_reflections_test": int(test_mask.sum().item()),
        }
    except Exception as e:
        if args.verbose > 1:
            print(f"  Warning: Could not compute final statistics: {e}")

    with open(output_json, "w") as f:
        json.dump(history_data, f, indent=2)

    if args.verbose > 0:
        print(f"  Refinement history: {output_json}")
        sys.stdout.flush()

    # Print final summary
    if args.verbose > 0:
        print("\n" + "=" * 80)
        print("Refinement Summary")
        print("=" * 80)

        if "final_statistics" in history_data and history_data["final_statistics"]:
            stats = history_data["final_statistics"]
            print(
                f"R-work:  {stats['R_work']:.4f} ({stats['n_reflections_work']} reflections)"
            )
            print(
                f"R-free:  {stats['R_free']:.4f} ({stats['n_reflections_test']} reflections)"
            )
            print(f"NLL work: {stats['NLL_work']:.2f}")
            print(f"NLL test: {stats['NLL_test']:.2f}")

        print("=" * 80)
        print("\nOutput files:")
        for fmt in ("pdb", "cif"):
            if outputs.get(fmt) is not None:
                print(f"  - {outputs[fmt]}")
        if (outdir / "refined.mtz").exists():
            print(f"  - {outdir / 'refined.mtz'}")
        print(f"  - {output_json}")
        print("\nRefinement completed successfully!")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
