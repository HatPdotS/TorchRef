#!/usr/bin/env python3 -u

"""
Command-line script for LBFGS refinement with STATIC weighting.

This script uses ``ManualWeighting`` with fixed component weights
(default: xray=1.0, geometry=10.0, adp=5.0).  Weights can be overridden
via a JSON file passed with ``--weights``.

Use this as a baseline to compare against hyperparameter-tuned refinement.

Examples
--------
::

    torchref.refine-static -m model.pdb -sf reflections.mtz -o output_dir/
    torchref.refine-static -m model.pdb -sf reflections.mtz -o output/ -n 10
"""

import argparse
import json
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
    add_weights_arg,
    build_column_names,
    configure_unbuffered_output,
    parse_weights,
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
        description="Run LBFGS refinement with static/default weighting (no hyperparameter optimization)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic refinement with static weights
  torchref.refine-static -m model.pdb -sf reflections.mtz -o output_dir/

  # With 10 refinement cycles
  torchref.refine-static -m model.pdb -sf reflections.mtz -o output/ -n 10
        """,
    )

    add_single_model_args(parser)

    output = parser.add_argument_group("Output")
    add_outdir_arg(output)
    add_output_format_args(output)
    add_metadata_args(output)

    refine = parser.add_argument_group("Refinement")
    add_n_cycles_arg(refine)
    add_weights_arg(refine, default_weights={"xray": 1.0, "geometry": 10.0, "adp": 5.0})

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
        print("TorchRef LBFGS Refinement - STATIC WEIGHTING")
        print("=" * 80)
        print(
            "Weighting scheme: ComponentWeighting (default, no hyperparameter tuning)"
        )
        print(f"Model:            {model_path}")
        print(f"Structure factor: {sf_path}")
        print(f"Output directory: {outdir}")
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

    refinement = LBFGSRefinement(
        data_file=str(sf_path),
        pdb=str(model_path),
        cif=args.cif,
        verbose=args.verbose,
        max_res=args.dmin,
        device=device,
        column_names=column_names,
    )

    from torchref.refinement.weighting import ManualWeighting

    base_weights, weights_error = parse_weights(
        args.weights, defaults={"xray": 1.0, "geometry": 10.0, "adp": 5.0}
    )
    if weights_error is not None:
        print(f"Error: {weights_error}", file=sys.stderr)
        return 1

    manual_weighting = ManualWeighting(weights=base_weights, device=device)

    refinement.component_weighting = manual_weighting

    if args.verbose > 0:
        print("Refinement initialized successfully.")
        print(
            "Using static ComponentWeighting (XrayScale + TargetOffset + Overfitting)"
        )
        print("No hyperparameters loaded - using default values.\n")
        sys.stdout.flush()

    # Run refinement (no hyperparameters applied)
    try:
        if args.verbose > 0:
            print(f"Starting refinement with {args.n_cycles} macro cycles...\n")
            sys.stdout.flush()

        refinement.refine_everything(macro_cycles=args.n_cycles)

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

    history_data = {
        "weighting_scheme": "static",
        "input_files": {
            "model": str(model_path),
            "structure_factor": str(sf_path),
            "cif": args.cif,
        },
        "parameters": {
            "n_cycles": args.n_cycles,
            "dmin": args.dmin,
            "device": str(device),
        },
        "history": refinement.history if hasattr(refinement, "history") else {},
        "final_statistics": {},
    }

    # Add final R-factors if available
    try:
        work_nll, test_nll = refinement.nll_xray()
        hkl, fobs, sigma, rfree = refinement.reflection_data()
        fcalc = refinement.get_F_calc_scaled(hkl, recalc=True)

        work_mask = rfree
        test_mask = ~rfree

        r_work = torch.sum(
            torch.abs(fobs[work_mask] - fcalc[work_mask])
        ) / torch.sum(fobs[work_mask])
        r_free = torch.sum(
            torch.abs(fobs[test_mask] - fcalc[test_mask])
        ) / torch.sum(fobs[test_mask])

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
        print(f"  - {output_pdb}")
        if (outdir / "refined.mtz").exists():
            print(f"  - {outdir / 'refined.mtz'}")
        print(f"  - {output_json}")
        print("\nRefinement completed successfully!")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
