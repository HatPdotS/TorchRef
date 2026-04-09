#!/usr/bin/env python3 -u

"""
Command-line script for LBFGS refinement with POLICY-BASED weighting.

This script uses ``PolicyComponentWeighting`` which predicts component weights
from the current refinement state using a trained neural network policy.

The policy network was trained on trajectories collected with random weighting
using Advantage-Weighted Regression (AWR) to learn optimal weight selection.

Two modes:

- Evaluation mode (default): Use mean policy predictions for deterministic refinement.
- Sampling mode: Sample from policy distribution for exploration/data collection.

Examples
--------
::

    torchref.refine-policy -m model.pdb -sf reflections.mtz -o output/ --policy policy.pt
    torchref.refine-policy -m model.pdb -sf reflections.mtz -o output/ --policy policy.pt --sample
"""

import argparse
import json
import sys
import time
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
        description="Run LBFGS refinement with policy-based weighting (trained neural network)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic refinement with trained policy
  torchref.refine-policy -m model.pdb -sf reflections.mtz -o output_dir/ --policy policy.pt

  # With sampling for exploration
  torchref.refine-policy -m model.pdb -sf reflections.mtz -o output/ --policy policy.pt --sample

  # With temperature for controlled exploration
  torchref.refine-policy -m model.pdb -sf reflections.mtz -o output/ --policy policy.pt --sample --temperature 0.5
        """,
    )

    add_single_model_args(parser)

    output = parser.add_argument_group("Output")
    add_outdir_arg(output)
    add_output_format_args(output)
    add_metadata_args(output)

    refine = parser.add_argument_group("Refinement")
    refine.add_argument(
        "--policy",
        required=False,
        default="default",
        type=str,
        help="Path to trained policy checkpoint (.pt file)",
    )
    add_n_cycles_arg(refine)
    refine.add_argument(
        "--sample",
        action="store_true",
        default=False,
        help="Sample from policy distribution instead of using mean predictions",
    )
    refine.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (only used with --sample). Higher = more exploration (default: 1.0)",
    )
    refine.add_argument(
        "--record-trajectory",
        action="store_true",
        default=False,
        help="Record state-action-reward trajectory for analysis or further training",
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
    if args.policy.lower() == "default":
        from torchref import PATH_TORCHREF_DATA

        policy_path = Path(PATH_TORCHREF_DATA) / "policy_latest.pt"
    else:
        policy_path = Path(args.policy)

    validate_files(
        [
            (str(model_path), "Model"),
            (str(sf_path), "Structure factors"),
            (str(policy_path), "Policy checkpoint"),
        ],
        exit_on_error=True,
    )

    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)

    # Import here to avoid slow startup for --help
    try:
        from torchref.refinement.lbfgs_refinement import LBFGSRefinement
        from torchref.refinement.weighting.component_weighting import ComponentWeighting
        from torchref.refinement.weighting.policy_weighting import (
            COMPONENTS,
            PolicyComponentWeighting,
            trajectory_to_dict,
        )
    except ImportError as e:
        print(f"Error: Failed to import torchref modules: {e}", file=sys.stderr)
        print("Please ensure torchref is properly installed.", file=sys.stderr)
        sys.exit(1)

    # Print header
    if args.verbose > 0:
        print("=" * 80)
        print("TorchRef LBFGS Refinement - POLICY WEIGHTING")
        print("=" * 80)
        print("Weighting scheme: PolicyComponentWeighting")
        print(f"Policy checkpoint: {policy_path}")
        print(
            f"Mode:             {'Sampling' if args.sample else 'Evaluation (deterministic)'}"
        )
        if args.sample:
            print(f"Temperature:      {args.temperature}")
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

    start_time = time.time()

    # Build column_names for MTZ loading
    column_names = build_column_names(args.column_structure_factor, args.column_sigma)

    # Initialize refinement first
    refinement = LBFGSRefinement(
        data_file=str(sf_path),
        pdb=str(model_path),
        cif=args.cif,
        verbose=args.verbose,
        max_res=args.dmin,
        device=device,
        column_names=column_names,
    )

    # Create PolicyComponentWeighting
    policy_weighting = PolicyComponentWeighting(
        policy_path=str(policy_path),
        sample=args.sample,
        temperature=args.temperature,
    )

    refinement.component_weighting = policy_weighting

    if args.verbose > 0:
        print("Refinement initialized successfully.")
        print("Using PolicyComponentWeighting with trained policy")
        print()
        sys.stdout.flush()

    # Start trajectory recording if requested
    if args.record_trajectory:
        pdb_id = model_path.stem
        policy_weighting.start_recording(
            pdb_id=pdb_id,
            structure_path=str(model_path),
            sf_path=str(sf_path),
            policy_version=str(policy_path),
        )

    # Record initial R-factors
    with torch.no_grad():
        initial_rwork, initial_rfree = refinement.get_rfactor()

    # Run refinement
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
        if args.record_trajectory and policy_weighting.trajectory:
            policy_weighting.trajectory.success = False
            policy_weighting.trajectory.error_message = str(e)
        raise e

    total_time = time.time() - start_time

    # Stop trajectory recording and get data
    trajectory_data = None
    if args.record_trajectory:
        trajectory = policy_weighting.stop_recording()
        if trajectory:
            trajectory.total_time = total_time
            trajectory_data = trajectory_to_dict(trajectory)

    # Record final R-factors
    with torch.no_grad():
        final_rwork, final_rfree = refinement.get_rfactor()

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

    # Prepare history data with policy weighting specific information
    history_data = {
        "weighting_scheme": "policy",
        "input_files": {
            "model": str(model_path),
            "structure_factor": str(sf_path),
            "cif": args.cif,
            "policy_checkpoint": str(policy_path),
        },
        "parameters": {
            "n_cycles": args.n_cycles,
            "dmin": args.dmin,
            "device": str(device),
            "sample_mode": args.sample,
            "temperature": args.temperature,
        },
        "trajectory_summary": {
            "initial_rwork": float(initial_rwork),
            "initial_rfree": float(initial_rfree),
            "final_rwork": float(final_rwork),
            "final_rfree": float(final_rfree),
            "delta_rfree": float(final_rfree - initial_rfree),
            "total_time": total_time,
        },
        "final_policy_weights": (
            policy_weighting.last_weights if policy_weighting.last_weights else {}
        ),
        "history": refinement.history if hasattr(refinement, "history") else {},
        "final_statistics": {},
    }

    # Add full trajectory data if recorded
    if trajectory_data:
        history_data["trajectory"] = trajectory_data

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

        print("\nTrajectory summary:")
        print(f"  Initial R-work: {initial_rwork:.4f}")
        print(f"  Initial R-free: {initial_rfree:.4f}")
        print(f"  Final R-work:   {final_rwork:.4f}")
        print(f"  Final R-free:   {final_rfree:.4f}")
        print(f"  Delta R-free:   {final_rfree - initial_rfree:+.4f}")
        print(f"  Total time:     {total_time:.1f}s")

        if "final_statistics" in history_data and history_data["final_statistics"]:
            stats = history_data["final_statistics"]
            print("\nFinal statistics:")
            print(
                f"  R-work:   {stats['R_work']:.4f} ({stats['n_reflections_work']} reflections)"
            )
            print(
                f"  R-free:   {stats['R_free']:.4f} ({stats['n_reflections_test']} reflections)"
            )
            print(f"  NLL work: {stats['NLL_work']:.2f}")
            print(f"  NLL test: {stats['NLL_test']:.2f}")

        if policy_weighting.last_weights:
            print("\nFinal policy-predicted weights:")
            for comp in COMPONENTS:
                if comp in policy_weighting.last_weights:
                    weight = policy_weighting.last_weights[comp]
                    log_w = policy_weighting.last_log_weights.get(comp, 0.0)
                    print(f"  {comp:12s}: log_w={log_w:+.3f}  weight={weight:.4f}")

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
