#!/das/work/p17/p17490/CONDA/torchref/bin/python -u

"""
Command-line script for LBFGS refinement with POLICY-BASED weighting.

This script uses PolicyComponentWeighting which predicts component weights
from the current refinement state using a trained neural network policy.

The policy network was trained on trajectories collected with random weighting
using Advantage-Weighted Regression (AWR) to learn optimal weight selection.

Two modes:
- Evaluation mode (default): Use mean policy predictions for deterministic refinement
- Sampling mode: Sample from policy distribution for exploration/data collection
"""

import argparse
import json
import sys
import os
from pathlib import Path
import torch
import time

# Force unbuffered output for batch systems like SLURM
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

# Import stats module early to patch json with StatEntry encoder
import torchref.utils.stats  # noqa: F401


def main():
    parser = argparse.ArgumentParser(
        description='Run LBFGS refinement with policy-based weighting (trained neural network)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic refinement with trained policy
  torchref-refine-policy -s model.pdb -f reflections.mtz -o output_dir/ --policy policy.pt

  # With sampling for exploration
  torchref-refine-policy -s model.pdb -f reflections.mtz -o output/ --policy policy.pt --sample

  # With temperature for controlled exploration
  torchref-refine-policy -s model.pdb -f reflections.mtz -o output/ --policy policy.pt --sample --temperature 0.5
        """
    )

    # Mandatory arguments
    parser.add_argument(
        '-s', '--structure',
        required=True,
        type=str,
        help='Input structure file (PDB or CIF format)'
    )

    parser.add_argument(
        '-f', '--structure-factors',
        required=True,
        type=str,
        help='Input structure factors file (MTZ or CIF format)'
    )

    parser.add_argument(
        '-o', '--outdir',
        required=True,
        type=str,
        help='Output directory for refined structure and results'
    )


    parser.add_argument(
        '--policy',
        required=True,
        type=str,
        help='Path to trained policy checkpoint (.pt file)'
    )

    # Optional arguments
    parser.add_argument(
        '-n', '--n-cycles',
        type=int,
        default=5,
        help='Number of refinement macro cycles (default: 5)'
    )

    parser.add_argument(
        '-c', '--cif-restraints',
        type=str,
        default=None,
        help='CIF restraints dictionary (auto-detected if not provided)'
    )

    parser.add_argument(
        '--max-res',
        type=float,
        default=None,
        help='Maximum resolution cutoff in Angstroms (optional)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Computation device (default: cpu)'
    )

    parser.add_argument(
        '--sample',
        action='store_true',
        default=False,
        help='Sample from policy distribution instead of using mean predictions'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature (only used with --sample). Higher = more exploration (default: 1.0)'
    )

    parser.add_argument(
        '--record-trajectory',
        action='store_true',
        default=False,
        help='Record state-action-reward trajectory for analysis or further training'
    )

    parser.add_argument(
        '-v', '--verbose',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Verbosity level: 0=quiet, 1=normal, 2=detailed (default: 1)'
    )

    args = parser.parse_args()

    # Validate inputs
    structure_path = Path(args.structure)
    sf_path = Path(args.structure_factors)
    outdir = Path(args.outdir)
    policy_path = Path(args.policy)

    if not structure_path.exists():
        print(f"Error: Structure file not found: {structure_path}", file=sys.stderr)
        sys.exit(1)

    if not sf_path.exists():
        print(f"Error: Structure factors file not found: {sf_path}", file=sys.stderr)
        sys.exit(1)

    if not policy_path.exists():
        print(f"Error: Policy checkpoint not found: {policy_path}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)

    # Import here to avoid slow startup for --help
    try:
        from torchref.refinement.lbfgs_refinement import LBFGSRefinement
        from torchref.refinement.weighting.policy_weighting import (
            PolicyComponentWeighting,
            trajectory_to_dict,
            COMPONENTS,
        )
        from torchref.refinement.weighting.component_weighting import ComponentWeighting
    except ImportError as e:
        print(f"Error: Failed to import torchref modules: {e}", file=sys.stderr)
        print("Please ensure torchref is properly installed.", file=sys.stderr)
        sys.exit(1)

    # Print header
    if args.verbose > 0:
        print("=" * 80)
        print("TorchRef LBFGS Refinement - POLICY WEIGHTING")
        print("=" * 80)
        print(f"Weighting scheme: PolicyComponentWeighting")
        print(f"Policy checkpoint: {policy_path}")
        print(f"Mode:             {'Sampling' if args.sample else 'Evaluation (deterministic)'}")
        if args.sample:
            print(f"Temperature:      {args.temperature}")
        print(f"Structure:        {structure_path}")
        print(f"Structure factors: {sf_path}")
        print(f"Output directory: {outdir}")
        print(f"Refinement cycles: {args.n_cycles}")
        print(f"Device:           {args.device}")
        if args.max_res:
            print(f"Resolution cutoff: {args.max_res:.2f} A")
        print("=" * 80)
        print()
        sys.stdout.flush()

    # Setup device
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU", file=sys.stderr)
        device = torch.device('cpu')

    if args.verbose > 0:
        print("Initializing refinement...")
        sys.stdout.flush()

    start_time = time.time()

    # Initialize refinement first
    refinement = LBFGSRefinement(
        data_file=str(sf_path),
        pdb=str(structure_path),
        cif=args.cif_restraints,
        verbose=args.verbose,
        max_res=args.max_res,
        device=device,
    )

    # Create PolicyComponentWeighting
    policy_weighting = PolicyComponentWeighting(
        refinement,
        policy_path=str(policy_path),
        sample=args.sample,
        temperature=args.temperature,
    )

    # Create a ComponentWeighting that uses the policy scheme
    # We wrap it in ComponentWeighting to integrate with the existing pipeline
    component_weighting = ComponentWeighting(
        refinement,
        schemes=[policy_weighting],
    )

    # Replace the default component_weighting with our policy-based version
    refinement.component_weighting = component_weighting

    if args.verbose > 0:
        print("Refinement initialized successfully.")
        print(f"Using PolicyComponentWeighting with trained policy")
        print()
        sys.stdout.flush()

    # Start trajectory recording if requested
    if args.record_trajectory:
        pdb_id = structure_path.stem
        policy_weighting.start_recording(
            pdb_id=pdb_id,
            structure_path=str(structure_path),
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

    # Save refined structure
    output_pdb = outdir / "refined.pdb"
    refinement.model.write_pdb(str(output_pdb))
    if args.verbose > 0:
        print(f"  Refined structure: {output_pdb}")
        sys.stdout.flush()

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
            "structure": str(structure_path),
            "structure_factors": str(sf_path),
            "cif_restraints": args.cif_restraints,
            "policy_checkpoint": str(policy_path),
        },
        "parameters": {
            "n_cycles": args.n_cycles,
            "max_resolution": args.max_res,
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
        "final_policy_weights": policy_weighting.last_weights if policy_weighting.last_weights else {},
        "history": refinement.history if hasattr(refinement, 'history') else {},
        "final_statistics": {}
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

        r_work = torch.sum(torch.abs(fobs[work_mask] - fcalc[work_mask])) / torch.sum(fobs[work_mask])
        r_free = torch.sum(torch.abs(fobs[test_mask] - fcalc[test_mask])) / torch.sum(fobs[test_mask])

        history_data["final_statistics"] = {
            "R_work": float(r_work.item()),
            "R_free": float(r_free.item()),
            "NLL_work": float(work_nll.item()),
            "NLL_test": float(test_nll.item()),
            "n_reflections_work": int(work_mask.sum().item()),
            "n_reflections_test": int(test_mask.sum().item())
        }
    except Exception as e:
        if args.verbose > 1:
            print(f"  Warning: Could not compute final statistics: {e}")

    with open(output_json, 'w') as f:
        json.dump(history_data, f, indent=2)

    if args.verbose > 0:
        print(f"  Refinement history: {output_json}")
        sys.stdout.flush()

    # Print final summary
    if args.verbose > 0:
        print("\n" + "=" * 80)
        print("Refinement Summary")
        print("=" * 80)

        print(f"\nTrajectory summary:")
        print(f"  Initial R-work: {initial_rwork:.4f}")
        print(f"  Initial R-free: {initial_rfree:.4f}")
        print(f"  Final R-work:   {final_rwork:.4f}")
        print(f"  Final R-free:   {final_rfree:.4f}")
        print(f"  Delta R-free:   {final_rfree - initial_rfree:+.4f}")
        print(f"  Total time:     {total_time:.1f}s")

        if "final_statistics" in history_data and history_data["final_statistics"]:
            stats = history_data["final_statistics"]
            print(f"\nFinal statistics:")
            print(f"  R-work:   {stats['R_work']:.4f} ({stats['n_reflections_work']} reflections)")
            print(f"  R-free:   {stats['R_free']:.4f} ({stats['n_reflections_test']} reflections)")
            print(f"  NLL work: {stats['NLL_work']:.2f}")
            print(f"  NLL test: {stats['NLL_test']:.2f}")

        if policy_weighting.last_weights:
            print(f"\nFinal policy-predicted weights:")
            for comp in COMPONENTS:
                if comp in policy_weighting.last_weights:
                    weight = policy_weighting.last_weights[comp]
                    log_w = policy_weighting.last_log_weights.get(comp, 0.0)
                    print(f"  {comp:12s}: log_w={log_w:+.3f}  weight={weight:.4f}")

        print("=" * 80)
        print("\nOutput files:")
        print(f"  - {output_pdb}")
        if (outdir / "refined.mtz").exists():
            print(f"  - {outdir / 'refined.mtz'}")
        print(f"  - {output_json}")
        print("\nRefinement completed successfully!")
        sys.stdout.flush()

    return 0


if __name__ == '__main__':
    sys.exit(main())
