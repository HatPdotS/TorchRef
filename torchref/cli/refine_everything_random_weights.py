#!/usr/bin/env python3 -u

"""
Command-line script for LBFGS refinement with RANDOM weighting.

This script uses ``RandomComponentWeighting`` which samples component weights
from log-normal distributions. This is used to generate diverse training
trajectories for policy network training via AWR (Advantage-Weighted Regression).

Two-level sampling strategy:

1. Trajectory-level base weights: Sampled once at initialization.
2. Step-level perturbations: Applied at each optimization step.

The sampled weights and resulting trajectories are recorded for training data.

Examples
--------
::

    torchref.refine-random-weights -s model.pdb -f reflections.mtz -o output_dir/
    torchref.refine-random-weights -s model.pdb -f reflections.mtz -o output/ --seed 42
"""

import argparse
import json
import os
import sys
import time
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

# Import stats module early to patch json with StatEntry encoder
import torchref.utils.stats  # noqa: F401


def main():
    parser = argparse.ArgumentParser(
        description="Run LBFGS refinement with random weighting for policy training data collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic refinement with random weights
  torchref.refine-random-weights -s model.pdb -f reflections.mtz -o output_dir/

  # With specific random seed for reproducibility
  torchref.refine-random-weights -s model.pdb -f reflections.mtz -o output/ --seed 42

  # With 10 refinement cycles
  torchref.refine-random-weights -s model.pdb -f reflections.mtz -o output/ -n 10
        """,
    )

    # Mandatory arguments
    parser.add_argument(
        "-s",
        "--structure",
        required=True,
        type=str,
        help="Input structure file (PDB or CIF format)",
    )

    parser.add_argument(
        "-f",
        "--structure-factors",
        required=True,
        type=str,
        help="Input structure factors file (MTZ or CIF format)",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        required=True,
        type=str,
        help="Output directory for refined structure and results",
    )

    # Optional arguments
    parser.add_argument(
        "-n",
        "--n-cycles",
        type=int,
        default=5,
        help="Number of refinement macro cycles (default: 5)",
    )

    parser.add_argument(
        "-c",
        "--cif-restraints",
        type=str,
        default=None,
        help="CIF restraints dictionary (auto-detected if not provided)",
    )

    parser.add_argument(
        "--max-res",
        type=float,
        default=None,
        help="Maximum resolution cutoff in Angstroms (optional)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computation device (default: cpu)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for weight sampling (default: random)",
    )

    parser.add_argument(
        "--trajectory-sigma",
        type=float,
        default=1.5,
        help="Standard deviation for trajectory-level base weight sampling in log-space (default: 1.5)",
    )

    parser.add_argument(
        "--step-sigma",
        type=float,
        default=0.3,
        help="Standard deviation for step-level perturbations in log-space (default: 0.3)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level: 0=quiet, 1=normal, 2=detailed (default: 1)",
    )

    args = parser.parse_args()

    # Validate inputs
    structure_path = Path(args.structure)
    sf_path = Path(args.structure_factors)
    outdir = Path(args.outdir)

    if not structure_path.exists():
        print(f"Error: Structure file not found: {structure_path}", file=sys.stderr)
        sys.exit(1)

    if not sf_path.exists():
        print(f"Error: Structure factors file not found: {sf_path}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)

    # Import here to avoid slow startup for --help
    try:
        from torchref.refinement.lbfgs_refinement import LBFGSRefinement
        from torchref.refinement.weighting.random_weighting import (
            DEFAULT_LOG_WEIGHTS,
            DEFAULT_STEP_SIGMAS,
            DEFAULT_TRAJECTORY_SIGMAS,
            RandomComponentWeighting,
        )
    except ImportError as e:
        print(f"Error: Failed to import torchref modules: {e}", file=sys.stderr)
        print("Please ensure torchref is properly installed.", file=sys.stderr)
        sys.exit(1)

    # Print header
    if args.verbose > 0:
        print("=" * 80)
        print("TorchRef LBFGS Refinement - RANDOM WEIGHTING")
        print("=" * 80)
        print("Weighting scheme: RandomComponentWeighting")
        print(f"Structure:        {structure_path}")
        print(f"Structure factors: {sf_path}")
        print(f"Output directory: {outdir}")
        print(f"Refinement cycles: {args.n_cycles}")
        print(f"Device:           {args.device}")
        print(f"Random seed:      {args.seed if args.seed is not None else 'random'}")
        print(f"Trajectory sigma: {args.trajectory_sigma}")
        print(f"Step sigma:       {args.step_sigma}")
        if args.max_res:
            print(f"Resolution cutoff: {args.max_res:.2f} A")
        print("=" * 80)
        print()
        sys.stdout.flush()

    # Setup device
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print(
            "Warning: CUDA requested but not available, falling back to CPU",
            file=sys.stderr,
        )
        device = torch.device("cpu")

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

    # Create custom sigmas if specified
    trajectory_sigmas = {
        k: args.trajectory_sigma for k in DEFAULT_TRAJECTORY_SIGMAS.keys()
    }
    step_sigmas = {k: args.step_sigma for k in DEFAULT_STEP_SIGMAS.keys()}

    # Create and install RandomComponentWeighting
    random_weighting = RandomComponentWeighting(
        refinement,
        default_log_weights=DEFAULT_LOG_WEIGHTS,
        trajectory_sigmas=trajectory_sigmas,
        step_sigmas=step_sigmas,
        seed=args.seed,
        resample_each_step=True,
    )

    # Replace the default component_weighting with our random version
    refinement.component_weighting = random_weighting

    if args.verbose > 0:
        print("Refinement initialized successfully.")
        print("Using RandomComponentWeighting (XrayScale + RandomWeighting)")
        print("\nBase log-weights (sampled once for this trajectory):")
        for name, log_w in random_weighting.get_base_log_weights().items():
            weight = torch.exp(torch.tensor(log_w)).item()
            print(f"  {name:12s}: log_w={log_w:+.3f}  weight={weight:.4f}")
        print()
        sys.stdout.flush()

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
        raise e

    total_time = time.time() - start_time

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

    # Save refinement history and trajectory data as JSON
    output_json = outdir / "refinement_history.json"

    # Prepare history data with random weighting specific information
    history_data = {
        "weighting_scheme": "random",
        "input_files": {
            "structure": str(structure_path),
            "structure_factors": str(sf_path),
            "cif_restraints": args.cif_restraints,
        },
        "parameters": {
            "n_cycles": args.n_cycles,
            "max_resolution": args.max_res,
            "device": str(device),
            "random_seed": args.seed,
            "trajectory_sigma": args.trajectory_sigma,
            "step_sigma": args.step_sigma,
        },
        "trajectory": {
            "initial_rwork": float(initial_rwork),
            "initial_rfree": float(initial_rfree),
            "final_rwork": float(final_rwork),
            "final_rfree": float(final_rfree),
            "total_time": total_time,
            "base_log_weights": random_weighting.get_base_log_weights(),
            "final_sampled_log_weights": random_weighting.get_sampled_log_weights(),
            "final_sampled_weights": random_weighting.get_sampled_weights(),
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

        print("\nFinal sampled weights:")
        for name, weight in random_weighting.get_sampled_weights().items():
            log_w = random_weighting.get_sampled_log_weights()[name]
            print(f"  {name:12s}: log_w={log_w:+.3f}  weight={weight:.4f}")

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
