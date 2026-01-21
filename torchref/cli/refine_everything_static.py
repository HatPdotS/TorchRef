#!/das/work/p17/p17490/CONDA/torchref/bin/python -u

"""
Command-line script for LBFGS refinement with STATIC weighting.

This script uses the default ComponentWeighting scheme (XrayScaleWeighting +
TargetOffsetWeighting + OverfittingWeighting) but without loading optimized
hyperparameters. The weighting scheme adapts during refinement based on
the loss values.

Use this as a baseline to compare against hyperparameter-tuned refinement.
"""

import argparse
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

# Import stats module early to patch json with StatEntry encoder
import torchref.utils.stats  # noqa: F401


def main():
    parser = argparse.ArgumentParser(
        description="Run LBFGS refinement with static/default weighting (no hyperparameter optimization)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic refinement with static weights
  torchref-refine-static -s model.pdb -f reflections.mtz -o output_dir/

  # With 10 refinement cycles
  torchref-refine-static -s model.pdb -f reflections.mtz -o output/ -n 10
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
        "-w",
        "--weights",
        type=str,
        default=None,
        help="Path to JSON file with manual weights for components (overrides defaults) defaults: {xray:1.0, geometry:10.0, adp:5.0}\n \
            Weights need to matcht the component names used in the refinement setup.",
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
    if args.device == "cuda" and not torch.cuda.is_available():
        print(
            "Warning: CUDA requested but not available, falling back to CPU",
            file=sys.stderr,
        )
        device = torch.device("cpu")

    if args.verbose > 0:
        print("Initializing refinement...")
        sys.stdout.flush()

    refinement = LBFGSRefinement(
        data_file=str(sf_path),
        pdb=str(structure_path),
        cif=args.cif_restraints,
        verbose=args.verbose,
        max_res=args.max_res,
        device=device,
    )

    from torchref.refinement.weighting import ManualWeighting

    base_weights = {"xray": 1.0, "geometry": 10.0, "adp": 5.0}
    if args.weights is not None:
        with open(args.weights, "r") as f:
            base_weights.update(json.load(f))

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

        # Prepare history data
        history_data = {
            "weighting_scheme": "static",
            "input_files": {
                "structure": str(structure_path),
                "structure_factors": str(sf_path),
                "cif_restraints": args.cif_restraints,
            },
            "parameters": {
                "n_cycles": args.n_cycles,
                "max_resolution": args.max_res,
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

            # Calculate R-factors
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
