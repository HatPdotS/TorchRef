#!/das/work/p17/p17490/CONDA/torchref/bin/python -u

"""
Command-line script for LBFGS crystallographic refinement with Phenix-style
weight screening.

This script implements the Phenix approach to weight optimization:
- Screen multiple weights on a log scale
- Run complete LBFGS optimization for each weight
- Select best weight based on Rfree while respecting gap constraints

This is fundamentally different from GradNorm-based adaptive weighting:
- GradNorm: Adjusts weights dynamically during optimization
- Screening: Runs multiple complete optimizations with fixed weights, selects best

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


def main():
    parser = argparse.ArgumentParser(
        description="Run LBFGS crystallographic refinement with Phenix-style weight screening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic refinement with weight screening
  torchref-refine-screened -s model.pdb -f reflections.mtz -o output_dir/
  
  # With 10 refinement cycles and custom weight range
  torchref-refine-screened -s model.pdb -f reflections.mtz -o output/ -n 10 \\
      --xyz-min-weight 0.5 --xyz-max-weight 50.0
  
  # With more weight samples for finer optimization
  torchref-refine-screened -s model.pdb -f reflections.mtz -o output/ \\
      --n-xyz-weights 30 --n-adp-weights 30
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

    # Refinement cycle arguments
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

    # Weight screening parameters for XYZ
    parser.add_argument(
        "--n-xyz-weights",
        type=int,
        default=20,
        help="Number of XYZ weights to screen (default: 20)",
    )

    parser.add_argument(
        "--xyz-min-weight",
        type=float,
        default=1,
        help="Minimum XYZ restraint weight (default: 0.1)",
    )

    parser.add_argument(
        "--xyz-max-weight",
        type=float,
        default=100.0,
        help="Maximum XYZ restraint weight (default: 100.0)",
    )

    # Weight screening parameters for ADP
    parser.add_argument(
        "--n-adp-weights",
        type=int,
        default=20,
        help="Number of ADP weights to screen (default: 20)",
    )

    parser.add_argument(
        "--adp-min-weight",
        type=float,
        default=1,
        help="Minimum ADP restraint weight (default: 0.01)",
    )

    parser.add_argument(
        "--adp-max-weight",
        type=float,
        default=100.0,
        help="Maximum ADP restraint weight (default: 10.0)",
    )

    # Constraint parameters
    parser.add_argument(
        "--max-gap",
        type=float,
        default=0.06,
        help="Maximum allowed Rfree-Rwork gap (default: 0.06 = 6%%)",
    )

    parser.add_argument(
        "--max-bi-bj",
        type=float,
        default=10.0,
        help="Maximum allowed mean |Bi-Bj| for bonded atoms (default: 10.0 Å²)",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="Maximum LBFGS iterations per weight trial (default: 20)",
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
        print("TorchRef LBFGS Refinement with Phenix-Style Weight Screening")
        print("=" * 80)
        print(f"Structure:         {structure_path}")
        print(f"Structure factors: {sf_path}")
        print(f"Output directory:  {outdir}")
        print(f"Refinement cycles: {args.n_cycles}")
        print(f"Device:            {args.device}")
        if args.max_res:
            print(f"Resolution cutoff: {args.max_res:.2f} Å")
        print()
        print("Weight screening parameters:")
        print(
            f"  XYZ weights: {args.n_xyz_weights} samples from {args.xyz_min_weight} to {args.xyz_max_weight}"
        )
        print(
            f"  ADP weights: {args.n_adp_weights} samples from {args.adp_min_weight} to {args.adp_max_weight}"
        )
        print(f"  Max Rfree-Rwork gap: {args.max_gap*100:.1f}%")
        print(f"  Max <Bi-Bj>: {args.max_bi_bj:.1f} Å²")
        print(f"  LBFGS iterations per trial: {args.max_iter}")
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

    # Initialize refinement (no weighter needed for screened approach)
    refinement = LBFGSRefinement(
        data_file=str(sf_path),
        pdb=str(structure_path),
        cif=args.cif_restraints,
        verbose=args.verbose,
        max_res=args.max_res,
        device=device,
        weighter=None,  # No weighter - using screening instead
    )

    if args.verbose > 0:
        print("Refinement initialized successfully.\n")
        sys.stdout.flush()

    # Run refinement with weight screening
    try:
        if args.verbose > 0:
            print(f"Starting refinement with {args.n_cycles} macro cycles...\n")
            sys.stdout.flush()

        refinement.refine_screened(
            macro_cycles=args.n_cycles,
            n_xyz_weights=args.n_xyz_weights,
            n_adp_weights=args.n_adp_weights,
            xyz_min_weight=args.xyz_min_weight,
            xyz_max_weight=args.xyz_max_weight,
            adp_min_weight=args.adp_min_weight,
            adp_max_weight=args.adp_max_weight,
            max_gap=args.max_gap,
            max_bi_bj=args.max_bi_bj,
            max_iter=args.max_iter,
        )

        # Final scaling
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
    refinement.write_out_mtz(str(output_mtz))

    if args.verbose > 0:
        print(f"  Refined structure factors: {output_mtz}")
        sys.stdout.flush()

    # Save refinement history as JSON
    output_json = outdir / "refinement_history.json"

    # Prepare history data
    history_data = {
        "input_files": {
            "structure": str(structure_path),
            "structure_factors": str(sf_path),
            "cif_restraints": args.cif_restraints,
        },
        "parameters": {
            "n_cycles": args.n_cycles,
            "max_resolution": args.max_res,
            "device": str(device),
            "weight_screening": {
                "n_xyz_weights": args.n_xyz_weights,
                "xyz_min_weight": args.xyz_min_weight,
                "xyz_max_weight": args.xyz_max_weight,
                "n_adp_weights": args.n_adp_weights,
                "adp_min_weight": args.adp_min_weight,
                "adp_max_weight": args.adp_max_weight,
                "max_gap": args.max_gap,
                "max_bi_bj": args.max_bi_bj,
                "max_iter": args.max_iter,
            },
        },
        "history": refinement.history if hasattr(refinement, "history") else {},
        "final_statistics": {},
    }

    # Add final R-factors
    try:
        rwork, rfree = refinement.get_rfactor()
        mean_b = refinement.model.b().mean().item()

        history_data["final_statistics"] = {
            "R_work": float(rwork),
            "R_free": float(rfree),
            "R_gap": float(rfree - rwork),
            "mean_B": float(mean_b),
        }

        # Add restraint statistics if available
        try:
            bond_devs, _ = refinement.restraints.bond_deviations()
            history_data["final_statistics"]["rmsd_bonds"] = torch.sqrt(
                (bond_devs**2).mean()
            ).item()
            angle_devs, _ = refinement.restraints.angle_deviations()
            history_data["final_statistics"]["rmsd_angles"] = torch.sqrt(
                (angle_devs**2).mean()
            ).item()
        except:
            pass

    except Exception as e:
        if args.verbose > 1:
            print(f"  Warning: Could not compute final statistics: {e}")

    # Write JSON
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
            print(f"R-work:     {stats['R_work']*100:.2f}%")
            print(f"R-free:     {stats['R_free']*100:.2f}%")
            print(f"R-gap:      {stats['R_gap']*100:.2f}%")
            print(f"Mean B:     {stats['mean_B']:.1f} Å²")
            if "rmsd_bonds" in stats:
                print(f"RMSD bonds: {stats['rmsd_bonds']:.4f} Å")
            if "rmsd_angles" in stats:
                print(f"RMSD angles: {stats['rmsd_angles']:.2f}°")

        print("=" * 80)
        print("\nOutput files:")
        print(f"  - {output_pdb}")
        print(f"  - {output_mtz}")
        print(f"  - {output_json}")
        print("\nRefinement completed successfully!")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
