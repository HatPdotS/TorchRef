#!/das/work/p17/p17490/CONDA/torchref/bin/python -u

"""
Command-line script for LBFGS refinement with HYPERPARAMETER-TUNED weighting.

This script uses ComponentWeighting (XrayScaleWeighting + TargetOffsetWeighting +
OverfittingWeighting) with Optuna-optimized hyperparameters loaded from file.

The hyperparameters were optimized to achieve the best R-free across a diverse
set of protein structures at various resolutions.

Options:
- "default": Load optimized hyperparameters from package data
- Custom path: Load from user-specified JSON file
- "none": Skip hyperparameter loading (same as static weighting)
"""

import argparse
import json
import sys
import os
from pathlib import Path
import torch

# Force unbuffered output for batch systems like SLURM
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

# Import stats module early to patch json with StatEntry encoder
import torchref.utils.stats  # noqa: F401


def main():
    parser = argparse.ArgumentParser(
        description='Run LBFGS refinement with Optuna-optimized hyperparameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Refinement with default optimized hyperparameters
  torchref-refine-hyper -s model.pdb -f reflections.mtz -o output_dir/

  # With custom hyperparameters file
  torchref-refine-hyper -s model.pdb -f reflections.mtz -o output/ --hyperparameters my_params.json

  # Skip hyperparameters (equivalent to static weighting)
  torchref-refine-hyper -s model.pdb -f reflections.mtz -o output/ --hyperparameters none
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
        '--hyperparameters',
        type=str,
        default='default',
        help='Path to hyperparameters JSON file, or "default" to use optimized defaults, '
             'or "none" to skip. The JSON file can be edited to customize refinement behavior. '
             '(default: "default" uses Optuna-optimized hyperparameters)'
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
        print("TorchRef LBFGS Refinement - HYPERPARAMETER-TUNED")
        print("=" * 80)
        print(f"Weighting scheme: ComponentWeighting with optimized hyperparameters")
        print(f"Hyperparameters:  {args.hyperparameters}")
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

    # Initialize refinement
    refinement = LBFGSRefinement(
        data_file=str(sf_path),
        pdb=str(structure_path),
        cif=args.cif_restraints,
        verbose=args.verbose,
        max_res=args.max_res,
        device=device,
    )

    if args.verbose > 0:
        print("Refinement initialized successfully.\n")
        sys.stdout.flush()

    # Load and apply hyperparameters
    hyperparams_source = None
    n_hyperparams = 0

    if args.hyperparameters.lower() != 'none':
        try:
            from torchref.utils.utils import dict_to_state_dict
            import importlib.resources

            if args.hyperparameters.lower() == 'default':
                # Load default hyperparameters from package data
                try:
                    # Python 3.9+
                    with importlib.resources.files('torchref.data').joinpath('default_hyperparameters.json').open() as f:
                        hyperparams_raw = json.load(f)
                except (AttributeError, TypeError):
                    # Fallback for older Python
                    import pkg_resources
                    hyperparams_path = pkg_resources.resource_filename('torchref', 'data/default_hyperparameters.json')
                    with open(hyperparams_path) as f:
                        hyperparams_raw = json.load(f)

                hyperparams_source = "package default (Optuna-optimized)"
                if args.verbose > 0:
                    print("Loading optimized default hyperparameters...")
                    sys.stdout.flush()
            else:
                # Load from user-specified file
                hyperparams_path = Path(args.hyperparameters)
                if not hyperparams_path.exists():
                    print(f"Error: Hyperparameters file not found: {hyperparams_path}", file=sys.stderr)
                    sys.exit(1)

                with open(hyperparams_path) as f:
                    hyperparams_raw = json.load(f)

                hyperparams_source = str(hyperparams_path)
                if args.verbose > 0:
                    print(f"Loading hyperparameters from: {hyperparams_path}")
                    sys.stdout.flush()

            # Convert to state dict and apply
            hyperparams = dict_to_state_dict(hyperparams_raw)
            refinement.apply_hyperparameters(hyperparams, verbose=(args.verbose > 1))
            n_hyperparams = len(hyperparams)

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
        "weighting_scheme": "hyperparameters",
        "input_files": {
            "structure": str(structure_path),
            "structure_factors": str(sf_path),
            "cif_restraints": args.cif_restraints
        },
        "parameters": {
            "n_cycles": args.n_cycles,
            "max_resolution": args.max_res,
            "device": str(device),
            "hyperparameters_source": hyperparams_source,
            "n_hyperparameters": n_hyperparams,
        },
        "history": refinement.history if hasattr(refinement, 'history') else {},
        "final_statistics": {}
    }

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

        if "final_statistics" in history_data and history_data["final_statistics"]:
            stats = history_data["final_statistics"]
            print(f"R-work:  {stats['R_work']:.4f} ({stats['n_reflections_work']} reflections)")
            print(f"R-free:  {stats['R_free']:.4f} ({stats['n_reflections_test']} reflections)")
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


if __name__ == '__main__':
    sys.exit(main())
