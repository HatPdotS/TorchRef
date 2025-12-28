#!/das/work/p17/p17490/CONDA/torchref/bin/python -u

"""

Command-line script for LBFGS crystallographic refinement using torchref.

This script provides a simple interface to run structure refinement with
reflection data, producing refined coordinates, structure factors, and 
refinement statistics.

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
            # Try to convert numpy arrays and similar
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
        except ImportError:
            pass
        return obj


def main():
    parser = argparse.ArgumentParser(
        description='Run LBFGS crystallographic refinement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic refinement
  torchref-refine -s model.pdb -f reflections.mtz -o output_dir/
  
  # With 10 refinement cycles
  torchref-refine -s model.pdb -f reflections.mtz -o output/ -n 10
  
  # Using CIF files
  torchref-refine -s model.cif -f reflections.cif -o output/
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
        '--weights',
        type=str,
        default=None,
        help='Target weights for loss components as JSON string (default: {"xray": 1.0, "restraints": 5.0, "adp": 5.0}). Example: \'{"xray": 1.0, "restraints": 5.0, "adp": 0.3}\''
        )

    parser.add_argument(
        '-v', '--verbose',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Verbosity level: 0=quiet, 1=normal, 2=detailed (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Parse weights argument
    if args.weights is None:
        # Use default weights
        weights = {
            'xray': 1.0,
            'restraints': 5.0,
            'adp': 5.0
        }
    else:
        try:
            weights = json.loads(args.weights)
            if not isinstance(weights, dict):
                print("Error: --weights must be a JSON dictionary", file=sys.stderr)
                sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON for --weights: {e}", file=sys.stderr)
            sys.exit(1)
    
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
        from torchref.refinement.loss_weighting import ResolutionDependentWeighting, HybridGradNormMLWeighting
    except ImportError as e:
        print(f"Error: Failed to import torchref modules: {e}", file=sys.stderr)
        print("Please ensure torchref is properly installed.", file=sys.stderr)
        sys.exit(1)
    
    # Print header
    if args.verbose > 0:
        print("=" * 80)
        print("TorchRef LBFGS Refinement")
        print("=" * 80)
        print(f"Structure:        {structure_path}")
        print(f"Structure factors: {sf_path}")
        print(f"Output directory: {outdir}")
        print(f"Refinement cycles: {args.n_cycles}")
        print(f"Device:           {args.device}")
        if args.max_res:
            print(f"Resolution cutoff: {args.max_res:.2f} Å")
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
    
    # Create weighting module

    
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

    if args.verbose > 0:
        print("Setting up Hybrid weighting...")
        sys.stdout.flush()
    
    # weighter = HybridGradNormMLWeighting(
    #     target_weights=weights, refinement=refinement, verbose=args.verbose
    # )

    # refinement.weighter = weighter


        

    
    # Run refinement
    try:
        if args.verbose > 0:
            print(f"Starting refinement with {args.n_cycles} macro cycles...\n")
            sys.stdout.flush()
        
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
    
    # Save refined structure
    output_pdb = outdir / "refined.pdb"
    refinement.model.write_pdb(str(output_pdb))
    if args.verbose > 0:
        print(f"  Refined structure: {output_pdb}")
        sys.stdout.flush()
        
# Save refined structure factors (if available)
    output_mtz = outdir / "refined.mtz"
    # Get calculated structure factors
    hkl, fobs, sigma, rfree = refinement.reflection_data()
    fcalc = refinement.get_F_calc_scaled(hkl, recalc=True)
    
    
    # Write MTZ
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
                "cif_restraints": args.cif_restraints
            },
            "parameters": {
                "n_cycles": args.n_cycles,
                "max_resolution": args.max_res,
                "device": str(device)
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
        
        # Write JSON - convert any tensors to serializable types
        with open(output_json, 'w') as f:
            json.dump(convert_to_serializable(history_data), f, indent=2)
        
        if args.verbose > 0:
            print(f"  Refinement history: {output_json}")
            sys.stdout.flush()
    
        sys.exit(1)
    
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
