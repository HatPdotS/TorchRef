#!/usr/bin/env python3 -u

"""
Command-line script for LBFGS crystallographic refinement using torchref.

Supports the Bhattacharyya overlap target by default; Gaussian / least-squares
/ maximum-likelihood targets remain available via ``--xray-mode``.

Examples
--------
::

    # Default: Bhattacharyya target, joint XYZ+ADP+scaler LBFGS
    torchref.refine -m model.pdb -sf reflections.mtz -o output_dir/

    # 10 refinement cycles
    torchref.refine -m model.pdb -sf reflections.mtz -o output/ -n 10

    # Separated XYZ then ADP cycles
    torchref.refine -m model.pdb -sf reflections.mtz -o output/ --mode refine

    # Legacy maximum-likelihood target
    torchref.refine -m model.pdb -sf reflections.mtz -o output/ --xray-mode ml
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
    build_column_names,
    configure_unbuffered_output,
    register_timing,
    resolve_device,
    validate_files,
    write_refinement_outputs,
)
from torchref.utils.serialization import convert_to_serializable

configure_unbuffered_output()

# Import stats module early to patch json with StatEntry encoder
import torchref.utils.stats  # noqa: F401,E402


def main():
    parser = argparse.ArgumentParser(
        description="Run LBFGS crystallographic refinement with torchref.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Bhattacharyya target, joint XYZ+ADP+scaler LBFGS
  torchref.refine -m model.pdb -sf reflections.mtz -o output_dir/

  # 10 refinement cycles
  torchref.refine -m model.pdb -sf reflections.mtz -o output/ -n 10

  # Separated XYZ then ADP cycles
  torchref.refine -m model.pdb -sf reflections.mtz -o output/ --mode refine

  # Legacy maximum-likelihood target
  torchref.refine -m model.pdb -sf reflections.mtz -o output/ --xray-mode ml
        """,
    )

    add_single_model_args(parser)

    output = parser.add_argument_group("Output")
    add_outdir_arg(output)
    add_output_format_args(output)
    add_metadata_args(output)

    refine_group = parser.add_argument_group("Refinement")
    add_n_cycles_arg(refine_group)
    refine_group.add_argument(
        "--mode",
        type=str,
        default="everything",
        choices=["everything", "refine"],
        help='Refinement mode: "everything" for joint XYZ+ADP+scaler LBFGS, '
        '"refine" for separated XYZ then ADP cycles (default: "everything")',
    )
    refine_group.add_argument(
        "--xray-mode",
        type=str,
        default="bhattacharyya",
        choices=["gaussian", "ls", "ml", "bhattacharyya"],
        help="X-ray target function. 'bhattacharyya' (default) uses the "
        "Bhattacharyya overlap loss with first-principles model error "
        "estimation and needs no manual weight tuning.",
    )
    refine_group.add_argument(
        "--sigma-m-scale",
        type=float,
        default=1.0,
        help="Global multiplier applied to σ_m for the Bhattacharyya target. "
        "Ignored for other targets. Default 1.0.",
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

    outdir.mkdir(parents=True, exist_ok=True)

    # Import here to avoid slow startup for --help
    try:
        from torchref.refinement.lbfgs_refinement import LBFGSRefinement
    except ImportError as e:
        print(f"Error: Failed to import torchref modules: {e}", file=sys.stderr)
        print("Please ensure torchref is properly installed.", file=sys.stderr)
        sys.exit(1)

    if args.verbose > 0:
        print("=" * 80)
        print("TorchRef LBFGS Refinement")
        print("=" * 80)
        print(f"Model:             {model_path}")
        print(f"Structure factor:  {sf_path}")
        print(f"Output directory:  {outdir}")
        print(f"Refinement mode:   {args.mode}")
        print(f"X-ray target:      {args.xray_mode}")
        if args.xray_mode == "bhattacharyya":
            print(f"sigma_m scale:     {args.sigma_m_scale}")
        print(f"Refinement cycles: {args.n_cycles}")
        print(f"Device:            {args.device}")
        if args.dmin:
            print(f"Resolution cutoff: {args.dmin:.2f} A")
        print("=" * 80)
        print()
        sys.stdout.flush()

    device = resolve_device(args.device)

    if args.verbose > 0:
        print("Initializing refinement...")
        sys.stdout.flush()

    column_names = build_column_names(args.column_structure_factor, args.column_sigma)

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
    )

    if args.verbose > 0:
        print("Refinement initialized successfully.\n")
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
    refinement.write_out_mtz(str(output_mtz))

    if args.verbose > 0:
        print(f"  Refined structure factors: {output_mtz}")
        sys.stdout.flush()

    # Save refinement history as JSON
    output_json = outdir / "refinement_history.json"
    history_data = {
        "input_files": {
            "model": str(model_path),
            "structure_factor": str(sf_path),
            "cif": args.cif,
        },
        "parameters": {
            "n_cycles": args.n_cycles,
            "mode": args.mode,
            "xray_mode": args.xray_mode,
            "sigma_m_scale": args.sigma_m_scale,
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
        json.dump(convert_to_serializable(history_data), f, indent=2)

    if args.verbose > 0:
        print(f"  Refinement history: {output_json}")
        sys.stdout.flush()

    # Print final summary
    if args.verbose > 0:
        print("\n" + "=" * 80)
        print("Refinement Summary")
        print("=" * 80)

        if history_data["final_statistics"]:
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
