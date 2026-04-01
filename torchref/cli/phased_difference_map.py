#!/usr/bin/env python3 -u

"""
Command-line script for computing phased difference and extrapolated
map coefficients from dark/light crystallographic data.

Uses the same pipeline as ``torchref.difference-refine`` but performs
**no refinement** — the input models are used as-is to compute phases,
scale factors, and all flavours of extrapolated structure-factor
amplitudes.  The result is a single MTZ file containing observed,
calculated, difference, and extrapolated columns.

Examples
--------
::

    # Basic usage
    torchref.phased-difference-map \\
        -dm dark.pdb -lm light.pdb \\
        -dsf dark.mtz -lsf light.mtz \\
        --fraction 0.37 -o results.mtz

    # With resolution cutoff and restraints
    torchref.phased-difference-map \\
        -dm dark.pdb -lm light.pdb \\
        -dsf dark.mtz -lsf light.mtz \\
        --fraction 0.37 --dmin 1.7 \\
        --cif ligand.cif -o results.mtz
"""

import argparse
import sys
from pathlib import Path

import torch

from torchref.cli._common import (
    add_dual_model_args,
    add_dmin_arg,
    add_general_args,
    add_output_arg,
    build_dual_column_names,
    configure_unbuffered_output,
    register_timing,
    resolve_device,
    validate_cif_files,
    validate_files,
)

configure_unbuffered_output()


def main():
    parser = argparse.ArgumentParser(
        description="Compute phased difference and extrapolated map "
                    "coefficients (no refinement).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  torchref.phased-difference-map \\
      -dm dark.pdb -lm light.pdb \\
      -dsf dark.mtz -lsf light.mtz \\
      --fraction 0.37 -o results.mtz

  torchref.phased-difference-map \\
      -dm dark.pdb -lm light.pdb \\
      -dsf dark.mtz -lsf light.mtz \\
      --fraction 0.37 --dmin 1.7 -o results.mtz
        """,
    )

    # --- Input files (creates "Input files" and "Column selection" groups) ---
    add_dual_model_args(parser)

    output = parser.add_argument_group("Output")
    add_output_arg(output, help="Output MTZ file path (e.g. results.mtz)")

    res = parser.add_argument_group("Resolution")
    add_dmin_arg(res)

    add_general_args(parser)

    args = parser.parse_args()

    register_timing()

    # --- Parse fraction ---
    fractions = [1.0 - args.fraction, args.fraction]

    # --- Validate input files ---
    if validate_files([
        (args.dark_model, "dark model"),
        (args.light_model, "light model"),
        (args.dark_structure_factor, "dark structure factor"),
        (args.light_structure_factor, "light structure factor"),
    ]):
        return 1

    if validate_cif_files(args.cif):
        return 1

    # Ensure output directory exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    device = resolve_device(args.device)

    # --- Header ---
    if args.verbose > 0:
        print("=" * 72)
        print("TorchRef Phased Difference Map")
        print("=" * 72)
        print(f"Dark model:        {args.dark_model}")
        print(f"Light model:       {args.light_model}")
        print(f"Dark SF:           {args.dark_structure_factor}")
        print(f"Light SF:          {args.light_structure_factor}")
        print(f"Fraction:          light={args.fraction} (dark={1.0 - args.fraction})")
        print(f"Output:            {args.output}")
        print(f"Device:            {device}")
        if args.dmin:
            print(f"Resolution cutoff: {args.dmin:.2f} A")
        if args.cif:
            print(f"CIF restraints:    {', '.join(args.cif)}")
        print("=" * 72)
        print()
        sys.stdout.flush()

    from torchref.cli.collection_difference_refine import (
        compute_rfactors,
        setup_model_collection,
        setup_dataset_collection,
        setup_scaler,
        write_results_mtz,
    )

    # --- Resolution ---
    d_min = args.dmin if args.dmin is not None else 1.0

    # --- Setup models ---
    if args.verbose > 0:
        print("Setting up models...")
        sys.stdout.flush()

    mc = setup_model_collection(
        args.dark_model, args.light_model, fractions,
        args.cif, d_min, device, args.verbose,
    )
    mc["light"].freeze_fractions()

    # --- Load data ---
    if args.verbose > 0:
        print("Loading reflection data...")
        sys.stdout.flush()

    col_dark, col_light = build_dual_column_names(args)

    dc = setup_dataset_collection(
        args.dark_structure_factor, args.light_structure_factor, args.dmin, device,
        column_names_dark=col_dark, column_names_light=col_light,
    )

    # --- Scale ---
    if args.verbose > 0:
        print("Setting up joint scaler...")
        sys.stdout.flush()

    scaler = setup_scaler(dc, mc, device, args.verbose)

    if args.verbose > 0:
        r_work_d, r_free_d = compute_rfactors(mc.dark_model, dc["dark"], scaler)
        r_work_l, r_free_l = compute_rfactors(mc["light"], dc["light"], scaler)
        print(f"  R-factor (dark):  R_work={r_work_d:.4f}  R_free={r_free_d:.4f}")
        print(f"  R-factor (mixed): R_work={r_work_l:.4f}  R_free={r_free_l:.4f}")
        print()
        sys.stdout.flush()

    # --- Write MTZ ---
    if args.verbose > 0:
        print("Computing map coefficients...")
        sys.stdout.flush()

    with torch.no_grad():
        write_results_mtz(dc, mc, scaler, str(out_path))

    if args.verbose > 0:
        print()
        print("Done.")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
