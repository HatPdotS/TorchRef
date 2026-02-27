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
        --dark-pdb dark.pdb --light-pdb light.pdb \\
        --dark-mtz dark.mtz --light-mtz light.mtz \\
        --fractions 0.63,0.37 -o results.mtz

    # With resolution cutoff and restraints
    torchref.phased-difference-map \\
        --dark-pdb dark.pdb --light-pdb light.pdb \\
        --dark-mtz dark.mtz --light-mtz light.mtz \\
        --fractions 0.63,0.37 --max-res 1.7 \\
        --restraints-cif ligand.cif -o results.mtz
"""

import argparse
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
        description="Compute phased difference and extrapolated map "
                    "coefficients (no refinement).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  torchref.phased-difference-map \\
      --dark-pdb dark.pdb --light-pdb light.pdb \\
      --dark-mtz dark.mtz --light-mtz light.mtz \\
      --fractions 0.63,0.37 -o results.mtz

  torchref.phased-difference-map \\
      --dark-pdb dark.pdb --light-pdb light.pdb \\
      --dark-mtz dark.mtz --light-mtz light.mtz \\
      --fractions 0.63,0.37 --max-res 1.7 -o results.mtz
        """,
    )

    # --- Required arguments ---
    parser.add_argument(
        "--dark-pdb",
        required=True,
        type=str,
        help="PDB file for the dark / reference state",
    )
    parser.add_argument(
        "--light-pdb",
        required=True,
        type=str,
        help="PDB file for the light / triggered state",
    )
    parser.add_argument(
        "--dark-mtz",
        required=True,
        type=str,
        help="MTZ file with dark / reference reflection data",
    )
    parser.add_argument(
        "--light-mtz",
        required=True,
        type=str,
        help="MTZ file with light / triggered reflection data",
    )
    parser.add_argument(
        "--fractions",
        required=True,
        type=str,
        help="Comma-separated dark,light occupancy fractions "
             "(e.g. '0.63,0.37')",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=str,
        help="Output MTZ file path (e.g. results.mtz)",
    )

    # --- Optional arguments ---
    parser.add_argument(
        "--restraints-cif",
        type=str,
        nargs="+",
        default=None,
        help="One or more CIF restraint dictionaries",
    )
    parser.add_argument(
        "--max-res",
        type=float,
        default=None,
        help="High-resolution cutoff in Angstroms (uses data limit if not set)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device (default: auto, uses CUDA if available)",
    )
    parser.add_argument(
        "-v", "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level: 0=quiet, 1=normal, 2=detailed (default: 1)",
    )

    args = parser.parse_args()

    # --- Parse fractions ---
    try:
        fractions = [float(x) for x in args.fractions.split(",")]
        if len(fractions) != 2:
            raise ValueError
    except ValueError:
        print(
            "Error: --fractions must be two comma-separated floats, "
            "e.g. '0.63,0.37'",
            file=sys.stderr,
        )
        return 1

    # --- Validate input files ---
    for label, path in [
        ("dark PDB", args.dark_pdb),
        ("light PDB", args.light_pdb),
        ("dark MTZ", args.dark_mtz),
        ("light MTZ", args.light_mtz),
    ]:
        if not Path(path).exists():
            print(f"Error: {label} file not found: {path}", file=sys.stderr)
            return 1

    if args.restraints_cif:
        for cif_path in args.restraints_cif:
            if not Path(cif_path).exists():
                print(
                    f"Error: Restraints CIF not found: {cif_path}",
                    file=sys.stderr,
                )
                return 1

    # Ensure output directory exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if args.device == "cuda" and not torch.cuda.is_available():
            print(
                "Warning: CUDA requested but not available, falling back to CPU",
                file=sys.stderr,
            )
            device = torch.device("cpu")

    # --- Header ---
    if args.verbose > 0:
        print("=" * 72)
        print("TorchRef Phased Difference Map")
        print("=" * 72)
        print(f"Dark PDB:          {args.dark_pdb}")
        print(f"Light PDB:         {args.light_pdb}")
        print(f"Dark MTZ:          {args.dark_mtz}")
        print(f"Light MTZ:         {args.light_mtz}")
        print(f"Fractions:         dark={fractions[0]}, light={fractions[1]}")
        print(f"Output:            {args.output}")
        print(f"Device:            {device}")
        if args.max_res:
            print(f"Resolution cutoff: {args.max_res:.2f} A")
        if args.restraints_cif:
            print(f"Restraints CIF:    {', '.join(args.restraints_cif)}")
        print("=" * 72)
        print()
        sys.stdout.flush()

    # --- Import pipeline helpers from difference_refine ---
    from torchref.cli.difference_refine import (
        setup_mixed_model,
        setup_data,
        setup_collection,
        setup_scaler,
        write_results_mtz,
    )

    # --- Resolution ---
    d_min = args.max_res if args.max_res is not None else 1.0

    # --- Setup models ---
    if args.verbose > 0:
        print("Setting up models...")
        sys.stdout.flush()

    mixed, dark, light = setup_mixed_model(
        args.dark_pdb, args.light_pdb, fractions,
        args.restraints_cif, d_min, device, args.verbose,
    )
    mixed.freeze_fractions()

    # --- Load data ---
    if args.verbose > 0:
        print("Loading reflection data...")
        sys.stdout.flush()

    data_dark, data_light = setup_data(
        args.dark_mtz, args.light_mtz, args.max_res, device,
    )

    # --- Scale ---
    if args.verbose > 0:
        print("Scaling datasets...")
        sys.stdout.flush()

    collection = setup_collection(data_dark, data_light, device)

    if args.verbose > 0:
        print("Setting up scalers...")
        sys.stdout.flush()

    scaler_dark = setup_scaler(dark, data_dark, device)
    scaler_mixed = setup_scaler(mixed, data_light, device)

    if args.verbose > 0:
        r_work_d, r_free_d = scaler_dark.rfactor()
        r_work_l, r_free_l = scaler_mixed.rfactor()
        print(f"  R-factor (dark):  R_work={r_work_d:.4f}  R_free={r_free_d:.4f}")
        print(f"  R-factor (mixed): R_work={r_work_l:.4f}  R_free={r_free_l:.4f}")
        print()
        sys.stdout.flush()

    # --- Write MTZ ---
    if args.verbose > 0:
        print("Computing map coefficients...")
        sys.stdout.flush()

    with torch.no_grad():
        write_results_mtz(
            data_dark, data_light, mixed, dark, light,
            scaler_dark, scaler_mixed, str(out_path),
        )

    if args.verbose > 0:
        print()
        print("Done.")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
