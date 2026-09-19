#!/usr/bin/env python3 -u

"""Difference and extrapolated map coefficients from dark/light data.

Uses the ``torchref.difference-refine`` pipeline but performs **no refinement**: the input
models are used as-is.

The default output is the difference map: the amplitude difference
``|Fo_light| - |Fo_dark|`` as ``DF``/``SIGDF`` carried on the **dark** model's phases
``PHDELWT``, with the per-reflection weights of every registered scheme beside it as
``W_IVW`` (inverse variance, the default) and ``W_SD`` (sigma_D Wiener weight), and the
observed-to-model scale ``KSCALE``. Build the map with
``torchref.mtz2map -csf DF -cw W_IVW -cphi PHDELWT`` (``--units electrons`` for e/A^3).
That needs no light-state model, so ``-lm`` is optional. It is also deliberately not a
*phased* difference map: putting the light state's model phases into the observed
amplitude biases the map toward the very model the experiment is testing.

Given ``-lm``, the light state's amplitude and phase and the extrapolated map follow.
``--all-columns`` adds the alternative constructions of both.

Examples
--------
::

    # difference map only -- no light model, no fraction needed
    torchref.difference-map \\
        -dm dark.pdb -dsf dark.mtz -lsf light.mtz -o results.mtz

    torchref.difference-map \\
        -dm dark.pdb -lm light.pdb -dsf dark.mtz -lsf light.mtz \\
        --fraction 0.37 --dmin 1.7 --cif ligand.cif -o results.mtz
"""

import argparse
import sys
from pathlib import Path

import torch

from torchref.cli._common import (
    add_all_columns_arg,
    add_ded_weight_args,
    add_dual_model_args,
    add_dmin_arg,
    add_general_args,
    add_output_arg,
    build_dual_column_names,
    configure_unbuffered_output,
    register_timing,
    parse_device_str,
    sigma_d_config_from_args,
    validate_cif_files,
    validate_files,
)

configure_unbuffered_output()


def main():
    """Entry point for ``torchref.difference-map``; returns the exit code."""
    parser = argparse.ArgumentParser(
        description="Compute difference and extrapolated map coefficients "
                    "(no refinement).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # difference map only -- needs no light model and no fraction
  torchref.difference-map \\
      -dm dark.pdb \\
      -dsf dark.mtz -lsf light.mtz -o results.mtz

  torchref.difference-map \\
      -dm dark.pdb -lm light.pdb \\
      -dsf dark.mtz -lsf light.mtz \\
      --fraction 0.37 -o results.mtz

  torchref.difference-map \\
      -dm dark.pdb -lm light.pdb \\
      -dsf dark.mtz -lsf light.mtz \\
      --fraction 0.37 --dmin 1.7 --all-columns -o results.mtz
        """,
    )

    add_dual_model_args(parser, fraction_required=False, light_model_required=False)

    output = parser.add_argument_group("Output")
    add_output_arg(output, help="Output MTZ file path (e.g. results.mtz)")
    add_all_columns_arg(output)
    add_ded_weight_args(output)

    res = parser.add_argument_group("Resolution")
    add_dmin_arg(res)

    add_general_args(parser)

    args = parser.parse_args()

    register_timing()

    has_light_model = args.light_model is not None
    if has_light_model:
        if args.fraction is None:
            parser.error(
                "--fraction is required with -lm/--light-model: the extrapolation "
                "divides by the light-state occupancy."
            )
        fractions = [1.0 - args.fraction, args.fraction]
    else:
        fractions = None
        if args.fraction is not None:
            parser.error(
                "--fraction needs -lm/--light-model. Without a light model only the "
                "weighted difference map is written, and it carries no occupancy: the "
                "amplitude is |Fo_light| - |Fo_dark| and the weight comes from the "
                "sigmas."
            )

    to_check = [
        (args.dark_model, "dark model"),
        (args.dark_structure_factor, "dark structure factor"),
        (args.light_structure_factor, "light structure factor"),
    ]
    if has_light_model:
        to_check.insert(1, (args.light_model, "light model"))
    if validate_files(to_check):
        return 1

    if validate_cif_files(args.cif):
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = parse_device_str(args.device)

    if args.verbose > 0:
        print("=" * 72)
        print("TorchRef Difference Map")
        print("=" * 72)
        print(f"Dark model:        {args.dark_model}")
        if has_light_model:
            print(f"Light model:       {args.light_model}")
            print(f"Fraction:          light={args.fraction} "
                  f"(dark={1.0 - args.fraction})")
        else:
            print("Light model:       (none) -- difference map only")
        print(f"Dark SF:           {args.dark_structure_factor}")
        print(f"Light SF:          {args.light_structure_factor}")
        print(f"Output:            {args.output}")
        print(f"Device:            {device}")
        if args.dmin:
            print(f"Resolution cutoff: {args.dmin:.2f} A")
        if args.cif:
            print(f"CIF restraints:    {', '.join(args.cif)}")
        if args.all_columns:
            print("Columns:           all")
        print("=" * 72)
        print()
        sys.stdout.flush()

    from torchref.cli.collection_difference_refine import (
        compute_rfactors,
        setup_dark_only,
        setup_model_collection,
        setup_dataset_collection,
        setup_scaler,
        write_results_mtz,
    )

    d_min = args.dmin if args.dmin is not None else 1.0

    if args.verbose > 0:
        print("Loading reflection data...")
        sys.stdout.flush()

    col_dark, col_light = build_dual_column_names(args)

    dc = setup_dataset_collection(
        args.dark_structure_factor, args.light_structure_factor, args.dmin, device,
        column_names_dark=col_dark, column_names_light=col_light,
    )

    if args.verbose > 0:
        print("Setting up models...")
        sys.stdout.flush()

    if has_light_model:
        mc = setup_model_collection(
            args.dark_model, args.light_model, fractions,
            args.cif, d_min, device, args.verbose,
        )
        mc["light"].freeze_fractions()

        if args.verbose > 0:
            print("Setting up joint scaler...")
            sys.stdout.flush()
        scaler = setup_scaler(dc, mc, device, args.verbose)
        dark_model = mc.dark_model

        if args.verbose > 0:
            r_work_d, r_free_d = compute_rfactors(dark_model, dc["dark"], scaler)
            r_work_l, r_free_l = compute_rfactors(mc["light"], dc["light"], scaler)
            print(f"  R-factor (dark):  R_work={r_work_d:.4f}  R_free={r_free_d:.4f}")
            print(f"  R-factor (mixed): R_work={r_work_l:.4f}  R_free={r_free_l:.4f}")
            print()
            sys.stdout.flush()
    else:
        mc = None
        dark_model, scaler = setup_dark_only(
            args.dark_model, dc, args.cif, d_min, device, args.verbose,
        )
        if args.verbose > 0:
            r_work_d, r_free_d = compute_rfactors(dark_model, dc["dark"], scaler)
            print(f"  R-factor (dark):  R_work={r_work_d:.4f}  R_free={r_free_d:.4f}")
            print()
            sys.stdout.flush()

    if args.verbose > 0:
        print("Computing map coefficients...")
        sys.stdout.flush()

    with torch.no_grad():
        write_results_mtz(
            dc,
            dark_model,
            scaler,
            str(out_path),
            mc=mc,
            all_columns=args.all_columns,
            verbose=args.verbose,
            ded_weight=args.ded_weight,
            sigma_d_config=sigma_d_config_from_args(args),
        )

    if args.verbose > 0:
        print()
        print("Done.")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
