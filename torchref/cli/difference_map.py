#!/usr/bin/env python3 -u

"""Difference and extrapolated map coefficients from dark/light data.

Uses the ``torchref.difference-refine`` pipeline but performs **no refinement**: the input
models are used as-is. Writes ``difference_map.mtz`` and ``summary.json`` (inputs,
dataset scaling, R-factors, weight and shrinkage fits) to the ``-o`` directory.

The default output is the difference map: ``DELFWT``/``PHDELWT``, the amplitude
difference ``dF = |Fo_light| - |Fo_dark|`` times the ``--ded-weight`` weight on the
**dark** model's phases, which Coot opens directly. ``dF``/``SIGdF``, the weights of
every registered scheme (``W_Q``, ``W_InVa``) and the observed-to-model scale ``KSCALE``
sit beside it, so another weighting is ``torchref.mtz2map -csf dF -cw W_InVa -cphi
PHDELWT`` (``--units electrons`` for e/A^3). The phases come from a fit of the dark
model alone, so the difference map is the same with or without ``-lm``. It is
deliberately not a *phased* difference map: putting the light state's model phases
into the observed amplitude biases the map toward the model the experiment is testing.

Optionally, ``-lm`` with ``--fraction`` adds the light data's model ``Fc_light``/
``PHFc_light`` and three extrapolated light-state maps: the shrunk ``FWT``/``PHWT``, the
phase-aware ``2FEXT_PHASED-Fc`` and the classic ``2FEXT_SCALAR-Fc``. ``--all-columns``
adds the model-phased difference columns.

Examples
--------
::

    # difference map only -- no light model, no fraction needed
    torchref.difference-map \\
        -dm dark.pdb -dsf dark.mtz -lsf light.mtz -o results/

    torchref.difference-map \\
        -dm dark.pdb -lm light.pdb -dsf dark.mtz -lsf light.mtz \\
        --fraction 0.37 --dmin 1.7 --cif ligand.cif -o results/
"""

import argparse
import json
import sys
from pathlib import Path

from torchref.cli._common import (
    add_all_columns_arg,
    add_ded_weight_args,
    add_dual_model_args,
    add_general_args,
    add_outdir_arg,
    add_resolution_args,
    build_dual_column_names,
    configure_unbuffered_output,
    difference_config_from_args,
    load_model,
    parse_device_str,
    register_timing,
    validate_cif_files,
    validate_files,
)

configure_unbuffered_output()


def main():
    """Entry point for ``torchref.difference-map``; returns the exit code."""
    parser = argparse.ArgumentParser(
        prog="torchref.difference-map",
        description="Compute difference and extrapolated map coefficients "
                    "(no refinement).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # difference map only -- needs no light model and no fraction
  torchref.difference-map \\
      -dm dark.pdb \\
      -dsf dark.mtz -lsf light.mtz -o results/

  # adds the light data's model and the extrapolated light-state maps
  torchref.difference-map \\
      -dm dark.pdb -lm light.pdb \\
      -dsf dark.mtz -lsf light.mtz \\
      --fraction 0.37 --dmin 1.7 --dmax 20 -o results/
        """,
    )

    add_dual_model_args(parser, fraction_required=False, light_model_required=False)

    output = parser.add_argument_group("Output")
    add_outdir_arg(
        output, help="Output directory for difference_map.mtz and summary.json"
    )
    add_all_columns_arg(output)
    add_ded_weight_args(output)

    res = parser.add_argument_group("Resolution")
    add_resolution_args(res)

    add_general_args(parser)

    args = parser.parse_args()

    has_light_model = args.light_model is not None
    if has_light_model != (args.fraction is not None):
        parser.error("-lm/--light-model and --fraction go together, or not at all")
    if args.all_columns and not has_light_model:
        parser.error("--all-columns needs -lm/--light-model: its columns use the model")

    to_check = [
        (args.dark_model, "dark model"),
        (args.dark_structure_factor, "dark structure factor"),
        (args.light_structure_factor, "light structure factor"),
    ]
    if has_light_model:
        to_check.insert(1, (args.light_model, "light model"))
    if validate_files(to_check) or validate_cif_files(args.cif):
        return 1

    col_dark, col_light = build_dual_column_names(args)

    import torch

    from torchref.cli.collection_difference_refine import (
        compute_rfactors,
        setup_dark_scaler,
        setup_dataset_collection,
        setup_model_collection,
        setup_scaler,
        write_results_mtz,
    )
    from torchref.utils.serialization import convert_to_serializable

    register_timing()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mtz_path, summary_path = outdir / "difference_map.mtz", outdir / "summary.json"
    device = parse_device_str(args.device)
    fractions = [1.0 - args.fraction, args.fraction] if has_light_model else None

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
        print(f"Output:            {outdir}")
        print(f"Device:            {device}")
        if args.dmin or args.dmax:
            print(f"Resolution:        {args.dmax or 'all'} - {args.dmin or 'all'} A")
        if args.cif:
            print(f"CIF restraints:    {', '.join(args.cif)}")
        print("=" * 72)
        print("Loading reflection data...")
        sys.stdout.flush()

    dc = setup_dataset_collection(
        args.dark_structure_factor, args.light_structure_factor, args.dmin, device,
        column_names_dark=col_dark, column_names_light=col_light, d_max=args.dmax,
    )
    # The models' FFT grid has to reach the data, so without --dmin it follows them.
    d_min = args.dmin or min(dc["dark"].d_min, dc["light"].d_min)

    if args.verbose > 0:
        print("Setting up models and scalers...")
        sys.stdout.flush()

    mc = joint_scaler = None
    if has_light_model:
        mc = setup_model_collection(
            args.dark_model, args.light_model, fractions,
            args.cif, d_min, device, args.verbose,
        )
        mc["light"].freeze_fractions()
        joint_scaler = setup_scaler(dc, mc, device, args.verbose)
        dark_model = mc.base_models[0]
    else:
        dark_model = load_model(
            args.dark_model, max_res=d_min, device=device, verbose=args.verbose,
            cif=args.cif,
        )
    scaler = setup_dark_scaler(dark_model, dc["dark"], device, args.verbose)

    fits = {"dark": (dark_model, dc["dark"], scaler)}
    if has_light_model:
        fits["mixed"] = (mc["light"], dc["light"], joint_scaler)
    r_factors = {}
    for name, (model, data, fit) in fits.items():
        r_work, r_free = compute_rfactors(model, data, fit)
        r_factors[name] = {"r_work": r_work, "r_free": r_free}
        if args.verbose > 0:
            print(f"  R-factor ({name}): R_work={r_work:.4f}  R_free={r_free:.4f}")

    if args.verbose > 0:
        print("Computing map coefficients...")
        sys.stdout.flush()

    with torch.no_grad():
        map_diagnostics = write_results_mtz(
            dc,
            dark_model,
            scaler,
            str(mtz_path),
            mc=mc,
            joint_scaler=joint_scaler,
            all_columns=args.all_columns,
            verbose=args.verbose,
            ded_weight=args.ded_weight,
            difference_config=difference_config_from_args(args),
            program="difference-map",
        )

    summary = {
        "input": {
            "dark_model": args.dark_model,
            "light_model": args.light_model,
            "dark_structure_factor": args.dark_structure_factor,
            "light_structure_factor": args.light_structure_factor,
            "fractions": fractions,
            "cif": args.cif,
            "dmin": args.dmin,
            "dmax": args.dmax,
            "model_grid_dmin": d_min,
        },
        "dataset_scaling": dc.scaling_metrics,
        "results": {"r_factors": r_factors, **map_diagnostics},
        "output_files": {"mtz": str(mtz_path), "summary": str(summary_path)},
    }
    with open(summary_path, "w") as f:
        json.dump(convert_to_serializable(summary), f, indent=2)

    if args.verbose > 0:
        print(f"  Summary written to {summary_path}")
        print("Done.")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
