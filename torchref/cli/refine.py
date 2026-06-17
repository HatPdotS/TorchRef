#!/usr/bin/env python3 -u

"""
Command-line script for LBFGS crystallographic refinement using torchref.

Uses the maximum-likelihood σ_A (Read MLF) target by default; Bhattacharyya /
Gaussian / least-squares / plain maximum-likelihood targets remain available
via ``--xray-mode``.

Examples
--------
::

    # Default: ml (maximum-likelihood Luzzati σ_A) target, joint XYZ+ADP+scaler LBFGS
    torchref.refine -m model.pdb -sf reflections.mtz -o output_dir/

    # 10 refinement cycles
    torchref.refine -m model.pdb -sf reflections.mtz -o output/ -n 10

    # Alternative targets
    torchref.refine -m model.pdb -sf reflections.mtz -o output/ --xray-mode bhattacharyya
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
    add_weights_arg,
    build_column_names,
    configure_unbuffered_output,
    parse_device_str,
    parse_weights,
    register_timing,
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
  # Default: SigmaA target, separate XYZ and ADP+scaler LBFGS
  torchref.refine -m model.pdb -sf reflections.mtz -o output_dir/

  # 10 refinement cycles
  torchref.refine -m model.pdb -sf reflections.mtz -o output/ -n 10

  # Separated XYZ then ADP cycles
  torchref.refine -m model.pdb -sf reflections.mtz -o output/ --mode separate

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
        default="separate",
        choices=["separate", "everything"],
        help='Refinement mode: "everything" for joint XYZ+ADP+scaler LBFGS, '
        '"separate" for separated XYZ then ADP cycles (default: "separate")',
    )
    refine_group.add_argument(
        "--xray-mode",
        type=str,
        default="ml",
        choices=["gaussian", "ls", "ls_wunit_k1", "ml", "bhattacharyya"],
        help="X-ray target function. 'ml' (default) is the "
        "maximum-likelihood Read MLF target with a cross-validated Luzzati "
        "sigma_A term (Phenix-style alpha/beta). 'bhattacharyya' uses the "
        "Bhattacharyya overlap loss with first-principles model error "
        "estimation; 'ls'/'gaussian' are simpler alternatives.",
    )
    refine_group.add_argument(
        "--sigma-m-scale",
        type=float,
        default=1.0,
        help="Global multiplier applied to σ_m for the Bhattacharyya target. "
        "Ignored for other targets. Default 1.0.",
    )
    add_weights_arg(refine_group)
    refine_group.add_argument(
        "--with-rigid-body",
        action="store_true",
        help="Run one multi-resolution rigid-body refinement step (per-chain "
        "rotation + translation) once at the start of refinement, before any "
        "macro cycle. Useful when the starting model has small global "
        "misorientation/shift. Combine with -n 0 to run rigid-body only.",
    )
    refine_group.add_argument(
        "--rigid-body-iter",
        type=int,
        default=30,
        help="LBFGS max_iter for each per-cutoff rigid-body step. "
        "Only used when --with-rigid-body is set. Default 30.",
    )
    refine_group.add_argument(
        "--rigid-body-cutoffs",
        type=str,
        default=None,
        help="Comma-separated high-resolution cutoffs (A) for the rigid-body "
        "schedule, coarse to fine, e.g. '10.15,8.80,5.89,4.10,3.00'. "
        "If unset, an auto schedule is derived from native d_min. "
        "Only used when --with-rigid-body is set.",
    )

    res = parser.add_argument_group("Resolution")
    add_dmin_arg(res)

    add_general_args(parser)

    args = parser.parse_args()

    register_timing()

    # Parse weights
    manual_weights, weights_err = parse_weights(args.weights)
    if weights_err:
        print(f"Error: {weights_err}", file=sys.stderr)
        sys.exit(1)

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
        if args.with_rigid_body:
            print(f"Rigid-body step:   on (iterations/cutoff = {args.rigid_body_iter})")
        print(f"Device:            {args.device}")
        if args.dmin:
            print(f"Resolution cutoff: {args.dmin:.2f} A")
        if manual_weights:
            print(f"Manual weights:    {json.dumps(manual_weights)}")
        print("=" * 80)
        print()
        sys.stdout.flush()

    device = parse_device_str(args.device)

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

    # Apply manual group weights, if given. Merge onto DEFAULT_GROUP_WEIGHTS so
    # unspecified groups keep their defaults (e.g. --weights '{"xray": 5}' sets
    # xray=5 while geometry=1 / adp=0.1 stay). reset_loss_state() forces the lazy
    # LossState to rebuild with the new weighting.
    if manual_weights:
        from torchref.refinement.base_refinement import DEFAULT_GROUP_WEIGHTS
        from torchref.refinement.weighting import ManualWeighting

        merged = {**DEFAULT_GROUP_WEIGHTS, **manual_weights}
        refinement.weighting = ManualWeighting(merged)
        refinement.reset_loss_state()
        if args.verbose > 0:
            print(f"Applied manual group weights: {merged}")

    if args.verbose > 0:
        print("Refinement initialized successfully.\n")
        sys.stdout.flush()

    # Run refinement
    try:
        if args.verbose > 0:
            print(f"Starting refinement with {args.n_cycles} macro cycles...\n")
            if args.with_rigid_body:
                print(
                    "Rigid-body step enabled: one multi-resolution rigid-body "
                    "pass will run once at the start of refinement.\n"
                )
            sys.stdout.flush()

        if args.mode == "separate":
            cycle_fn = refinement.refine
        elif args.mode == "everything":
            cycle_fn = refinement.refine_everything
        else:
            raise ValueError(f"Invalid refinement mode: {args.mode}")

        if args.with_rigid_body:
            if args.verbose > 0:
                print("\n--- Rigid-body step (once, at start) ---")
                sys.stdout.flush()
            rb_cutoffs = None
            if args.rigid_body_cutoffs:
                rb_cutoffs = [float(x) for x in args.rigid_body_cutoffs.split(",")]
            refinement.refine_rigid_body(
                cutoffs=rb_cutoffs,
                iterations_per_step=args.rigid_body_iter,
                commit=True,
            )

        if args.n_cycles > 0:
            cycle_fn(macro_cycles=args.n_cycles)

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
            "weights": manual_weights if manual_weights else None,
            "dmin": args.dmin,
            "device": str(device),
            "with_rigid_body": args.with_rigid_body,
            "rigid_body_iter": args.rigid_body_iter if args.with_rigid_body else None,
            "rigid_body_cutoffs": (
                args.rigid_body_cutoffs if args.with_rigid_body else None
            ),
        },
        "history": refinement.history if hasattr(refinement, "history") else {},
        "final_statistics": {},
    }

    # Add final R-factors if available
    try:
        work_nll, test_nll = refinement.nll_xray()
        rd = refinement.reflection_data
        # Signed HKL so |F_calc| matches what refinement optimized (anomalous mates).
        fcalc = refinement.get_F_calc_scaled(rd.hkl_for_sf(), recalc=True)

        # work/free accessor: scaled, validity-masked |F_obs| per subset; .select()
        # aligns the full-size |F_calc| onto the same subset. Replaces the
        # deprecated data() unpack and applies the validity mask to the counts too.
        fobs_work, fobs_free = rd.work.F, rd.free.F
        r_work = torch.sum(
            torch.abs(fobs_work - rd.work.select(fcalc))
        ) / torch.sum(fobs_work)
        r_free = torch.sum(
            torch.abs(fobs_free - rd.free.select(fcalc))
        ) / torch.sum(fobs_free)

        history_data["final_statistics"] = {
            "R_work": float(r_work.item()),
            "R_free": float(r_free.item()),
            "NLL_work": float(work_nll.item()),
            "NLL_test": float(test_nll.item()),
            "n_reflections_work": rd.work.n,
            "n_reflections_test": rd.free.n,
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
