#!/usr/bin/env python3 -u

"""
Command-line script for LBFGS crystallographic refinement using torchref.

Uses the maximum-likelihood σ_A (Read MLF) target by default. Four other x-ray
targets are selectable via ``--xray-mode``; see
:mod:`torchref.refinement.targets.xray._specs` for the taxonomy.


"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch

from torchref.cli._common import (
    add_adp_mode_arg,
    add_dmin_arg,
    add_general_args,
    add_metadata_args,
    add_n_cycles_arg,
    add_outdir_arg,
    add_output_format_args,
    add_single_model_args,
    add_wavelength_arg,
    add_weights_arg,
    build_column_names,
    configure_unbuffered_output,
    parse_device_str,
    parse_weights,
    register_timing,
    validate_files,
    write_refinement_outputs,
)
from torchref.refinement.targets.xray._specs import XRAY_TARGETS
from torchref.scaling.scaler_base import DEFAULT_SCALE_TARGET, SCALE_TARGETS
from torchref.utils.serialization import convert_to_serializable

configure_unbuffered_output()

# Import stats module early to patch json with StatEntry encoder
import torchref.utils.stats  # noqa: F401,E402


def _sigma_a_kwargs(args) -> dict:
    """Only the σ_A estimator knobs the user actually set.

    Passing ``None`` through would override the library default with ``None``; omitting
    unset flags keeps :mod:`torchref.refinement.model_error_estimation.sigma_a` the
    single source of
    truth for the defaults, so the CLI help and the code cannot drift apart.
    """
    out = {}
    if getattr(args, "sigma_a_max", None) is not None:
        out["sigma_a_max"] = float(args.sigma_a_max)
    if getattr(args, "no_shrink", False):
        out["shrink"] = False
    # `--shrink-passes` is retired: the shrinkage is one-shot, so a pass count would be
    # a flag whose value changes nothing. Accepted as an on/off alias, with a warning.
    passes = getattr(args, "shrink_passes", None)
    if passes is not None:
        warnings.warn(
            "--shrink-passes is deprecated: the stability shrinkage is one-shot now "
            "(it shrinks toward a fitted sigma_A(d*^2) curve, not toward neighbouring "
            f"shells, so passes no longer apply). Treating {passes} as "
            f"{'--no-shrink' if int(passes) <= 0 else 'shrinkage enabled'}; use "
            "--no-shrink instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        out["shrink"] = int(passes) > 0
    return out


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

  # Joined XYZ then ADP cycles
  torchref.refine -m model.pdb -sf reflections.mtz -o output/ --mode everything

  # Plain sigma-weighted Gaussian NLL (no model-error term)
  torchref.refine -m model.pdb -sf reflections.mtz -o output/ --xray-mode nll

Loss weights:
  Default group weights are xray=1 / geometry=0.2 / adp=0.02, with
  geometry/ramachandran=0 (the Ramachandran restraint is OFF by default).
  Weights are hierarchical and MULTIPLICATIVE: a target's effective weight is the 
  product of its path levels (e.g. geometry/ramachandran = weight[geometry] x weight[geometry/ramachandran]),
  so a component key scales within its group. Override any subset with --weights,
  e.g. re-enable Ramachandran with --weights '{"geometry/ramachandran": 1.0}'.
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
        choices=list(XRAY_TARGETS.names),
        # Generated from the table, not hand-written. The hand-written version drifted
        # badly: it advertised 'bhattacharyya' (not in `choices`, so argparse rejected it)
        # and 'gaussian' (an alias), while never mentioning ml_noalpha, nll_beta or nll --
        # three of the modes it actually accepted. `spec.doc` promised it was "surfaced in
        # --help" and had no reader at all.
        help="X-ray target function. " + "  ".join(
            f"'{spec.name}': {spec.doc}" for spec in XRAY_TARGETS.specs
        ),
    )
    refine_group.add_argument(
        "--scale-target",
        type=str,
        default=DEFAULT_SCALE_TARGET,
        choices=list(SCALE_TARGETS),
        help=f"Objective for the scaler's own L-BFGS scale fit (NOT the body target). The "
        f"choices are --xray-mode rows, evaluated by the same target classes the body uses; "
        f"only rows that do not centre on alpha are offered, because alpha is degenerate "
        f"with the scale being fitted. '{DEFAULT_SCALE_TARGET}' (default) is unit-weight "
        f"least squares, weighting every reflection as R itself does. 'nll' is the "
        f"sigma_obs-weighted Gaussian, which up-weights weak reflections. 'ml_noalpha' is "
        f"the Read-MLF sigma_A likelihood; prefer it if the scale collapses in weak shells, "
        f"since its beta absorbs the mismatch instead.",
    )
    refine_group.add_argument(
        "--sigma-a-max",
        type=float,
        default=None,
        help="Upper bound on the per-shell Luzzati σ_A, i.e. the floor on the "
        "model-error variance at (1 - σ_A_max²)·Σ_N. Raising it lets very "
        "well-fitting shells reach a smaller variance. Default 0.99. Applies to the "
        "σ_A-family targets (ml, ml_full, nll_beta) only.",
    )
    refine_group.add_argument(
        "--no-shrink",
        action="store_true",
        help="Disable the per-shell stability shrinkage, which moves each shell's σ_A "
        "toward a fitted σ_A(d*²) curve in proportion to that shell's own sampling "
        "variance (a shell measured precisely is left alone). On by default.",
    )
    refine_group.add_argument(
        "--shrink-passes",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # deprecated on/off alias for --no-shrink; see
        # _sigma_a_kwargs. The shrinkage is one-shot now, so a pass count is meaningless.
    )
    # Defaults are documented in the epilog above; we avoid importing
    # DEFAULT_GROUP_WEIGHTS here so --help stays fast (it would pull in torch).
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
    add_adp_mode_arg(refine_group)
    add_wavelength_arg(refine_group)

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
        print(f"Refinement cycles: {args.n_cycles}")
        if args.with_rigid_body:
            print(f"Rigid-body step:   on (iterations/cutoff = {args.rigid_body_iter})")
        print(f"Device:            {args.device}")
        if args.dmin:
            print(f"Resolution cutoff: {args.dmin:.2f} A")
        adp_line = f"ADP mode:          {args.adp_mode}"
        if args.adp_mode == "anisotropic":
            adp_line += (
                "  (selection: "
                f"{args.anisotropic_selection or 'not resname HOH and not element H'})"
            )
        print(adp_line)
        if args.wavelength == 0:
            print("Anomalous:         off (wavelength 0 -> Friedel-merged read)")
        else:
            print(f"Wavelength:        {args.wavelength:.4g} A")
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
        scale_target=args.scale_target,
        **_sigma_a_kwargs(args),
        adp_mode=args.adp_mode,
        aniso_selection=args.anisotropic_selection,
        wavelength=args.wavelength,
    )

    # Merge onto DEFAULT_GROUP_WEIGHTS so unspecified groups keep their defaults;
    # reset_loss_state() forces the lazy LossState to rebuild with the new weighting.
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
            "adp_mode": args.adp_mode,
            "anisotropic_selection": (
                args.anisotropic_selection if args.adp_mode == "anisotropic" else None
            ),
            "wavelength": args.wavelength,
            "xray_mode": args.xray_mode,
            # Recorded because it changes the R-factors this file reports: without it an
            # archived run cannot be attributed to a scale target, and a change to the
            # default silently invalidates every cached score derived from these numbers.
            "scale_target": args.scale_target,
            **_sigma_a_kwargs(args),
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
        # Same canonical-convention |F_calc| the refinement optimized.
        fcalc = refinement.get_F_calc_scaled(recalc=True)

        # work/free accessor: scaled, validity-masked |F_obs| per subset; .select()
        # aligns the full-size |F_calc| onto the same subset.
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
