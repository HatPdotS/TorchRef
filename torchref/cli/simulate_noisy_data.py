#!/usr/bin/env python3
"""Simulate a noisy reflection MTZ from a structure file.

Two modes are supported, chosen by whether ``--reference-hkl`` is passed:

**Reference-driven (preferred)**
    The model is scaled to the reference dataset with torchref's
    ``Scaler`` so that Fcalc and reference intensities share an absolute
    scale. The simulation then inherits the reference's HKL list, and
    noise for each reflection uses the reference's reported ``sigma(I)``
    directly — no parametric model, no fitting. Use this when you have a
    CrystFEL ``.hkl`` (or equivalent) from a real experiment.

**Parametric fallback**
    When ``--reference-hkl`` is omitted, intensity sigma is built from a
    three-term variance-additive model
    ``sigma_I^2 = sigma_lin^2 * I + sigma_mul^2 * I^2 + sigma_abs_I^2``.

In both modes, two independent noise realizations are drawn per
reflection; R-split and Pearson CC between the halves are printed, and
the mean is written as F-obs/SIGF-obs (default) or I-obs/SIGI-obs.

Usage
-----
::

    # Reference-driven
    torchref.simulate-noisy-data input.pdb out.mtz \
        --reference-hkl td1.hkl --d-min 2.0

    # Parametric (no reference)
    torchref.simulate-noisy-data input.pdb out.mtz \
        --sigma-lin 5 --sigma-mul 0.05 --sigma-abs 0.33

    # Intensity output
    torchref.simulate-noisy-data input.pdb out.mtz \
        --reference-hkl td1.hkl --output-type intensities
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from torchref.cli._common import (
    add_device_arg,
    add_verbose_arg,
    configure_unbuffered_output,
    load_model,
    parse_device_str,
)
from torchref.io import mtz
from torchref.io.datasets import FcalcDataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torchref.simulate-noisy-data",
        description=(
            "Compute Fcalc from a structure, add Gaussian intensity noise "
            "(optionally grafted from a CrystFEL reference hkl), and write "
            "an MTZ as F-obs/SIGF-obs (default) or I-obs/SIGI-obs."
        ),
    )
    parser.add_argument("input", help="Input structure file (.pdb / .cif / .mmcif)")
    parser.add_argument("output", help="Output MTZ file")
    parser.add_argument(
        "--d-min",
        type=float,
        default=2.0,
        help="High-resolution limit in Angstroms (default: 2.0)",
    )
    parser.add_argument(
        "--d-max",
        type=float,
        default=None,
        help="Low-resolution limit in Angstroms (default: no cutoff)",
    )
    parser.add_argument(
        "--output-type",
        choices=("amplitudes", "intensities"),
        default="amplitudes",
        help="Write F-obs/SIGF-obs (amplitudes, default) or I-obs/SIGI-obs",
    )
    parser.add_argument(
        "--reference-hkl",
        type=str,
        default=None,
        help=("CrystFEL partialator .hkl file. When given, the model is "
              "scaled to this reference and per-reflection sigmas are "
              "grafted from it (parametric sigma-* flags are ignored)."),
    )
    parser.add_argument(
        "--sigma-lin",
        type=float,
        default=0.0,
        help=("Parametric only: Poisson coefficient — sigma_I^2 += "
              "sigma_lin^2 * I (default: 0.0). Ignored if --reference-hkl."),
    )
    parser.add_argument(
        "--sigma-mul",
        type=float,
        default=0.05,
        help=("Parametric only: multiplicative coefficient — sigma_I^2 += "
              "(sigma_mul * I)^2 (default: 0.05). Ignored if --reference-hkl."),
    )
    parser.add_argument(
        "--sigma-abs",
        type=float,
        default=0.0,
        help=("Parametric only: target 1/SNR at the resolution limit from "
              "the absolute-noise term (default: 0.0). Ignored if "
              "--reference-hkl."),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random generator (default: non-reproducible)",
    )
    add_device_arg(parser)
    add_verbose_arg(parser)
    return parser


def main():
    configure_unbuffered_output()
    args = _build_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1
    if args.reference_hkl is not None and not Path(args.reference_hkl).is_file():
        print(
            f"Error: reference hkl not found: {args.reference_hkl}",
            file=sys.stderr,
        )
        return 1
    if args.sigma_lin < 0 or args.sigma_mul < 0 or args.sigma_abs < 0:
        print("Error: all --sigma-* flags must be non-negative", file=sys.stderr)
        return 1

    device = parse_device_str(args.device)

    if args.verbose:
        print(f"Loading structure: {args.input}")
    model = load_model(
        args.input, max_res=args.d_min, device=device, verbose=args.verbose
    )

    if args.reference_hkl is not None:
        return _run_reference_mode(args, model, device)
    return _run_parametric_mode(args, model, device)


def _run_reference_mode(args, model, device) -> int:
    """Reference-driven: scale model to CrystFEL hkl and graft sigmas."""
    from torchref import ReflectionData
    from torchref.base.reciprocal import get_d_spacing
    from torchref.scaling import Scaler

    if args.verbose:
        print(f"Loading CrystFEL reference: {args.reference_hkl}")
    ref = ReflectionData(device=str(device), verbose=args.verbose).load_crystfel_hkl(
        args.reference_hkl, cell=model.cell, spacegroup=model.spacegroup,
    )
    # Prune reference tensors by resolution so the simulation output only
    # covers the requested range. cut_res() only masks — we want the HKL
    # list itself to be filtered so Scaler, model(), and add_noise all see
    # the same reflection set.
    if args.d_min is not None or args.d_max is not None:
        res = ref.resolution
        mask = torch.ones_like(res, dtype=torch.bool)
        if args.d_min is not None:
            mask &= res >= args.d_min
        if args.d_max is not None:
            mask &= res <= args.d_max
        for field in ("hkl", "I", "I_sigma", "F", "F_sigma", "resolution", "rfree_flags"):
            t = getattr(ref, field, None)
            if t is not None:
                setattr(ref, field, t[mask])
        # Replace the masks with a single all-True flagged_initial matching
        # the new tensor size. An empty TensorMasks() returns None from
        # __call__(), which downstream consumers (Scaler.get_bins) don't
        # tolerate.
        ref.masks = type(ref.masks)(device=ref.device)
        ref.masks["flagged_initial"] = torch.ones(
            len(ref.hkl), dtype=torch.bool, device=ref.device
        )
    if args.verbose:
        print(f"Reference: {len(ref.hkl)} reflections after resolution cuts")

    if args.verbose:
        print("Scaling model to reference...")
    scaler = Scaler(model, ref, device=device, verbose=args.verbose)
    scaler.initialize().refine_lbfgs()

    if args.verbose:
        print(f"Computing scaled Fcalc on {len(ref.hkl)} reference HKLs")
    with torch.no_grad():
        fcalc_scaled = scaler(model(ref.hkl))

    sim = FcalcDataset(
        hkl=ref.hkl.clone(),
        resolution=get_d_spacing(ref.hkl.float(), ref.cell.data),
        cell=ref.cell,
        spacegroup=ref.spacegroup,
        device=device,
    )
    sim.set_fcalc(fcalc_scaled)

    noisy = sim.add_noise(reference=ref, seed=args.seed, verbose=bool(args.verbose))
    _write_output(args, noisy)
    return 0


def _run_parametric_mode(args, model, device) -> int:
    """No reference: build HKL from cell+resolution, use three-term model."""
    if args.verbose:
        print(
            f"Generating HKL to d_min={args.d_min} A"
            + (f", d_max={args.d_max} A" if args.d_max is not None else "")
        )
    dataset = FcalcDataset.from_cell_and_resolution(
        cell=model.cell,
        spacegroup=model.spacegroup,
        d_min=args.d_min,
        d_max=args.d_max,
        device=device,
    )
    if args.verbose:
        print(f"Computing Fcalc for {len(dataset)} reflections")
    hkl = dataset.hkl.to(device)
    fcalc = model(hkl, recalc=True)
    dataset.set_fcalc(fcalc)

    noisy = dataset.add_noise(
        sigma_lin=args.sigma_lin,
        sigma_mul=args.sigma_mul,
        sigma_abs=args.sigma_abs,
        seed=args.seed,
        verbose=bool(args.verbose),
    )
    _write_output(args, noisy)
    return 0


def _write_output(args, noisy: FcalcDataset) -> None:
    """Write noisy amplitudes or intensities and their uncertainties to MTZ."""
    hkl_np = noisy.hkl.cpu().numpy()
    columns = {
        "H": hkl_np[:, 0],
        "K": hkl_np[:, 1],
        "L": hkl_np[:, 2],
    }

    if args.output_type == "intensities":
        # The intensity add_noise actually drew, not the square of the clamped
        # amplitude. Squaring back would floor every negative reflection at zero, and
        # that is a positive bias concentrated exactly where the noise dominates -- the
        # same signature as a real positive perturbation of the merged intensity.
        columns["I-obs"] = noisy.I.detach().cpu().numpy()
        columns["SIGI-obs"] = noisy.I_sigma.detach().cpu().numpy()
    else:
        columns["F-obs"] = noisy.fcalc_amp.detach().cpu().numpy()
        columns["SIGF-obs"] = noisy.fobs_sigma.detach().cpu().numpy()

    df = pd.DataFrame(columns)
    mtz.write(df, noisy.cell.data, noisy.spacegroup, args.output)

    if args.verbose:
        print(
            f"Wrote {args.output} "
            f"({args.output_type}, n={len(noisy)})"
        )


if __name__ == "__main__":
    sys.exit(main() or 0)
