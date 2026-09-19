#!/usr/bin/env python3 -u
"""Convert MTZ map coefficients to a CCP4 map file.

Reads amplitude and phase columns, expands to P1, and computes the real-space electron
density by FFT. Like phenix.mtz2map, but with explicit column-name control.

Examples
--------
::

    torchref.mtz2map -sf refined.mtz -csf FWT -cphi PHWT -o 2fofc.ccp4
    torchref.mtz2map -sf refined.mtz -csf DELFWT -cphi PHDELWT --dmin 2.0 -o fofc.ccp4
"""

import argparse
import sys

import numpy as np
import torch

from torchref.config import get_float_dtype
from torchref.cli._common import (
    add_general_args,
    add_resolution_args,
    register_timing,
    parse_device_str,
)


def main():
    """Entry point for ``torchref.mtz2map``."""
    parser = argparse.ArgumentParser(
        description="Convert MTZ map coefficients to a CCP4 map.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  torchref.mtz2map -sf refined.mtz -csf FWT -cphi PHWT -o 2fofc.ccp4
  torchref.mtz2map -sf refined.mtz -csf DELFWT -cphi PHDELWT -o fofc.ccp4
  torchref.mtz2map -sf data.mtz -csf 2FOFCWT -cphi PH2FOFCWT --dmin 2.0 -o map.ccp4
        """,
    )

    inp = parser.add_argument_group("Input")
    inp.add_argument(
        "-sf", "--structure-factor", required=True, type=str, help="Input MTZ file."
    )
    inp.add_argument(
        "-csf",
        "--column-structure-factor",
        required=True,
        type=str,
        metavar="COL",
        help="Column name for amplitudes (e.g. FWT, DELFWT, 2FOFCWT).",
    )
    inp.add_argument(
        "-cphi",
        "--column-phase",
        required=True,
        type=str,
        metavar="COL",
        help="Column name for phases in degrees (e.g. PHWT, PHDELWT, PH2FOFCWT).",
    )
    inp.add_argument(
        "-cw",
        "--column-weight",
        default=None,
        type=str,
        metavar="COL",
        help="Weight column multiplied into the amplitudes before the FFT "
        "(e.g. W_SD, W_IVW from torchref.difference-map). Default: none.",
    )
    inp.add_argument(
        "-ck",
        "--column-scale",
        default=None,
        type=str,
        metavar="COL",
        help="Per-reflection observed-to-model scale factor the amplitudes are divided "
        "by for --units electrons (e.g. KSCALE from torchref.difference-map). "
        "Default: KSCALE when the file has it.",
    )

    output = parser.add_argument_group("Output")
    output.add_argument(
        "-o", "--output", required=True, type=str, help="Output CCP4 map file."
    )

    mapopts = parser.add_argument_group("Map options")
    mapopts.add_argument(
        "--gridsize",
        type=int,
        nargs=3,
        default=None,
        metavar=("NX", "NY", "NZ"),
        help="Override grid dimensions. Default: auto from cell and resolution.",
    )
    mapopts.add_argument(
        "--units",
        type=str,
        choices=["sigma", "electrons", "raw"],
        default=None,
        help="Map units. 'sigma': zero mean and unit standard deviation (default). "
        "'electrons': electrons per cubic Angstrom, sum_h F(h) exp(-2 pi i h.x) / V "
        "with F divided by the --column-scale factor. 'raw': the plain FFT with the "
        "1/N normalisation, no rescaling.",
    )
    mapopts.add_argument(
        "-n",
        "--normalize",
        type=str,
        choices=["True", "False"],
        default=None,
        help="Deprecated alias: '-n True' is '--units sigma', '-n False' is "
        "'--units raw'.",
    )

    res = parser.add_argument_group("Resolution")
    add_resolution_args(res)

    add_general_args(parser)

    args = parser.parse_args()

    register_timing()

    # --- Device ---
    device = parse_device_str(args.device)

    # --- Read MTZ ---
    import reciprocalspaceship as rs

    if args.verbose >= 1:
        print(f"Reading {args.structure_factor}")

    mtz = rs.read_mtz(args.structure_factor)
    available = list(mtz.columns)

    if args.units is not None and args.normalize is not None:
        print("Error: --units and --normalize cannot both be given", file=sys.stderr)
        sys.exit(1)
    if args.units is not None:
        units = args.units
    elif args.normalize is not None:
        units = "sigma" if args.normalize == "True" else "raw"
    else:
        units = "sigma"
    scale_column = args.column_scale
    if scale_column is None and units == "electrons" and "KSCALE" in available:
        scale_column = "KSCALE"
    if units == "electrons" and scale_column is None:
        print(
            "Error: --units electrons needs --column-scale (no KSCALE column found).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.column_structure_factor not in available:
        print(
            f"Error: amplitude column '{args.column_structure_factor}' not found.\n"
            f"Available columns: {available}",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.column_phase not in available:
        print(
            f"Error: phase column '{args.column_phase}' not found.\n"
            f"Available columns: {available}",
            file=sys.stderr,
        )
        sys.exit(1)
    for label, col in (("weight", args.column_weight), ("scale", scale_column)):
        if col is not None and col not in available:
            print(
                f"Error: {label} column '{col}' not found.\n"
                f"Available columns: {available}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Extract cell and spacegroup
    cell = np.array(
        [mtz.cell.a, mtz.cell.b, mtz.cell.c,
         mtz.cell.alpha, mtz.cell.beta, mtz.cell.gamma]
    )
    spacegroup = mtz.spacegroup.hm

    if args.verbose >= 1:
        print(f"  Cell: {cell[0]:.2f} {cell[1]:.2f} {cell[2]:.2f}  "
              f"{cell[3]:.1f} {cell[4]:.1f} {cell[5]:.1f}")
        print(f"  Spacegroup: {spacegroup}")
        print(f"  Columns: {args.column_structure_factor} (amplitude), {args.column_phase} (phase)")

    # Extract HKL, amplitudes, phases
    df = mtz.reset_index()
    hkl = df[["H", "K", "L"]].to_numpy().astype(np.int32)
    amplitudes = df[args.column_structure_factor].to_numpy().astype(np.float32)
    phases_deg = df[args.column_phase].to_numpy().astype(np.float32)
    valid = np.isfinite(amplitudes) & np.isfinite(phases_deg)
    if args.column_weight is not None:
        weights = df[args.column_weight].to_numpy().astype(np.float32)
        valid &= np.isfinite(weights)
        amplitudes = amplitudes * weights
        if args.verbose >= 1:
            print(f"  Weights: {args.column_weight} (mean {np.nanmean(weights):.3f})")
    if units == "electrons":
        kscale = df[scale_column].to_numpy().astype(np.float32)
        valid &= np.isfinite(kscale) & (kscale > 0)
        # Observed amplitudes carry the scaler's overall scale, B and anisotropy;
        # dividing by that factor returns them to electrons.
        amplitudes = amplitudes / np.where(valid, kscale, 1.0)
        if args.verbose >= 1:
            print(f"  Absolute scale: dividing by {scale_column}")

    # Drop NaN reflections
    if not valid.all():
        n_drop = (~valid).sum()
        if args.verbose >= 1:
            print(f"  Dropping {n_drop} reflections with NaN values")
        hkl = hkl[valid]
        amplitudes = amplitudes[valid]
        phases_deg = phases_deg[valid]

    # --- Resolution filter ---
    import gemmi

    gemmi_cell = gemmi.UnitCell(*cell.tolist())
    d_spacings = np.array(
        [gemmi_cell.calculate_d(h) for h in hkl.tolist()], dtype=np.float32
    )

    res_mask = np.ones(len(hkl), dtype=bool)
    if args.dmin is not None:
        res_mask &= d_spacings >= args.dmin
    if args.dmax is not None:
        res_mask &= d_spacings <= args.dmax
    if not res_mask.all():
        n_before = len(hkl)
        hkl = hkl[res_mask]
        amplitudes = amplitudes[res_mask]
        phases_deg = phases_deg[res_mask]
        d_spacings = d_spacings[res_mask]
        if args.verbose >= 1:
            print(f"  Resolution filter: {n_before} -> {len(hkl)} reflections")

    if args.verbose >= 1:
        print(f"  {len(hkl)} reflections, resolution range: "
              f"{d_spacings.max():.2f} - {d_spacings.min():.2f} A")

    # --- Convert to torch ---
    hkl_t = torch.tensor(hkl, dtype=torch.int32, device=device)  # dtype-ok: hkl Miller indices fed to symmetry expand; fixed int32 crystallographic representation
    amp_t = torch.tensor(amplitudes, dtype=get_float_dtype(), device=device)
    phi_t = torch.tensor(phases_deg, dtype=get_float_dtype(), device=device) * (np.pi / 180.0)

    # --- Expand to P1 ---
    from torchref.symmetry import Cell, SpaceGroup

    sg = SpaceGroup(spacegroup)
    hkl_p1, orig_idx, phase_shifts = sg.expand_hkl(
        hkl_t, include_friedel=False, remove_absences=True
    )

    amp_p1 = amp_t[orig_idx]
    phi_p1 = phi_t[orig_idx] + phase_shifts

    if args.verbose >= 1:
        print(f"  Expanded: {len(hkl_t)} -> {len(hkl_p1)} reflections (P1)")

    # Complex map coefficients: F * exp(i * phi)
    coefficients = amp_p1 * torch.exp(1j * phi_p1)

    # --- Grid size ---
    if args.gridsize is not None:
        gridsize = tuple(args.gridsize)
    else:
        max_res = float(d_spacings.min())
        gridsize = sg.optimal_grid_size(Cell(cell), max_res)

    if args.verbose >= 1:
        print(f"  Grid size: {gridsize[0]} x {gridsize[1]} x {gridsize[2]}")

    # --- Place on grid and FFT ---
    from torchref.base.reciprocal.grid_operations import place_on_grid

    grid = place_on_grid(hkl_p1, coefficients, gridsize, enforce_hermitian=True)

    # FFT to real space with the 1/N normalisation: rho_raw(r) = (1/N) sum_h F(h) exp(-2 pi i h.r)
    real_map = torch.fft.fftn(grid, dim=(0, 1, 2), norm="forward").real

    if units == "sigma":
        real_map = (real_map - real_map.mean()) / real_map.std()
    elif units == "electrons":
        # rho(r) = (1/V) sum_h F(h) exp(-2 pi i h.r): undo the 1/N and divide by the
        # cell volume, so the map is in electrons per cubic Angstrom.
        volume = Cell(cell, device=device).volume.to(real_map.dtype)
        real_map = real_map * (real_map.numel() / volume)

    # --- Write output ---
    from torchref.io.cif import write_map

    write_map(real_map, cell, args.output, spacegroup="P1")

    if args.verbose >= 1:
        print(f"  Written: {args.output}")
        sigma = float(real_map.std())
        print(f"  Units: {units}")
        print(f"  Map sigma: {sigma:.4f}")


if __name__ == "__main__":
    main()
