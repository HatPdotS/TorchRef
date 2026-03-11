#!/usr/bin/env python3 -u
"""
Convert MTZ map coefficients to CCP4 map file.

Reads amplitude and phase columns from an MTZ file, expands to P1,
and computes a real-space electron density map via FFT.

Similar to phenix.mtz2map but with explicit column name control.

Examples
--------
::

    # 2mFo-DFc map from refinement output
    torchref.mtz2map -sf refined.mtz -csf FWT -cphi PHWT -o 2fofc.ccp4

    # mFo-DFc difference map
    torchref.mtz2map -sf refined.mtz -csf DELFWT -cphi PHDELWT -o fofc.ccp4

    # Custom columns with resolution cutoff
    torchref.mtz2map -sf data.mtz -csf 2FOFCWT -cphi PH2FOFCWT --dmin 2.0 -o map.ccp4
"""

import argparse
import sys

import numpy as np
import torch

from torchref.cli._common import (
    add_general_args,
    add_resolution_args,
    register_timing,
    resolve_device,
)


def main():
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
        "-n",
        "--normalize",
        type=str,
        choices=['True', 'False'],
        default='True',
        help="Normalize amplitudes to unit variance. Default: True.",
    )

    res = parser.add_argument_group("Resolution")
    add_resolution_args(res)

    add_general_args(parser)

    args = parser.parse_args()

    register_timing()

    # --- Device ---
    device = resolve_device(args.device)

    # --- Read MTZ ---
    import reciprocalspaceship as rs

    if args.verbose >= 1:
        print(f"Reading {args.structure_factor}")

    mtz = rs.read_mtz(args.structure_factor)
    available = list(mtz.columns)

    normalize = args.normalize == 'True'

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

    # Drop NaN reflections
    valid = np.isfinite(amplitudes) & np.isfinite(phases_deg)
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
    hkl_t = torch.tensor(hkl, dtype=torch.int32, device=device)
    amp_t = torch.tensor(amplitudes, dtype=torch.float32, device=device)
    phi_t = torch.tensor(phases_deg, dtype=torch.float32, device=device) * (np.pi / 180.0)

    # --- Expand to P1 ---
    from torchref.symmetry.reciprocal_symmetry import expand_hkl

    hkl_p1, orig_idx, phase_shifts = expand_hkl(
        hkl_t, spacegroup, include_friedel=False, remove_absences=True
    )

    amp_p1 = amp_t[orig_idx]
    phi_p1 = phi_t[orig_idx] + phase_shifts

    if args.verbose >= 1:
        print(f"  Expanded: {len(hkl_t)} -> {len(hkl_p1)} reflections (P1)")

    # Complex map coefficients: F * exp(i * phi)
    coefficients = amp_p1 * torch.exp(1j * phi_p1)

    # --- Grid size ---
    from torchref.symmetry.grid_utils import calculate_optimal_grid_size

    if args.gridsize is not None:
        gridsize = tuple(args.gridsize)
    else:
        max_res = float(d_spacings.min())
        gridsize = calculate_optimal_grid_size(cell, max_res, spacegroup)

    if args.verbose >= 1:
        print(f"  Grid size: {gridsize[0]} x {gridsize[1]} x {gridsize[2]}")

    # --- Place on grid and FFT ---
    from torchref.base.reciprocal.grid_operations import place_on_grid

    grid = place_on_grid(hkl_p1, coefficients, gridsize, enforce_hermitian=True)

    # FFT to real space: rho(r) = sum_h F(h) * exp(-2*pi*i * h.r)
    real_map = torch.fft.fftn(grid, dim=(0, 1, 2), norm="forward").real

    if normalize:
        real_map = (real_map - real_map.mean()) / real_map.std()
        
    # --- Write output ---
    from torchref.io.cif import write_map

    write_map(real_map, cell, args.output, spacegroup="P1")

    if args.verbose >= 1:
        print(f"  Written: {args.output}")
        sigma = float(real_map.std())
        print(f"  Map sigma: {sigma:.4f}")


if __name__ == "__main__":
    main()
