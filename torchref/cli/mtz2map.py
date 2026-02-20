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
    torchref-mtz2map -f refined.mtz --amplitude FWT --phase PHWT -o 2fofc.ccp4

    # mFo-DFc difference map
    torchref-mtz2map -f refined.mtz --amplitude DELFWT --phase PHDELWT -o fofc.ccp4

    # Custom columns with resolution cutoff
    torchref-mtz2map -f data.mtz --amplitude 2FOFCWT --phase PH2FOFCWT --high-res 2.0 -o map.ccp4
"""

import argparse
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Convert MTZ map coefficients to a CCP4 map.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  torchref-mtz2map -f refined.mtz --amplitude FWT --phase PHWT -o 2fofc.ccp4
  torchref-mtz2map -f refined.mtz --amplitude DELFWT --phase PHDELWT -o fofc.ccp4
  torchref-mtz2map -f data.mtz -F 2FOFCWT -P PH2FOFCWT --high-res 2.0 -o map.ccp4
        """,
    )

    parser.add_argument(
        "-f", "--mtz", required=True, type=str, help="Input MTZ file."
    )
    parser.add_argument(
        "-F",
        "--amplitude",
        required=True,
        type=str,
        help="Column name for amplitudes (e.g. FWT, DELFWT, 2FOFCWT).",
    )
    parser.add_argument(
        "-P",
        "--phase",
        required=True,
        type=str,
        help="Column name for phases in degrees (e.g. PHWT, PHDELWT, PH2FOFCWT).",
    )
    parser.add_argument(
        "-o", "--output", required=True, type=str, help="Output CCP4 map file."
    )
    parser.add_argument(
        "--high-res",
        type=float,
        default=None,
        help="High-resolution cutoff in Angstroms. Default: use all data.",
    )
    parser.add_argument(
        "--low-res",
        type=float,
        default=None,
        help="Low-resolution cutoff in Angstroms. Default: use all data.",
    )
    parser.add_argument(
        "--gridsize",
        type=int,
        nargs=3,
        default=None,
        metavar=("NX", "NY", "NZ"),
        help="Override grid dimensions. Default: auto from cell and resolution.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity (0=silent, 1=normal, 2=debug). Default: 1.",
    )

    args = parser.parse_args()

    # --- Read MTZ ---
    import reciprocalspaceship as rs

    if args.verbose >= 1:
        print(f"Reading {args.mtz}")

    mtz = rs.read_mtz(args.mtz)
    available = list(mtz.columns)

    if args.amplitude not in available:
        print(
            f"Error: amplitude column '{args.amplitude}' not found.\n"
            f"Available columns: {available}",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.phase not in available:
        print(
            f"Error: phase column '{args.phase}' not found.\n"
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
        print(f"  Columns: {args.amplitude} (amplitude), {args.phase} (phase)")

    # Extract HKL, amplitudes, phases
    df = mtz.reset_index()
    hkl = df[["H", "K", "L"]].to_numpy().astype(np.int32)
    amplitudes = df[args.amplitude].to_numpy().astype(np.float32)
    phases_deg = df[args.phase].to_numpy().astype(np.float32)

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
    if args.high_res is not None:
        res_mask &= d_spacings >= args.high_res
    if args.low_res is not None:
        res_mask &= d_spacings <= args.low_res
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
    hkl_t = torch.tensor(hkl, dtype=torch.int32)
    amp_t = torch.tensor(amplitudes, dtype=torch.float32)
    phi_t = torch.tensor(phases_deg, dtype=torch.float32) * (np.pi / 180.0)

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

    # --- Write output ---
    from torchref.io.cif import write_map

    write_map(real_map, cell, args.output, spacegroup="P1")

    if args.verbose >= 1:
        print(f"  Written: {args.output}")
        sigma = float(real_map.std())
        print(f"  Map sigma: {sigma:.4f}")


if __name__ == "__main__":
    main()
