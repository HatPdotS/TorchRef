#!/usr/bin/env python3 -u
"""
Validate difference electron density (DED) by correlating DFo and DFc maps.

Takes separate dark and light MTZ files, computes weighted difference
amplitudes internally, then compares weighted DFo and DFcalc maps using
dark-state phases.

Supports Phenix-style atom selections for regional correlation analysis
(e.g., around a ligand binding site).

Examples
--------
::

    # Basic validation (full cell correlation)
    torchref.validate-ded -dsf dark.mtz -lsf light.mtz \\
        -dm dark.pdb -lm light.pdb

    # With light fraction and ligand masking (both models, default)
    torchref.validate-ded -dsf dark.mtz -lsf light.mtz \\
        -dm dark.pdb -lm light.pdb \\
        --fraction 0.20 --selection "chain B and resname IBL" --mask-radius 2.5

    # Mask from light model only
    torchref.validate-ded -dsf dark.mtz -lsf light.mtz \\
        -dm dark.pdb -lm light.pdb \\
        --fraction 0.20 --selection "resname IBL" --mask-source light

    # Full output with plots and CCP4 maps
    torchref.validate-ded -dsf dark.mtz -lsf light.mtz \\
        -dm dark.pdb -lm light.pdb \\
        --fraction 0.20 --selection "resname IBL" --plot --write-maps -o validation/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from torchref.cli._common import (
    add_dual_model_args,
    add_dmin_arg,
    add_general_args,
    add_outdir_arg,

    build_dual_column_names,
    configure_unbuffered_output,
    load_model,
    load_reflection_data,
    register_timing,
    resolve_device,
    validate_cif_files,
    validate_files,
)
from torchref.utils.serialization import convert_to_serializable

configure_unbuffered_output()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def build_atom_mask(selection_xyz, real_space_grid, cell, mask_radius, device):
    """Build a boolean voxel mask around selected atom positions.

    Parameters
    ----------
    selection_xyz : torch.Tensor
        Cartesian coordinates of selected atoms, shape (N, 3).
    real_space_grid : torch.Tensor
        Real-space grid from ``get_real_grid()``.
    cell : Cell or torch.Tensor
        Unit cell parameters (or Cell object with ``.data`` attribute).
    mask_radius : float
        Radius in Angstroms around each atom to include.
    device : torch.device
        Device for tensor operations.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape ``grid_shape[:3]``.
    """
    from torchref.base.coordinates.transforms_torch import (
        get_fractional_matrix,
        get_inv_fractional_matrix_torch as get_inverse_fractional_matrix,
    )
    from torchref.base.electron_density.solvent_mask import add_to_solvent_mask
    from torchref.base.electron_density.voxel_utils import find_relevant_voxels

    grid_shape = real_space_grid.shape[:3]
    cell_t = cell.data if hasattr(cell, "data") else cell
    inv_frac = get_inverse_fractional_matrix(cell_t).to(device)
    frac = get_fractional_matrix(cell_t).to(device)

    surrounding_coords, voxel_indices = find_relevant_voxels(
        real_space_grid,
        selection_xyz,
        radius_angstrom=mask_radius,
        inv_frac_matrix=inv_frac,
    )

    mask = torch.zeros(grid_shape, dtype=torch.int32, device=device)
    mask = add_to_solvent_mask(
        surrounding_coords,
        voxel_indices,
        mask,
        selection_xyz,
        mask_radius,
        inv_frac,
        frac,
    )
    return mask > 0


def compute_correlation(map1, map2, mask):
    """Pearson correlation coefficient between two maps within a mask."""
    v1 = map1[mask]
    v2 = map2[mask]
    v1 = v1 - v1.mean()
    v2 = v2 - v2.mean()
    cc = (v1 * v2).sum() / (torch.sqrt((v1**2).sum() * (v2**2).sum()) + 1e-12)
    return cc.item()


def compute_map_from_coefficients(amplitudes, phases_rad, hkl_p1, gridsize):
    """Compute a real-space map from Fourier coefficients.

    Parameters
    ----------
    amplitudes : torch.Tensor
        Structure factor amplitudes (can be signed), shape (N,).
    phases_rad : torch.Tensor
        Phases in radians, shape (N,).
    hkl_p1 : torch.Tensor
        P1-expanded Miller indices, shape (N, 3).
    gridsize : tuple
        Grid dimensions (nx, ny, nz).

    Returns
    -------
    torch.Tensor
        Real-space 3D map.
    """
    from torchref.base.reciprocal.grid_operations import place_on_grid

    coefficients = amplitudes * torch.exp(1j * phases_rad)
    grid = place_on_grid(hkl_p1, coefficients, gridsize, enforce_hermitian=True)
    return torch.fft.fftn(grid, dim=(0, 1, 2), norm="forward").real


def generate_plots(results, map_dfo, map_dfc, mask_dict, outdir, verbose):
    """Generate a 2-panel validation figure.

    Parameters
    ----------
    results : dict
        Correlation results dictionary.
    map_dfo, map_dfc : torch.Tensor
        Real-space DFo and DFc maps.
    mask_dict : dict
        Mapping of region name to boolean mask tensor.
    outdir : Path
        Output directory for plots.
    verbose : int
        Verbosity level.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("DED Validation: WDFo vs WDFcalc", fontsize=14)

    rs_corr = results["realspace_correlation"]


    # Panel 1: Resolution-binned CC
    ax = axes[0]
    bins = results["resolution_bins"]
    d_mins = [b["d_min"] for b in bins]
    cc_vals = [b["cc"] for b in bins]
    ax.plot(d_mins, cc_vals, "o-", color="steelblue", markersize=5)
    ax.set_xlabel("Resolution (A)")
    ax.set_ylabel("CC(WDFo, WDFcalc)")
    ax.set_title("Amplitude CC by Resolution")
    ax.invert_xaxis()
    ax.axhline(y=0, color="gray", ls="--", alpha=0.3)
    ax.set_ylim(-0.2, 1.0)

    # Panel 2: Scatter of map values in first masked region (or full cell)
    ax = axes[1]
    # Use the first non-full_cell mask, or full_cell if no selection masks
    scatter_region = "full_cell"
    for name in mask_dict:
        if name != "full_cell":
            scatter_region = name
            break
    scatter_mask = mask_dict[scatter_region]
    v_dfo = map_dfo[scatter_mask].detach().cpu().numpy()
    v_dfc = map_dfc[scatter_mask].detach().cpu().numpy()
    ax.scatter(v_dfo, v_dfc, s=2, alpha=0.3, color="steelblue")
    ax.set_xlabel("WDFo map (sigma)")
    ax.set_ylabel("WDFcalc map (sigma)")
    cc_val = rs_corr[scatter_region]["cc"]
    ax.set_title(f"{scatter_region} Scatter (CC={cc_val:.3f})")
    if len(v_dfo) > 2:
        # Identity line (ideal for sigma-normalized maps)
        lim = max(abs(v_dfo).max(), abs(v_dfc).max())
        ax.plot([-lim, lim], [-lim, lim], "r-", linewidth=1.5, alpha=0.5, label="y=x")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)

    plt.tight_layout()
    png_path = outdir / "validate_ded.png"
    pdf_path = outdir / "validate_ded.pdf"
    plt.savefig(str(png_path), dpi=200)
    plt.savefig(str(pdf_path))
    plt.close(fig)
    if verbose >= 1:
        print(f"  Plots saved to {png_path}")


# ---------------------------------------------------------------------------
# Core validation pipeline
# ---------------------------------------------------------------------------


def run_validation(args):
    """Run the DED validation pipeline."""
    import gemmi

    from torchref import DatasetCollection
    from torchref.model.model_collection import ModelCollection
    from torchref.cli.collection_difference_refine import (
        compute_rfactors, setup_scaler as setup_collection_scaler,
    )
    from torchref.base.fourier.grid import get_real_grid
    from torchref.symmetry.grid_utils import calculate_optimal_grid_size
    from torchref.symmetry.reciprocal_symmetry import expand_hkl

    device = resolve_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.verbose >= 1:
        print("=" * 70)
        print("DED Validation: WDFo vs WDFcalc")
        print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load dark and light MTZ, compute weighted differences
    # ------------------------------------------------------------------
    if args.verbose >= 1:
        print(f"\n--- Loading reflection data ---")
        print(f"  Dark SF:   {args.dark_structure_factor}")
        print(f"  Light SF:  {args.light_structure_factor}")

    col_dark, col_light = build_dual_column_names(args)

    data_dark = load_reflection_data(
        args.dark_structure_factor, device=device, column_names=col_dark, verbose=0
    )
    data_light = load_reflection_data(
        args.light_structure_factor, device=device, column_names=col_light, verbose=0
    )

    # Resolution cutoff
    d_min = args.dmin
    if d_min is not None:
        data_dark.cut_res(highres=d_min)
        data_light.cut_res(highres=d_min)

    # Align HKL and scale datasets to a common scale
    collection = DatasetCollection(device=str(device))
    collection.add_dataset("dark", data_dark)
    collection.add_dataset("light", data_light)
    collection.scale()

    if args.verbose >= 1:
        print(f"  Scale parameters after optimization:")
        for name, ds in collection:
            if hasattr(ds, "log_scale") and ds.log_scale is not None:
                print(f"    {name}: log_scale={ds.log_scale.item():.6f} "
                      f"(scale={torch.exp(ds.log_scale).item():.6f})")
            if hasattr(ds, "U_aniso") and ds.U_aniso is not None:
                print(f"    {name}: U_aniso={ds.U_aniso.detach().cpu().tolist()}")

    # Extract matched F and sigma
    hkl_all, F_dark_mt, sig_dark_mt, _ = data_dark()
    _, F_light_mt, sig_light_mt, _ = data_light()
    mask = F_dark_mt.get_mask() & F_light_mt.get_mask()

    F_dark = F_dark_mt.get_data()[mask]
    F_light = F_light_mt.get_data()[mask]
    sig_dark = sig_dark_mt.get_data()[mask]
    sig_light = sig_light_mt.get_data()[mask]
    hkl = hkl_all[mask]

    # Compute weighted differences (same as difference_refine.py)
    dfo = F_light - F_dark
    sig_diff = (sig_dark**2 + sig_light**2) ** 0.5
    weights = 1 / sig_diff**2
    weights = weights / weights.mean()
    w_dfo = dfo * weights

    # Cell and spacegroup from dark dataset
    cell_t = data_dark.cell.data.to(device)
    cell_np = cell_t.cpu().numpy()
    sg_name = data_dark.spacegroup.hm

    # Determine resolution from data if not specified
    gemmi_cell = gemmi.UnitCell(*cell_np.tolist())
    hkl_np = hkl.cpu().numpy().astype(np.int32)
    d_spacings = np.array(
        [gemmi_cell.calculate_d(h) for h in hkl_np.tolist()], dtype=np.float32
    )
    if d_min is None:
        d_min = float(d_spacings.min())

    if args.verbose >= 1:
        print(f"  Matched reflections: {len(hkl)}")
        print(f"  Cell: {cell_np}")
        print(f"  Spacegroup: {sg_name}")
        print(f"  Resolution cutoff: {d_min:.2f} A")
        n_neg = int((dfo < 0).sum().item())
        print(f"  DFo: {len(dfo)} reflections ({n_neg} negative)")
        print(f"  Weight range: {weights.min():.3f} - {weights.max():.3f}")

    # ------------------------------------------------------------------
    # 2. Load models
    # ------------------------------------------------------------------
    if args.verbose >= 1:
        print(f"\n--- Loading models ---")

    model_dark = load_model(
        args.dark_model, max_res=d_min, device=device, verbose=0, cif=args.cif
    )
    model_light = load_model(
        args.light_model, max_res=d_min, device=device, verbose=0, cif=args.cif
    )

    if args.verbose >= 1:
        print(f"  Dark model: {len(model_dark.pdb)} atoms")
        print(f"  Light model: {len(model_light.pdb)} atoms")
        print(f"  Fraction: {args.fraction}")

    # Build ModelCollection + CollectionScaler
    f = args.fraction
    mc = ModelCollection([model_dark, model_light], dark_key="dark")
    mc.add_dark()
    mc.add_timepoint("light", fractions=[1.0 - f, f])
    dark_model = mc.dark_model
    mixed_model = mc["light"]

    if args.verbose >= 1:
        print(f"\n--- Setting up joint scaler ---")

    scaler = setup_collection_scaler(collection, mc, device, verbose=args.verbose)

    if args.verbose >= 1:
        r_work_d, r_free_d = compute_rfactors(dark_model, data_dark, scaler)
        r_work_m, r_free_m = compute_rfactors(mixed_model, data_light, scaler)
        print(f"  R-factor (dark):  R_work={r_work_d:.4f}  R_free={r_free_d:.4f}")
        print(f"  R-factor (mixed): R_work={r_work_m:.4f}  R_free={r_free_m:.4f}")

    # ------------------------------------------------------------------
    # 3. P1 expansion and grid setup
    # ------------------------------------------------------------------
    if args.verbose >= 1:
        print(f"\n--- Setting up P1 expansion and grid ---")

    gridsize = calculate_optimal_grid_size(cell_t, d_min, sg_name)
    if args.verbose >= 1:
        print(f"  Grid size: {gridsize}")

    # Expand HKL to P1
    hkl_p1, orig_idx, phase_shifts = expand_hkl(
        hkl, sg_name, include_friedel=False, remove_absences=True
    )
    w_dfo_p1 = w_dfo[orig_idx]
    weights_p1 = weights[orig_idx]

    if args.verbose >= 1:
        print(f"  ASU reflections: {len(hkl)}")
        print(f"  P1 reflections: {len(hkl_p1)}")

    # ------------------------------------------------------------------
    # 4. Compute Fcalc at ASU level, then expand to P1 via symmetry
    # ------------------------------------------------------------------
    if args.verbose >= 1:
        print(f"\n--- Computing structure factors ---")

    # Compute scaled Fcalc at ASU level using joint scaler with per-model solvent.
    with torch.no_grad():
        fcalc_dark_asu = scaler.forward_mixed(
            dark_model(hkl_all, recalc=True), dark_model.fractions
        )[mask]
        fcalc_mixed_asu = scaler.forward_mixed(
            mixed_model(hkl_all, recalc=True), mixed_model.fractions
        )[mask]

    # Expand to P1 using symmetry phase shifts (exact, no FFT needed)
    phase_factors = torch.exp(1j * phase_shifts.to(fcalc_dark_asu.dtype))
    fcalc_dark_p1 = fcalc_dark_asu[orig_idx] * phase_factors
    fcalc_mixed_p1 = fcalc_mixed_asu[orig_idx] * phase_factors

    delta_fcalc = fcalc_mixed_p1.abs() - fcalc_dark_p1.abs()
    w_delta_fcalc = delta_fcalc * weights_p1

    # Dark-state phases (from scaled Fcalc)
    phi_dark_p1 = torch.angle(fcalc_dark_p1)

    if args.verbose >= 1:
        print(f"  |DFcalc| mean: {delta_fcalc.abs().mean():.3f}")
        print(f"  |WDFcalc| mean: {w_delta_fcalc.abs().mean():.3f}")
        print(f"  WDFo mean: {w_dfo_p1.mean():.3f}, std: {w_dfo_p1.std():.3f}")

    # ------------------------------------------------------------------
    # 5. Compute maps (both weighted)
    # ------------------------------------------------------------------
    if args.verbose >= 1:
        print(f"\n--- Computing maps ---")

    with torch.no_grad():
        map_dfo = compute_map_from_coefficients(
            w_dfo_p1, phi_dark_p1, hkl_p1, gridsize
        )
        map_dfc = compute_map_from_coefficients(
            w_delta_fcalc, phi_dark_p1, hkl_p1, gridsize
        )

    # Normalize maps to sigma units (zero mean, unit variance)
    map_dfo = (map_dfo - map_dfo.mean()) / (map_dfo.std() + 1e-12)
    map_dfc = (map_dfc - map_dfc.mean()) / (map_dfc.std() + 1e-12)

    if args.verbose >= 1:
        print(f"  WDFo map: shape={map_dfo.shape} (sigma-normalized)")
        print(f"  WDFc map: shape={map_dfc.shape} (sigma-normalized)")

    # ------------------------------------------------------------------
    # 6. Build masks
    # ------------------------------------------------------------------
    mask_dict = {}
    mask_dict["full_cell"] = torch.ones(map_dfo.shape, dtype=torch.bool, device=device)

    if args.selection and args.selection.strip():
        if args.verbose >= 1:
            print(f'\n--- Building masks for "{args.selection}" ---')

        # Build atom positions for mask based on --mask-source
        xyz_parts = []
        if args.mask_source in ("dark", "both"):
            dark_atom_mask = model_dark.get_selection_mask(args.selection)
            xyz_parts.append(model_dark.xyz()[dark_atom_mask])
            if args.verbose >= 1:
                print(f"  Selected atoms (dark):  {dark_atom_mask.sum().item()}")
        if args.mask_source in ("light", "both"):
            light_atom_mask = model_light.get_selection_mask(args.selection)
            xyz_parts.append(model_light.xyz()[light_atom_mask])
            if args.verbose >= 1:
                print(f"  Selected atoms (light): {light_atom_mask.sum().item()}")
        selected_xyz = torch.cat(xyz_parts, dim=0)

        if args.verbose >= 1:
            print(f"  Mask source: {args.mask_source}")
            print(f"  Total atoms for mask: {len(selected_xyz)}")

        real_space_grid = get_real_grid(
            cell_t, max_res=d_min, gridsize=torch.tensor(gridsize), device=device
        )

        # Selection mask from combined dark + light positions
        mask_sel = build_atom_mask(
            selected_xyz, real_space_grid, cell_t, args.mask_radius, device
        )
        mask_dict["selection"] = mask_sel
        if args.verbose >= 1:
            print(f"  Selection voxels: {mask_sel.sum().item()}")


    # ------------------------------------------------------------------
    # 7. Compute real-space correlations
    # ------------------------------------------------------------------
    if args.verbose >= 1:
        print(f"\n--- Real-space correlations ---")

    rs_corr = {}
    for name, m in mask_dict.items():
        cc = compute_correlation(map_dfo, map_dfc, m)
        rs_corr[name] = {"cc": cc, "n_voxels": m.sum().item()}
        if args.verbose >= 1:
            print(f"  {name:>15s}: CC = {cc:.4f}  (n_vox = {m.sum().item()})")

    # ------------------------------------------------------------------
    # 8. Resolution-binned reciprocal-space correlation
    # ------------------------------------------------------------------
    if args.verbose >= 1:
        print(f"\n--- Resolution-binned amplitude correlation ---")

    # Reuse ASU-level scaled Fcalc from step 4
    delta_fcalc_asu = fcalc_mixed_asu.abs() - fcalc_dark_asu.abs()
    w_delta_fcalc_asu = delta_fcalc_asu * weights

    d_spacing = torch.tensor(d_spacings, dtype=torch.float32, device=device)
    n_bins = args.n_bins
    d_sorted, sort_idx = d_spacing.sort(descending=True)
    bin_size = len(d_spacing) // n_bins

    bin_results = []
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(d_spacing)
        idx = sort_idx[start:end]

        obs_bin = w_dfo[idx]
        calc_bin = w_delta_fcalc_asu[idx]
        ao = obs_bin - obs_bin.mean()
        ac = calc_bin - calc_bin.mean()
        cc_amp = (ao * ac).sum() / (
            torch.sqrt((ao**2).sum() * (ac**2).sum()) + 1e-12
        )

        d_lo = d_spacing[idx].max().item()
        d_hi = d_spacing[idx].min().item()

        bin_results.append(
            {
                "bin": i,
                "d_min": round(d_hi, 3),
                "d_max": round(d_lo, 3),
                "n_refl": len(idx),
                "cc": round(cc_amp.item(), 4),
            }
        )
        if args.verbose >= 1:
            print(
                f"  Bin {i:2d}: {d_lo:5.2f}-{d_hi:5.2f} A  "
                f"n={len(idx):5d}  CC={cc_amp.item():.4f}"
            )

    # Overall reciprocal-space CC
    cc_overall = torch.corrcoef(torch.stack([w_dfo, w_delta_fcalc_asu]))[0, 1].item()
    if args.verbose >= 1:
        print(f"  Overall CC(WDFo, WDFcalc) = {cc_overall:.4f}")

    # ------------------------------------------------------------------
    # 9. Assemble results
    # ------------------------------------------------------------------
    results = {
        "input": {
            "dark_structure_factor": str(args.dark_structure_factor),
            "light_structure_factor": str(args.light_structure_factor),
            "dark_model": str(args.dark_model),
            "light_model": str(args.light_model),
            "fraction": args.fraction,
            "selection": args.selection,
            "mask_radius": args.mask_radius,
            "dmin": d_min,
        },
        "realspace_correlation": rs_corr,
        "reciprocal_cc_overall": round(cc_overall, 4),
        "resolution_bins": bin_results,
        "map_statistics": {
            "WDFo": {
                "std": round(float(map_dfo.std()), 6),
                "mean": round(float(map_dfo.mean()), 6),
            },
            "WDFc": {
                "std": round(float(map_dfc.std()), 6),
                "mean": round(float(map_dfc.mean()), 6),
            },
        },
        "n_reflections_asu": len(hkl),
        "n_reflections_p1": len(hkl_p1),
    }

    # Write JSON
    json_path = outdir / "validate_ded_results.json"
    with open(json_path, "w") as fh:
        json.dump(convert_to_serializable(results), fh, indent=2)
    if args.verbose >= 1:
        print(f"\nResults saved to {json_path}")

    # ------------------------------------------------------------------
    # 10. Optional: write maps
    # ------------------------------------------------------------------
    if args.write_maps:
        from torchref.io.cif import write_map

        dfo_path = outdir / "validate_WDFo.ccp4"
        dfc_path = outdir / "validate_WDFc.ccp4"
        write_map(map_dfo, cell_np, str(dfo_path), spacegroup="P1")
        write_map(map_dfc, cell_np, str(dfc_path), spacegroup="P1")
        if args.verbose >= 1:
            print(f"  WDFo map: {dfo_path}")
            print(f"  WDFc map: {dfc_path}")

    # ------------------------------------------------------------------
    # 11. Optional: plots
    # ------------------------------------------------------------------
    if args.plot:
        generate_plots(results, map_dfo, map_dfc, mask_dict, outdir, args.verbose)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if args.verbose >= 1:
        print(f"\n{'=' * 70}")
        print("Summary:")
        for name, corr in rs_corr.items():
            print(f"  {name}: CC = {corr['cc']:.4f}")
        print(f"  Reciprocal-space CC (overall): {cc_overall:.4f}")
        print(f"{'=' * 70}")

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Validate difference electron density by correlating "
        "weighted DFo and DFcalc maps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  torchref.validate-ded -dsf dark.mtz -lsf light.mtz \\
      -dm dark.pdb -lm light.pdb

  # With fraction and ligand masking (mask from both models by default)
  torchref.validate-ded -dsf dark.mtz -lsf light.mtz \\
      -dm dark.pdb -lm light.pdb \\
      --fraction 0.20 --selection "chain B and resname IBL" --mask-radius 2.5

  # Mask from light model only
  torchref.validate-ded -dsf dark.mtz -lsf light.mtz \\
      -dm dark.pdb -lm light.pdb \\
      --fraction 0.20 --selection "resname IBL" --mask-source light

  # Full output
  torchref.validate-ded -dsf dark.mtz -lsf light.mtz \\
      -dm dark.pdb -lm light.pdb \\
      --fraction 0.20 --selection "resname IBL" --plot --write-maps -o validation/
        """,
    )

    # --- Input files (creates "Input files" and "Column selection" groups) ---
    add_dual_model_args(parser, fraction_required=False, fraction_default=1.0)

    output = parser.add_argument_group("Output")
    add_outdir_arg(
        output,
        required=False,
        default=".",
        help="Output directory (default: current directory)",
    )

    # --- Validation options ---
    analysis = parser.add_argument_group("Analysis")
    analysis.add_argument(
        "--selection",
        type=str,
        default=None,
        help='Phenix-style atom selection for masking '
        '(e.g., "chain B and resname IBL")',
    )
    analysis.add_argument(
        "--mask-source",
        type=str,
        default="both",
        choices=["light", "dark", "both"],
        help="Which model(s) to use for building the atom selection mask. "
        "'light' uses only the light/triggered model, 'dark' uses only the "
        "dark/reference model, 'both' combines atoms from both models "
        "(default: both).",
    )
    analysis.add_argument(
        "--mask-radius",
        type=float,
        default=2.5,
        help="Mask sphere radius in Angstroms (default: 2.5)",
    )
    analysis.add_argument(
        "--n-bins",
        type=int,
        default=20,
        help="Number of resolution bins (default: 20)",
    )
    analysis.add_argument(
        "--plot",
        action="store_true",
        help="Generate matplotlib validation plots (PNG + PDF)",
    )
    analysis.add_argument(
        "--write-maps",
        action="store_true",
        help="Write CCP4 map files for WDFo and WDFcalc",
    )

    res = parser.add_argument_group("Resolution")
    add_dmin_arg(res)

    add_general_args(parser)

    args = parser.parse_args()

    register_timing()

    # Validate input files exist
    if validate_files([
        (args.dark_structure_factor, "--dark-structure-factor"),
        (args.light_structure_factor, "--light-structure-factor"),
        (args.dark_model, "--dark-model"),
        (args.light_model, "--light-model"),
    ]):
        return 1

    if validate_cif_files(args.cif):
        return 1

    return run_validation(args)


if __name__ == "__main__":
    sys.exit(main())
