#!/usr/bin/env python3 -u
"""Validate difference electron density (DED) by correlating DFo and DFc maps.

Takes separate dark and light MTZ files, computes weighted difference amplitudes
internally, then compares the weighted DFo and DFcalc maps using dark-state phases.
Phenix-style atom selections give regional correlations, e.g. around a ligand site.

Examples
--------
::

    torchref.validate-ded -dsf dark.mtz -lsf light.mtz -dm dark.pdb -lm light.pdb \\
        --fraction 0.20 --selection "chain B and resname IBL" --mask-radius 2.5 \\
        --mask-source light --plot --write-maps -o validation/

Or programmatically::

    ctx = setup_ded_context("dark.mtz", "light.mtz", dmin=2.2)
    result = compute_ded_maps(ctx, model_dark, model_light, fraction=0.18,
                              selection="resname IBL", mask_radius=2.5)
    print(result["reciprocal_cc_overall"])
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
    parse_device_str,
    validate_cif_files,
    validate_files,
)
from torchref.utils.serialization import convert_to_serializable

configure_unbuffered_output()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def build_atom_mask(selection_xyz, real_space_grid, cell, mask_radius, device):
    """Boolean voxel mask of shape ``grid_shape[:3]``, True within ``mask_radius``
    Angstrom of any atom in ``selection_xyz`` ``(N, 3)``.

    ``real_space_grid`` comes from ``get_real_grid()`` and ``cell`` may be a ``Cell`` or the
    raw parameter tensor.
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
    """Real-space 3D map from Fourier coefficients on a ``gridsize`` grid.

    Takes ``amplitudes`` ``(N,)`` (which may be signed), ``phases_rad`` ``(N,)`` and
    P1-expanded ``hkl_p1`` ``(N, 3)``.
    """
    from torchref.base.reciprocal.grid_operations import place_on_grid

    coefficients = amplitudes * torch.exp(1j * phases_rad)
    grid = place_on_grid(hkl_p1, coefficients, gridsize, enforce_hermitian=True)
    return torch.fft.fftn(grid, dim=(0, 1, 2), norm="forward").real


def generate_plots(results, map_dfo, map_dfc, mask_dict, outdir, verbose):
    """Write the 2-panel validation figure for ``results`` into ``outdir``.

    ``map_dfo``/``map_dfc`` are the real-space maps and ``mask_dict`` maps region name to a
    boolean mask.
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
# Reusable core functions
# ---------------------------------------------------------------------------


def setup_ded_context(
    dark_sf,
    light_sf,
    dmin=None,
    device=None,
    col_dark=None,
    col_light=None,
    n_bins=20,
    verbose=0,
):
    """Load reflection data and prepare shared state for DED validation.

    This sets up the observation side (weighted DFo, P1 expansion, resolution
    bins, free/work masks) that is independent of any particular model.

    Parameters
    ----------
    dark_sf, light_sf : str or Path
        Paths to dark and light structure factor files.
    dmin : float, optional
        High-resolution cutoff in Angstroms.
    device : torch.device, optional
        Compute device. Defaults to CPU.
    col_dark, col_light : dict, optional
        Column name overrides for data loading.
    n_bins : int
        Number of resolution bins for reciprocal-space CC (default 20).
    verbose : int
        Verbosity level.

    Returns
    -------
    dict
        Context dictionary with keys: device, collection, data_dark,
        data_light, hkl_all, hkl, refl_mask, w_dfo, weights, d_spacing,
        cell_t, cell_np, sg_name, d_min, gridsize, hkl_p1, orig_idx,
        phase_shifts, w_dfo_p1, weights_p1, work_mask, free_mask.
    """
    import gemmi

    from torchref import DatasetCollection
    from torchref.symmetry.grid_utils import calculate_optimal_grid_size
    from torchref.symmetry.reciprocal_symmetry import expand_hkl

    from torchref.config import normalize_device

    device = normalize_device(device)

    # Load and scale
    data_dark = load_reflection_data(
        str(dark_sf), device=device, column_names=col_dark, verbose=0
    )
    data_light = load_reflection_data(
        str(light_sf), device=device, column_names=col_light, verbose=0
    )
    if dmin is not None:
        data_dark.cut_res(highres=dmin)
        data_light.cut_res(highres=dmin)

    collection = DatasetCollection(device=str(device))
    collection.add_dataset("dark", data_dark)
    collection.add_dataset("light", data_light)
    collection.scale()

    if verbose >= 1:
        print(f"Scale parameters after optimization:")
        for name, ds in collection:
            if hasattr(ds, "log_scale") and ds.log_scale is not None:
                print(
                    f"  {name}: log_scale={ds.log_scale.item():.6f} "
                    f"(scale={torch.exp(ds.log_scale).item():.6f})"
                )

    # Extract matched reflections
    hkl_all = data_dark.hkl
    F_dark_full, sig_dark_full = data_dark.get_corrected_data()
    rfree_dark = data_dark.rfree_flags
    F_light_full, sig_light_full = data_light.get_corrected_data()
    rfree_light = data_light.rfree_flags
    refl_mask = data_dark.masks().to(torch.bool) & data_light.masks().to(torch.bool)

    F_dark = F_dark_full[refl_mask]
    F_light = F_light_full[refl_mask]
    sig_dark = sig_dark_full[refl_mask]
    sig_light = sig_light_full[refl_mask]
    hkl = hkl_all[refl_mask]

    # Free/work masks
    _has_rfree = rfree_dark is not None or rfree_light is not None
    if _has_rfree:
        rfree_flags = (
            rfree_dark[refl_mask].bool()
            if rfree_dark is not None
            else rfree_light[refl_mask].bool()
        )
        work_mask = rfree_flags
        free_mask = ~rfree_flags
    else:
        free_mask = work_mask = None

    # Weighted difference Fo
    dfo = F_light - F_dark
    sig_diff = (sig_dark**2 + sig_light**2) ** 0.5
    weights = 1 / sig_diff**2
    weights = weights / weights.mean()
    w_dfo = dfo * weights

    # Cell, spacegroup, d-spacings
    cell_t = data_dark.cell.data.to(device)
    cell_np = cell_t.cpu().numpy()
    sg_name = data_dark.spacegroup.hm

    gemmi_cell = gemmi.UnitCell(*cell_np.tolist())
    hkl_np = hkl.cpu().numpy().astype(np.int32)
    d_spacings = np.array(
        [gemmi_cell.calculate_d(h) for h in hkl_np.tolist()], dtype=np.float32
    )
    if dmin is None:
        dmin = float(d_spacings.min())
    d_spacing = torch.tensor(d_spacings, dtype=torch.float32, device=device)

    # P1 expansion and grid
    gridsize = calculate_optimal_grid_size(cell_t, dmin, sg_name)
    hkl_p1, orig_idx, phase_shifts = expand_hkl(
        hkl, sg_name, include_friedel=False, remove_absences=True
    )
    w_dfo_p1 = w_dfo[orig_idx]
    weights_p1 = weights[orig_idx]

    if verbose >= 1:
        print(f"Matched reflections: {len(hkl)}")
        print(f"Cell: {cell_np}, Spacegroup: {sg_name}")
        print(f"Resolution: {dmin:.2f} A, Grid: {gridsize}")
        print(f"ASU: {len(hkl)}, P1: {len(hkl_p1)}")

    return {
        "device": device,
        "collection": collection,
        "data_dark": data_dark,
        "data_light": data_light,
        "hkl_all": hkl_all,
        "hkl": hkl,
        "refl_mask": refl_mask,
        "w_dfo": w_dfo,
        "weights": weights,
        "d_spacing": d_spacing,
        "cell_t": cell_t,
        "cell_np": cell_np,
        "sg_name": sg_name,
        "d_min": dmin,
        "gridsize": gridsize,
        "hkl_p1": hkl_p1,
        "orig_idx": orig_idx,
        "phase_shifts": phase_shifts,
        "w_dfo_p1": w_dfo_p1,
        "weights_p1": weights_p1,
        "work_mask": work_mask,
        "free_mask": free_mask,
        "n_bins": n_bins,
    }


def compute_ded_maps(
    ctx,
    model_dark,
    model_light,
    fraction,
    selection=None,
    mask_source="both",
    mask_radius=2.5,
    n_bins=None,
    verbose=0,
):
    """Compute DED maps and correlations for one (dark, light) model pair.

    Parameters
    ----------
    ctx : dict
        Context from :func:`setup_ded_context`.
    model_dark, model_light : ModelFT
        Dark and light atomic models.
    fraction : float
        Activated fraction for the light state.
    selection : str, optional
        Phenix-style atom selection for masked correlation.
    mask_source : str
        Which model(s) build the selection mask: "dark", "light", or "both".
    mask_radius : float
        Mask sphere radius in Angstroms.
    n_bins : int, optional
        Number of resolution bins. Falls back to ``ctx["n_bins"]``.
    verbose : int
        Verbosity level.

    Returns
    -------
    dict
        Keys: map_dfo, map_dfc, mask_dict, realspace_correlation,
        resolution_bins, reciprocal_cc_overall, reciprocal_cc_work,
        reciprocal_cc_free, w_delta_fcalc_asu.
    """
    from torchref.model.model_collection import ModelCollection
    from torchref.cli.collection_difference_refine import (
        setup_scaler as setup_collection_scaler,
    )
    from torchref.base.fourier.grid import get_real_grid

    device = ctx["device"]

    # Build ModelCollection + scaler
    mc = ModelCollection([model_dark, model_light], dark_key="dark")
    mc.add_dark()
    mc.add_timepoint("light", fractions=[1.0 - fraction, fraction])
    dark_mm = mc.dark_model
    mixed_mm = mc["light"]

    scaler = setup_collection_scaler(ctx["collection"], mc, device, verbose=0)

    # Fcalc at ASU level
    with torch.no_grad():
        fcalc_dark_asu = scaler.forward_mixed(
            dark_mm(ctx["hkl_all"], recalc=True), dark_mm.fractions
        )[ctx["refl_mask"]]
        fcalc_mixed_asu = scaler.forward_mixed(
            mixed_mm(ctx["hkl_all"], recalc=True), mixed_mm.fractions
        )[ctx["refl_mask"]]

    # Expand to P1
    phase_factors = torch.exp(
        1j * ctx["phase_shifts"].to(fcalc_dark_asu.dtype)
    )
    fcalc_dark_p1 = fcalc_dark_asu[ctx["orig_idx"]] * phase_factors
    fcalc_mixed_p1 = fcalc_mixed_asu[ctx["orig_idx"]] * phase_factors

    delta_fcalc = fcalc_mixed_p1.abs() - fcalc_dark_p1.abs()
    w_delta_fcalc = delta_fcalc * ctx["weights_p1"]
    phi_dark_p1 = torch.angle(fcalc_dark_p1)

    # ASU-level weighted DFcalc
    delta_fcalc_asu = fcalc_mixed_asu.abs() - fcalc_dark_asu.abs()
    w_delta_fcalc_asu = delta_fcalc_asu * ctx["weights"]

    if verbose >= 1:
        print(f"  |DFcalc| mean: {delta_fcalc.abs().mean():.3f}")
        print(f"  |WDFcalc| mean: {w_delta_fcalc.abs().mean():.3f}")

    # Compute maps
    with torch.no_grad():
        map_dfo = compute_map_from_coefficients(
            ctx["w_dfo_p1"], phi_dark_p1, ctx["hkl_p1"], ctx["gridsize"]
        )
        map_dfc = compute_map_from_coefficients(
            w_delta_fcalc, phi_dark_p1, ctx["hkl_p1"], ctx["gridsize"]
        )

    map_dfo = (map_dfo - map_dfo.mean()) / (map_dfo.std() + 1e-12)
    map_dfc = (map_dfc - map_dfc.mean()) / (map_dfc.std() + 1e-12)

    # Build masks
    mask_dict = {
        "full_cell": torch.ones(map_dfo.shape, dtype=torch.bool, device=device)
    }
    if selection and selection.strip():
        xyz_parts = []
        if mask_source in ("dark", "both"):
            sel = model_dark.get_selection_mask(selection)
            xyz_parts.append(model_dark.xyz()[sel])
        if mask_source in ("light", "both"):
            sel = model_light.get_selection_mask(selection)
            xyz_parts.append(model_light.xyz()[sel])
        selected_xyz = torch.cat(xyz_parts, dim=0)

        real_space_grid = get_real_grid(
            ctx["cell_t"],
            max_res=ctx["d_min"],
            gridsize=torch.tensor(ctx["gridsize"]),
            device=device,
        )
        mask_dict["selection"] = build_atom_mask(
            selected_xyz, real_space_grid, ctx["cell_t"], mask_radius, device
        )

    # Real-space correlations
    rs_corr = {}
    for name, m in mask_dict.items():
        cc = compute_correlation(map_dfo, map_dfc, m)
        rs_corr[name] = {"cc": cc, "n_voxels": m.sum().item()}
        if verbose >= 1:
            print(f"  {name:>15s}: CC = {cc:.4f}  (n_vox = {m.sum().item()})")

    # Resolution-binned CC
    if n_bins is None:
        n_bins = ctx.get("n_bins", 20)
    d_sorted, sort_idx = ctx["d_spacing"].sort(descending=True)
    bin_size = len(ctx["d_spacing"]) // n_bins
    bin_results = []
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(ctx["d_spacing"])
        idx = sort_idx[start:end]

        obs_bin = ctx["w_dfo"][idx]
        calc_bin = w_delta_fcalc_asu[idx]
        ao = obs_bin - obs_bin.mean()
        ac = calc_bin - calc_bin.mean()
        cc_amp = (ao * ac).sum() / (
            torch.sqrt((ao**2).sum() * (ac**2).sum()) + 1e-12
        )

        bin_results.append({
            "bin": i,
            "d_min": round(ctx["d_spacing"][idx].min().item(), 3),
            "d_max": round(ctx["d_spacing"][idx].max().item(), 3),
            "n_refl": len(idx),
            "cc": round(cc_amp.item(), 4),
        })
        if verbose >= 1:
            d_lo = ctx["d_spacing"][idx].max().item()
            d_hi = ctx["d_spacing"][idx].min().item()
            print(
                f"  Bin {i:2d}: {d_lo:5.2f}-{d_hi:5.2f} A  "
                f"n={len(idx):5d}  CC={cc_amp.item():.4f}"
            )

    # Overall and free/work CC
    cc_overall = torch.corrcoef(
        torch.stack([ctx["w_dfo"], w_delta_fcalc_asu])
    )[0, 1].item()

    cc_free = cc_work = None
    free_mask = ctx["free_mask"]
    work_mask = ctx["work_mask"]
    if free_mask is not None and free_mask.sum() > 10:
        cc_free = torch.corrcoef(
            torch.stack([ctx["w_dfo"][free_mask], w_delta_fcalc_asu[free_mask]])
        )[0, 1].item()
        cc_work = torch.corrcoef(
            torch.stack([ctx["w_dfo"][work_mask], w_delta_fcalc_asu[work_mask]])
        )[0, 1].item()

    if verbose >= 1:
        print(f"  Overall CC = {cc_overall:.4f}")
        if cc_work is not None:
            print(f"  Work CC = {cc_work:.4f}, Free CC = {cc_free:.4f}")

    return {
        "map_dfo": map_dfo,
        "map_dfc": map_dfc,
        "mask_dict": mask_dict,
        "realspace_correlation": rs_corr,
        "resolution_bins": bin_results,
        "reciprocal_cc_overall": round(cc_overall, 4),
        "reciprocal_cc_work": round(cc_work, 4) if cc_work is not None else None,
        "reciprocal_cc_free": round(cc_free, 4) if cc_free is not None else None,
        "w_delta_fcalc_asu": w_delta_fcalc_asu,
    }


# ---------------------------------------------------------------------------
# CLI validation pipeline (wraps the core functions above)
# ---------------------------------------------------------------------------


def run_validation(args):
    """Run the DED validation pipeline."""
    from torchref.cli.collection_difference_refine import compute_rfactors
    from torchref.model.model_collection import ModelCollection
    from torchref.cli.collection_difference_refine import (
        setup_scaler as setup_collection_scaler,
    )

    device = parse_device_str(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.verbose >= 1:
        print("=" * 70)
        print("DED Validation: WDFo vs WDFcalc")
        print("=" * 70)
        print(f"\n--- Loading reflection data ---")
        print(f"  Dark SF:   {args.dark_structure_factor}")
        print(f"  Light SF:  {args.light_structure_factor}")

    col_dark, col_light = build_dual_column_names(args)
    ctx = setup_ded_context(
        args.dark_structure_factor,
        args.light_structure_factor,
        dmin=args.dmin,
        device=device,
        col_dark=col_dark,
        col_light=col_light,
        n_bins=args.n_bins,
        verbose=args.verbose,
    )
    d_min = ctx["d_min"]

    # Load models
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

    # R-factors (verbose only, before DED computation)
    if args.verbose >= 1:
        print(f"\n--- Setting up joint scaler ---")
        f = args.fraction
        mc = ModelCollection([model_dark, model_light], dark_key="dark")
        mc.add_dark()
        mc.add_timepoint("light", fractions=[1.0 - f, f])
        scaler = setup_collection_scaler(
            ctx["collection"], mc, device, verbose=args.verbose
        )
        r_work_d, r_free_d = compute_rfactors(mc.dark_model, ctx["data_dark"], scaler)
        r_work_m, r_free_m = compute_rfactors(mc["light"], ctx["data_light"], scaler)
        print(f"  R-factor (dark):  R_work={r_work_d:.4f}  R_free={r_free_d:.4f}")
        print(f"  R-factor (mixed): R_work={r_work_m:.4f}  R_free={r_free_m:.4f}")

    # Core computation
    if args.verbose >= 1:
        print(f"\n--- Computing DED maps and correlations ---")

    result = compute_ded_maps(
        ctx,
        model_dark,
        model_light,
        fraction=args.fraction,
        selection=args.selection,
        mask_source=args.mask_source,
        mask_radius=args.mask_radius,
        n_bins=args.n_bins,
        verbose=args.verbose,
    )

    # Assemble output JSON
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
        "realspace_correlation": result["realspace_correlation"],
        "reciprocal_cc_overall": result["reciprocal_cc_overall"],
        "reciprocal_cc_work": result["reciprocal_cc_work"],
        "reciprocal_cc_free": result["reciprocal_cc_free"],
        "resolution_bins": result["resolution_bins"],
        "map_statistics": {
            "WDFo": {
                "std": round(float(result["map_dfo"].std()), 6),
                "mean": round(float(result["map_dfo"].mean()), 6),
            },
            "WDFc": {
                "std": round(float(result["map_dfc"].std()), 6),
                "mean": round(float(result["map_dfc"].mean()), 6),
            },
        },
        "n_reflections_asu": len(ctx["hkl"]),
        "n_reflections_p1": len(ctx["hkl_p1"]),
    }

    # Write JSON
    json_path = outdir / "validate_ded_results.json"
    with open(json_path, "w") as fh:
        json.dump(convert_to_serializable(results), fh, indent=2)
    if args.verbose >= 1:
        print(f"\nResults saved to {json_path}")

    # Optional: write maps
    if args.write_maps:
        from torchref.io.cif import write_map

        dfo_path = outdir / "validate_WDFo.ccp4"
        dfc_path = outdir / "validate_WDFc.ccp4"
        write_map(result["map_dfo"], ctx["cell_np"], str(dfo_path), spacegroup="P1")
        write_map(result["map_dfc"], ctx["cell_np"], str(dfc_path), spacegroup="P1")
        if args.verbose >= 1:
            print(f"  WDFo map: {dfo_path}")
            print(f"  WDFc map: {dfc_path}")

    # Optional: plots
    if args.plot:
        generate_plots(
            results, result["map_dfo"], result["map_dfc"],
            result["mask_dict"], outdir, args.verbose,
        )

    # Summary
    if args.verbose >= 1:
        print(f"\n{'=' * 70}")
        print("Summary:")
        for name, corr in result["realspace_correlation"].items():
            print(f"  {name}: CC = {corr['cc']:.4f}")
        print(f"  Reciprocal-space CC (overall): {result['reciprocal_cc_overall']}")
        print(f"{'=' * 70}")

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """Entry point for ``torchref.validate-ded``; returns the exit code."""
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
