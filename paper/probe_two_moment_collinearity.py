#!/usr/bin/env python
"""How much of the activation-dispersion signal can the scale model absorb?

The contamination ``sigma_alpha^2 |dF/dalpha|^2`` is smooth, strictly positive and
concentrated at low resolution -- the same shape a scale or overall-B error takes. If the
scale parameters can reproduce it, a refined ``lambda`` is measuring scale error rather
than activation heterogeneity, and no amount of refinement will tell the two apart.

This is answerable exactly, by linear algebra, with no refinement at all. Whiten every
quantity by the measurement error, treat the contamination as a template ``t`` and the
scale parameters' derivatives as a design matrix ``X``, and project::

    P    = X (X'X)^-1 X'
    leak = ||P t|| / ||t||          fraction of the template the scale model can absorb
    vif  = ||t|| / ||t - P t||      how much the surviving signal is degraded

Two designs are compared, because they are the two places scale is fitted:

* **per-dataset** -- the ``log_scale`` + ``U_aniso`` that ``DatasetCollection.scale()``
  fits on the light dataset alone. Free to shape the light data however it likes.
* **shared** -- the ``CollectionScaler`` parameters, which are fitted jointly against dark
  and light. One column per parameter spanning *both* datasets, so a light-only template
  cannot be matched without spoiling the dark.

The derivatives are taken numerically from the live scaler rather than from textbook
formulae, so the design matrix is the parameterisation actually in use.

The template is then split into a smooth resolution envelope and the residual speckle,
and each projected separately: the envelope is what a scale model can absorb, the speckle
is what identifies the dispersion. Which half survives decides how a fitted lambda should
be read.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def build(dark_sf, light_sf, dark_pdb, light_pdb, cif, d_min, fraction, lam, device):
    """Rebuild the figure-4 collection exactly as the CLI does."""
    from torchref.cli.collection_difference_refine import (
        setup_dataset_collection,
        setup_model_collection,
        setup_scaler,
    )

    mc = setup_model_collection(
        dark_pdb, light_pdb, [1.0 - fraction, fraction], cif, d_min, device, 0
    )
    dc = setup_dataset_collection(dark_sf, light_sf, d_min, device)
    scaler = setup_scaler(dc, mc, device, verbose=0)
    mc.set_lambda_twin(lam)
    return dc, mc, scaler


def light_intensity(dc, mc, scaler):
    """Scaled model intensity for the light dataset, shape (n_hkl,)."""
    keys = [mc.dark_key, "light"]
    rows = [mc.keys().index(k) for k in keys]
    comps = dc.component_structure_factors(mc, recalc=True)
    w = mc.fractions_matrix()[rows]
    return scaler.forward_batched(mc.mix_component_fcalcs(comps, w), w)[1].abs() ** 2


def contamination(dc, mc, scaler):
    """sigma_alpha^2 |dF/dalpha|^2 on the light dataset."""
    keys = [mc.dark_key, "light"]
    rows = [mc.keys().index(k) for k in keys]
    comps = dc.component_structure_factors(mc, recalc=True)
    jac = mc.activation_jacobian()[rows]
    deriv = scaler.forward_batched(mc.mix_component_fcalcs(comps, jac), jac)[1]
    return mc.sigma_alpha_sq * deriv.abs() ** 2


def numeric_columns(params, evaluate, rel_step=1e-3):
    """d(model intensity)/d(theta) for every scalar in `params`, by central difference."""
    cols = []
    for p in params:
        flat = p.detach().reshape(-1)
        for i in range(flat.numel()):
            step = rel_step * max(abs(float(flat[i])), 1e-3)
            saved = float(flat[i])
            # No autograd: these are numerical derivatives, and retaining a graph per
            # evaluation is what pushed this over the memory limit.
            with torch.no_grad():
                flat[i] = saved + step
                plus = evaluate()
                flat[i] = saved - step
                minus = evaluate()
                flat[i] = saved
                col = ((plus - minus) / (2 * step)).cpu().numpy()
            del plus, minus
            cols.append(col)
    return np.asarray(cols).T  # (n_hkl, n_param)


def leakage(t, X):
    """(leak, vif) for template `t` against design `X`, both already whitened."""
    keep = ~np.any(~np.isfinite(X), axis=1) & np.isfinite(t)
    t, X = t[keep], X[keep]
    # Drop null columns, then least-squares project (lstsq handles rank deficiency).
    good = np.linalg.norm(X, axis=0) > 0
    X = X[:, good]
    if X.shape[1] == 0:
        return 0.0, 1.0
    coef, *_ = np.linalg.lstsq(X, t, rcond=None)
    fit = X @ coef
    nt = np.linalg.norm(t)
    resid = np.linalg.norm(t - fit)
    return float(np.linalg.norm(fit) / nt), float(nt / max(resid, 1e-30))


def envelope_and_speckle(t, res, nbin=30):
    """Split a template into its smooth resolution envelope and the residual."""
    order = np.argsort(-res)
    env = np.zeros_like(t)
    for chunk in np.array_split(order, nbin):
        env[chunk] = t[chunk].mean()
    return env, t - env


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    fig4 = Path(__file__).resolve().parent / "figure4_difference_refinement"
    ap.add_argument("--dark-sf", default=str(fig4 / "data/8QL2-sf.cif"))
    ap.add_argument("--light-sf", default=str(fig4 / "data/7YYZ-light.mtz"))
    ap.add_argument("--dark-pdb", default=str(fig4 / "data/8QL2_no_altloc.pdb"))
    ap.add_argument("--light-pdb", default=str(fig4 / "work_no_altloc.pdb"))
    ap.add_argument("--cif", nargs="*", default=[str(fig4 / "data/IBL_grade.cif")])
    ap.add_argument("--dmin", type=float, default=2.2)
    ap.add_argument("--fraction", type=float, default=0.22)
    ap.add_argument("--lambda-twin", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device)
    print("Rebuilding the collection...", flush=True)
    dc, mc, scaler = build(
        args.dark_sf, args.light_sf, args.dark_pdb, args.light_pdb,
        args.cif, args.dmin, args.fraction, args.lambda_twin, dev,
    )
    light = dc["light"]

    with torch.no_grad():
        t_raw = contamination(dc, mc, scaler).cpu().numpy()
        _, sig_I = light.get_corrected_intensities()
        sig = sig_I.cpu().numpy()
        res = light.resolution.cpu().numpy()
        mask = light.masks().cpu().numpy().astype(bool)

    ok = mask & np.isfinite(sig) & (sig > 0) & np.isfinite(t_raw) & np.isfinite(res)
    print(f"reflections: {ok.sum()} of {len(ok)}")

    # Whiten: everything is measured in units of the error it has to beat.
    t = (t_raw / sig)[ok]

    # --- design matrices, differentiated numerically ---
    #
    # Two different response functions, because the two parameter sets act on opposite
    # sides of the residual. The shared scaler shapes the *model* intensity; the
    # per-dataset log_scale / U_aniso shape the *observed* one. Using the model response
    # for both would give the dataset parameters an identically zero column and report
    # no leak at all.
    #
    # The component structure factors are computed once: no parameter here moves an
    # atom, so recomputing them per derivative is ~50x of pure waste.
    print("Differentiating the scale model...", flush=True)
    keys = [mc.dark_key, "light"]
    rows = [mc.keys().index(k) for k in keys]
    with torch.no_grad():
        comps = dc.component_structure_factors(mc, recalc=True)

    def model_response():
        w = mc.fractions_matrix()[rows]
        return scaler.forward_batched(
            mc.mix_component_fcalcs(comps, w), w
        )[1].abs() ** 2

    def obs_response():
        light._corrected_I_fp = None
        return light.get_corrected_intensities()[0]

    shared_params = list(scaler.parameters())
    X_shared = numeric_columns(shared_params, model_response)[ok] / sig[ok, None]

    data_params = [p for p in light.parameters() if p is not None]
    X_data = numeric_columns(data_params, obs_response)[ok] / sig[ok, None]

    env, speck = envelope_and_speckle(t, res[ok])

    rows = []
    for tname, tv in (("full", t), ("envelope", env), ("speckle", speck)):
        for xname, X in (("per-dataset", X_data), ("shared", X_shared)):
            leak, vif = leakage(tv, X)
            rows.append(dict(template=tname, design=xname, n_param=X.shape[1],
                             leak=leak, vif=vif))

    print()
    print("Fraction of the activation template the scale model can absorb")
    print("=" * 66)
    print(f"{'template':10s} {'design':13s} {'n_param':>8s} {'leak':>8s} {'VIF':>8s}")
    print("-" * 66)
    for r in rows:
        print(f"{r['template']:10s} {r['design']:13s} {r['n_param']:8d} "
              f"{r['leak']:8.3f} {r['vif']:8.2f}")
    print("-" * 66)
    print(f"envelope carries {np.linalg.norm(env) / np.linalg.norm(t):.3f} of the "
          f"template norm, speckle {np.linalg.norm(speck) / np.linalg.norm(t):.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
