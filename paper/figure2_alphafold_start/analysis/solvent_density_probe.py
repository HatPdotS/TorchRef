#!/usr/bin/env python
"""
A/B probe: density-derived bulk solvent vs the current vdW-sphere mask.

Standalone analysis -- does NOT wire DensitySolventModel into the production
Scaler/refinement. It scores every solvent variant under ONE identical
per-resolution-bin least-squares scale, so the comparison isolates the *solvent
model* (not differences in overall-scale/aniso parametrization).

Three things are reported per structure:

1. R-factor under a common per-bin scale for:
     - no solvent
     - current mask  (SolventModel.get_rec_solvent, raw mask SF)
     - density       (DensitySolventModel, fitted rho_s/rho0)
   demonstrating the co-refinement of {rho_s, rho0}.

2. rho0 sweep: correlation of density F_solv to (a) the current mask F_solv and
   (b) -F_protein, showing the continuous flat-mask <-> Babinet interpolation.

3. Amplitude-vs-resolution of the density F_solv vs the current mask.

Usage::

    python solvent_density_probe.py --pdb 1DAW.pdb --mtz 1DAW.mtz [--occupancy exp]

Run on SLURM (hour partition); write logs under /das, not /tmp.
"""

import argparse

import torch

from torchref import ModelFT, ReflectionData
from torchref.base import get_scattering_vectors
from torchref.scaling.scaler import Scaler
from torchref.scaling.density_solvent import DensitySolventModel


# ----------------------------------------------------------------------
# Common scoring: one per-resolution-bin real scale for every variant
# ----------------------------------------------------------------------
def resolution_bins(hkl, model, nbins=20):
    """Per-reflection resolution-bin index (equal-population bins in s)."""
    s = (
        torch.norm(
            get_scattering_vectors(hkl, model.cell, recB=model.recB), dim=1
        )
        / 2.0
    )
    order = torch.argsort(s)
    bin_idx = torch.empty_like(order)
    edges = torch.linspace(0, len(s), nbins + 1).long()
    for b in range(nbins):
        bin_idx[order[edges[b] : edges[b + 1]]] = b
    return s, bin_idx


def binwise_scaled_R(Fobs, Fmodel_unscaled, bin_idx, work, free, nbins=20):
    """
    Fit one positive real scale per resolution bin on the WORK set minimizing
    sum (|Fobs| - k*|Fmodel|)^2, then report R_work / R_free.

    Returns (R_work, R_free).
    """
    Aobs = Fobs.abs()
    Amod = Fmodel_unscaled.abs()
    k = torch.ones(nbins, device=Aobs.device, dtype=Aobs.dtype)
    for b in range(nbins):
        m = (bin_idx == b) & work
        denom = (Amod[m] ** 2).sum()
        if denom > 0:
            k[b] = (Aobs[m] * Amod[m]).sum() / denom
    Ascaled = k[bin_idx] * Amod

    def R(mask):
        return (
            (Aobs[mask] - Ascaled[mask]).abs().sum() / Aobs[mask].sum()
        ).item()

    return R(work), R(free)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--mtz", required=True)
    ap.add_argument("--occupancy", default="exp", choices=["exp", "sigmoid"])
    ap.add_argument("--residual-bsol", action="store_true",
                    help="co-refine a residual B_sol damping on the density solvent")
    ap.add_argument("--solvent-res", type=float, default=4.0)
    ap.add_argument("--max-res", type=float, default=2.0)
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--fit-steps", type=int, default=40)
    args = ap.parse_args()

    torch.manual_seed(0)

    # --- load data + model ---
    # Drive everything off data.hkl (raw amplitudes); we fit our own per-bin
    # scale below, so raw vs corrected Fobs is irrelevant to the comparison.
    data = ReflectionData()
    data.load_mtz(args.mtz)
    hkl = data.hkl.to(torch.long)
    Fobs = data.F.detach().clone().to(torch.get_default_dtype())
    rfree = data.rfree_flags
    # rfree convention: 1=work, 0=test (see memory: rfree_flag_convention)
    if rfree is not None:
        work = rfree.detach().to(torch.bool)
    else:
        work = torch.ones_like(Fobs, dtype=torch.bool)
    free = ~work

    model = ModelFT(max_res=args.max_res, radius_angstrom=4.0)
    model.load_pdb(args.pdb)
    model.setup_grid()

    Fprotein = model.get_structure_factor(hkl).detach()
    s, bin_idx = resolution_bins(hkl, model, args.nbins)

    # Align Fobs length to hkl_for_sf (anomalous expansion may differ); guard.
    n = min(len(Fobs), len(Fprotein))
    Fobs, Fprotein, bin_idx, work, free = (
        Fobs[:n], Fprotein[:n], bin_idx[:n], work[:n], free[:n],
    )
    hkl = hkl[:n]

    print(f"\n=== {args.pdb} | {len(Fprotein)} refl | occupancy={args.occupancy} ===")

    # --- production baseline: full scaler with the current vdW-sphere mask ---
    scaler = Scaler(model, data, nbins=args.nbins)
    scaler.initialize()
    scaler.refine_lbfgs(nsteps=5, verbose=False)
    Rw_base, Rf_base = scaler.rfactor()
    Rw_base, Rf_base = float(Rw_base), float(Rf_base)

    # Faithfully reconstruct the production mask solvent contribution
    # (k_sol * exp(-B_sol s^2) * phase * F_mask_raw) so it is scored on equal
    # footing with the density solvent under the common per-bin scale below.
    sv = scaler.solvent
    Fmask_raw = sv.get_rec_solvent(hkl).detach()[:n]
    k_sol = float(torch.exp(sv.log_k_solvent))
    s_half_sq = (
        torch.norm(get_scattering_vectors(hkl, model.cell, recB=model.recB), dim=1)
        / 2.0
    )[:n] ** 2
    bfac = torch.exp((-sv.b_solvent.detach() * s_half_sq).clamp(min=-50, max=50))
    Fsol_mask = k_sol * bfac * Fmask_raw
    if getattr(sv, "optimize_phase", False):
        Fsol_mask = Fsol_mask * torch.exp(1j * sv.phase_offset.detach())

    # --- density solvent ---
    sol = DensitySolventModel(
        model, occupancy=args.occupancy, solvent_res=args.solvent_res,
        residual_bsol=args.residual_bsol,
    )

    # ------------------------------------------------------------------
    # (1) R under a common per-bin scale: no solvent / mask / density(fitted)
    # ------------------------------------------------------------------
    Rw0, Rf0 = binwise_scaled_R(Fobs, Fprotein, bin_idx, work, free, args.nbins)
    Rwm, Rfm = binwise_scaled_R(
        Fobs, Fprotein + Fsol_mask, bin_idx, work, free, args.nbins
    )

    # co-refine rho_s, rho0 against the work-set per-bin-scaled LS
    opt = torch.optim.LBFGS(
        list(sol.parameters()), max_iter=args.fit_steps, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad()
        fsol = sol(hkl)[:n]
        Amod = (Fprotein + fsol).abs()
        Aobs = Fobs.abs()
        # per-bin scale computed on work set (detached so it acts as a target scale)
        loss = 0.0
        for b in range(args.nbins):
            m = (bin_idx == b) & work
            denom = (Amod[m].detach() ** 2).sum()
            if denom > 0:
                k = (Aobs[m] * Amod[m].detach()).sum() / denom
                loss = loss + ((Aobs[m] - k * Amod[m]) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)

    fsol_d = sol(hkl).detach()[:n]
    Rwd, Rfd = binwise_scaled_R(
        Fobs, Fprotein + fsol_d, bin_idx, work, free, args.nbins
    )

    print(f"  production baseline (full scaler, current mask): "
          f"{Rw_base:.4f} / {Rf_base:.4f}   [reference, has aniso DOF]")
    print("  common per-bin-scaled R (work / free) -- isolates the solvent model:")
    print(f"    no solvent  : {Rw0:.4f} / {Rf0:.4f}")
    print(f"    current mask: {Rwm:.4f} / {Rfm:.4f}   [k_sol={k_sol:.3f}]")
    bsol_str = (f", B_sol={sol.b_solvent.item():.1f}" if args.residual_bsol else "")
    print(f"    density     : {Rwd:.4f} / {Rfd:.4f}   "
          f"[rho_s={sol.rho_s.item():.3f}, rho0={sol.rho0.item():.3f}{bsol_str}]")

    # ------------------------------------------------------------------
    # (2) rho0 sweep: interpolation flat-mask <-> Babinet
    # ------------------------------------------------------------------
    print("  rho0 sweep  (corr to current-mask F_sol | corr to -F_protein, low-res):")
    lowres = s[:n] < (1.0 / 6.0)

    def corr(a, b, mask):
        a, b = a[mask], b[mask]
        return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

    for rho0 in [0.5, 1.0, 2.0, 5.0, 20.0, 1e4]:
        sweep = DensitySolventModel(
            model, rho0=rho0, occupancy=args.occupancy, solvent_res=args.solvent_res
        )
        fs = sweep(hkl).detach()[:n]
        c_mask = corr(fs.real, Fsol_mask.real, lowres & (fs.abs() > 0))
        c_bab = corr(fs.real, (-Fprotein).real, lowres & (fs.abs() > 0))
        print(f"    rho0={rho0:>8.2f}:  {c_mask:+.3f}  |  {c_bab:+.3f}")

    # ------------------------------------------------------------------
    # (3) amplitude vs resolution
    # ------------------------------------------------------------------
    print("  |F_solv| median by resolution shell (mask | density):")
    for b in range(0, args.nbins, max(1, args.nbins // 5)):
        m = (bin_idx == b) & (fsol_d.abs() > 0)
        if m.sum() == 0:
            continue
        d = 1.0 / (2 * s[:n][m].median().item() + 1e-9)
        print(f"    ~{d:5.1f} A:  {Fsol_mask.abs()[m].median():8.2f}  | "
              f"{fsol_d.abs()[m].median():8.2f}")


if __name__ == "__main__":
    main()
