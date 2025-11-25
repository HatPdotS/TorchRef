#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/analytical_scaling_test.out

import torch
import numpy as np
from typing import Tuple, Dict

def analytical_kmask_K(
    fcalc: torch.Tensor,              # complex tensor shape (N,) dtype=torch.cfloat
    fmask: torch.Tensor,              # complex tensor shape (N,)
    hkl: torch.Tensor,                # (N,3) integers (not used for math but kept)
    scattering_vecs: torch.Tensor,    # (N,3) real: reciprocal-space/scattering vectors
    Fobs: torch.Tensor,               # real or complex (N,) observed amplitudes (if complex, |.| used)
    n_bins: int = 20,
    overall_factor: float = 1.0,      # k_overall * kanisotropic prefactor (assumed applied to Fobs)
    anisotropic_factor: float = 1.0,  # included for clarity; total prefactor = overall_factor * anisotropic_factor
    min_refl_per_bin: int = 30
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Compute per-resolution-shell kmask and K = k_isotropic^{-2} analytically as in Afonine et al. (2013).

    Returns:
      kmask_per_reflection: torch.Tensor (N,) real
      K_per_reflection:     torch.Tensor (N,) real
      bin_info: dict with details per bin:
        - 'bin_edges', 'bin_centers', 'kmask_bin', 'K_bin', 'n_refl'
    Notes:
      - This implementation assumes overall_factor * anisotropic_factor is known and multiplies Fobs.
      - If Fobs given as amplitudes, pass them directly. If Fobs are intensities, adapt appropriately.
      - The cubic coefficients are obtained by algebraic substitution of K from Eq.(1) into Eq.(2)
        as described in the paper; then numpy.roots is used to get candidate real roots.
    """

    # --- input checks & normalize types ---
    assert fcalc.shape == fmask.shape, "fcalc and fmask must have the same shape"
    N = fcalc.shape[0]
    device = fcalc.device

    # Make sure Fobs are amplitudes (real). If complex given, take abs.
    Fobs_amp = Fobs.abs() if torch.is_complex(Fobs) else Fobs.clone().to(dtype=torch.float32)
    # per-paper Is variable: I_s = [k_overall * k_anisotropic(s)]^2 * Fobs^2.
    # We assume overall/anisotropic scalars here:
    Is = (overall_factor * anisotropic_factor)**2 * (Fobs_amp**2)              # (N,)

    # compute resolution per reflection using scattering_vecs: d = 1 / |s|
    s_norm = scattering_vecs.norm(dim=1) + 1e-20
    d_res = 1.0 / s_norm    # resolution in same units as scattering vector convention
    # log-spaced binning over ln(d)
    ln_d = torch.log(d_res)
    ln_min, ln_max = ln_d.min().item(), ln_d.max().item()
    bin_edges = torch.exp(torch.linspace(ln_min, ln_max, steps=n_bins+1, device=device))
    # For grouping, use ln(d) edges to avoid numerical issues:
    ln_edges = torch.linspace(ln_min, ln_max, steps=n_bins+1, device=device)

    # Prepare outputs
    kmask_per_ref = torch.zeros(N, dtype=torch.float32, device=device)
    K_per_ref = torch.zeros(N, dtype=torch.float32, device=device)
    kmask_bin = torch.zeros(n_bins, dtype=torch.float32, device=device)
    K_bin = torch.zeros(n_bins, dtype=torch.float32, device=device)
    n_refl_bin = torch.zeros(n_bins, dtype=torch.int32, device=device)
    bin_centers = torch.exp(0.5*(ln_edges[:-1] + ln_edges[1:]))

    # Precompute u, v, w per reflection
    # us = |Fcalc|^2
    us = (fcalc.real**2 + fcalc.imag**2).to(dtype=torch.float64)   # better numeric precision for sums
    # ws = |Fmask|^2
    ws = (fmask.real**2 + fmask.imag**2).to(dtype=torch.float64)
    # vs = Re(Fcalc * conj(Fmask))
    # conj(Fmask) = fmask.real - i fmask.imag ; product -> complex; we want real part
    vs = (fcalc.real * fmask.real + fcalc.imag * fmask.imag).to(dtype=torch.float64)

    Is_d = Is.to(dtype=torch.float64)

    # iterate bins
    for ib in range(n_bins):
        ln_left = ln_edges[ib].item()
        ln_right = ln_edges[ib+1].item()
        sel = (ln_d >= ln_left) & (ln_d < ln_right)
        idx = torch.nonzero(sel, as_tuple=False).squeeze(1)
        n_refl = idx.numel()
        n_refl_bin[ib] = n_refl
        if n_refl < min_refl_per_bin:
            # skip: assign zeros (or fallback)
            kmask_bin[ib] = 0.0
            # compute K by simple least-squares from Eq1 with kmask=0:
            # Eq1 with k_mask=0 -> S_uI - K*S_I2 = 0  => K = S_uI / S_I2 (if S_I2>0)
            S_uI = us[idx].dot(Is_d[idx]) if n_refl>0 else 0.0
            S_I2 = (Is_d[idx]**2).sum() if n_refl>0 else 0.0
            if n_refl>0 and S_I2 > 0:
                K_val = (S_uI / S_I2).item()
            else:
                K_val = 1.0
            K_bin[ib] = K_val
            K_per_ref[idx] = float(K_val)
            continue

        # compute the sums used in the algebra (in double precision)
        S_wI  = (ws[idx] * Is_d[idx]).sum().item()
        S_vI  = (vs[idx] * Is_d[idx]).sum().item()
        S_uI  = (us[idx] * Is_d[idx]).sum().item()
        S_I2  = (Is_d[idx] * Is_d[idx]).sum().item()

        # sums for cubic coefficient
        S_w2  = (ws[idx] * ws[idx]).sum().item()
        S_wv  = (ws[idx] * vs[idx]).sum().item()
        S_v2  = (vs[idx] * vs[idx]).sum().item()
        S_uw  = (us[idx] * ws[idx]).sum().item()
        S_Iw  = (Is_d[idx] * ws[idx]).sum().item()
        S_uv  = (us[idx] * vs[idx]).sum().item()
        S_Iv  = (Is_d[idx] * vs[idx]).sum().item()

        # first equation: k^2 * S_wI + 2 k * S_vI + S_uI - K * S_I2 = 0
        # => K = (k^2 * S_wI + 2k * S_vI + S_uI) / S_I2    (denominator S_I2 assumed >0)
        if abs(S_I2) < 1e-20:
            # degenerate case -> no reliable solution; fallback to kmask=0
            kmask_bin[ib] = 0.0
            K_val = (S_uI / (S_I2 + 1e-20)) if S_I2 != 0 else 1.0
            K_bin[ib] = float(K_val)
            K_per_ref[idx] = float(K_val)
            continue

        a = S_wI
        b = 2.0 * S_vI
        c = S_uI
        d = S_I2

        # Build coefficients of cubic: coef3*k^3 + coef2*k^2 + coef1*k + coef0 = 0
        coef3 = S_w2 - (S_Iw / d) * a
        coef2 = 3.0 * S_wv - (S_Iw / d) * b - (S_Iv / d) * a
        coef1 = (2.0 * S_v2 + S_uw) - (S_Iw / d) * c - (S_Iv / d) * b
        coef0 = S_uv - (S_Iv / d) * c

        # If coefficients are all ~0, fallback:
        if (abs(coef3) < 1e-30 and abs(coef2) < 1e-30 and abs(coef1) < 1e-30 and abs(coef0) < 1e-30):
            kmask_bin[ib] = 0.0
            K_val = (S_uI / d)
            K_bin[ib] = float(K_val)
            K_per_ref[idx] = float(K_val)
            continue

        # Solve cubic using numpy (coerce to CPU numpy floats)
        poly = np.array([coef3, coef2, coef1, coef0], dtype=np.float64)
        # if leading coef is nearly zero, reduce to quadratic/linear robustly
        # use numpy.roots which handles leading zeros automatically
        roots = np.roots(poly)   # complex roots possible

        # filter real, non-negative roots (allow tiny negative tolerance)
        cand = []
        for r in roots:
            if abs(r.imag) < 1e-8:
                rreal = float(r.real)
                if rreal >= -1e-8:   # allow tiny negative numerical errors
                    cand.append(max(rreal, 0.0))
        # Always include zero as candidate (paper uses kmask=0 if no positive root)
        cand.append(0.0)

        # Evaluate LS value for each candidate and choose the one with smallest LS
        best_k = None
        best_LS = None
        for k_trial in cand:
            # compute K from Eq1
            K_trial = ( (k_trial**2) * S_wI + (2.0 * k_trial) * S_vI + S_uI ) / d
            # compute per-reflection residuals and LS
            # residual r_s = (k^2 ws + 2 k vs + us) - K Is
            # LS = sum(r_s^2)
            k2_ws = (k_trial**2) * ws[idx]
            two_k_vs = (2.0 * k_trial) * vs[idx]
            us_sel = us[idx]
            resid = (k2_ws + two_k_vs + us_sel) - (K_trial * Is_d[idx])
            LS_val = float((resid * resid).sum().item())
            if (best_LS is None) or (LS_val < best_LS):
                best_LS = LS_val
                best_k = k_trial
                best_K = float(K_trial)

        # assign best values
        kmask_bin[ib] = float(best_k)
        K_bin[ib] = float(best_K)
        kmask_per_ref[idx] = float(best_k)
        K_per_ref[idx] = float(best_K)

    # return tensors on original device
    kmask_per_ref = kmask_per_ref.to(device=device)
    K_per_ref = K_per_ref.to(device=device)

    bin_info = {
        'bin_edges_ln': ln_edges.cpu().numpy(),
        'bin_centers': bin_centers.cpu().numpy(),
        'kmask_bin': kmask_bin.cpu().numpy(),
        'K_bin': K_bin.cpu().numpy(),
        'n_refl_bin': n_refl_bin.cpu().numpy()
    }
    return kmask_per_ref, K_per_ref, bin_info

import torch
from typing import Tuple

def apply_bulk_solvent_and_isotropic_scale(
    fcalc: torch.Tensor,               # complex tensor (N,)
    fmask: torch.Tensor,               # complex tensor (N,)
    K_per_ref: torch.Tensor,           # real tensor (N,), corresponds to k_isotropic^{-2}
    kmask_per_ref: torch.Tensor,       # real tensor (N,), bulk-solvent scale factor
    k_overall: float = 1.0,            # optional overall scale (usually 1)
    kanisotropic_per_ref: torch.Tensor = None  # optional anisotropic factor per reflection
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled model structure factors F_model and their intensities I_model
    according to Afonine et al. (2013).

    Model:
        F_model(s) = k_total(s) * ( Fcalc(s) + kmask(s) * Fmask(s) )

    where
        k_total(s)^2 = K^{-1}(s) = k_overall^2 * k_isotropic(s)^2 * k_anisotropic(s)^2
        -> hence k_total(s) = k_overall * k_anisotropic(s) / sqrt(K(s))

    Args:
        fcalc: complex (N,) atomic model structure factors
        fmask: complex (N,) bulk solvent mask structure factors
        K_per_ref: real (N,), equal to k_isotropic^{-2}(s)
        kmask_per_ref: real (N,), bulk solvent scale factors
        k_overall: scalar, overall scale (default = 1)
        kanisotropic_per_ref: real (N,) optional, anisotropic scaling factors (default = 1)

    Returns:
        F_model: complex (N,) scaled model structure factors
        I_model: real (N,) model intensities = |F_model|^2
    """

    # Ensure shapes are consistent
    assert fcalc.shape == fmask.shape == K_per_ref.shape == kmask_per_ref.shape, \
        "All inputs must have the same shape per reflection."

    # Compute total scaling factor per reflection
    # K = k_isotropic^{-2}  ->  k_isotropic = 1/sqrt(K)
    k_isotropic = torch.sqrt(1.0 / (K_per_ref + 1e-20))

    if kanisotropic_per_ref is None:
        kanisotropic_per_ref = torch.ones_like(k_isotropic)

    # Total scale factor per reflection:
    k_total = k_overall * kanisotropic_per_ref * k_isotropic

    # Compute scaled model structure factors
    F_model = k_total * (fcalc + kmask_per_ref * fmask)

    # Compute corresponding intensities
    I_model = (F_model.real**2 + F_model.imag**2)

    return F_model, I_model

def get_rfactor(Fobs: torch.Tensor, Fcalc: torch.Tensor) -> float:
    """
    Compute R-factor between observed and calculated structure factors.

    R = sum(| |Fobs| - |Fcalc| |) / sum(|Fobs|)

    Args:
        Fobs: complex or real tensor (N,) observed structure factors
        Fcalc: complex tensor (N,) calculated structure factors
    """
    Fobs_amp = Fobs.abs() if torch.is_complex(Fobs) else Fobs.clone().to(dtype=torch.float32)
    Fcalc_amp = Fcalc.abs().to(dtype=torch.float32)

    numerator = torch.sum(torch.abs(Fobs_amp - Fcalc_amp))
    denominator = torch.sum(Fobs_amp) + 1e-20  # avoid division by zero

    R = (numerator / denominator).item()
    return R

from torchref.Data import ReflectionData
from torchref.model_ft import ModelFT
from torchref.solvent import SolventModel
from torchref.math_functions.math_torch import get_scattering_vectors

pdbin = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler/dark.pdb'
mtzin = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler/dark.mtz'

data = ReflectionData().load_from_mtz(mtzin)
model = ModelFT(verbose=2,max_res=1.7).load_pdb_from_file(pdbin)

solvent = SolventModel(model)

fcalc = model(data.get_hkl())
fmask = solvent.get_rec_solvent(data.get_hkl())
hkl = data.get_hkl()

s = get_scattering_vectors(hkl, data.cell)

hkl, fobs, _, _ = data()

kmask_per_ref, K_per_ref, bin_info = analytical_kmask_K(
    fcalc=fcalc,
    fmask=fmask,
    hkl=hkl,
    scattering_vecs=s,
    Fobs=fobs,
    n_bins=20,
    overall_factor=1.0,
    anisotropic_factor=1.0,
    min_refl_per_bin=30
)


F_model, I_model = apply_bulk_solvent_and_isotropic_scale(
    fcalc=fcalc,
    fmask=fmask,
    K_per_ref=K_per_ref,
    kmask_per_ref=kmask_per_ref)




print(F_model.dtype)
R = get_rfactor(fobs, torch.abs(F_model))

print(f"R-factor after analytical scaling: {R:.4f}")    


