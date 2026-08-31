"""Does the translation LLG rank badly because each candidate fits its own sigma_A?

Over ten seeds on 2DQ6 the plain correlation puts truth at rank 0 in 10/10 while
the LLG -- a likelihood, strictly more information -- manages 2/10. A weaker
score beating a stronger one is a symptom, not a result.

The suspect is how many parameters each score fits PER CANDIDATE:

    correlation   0            sum w E_obs^2_c |Fc|^2 / sum w |Fc|^2
    analytic R    1            the global scale k
    LLG           n_shells     fit_sigma_a_per_shell, on THAT candidate's
                               own top translation

`_llg_tf_rescore` is called once per rotation candidate and refits sigma_A each
time, so every wrong orientation is scored against a likelihood tuned to itself.
The docstring there warns against exactly this one level down -- refitting per
translation -- and the pipeline then does it per rotation.

This recomputes the LLG with sigma_A held FIXED across candidates, three ways:

``per_cand``   what the pipeline does now, as the control.
``shared``     fitted once, on the FRF's top-ranked candidate. Model-dependent
               but not candidate-dependent, so it cannot flatter any one of them.
``empirical``  from Sigma_obs/Sigma_calc via weighting.empirical_sigma_a, which
               is rotation-invariant by construction -- total scattering per
               shell does not depend on orientation -- and is the estimate that
               exists for precisely this reason.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import (BENCH_PDBS, rotated_case, seed_for,  # noqa: E402
                 symmetry_orbit)
from lab.truth import angle_to_orbit  # noqa: E402


def _rank_of_truth(scores, is_truth, higher_is_better=True):
    order = sorted(range(len(scores)), key=lambda i: scores[i],
                   reverse=higher_is_better)
    for pos, i in enumerate(order):
        if is_truth[i]:
            return pos
    return -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="2DQ6", choices=list(BENCH_PDBS))
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--n-cand", type=int, default=15)
    ap.add_argument("--thr-deg", type=float, default=8.0)
    args = ap.parse_args()

    from torchref.experimental.alignment.frf.rotation_utils import (
        rotation_matrix_from_edmonds_euler,
    )
    from torchref.experimental.alignment.pipeline import (
        MolecularReplacementPipeline,
    )
    from torchref.experimental.alignment.rotation_search import prepare_frf_inputs
    from torchref.experimental.alignment.translation import (
        DirectModelEvaluator, amplitude_translation_search, fit_sigma_a_per_shell,
        llg_translation_rescore, normalise_calc, precompute_G_for_rotation,
    )
    from torchref.scaling import WilsonNormaliser
    from torchref.scaling.weighting import empirical_sigma_a

    for trial in range(args.trials):
        seed = seed_for(args.pdb, trial)
        model, data, R_true = rotated_case(args.pdb, seed)
        pipe = MolecularReplacementPipeline(
            data, model, verbose=0, n_rotation_peaks=200,
            n_rotation_candidates=args.n_cand, use_llg_tf=False)
        frf = prepare_frf_inputs(model, data, d_min=pipe.d_min, d_max=pipe.d_max,
                                 n_shells=pipe.n_shells, verbose=0)
        pipe._frf = frf
        peaks = pipe._rotation_candidates(frf)[: args.n_cand]
        pipe._prepare_translation_arrays()
        obs = pipe._obs
        eye3 = pipe._eye3
        orbit = symmetry_orbit(
            R_true, data.spacegroup.matrices.to(torch.float64).cpu(),
            side="left", frame="cart",
            reciprocal_basis=data.cell.reciprocal_basis_matrix.to(
                torch.float64).cpu())

        # One pass to collect each candidate's G, its top translation and its
        # own E_calc there; sigma_A choices are applied afterwards so every
        # variant scores the SAME placements.
        cand = []
        for p in peaks:
            ang = angle_to_orbit(
                rotation_matrix_from_edmonds_euler(p.alpha, p.beta, p.gamma),
                orbit)
            rot = pipe._make_rotated(p)[0]
            rot.spacegroup = data.spacegroup.hm
            p1 = rot.copy(); p1.spacegroup = "P 1"
            ev = DirectModelEvaluator(p1)
            G, h_R = precompute_G_for_rotation(
                ev, eye3, obs.hkl, data.spacegroup, data.cell)
            _, _, tp = amplitude_translation_search(
                obs=obs, interpolator=ev, R_rotation=eye3,
                spacegroup=data.spacegroup, real_cell=data.cell,
                grid_steps=pipe.translation_grid_steps,
                n_peaks=pipe.n_translation_peaks, cluster_radius=0.05,
                precomputed_G=G, precomputed_h_R=h_R)
            t_top = torch.as_tensor(tp[0].translation, dtype=torch.float64,
                                    device=G.device)
            ph = torch.exp(2j * torch.pi * torch.einsum(
                "ind,d->in", h_R.to(torch.float64), t_top).to(G.dtype))
            Fc_top = (G * ph).sum(dim=0).abs().to(torch.float64)
            cand.append(dict(ang=float(ang), corr=float(tp[0].score), G=G,
                             h_R=h_R, t=t_top, Fc=Fc_top,
                             E_calc=normalise_calc(Fc_top, obs)))

        is_truth = [c["ang"] <= args.thr_deg for c in cand]
        if not any(is_truth):
            print(f"ROW pdb={args.pdb} trial={trial} truth_found=0", flush=True)
            continue

        def sa_per_cand(c):
            return fit_sigma_a_per_shell(obs.E_obs, c["E_calc"], obs.centric,
                                         obs.shell_idx, obs.n_shells, n_grid=81)
        sa_shared = sa_per_cand(cand[0])          # the FRF's own top candidate
        fit_calc = WilsonNormaliser(
            cand[0]["Fc"] ** 2, obs.s_mag, n_coeff=6,
            s_lo=float(obs.s_mag.min()), s_hi=float(obs.s_mag.max()))
        sa_emp_per_refl = empirical_sigma_a(
            obs.fit.evaluate(obs.s_mag).to(torch.float64),
            fit_calc.evaluate(obs.s_mag).to(torch.float64))
        # collapse to per-shell, the shape llg_translation_rescore expects
        cnt = torch.bincount(obs.shell_idx, minlength=obs.n_shells).to(torch.float64)
        tot = torch.zeros(obs.n_shells, dtype=torch.float64).scatter_add_(
            0, obs.shell_idx, sa_emp_per_refl.to(torch.float64))
        sa_emp = (tot / cnt.clamp(min=1.0)).clamp(1e-3, 1 - 1e-6)

        def llg_of(c, sigma_a):
            return float(llg_translation_rescore(
                obs=obs, G=c["G"], h_R=c["h_R"],
                t_candidates=c["t"].view(1, 3), sigma_a=sigma_a)[0])

        variants = {
            "per_cand": [llg_of(c, sa_per_cand(c)) for c in cand],
            "shared":   [llg_of(c, sa_shared) for c in cand],
            "empirical": [llg_of(c, sa_emp) for c in cand],
        }
        r_corr = _rank_of_truth([c["corr"] for c in cand], is_truth)
        parts = " ".join(
            f"rank_{k}={_rank_of_truth(v, is_truth)}" for k, v in variants.items())
        print(f"ROW pdb={args.pdb} trial={trial} n_truth={sum(is_truth)} "
              f"rank_corr={r_corr} {parts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
