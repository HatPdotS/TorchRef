"""Where does the likelihood ranking's 2.5x actually go?

The model-error fit used to be a local 81-point scan over every reflection, in
float64, evaluating both likelihood branches everywhere and discarding half --
295 ms per candidate on 2DQ6, three times the translation refine beside it. It
now goes through the shared `SigmaAEstimator`.

This times the pieces so the effect is measured rather than assumed: the
model-error fit, the Wilson normalisation of the calculated side, the likelihood
evaluation itself, and the translation refine they sit alongside.
"""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)
from lab import load_case, random_rotation, seed_for  # noqa: E402


def _t(fn, n=3):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return min(ts), out


def main():
    from torchref.experimental.alignment.translation import (
        DirectModelEvaluator, TranslationObs, amplitude_translation_search,
        correlation_at, fit_model_error, llg_at, local_translation_refine,
        normalise_calc, precompute_G_for_rotation)

    for pdb in sys.argv[1:] or ["1DAW", "2DQ6"]:
        seed = seed_for(pdb, 0)
        model, data = load_case(pdb)
        rot = model.copy().rotate(random_rotation(seed).to(model.dtype_float),
                                  center=model.xyz().mean(0))
        rot.spacegroup = data.spacegroup.hm
        p1 = rot.copy(); p1.spacegroup = "P 1"
        mask = data.get_valid_mask()
        sig = getattr(data, "F_sigma", None)
        obs = TranslationObs.build(data.F[mask], data.hkl[mask], data.spacegroup,
                                   data.cell,
                                   sig_F=None if sig is None else sig[mask],
                                   n_shells=10)
        ev = DirectModelEvaluator(p1); eye3 = torch.eye(3, dtype=torch.float64)
        G, h_R = precompute_G_for_rotation(ev, eye3, obs.hkl, data.spacegroup,
                                           data.cell)
        _, _, peaks = amplitude_translation_search(
            obs=obs, interpolator=ev, R_rotation=eye3,
            spacegroup=data.spacegroup, real_cell=data.cell, grid_steps=16,
            n_peaks=20, precomputed_G=G, precomputed_h_R=h_R)
        t0 = torch.as_tensor(peaks[0].translation, dtype=torch.float64)

        t_ref, _ = _t(lambda: local_translation_refine(
            obs=obs, interpolator=ev, R_rotation=eye3,
            spacegroup=data.spacegroup, real_cell=data.cell, t_init=t0,
            radius=0.06, grid_steps=13, n_refinement_passes=1,
            precomputed_G=G, precomputed_h_R=h_R))
        ph = torch.exp(2j * torch.pi * torch.einsum(
            "ind,d->in", h_R.to(torch.float64), t0.to(G.device)).to(G.dtype))
        Fc = (G * ph).sum(dim=0).abs().to(torch.float64)
        t_norm, E_calc = _t(lambda: normalise_calc(Fc, obs))
        t_sa, (alpha, beta) = _t(lambda: fit_model_error(obs, E_calc))
        t_llg, _ = _t(lambda: llg_at(obs, G, h_R, t0, alpha, beta))
        t_corr, _ = _t(lambda: correlation_at(obs, G, h_R, t0))
        N = obs.hkl.numel() // 3
        print(f"ROW pdb={pdb} N={N} shells={obs.n_shells} "
              f"refine={1000*t_ref:.1f}ms norm_calc={1000*t_norm:.1f}ms "
              f"model_err={1000*t_sa:.1f}ms llg={1000*t_llg:.1f}ms "
              f"corr={1000*t_corr:.1f}ms", flush=True)


main()
