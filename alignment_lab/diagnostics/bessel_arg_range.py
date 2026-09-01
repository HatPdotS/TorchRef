"""Does the base Rice's Bessel-argument clamp bite in E-space?

`xray_likelihoods._rice_body` clamps `2 F_calc F_obs / Sigma` at 1e6. That cap is
sized for F-space; the translation likelihood runs on E values with
`Sigma = 1 - D^2` floored at 1e-4, where the ratio can in principle be far
larger. A clamp that fires silently truncates the likelihood exactly where it is
most discriminating, so this measures the real range instead of reasoning about
the bound.
"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)
from lab import BENCH_PDBS, load_case, random_rotation, seed_for  # noqa: E402


def main():
    from torchref.experimental.alignment.translation import (
        DirectModelEvaluator, TranslationObs, amplitude_translation_search,
        normalise_calc, precompute_G_for_rotation)

    for pdb in sys.argv[1:] or list(BENCH_PDBS):
        model, data = load_case(pdb)
        rot = model.copy().rotate(
            random_rotation(seed_for(pdb, 0)).to(model.dtype_float),
            center=model.xyz().mean(0))
        rot.spacegroup = data.spacegroup.hm
        p1 = rot.copy(); p1.spacegroup = "P 1"
        mask = data.get_valid_mask()
        sig = getattr(data, "F_sigma", None)
        obs = TranslationObs.build(data.F[mask], data.hkl[mask], data.spacegroup,
                                   data.cell,
                                   sig_F=None if sig is None else sig[mask],
                                   n_shells=10)
        ev = DirectModelEvaluator(p1)
        eye3 = torch.eye(3, dtype=torch.float64)
        G, h_R = precompute_G_for_rotation(ev, eye3, obs.hkl, data.spacegroup,
                                           data.cell)
        _, _, peaks = amplitude_translation_search(
            obs=obs, interpolator=ev, R_rotation=eye3,
            spacegroup=data.spacegroup, real_cell=data.cell, grid_steps=16,
            n_peaks=1, precomputed_G=G, precomputed_h_R=h_R)
        t = torch.as_tensor(peaks[0].translation, dtype=torch.float64,
                            device=G.device)
        ph = torch.exp(2j * torch.pi * torch.einsum(
            "ind,d->in", h_R.to(torch.float64), t).to(G.dtype))
        E_calc = normalise_calc((G * ph).sum(dim=0).abs().to(torch.float64), obs)

        # Worst case over the whole sigma_A grid: D -> 0.99, Sigma -> 1 - D^2.
        D = 0.99
        Sigma = max(1.0 - D * D, 1e-4)
        arg = (2.0 * (D * E_calc) * obs.E_obs / Sigma)
        print(f"ROW pdb={pdb} N={obs.E_obs.numel()} "
              f"maxE_obs={float(obs.E_obs.max()):.2f} "
              f"maxE_calc={float(E_calc.max()):.2f} "
              f"max_bessel_arg={float(arg.max()):.3e} "
              f"clamped={int((arg > 1e6).sum())}", flush=True)


main()
