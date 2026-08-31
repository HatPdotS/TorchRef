"""What does it cost to normalise the LLG's calculated side with the shared fit?

Two per-shell normalisations survive in the translation likelihood -- the
``E_calc`` of each candidate translation, and the ``E_calc`` of the top peak that
the sigma_A fit runs against. They are the last places in the alignment package
that answer "what is the mean intensity here" without going through
:class:`~torchref.scaling.WilsonNormaliser`.

The argument for keeping them was cost: the shared fit is a Gamma GLM by IRLS and
the calc side needs one fit per candidate, K of them per rotation. This measures
that instead of asserting it, and also asks the two questions that decide whether
the swap is safe at all:

* does the fit **converge** on a calculated set, which has near-zeros at the
  nodes of the molecular transform where an observed set has none, and
* how far do the two normalisations actually differ, per reflection and in the
  LLG ranking they feed.

Warm-up first: on this filesystem a first call pays cold package reads inside
whatever timer surrounds it, which is worth ~100x and is not compute.

Usage::

    python alignment_lab/diagnostics/calc_norm_cost.py --pdb 1DAW --k 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import BENCH_PDBS, load_case, random_rotation, seed_for  # noqa: E402


def _time(fn, repeats=3):
    fn()                                   # warm: discard the cold-read call
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="1DAW", choices=list(BENCH_PDBS))
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--k", type=int, default=20, help="candidate translations")
    ap.add_argument("--n-coeff", type=int, default=6)
    args = ap.parse_args()

    from torchref.experimental.alignment.translation import (
        DirectModelEvaluator, TranslationObs, amplitude_translation_search,
        precompute_G_for_rotation,
    )
    from torchref.scaling import WilsonNormaliser

    seed = seed_for(args.pdb, args.trial)
    model, data = load_case(args.pdb)
    R_true = random_rotation(seed)
    rot = model.copy()
    rot = rot.rotate(R_true.to(model.dtype_float), center=model.xyz().mean(0))
    rot.spacegroup = data.spacegroup.hm
    p1 = rot.copy()
    p1.spacegroup = "P 1"

    mask = data.get_valid_mask()
    sig = getattr(data, "F_sigma", None)
    obs = TranslationObs.build(
        data.F[mask], data.hkl[mask], data.spacegroup, data.cell,
        sig_F=None if sig is None else sig[mask],
    )
    ev = DirectModelEvaluator(p1)
    eye3 = torch.eye(3, dtype=torch.float64)
    G, h_R = precompute_G_for_rotation(
        ev, eye3, obs.hkl, data.spacegroup, data.cell)
    _, _, peaks = amplitude_translation_search(
        obs=obs, interpolator=ev, R_rotation=eye3,
        spacegroup=data.spacegroup, real_cell=data.cell,
        grid_steps=16, n_peaks=args.k, precomputed_G=G, precomputed_h_R=h_R)

    K = min(args.k, len(peaks))
    N = obs.hkl.shape[0]
    t_cand = torch.as_tensor(
        [p.translation for p in peaks[:K]], dtype=torch.float64,
        device=G.device)
    phase = torch.exp(2j * torch.pi * torch.einsum(
        "ind,kd->kin", h_R.to(torch.float64), t_cand).to(G.dtype))
    F_calc = (G.view(1, *G.shape) * phase).sum(dim=1).abs().to(torch.float64)

    print(f"# {args.pdb} trial={args.trial} N={N} K={K} "
          f"n_coeff={args.n_coeff}", flush=True)

    # --- current: one per-shell mean per candidate, all K at once ---
    def per_shell():
        idx = obs.shell_idx.view(1, -1).expand(K, N)
        cnt = torch.bincount(obs.shell_idx, minlength=obs.n_shells).to(torch.float64)
        tot = torch.zeros((K, obs.n_shells), dtype=torch.float64, device=G.device)
        tot.scatter_add_(1, idx, F_calc * F_calc)
        mean = (tot / cnt.clamp(min=1.0).unsqueeze(0)).clamp(min=1e-30)
        return F_calc / mean.sqrt().gather(1, idx)

    # --- proposed: the shared Wilson fit, once per candidate ---
    def wilson():
        out = torch.empty_like(F_calc)
        iters = []
        for k in range(K):
            w = WilsonNormaliser(
                F_calc[k] * F_calc[k], obs.s_mag, n_coeff=args.n_coeff,
                s_lo=float(obs.s_mag.min()), s_hi=float(obs.s_mag.max()),
            )
            out[k] = w.E.to(torch.float64)
            iters.append(w.n_iter)
        return out, iters

    t_shell, E_shell = _time(per_shell)
    t_wilson, (E_wilson, iters) = _time(wilson)

    # How different are they, and does the *ranking* they feed move?
    rel = ((E_wilson - E_shell).abs()
           / E_shell.abs().clamp(min=1e-12)).median().item()
    m_shell = (E_shell ** 2).mean(dim=1)
    m_wilson = (E_wilson ** 2).mean(dim=1)

    print(f"ROW pdb={args.pdb} N={N} K={K} "
          f"t_per_shell_ms={1000 * t_shell:.2f} "
          f"t_wilson_ms={1000 * t_wilson:.1f} "
          f"ratio={t_wilson / max(t_shell, 1e-9):.0f}x "
          f"per_cand_ms={1000 * t_wilson / K:.1f} "
          f"iter_min={min(iters)} iter_max={max(iters)} "
          f"median_rel_dE={rel:.4f} "
          f"meanE2_shell={m_shell.mean():.4f} "
          f"meanE2_wilson={m_wilson.mean():.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
