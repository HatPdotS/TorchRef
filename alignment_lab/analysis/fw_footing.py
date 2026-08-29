"""Is French-Wilson's `eEobs` supposed to have unit mean square?

The conformance table reports `obs_calc_ratio` ~2.1 for `FrenchWilsonE`, i.e.
`<eEobs**2>` sits near 0.5 where the calc companion sits at 1. Two readings, and
they call for different actions:

* the FW port is mis-scaled -- a real defect in the FRF's observed side; or
* `eEobs` is a DEFLATED amplitude by construction and the check is asking the
  wrong question of it.

`eEobs**2 = eEsqFW + (DFAC**2 - 1)/DFAC**2` with `DFAC < 1`, so the second term
is strictly negative: the assembly subtracts the share of the measured intensity
that is measurement error. That is a deconvolution, not a normalisation, and
`eEobs` travels with `DFAC` as a pair.

So the question is not "is `<eEobs**2>` one" but "is the quantity the CONSUMER
forms centred". The consumer is `build_lerf1_obs_intensity`, which forms
`cw * (eEobs**2 - 1) * DFAC**2` -- it subtracts a literal 1. If `<eEobs**2>` is
really 0.5, that term carries a systematic negative offset into the Patterson
correlation, and whether that matters is a separate question from whether the
port is faithful.

This decomposes the assembly term by term, per resolution decile, so the answer
comes from the numbers rather than from reading the formula.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import load_case  # noqa: E402


def main() -> int:
    from torchref.experimental.alignment.frf.french_wilson import (
        french_wilson_preprocess,
    )
    from torchref.experimental.alignment.frf.preprocessing import (
        build_lerf1_intensity,
    )
    from lab.reference_normalisers import wilson_normalise

    for pdb in ("1DAW", "3K7M", "2DQ6"):
        model, data = load_case(pdb)
        F = data.F.to(torch.float64).abs().cpu()
        sig = data.F_sigma.to(torch.float64).cpu()
        hkl = data.hkl.cpu()
        rec = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
        s = (hkl.to(torch.float64) @ rec).norm(dim=-1)
        cen = data.centric.cpu().to(torch.bool)
        keep = torch.isfinite(F) & torch.isfinite(sig) & (sig > 0) & (F > 0)
        F, sig, s, cen = F[keep], sig[keep], s[keep], cen[keep]

        fw = french_wilson_preprocess(F, sig, s, cen, n_wilson_shells=20)
        eE, dfac = fw["eEobs"].to(torch.float64), fw["DFAC"].to(torch.float64)
        # eEsqFW is what eEobs**2 would be before the deconvolution term.
        corr = (dfac * dfac - 1.0) / (dfac * dfac)
        eEsq = eE * eE - corr
        wil = wilson_normalise(F, s, 20)[0].to(torch.float64)
        lerf = build_lerf1_intensity(
            fw["eEobs"], cen, weight=dfac * dfac, use_centric_weight=True,
        ).to(torch.float64)

        order = torch.argsort(s)
        dec = torch.zeros_like(s, dtype=torch.long)
        ch = max(1, s.numel() // 10)
        for k in range(10):
            hi = (k + 1) * ch if k < 9 else s.numel()
            dec[order[k * ch:hi]] = k

        print(f"\n=== {pdb}  n={F.numel()}  centric={int(cen.sum())} ===")
        print(f"  {'dec':>3s} {'d(A)':>12s} {'<Ewil^2>':>9s} {'<eEsqFW>':>9s} "
              f"{'<eEobs^2>':>10s} {'<DFAC>':>7s} {'<lerf1>':>9s}")
        for k in range(10):
            m = dec == k
            lo, hi = float(1 / s[m].max()), float(1 / s[m].min())
            print(f"  {k:>3d} {f'{hi:5.1f}-{lo:4.2f}':>12s} "
                  f"{float((wil[m] ** 2).mean()):>9.4f} "
                  f"{float(eEsq[m].mean()):>9.4f} "
                  f"{float((eE[m] ** 2).mean()):>10.4f} "
                  f"{float(dfac[m].mean()):>7.4f} {float(lerf[m].mean()):>9.4f}")
        print(f"  {'ALL':>3s} {'':>12s} {float((wil ** 2).mean()):>9.4f} "
              f"{float(eEsq.mean()):>9.4f} {float((eE ** 2).mean()):>10.4f} "
              f"{float(dfac.mean()):>7.4f} {float(lerf.mean()):>9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
