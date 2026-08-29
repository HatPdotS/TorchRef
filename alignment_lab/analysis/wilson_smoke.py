"""Does the Wilson normaliser hold its identity on real data, both sides?

The synthetic test draws from the distribution the fit assumes, so it can only
show the arithmetic is right. These are the cases the assumption is wrong in:
observations carry measurement error and a real solvent deficit, and the
rotation function's calc side is an oversampled molecular transform in a P1 box,
where adjacent samples are correlated and Wilson independence does not hold at
all. The mean estimate survives that by quasi-likelihood -- a log-link Gamma GLM
is consistent for the mean under a misspecified variance function -- and this is
where that claim gets checked rather than asserted.

Also checks the property the whole weighting design rests on: fitted with a
SHARED abscissa, the two curves are comparable, and their ratio is the
resolution-dependent model deficiency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import load_case  # noqa: E402

CASES = ("1DAW", "2DQ6", "3K7M", "4BX9")


def main() -> int:
    from torchref.scaling.wilson import WilsonNormaliser

    for pdb in CASES:
        model, data = load_case(pdb)
        hkl = data.hkl
        rec = data.cell.reciprocal_basis_matrix.to(torch.float64)
        s = (hkl.to(torch.float64) @ rec).norm(dim=-1)
        F = data.F.to(torch.float64).abs()
        keep = torch.isfinite(F) & (F > 0)
        has_I = getattr(data, "I", None) is not None
        print(f"\n=== {pdb}  {data.spacegroup.hm}  n={int(keep.sum())}  "
              f"raw intensities available: {has_I} ===")

        # --- observed side -------------------------------------------------
        I_obs = (F * F)[keep]
        obs = WilsonNormaliser.from_hkl(
            I_obs, hkl[keep], data.spacegroup, data.cell, n_coeff=6,
            s_lo=float(s.min()), s_hi=float(s.max()),
        )
        cen = data.spacegroup.is_centric(hkl[keep].to(torch.long)).to(torch.bool)
        k = torch.where(cen, 0.5, 1.0).to(torch.float64)
        e2 = obs.E_squared.to(torch.float64)
        print(f"  obs  {obs!r}")
        print(f"       k-weighted <E^2> = {float((k*e2).sum()/k.sum()):.10f}")
        _deciles("       obs decile <E^2>", s[keep], e2)

        # --- calculated side, on the SAME abscissa --------------------------
        F_calc = model.get_structure_factor(hkl[keep], recalc=True).abs()
        I_calc = (F_calc.to(torch.float64) ** 2)
        calc = WilsonNormaliser.from_hkl(
            I_calc, hkl[keep], data.spacegroup, data.cell, n_coeff=6,
            s_lo=float(s.min()), s_hi=float(s.max()),
        )
        e2c = calc.E_squared.to(torch.float64)
        print(f"  calc {calc!r}")
        print(f"       k-weighted <E^2> = {float((k*e2c).sum()/k.sum()):.10f}")
        _deciles("       calc decile <E^2>", s[keep], e2c)

        # --- the ratio the weight will be built from ------------------------
        # Same basis, so the two curves are directly comparable. Reported as a
        # shape: what the model under-explains, versus resolution.
        grid = torch.linspace(float(s[keep].min()), float(s[keep].max()), 8)
        r = (obs.evaluate(grid).to(torch.float64)
             / calc.evaluate(grid).to(torch.float64))
        r = r / r.mean()
        print("       Sigma_obs/Sigma_calc (normalised) vs d(A):")
        print("         " + "  ".join(f"{1/float(x):5.1f}:{float(v):5.2f}"
                                      for x, v in zip(grid, r)))
    return 0


def _deciles(label, s, v):
    order = torch.argsort(s)
    d = [float(v[order[i::10]].mean()) for i in range(10)]
    print(f"{label}: min {min(d):.4f} max {max(d):.4f}  "
          + " ".join(f"{x:.2f}" for x in d))


if __name__ == "__main__":
    sys.exit(main())
