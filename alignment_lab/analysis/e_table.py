"""Conformance table over every E convention the alignment package uses.

Establishes what we currently have, before anything changes. Real benchmark data
rather than synthetic, because the properties that matter (the Wilson shape, the
epsilon behaviour on high-symmetry lattices) are properties of real reflection
sets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
torch.set_grad_enabled(False)

from e_conformance import check_e_convention, format_table   # noqa: E402
from lab import BENCH_PDBS, load_case                        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="1DAW", choices=list(BENCH_PDBS))
    ap.add_argument("--n-shells", type=int, default=20)
    args = ap.parse_args()

    from torchref.experimental.alignment.e_values import (
        CalcGlobalE, CalcShellE, FrenchWilsonE, SmoothSigmaE, WilsonShellE,
        WilsonShellEpsE,
    )
    from torchref.experimental.alignment.frf.preprocessing import compute_epsilon

    model, data = load_case(args.pdb)[:2]
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64)
    s_mag = (data.hkl.to(torch.float64) @ rec).norm(dim=-1)
    F = data.F.to(torch.float64).abs()
    sig = None if getattr(data, "F_sigma", None) is None else \
        data.F_sigma.to(torch.float64)
    cen = data.centric.to(torch.bool)
    eps = compute_epsilon(data.hkl.to(torch.long),
                          data.spacegroup.matrices.to(torch.float64))

    # A calc set from the deposited coordinates: the "perfect model" case, where
    # obs and calc genuinely should land on the same scale.
    with torch.no_grad():
        F_calc = model.get_structure_factor(
            data.hkl, recalc=True).abs().to(torch.float64)

    finite = torch.isfinite(F) & torch.isfinite(F_calc) & (F > 0)
    if sig is not None:
        finite &= torch.isfinite(sig) & (sig > 0)
    F, s_mag, cen, eps, F_calc = (F[finite], s_mag[finite], cen[finite],
                                  eps[finite], F_calc[finite])
    sig = None if sig is None else sig[finite]

    uniq, cnt = torch.unique(eps, return_counts=True)
    n_ops = int(data.spacegroup.matrices.shape[0])
    print(f"=== {args.pdb}  {data.spacegroup.hm}  N={int(finite.sum())} "
          f"(dropped {int((~finite).sum())})  n_shells={args.n_shells} ===")
    print(f"  n_ops={n_ops}  epsilon: "
          + ", ".join(f"{float(u):g}x{int(c)}" for u, c in zip(uniq, cnt)))
    reports = []
    for cls in (FrenchWilsonE, WilsonShellE, WilsonShellEpsE, CalcShellE,
                CalcGlobalE, SmoothSigmaE):
        try:
            reports.append(check_e_convention(
                cls, F, s_mag, cen, sig_F=sig, eps=eps,
                n_shells=args.n_shells, F_calc=F_calc,
            ))
        except Exception as exc:                      # noqa: BLE001 - report it
            print(f"  {cls.__name__}: RAISED {type(exc).__name__}: {exc}")
    print(format_table(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
