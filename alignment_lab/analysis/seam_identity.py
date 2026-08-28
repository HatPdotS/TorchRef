"""Is the E-convention seam inert?

Routing the rotation function's normalisation through an `EConvention` class is
only safe to build on if it changes nothing while the default is in place. A
peak-list hash would answer that, but it needs a second worktree to compare
against and it only samples the structures it is run on.

This is stronger and cheaper. The seam replaced exactly three tensors --
`eEobs`, the LERF1 weight, and `E_calc` -- and every line downstream of them is
untouched. So bit-identity on those three is not evidence that the peak list is
unchanged, it is a proof of it, on whatever data this is run over.

The one at real risk is the calc side. `wilson_normalise` accumulates its shell
sums with `index_add_` and clamps the mean at 1e-12; `EConvention` uses
`scatter_add_` and clamps at 1e-30. Same arithmetic in exact terms, and on CPU
both reduce in index order -- but "should be identical" is the claim under test,
not the assumption behind it.

Reports max absolute and relative deviation rather than a bare pass/fail, so a
non-zero result says how big it is instead of only that it exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import load_case  # noqa: E402

CASES = ("1DAW", "3K7M", "2DQ6", "4BX9")


def _dev(new: torch.Tensor, old: torch.Tensor):
    """``(n_differing, max_abs, max_rel)`` between two tensors."""
    d = (new.to(torch.float64) - old.to(torch.float64)).abs()
    rel = d / old.to(torch.float64).abs().clamp(min=1e-30)
    return int((d > 0).sum()), float(d.max()), float(rel.max())


def main() -> int:
    from torchref.experimental.alignment.e_values import (
        FrenchWilsonE, WilsonShellE,
    )
    from torchref.experimental.alignment.frf.french_wilson import (
        french_wilson_preprocess,
    )
    from torchref.experimental.alignment.frf.preprocessing import (
        build_lerf1_intensity, wilson_normalise,
    )
    from torchref.experimental.alignment.sh import (
        assign_shells, equal_count_shell_edges,
    )

    n_shells = 20
    worst = 0.0
    print(f"{'case':>6s} {'tensor':>12s} {'n':>8s} {'n_diff':>7s} "
          f"{'max abs':>10s} {'max rel':>10s}")
    for pdb in CASES:
        model, data = load_case(pdb)
        rec = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
        hkl = data.hkl.cpu()
        s = (hkl.to(torch.float64) @ rec).norm(dim=-1)
        F = data.F.to(torch.float64).abs().cpu()
        sig = data.F_sigma.to(torch.float64).cpu()
        cen = data.centric.cpu().to(torch.bool)
        keep = torch.isfinite(F) & torch.isfinite(sig) & (sig > 0) & (F > 0)
        F, sig, s, cen = F[keep], sig[keep], s[keep], cen[keep]

        edges, _ = equal_count_shell_edges(s, n_shells)
        shell_idx = assign_shells(s, edges)

        # --- obs side -------------------------------------------------
        fw = french_wilson_preprocess(F, sig, s, cen, n_wilson_shells=n_shells,
                                      shell_idx=shell_idx)
        conv = FrenchWilsonE(F, s, cen, sig_F=sig, shell_idx=shell_idx,
                             n_shells=n_shells)
        for name, new, old in (
            ("eEobs", conv.E, fw["eEobs"]),
            ("weight", conv.weight, fw["DFAC"] * fw["DFAC"]),
            ("lerf1",
             build_lerf1_intensity(conv.E, cen, weight=conv.weight),
             build_lerf1_intensity(fw["eEobs"], cen,
                                   weight=fw["DFAC"] * fw["DFAC"])),
        ):
            nd, a, r = _dev(new, old)
            worst = max(worst, a)
            print(f"{pdb:>6s} {name:>12s} {new.numel():>8d} {nd:>7d} "
                  f"{a:>10.3e} {r:>10.3e}")

        # --- calc side: the one where the two implementations differ ----
        # A stand-in calc set; the check is about the normaliser, not the model.
        F_calc = (F * 1.37 + 5.0)
        old_E, _ = wilson_normalise(F_calc, s, n_shells)
        new_E = WilsonShellE(F_calc, s, cen, n_shells=n_shells).E
        nd, a, r = _dev(new_E, old_E)
        worst = max(worst, a)
        print(f"{pdb:>6s} {'E_calc':>12s} {new_E.numel():>8d} {nd:>7d} "
              f"{a:>10.3e} {r:>10.3e}")

    print(f"\nSEAM_WORST_ABS {worst:.6e}")
    print("SEAM_INERT" if worst == 0.0 else "SEAM_NOT_INERT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
