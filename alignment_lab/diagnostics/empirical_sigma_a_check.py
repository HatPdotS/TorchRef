"""What does the rotation function's "empirical" sigma_A actually evaluate to?

``empirical_sigma_a`` divides the observed Wilson curve by the calculated one
and takes ``sqrt(min(R, 1/R))``. The two curves sit on different absolute
scales -- the MTZ's arbitrary one and the model's electron scale -- and the
function does not remove that factor, so the ratio's level, not only its shape,
sets the answer. This records the ratio and the resulting sigma_A by resolution
during a real rotation search.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import BENCH_PDBS, load_case  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="1DAW", choices=list(BENCH_PDBS))
    args = ap.parse_args()

    import torchref.experimental.alignment.frf.api as api
    from torchref.experimental.alignment import rotation_search

    seen = {}
    real = api.empirical_sigma_a

    def spy(sigma_obs, sigma_calc, **kw):
        out = real(sigma_obs, sigma_calc, **kw)
        seen["ratio"] = (sigma_obs / sigma_calc).detach().cpu()
        seen["sigma_a"] = out.detach().cpu()
        return out

    api.empirical_sigma_a = spy
    model, data = load_case(args.pdb)
    rotation_search(model, data, model_error_A=0.8, n_peaks=5)
    r, sa = seen["ratio"], seen["sigma_a"]
    q = lambda x: [round(float(v), 4) for v in torch.quantile(x, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=x.dtype))]
    print(f"ROW pdb={args.pdb} ratio_quantiles={q(r)} sigma_a_quantiles={q(sa)} "
          f"ratio_geomean={float(r.log().mean().exp()):.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
