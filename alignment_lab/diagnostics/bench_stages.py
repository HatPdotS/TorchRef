"""Stage-resolved timing of one cold FRF call.

The eventual goal is placement cheap enough to sit inside a training loop, so
what matters is where the time goes, not just the total. Stage functions are
wrapped for the duration of one call and restored afterwards.

Timings are **cold by default**: the first call in a process pays one-off costs
(parametrisation, grid setup, any compile). Pass ``--warmup`` for steady-state
numbers, and say which one a reported figure is.

Usage::

    python alignment_lab/diagnostics/bench_stages.py --pdb 1DAW --lmax-cap 64
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import (BENCH_PDBS, FRFConfig, ResultWriter, orbit_rank,  # noqa: E402
                 rotated_case, run_frf, seed_for)
from lab.frf import patched  # noqa: E402

#: (module path, attribute) pairs timed individually.
STAGES = [
    ("torchref.experimental.alignment.frf.dense_calc", "dense_calc_via_box"),
    ("torchref.experimental.alignment.frf.api", "phaser_rotation_search"),
    ("torchref.experimental.alignment.frf.data_mr", "spherical_bessel_table"),
    ("torchref.experimental.alignment.frf.data_mr", "bessel_sh_expand"),
    ("torchref.experimental.alignment.frf.data_mr", "cross_correlate_xi"),
    ("torchref.experimental.alignment.frf.wigner_d", "wigner_contraction_per_beta"),
    ("torchref.experimental.alignment.frf.sitelist_ang", "evaluate_rotation_function"),
    ("torchref.experimental.alignment.frf.peak_finder", "find_rotation_peaks"),
]


def _instrument(stack, totals, counts, skipped):
    """Wrap each resolvable stage with a timer, via the exit stack."""
    import importlib

    for mod_path, attr in STAGES:
        try:
            mod = importlib.import_module(mod_path)
            original = getattr(mod, attr)
        except (ImportError, AttributeError):
            # Report it: a silently skipped stage reads as "that stage is free".
            skipped.append(f"{mod_path.rsplit('.', 1)[-1]}.{attr}")
            continue

        def make(orig, key):
            def timed(*a, **k):
                t0 = time.perf_counter()
                try:
                    return orig(*a, **k)
                finally:
                    totals[key] += time.perf_counter() - t0
                    counts[key] += 1
            return timed

        # Register at zero so a resolved-but-never-called stage still prints:
        # an absent row is indistinguishable from a free one.
        totals[attr] += 0.0
        counts[attr] += 0
        stack.enter_context(patched(mod, attr, make(original, attr)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="1DAW", choices=list(BENCH_PDBS))
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--lmax-cap", type=int, default=64)
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=15.0)
    ap.add_argument("--n-peaks", type=int, default=500)
    ap.add_argument("--warmup", action="store_true",
                    help="discard one call first and report steady state")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    from contextlib import ExitStack

    seed = seed_for(args.pdb, args.trial)
    rotated, data, R_true = rotated_case(args.pdb, seed)
    cfg = FRFConfig(d_min=args.d_min, d_max=args.d_max,
                    n_peaks=args.n_peaks, lmax_cap=args.lmax_cap)

    if args.warmup:
        run_frf(rotated, data, cfg, capture_arf=False)

    totals, counts, skipped = defaultdict(float), defaultdict(int), []
    with ExitStack() as stack:
        _instrument(stack, totals, counts, skipped)
        res = run_frf(rotated, data, cfg)

    sym = data.spacegroup.matrices.to(torch.float64).cpu()
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
    rank, ang = orbit_rank(res.peaks, R_true, sym, reciprocal_basis=rec)

    kind = "steady-state" if args.warmup else "cold"
    print(f"=== {args.pdb} lmax_cap={args.lmax_cap} ({kind}) ===")
    print(f"  {'stage':28s} {'calls':>6s} {'seconds':>9s} {'% of run':>9s}")
    accounted = 0.0
    for key, secs in sorted(totals.items(), key=lambda kv: -kv[1]):
        accounted += secs
        print(f"  {key:28s} {counts[key]:6d} {secs:9.3f} "
              f"{100.0 * secs / max(res.seconds, 1e-9):9.1f}")
    print(f"  {'(unattributed)':28s} {'':6s} {res.seconds - accounted:9.3f} "
          f"{100.0 * (res.seconds - accounted) / max(res.seconds, 1e-9):9.1f}")
    print(f"  {'TOTAL':28s} {'':6s} {res.seconds:9.3f}")
    if skipped:
        print(f"  NOT INSTRUMENTED (renamed or absent): {', '.join(skipped)}")
    print(f"  truth rank {rank} at {ang:.2f} deg")

    if args.out_csv:
        w = ResultWriter(args.out_csv, "bench_stages",
                         extra_fields=("timing_kind", "total_seconds",
                                       "unattributed_seconds") +
                                      tuple(f"t_{a}" for _, a in STAGES))
        row = dict(pdb=args.pdb, seed=seed, trial=args.trial,
                   spacegroup=str(data.spacegroup), n_ops=int(sym.shape[0]),
                   truth_rank=rank, truth_angle_deg=round(ang, 4),
                   orbit_side="left", orbit_frame="cart",
                   lmax_cap=args.lmax_cap, d_min=args.d_min, d_max=args.d_max,
                   device="cpu", timing_kind=kind,
                   total_seconds=round(res.seconds, 4),
                   unattributed_seconds=round(res.seconds - accounted, 4))
        for _, attr in STAGES:
            row[f"t_{attr}"] = round(totals.get(attr, 0.0), 4)
        w.write(**row)
        print(f"  wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
