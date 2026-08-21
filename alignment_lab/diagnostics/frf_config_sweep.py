"""Settle the FRF's remaining free constants by measurement, one arm each.

Four engine settings are still switches because nobody chose a value. Making
the rotation search a three-input call means choosing them, and each choice
gets a number first:

``lmax_cap``
    The signature default is 48, its own docstring claims 100, and the
    benchmarks run 64. Phaser's ``DEF_CLMN_LMAX`` is 100. The "high l
    under-determines the SH modes" argument for 48 predates the dense P1-box
    calc, so it is not evidence about the current engine.
anisotropy
    ``production`` is the log-space fit with no intercept; ``fixed_fit`` is the
    intensity-space replacement; ``iso_only`` keeps its radial part; ``no_aniso``
    drops the correction. Measured before at seven structures, where
    ``fixed_fit`` was indistinguishable from ``no_aniso`` in aggregate.
``_orbit_unroll``
    Off, on the strength of a run that predates the reciprocal-space
    convention fix, so its evidence is void.
Patterson radius
    Never exercised in production. Two structures want radii a factor 2.4
    apart with no rule to pick between them, so the candidate is the *union*:
    two runs merged by z-score. It doubles the cost, so it has to earn it.

Every arm runs in one process per (structure, trial) cell, so the paired
comparison against ``production`` is exact. ``production_dup`` repeats the
baseline arm verbatim: it measures the engine's own run-to-run spread, which
bounds how small a real effect this sweep can resolve.

Usage
-----
    python -m diagnostics.frf_config_sweep --pdb 3GR5 --trial 0
    python -m diagnostics.frf_config_sweep --pdb 3GR5 --trials 10 --stage 2
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import (BENCH_PDBS, FRFConfig, aniso_arm, merge_peak_lists,  # noqa: E402
                 orbit_rank, rotated_case, run_frf, seed_for, tensor_report)
from lab.results import append_row, provenance  # noqa: E402

EXPERIMENT = "frf_config_sweep"

#: Suppression radius for the union merge. The engine uses
#: ``max(2 * grid_sampling_deg, 6)`` internally, and the production sampling is
#: 3 degrees, so 6 degrees keeps the merged list on the same footing.
UNION_NMS_DEG = 6.0


@dataclass(frozen=True)
class Arm:
    """One engine configuration to measure.

    ``radius_scales`` with more than one entry means the union arm: one FRF
    evaluation per scale, merged by z-score.
    """

    name: str
    lmax_cap: int = 64
    aniso: str = "production"
    orbit_unroll: bool = False
    radius_scales: Tuple[float, ...] = (1.0,)

    def config(self, base: FRFConfig) -> Tuple[FRFConfig, ...]:
        out = []
        for scale in self.radius_scales:
            extra: Dict[str, object] = {"_orbit_unroll": self.orbit_unroll}
            if scale != 1.0:
                extra["frf_patterson_radius_scale"] = scale
            out.append(FRFConfig(
                d_min=base.d_min, d_max=base.d_max, n_shells=base.n_shells,
                n_peaks=base.n_peaks, lmax_cap=self.lmax_cap,
                dense_pad=base.dense_pad, extra=extra,
            ))
        return tuple(out)


def _factorial_arms() -> Tuple[Arm, ...]:
    """lmax_cap x anisotropy, plus the repeat-baseline control."""
    arms = [Arm("production_dup")]
    for cap in (48, 64, 100):
        for aniso in ("production", "fixed_fit", "iso_only", "no_aniso"):
            arms.append(Arm(f"cap{cap}_{aniso}", lmax_cap=cap, aniso=aniso))
    return tuple(arms)


def _followup_arms(cap: int, aniso: str) -> Tuple[Arm, ...]:
    """One-at-a-time from the winning cell of stage 1."""
    base = Arm(f"cap{cap}_{aniso}", lmax_cap=cap, aniso=aniso)
    return (
        base,
        Arm(f"{base.name}_unroll", lmax_cap=cap, aniso=aniso, orbit_unroll=True),
        Arm(f"{base.name}_union", lmax_cap=cap, aniso=aniso,
            radius_scales=(1.0, 0.5)),
    )


#: The baseline every paired difference is taken against: today's shipped
#: configuration (broken anisotropy fit, cap 64, no unroll, single radius).
BASELINE = Arm("production", lmax_cap=64, aniso="production")


def run_one(pdb: str, trial: int, arm: Arm, base: FRFConfig,
            *, thr_deg: float) -> dict:
    seed = seed_for(pdb, trial)
    model, data, R_true = rotated_case(pdb, seed)
    configs = arm.config(base)

    captured: dict = {}
    peak_lists = []
    t0 = time.time()
    for cfg in configs:
        with aniso_arm(arm.aniso if arm.aniso != "production" else "production",
                       data, d_min=cfg.d_min, d_max=cfg.d_max,
                       captured=captured):
            res = run_frf(model, data, cfg, capture_arf=False, verbose=0)
        peak_lists.append(res.peaks)
    seconds = time.time() - t0

    peaks = (peak_lists[0] if len(peak_lists) == 1 else
             merge_peak_lists(peak_lists, n_peaks=base.n_peaks,
                              nms_radius_deg=UNION_NMS_DEG))

    rank, ang = orbit_rank(
        peaks, R_true, data.spacegroup.matrices.to(torch.float64).cpu(),
        reciprocal_basis=data.cell.reciprocal_basis_matrix.to(torch.float64).cpu(),
        side="left", frame="cart", thr_deg=thr_deg,
    )
    row = {"experiment": EXPERIMENT, "pdb": pdb, "trial": trial,
           "arm": arm.name, "seed": seed}
    row.update(provenance())
    row.update(configs[0].as_row())
    row.update({
        "arm_lmax_cap": arm.lmax_cap,
        "arm_aniso": arm.aniso,
        "arm_orbit_unroll": int(arm.orbit_unroll),
        "arm_radius_scales": "|".join(f"{s:g}" for s in arm.radius_scales),
        "n_frf_calls": len(configs),
        "spacegroup": str(data.spacegroup.hm),
        "truth_rank": rank,
        # orbit_rank returns -1 for "no peak within thr_deg". A miss must not
        # sort as a good rank, so for pairing it counts as worse than the worst
        # hit, i.e. the length of the peak list.
        "rank_for_compare": rank if rank >= 0 else base.n_peaks,
        "found": int(rank >= 0),
        "in_top20": int(0 <= rank < 20),
        "truth_angle_deg": None if ang is None else round(float(ang), 3),
        "n_peaks_found": len(peaks),
        "orbit_side": "left", "orbit_frame": "cart", "thr_deg": thr_deg,
        "seconds": round(seconds, 1),
    })
    # Emit both tensor reports for every arm, blank where the arm does not
    # produce one: a row carrying columns the file's header lacks is a schema
    # error, and silently-widened rows lose exactly these values.
    for tag in ("raw", "fixed"):
        if tag in captured:
            row.update(tensor_report(captured[tag], tag))
        else:
            row.update({f"{tag}_B_min": "", f"{tag}_B_max": "",
                        f"{tag}_B_spread": ""})
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", required=True, choices=list(BENCH_PDBS))
    ap.add_argument("--trial", type=int, default=None,
                    help="single trial index; omit to run --trials of them")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2),
                    help="1 = lmax x aniso factorial; 2 = follow-ups")
    ap.add_argument("--stage2-cap", type=int, default=64)
    ap.add_argument("--stage2-aniso", default="fixed_fit")
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=15.0)
    ap.add_argument("--n-peaks", type=int, default=500)
    ap.add_argument("--thr-deg", type=float, default=5.0)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    arms = (BASELINE,) + (
        _factorial_arms() if args.stage == 1
        else _followup_arms(args.stage2_cap, args.stage2_aniso)
    )
    base = FRFConfig(d_min=args.d_min, d_max=args.d_max, n_peaks=args.n_peaks)

    if args.out_csv:
        csv_path = Path(args.out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path(args.outdir) if args.outdir else (
            Path(__file__).resolve().parents[1] / "runs" / EXPERIMENT)
        outdir.mkdir(parents=True, exist_ok=True)
        csv_path = outdir / f"{EXPERIMENT}_{args.pdb}.csv"

    trials = [args.trial] if args.trial is not None else list(range(args.trials))
    print(f"{args.pdb}: stage {args.stage}, {len(arms)} arms x {len(trials)} "
          f"trial(s)", flush=True)

    ranks: Dict[str, list] = {a.name: [] for a in arms}
    n_fail = 0
    for trial in trials:
        for arm in arms:
            try:
                row = run_one(args.pdb, trial, arm, base, thr_deg=args.thr_deg)
            except Exception as exc:
                n_fail += 1
                ranks[arm.name].append(None)
                print(f"  trial {trial} {arm.name}: FAILED {type(exc).__name__}: "
                      f"{exc}", flush=True)
                continue
            append_row(csv_path, row)
            ranks[arm.name].append(row["rank_for_compare"])
            shown = (str(row["truth_rank"]) if row["found"]
                     else f"miss@{row['truth_angle_deg']:.0f}deg")
            print(f"  trial {trial} {arm.name:<26} rank={shown:<12} "
                  f"top20={row['in_top20']} {row['seconds']:>6.1f}s", flush=True)

    base_ranks = ranks[BASELINE.name]
    print("\npaired vs production (negative = better rank):", flush=True)
    for arm in arms:
        if arm.name == BASELINE.name:
            continue
        d = [(a - b) for a, b in zip(ranks[arm.name], base_ranks)
             if a is not None and b is not None]
        if not d:
            print(f"  {arm.name:<26} no paired trials", flush=True)
            continue
        sd = sorted(d)
        med = (sd[len(sd) // 2] if len(sd) % 2
               else 0.5 * (sd[len(sd) // 2 - 1] + sd[len(sd) // 2]))
        print(f"  {arm.name:<26} n={len(d):<3} better={sum(x < 0 for x in d)} "
              f"same={sum(x == 0 for x in d)} worse={sum(x > 0 for x in d)} "
              f"median={med:+.1f}  per-trial={d}", flush=True)
    print(f"\nwrote {csv_path}  ({n_fail} failures)", flush=True)
    return 1 if n_fail == len(arms) * len(trials) else 0


if __name__ == "__main__":
    raise SystemExit(main())
