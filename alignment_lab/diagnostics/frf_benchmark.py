"""The rotation search's standing benchmark: accuracy, memory and runtime.

One row per (structure, trial, arm), carrying all three so a change cannot
improve one at the silent expense of another:

**Accuracy** -- where the true orientation lands in the peak list, and whether
it is inside the top ``--top-n``. That window, not rank 0, is the thing that
matters: the placement search carries its top candidates forward, so rank 7 and
rank 0 are the same outcome downstream and rank 223 is not. Reported per
structure, because an average lets one failing space group be cancelled by nine
easy ones.

**Memory** -- peak resident set over the search, as a delta over the value on
entry, plus the process high-water mark. Read
:mod:`alignment_lab.lab.profile` for what a sampler can and cannot see. Memory
is why the bandwidth ceiling is not simply "as high as possible": cap 100 needs
more than 32 GB on the two P432 structures, where the symmetry expansion
multiplies the reflection count by 24.

**Runtime** -- total, plus per-stage. Cold by default, since a caller placing one
model pays cold costs; ``--warmup`` gives steady state, and the row says which.
Every row carries a fixed calibration workload timed in the same process, and the
node's identity: wall clock on a shared cluster measures the cluster unless it is
normalised or confined to one node. Run with ``--exclusive`` and compare
``seconds_per_calibration`` across nodes, or raw seconds only within a node.

Usage
-----
    python -m diagnostics.frf_benchmark --pdb 1DAW --trials 3
    python -m diagnostics.frf_benchmark --pdb 3K7M --arms cap48,cap64,cap100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import (BENCH_PDBS, FRFConfig, orbit_rank, rotated_case,  # noqa: E402
                 run_frf, seed_for)
from lab.profile import (FRF_STAGES, PeakMemory, calibration_seconds,  # noqa: E402
                         exclusive_times, host_info, stage_timers)
from lab.results import append_row, provenance  # noqa: E402

EXPERIMENT = "frf_benchmark"

#: Named bandwidth arms. ``shipped`` reads the engine's own constant, so the
#: benchmark follows the code rather than restating it -- if the constant moves
#: and this row does not, the harness is measuring history.
ARMS = {"cap48": 48, "cap64": 64, "cap100": 100, "shipped": None}


def _shipped_lmax_cap() -> int:
    import importlib

    return importlib.import_module(
        "torchref.experimental.alignment.rotation_search").LMAX_CAP


def warmup_run(pdb: str, lmax_cap: int, n_peaks: int) -> float:
    """One discarded search, to move the process's start-up out of the way.

    On this cluster the first real computation in a process is dominated by
    PyTorch loading its backend libraries. That happens lazily on first use
    rather than at ``import torch``, and the environment lives on GPFS, where it
    is tens of seconds of many small reads. Measured on 3A5V: **41.4 s for the
    first search against 1.8 s for the same search afterwards**, with 39.4 of
    those seconds attributable to no stage at all.

    Without this, whichever arm runs first carries the lot and reads as an order
    of magnitude slower than it is. Run at full fidelity rather than on a token
    problem, so the kernels and FFT plans the measured searches use are the ones
    already paid for.

    Returns the seconds it took. Every row carries it: the cost is real and
    worth reporting, it just is not the search's.
    """
    model, data, _ = rotated_case(pdb, seed_for(pdb, 0))
    t0 = time.perf_counter()
    run_frf(model, data, FRFConfig(n_peaks=n_peaks, lmax_cap=lmax_cap),
            capture_arf=False, verbose=0)
    return time.perf_counter() - t0


def run_one(pdb: str, trial: int, arm: str, *, n_peaks: int, top_n: int,
            thr_deg: float, warmup: bool, mem_interval_s: float,
            prewarm_seconds: float = float("nan"),
            first_in_process: bool = False) -> dict:
    """One measurement: accuracy, memory and runtime for a single search."""
    lmax_cap = ARMS[arm] if ARMS[arm] is not None else _shipped_lmax_cap()
    seed = seed_for(pdb, trial)
    model, data, R_true = rotated_case(pdb, seed)
    cfg = FRFConfig(n_peaks=n_peaks, lmax_cap=lmax_cap)

    if warmup:
        run_frf(model, data, cfg, capture_arf=False, verbose=0)

    # Calibrate before the measurement, so a node that is busy *now* is visible.
    calib = calibration_seconds()

    mem = PeakMemory(interval_s=mem_interval_s)
    t0 = time.perf_counter()
    with mem.window() as mem_out, stage_timers() as (totals, counts, unresolved):
        res = run_frf(model, data, cfg, capture_arf=False, verbose=0)
    wall = time.perf_counter() - t0

    rank, ang = orbit_rank(
        res.peaks, R_true, data.spacegroup.matrices.to(torch.float64).cpu(),
        reciprocal_basis=data.cell.reciprocal_basis_matrix.to(torch.float64).cpu(),
        side="left", frame="cart", thr_deg=thr_deg,
    )
    # Exclusive, not inclusive: the nested stages would otherwise be counted
    # twice and "unattributed" could come out negative.
    excl = exclusive_times(totals)
    attributed = sum(v for v in excl.values() if v == v)

    row = {"experiment": EXPERIMENT, "pdb": pdb, "trial": trial, "arm": arm,
           "seed": seed}
    row.update(provenance())
    row.update(host_info())
    row.update({
        "spacegroup": str(data.spacegroup.hm),
        "n_ops": int(data.spacegroup.matrices.shape[0]),
        "n_atoms": int(model.xyz().shape[0]),
        "n_reflections": int(data.hkl.shape[0]),
        "lmax_cap": lmax_cap,
        "n_peaks": n_peaks,
        # --- accuracy ---
        "truth_rank": rank,
        # orbit_rank returns -1 when nothing matched. That must not order as a
        # good rank, so for any comparison a miss counts as worse than the worst
        # hit, i.e. the length of the peak list.
        "rank_for_compare": rank if rank >= 0 else n_peaks,
        "found": int(rank >= 0),
        "in_top_n": int(0 <= rank < top_n),
        "top_n": top_n,
        "truth_angle_deg": None if ang is None else round(float(ang), 3),
        "n_peaks_found": len(res.peaks),
        "orbit_side": "left", "orbit_frame": "cart", "thr_deg": thr_deg,
        # --- runtime ---
        "timing_kind": "steady" if warmup else "post_warmup",
        "prewarm_seconds": round(prewarm_seconds, 2),
        # Flagged rather than assumed away: if the warm-up ever misses a shared
        # cost, it lands in this row and stays identifiable.
        "first_in_process": int(first_in_process),
        # Flagged rather than assumed away: if the pre-warm ever misses a shared
        # cost, it lands here and is identifiable.
        "first_in_process": int(first_in_process),
        "seconds": round(wall, 3),
        "seconds_attributed": round(attributed, 3),
        "seconds_unattributed": round(wall - attributed, 3),
        "calibration_seconds": round(calib, 5),
        "seconds_per_calibration": round(wall / max(calib, 1e-9), 1),
        # --- memory ---
        **mem_out,
        "stages_unresolved": "|".join(unresolved),
    })
    # Inclusive time for reading a single stage, exclusive for adding them up.
    for _, attr in FRF_STAGES:
        row[f"t_{attr}"] = round(totals.get(attr, float("nan")), 4)
        row[f"x_{attr}"] = round(excl.get(attr, float("nan")), 4)
        row[f"n_{attr}"] = counts.get(attr, 0)
    return row


def _fmt(row: dict) -> str:
    hit = "yes" if row["in_top_n"] else "NO "
    return (f"  {row['arm']:<8} rank={str(row['truth_rank']):<6} "
            f"top{row['top_n']}={hit} "
            f"{row['seconds']:>7.2f}s  peak {row['rss_peak_mb']:>8.0f} MB "
            f"(+{row['rss_delta_mb']:>7.0f})  {row['seconds_per_calibration']:>7.1f} cal")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", required=True, choices=list(BENCH_PDBS))
    ap.add_argument("--trial", type=int, default=None,
                    help="single trial; omit to run --trials of them")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--arms", default="shipped",
                    help=f"comma-separated, from {sorted(ARMS)}")
    ap.add_argument("--n-peaks", type=int, default=500)
    ap.add_argument("--top-n", type=int, default=20,
                    help="candidates the placement search carries forward")
    ap.add_argument("--thr-deg", type=float, default=5.0)
    ap.add_argument("--warmup", action="store_true",
                    help="discard one search first and report steady state")
    ap.add_argument("--no-warmup-run", dest="prewarm", action="store_false",
                    help="skip the discarded warm-up search; the first "
                         "measurement then carries the process start-up cost")
    ap.add_argument("--mem-interval", type=float, default=0.02,
                    help="RSS sampling period in seconds")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; expected {sorted(ARMS)}")
    trials = [args.trial] if args.trial is not None else list(range(args.trials))

    csv_path = None
    if args.out_csv:
        csv_path = Path(args.out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    prewarm_s = float("nan")
    if args.prewarm:
        prewarm_s = warmup_run(args.pdb, ARMS[arms[0]] or _shipped_lmax_cap(),
                               args.n_peaks)
        print(f"warm-up search: {prewarm_s:.1f}s -- the process's start-up, "
              f"mostly PyTorch loading its backend off GPFS. Excluded from the "
              f"measurements below and reported as prewarm_seconds.", flush=True)

    info = host_info()
    print(f"{args.pdb}: {len(arms)} arm(s) x {len(trials)} trial(s), "
          f"{'steady-state' if args.warmup else 'post-warm-up'}", flush=True)
    print(f"  host {info['host']} / {info['torch_threads']} threads / "
          f"{info['cpu_model'] or 'unknown cpu'}", flush=True)

    rows, n_fail = [], 0
    first = True
    for trial in trials:
        print(f" trial {trial}", flush=True)
        for arm in arms:
            try:
                row = run_one(args.pdb, trial, arm, n_peaks=args.n_peaks,
                              top_n=args.top_n, thr_deg=args.thr_deg,
                              warmup=args.warmup,
                              mem_interval_s=args.mem_interval,
                              first_in_process=first)
                first = False
            except Exception as exc:
                n_fail += 1
                print(f"  {arm:<8} FAILED {type(exc).__name__}: {exc}", flush=True)
                continue
            rows.append(row)
            if csv_path:
                append_row(csv_path, row)
            print(_fmt(row), flush=True)
            if row["stages_unresolved"]:
                print(f"    NOT INSTRUMENTED: {row['stages_unresolved']}",
                      flush=True)

    if rows:
        def med(key):
            vals = sorted(r.get(key, 0.0) or 0.0 for r in rows)
            return vals[len(vals) // 2]

        med_total = med("seconds")
        print("\nwhere the time goes (median over the rows above; exclusive of "
              "nested stages, so the column sums)", flush=True)
        print(f"  {'stage':32s} {'excl s':>8s} {'%':>6s} {'incl s':>8s}",
              flush=True)
        order = sorted((a for _, a in FRF_STAGES), key=lambda a: -med(f"x_{a}"))
        for attr in order:
            x, t = med(f"x_{attr}"), med(f"t_{attr}")
            if t <= 0:
                continue
            print(f"  {attr:32s} {x:8.3f} {100 * x / max(med_total, 1e-9):6.1f} "
                  f"{t:8.3f}", flush=True)
        unatt = med("seconds_unattributed")
        print(f"  {'(unattributed)':32s} {unatt:8.3f} "
              f"{100 * unatt / max(med_total, 1e-9):6.1f}", flush=True)
    if csv_path:
        print(f"\nwrote {csv_path}  ({n_fail} failures)", flush=True)
    return 1 if rows == [] else 0


if __name__ == "__main__":
    raise SystemExit(main())
