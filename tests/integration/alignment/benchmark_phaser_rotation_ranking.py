#!/usr/bin/env python
"""
Phaser FRF-only baseline for the rotation-ranking benchmark.

Mirrors `benchmark_rotation_ranking.py` but invokes Phaser in `MODE MR_FRF`
so we measure ONLY the rotation-function peak list — no rescore, no
translation. This isolates rotation-function quality from any
downstream-stage difference.

For each (PDB, seed):
  1. Apply the same `R_true` that the torchref benchmark applies.
  2. Write rotated PDB.
  3. Run `phenix.phaser` MODE MR_FRF with PEAKS ROT NUMBER 500.
  4. Parse `SOLU TRIAL` lines from the .sol; each gives Euler + RFZ.
  5. Convert to rotation matrices, compute symmetry-orbit rank-of-truth
     against R_true (modulo spacegroup), report rank + best angle.
  6. CSV output.

The orbit-distance convention is the same as the torchref benchmark:
   target = R_true   (verified empirically against the Phaser-placed PDB
                      in the prior MR_AUTO sweep — if the convention
                      were R_true.T, the placement's err° would not be
                      0.03° on 1DAW.)

`module load phenix/phenix-1.20-4459` must be done in the surrounding
slurm script.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from torchref.experimental.alignment.frf.rotation_utils import (
    rotation_angular_distance_deg,
    rotation_matrix_from_edmonds_euler,
)
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT


TEST_FILES = Path(
    "/das/work/p17/p17490/Peter/Library/work_trees_torchref/fix_alignment/tests/files"
)
PAIRS = {
    "1AK5": (TEST_FILES / "pdb" / "1AK5_with_H.pdb", TEST_FILES / "mtz" / "1AK5.mtz"),
    "1DAW": (TEST_FILES / "pdb" / "1DAW.pdb", TEST_FILES / "mtz" / "1DAW.mtz"),
    "2DQ6": (TEST_FILES / "pdb" / "2DQ6.pdb", TEST_FILES / "mtz" / "2DQ6.mtz"),
    "3A5V": (TEST_FILES / "pdb" / "3A5V.pdb", TEST_FILES / "mtz" / "3A5V.mtz"),
    "3E98": (TEST_FILES / "pdb" / "3E98.pdb", TEST_FILES / "mtz" / "3E98.mtz"),
    "3GR5": (TEST_FILES / "pdb" / "3GR5.pdb", TEST_FILES / "mtz" / "3GR5.mtz"),
    "3K7M": (TEST_FILES / "pdb" / "3K7M.pdb", TEST_FILES / "mtz" / "3K7M.mtz"),
    "3VRJ": (TEST_FILES / "pdb" / "3VRJ.pdb", TEST_FILES / "mtz" / "3VRJ.mtz"),
    "4BX9": (TEST_FILES / "pdb" / "4BX9.pdb", TEST_FILES / "mtz" / "4BX9.mtz"),
    "6G9X": (TEST_FILES / "pdb" / "6G9X.pdb", TEST_FILES / "mtz" / "6G9X.mtz"),
}


@dataclass
class PhaserRotPeak:
    """One row from Phaser's FRF peak list."""
    alpha_deg: float
    beta_deg: float
    gamma_deg: float
    rfz: float


def _random_rotation(seed: int) -> torch.Tensor:
    """Identical to torchref's benchmark — same seed → same R_true."""
    g = torch.Generator().manual_seed(int(seed))
    A = torch.randn(3, 3, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _discover_mtz_amplitude_labels(mtz_path: Path) -> tuple[str, str]:
    import gemmi
    mtz = gemmi.read_mtz_file(str(mtz_path))
    f_label, sigf_label = None, None
    for c in mtz.columns:
        if c.type == "F" and f_label is None:
            f_label = c.label
        elif c.type == "Q" and sigf_label is None:
            sigf_label = c.label
        if f_label and sigf_label:
            break
    if f_label is None or sigf_label is None:
        raise RuntimeError(f"no F/SIGF columns in {mtz_path}")
    return f_label, sigf_label


_SOLU_TRIAL_RE = re.compile(
    r"SOLU\s+TRIAL\s+ENSEMBLE\s+\S+\s+"
    r"EULER\s+([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)\s+"
    r"RF\s+([-+\d.eE]+)\s+RFZ\s+([-+\d.eE]+)",
    re.IGNORECASE,
)


def _parse_phaser_frf_peaks(sol_path: Path) -> List[PhaserRotPeak]:
    """Extract FRF peaks from Phaser's .rlist file:

        SOLU TRIAL ENSEMBLE search EULER 39.764 69.564 211.710 RF 78.6 RFZ 10.25
    """
    if not sol_path.exists():
        return []
    text = sol_path.read_text()
    out: List[PhaserRotPeak] = []
    for ln in text.splitlines():
        if "SOLU TRIAL" not in ln.upper():
            continue
        m = _SOLU_TRIAL_RE.search(ln)
        if m is None:
            continue
        a, b, g, _rf, rfz = (float(x) for x in m.groups())
        out.append(PhaserRotPeak(a, b, g, rfz))
    # Phaser writes peaks in descending-RFZ order. Re-sort defensively.
    out.sort(key=lambda p: p.rfz, reverse=True)
    return out


def _write_phaser_frf_kw(
    work: Path, *, mtz_path: Path, rotated_pdb: Path,
    f_label: str, sigf_label: str, root: str,
    n_peaks: int,
) -> Path:
    """Phaser keywords for FRF-only mode."""
    kw = work / "phaser_frf.kw"
    # Phaser keyword grammar (from PEAK.cc):
    #   PEAKS ROT SELECT {SIG|NUM|PERCENT|ALL}
    #   PEAKS ROT CUTOFF <num>      (interpretation depends on SELECT)
    #   PEAKS ROT CLUSTER {ON|OFF}
    # With `SELECT NUM` + `CUTOFF n`, Phaser keeps top-n peaks. Disable
    # clustering so symmetry-equivalents aren't merged before we rank them.
    kw.write_text(f"""TITLE FRF-only rotation ranking
MODE MR_FRF
HKLIN {mtz_path}
LABIN F={f_label} SIGF={sigf_label}
ENSEMBLE search PDB {rotated_pdb} IDENT 1.0
COMPOSITION BY AVERAGE
SEARCH ENSEMBLE search
PEAKS ROT SELECT ALL
PEAKS ROT CLUSTER OFF
PEAKS ROT LEVEL 0
ROOT {root}
""")
    return kw


def _run_phaser_frf(work: Path, kw_path: Path, timeout_s: int) -> tuple[int, float]:
    """Pipe keywords to `phenix.phaser` stdin (CCP4-keyword mode)."""
    t0 = time.time()
    keywords = kw_path.read_text()
    try:
        proc = subprocess.run(
            ["phenix.phaser"],
            cwd=str(work),
            input=keywords,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        rc = proc.returncode
        (work / "phaser.stdout").write_text(proc.stdout or "")
        (work / "phaser.stderr").write_text(proc.stderr or "")
    except subprocess.TimeoutExpired:
        rc = -1
    return rc, time.time() - t0


def _euler_zyz_to_R(alpha, beta, gamma) -> torch.Tensor:
    """Edmonds ZYZ active rotation: R = Rz(α) · Ry(β) · Rz(γ). Vectorised."""
    a, b, g = (torch.as_tensor(x, dtype=torch.float64) for x in (alpha, beta, gamma))
    ca, sa = a.cos(), a.sin()
    cb, sb = b.cos(), b.sin()
    cg, sg = g.cos(), g.sin()
    # R = Rz(a) Ry(b) Rz(g), 3x3 column-vector convention.
    R = torch.stack([
        torch.stack([ca*cb*cg - sa*sg,  -ca*cb*sg - sa*cg,  ca*sb], dim=-1),
        torch.stack([sa*cb*cg + ca*sg,  -sa*cb*sg + ca*cg,  sa*sb], dim=-1),
        torch.stack([-sb*cg,             sb*sg,             cb   ], dim=-1),
    ], dim=-2)
    return R


def orbit_rank(
    peaks: List[PhaserRotPeak],
    R_target: torch.Tensor,
    sym_mats: torch.Tensor,
    threshold_deg: float = 5.0,
) -> dict:
    """Symmetry-orbit-aware rank, vectorised over all peaks at once."""
    if not peaks:
        return {
            "best_rank": -1, "best_ang_deg": float("inf"),
            "best_rfz": 0.0, "top1_ang_deg": float("inf"),
            "any_below_threshold": False, "n_peaks": 0,
        }
    R_target = R_target.to(torch.float64).cpu()
    sym_mats = sym_mats.to(torch.float64).cpu()

    alphas = torch.tensor(
        [math.radians(p.alpha_deg) for p in peaks], dtype=torch.float64,
    )
    betas = torch.tensor(
        [math.radians(p.beta_deg) for p in peaks], dtype=torch.float64,
    )
    gammas = torch.tensor(
        [math.radians(p.gamma_deg) for p in peaks], dtype=torch.float64,
    )
    R_peaks = _euler_zyz_to_R(alphas, betas, gammas)         # (N, 3, 3)
    orbit = sym_mats @ R_target.unsqueeze(0)                 # (n_ops, 3, 3)

    # For each peak and each sym op: trace(R_peak · (S_k R_target)^T) / 2
    # Then arccos for angular distance; min over sym ops; argmin over peaks.
    Rk_T = orbit.transpose(-1, -2)                           # (n_ops, 3, 3)
    # Batched matrix-multiply: (N, 1, 3, 3) @ (1, n_ops, 3, 3) → (N, n_ops, 3, 3)
    M = R_peaks.unsqueeze(1) @ Rk_T.unsqueeze(0)
    tr = M.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)      # (N, n_ops)
    cos_a = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    angles = cos_a.arccos() * (180.0 / math.pi)              # (N, n_ops)
    per_peak_min, _ = angles.min(dim=-1)                     # (N,)

    top1_ang = float(per_peak_min[0].item())
    # Find first peak below threshold; otherwise return global argmin.
    below = (per_peak_min <= threshold_deg).nonzero(as_tuple=True)[0]
    if below.numel() > 0:
        rank = int(below[0].item())
    else:
        rank = int(per_peak_min.argmin().item())
    best_ang = float(per_peak_min[rank].item())
    best_rfz = peaks[rank].rfz

    return {
        "best_rank": rank,
        "best_ang_deg": best_ang,
        "best_rfz": best_rfz,
        "top1_ang_deg": top1_ang,
        "any_below_threshold": best_ang <= threshold_deg,
        "n_peaks": len(peaks),
    }


def run_one(
    pdb_key: str,
    seed: int,
    *,
    work_root: Path,
    n_peaks: int,
    threshold_deg: float,
    timeout_s: int,
    verbose: int,
) -> dict:
    pdb_path, mtz_path = PAIRS[pdb_key]
    print(f"\n=== PHASER FRF {pdb_key} seed={seed} ===", flush=True)

    t0 = time.time()
    model = ModelFT(device=torch.device("cpu")).load_pdb(str(pdb_path))
    data = ReflectionData(device="cpu").load_mtz(str(mtz_path))
    sym_mats = data.spacegroup.matrices.to(dtype=torch.float64)

    R_true = _random_rotation(seed)
    centroid = model.xyz().mean(0)
    rotated = model.rotate(R_true.to(model.dtype_float), center=centroid)

    work = work_root / f"{pdb_key}_seed{seed}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    rotated_pdb = work / "rotated.pdb"
    rotated.write_pdb(str(rotated_pdb))

    f_label, sigf_label = _discover_mtz_amplitude_labels(mtz_path)
    root = "phaser_frf"
    kw_path = _write_phaser_frf_kw(
        work, mtz_path=mtz_path, rotated_pdb=rotated_pdb,
        f_label=f_label, sigf_label=sigf_label, root=root,
        n_peaks=n_peaks,
    )
    rc, t_phaser = _run_phaser_frf(work, kw_path, timeout_s)
    # Phaser FRF writes the rotation peak list to <ROOT>.rlist, not .sol.
    # (.sol is reserved for full MR_AUTO solutions.)
    rlist_path = work / f"{root}.rlist"

    peaks = _parse_phaser_frf_peaks(rlist_path)
    if verbose:
        print(f"  rc={rc}  n_peaks={len(peaks)}  t={t_phaser:.1f}s", flush=True)

    # Test convention against R_true (the same target the torchref benchmark uses).
    rank_R_true = orbit_rank(peaks, R_true, sym_mats, threshold_deg=threshold_deg)
    # Also evaluate against R_true.T to check whether Phaser's Euler points
    # the opposite direction. The smaller `top1_ang_deg` indicates the
    # correct convention for this pipeline.
    rank_R_true_T = orbit_rank(peaks, R_true.T, sym_mats, threshold_deg=threshold_deg)

    if verbose:
        print(
            f"  against R_true:   rank={rank_R_true['best_rank']:>4}  "
            f"ang={rank_R_true['best_ang_deg']:.2f}°  "
            f"top1ang={rank_R_true['top1_ang_deg']:.2f}°",
            flush=True,
        )
        print(
            f"  against R_true.T: rank={rank_R_true_T['best_rank']:>4}  "
            f"ang={rank_R_true_T['best_ang_deg']:.2f}°  "
            f"top1ang={rank_R_true_T['top1_ang_deg']:.2f}°",
            flush=True,
        )

    # Pick the convention with smaller top1 angular distance as the
    # "canonical" measurement for this row.
    pick = (
        rank_R_true if rank_R_true["top1_ang_deg"] <= rank_R_true_T["top1_ang_deg"]
        else rank_R_true_T
    )
    convention = "R_true" if pick is rank_R_true else "R_true.T"

    return {
        "pdb": pdb_key,
        "seed": seed,
        "spacegroup": str(data.spacegroup),
        "phaser_exit": rc,
        "n_peaks": len(peaks),
        "convention_used": convention,
        "rank": pick["best_rank"],
        "ang_deg": pick["best_ang_deg"],
        "rfz": pick["best_rfz"],
        "top1_ang_deg": pick["top1_ang_deg"],
        "rank_R_true": rank_R_true["best_rank"],
        "ang_R_true_deg": rank_R_true["best_ang_deg"],
        "rank_R_true_T": rank_R_true_T["best_rank"],
        "ang_R_true_T_deg": rank_R_true_T["best_ang_deg"],
        "phaser_time_s": t_phaser,
        "total_time_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default=None, choices=sorted(PAIRS.keys()))
    ap.add_argument("--seed", type=int, default=None,
                    help="Top-level seed; same semantics as the torchref "
                         "benchmark so the per-trial R_true matches.")
    ap.add_argument("--n-trials", type=int, default=1)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--n-peaks", type=int, default=500,
                    help="PEAKS ROT NUMBER for Phaser FRF (default 500 to "
                         "match the torchref benchmark's n_rotation_peaks).")
    ap.add_argument("--threshold-deg", type=float, default=5.0)
    ap.add_argument("--timeout-s", type=int, default=1200)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    if args.seed is None:
        args.seed = int(time.time())
    rng = random.Random(args.seed)
    print(f"top-level seed = {args.seed}", flush=True)

    if args.sweep:
        worklist = [(pdb, rng.randint(0, 10 ** 9))
                    for pdb in sorted(PAIRS.keys())
                    for _ in range(args.n_trials)]
    else:
        worklist = []
        for _ in range(args.n_trials):
            pdb = args.pdb if args.pdb is not None else rng.choice(list(PAIRS.keys()))
            worklist.append((pdb, rng.randint(0, 10 ** 9)))

    work_root = Path(args.workdir or f"/tmp/phaser_frf_{int(time.time())}")
    work_root.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv or f"phaser_frf_results_{int(time.time())}.csv")

    rows = []
    for pdb_key, trial_seed in worklist:
        try:
            r = run_one(
                pdb_key, trial_seed,
                work_root=work_root,
                n_peaks=args.n_peaks,
                threshold_deg=args.threshold_deg,
                timeout_s=args.timeout_s,
                verbose=args.verbose,
            )
            rows.append(r)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  TRIAL FAILED on {pdb_key} seed={trial_seed}: {exc!r}",
                  flush=True)
            rows.append({"pdb": pdb_key, "seed": trial_seed,
                         "error": repr(exc)})

    if rows:
        cols = sorted({k for r in rows for k in r})
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {out_csv}", flush=True)

    # Console summary
    print("\n=== Phaser FRF summary (best rank / best ang°, both conventions) ===",
          flush=True)
    print(
        f"{'pdb':>6}  {'seed':>10}  {'n_pk':>5}  "
        f"{'pick':>9}  {'rank':>4}  {'ang°':>6}  {'RFZ':>6}  "
        f"{'R_true rank':>11}  {'R_T rank':>9}  {'t(s)':>6}",
        flush=True,
    )
    for r in rows:
        if r.get("error"):
            print(f"{r['pdb']:>6}  {r['seed']:>10}  FAILED: {r['error']}",
                  flush=True)
            continue
        print(
            f"{r['pdb']:>6}  {r['seed']:>10}  {r['n_peaks']:>5}  "
            f"{r['convention_used']:>9}  {r['rank']:>4}  "
            f"{r['ang_deg']:>6.2f}  {r['rfz']:>6.2f}  "
            f"{r['rank_R_true']:>11}  {r['rank_R_true_T']:>9}  "
            f"{r['phaser_time_s']:>6.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
