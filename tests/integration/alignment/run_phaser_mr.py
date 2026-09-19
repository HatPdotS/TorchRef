#!/usr/bin/env python
"""
Phaser-MR baseline for the same problem `run_random_pdb_fit.py` runs against
torchref's alignment pipeline.

Flow:
  1. Pick a PDB / MTZ pair from `tests/files`.
  2. Load the model + data; apply the SAME random rotation R_true that
     `run_random_pdb_fit.py` would (same seed → same R_true).
  3. Write the rotated model to a temporary PDB.
  4. Invoke `phenix.phaser` MODE MR_AUTO on (rotated_model, original_mtz).
  5. Parse Phaser's `.sum` for LLG / TFZ / R-factor.
  6. Load Phaser's placed PDB; compute angular distance to the canonical
     model, modulo the spacegroup symmetry orbit (same metric as
     `run_random_pdb_fit.py`'s `aligned-vs-canonical` line).
  7. Append a CSV row.

The script does NOT compute torchref's Scaler R-work on Phaser's placement —
per the design decision, both pipelines report their OWN R-work and we
display them side-by-side. Differences between the columns are partly
scaler-vs-scaler effects, not just placement quality.

Requires `module load phenix/phenix-1.20-4459` to be done in the surrounding
slurm script (PATH-resolved at subprocess.run time).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import torch

from torchref.base.metrics.rfactor import rfactor_work_free
from torchref.experimental.alignment.frf.rotation_utils import rotation_angular_distance_deg
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT
from torchref.scaling import Scaler


# Same PAIRS as run_random_pdb_fit.py — keep in sync.
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


def _random_rotation(seed: int) -> torch.Tensor:
    """Identical to run_random_pdb_fit.py — must produce byte-equal R_true."""
    g = torch.Generator().manual_seed(int(seed))
    A = torch.randn(3, 3, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _kabsch_rotation(xyz_a: torch.Tensor, xyz_b: torch.Tensor) -> torch.Tensor:
    """Identical to run_random_pdb_fit.py."""
    a = (xyz_a.detach() - xyz_a.detach().mean(0)).to(torch.float64)
    b = (xyz_b.detach() - xyz_b.detach().mean(0)).to(torch.float64)
    H = b.T @ a
    U, _, Vt = torch.linalg.svd(H)
    d = float(torch.sign(torch.det(Vt.T @ U.T)))
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=H.dtype, device=H.device))
    return Vt.T @ D @ U.T


def _discover_mtz_amplitude_labels(mtz_path: Path) -> tuple[str, str]:
    """
    Read the MTZ column descriptions and return the first (F, SIGF) pair.

    Column types we look for: 'F' (structure-factor amplitude) and 'Q'
    (standard deviation) per CCP4 MTZ column-type convention.
    """
    import gemmi
    mtz = gemmi.read_mtz_file(str(mtz_path))
    cols = mtz.columns
    f_label, sigf_label = None, None
    for c in cols:
        if c.type == "F" and f_label is None:
            f_label = c.label
        elif c.type == "Q" and sigf_label is None:
            sigf_label = c.label
        if f_label and sigf_label:
            break
    if f_label is None or sigf_label is None:
        raise RuntimeError(
            f"could not find F/SIGF columns in {mtz_path}; "
            f"saw {[(c.label, c.type) for c in cols]}"
        )
    return f_label, sigf_label


_SOLU_RE = re.compile(
    r"SOLU\s+SET\s+.*?RFZ=([-\d.eE+]+).*?TFZ=([-\d.eE+]+)"
    r".*?LLG=([-\d.eE+]+).*?Rfac=([-\d.eE+%]+)?",
    re.IGNORECASE | re.DOTALL,
)


def _parse_phaser_sol(sol_path: Path) -> dict:
    """
    Extract LLG / TFZ / RFZ from Phaser's `.sol` solution file.

    Phaser writes one or more `SOLU SET` lines summarising each solution;
    the top one is the highest-LLG. Format (Phaser 2.x):
        SOLU SET  RFZ=10.2 TFZ=8.9 PAK=1 LLG=260 TFZ==11.9 LLG=6176 TFZ==60.2 ...

    The double-equals (`TFZ==`, `LLG=...`) entries are post-refinement
    scores; the first single-equals are pre-refinement. We report the
    final refined LLG and TFZ (the last LLG / TFZ== occurrences).

    Phaser does NOT print an R-factor in the .sol — we compute one
    downstream via torchref's Scaler on the placed PDB.
    """
    if not sol_path.exists():
        return {"solved": False, "llg": None, "tfz": None, "rfz": None,
                "n_solutions": 0}
    text = sol_path.read_text()
    solu_lines = [ln for ln in text.splitlines() if ln.strip().startswith("SOLU SET")]
    n = len(solu_lines)
    if n == 0:
        return {"solved": False, "llg": None, "tfz": None, "rfz": None,
                "n_solutions": 0}

    top = solu_lines[0]
    # Final refined LLG / TFZ are the LAST occurrences on the line.
    # (Phaser writes the pre-refinement then the refined version after.)
    llg_matches = re.findall(r"LLG=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", top)
    tfz_matches = re.findall(
        r"TFZ=={0,1}([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", top,
    )
    rfz_matches = re.findall(r"RFZ=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", top)
    llg = float(llg_matches[-1]) if llg_matches else None
    tfz = float(tfz_matches[-1]) if tfz_matches else None
    rfz = float(rfz_matches[-1]) if rfz_matches else None

    return {
        "solved": True,
        "llg": llg, "tfz": tfz, "rfz": rfz,
        "n_solutions": n,
    }


def _write_phaser_keywords(
    work: Path, *, mtz_path: Path, rotated_pdb: Path,
    f_label: str, sigf_label: str, root: str,
) -> Path:
    """Build the Phaser keyword script. MODE MR_AUTO runs through to RNP."""
    kw = work / "phaser.kw"
    kw.write_text(f"""TITLE torchref Phaser MR comparison
MODE MR_AUTO
HKLIN {mtz_path}
LABIN F={f_label} SIGF={sigf_label}
ENSEMBLE search PDB {rotated_pdb} IDENT 1.0
COMPOSITION BY AVERAGE
SEARCH ENSEMBLE search NUM 1
ROOT {root}
JOBS 1
""")
    return kw


def _run_phaser(work: Path, kw_path: Path, timeout_s: int,
                verbose: int) -> tuple[int, float]:
    """Invoke `phenix.phaser` with the keyword script piped to stdin.

    `phenix.phaser` with no command-line argument launches the CCP4-style
    binary that reads keywords from stdin. Passing the keywords as a file
    argument triggers the PHIL-parameter path, which doesn't understand
    CCP4 syntax.
    """
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
        # Always save stdout/stderr — Phaser's solution summary lands in
        # stdout, and we want to keep a record of failed runs too.
        (work / "phaser.stdout").write_text(proc.stdout or "")
        (work / "phaser.stderr").write_text(proc.stderr or "")
    except subprocess.TimeoutExpired:
        rc = -1
    return rc, time.time() - t0


def run(pdb_key: str, seed: int, *, work_root: Path,
        verbose: int = 1, timeout_s: int = 3600) -> dict:
    pdb_path, mtz_path = PAIRS[pdb_key]
    print(f"\n=== PHASER {pdb_key} seed={seed} ===", flush=True)

    t0 = time.time()
    device = torch.device("cpu")
    # Load model on CPU — Phaser doesn't care, and we only need the rotated
    # PDB on disk. The Scaler comparison at the end also runs on CPU.
    model = ModelFT(device=device).load_pdb(str(pdb_path))
    data = ReflectionData(device=str(device)).load_mtz(str(mtz_path))
    sym_mats = data.spacegroup.matrices.to(dtype=torch.float64, device=device)
    xyz_canonical = model.xyz().detach().clone().to(device)

    # Apply R_true (same random rotation `run_random_pdb_fit.py` applies).
    R_true = _random_rotation(seed).to(device)
    centroid = xyz_canonical.mean(0)
    rotated = model.rotate(
        R_true.to(model.dtype_float), center=centroid,
    )

    # Per-trial workspace
    work = work_root / f"{pdb_key}_seed{seed}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    rotated_pdb = work / "rotated.pdb"
    rotated.write_pdb(str(rotated_pdb))

    f_label, sigf_label = _discover_mtz_amplitude_labels(mtz_path)
    if verbose:
        print(f"  MTZ labels: F={f_label}, SIGF={sigf_label}", flush=True)

    root = "phaser_out"
    kw_path = _write_phaser_keywords(
        work, mtz_path=mtz_path, rotated_pdb=rotated_pdb,
        f_label=f_label, sigf_label=sigf_label, root=root,
    )
    rc, phaser_time_s = _run_phaser(work, kw_path, timeout_s, verbose)
    sol_path = work / f"{root}.sol"
    placed_pdb = work / f"{root}.1.pdb"

    phaser = _parse_phaser_sol(sol_path)
    if verbose:
        print(
            f"  Phaser: rc={rc}, solved={phaser['solved']}, "
            f"LLG={phaser['llg']}, TFZ={phaser['tfz']}, "
            f"time={phaser_time_s:.1f}s",
            flush=True,
        )

    # Angular distance + R-work on Phaser's placed model.
    err_canonical_deg = float("nan")
    rwork_torchref_scaler = float("nan")
    rfree_torchref_scaler = float("nan")
    if placed_pdb.exists():
        try:
            placed = ModelFT(device=device).load_pdb(str(placed_pdb))
            # If atom counts disagree (Phaser may include only one ensemble
            # member) we Kabsch on the common prefix. Phaser's placement
            # preserves atom order.
            xyz_placed = placed.xyz().detach().to(device)
            n_match = min(xyz_placed.shape[0], xyz_canonical.shape[0])
            R_residual = _kabsch_rotation(
                xyz_placed[:n_match], xyz_canonical[:n_match],
            )
            err_canonical_deg = min(
                rotation_angular_distance_deg(
                    R_residual.to(torch.float64), sym_mats[k],
                )
                for k in range(sym_mats.shape[0])
            )
        except Exception as exc:
            print(f"  Kabsch on placed.pdb failed: {exc!r}", flush=True)

        # torchref-Scaler R-work on Phaser's placement (same Scaler the
        # torchref pipeline uses for its end-of-pipeline R-work — apples
        # to apples for the post-MR placement quality).
        try:
            placed = ModelFT(device=device).load_pdb(str(placed_pdb))
            scaler = Scaler(model=placed, data=data, nbins=20, verbose=0,
                            device=device)
            with torch.no_grad():
                fcalc = placed(data.hkl).detach()
            scaler.initialize(fcalc)
            scaler.refine_lbfgs(fcalc=fcalc)
            with torch.no_grad():
                rw, rf = rfactor_work_free(data, torch.abs(scaler.forward(fcalc)))
            rwork_torchref_scaler = (
                rw.item() if hasattr(rw, "item") else float(rw)
            )
            rfree_torchref_scaler = (
                rf.item() if hasattr(rf, "item") else float(rf)
            )
        except Exception as exc:
            print(f"  torchref-Scaler on placed.pdb failed: {exc!r}",
                  flush=True)

    if verbose:
        print(
            f"  Phaser placed-vs-canonical angular distance "
            f"(mod {data.spacegroup}-symmetry): {err_canonical_deg:.2f}°  "
            f"R-work (torchref Scaler) = {rwork_torchref_scaler:.4f}",
            flush=True,
        )

    return {
        "pdb": pdb_key,
        "seed": seed,
        "spacegroup": str(data.spacegroup),
        "phaser_exit": rc,
        "phaser_solved": phaser["solved"],
        "phaser_n_solutions": phaser["n_solutions"],
        "phaser_llg": phaser["llg"],
        "phaser_tfz": phaser["tfz"],
        "phaser_rfz": phaser.get("rfz"),
        "err_canonical_deg": err_canonical_deg,
        "rwork_torchref_scaler": rwork_torchref_scaler,
        "rfree_torchref_scaler": rfree_torchref_scaler,
        "phaser_time_s": phaser_time_s,
        "total_time_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default=None, choices=sorted(PAIRS.keys()))
    ap.add_argument("--seed", type=int, default=None,
                    help="Same semantics as run_random_pdb_fit.py: top-level "
                         "rng seed that derives the per-trial seeds.")
    ap.add_argument("--n-trials", type=int, default=1)
    ap.add_argument("--sweep", action="store_true",
                    help="Iterate over all PDBs in PAIRS, n-trials per PDB.")
    ap.add_argument("--timeout-s", type=int, default=3600,
                    help="Per-trial Phaser timeout in seconds (default 1h).")
    ap.add_argument("--workdir", default=None,
                    help="Per-trial workspace root (default: /tmp/phaser_mr_<ts>).")
    ap.add_argument("--out-csv", default=None,
                    help="CSV output path. Default: timestamped under cwd.")
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

    work_root = Path(args.workdir or f"/tmp/phaser_mr_{int(time.time())}")
    work_root.mkdir(parents=True, exist_ok=True)
    print(f"work root: {work_root}", flush=True)

    out_csv = Path(args.out_csv or f"phaser_mr_results_{int(time.time())}.csv")
    rows = []
    for pdb_key, trial_seed in worklist:
        try:
            r = run(pdb_key, trial_seed, work_root=work_root,
                    verbose=args.verbose, timeout_s=args.timeout_s)
            rows.append(r)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  TRIAL FAILED on {pdb_key} seed={trial_seed}: {exc!r}",
                  flush=True)
            rows.append({
                "pdb": pdb_key, "seed": trial_seed,
                "phaser_solved": False, "phaser_exit": -2,
                "error": repr(exc),
            })

    if rows:
        # Union of all columns across all rows.
        cols = sorted({k for r in rows for k in r})
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {out_csv}", flush=True)

    # Quick console summary
    print("\n=== Phaser MR summary ===", flush=True)
    print(
        f"{'pdb':>6}  {'seed':>10}  {'solved':>6}  {'LLG':>8}  "
        f"{'TFZ':>6}  {'r_torchref':>10}  {'err°':>7}  {'time_s':>8}",
        flush=True,
    )
    for r in rows:
        if r.get("error"):
            print(f"{r['pdb']:>6}  {r['seed']:>10}  FAILED: {r['error']}",
                  flush=True)
            continue
        def _f(x, default=float('nan')):
            return x if x is not None else default
        print(
            f"{r['pdb']:>6}  {r['seed']:>10}  "
            f"{str(r.get('phaser_solved', False)):>6}  "
            f"{_f(r.get('phaser_llg')):>8.2f}  "
            f"{_f(r.get('phaser_tfz')):>6.2f}  "
            f"{_f(r.get('rwork_torchref_scaler')):>10.4f}  "
            f"{_f(r.get('err_canonical_deg')):>7.2f}  "
            f"{r.get('phaser_time_s', 0.0):>8.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
