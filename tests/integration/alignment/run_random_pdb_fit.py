#!/usr/bin/env python
"""
End-to-end demo / sanity check for the alignment pipeline.

Flow:
  1. Pick a random PDB / MTZ pair from `tests/files`.
  2. Load model + data (real cell, real F_obs).
  3. Apply a random rotation to the model atoms.
  4. Run `ModelFT.fit_to_data` to recover an aligned orientation.
  5. Fit an anisotropic Scaler against the data.
  6. Report R-work / R-free before and after.

Run as a script (NOT a pytest test) — it's a one-off integration probe:

    cd /das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/fix_alignment
    python tests/integration/alignment/run_random_pdb_fit.py [--seed N] [--pdb 1DAW]
"""


from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch

from torchref.experimental.alignment.frf.rotation_utils import rotation_angular_distance_deg
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT
from torchref.scaling import Scaler


TEST_FILES =Path('/das/work/p17/p17490/Peter/Library/work_trees_torchref/fix_alignment/tests/files')

PAIRS = {
    # PDB stem → (pdb_path, mtz_path). Some PDBs use a non-standard filename.
    "1AK5": (TEST_FILES / "pdb" / "1AK5_with_H.pdb", TEST_FILES / "mtz" / "1AK5.mtz"),
    "1DAW": (TEST_FILES / "pdb" / "1DAW.pdb", TEST_FILES / "mtz" / "1DAW.mtz"),
    "2DQ6": (TEST_FILES / "pdb" / "2DQ6.pdb", TEST_FILES / "mtz" / "2DQ6.mtz"),
    "3A5V": (TEST_FILES / "pdb" / "3A5V.pdb", TEST_FILES / "mtz" / "3A5V.mtz"),
    "3E98": (TEST_FILES / "pdb" / "3E98.pdb", TEST_FILES / "mtz" / "3E98.mtz"),
    "3GR5": (TEST_FILES / "pdb" / "3GR5.pdb", TEST_FILES / "mtz" / "3GR5.mtz"),
    "3K7M": (TEST_FILES / "pdb" / "3K7M.pdb", TEST_FILES / "mtz" / "3K7M.mtz"),
    "3VRJ": (TEST_FILES / "pdb" / "3VRJ.pdb", TEST_FILES / "mtz" / "3VRJ.mtz"),
    "4BX9": (TEST_FILES / "pdb" / "4BX9.pdb", TEST_FILES / "mtz" / "4BX9.mtz"),
    # 5BOV excluded — too large (4.6 GiB single TF allocation OOMs on A100-40GB).
    "6G9X": (TEST_FILES / "pdb" / "6G9X.pdb", TEST_FILES / "mtz" / "6G9X.mtz"),
}


def _random_rotation(seed: int) -> torch.Tensor:
    """Uniform random rotation on SO(3) via QR of a Gaussian matrix."""
    g = torch.Generator().manual_seed(int(seed))
    A = torch.randn(3, 3, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _min_err_over_sym(R_test: torch.Tensor, R_ref: torch.Tensor,
                       sym_mats: torch.Tensor) -> float:
    """Minimum angular distance of R_test to any S·R_ref over sym_mats."""
    best = float("inf")
    R_test = R_test.to(torch.float64)
    R_ref = R_ref.to(torch.float64)
    for k in range(sym_mats.shape[0]):
        e = rotation_angular_distance_deg(R_test, sym_mats[k] @ R_ref)
        if e < best:
            best = e
    return best


def _kabsch_rotation(xyz_a: torch.Tensor, xyz_b: torch.Tensor) -> torch.Tensor:
    """Return R minimising ||xyz_a - xyz_b @ R^T|| (both centred)."""
    a = (xyz_a.detach() - xyz_a.detach().mean(0)).to(torch.float64)
    b = (xyz_b.detach() - xyz_b.detach().mean(0)).to(torch.float64)
    H = b.T @ a
    U, _, Vt = torch.linalg.svd(H)
    d = float(torch.sign(torch.det(Vt.T @ U.T)))
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=H.dtype, device=H.device))
    return Vt.T @ D @ U.T


def run(pdb_key: str, seed: int, verbose: int = 1,
         device: torch.device = torch.device("cpu"),
         use_interp_var: bool = False,
         use_llg_tf: bool = False,
         refine_b: bool = False,
         sigma_rot_deg: float = 0.0,
         sigma_trans_ang: float = 0.0,
         sigma_b: float = 0.0,
         use_sigma_a_frf: bool = False,
         frf_delta_vrms_A: float = 1.0,
         frf_weight_combine: str = "sigma_a_only",
         n_rotation_candidates: int = 15,
         use_m_symmetry_filter: bool = False,
         use_lerf1_intensity: bool = False,
         use_fitted_delta_vrms: bool = False,
         use_even_l_only: bool = False) -> dict:
    pdb_path, mtz_path = PAIRS[pdb_key]
    print(f"\n=== {pdb_key}: {pdb_path.name} + {mtz_path.name} ===", flush=True)

    # 1. Construct model + data directly on `device` so the alignment
    # pipeline runs end-to-end on GPU without re-creating the SfFFT.
    # Post-hoc `.to(device)` reassigns Model.cell via the setter, which
    # triggers `_maybe_initialize_fft` and drops the already-built grid /
    # map_symmetry state.
    t0 = time.time()
    model = ModelFT(device=device).load_pdb(str(pdb_path))
    data = ReflectionData(device=str(device)).load_mtz(str(mtz_path))
    sym_mats = data.spacegroup.matrices.to(dtype=torch.float64, device=device)
    print(f"  spacegroup: {data.spacegroup}  cell: "
          f"a={data.cell.a:.1f}, b={data.cell.b:.1f}, c={data.cell.c:.1f} Å  "
          f"atoms: {model.xyz().shape[0]}  ({time.time()-t0:.1f}s)", flush=True)

    def _scale_and_r(m: ModelFT) -> tuple[float, float]:
        s = Scaler(model=m, data=data, nbins=20, verbose=0,
                   device=m.xyz().device)
        # Detach fcalc — the scaler only needs grad through its own
        # parameters. Without this, m's autograd graph (and the
        # CachedForwardMixin hook that pins it via a register_hook closure)
        # keeps multi-GB of fft intermediates alive across trials.
        with torch.no_grad():
            fcalc = m(data.hkl).detach()
        s.initialize(fcalc)
        s.refine_lbfgs(fcalc=fcalc)
        with torch.no_grad():
            rw, rf = s.rfactor(fcalc)
        rw = rw.item() if hasattr(rw, "item") else float(rw)
        rf = rf.item() if hasattr(rf, "item") else float(rf)
        return rw, rf

    # Reference R-factor of the un-rotated model (the optimal we could hope for).
    rwork_ref, rfree_ref = _scale_and_r(model)
    print(f"  reference R-work (un-rotated model): {rwork_ref:.4f}  "
          f"R-free: {rfree_ref:.4f}", flush=True)

    # 2. Apply random rotation to atom coords.
    R_true = _random_rotation(seed)
    xyz_canonical = model.xyz().clone()
    centroid = xyz_canonical.mean(0)
    rotated_search = model.rotate(
        R_true.to(model.dtype_float).to(device), center=centroid,
    )

    # R-factor of the rotated search model (should be ~50% — random).
    rwork_pre, rfree_pre = _scale_and_r(rotated_search)
    print(f"  rotated search R-work (no alignment): {rwork_pre:.4f}  "
          f"R-free: {rfree_pre:.4f}  (should be ~0.5)", flush=True)

    # 3. Run fit_to_data: recover the alignment.
    t1 = time.time()
    aligned = rotated_search.fit_to_data(
        data,
        d_min=4.0, d_max=15.0,
        L=32, n_shells=20,
        n_rotation_peaks=200, n_ml_refine=200,
        verbose=verbose,
        use_interp_var=use_interp_var,
        use_llg_tf=use_llg_tf,
        refine_b=refine_b,
        sigma_rot_deg=sigma_rot_deg,
        sigma_trans_ang=sigma_trans_ang,
        sigma_b=sigma_b,
        use_sigma_a_frf=use_sigma_a_frf,
        frf_delta_vrms_A=frf_delta_vrms_A,
        frf_weight_combine=frf_weight_combine,
        n_rotation_candidates=n_rotation_candidates,
        use_m_symmetry_filter=use_m_symmetry_filter,
        use_lerf1_intensity=use_lerf1_intensity,
        use_fitted_delta_vrms=use_fitted_delta_vrms,
        use_even_l_only=use_even_l_only,
    )
    fit_time = time.time() - t1
    print(f"  fit_to_data took {fit_time:.1f}s", flush=True)

    # Effective rotation between aligned coords and the canonical reference.
    R_residual = _kabsch_rotation(aligned.xyz(), xyz_canonical)
    err_to_canonical = min(
        rotation_angular_distance_deg(R_residual.to(torch.float64), sym_mats[k])
        for k in range(sym_mats.shape[0])
    )
    print(f"  aligned-vs-canonical angular distance "
          f"(mod {data.spacegroup}-symmetry): {err_to_canonical:.2f}°", flush=True)

    # 4. Scale the aligned model and report R-factor.
    rwork_post, rfree_post = _scale_and_r(aligned)
    print(f"  aligned R-work: {rwork_post:.4f}  R-free: {rfree_post:.4f}", flush=True)

    return {
        "pdb": pdb_key,
        "spacegroup": str(data.spacegroup),
        "ref_rwork": rwork_ref,
        "ref_rfree": rfree_ref,
        "pre_rwork": rwork_pre,
        "pre_rfree": rfree_pre,
        "post_rwork": rwork_post,
        "post_rfree": rfree_post,
        "err_canonical_deg": err_to_canonical,
        "fit_time_s": fit_time,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed (rotation + PDB pick). Default: time-based.")
    ap.add_argument("--pdb", default=None, choices=sorted(PAIRS.keys()),
                    help="PDB key to use. Default: random.")
    ap.add_argument("--n-trials", type=int, default=1,
                    help="Number of random trials with different rotations / PDBs.")
    ap.add_argument("--sweep", action="store_true",
                    help="Iterate over every PDB in PAIRS, n-trials per PDB.")
    ap.add_argument("--verbose", type=int, default=0,
                    help="Verbosity passed to fit_to_data.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Run the alignment on this device.")
    ap.add_argument("--use-interp-var", action="store_true",
                    help="Phase A flag: add Phaser-style totvar_search "
                         "interpolation variance to the rotation rescore.")
    ap.add_argument("--use-llg-tf", action="store_true",
                    help="Phase B flag: re-rank the FFT-correlation "
                         "translation peaks by Rice/Woolfson LLG.")
    ap.add_argument("--refine-b", action="store_true",
                    help="Phase C: co-refine per-atom B-factors in rigid-body.")
    ap.add_argument("--sigma-rot-deg", type=float, default=0.0,
                    help="Phase C: Gaussian rotation restraint sigma (deg). "
                         "0 = no restraint. Phaser default ~5.")
    ap.add_argument("--sigma-trans-ang", type=float, default=0.0,
                    help="Phase C: Gaussian translation restraint sigma (Å). "
                         "0 = no restraint. Phaser default ~0.5.")
    ap.add_argument("--sigma-b", type=float, default=0.0,
                    help="Phase C: Gaussian B-factor restraint sigma (Å²). "
                         "0 = no restraint. Phaser default ~15.")
    ap.add_argument("--use-sigma-a-frf", action="store_true",
                    help="E3: σA-weight the FRF input field (Phaser FastRot "
                         "Eterm/Vterm analogue). Default off.")
    ap.add_argument("--frf-delta-vrms", type=float, default=1.0,
                    help="ΔVRMS for Luzzati σA(s) = exp(−2π²s²ΔVRMS²), Å. "
                         "Default 1.0.")
    ap.add_argument("--frf-weight-combine", default="sigma_a_only",
                    choices=["sigma_a_only", "sigma_a_x_variance"],
                    help="How to combine σA² and empirical variance weights.")
    ap.add_argument("--n-rotation-candidates", type=int, default=15,
                    help="Top-N rotations from MLRF rescore that get full "
                         "translation+polish. Default 15.")
    ap.add_argument("--use-m-symmetry-filter", action="store_true",
                    help="F1: zero SH coefficients with m not divisible by "
                         "ZSYMM (Phaser-style symmetry-aware denoiser).")
    ap.add_argument("--use-lerf1-intensity", action="store_true",
                    help="F2: replace patt_obs = E²−1 with "
                         "cweight·(E²−1)·DFAC² (Phaser LERF1 form).")
    ap.add_argument("--use-fitted-delta-vrms", action="store_true",
                    help="F3: fit ΔVRMS from <B>/(8π²) instead of "
                         "frf_delta_vrms_A.")
    ap.add_argument("--use-even-l-only", action="store_true",
                    help="F4: skip odd-l SH coefficients (perf, no SNR).")
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda not available")

    if args.seed is None:
        args.seed = int(time.time())
    rng = random.Random(args.seed)
    print(f"seed = {args.seed}", flush=True)

    # Build the (pdb, seed) work list.
    if args.sweep:
        worklist = [(pdb, rng.randint(0, 10 ** 9))
                    for pdb in sorted(PAIRS.keys())
                    for _ in range(args.n_trials)]
    else:
        worklist = []
        for _ in range(args.n_trials):
            pdb = args.pdb if args.pdb is not None else rng.choice(list(PAIRS.keys()))
            worklist.append((pdb, rng.randint(0, 10 ** 9)))

    results = []
    for pdb_key, trial_seed in worklist:
        try:
            r = run(pdb_key, trial_seed, verbose=args.verbose, device=device,
                    use_interp_var=args.use_interp_var,
                    use_llg_tf=args.use_llg_tf,
                    refine_b=args.refine_b,
                    sigma_rot_deg=args.sigma_rot_deg,
                    sigma_trans_ang=args.sigma_trans_ang,
                    sigma_b=args.sigma_b,
                    use_sigma_a_frf=args.use_sigma_a_frf,
                    frf_delta_vrms_A=args.frf_delta_vrms,
                    frf_weight_combine=args.frf_weight_combine,
                    n_rotation_candidates=args.n_rotation_candidates,
                    use_m_symmetry_filter=args.use_m_symmetry_filter,
                    use_lerf1_intensity=args.use_lerf1_intensity,
                    use_fitted_delta_vrms=args.use_fitted_delta_vrms,
                    use_even_l_only=args.use_even_l_only)
            results.append(r)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  TRIAL FAILED on {pdb_key}: {exc!r}", flush=True)
            results.append({"pdb": pdb_key, "error": repr(exc)})
        finally:
            # Release CUDA allocator caches between trials. Without this a
            # failed trial leaves its ~tens-of-GB residue in the allocator
            # pool and starves every subsequent trial of memory.
            if device.type == "cuda":
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                alloc = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"  [post-trial gc] alloc={alloc:5.2f} GB  "
                      f"reserved={reserved:5.2f} GB", flush=True)

    print("\n=== summary ===", flush=True)
    print(f"{'pdb':>6}  {'sg':>8}  {'rwork_ref':>10}  {'rwork_pre':>10}  "
          f"{'rwork_post':>10}  {'err_deg':>8}  {'time_s':>7}", flush=True)
    for r in results:
        if "error" in r:
            print(f"{r['pdb']:>6}  FAILED: {r['error']}", flush=True)
            continue
        print(f"{r['pdb']:>6}  {r['spacegroup'][12:20]:>8}  "
              f"{r['ref_rwork']:>10.4f}  {r['pre_rwork']:>10.4f}  "
              f"{r['post_rwork']:>10.4f}  {r['err_canonical_deg']:>8.2f}  "
              f"{r['fit_time_s']:>7.1f}", flush=True)


if __name__ == "__main__":
    main()
