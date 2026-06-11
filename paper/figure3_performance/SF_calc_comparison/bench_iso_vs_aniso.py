#!/usr/bin/env python
"""Benchmark: isotropic vs anisotropic structure-factor calculation.

Compares the cost of the ISO vs ANISO path for both TorchRef SF pipelines
(``sf_fft`` = electron-density + FFT, ``sf_ds`` = direct summation) on the SAME
atoms and the SAME reflections, so the time delta is purely the per-atom math
overhead of anisotropic ADPs (3x3 covariance + inverse) over isotropic ones.

Every atom is run twice:
  * ISO   — isotropic B-factor ``b`` (the structure's B_eq).
  * ANISO — anisotropic ``U = (b / 8*pi^2) * I`` (iso-equivalent diagonal U).
Because the ANISO U is the iso-equivalent, both paths produce the SAME structure
factors (checked), isolating the kernel cost rather than a different workload.

Reuses the timing helpers from ``benchmark_worker.py``.

Usage:
    TORCHREF_NUM_THREADS=8 .dev/bin/python bench_iso_vs_aniso.py \
        --device cuda --structures 1DAW 2DQ6 --n-iterations 20 --n-warmup 5
"""

import argparse
import math
import os

import torch

from benchmark_worker import (
    _resolve_files,
    _summarize_times,
    _time_iterations,
    _time_iterations_gpu,
)

EIGHT_PI_SQ = 8.0 * math.pi**2


def _all_atoms(M):
    """Concatenate iso + aniso atoms into one set with both ADP representations.

    Returns xyz, occ, A, B (all atoms) plus matched ``b`` (isotropic) and ``u``
    (anisotropic, 6-comp) where the iso-origin atoms get ``U = b/(8*pi^2)*I`` and
    the aniso-origin atoms get a B_eq = 8*pi^2*tr(U)/3.
    """
    xi, adpi, occi, Ai, Bi = M.get_iso()
    xa, ua, occa, Aa, Ba = M.get_aniso()

    def cat(a, b):
        if a is None or len(a) == 0:
            return b
        if b is None or len(b) == 0:
            return a
        return torch.cat([a, b], dim=0)

    # iso-origin atoms -> diagonal U; aniso-origin atoms -> B_eq
    u_from_iso = torch.zeros(len(xi), 6, device=xi.device, dtype=xi.dtype)
    u_from_iso[:, 0] = u_from_iso[:, 1] = u_from_iso[:, 2] = adpi / EIGHT_PI_SQ
    if xa is not None and len(xa) > 0:
        b_from_aniso = EIGHT_PI_SQ * (ua[:, 0] + ua[:, 1] + ua[:, 2]) / 3.0
    else:
        b_from_aniso = None

    xyz = cat(xi, xa)
    occ = cat(occi, occa)
    A = cat(Ai, Aa)
    B = cat(Bi, Ba)
    b = cat(adpi, b_from_aniso)
    u = cat(u_from_iso, ua)
    return xyz, occ, A, B, b, u


def _empty(device, dtype):
    return (
        torch.zeros(0, 3, device=device, dtype=dtype),
        torch.zeros(0, device=device, dtype=dtype),
        torch.zeros(0, device=device, dtype=dtype),
        torch.zeros(0, 5, device=device, dtype=dtype),
        torch.zeros(0, 5, device=device, dtype=dtype),
    )


def _build(engine, hkl, xyz, occ, A, B, b, u, mode):
    """Return a (fwd, fwd_bwd) pair for ``mode`` in {'iso','aniso'}."""
    device, dtype = xyz.device, xyz.dtype
    ex, eadp, eocc, eA, eB = _empty(device, dtype)

    if mode == "iso":
        xyz_p = xyz.clone().detach().requires_grad_(True)
        adp_p = b.clone().detach().requires_grad_(True)
        occ_d, A_d, B_d = occ.detach(), A.detach(), B.detach()
        grads = (xyz_p, adp_p)

        def call():
            return engine.compute_structure_factors(
                hkl, xyz_p, adp_p, occ_d, A_d, B_d
            )[0]
    else:  # aniso
        xyz_p = xyz.clone().detach().requires_grad_(True)
        u_p = u.clone().detach().requires_grad_(True)
        occ_d, A_d, B_d = occ.detach(), A.detach(), B.detach()
        grads = (xyz_p, u_p)

        def call():
            return engine.compute_structure_factors(
                hkl, ex, eadp, eocc, eA, eB,
                xyz_p, u_p, occ_d, A_d, B_d,
            )[0]

    def fwd():
        return call()

    def fwd_bwd():
        for t in grads:
            t.grad = None
        sf = call()
        loss = sf.abs().sum()
        loss.backward()
        return loss

    return fwd, fwd_bwd


def bench_structure(method, structure, device_str, n_iter, n_warmup):
    from torchref import ModelFT, ReflectionData
    from torchref.model import SfDS

    device = torch.device(device_str)
    is_gpu = device.type == "cuda"
    timer = _time_iterations_gpu if is_gpu else _time_iterations

    pdb, mtz = _resolve_files(structure)
    data = ReflectionData(device=device).load_mtz(mtz)
    d_min = float(data.d_min)
    M = ModelFT(max_res=d_min, device=device, radius_angstrom=3.0).load_pdb(pdb)
    hkl = data()[0]
    xyz, occ, A, B, b, u = _all_atoms(M)
    n_atoms, n_refl = int(xyz.shape[0]), int(hkl.shape[0])

    engine = M.fft if method == "sf_fft" else SfDS(M.cell, M.spacegroup, device=device)

    out = {"method": method, "structure": structure, "n_atoms": n_atoms,
           "n_refl": n_refl, "d_min": d_min}

    # correctness: iso vs aniso F should agree (iso-equivalent U)
    with torch.no_grad():
        f_iso = _build(engine, hkl, xyz, occ, A, B, b, u, "iso")[0]()
        f_ani = _build(engine, hkl, xyz, occ, A, B, b, u, "aniso")[0]()
        out["F_rel_err"] = ((f_iso - f_ani).abs().max() / f_iso.abs().max()).item()

    for mode in ("iso", "aniso"):
        fwd, fwd_bwd = _build(engine, hkl, xyz, occ, A, B, b, u, mode)
        with torch.no_grad():
            for _ in range(n_warmup):
                fwd()
            if is_gpu:
                torch.cuda.synchronize()
            out[f"{mode}_fwd"] = _summarize_times(timer(fwd, n_iter))["mean_time"]
        for _ in range(n_warmup):
            fwd_bwd()
        if is_gpu:
            torch.cuda.synchronize()
        out[f"{mode}_fwd_bwd"] = _summarize_times(timer(fwd_bwd, n_iter))["mean_time"]

    return out


def main():
    p = argparse.ArgumentParser(description="iso vs aniso SF benchmark")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--methods", nargs="+", default=["sf_fft", "sf_ds"])
    p.add_argument("--structures", nargs="+", default=["1DAW"])
    p.add_argument("--n-iterations", type=int, default=20)
    p.add_argument("--n-warmup", type=int, default=5)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    hdr = (f"{'method':7s} {'struct':6s} {'atoms':>6s} {'refl':>6s} "
           f"{'iso_fwd':>9s} {'ani_fwd':>9s} {'x':>5s}  "
           f"{'iso_f+b':>9s} {'ani_f+b':>9s} {'x':>5s}  {'F_relerr':>9s}")
    print(f"\n=== iso vs aniso SF  [{args.device}] "
          f"(times in ms, x = aniso/iso) ===")
    print(hdr)
    print("-" * len(hdr))
    for method in args.methods:
        for s in args.structures:
            try:
                r = bench_structure(method, s, args.device,
                                    args.n_iterations, args.n_warmup)
            except Exception as e:
                print(f"{method:7s} {s:6s}  ERROR: {type(e).__name__}: {e}")
                continue
            print(
                f"{r['method']:7s} {r['structure']:6s} {r['n_atoms']:6d} "
                f"{r['n_refl']:6d} "
                f"{r['iso_fwd']*1e3:9.3f} {r['aniso_fwd']*1e3:9.3f} "
                f"{r['aniso_fwd']/r['iso_fwd']:5.2f}  "
                f"{r['iso_fwd_bwd']*1e3:9.3f} {r['aniso_fwd_bwd']*1e3:9.3f} "
                f"{r['aniso_fwd_bwd']/r['iso_fwd_bwd']:5.2f}  "
                f"{r['F_rel_err']:9.1e}"
            )


if __name__ == "__main__":
    main()
