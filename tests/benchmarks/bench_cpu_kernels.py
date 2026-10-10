"""Benchmark the Rust CPU kernels against the C++ ``-march=native`` JIT builds.

Runs the density splat (iso/aniso x forward/backward x float32/float64, 1DAW-sized and
large scenes) and the FRF Legendre/shell kernel, at one thread (code generation) and at
the configured thread count (scaling), checks that both implementations agree, and
prints the Rust/C++ time ratio. Gate: ratio <= 1.05 at one thread.

The C++ Legendre kernel ignores the thread count on macOS (it is built without OpenMP and
uses every core), so its one-thread ratio is only meaningful on Linux.

    python tests/benchmarks/bench_cpu_kernels.py [--threads N] [--reps R] [--json out.json]

Needs a C++20 compiler and ninja for the baseline (it is compiled on first run).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torchref.base.electron_density.kernels.cpu import sphere_splat  # noqa: E402
from torchref.utils import native  # noqa: E402

GATE = 1.05

# (name, cell lengths A, angles deg, grid, atoms)
SCENES = [
    ("1daw-sized", (35.0, 64.0, 33.0), (90.0, 106.0, 90.0), (72, 128, 66), 1500),
    ("large", (60.0, 54.0, 48.0), (90.0, 100.0, 90.0), (120, 108, 96), 3000),
]


def _matrices(lengths, angles, dtype):
    a, b, c = lengths
    al, be, ga = (math.radians(x) for x in angles)
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(c * c - cx * cx - cy * cy)
    frac = torch.tensor(
        [[a, b * math.cos(ga), cx], [0.0, b * math.sin(ga), cy], [0.0, 0.0, cz]],
        dtype=torch.float64,
    )
    return frac.inverse().to(dtype).contiguous(), frac.to(dtype).contiguous()


def _scene(spec, dtype, aniso, seed=0):
    _, lengths, angles, grid, n = spec
    g = torch.Generator().manual_seed(seed)
    inv_frac, frac = _matrices(lengths, angles, dtype)
    xyz = (torch.rand(n, 3, generator=g, dtype=torch.float64) @ frac.double().T).to(dtype)
    occ = (0.5 + 0.5 * torch.rand(n, generator=g)).to(dtype)
    A = (0.5 + 2.0 * torch.rand(n, 5, generator=g)).to(dtype)
    B = (0.3 + 30.0 * torch.rand(n, 5, generator=g)).to(dtype)
    r2cut = (2.0 + 2.0 * torch.rand(n, generator=g)).pow(2).to(dtype)
    if aniso:
        u = torch.zeros(n, 6, dtype=torch.float64)
        u[:, :3] = 0.2 + 0.3 * torch.rand(n, 3, generator=g, dtype=torch.float64)
        u[:, 3:] = 0.02 * (torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5)
        adp = u.to(dtype)
    else:
        adp = (15.0 + 25.0 * torch.rand(n, generator=g)).to(dtype)
    grad = torch.rand(grid, generator=g).to(dtype)
    return dict(xyz=xyz, adp=adp, occ=occ, A=A, B=B, r2cut=r2cut,
                inv_frac=inv_frac.view(-1), frac=frac.view(-1), grid=grid, grad=grad)


def _run(mod, s, aniso, backward):
    nx, ny, nz = s["grid"]
    args = (s["xyz"], s["adp"], s["occ"], s["A"], s["B"], s["r2cut"],
            s["inv_frac"], s["frac"], nx, ny, nz)
    if not backward:
        out = torch.zeros(nx * ny * nz, dtype=s["xyz"].dtype)
        (mod.aniso_fwd if aniso else mod.iso_fwd)(out, *args)
        return (out,)
    gx = torch.zeros_like(s["xyz"]).view(-1)
    ga = torch.zeros_like(s["adp"]).view(-1)
    go = torch.zeros_like(s["occ"])
    (mod.aniso_bwd if aniso else mod.iso_bwd)(gx, ga, go, s["grad"].view(-1), *args)
    return gx, ga, go


def _time(fn, reps):
    fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _rel_l2(a, b):
    # Relative L2, as the kernel tests use: in float32 a voxel exactly on the cutoff sphere
    # can land in or out depending on rounding, which a max-norm reports as a large error.
    worst = 0.0
    for x, y in zip(a, b):
        scale = y.double().norm().item() or 1.0
        worst = max(worst, (x.double() - y.double()).norm().item() / scale)
    return worst


def _legendre_case(L, n_clusters, n_shells, seed=4):
    from torchref.experimental.alignment.sh import legendre_recurrence_coefficients

    g = torch.Generator().manual_seed(seed)
    cos_t = 2 * torch.rand(n_clusters, generator=g) - 1
    sin_t = (1 - cos_t * cos_t).clamp(min=0).sqrt()
    Dr = torch.randn(n_clusters, L, generator=g)
    Di = torch.randn(n_clusters, L, generator=g)
    shell = torch.sort(torch.randint(0, n_shells, (n_clusters,), generator=g))[0]
    a, b, sect = legendre_recurrence_coefficients(L, torch.float32, torch.device("cpu"))
    n_even = (L - 1 if (L - 1) % 2 == 0 else L - 2) // 2
    return (n_even, n_shells, L), (cos_t, sin_t, Dr, Di, shell, a, b, sect)


def _legendre_rust(shape, args):
    from torchref.experimental.alignment.frf.kernels.cpu import legendre_shell

    Tr, Ti = torch.zeros(shape), torch.zeros(shape)
    legendre_shell.legendre_shell_accumulate(Tr, Ti, *args)
    return Tr, Ti


def _legendre_cpp(mod, shape, args):
    from torchref.experimental.alignment.frf.kernels.cpu import legendre_shell
    from torchref.experimental.alignment.sh import LEGENDRE_SEED

    cos_t, sin_t, Dr, Di, shell, a, b, sect = args
    Tr, Ti = torch.zeros(shape), torch.zeros(shape)
    off = legendre_shell.shell_offsets(shell, shape[1])
    mod.legendre_shell_accumulate(Tr, Ti, cos_t, sin_t, Dr, Di, shell, off, a, b, sect,
                                  float(LEGENDRE_SEED))
    return Tr, Ti


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=torch.get_num_threads())
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--json", help="write the results here")
    opts = ap.parse_args()

    import _legendre_shell_jit
    import _sphere_splat_jit

    cpp = _sphere_splat_jit.load()
    cpp_leg = _legendre_shell_jit.load()
    rust = sphere_splat._require_module()
    info = native.build_info()
    print(f"host: {platform.machine()} {platform.processor() or ''} "
          f"| torch {torch.__version__} | kernels {info['version']} "
          f"cpu_variant={info['cpu_variant']}")

    rows, failed = [], False
    for threads in sorted({1, opts.threads}):
        torch.set_num_threads(threads)
        for spec in SCENES:
            for dtype in (torch.float32, torch.float64):  # dtype-ok: benchmark matrix
                for aniso in (False, True):
                    s = _scene(spec, dtype, aniso)
                    for backward in (False, True):
                        a = _run(rust, s, aniso, backward)
                        b = _run(cpp, s, aniso, backward)
                        err = _rel_l2(a, b)
                        tr = _time(lambda: _run(rust, s, aniso, backward), opts.reps)
                        tc = _time(lambda: _run(cpp, s, aniso, backward), opts.reps)
                        tol = 2e-4 if dtype == torch.float32 else 1e-12  # dtype-ok: tolerance by precision
                        ratio = tr / tc
                        bad = err > tol or (threads == 1 and ratio > GATE)
                        failed |= bad
                        row = dict(threads=threads, scene=spec[0], dtype=str(dtype)[6:],
                                   kind="aniso" if aniso else "iso",
                                   pass_="bwd" if backward else "fwd",
                                   rust_ms=tr * 1e3, cpp_ms=tc * 1e3, ratio=ratio,
                                   rel_l2_diff=err, ok=not bad)
                        rows.append(row)
                        print(f"{threads:>2}t {spec[0]:>10} {row['dtype']:>7} "
                              f"{row['kind']:>5} {row['pass_']} "
                              f"rust {tr*1e3:8.2f} ms  c++ {tc*1e3:8.2f} ms  "
                              f"ratio {ratio:5.2f}  diff {err:.1e}"
                              f"{'' if not bad else '   <-- FAIL'}")
        for L, nc, ns in ((65, 20000, 300), (101, 40000, 400)):
            shape, args = _legendre_case(L, nc, ns)
            err = _rel_l2(_legendre_rust(shape, args), _legendre_cpp(cpp_leg, shape, args))
            tr = _time(lambda: _legendre_rust(shape, args), opts.reps)
            tc = _time(lambda: _legendre_cpp(cpp_leg, shape, args), opts.reps)
            ratio = tr / tc
            gated = threads == 1 and platform.system() != "Darwin"
            bad = err > 1e-4 or (gated and ratio > GATE)
            failed |= bad
            rows.append(dict(threads=threads, scene=f"legendre-L{L}", dtype="float32",
                             kind="legendre", pass_="fwd", rust_ms=tr * 1e3,
                             cpp_ms=tc * 1e3, ratio=ratio, rel_l2_diff=err, ok=not bad))
            print(f"{threads:>2}t {'legendre-L' + str(L):>10} float32 shell fwd "
                  f"rust {tr*1e3:8.2f} ms  c++ {tc*1e3:8.2f} ms  "
                  f"ratio {ratio:5.2f}  diff {err:.1e}"
                  f"{'' if not bad else '   <-- FAIL'}"
                  f"{'' if gated or threads > 1 else '   (c++ uses all cores here)'}")
    if opts.json:
        with open(opts.json, "w") as f:
            json.dump(dict(build_info=info, rows=rows), f, indent=1)
    print("GATE", "FAILED" if failed else "PASSED",
          f"(1-thread ratio <= {GATE}, outputs agree)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
