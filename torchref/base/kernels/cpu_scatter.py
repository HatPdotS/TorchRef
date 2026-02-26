"""Parallel partitioned scatter_add for structured (wa + wbwc) indices.

Uses a C++ OpenMP kernel compiled via torch.utils.cpp_extension.load_inline.
Each thread owns a contiguous output partition and only accumulates scatter
elements that fall in its range.  With atoms sorted by 1D center, per-thread
early-exit skips ~(T-1)/T of atoms — giving near-linear scaling.

Autograd: forward is the custom C++ scatter, backward is a standard gather
(embarrassingly parallel, no custom kernel needed).
"""

import torch
from torch.utils.cpp_extension import load_inline

_CPP_SRC = r"""
#include <torch/extension.h>
#include <cstdint>

// Parallel partitioned scatter_add from structured indices.
//
// Output space is divided among OpenMP threads.  Each thread only
// writes to its own [lo, hi) segment — zero synchronization.
// Atoms sorted by 1D center give efficient early-exit per thread.
//
// Args:
//   output      (M,)            float32, accumulated into
//   wa          (C, nx)         int64,   x-axis 1D indices
//   wbwc        (C, ny * nz)   int64,   yz-plane 2D indices (flattened)
//   values      (C, nx*ny*nz)  float32, density values (flattened cube)
//   nx, ny, nz  axis sizes
torch::Tensor structured_scatter_add(
    torch::Tensor output,
    torch::Tensor wa,
    torch::Tensor wbwc,
    torch::Tensor values,
    int64_t nx, int64_t ny, int64_t nz)
{
    TORCH_CHECK(output.is_contiguous() && output.scalar_type() == torch::kFloat32);
    TORCH_CHECK(wa.is_contiguous()     && wa.scalar_type() == torch::kInt64);
    TORCH_CHECK(wbwc.is_contiguous()   && wbwc.scalar_type() == torch::kInt64);
    TORCH_CHECK(values.is_contiguous() && values.scalar_type() == torch::kFloat32);

    const int64_t M      = output.size(0);
    const int64_t C      = wa.size(0);
    const int64_t ny_nz  = ny * nz;
    const int64_t nxyz   = nx * ny_nz;

    float*         __restrict__ out_p  = output.data_ptr<float>();
    const int64_t* __restrict__ wa_p   = wa.data_ptr<int64_t>();
    const int64_t* __restrict__ wbwc_p = wbwc.data_ptr<int64_t>();
    const float*   __restrict__ val_p  = values.data_ptr<float>();

    // Global wbwc bounds (for atom-level early exit)
    int64_t wbwc_min = wbwc_p[0], wbwc_max = wbwc_p[0];
    for (int64_t i = 1; i < C * ny_nz; i++) {
        int64_t v = wbwc_p[i];
        if (v < wbwc_min) wbwc_min = v;
        if (v > wbwc_max) wbwc_max = v;
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int64_t lo = (int64_t)tid * M / nth;
        int64_t hi = (int64_t)(tid + 1) * M / nth;

        for (int64_t c = 0; c < C; c++) {
            const int64_t* wa_row = wa_p + c * nx;

            // Atom-level early exit: check wa range vs partition
            int64_t amin = wa_row[0], amax = wa_row[0];
            for (int64_t i = 1; i < nx; i++) {
                int64_t v = wa_row[i];
                if (v < amin) amin = v;
                if (v > amax) amax = v;
            }
            if (amax + wbwc_max < lo || amin + wbwc_min >= hi) continue;

            const int64_t* wbwc_row = wbwc_p + c * ny_nz;
            const float*   val_base = val_p  + c * nxyz;

            for (int64_t ix = 0; ix < nx; ix++) {
                int64_t wa_val = wa_row[ix];
                // x-offset early exit
                if (wa_val + wbwc_max < lo || wa_val + wbwc_min >= hi) continue;

                const float* val_ix = val_base + ix * ny_nz;

                for (int64_t iyz = 0; iyz < ny_nz; iyz++) {
                    int64_t idx = wa_val + wbwc_row[iyz];
                    if (idx >= lo && idx < hi) {
                        out_p[idx] += val_ix[iyz];
                    }
                }
            }
        }
    }

    return output;
}

// Parallel structured gather (backward of scatter_add).
//
// grad_cube[c, ix, iyz] = grad_output[wa[c,ix] + wbwc[c,iyz]]
//
// Each atom's output is independent — parallelize over atoms directly.
// No index tensor allocation, no fancy indexing overhead.
torch::Tensor structured_gather(
    torch::Tensor grad_output,
    torch::Tensor wa,
    torch::Tensor wbwc,
    int64_t nx, int64_t ny, int64_t nz)
{
    TORCH_CHECK(grad_output.is_contiguous() && grad_output.scalar_type() == torch::kFloat32);
    TORCH_CHECK(wa.is_contiguous()   && wa.scalar_type() == torch::kInt64);
    TORCH_CHECK(wbwc.is_contiguous() && wbwc.scalar_type() == torch::kInt64);

    const int64_t C      = wa.size(0);
    const int64_t ny_nz  = ny * nz;
    const int64_t nxyz   = nx * ny_nz;

    auto grad_cube = torch::empty({C, nxyz}, grad_output.options());

    const float*   __restrict__ go_p   = grad_output.data_ptr<float>();
    const int64_t* __restrict__ wa_p   = wa.data_ptr<int64_t>();
    const int64_t* __restrict__ wbwc_p = wbwc.data_ptr<int64_t>();
    float*         __restrict__ gc_p   = grad_cube.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < C; c++) {
        const int64_t* wa_row   = wa_p   + c * nx;
        const int64_t* wbwc_row = wbwc_p + c * ny_nz;
        float*         out_row  = gc_p   + c * nxyz;

        for (int64_t ix = 0; ix < nx; ix++) {
            int64_t wa_val = wa_row[ix];
            float*  dst    = out_row + ix * ny_nz;

            for (int64_t iyz = 0; iyz < ny_nz; iyz++) {
                dst[iyz] = go_p[wa_val + wbwc_row[iyz]];
            }
        }
    }

    return grad_cube;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("structured_scatter_add", &structured_scatter_add,
          "Parallel partitioned scatter_add from structured (wa, wbwc) indices");
    m.def("structured_gather", &structured_gather,
          "Parallel structured gather (backward of scatter_add)");
}
"""

# ---------------------------------------------------------------------------
# Lazy compilation with POSIX lockf locking (works across cluster nodes).
#
# PyTorch's load_inline uses FileBaton (file-existence lock) which stays
# behind when a process is killed mid-compile, blocking all future imports.
# We wrap it with fcntl.lockf (POSIX record locks) which:
#   - are enforced by the filesystem → work across NFS/GPFS cluster nodes
#   - are released by the kernel on process death (even SIGKILL)
# First process compiles; all others (same node or different) reuse cache.
# ---------------------------------------------------------------------------
_module = None


def _get_module():
    global _module
    if _module is None:
        import fcntl
        import os
        import sys

        # Ensure ninja (installed via pip) is on PATH for compute nodes
        bin_dir = os.path.dirname(sys.executable)
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + ":" + os.environ.get("PATH", "")
        # Need GCC >= 9 for PyTorch C++ extensions
        for toolset in ("14", "13", "12"):
            gcc = f"/opt/rh/gcc-toolset-{toolset}/root/usr/bin/g++"
            if os.path.isfile(gcc):
                os.environ["CXX"] = gcc
                os.environ["CC"] = gcc.replace("g++", "gcc")
                break

        # Per-microarchitecture build directory — prevents Illegal Instruction
        # when different cluster nodes have different CPUs (e.g., AMD vs Intel).
        import platform
        cpu_tag = platform.machine()
        try:
            # Use the CPU model to distinguish microarchitectures
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        # e.g. "EPYC_7443P" or "Xeon_Gold_6248"
                        cpu_tag = line.split(":")[1].strip().replace(" ", "_")
                        break
        except OSError:
            pass
        build_dir = os.path.join(
            os.environ.get(
                "TORCH_EXTENSIONS_DIR",
                os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions"),
            ),
            f"cpu_scatter_{cpu_tag}",
        )
        os.makedirs(build_dir, exist_ok=True)

        # fcntl.lockf uses POSIX record locks (fcntl F_SETLKW) which are:
        #   1. filesystem-level → work across NFS/GPFS cluster nodes
        #   2. released by kernel on process death, even SIGKILL
        lock_fd = os.open(
            os.path.join(build_dir, "compile.lock"), os.O_CREAT | os.O_RDWR
        )
        try:
            fcntl.lockf(lock_fd, fcntl.LOCK_EX)
            # Clear any stale PyTorch FileBaton lock from a killed process
            try:
                os.unlink(os.path.join(build_dir, "lock"))
            except FileNotFoundError:
                pass
            _module = load_inline(
                name="cpu_scatter",
                cpp_sources=[_CPP_SRC],
                extra_cflags=["-O3", "-fopenmp", "-march=native"],
                extra_ldflags=["-fopenmp"],
                build_directory=build_dir,
                verbose=False,
            )
        finally:
            fcntl.lockf(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    return _module


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------

class _StructuredScatterAdd(torch.autograd.Function):
    """scatter_add with structured (wa, wbwc) indices.

    Forward:  C++ partitioned scatter (parallel, no conflicts)
    Backward: standard gather (embarrassingly parallel)
    """

    @staticmethod
    def forward(ctx, density_cube, wa, wbwc, map_size):
        # density_cube: (C, nx, ny, nz)  — requires grad
        # wa:           (C, nx)           — int64, no grad
        # wbwc:         (C, ny, nz)       — int64, no grad
        C, nx, ny, nz = density_cube.shape
        ctx.save_for_backward(wa, wbwc)
        ctx.cube_shape = density_cube.shape

        mod = _get_module()
        result = torch.zeros(map_size, dtype=density_cube.dtype,
                             device=density_cube.device)
        mod.structured_scatter_add(
            result,
            wa.contiguous(),
            wbwc.reshape(C, ny * nz).contiguous(),
            density_cube.reshape(C, nx * ny * nz).contiguous(),
            nx, ny, nz,
        )
        return result

    @staticmethod
    def backward(ctx, grad_output):
        wa, wbwc = ctx.saved_tensors
        C, nx, ny, nz = ctx.cube_shape
        mod = _get_module()
        grad_cube = mod.structured_gather(
            grad_output.contiguous(),
            wa.contiguous(),
            wbwc.reshape(C, ny * nz).contiguous(),
            nx, ny, nz,
        )
        return grad_cube.reshape(C, nx, ny, nz), None, None, None


def structured_scatter_add(density_cube, wa, wbwc, map_size):
    """Differentiable parallel scatter_add using structured indices.

    Parameters
    ----------
    density_cube : Tensor (C, nx, ny, nz) float32
        Values to scatter (from _separable_density).
    wa : Tensor (C, nx) int64
        Precomputed x-axis scatter indices.
    wbwc : Tensor (C, ny, nz) int64
        Precomputed yz-plane scatter indices.
    map_size : int
        Total number of voxels in flat density map.

    Returns
    -------
    Tensor (map_size,) float32
        Scattered result.  Differentiable w.r.t. density_cube.
    """
    return _StructuredScatterAdd.apply(density_cube, wa, wbwc, map_size)
