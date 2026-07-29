"""Parallel partitioned scatter_add for structured (wa + wbwc) indices.

This kernel is float32-only by design: the C++ kernel hardcodes
``torch::kFloat32`` for the density/output buffers. The Python wrapper
auto-casts non-float32 inputs to float32 and emits a one-time warning per
input dtype — so callers running with ``dtypes.float = torch.float64`` keep
working, just at float32 precision through this kernel.

Uses a C++ kernel compiled via torch.utils.cpp_extension.load_inline.
Each thread owns a contiguous output partition and only accumulates scatter
elements that fall in its range.  With atoms sorted by 1D center, per-thread
early-exit skips ~(T-1)/T of atoms — giving near-linear scaling.

Threading backend is selected at compile time:
  * OpenMP on Linux (and any compiler that accepts -fopenmp)
  * std::thread fallback otherwise (notably Apple Clang on macOS, whose
    -fopenmp is rejected because the OpenMP runtime is not bundled)

Two index dtypes are exposed:
  * int32  — default fast path. Fits any realistic crystallographic grid
             (nx*ny*nz < 2**31, roughly 1300^3 voxels) and halves the
             index bandwidth of the inner loop.
  * int64  — fallback for the rare edge cases that exceed INT32_MAX
             (e.g. very large cryo-EM grids).
The Python wrapper picks the binding from wa.dtype.

Autograd: forward is the custom C++ scatter, backward is a standard gather
(embarrassingly parallel, no custom kernel needed).
"""

import warnings

import torch

from torchref.base.electron_density.kernels.cpu._cpp_build import build_extension

_WARNED_CAST_DTYPES: set = set()

_CPP_SRC = r"""
#include <torch/extension.h>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#else
#include <thread>
#include <vector>
#endif

// Parallel partitioned scatter_add from structured indices.
//
// Output space is divided among threads.  Each thread only writes to its own
// [lo, hi) segment — zero synchronization.
// Atoms sorted by 1D center give efficient early-exit per thread.
//
// IdxT is int32_t (default fast path) or int64_t (for huge grids).
template <typename IdxT>
torch::Tensor structured_scatter_add_impl(
    torch::Tensor output,
    torch::Tensor wa,
    torch::Tensor wbwc,
    torch::Tensor values,
    int64_t nx, int64_t ny, int64_t nz)
{
    TORCH_CHECK(output.is_contiguous() && output.scalar_type() == torch::kFloat32);
    TORCH_CHECK(values.is_contiguous() && values.scalar_type() == torch::kFloat32);
    TORCH_CHECK(wa.is_contiguous());
    TORCH_CHECK(wbwc.is_contiguous());

    const int64_t M      = output.size(0);
    const int64_t C      = wa.size(0);
    const int64_t ny_nz  = ny * nz;
    const int64_t nxyz   = nx * ny_nz;

    float*       __restrict__ out_p  = output.data_ptr<float>();
    const IdxT*  __restrict__ wa_p   = wa.data_ptr<IdxT>();
    const IdxT*  __restrict__ wbwc_p = wbwc.data_ptr<IdxT>();
    const float* __restrict__ val_p  = values.data_ptr<float>();

    // Global wbwc bounds (for atom-level early exit)
    IdxT wbwc_min = wbwc_p[0], wbwc_max = wbwc_p[0];
    for (int64_t i = 1; i < C * ny_nz; i++) {
        IdxT v = wbwc_p[i];
        if (v < wbwc_min) wbwc_min = v;
        if (v > wbwc_max) wbwc_max = v;
    }

    auto worker = [&](int64_t lo, int64_t hi) {
        for (int64_t c = 0; c < C; c++) {
            const IdxT* wa_row = wa_p + c * nx;

            // Atom-level early exit: check wa range vs partition
            IdxT amin = wa_row[0], amax = wa_row[0];
            for (int64_t i = 1; i < nx; i++) {
                IdxT v = wa_row[i];
                if (v < amin) amin = v;
                if (v > amax) amax = v;
            }
            if ((int64_t)amax + wbwc_max < lo || (int64_t)amin + wbwc_min >= hi) continue;

            const IdxT*  wbwc_row = wbwc_p + c * ny_nz;
            const float* val_base = val_p  + c * nxyz;

            for (int64_t ix = 0; ix < nx; ix++) {
                IdxT wa_val = wa_row[ix];
                // x-offset early exit
                if ((int64_t)wa_val + wbwc_max < lo || (int64_t)wa_val + wbwc_min >= hi) continue;

                const float* val_ix = val_base + ix * ny_nz;

                for (int64_t iyz = 0; iyz < ny_nz; iyz++) {
                    int64_t idx = (int64_t)wa_val + (int64_t)wbwc_row[iyz];
                    if (idx >= lo && idx < hi) {
                        out_p[idx] += val_ix[iyz];
                    }
                }
            }
        }
    };

#ifdef _OPENMP
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int64_t lo = (int64_t)tid * M / nth;
        int64_t hi = (int64_t)(tid + 1) * M / nth;
        worker(lo, hi);
    }
#else
    int nth = (int)std::thread::hardware_concurrency();
    if (nth < 1) nth = 1;
    std::vector<std::thread> threads;
    threads.reserve(nth);
    for (int tid = 0; tid < nth; tid++) {
        int64_t lo = (int64_t)tid * M / nth;
        int64_t hi = (int64_t)(tid + 1) * M / nth;
        threads.emplace_back(worker, lo, hi);
    }
    for (auto& t : threads) t.join();
#endif

    return output;
}

// Parallel structured gather (backward of scatter_add).
//
// grad_cube[c, ix, iyz] = grad_output[wa[c,ix] + wbwc[c,iyz]]
//
// Each atom's output is independent — parallelize over atoms directly.
// No index tensor allocation, no fancy indexing overhead.
// Note: the worker_gather lambda below is used only by the non-OpenMP branch;
// under _OPENMP the same loop body is duplicated inline in the parallel-for.
template <typename IdxT>
torch::Tensor structured_gather_impl(
    torch::Tensor grad_output,
    torch::Tensor wa,
    torch::Tensor wbwc,
    int64_t nx, int64_t ny, int64_t nz)
{
    TORCH_CHECK(grad_output.is_contiguous() && grad_output.scalar_type() == torch::kFloat32);
    TORCH_CHECK(wa.is_contiguous());
    TORCH_CHECK(wbwc.is_contiguous());

    const int64_t C      = wa.size(0);
    const int64_t ny_nz  = ny * nz;
    const int64_t nxyz   = nx * ny_nz;

    auto grad_cube = torch::empty({C, nxyz}, grad_output.options());

    const float* __restrict__ go_p   = grad_output.data_ptr<float>();
    const IdxT*  __restrict__ wa_p   = wa.data_ptr<IdxT>();
    const IdxT*  __restrict__ wbwc_p = wbwc.data_ptr<IdxT>();
    float*       __restrict__ gc_p   = grad_cube.data_ptr<float>();

    auto worker_gather = [&](int64_t c_lo, int64_t c_hi) {
        for (int64_t c = c_lo; c < c_hi; c++) {
            const IdxT*  wa_row   = wa_p   + c * nx;
            const IdxT*  wbwc_row = wbwc_p + c * ny_nz;
            float*       out_row  = gc_p   + c * nxyz;

            for (int64_t ix = 0; ix < nx; ix++) {
                IdxT wa_val = wa_row[ix];
                float* dst  = out_row + ix * ny_nz;

                for (int64_t iyz = 0; iyz < ny_nz; iyz++) {
                    dst[iyz] = go_p[(int64_t)wa_val + (int64_t)wbwc_row[iyz]];
                }
            }
        }
    };

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < C; c++) {
        const IdxT*  wa_row   = wa_p   + c * nx;
        const IdxT*  wbwc_row = wbwc_p + c * ny_nz;
        float*       out_row  = gc_p   + c * nxyz;

        for (int64_t ix = 0; ix < nx; ix++) {
            IdxT wa_val = wa_row[ix];
            float* dst  = out_row + ix * ny_nz;

            for (int64_t iyz = 0; iyz < ny_nz; iyz++) {
                dst[iyz] = go_p[(int64_t)wa_val + (int64_t)wbwc_row[iyz]];
            }
        }
    }
#else
    int nth = (int)std::thread::hardware_concurrency();
    if (nth < 1) nth = 1;
    std::vector<std::thread> threads;
    threads.reserve(nth);
    for (int tid = 0; tid < nth; tid++) {
        int64_t c_lo = (int64_t)tid * C / nth;
        int64_t c_hi = (int64_t)(tid + 1) * C / nth;
        threads.emplace_back(worker_gather, c_lo, c_hi);
    }
    for (auto& t : threads) t.join();
#endif

    return grad_cube;
}

// Wrappers that enforce the matching dtype, so a wrong-dtype tensor gets a
// clear error instead of being reinterpreted by data_ptr<IdxT>().
torch::Tensor structured_scatter_add_i32(
    torch::Tensor output, torch::Tensor wa, torch::Tensor wbwc, torch::Tensor values,
    int64_t nx, int64_t ny, int64_t nz)
{
    TORCH_CHECK(wa.scalar_type()   == torch::kInt32, "wa must be int32");
    TORCH_CHECK(wbwc.scalar_type() == torch::kInt32, "wbwc must be int32");
    return structured_scatter_add_impl<int32_t>(output, wa, wbwc, values, nx, ny, nz);
}
torch::Tensor structured_scatter_add_i64(
    torch::Tensor output, torch::Tensor wa, torch::Tensor wbwc, torch::Tensor values,
    int64_t nx, int64_t ny, int64_t nz)
{
    TORCH_CHECK(wa.scalar_type()   == torch::kInt64, "wa must be int64");
    TORCH_CHECK(wbwc.scalar_type() == torch::kInt64, "wbwc must be int64");
    return structured_scatter_add_impl<int64_t>(output, wa, wbwc, values, nx, ny, nz);
}
torch::Tensor structured_gather_i32(
    torch::Tensor grad_output, torch::Tensor wa, torch::Tensor wbwc,
    int64_t nx, int64_t ny, int64_t nz)
{
    TORCH_CHECK(wa.scalar_type()   == torch::kInt32, "wa must be int32");
    TORCH_CHECK(wbwc.scalar_type() == torch::kInt32, "wbwc must be int32");
    return structured_gather_impl<int32_t>(grad_output, wa, wbwc, nx, ny, nz);
}
torch::Tensor structured_gather_i64(
    torch::Tensor grad_output, torch::Tensor wa, torch::Tensor wbwc,
    int64_t nx, int64_t ny, int64_t nz)
{
    TORCH_CHECK(wa.scalar_type()   == torch::kInt64, "wa must be int64");
    TORCH_CHECK(wbwc.scalar_type() == torch::kInt64, "wbwc must be int64");
    return structured_gather_impl<int64_t>(grad_output, wa, wbwc, nx, ny, nz);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("structured_scatter_add_i32", &structured_scatter_add_i32,
          "Partitioned scatter_add with int32 structured (wa, wbwc) indices");
    m.def("structured_scatter_add_i64", &structured_scatter_add_i64,
          "Partitioned scatter_add with int64 structured (wa, wbwc) indices");
    m.def("structured_gather_i32", &structured_gather_i32,
          "Structured gather (backward) with int32 indices");
    m.def("structured_gather_i64", &structured_gather_i64,
          "Structured gather (backward) with int64 indices");
}
"""

# ---------------------------------------------------------------------------
# Lazy compilation. The build itself (POSIX lockf locking, per-microarchitecture
# build directory, ninja/GCC discovery, no -fopenmp on macOS) lives in
# ``_cpp_build.build_extension``, shared with ``sphere_splat.py`` so the two
# extensions cannot drift apart on cluster deployment details.
# ---------------------------------------------------------------------------
_module = None
_module_failed = False
# Captured diagnostic from the most recent failed compile attempt. Populated
# by _get_module() on failure so test/debug code can surface the underlying
# error instead of just "module not available". Format: (error_str, traceback_str).
_module_error: "tuple[str, str] | None" = None


def _get_module():
    global _module, _module_failed, _module_error
    if _module is not None:
        return _module
    if _module_failed:
        return None
    _module, _module_error = build_extension("cpu_scatter", _CPP_SRC)
    if _module is None:
        _module_failed = True
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
        # density_cube: (C, nx, ny, nz)  — requires grad, must be float32
        # wa:           (C, nx)           — int32 or int64, no grad
        # wbwc:         (C, ny, nz)       — same dtype as wa
        if density_cube.dtype != torch.float32:
            if density_cube.dtype not in _WARNED_CAST_DTYPES:
                warnings.warn(
                    f"cpu_scatter is float32-only; casting density_cube from "
                    f"{density_cube.dtype} to float32 (precision will be reduced).",
                    stacklevel=3,
                )
                _WARNED_CAST_DTYPES.add(density_cube.dtype)
            density_cube = density_cube.to(torch.float32)
        if wa.dtype != wbwc.dtype:
            raise TypeError(
                f"wa.dtype ({wa.dtype}) and wbwc.dtype ({wbwc.dtype}) must match"
            )
        if wa.dtype == torch.int32:
            if map_size > _INT32_MAX:
                raise RuntimeError(
                    f"map_size {map_size} exceeds INT32_MAX ({_INT32_MAX}); "
                    "pass int64 indices for grids this large."
                )
            scatter_fn_name = "structured_scatter_add_i32"
            gather_fn_name = "structured_gather_i32"
        elif wa.dtype == torch.int64:
            scatter_fn_name = "structured_scatter_add_i64"
            gather_fn_name = "structured_gather_i64"
        else:
            raise TypeError(f"wa.dtype must be int32 or int64, got {wa.dtype}")

        C, nx, ny, nz = density_cube.shape
        ctx.save_for_backward(wa, wbwc)
        ctx.cube_shape = density_cube.shape
        ctx.gather_fn_name = gather_fn_name

        mod = _get_module()
        if mod is None:
            err = _module_error[0] if _module_error else "unknown reason"
            raise RuntimeError(
                f"C++ cpu_scatter module not available ({err}). "
                "See torchref.base.electron_density.kernels.cpu.scatter._module_error for the full traceback."
            )
        result = torch.zeros(
            map_size, dtype=density_cube.dtype, device=density_cube.device
        )
        getattr(mod, scatter_fn_name)(
            result,
            wa.contiguous(),
            wbwc.reshape(C, ny * nz).contiguous(),
            density_cube.reshape(C, nx * ny * nz).contiguous(),
            nx,
            ny,
            nz,
        )
        return result

    @staticmethod
    def backward(ctx, grad_output):
        wa, wbwc = ctx.saved_tensors
        C, nx, ny, nz = ctx.cube_shape
        # The scatter is linear, so its VJP is the adjoint gather. Routing
        # through the ``_StructuredGather`` Function (rather than calling the
        # C++ gather imperatively) keeps the operation in the autograd graph,
        # so higher-order derivatives compose: gather's own backward is the
        # scatter again (the two are mutual adjoints). Without this, a C++
        # gather returns a graph-less tensor and ``create_graph=True`` silently
        # drops the second-order term through the scatter transpose.
        grad_cube = _StructuredGather.apply(grad_output, wa, wbwc, ctx.cube_shape)
        return grad_cube, None, None, None


class _StructuredGather(torch.autograd.Function):
    """Adjoint of :class:`_StructuredScatterAdd` (structured gather).

    ``gather(grid)[c] = grid[scatter_index(c)]``. Linear, so its VJP is the
    scatter. Pairing the two as mutual adjoints makes both double-(and higher-)
    differentiable with no Hessian math.
    """

    @staticmethod
    def forward(ctx, grid, wa, wbwc, cube_shape):
        if grid.dtype != torch.float32:
            if grid.dtype not in _WARNED_CAST_DTYPES:
                warnings.warn(
                    f"cpu_scatter is float32-only; casting from {grid.dtype} to "
                    f"float32 (precision will be reduced).",
                    stacklevel=3,
                )
                _WARNED_CAST_DTYPES.add(grid.dtype)
            grid = grid.to(torch.float32)
        C, nx, ny, nz = cube_shape
        mod = _get_module()
        if mod is None:
            err = _module_error[0] if _module_error else "unknown reason"
            raise RuntimeError(
                f"C++ cpu_scatter module not available ({err}). "
                "See torchref.base.electron_density.kernels.cpu.scatter._module_error for the full traceback."
            )
        # int32 / int64 gather fn name matches the scatter's index dtype.
        gather_fn_name = (
            "structured_gather_i32"
            if wa.dtype == torch.int32
            else "structured_gather_i64"
        )
        grad_cube = getattr(mod, gather_fn_name)(
            grid.contiguous(),
            wa.contiguous(),
            wbwc.reshape(C, ny * nz).contiguous(),
            nx,
            ny,
            nz,
        ).reshape(C, nx, ny, nz)
        ctx.save_for_backward(wa, wbwc)
        ctx.map_size = int(grid.numel())
        return grad_cube

    @staticmethod
    def backward(ctx, grad_cube):
        wa, wbwc = ctx.saved_tensors
        # Adjoint of the gather is the scatter — reuse the differentiable
        # Function so a third-order graph would also compose.
        grad_grid = _StructuredScatterAdd.apply(grad_cube, wa, wbwc, ctx.map_size)
        return grad_grid, None, None, None


_INT32_MAX = 2**31 - 1


def structured_scatter_add(density_cube, wa, wbwc, map_size):
    """Differentiable parallel scatter_add using structured indices.

    Dispatches to the int32 or int64 kernel based on ``wa.dtype``. int32 is
    the default fast path (halves index bandwidth); int64 is available for
    grids larger than INT32_MAX voxels.

    Parameters
    ----------
    density_cube : Tensor (C, nx, ny, nz) float32
        Values to scatter (from _separable_density).
    wa : Tensor (C, nx)  int32 or int64
        Precomputed x-axis scatter indices.
    wbwc : Tensor (C, ny, nz)  same dtype as wa
        Precomputed yz-plane scatter indices.
    map_size : int
        Total number of voxels in flat density map. For int32 indices,
        must be <= INT32_MAX.

    Returns
    -------
    Tensor (map_size,) float32
        Scattered result.  Differentiable w.r.t. density_cube.
    """
    return _StructuredScatterAdd.apply(density_cube, wa, wbwc, map_size)


def structured_scatter_add_inplace(out_flat, density_cube, wa, wbwc):
    """Low-level in-place structured scatter: ``out_flat[idx] += cube`` (no autograd).

    The raw building block for :class:`_ScatterAccumulate`. The C++ kernel already
    does ``out[idx] += ...``, so passing ``out_flat`` straight in as the output
    buffer accumulates in place — no fresh ``zeros(map_size)`` and no out-of-place
    ``out + result`` add (the two full-grid touches per chunk that the functional
    :func:`structured_scatter_add` incurs).

    Returns ``out_flat`` (mutated), or ``None`` if the C++ module is unavailable
    (caller should fall back to the functional path).
    """
    mod = _get_module()
    if mod is None:
        return None
    if density_cube.dtype != torch.float32:
        density_cube = density_cube.to(torch.float32)
    C, nx, ny, nz = density_cube.shape
    if wa.dtype == torch.int32:
        if out_flat.numel() > _INT32_MAX:
            raise RuntimeError(
                f"map_size {out_flat.numel()} exceeds INT32_MAX ({_INT32_MAX}); "
                "pass int64 indices for grids this large."
            )
        fn_name = "structured_scatter_add_i32"
    elif wa.dtype == torch.int64:
        fn_name = "structured_scatter_add_i64"
    else:
        raise TypeError(f"wa.dtype must be int32 or int64, got {wa.dtype}")
    getattr(mod, fn_name)(
        out_flat,
        wa.contiguous(),
        wbwc.reshape(C, ny * nz).contiguous(),
        density_cube.reshape(C, nx * ny * nz).contiguous(),
        nx,
        ny,
        nz,
    )
    return out_flat


class _ScatterAccumulate(torch.autograd.Function):
    """Differentiable in-place accumulating scatter: ``out += scatter(cube)``.

    ``out_flat`` is mutated in place (one shared buffer accumulated across chunks),
    so there is NO per-chunk ``zeros(map_size)`` and NO per-chunk out-of-place
    ``out + result`` add — the two full-grid touches that dominate scatter-bound,
    many-chunk structures in the functional path.

    Correctness under autograd: the scatter is linear, so for ``out = out_in +
    scatter(cube)`` the VJP is ``grad_out`` w.r.t. ``out_in`` (identity) and
    ``gather(grad_out)`` w.r.t. ``cube``. Crucially the backward needs ONLY the
    (un-mutated) indices ``wa``/``wbwc`` and ``grad_out`` — never the mutated
    buffer — so the in-place version bump invalidates no saved tensor and the
    chunk-to-chunk in-place chain differentiates correctly. The gather is routed
    through :class:`_StructuredGather` so higher-order graphs still compose.
    """

    @staticmethod
    def forward(ctx, out_flat, density_cube, wa, wbwc):
        if density_cube.dtype != torch.float32:
            if density_cube.dtype not in _WARNED_CAST_DTYPES:
                warnings.warn(
                    f"cpu_scatter is float32-only; casting density_cube from "
                    f"{density_cube.dtype} to float32 (precision will be reduced).",
                    stacklevel=3,
                )
                _WARNED_CAST_DTYPES.add(density_cube.dtype)
            density_cube = density_cube.to(torch.float32)
        ctx.save_for_backward(wa, wbwc)
        ctx.cube_shape = density_cube.shape
        structured_scatter_add_inplace(out_flat, density_cube, wa, wbwc)
        ctx.mark_dirty(out_flat)
        return out_flat

    @staticmethod
    def backward(ctx, grad_out):
        wa, wbwc = ctx.saved_tensors
        # d out / d out_in = identity  -> grad_out passes through unchanged.
        # d out / d cube  = adjoint gather of grad_out.
        grad_cube = _StructuredGather.apply(grad_out, wa, wbwc, ctx.cube_shape)
        return grad_out, grad_cube, None, None


def structured_scatter_accumulate(out_flat, density_cube, wa, wbwc):
    """Differentiable ``out_flat += scatter(density_cube)``, accumulated in place.

    Returns the (mutated) ``out_flat``. Differentiable w.r.t. both ``out_flat`` and
    ``density_cube``. Use in the chunk loop to accumulate every chunk into one
    shared buffer with no per-chunk full-grid allocation or add.
    """
    return _ScatterAccumulate.apply(out_flat, density_cube, wa, wbwc)
