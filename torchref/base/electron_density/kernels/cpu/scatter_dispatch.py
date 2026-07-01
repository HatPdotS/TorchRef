"""Device-aware structured scatter for the box-splat density kernels.

On CPU, dispatches to the custom C++ parallel scatter (``cpu_scatter``) when the
extension built successfully; otherwise (and on every non-CPU device) falls back
to PyTorch's ``scatter_add_``.
"""

import torch

# Lazy-loaded C++ parallel scatter for CPU
_cpp_scatter_fn = None
_cpp_scatter_accumulate_fn = None
_cpp_scatter_checked = False


def _get_cpp_scatter():
    """Return the C++ parallel scatter_add, or None if unavailable.

    Eagerly triggers the C++ compilation so that failures (missing ninja,
    unsupported compiler flags, etc.) are caught here rather than mid-calculation.
    Also caches the in-place variant used by the no-grad fast path.
    """
    global _cpp_scatter_fn, _cpp_scatter_accumulate_fn, _cpp_scatter_checked
    if not _cpp_scatter_checked:
        try:
            from torchref.base.electron_density.kernels.cpu.scatter import (
                _get_module,
                structured_scatter_accumulate,
                structured_scatter_add,
            )

            # Trigger compilation now — _get_module returns None on failure
            if _get_module() is not None:
                _cpp_scatter_fn = structured_scatter_add
                _cpp_scatter_accumulate_fn = structured_scatter_accumulate
        except Exception:
            pass
        _cpp_scatter_checked = True
    return _cpp_scatter_fn


def _do_structured_scatter(
    density_cube: torch.Tensor,
    wa: torch.Tensor,
    wbwc: torch.Tensor,
    density_flat: torch.Tensor,
    map_size: int,
) -> torch.Tensor:
    """Pick the fastest structured scatter for the device.

    On CPU, dispatches to the custom C++ kernel (``cpu_scatter``) when
    available — partitioned, no atomics, ~2× faster than PyTorch's stock
    ``scatter_add_``. On every other device (MPS, CUDA, CPU without the
    extension built) falls back to PyTorch ``scatter_add_`` with int64
    indices.

    Returns the resulting flat density tensor. The C++ path accumulates
    out-of-place; the ``scatter_add_`` fallback mutates ``density_flat``
    in place. Both return the up-to-date tensor.
    """
    if density_cube.device.type == "cpu":
        cpp_fn = _get_cpp_scatter()
        if cpp_fn is not None:
            # Differentiable in-place accumulate: density_flat += scatter(cube),
            # mutating one shared buffer across chunks. Avoids the per-chunk
            # zeros(map_size) + out-of-place full-grid add (two full-grid touches
            # per chunk) that the functional path incurs and that dominate
            # scatter-bound, many-chunk structures. Gradients are preserved
            # (the scatter is linear; see _ScatterAccumulate).
            if _cpp_scatter_accumulate_fn is not None:
                return _cpp_scatter_accumulate_fn(density_flat, density_cube, wa, wbwc)
            return density_flat + cpp_fn(density_cube, wa, wbwc, map_size)
    # Fallback: PyTorch scatter_add_ requires int64 indices.
    idx_flat = wa[:, :, None, None] + wbwc[:, None, :, :]
    density_flat.scatter_add_(
        0,
        idx_flat.reshape(-1).to(torch.int64),
        density_cube.reshape(-1),
    )
    return density_flat
