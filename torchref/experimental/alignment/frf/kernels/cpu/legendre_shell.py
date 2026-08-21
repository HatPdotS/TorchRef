"""Fused Legendre recurrence and shell accumulation, as one C++ kernel.

The portable version runs the vertical recurrence as one torch operation per
``l`` and then scatters the row, so every row makes a round trip to memory. At
L=101 over 4.4e5 clusters that is ~108 GB for the recurrence and ~71 GB for the
scatter, both measured at ~50 GB/s -- the stages are bandwidth-bound, and the
arithmetic underneath is a small fraction of the time.

float32 throughout, matching the rest of this codebase's kernels. The radial
Bessel recurrence is a separate stage and keeps its float64 internals, where the
downward recurrence's cancellation actually needs them.

Fusing them removes the round trip: one cluster's three rows are 1.2 kB of stack,
so ``cur`` is produced, multiplied and accumulated without ever reaching memory.
Two further things fall out of writing it as a loop nest:

* **Ragged widths are free.** ``bar_P[l, m]`` is zero for m > l, so step ``l``
  needs only columns 0..l -- ``for (m = 0; m <= l; ++m)`` and nothing more. In
  torch the same saving needs narrowed views, and that was measured *slower*,
  because a strided scatter target costs more than the zeros it skips.
* **No atomics.** The clusters arrive sorted by shell, so a thread that owns a
  range of shells owns every write into those shells' rows. Parallelising over
  clusters instead would race on the shared accumulator.

The accumulator rows for one shell are ``n_even * L`` scalars -- 40 kB at L=101 --
so they stay in cache across that shell's clusters, which is the point of
grouping by shell in the first place.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from torchref.base.electron_density.kernels.cpu._cpp_build import build_extension

_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#else
#include <thread>
#endif

// One shell's worth of work: every cluster in [c0, c1) contributes
//   T[pos][s][m] += barP(l, m) * D[c][m]      for even l >= 2, m <= l
// with barP built by the vertical recurrence in registers/stack.
template <typename scalar_t>
static void shell_range(
    int64_t s_begin, int64_t s_end,
    const int64_t* __restrict off,
    const int64_t* __restrict shell,
    const scalar_t* __restrict rep_cos,
    const scalar_t* __restrict rep_sin,
    const scalar_t* __restrict Dr,
    const scalar_t* __restrict Di,
    const scalar_t* __restrict a_coef,
    const scalar_t* __restrict b_coef,
    const scalar_t* __restrict sect,
    scalar_t* __restrict Tr,
    scalar_t* __restrict Ti,
    int64_t L, int64_t nb, int64_t n_even, scalar_t seed) {

  std::vector<scalar_t> buf(3 * L, scalar_t(0));
  scalar_t* prev2 = buf.data();
  scalar_t* prev1 = buf.data() + L;
  scalar_t* cur   = buf.data() + 2 * L;

  for (int64_t s = s_begin; s < s_end; ++s) {
    for (int64_t c = off[s]; c < off[s + 1]; ++c) {
      const int64_t row = shell[c];          // == s, carried explicitly
      const scalar_t co = rep_cos[c];
      const scalar_t si = rep_sin[c];
      const scalar_t* dr = Dr + c * L;
      const scalar_t* di = Di + c * L;

      for (int64_t m = 0; m < L; ++m) { prev1[m] = scalar_t(0); prev2[m] = scalar_t(0); }
      prev1[0] = seed;                       // bar_P_0^0

      for (int64_t l = 1; l < L; ++l) {
        const scalar_t* a = a_coef + l * L;
        const scalar_t* b = b_coef + l * L;
        // Vertical recurrence, only where the row can be non-zero.
        for (int64_t m = 0; m < l; ++m) {
          cur[m] = a[m] * co * prev1[m] - b[m] * prev2[m];
        }
        // Sectoral m == l, which MUST be in place before the products below:
        // it is this row's diagonal entry.
        cur[l] = sect[l] * si * prev1[l - 1];

        if (l >= 2 && (l % 2) == 0) {
          const int64_t pos = (l - 2) / 2;
          scalar_t* tr = Tr + (pos * nb + row) * L;
          scalar_t* ti = Ti + (pos * nb + row) * L;
          for (int64_t m = 0; m <= l; ++m) {
            tr[m] += cur[m] * dr[m];
            ti[m] += cur[m] * di[m];
          }
        }
        // Rotate the three buffers; nothing is copied.
        scalar_t* t = prev2; prev2 = prev1; prev1 = cur; cur = t;
      }
    }
  }
}

void legendre_shell_accumulate(
    torch::Tensor Tr, torch::Tensor Ti,
    torch::Tensor rep_cos, torch::Tensor rep_sin,
    torch::Tensor Dr, torch::Tensor Di,
    torch::Tensor shell, torch::Tensor offsets,
    torch::Tensor a_coef, torch::Tensor b_coef, torch::Tensor sect,
    double seed) {

  TORCH_CHECK(Tr.is_contiguous() && Ti.is_contiguous(), "T must be contiguous");
  TORCH_CHECK(rep_cos.scalar_type() == Tr.scalar_type()
              && rep_sin.scalar_type() == Tr.scalar_type()
              && Dr.scalar_type() == Tr.scalar_type()
              && Di.scalar_type() == Tr.scalar_type()
              && a_coef.scalar_type() == Tr.scalar_type()
              && b_coef.scalar_type() == Tr.scalar_type()
              && sect.scalar_type() == Tr.scalar_type(),
              "every array must share the accumulator's dtype");
  TORCH_CHECK(Dr.is_contiguous() && Di.is_contiguous(), "D must be contiguous");
  TORCH_CHECK(shell.scalar_type() == torch::kLong, "shell must be int64");
  TORCH_CHECK(offsets.scalar_type() == torch::kLong, "offsets must be int64");

  const int64_t n_even = Tr.size(0);
  const int64_t nb     = Tr.size(1);
  const int64_t L      = Tr.size(2);
  TORCH_CHECK(offsets.numel() == nb + 1, "offsets must have n_shells + 1 entries");

  // float32 only, by policy: this codebase has no float64 kernels. The caller
  // is checked rather than dispatched on, so a float64 accumulator is a loud
  // error instead of a silent reinterpretation of the buffer.
  TORCH_CHECK(Tr.scalar_type() == torch::kFloat,
              "legendre_shell_accumulate is float32 only, got ", Tr.scalar_type());
  {
    using scalar_t = float;
    const int64_t* off = offsets.data_ptr<int64_t>();
    const int64_t* sh  = shell.data_ptr<int64_t>();
    const scalar_t* rc = rep_cos.data_ptr<scalar_t>();
    const scalar_t* rs = rep_sin.data_ptr<scalar_t>();
    const scalar_t* dr = Dr.data_ptr<scalar_t>();
    const scalar_t* di = Di.data_ptr<scalar_t>();
    const scalar_t* ac = a_coef.data_ptr<scalar_t>();
    const scalar_t* bc = b_coef.data_ptr<scalar_t>();
    const scalar_t* sc = sect.data_ptr<scalar_t>();
    scalar_t* tr = Tr.data_ptr<scalar_t>();
    scalar_t* ti = Ti.data_ptr<scalar_t>();
    const scalar_t sd = static_cast<scalar_t>(seed);

#ifdef _OPENMP
    // Dynamic, because clusters per shell varies (measured 2.7 to 39 across the
    // benchmark) so equal shell counts are not equal work.
#pragma omp parallel for schedule(dynamic, 8)
    for (int64_t s = 0; s < nb; ++s) {
      shell_range<scalar_t>(s, s + 1, off, sh, rc, rs, dr, di, ac, bc, sc,
                            tr, ti, L, nb, n_even, sd);
    }
#else
    // Apple Clang rejects -fopenmp, so carve the shells into contiguous blocks.
    int nthreads = std::max(1u, std::thread::hardware_concurrency());
    if (nthreads > nb) nthreads = static_cast<int>(std::max<int64_t>(nb, 1));
    std::vector<std::thread> pool;
    const int64_t per = (nb + nthreads - 1) / std::max(nthreads, 1);
    for (int t = 0; t < nthreads; ++t) {
      const int64_t s0 = t * per;
      const int64_t s1 = std::min(nb, s0 + per);
      if (s0 >= s1) break;
      pool.emplace_back([=] {
        shell_range<scalar_t>(s0, s1, off, sh, rc, rs, dr, di, ac, bc, sc,
                              tr, ti, L, nb, n_even, sd);
      });
    }
    for (auto& th : pool) th.join();
#endif
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("legendre_shell_accumulate", &legendre_shell_accumulate,
        "Fused Legendre recurrence and per-shell accumulation");
}
"""

_module = None
_module_failed = False
_module_error: Optional[Tuple[str, str]] = None


def _get_module():
    """The compiled extension, or None if it could not be built."""
    global _module, _module_failed, _module_error
    if _module is not None:
        return _module
    if _module_failed:
        return None
    _module, _module_error = build_extension("frf_legendre_shell", _CPP_SRC)
    if _module is None:
        _module_failed = True
    return _module


def why_unavailable() -> Optional[str]:
    """``None`` if the fused kernel is usable, else why it is not.

    The single availability probe for this backend, read by
    :mod:`torchref.utils.backends`. A missing compiler and a compile error are
    different problems, and the captured diagnostic is what separates them.
    """
    if _get_module() is not None:
        return None
    reason = _module_error[0] if _module_error else "unknown reason"
    return (
        f"the fused CPU Legendre/shell kernel is not available ({reason}); see "
        "torchref.experimental.alignment.frf.kernels.cpu.legendre_shell."
        "last_error()"
    )


def available() -> bool:
    """Whether the fused kernel compiled and is ready to dispatch."""
    return why_unavailable() is None


def last_error() -> Optional[Tuple[str, str]]:
    """``(message, traceback)`` from the last failed build attempt, if any."""
    _get_module()
    return _module_error


def clear_cache() -> None:
    """Forget the build result, so the next call retries. For tests."""
    global _module, _module_failed, _module_error
    _module, _module_failed, _module_error = None, False, None


def shell_offsets(shell: torch.Tensor, n_shells: int) -> torch.Tensor:
    """Start index of each shell in a shell-sorted cluster array, plus the end.

    ``(n_shells + 1,)`` int64. The kernel needs the ranges rather than the
    per-cluster labels so that a thread can own a set of shells outright and
    write their accumulator rows without atomics.
    """
    counts = torch.bincount(shell, minlength=n_shells)
    offsets = torch.zeros(n_shells + 1, dtype=torch.long, device=shell.device)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return offsets


def legendre_shell_accumulate(
    Tr: torch.Tensor,
    Ti: torch.Tensor,
    rep_cos: torch.Tensor,
    rep_sin: torch.Tensor,
    Dr: torch.Tensor,
    Di: torch.Tensor,
    shell: torch.Tensor,
    a_coef: torch.Tensor,
    b_coef: torch.Tensor,
    sect: torch.Tensor,
) -> None:
    """Fused recurrence and accumulation, in place on ``Tr``/``Ti``.

    Same signature and same effect as
    :func:`torchref.experimental.alignment.frf.kernels.portable.legendre_shell_accumulate`.
    ``shell`` must be sorted non-decreasing -- the kernel partitions work by
    shell to avoid atomics, and unsorted input would silently drop
    contributions rather than merely run slowly.
    """
    from ....sh import LEGENDRE_SEED

    module = _get_module()
    if module is None:
        raise RuntimeError(why_unavailable())
    offsets = shell_offsets(shell, Tr.shape[1])
    module.legendre_shell_accumulate(
        Tr, Ti, rep_cos.contiguous(), rep_sin.contiguous(),
        Dr.contiguous(), Di.contiguous(), shell.contiguous(), offsets,
        a_coef.contiguous(), b_coef.contiguous(), sect.contiguous(),
        float(LEGENDRE_SEED),
    )


__all__ = [
    "available",
    "clear_cache",
    "last_error",
    "legendre_shell_accumulate",
    "shell_offsets",
    "why_unavailable",
]
