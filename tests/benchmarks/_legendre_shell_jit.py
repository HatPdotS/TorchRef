"""The C++ ``legendre_shell`` JIT build, kept only as the benchmark baseline.

Deleted together with ``_cpp_build`` once the Rust kernels have passed
``bench_cpu_kernels.py`` on every target machine.
"""

from _cpp_build import build_extension

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


def load():
    """Compile (once, cached on disk) and return the C++ module, or raise."""
    module, err = build_extension("frf_legendre_shell", _CPP_SRC)
    if module is None:
        raise RuntimeError(err[0] + "\n" + err[1])
    return module
