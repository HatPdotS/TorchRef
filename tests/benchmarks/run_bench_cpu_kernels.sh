#!/bin/bash
# CPU-kernel gate on a cluster node: Rust torchref-kernels vs the C++ -march=native build.
#
#   sbatch -p <cpu-partition> tests/benchmarks/run_bench_cpu_kernels.sh
#
# Run from the repo root, in an environment with torchref installed (editable is fine),
# torchref-kernels built from this checkout (`pip install ./kernels`), and a C++20
# compiler plus `pip install ninja` for the baseline build. No GPU needed. Writes
# bench_cpu_kernels.<jobid>.{txt,json}; the last line reads GATE PASSED or GATE FAILED.

#SBATCH -J bench-cpu-kernels
#SBATCH -c 8
#SBATCH -o bench_cpu_kernels.%j.txt
#SBATCH -e bench_cpu_kernels.%j.err

set -euo pipefail
export TORCHREF_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# One build directory per job: the baseline is compiled with -march=native for this node.
export TORCH_EXTENSIONS_DIR="${TMPDIR:-/tmp}/torchref-bench-${SLURM_JOB_ID:-local}"

python -c "import torchref_kernels as k; print(k.build_info())"
grep -m1 "model name" /proc/cpuinfo || true
python -m pytest -q -p no:cacheprovider \
    tests/unit/base/test_canonical_sphere_cpu.py \
    tests/unit/frf_separate/test_legendre_kernel.py
python tests/benchmarks/bench_cpu_kernels.py \
    --threads "${TORCHREF_NUM_THREADS}" --reps 7 \
    --json "bench_cpu_kernels.${SLURM_JOB_ID:-local}.json"
