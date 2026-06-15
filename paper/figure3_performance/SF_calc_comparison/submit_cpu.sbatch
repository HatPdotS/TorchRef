#!/bin/bash
#SBATCH --job-name=sf_cmp_cpu
#SBATCH --partition=day
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --output=%x_%j.out

set -euo pipefail

REPO_ROOT="/das/work/p17/p17490/Peter/Library/work_trees_torchref/comparison_SFcalc"
BENCH_DIR="${REPO_ROOT}/paper/figure3_performance/SF_calc_comparison"
PYTHON="${REPO_ROOT}/.dev/bin/python"

cd "${BENCH_DIR}"

# 16-thread CPU benchmark. Note: cctbx structure_factors() is single-threaded
# regardless of these settings (documented in README).
export TORCHREF_NUM_THREADS=16
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

echo "=== SF comparison — CPU (16 threads) ==="
echo "Host: $(hostname)"
echo "Output dir: ${OUTPUT_DIR:-auto}"
echo

"${PYTHON}" run_benchmarks.py \
    --device cpu \
    --n-iterations 10 \
    --n-warmup 3 \
    ${OUTPUT_DIR:+--output-dir "${OUTPUT_DIR}"} \
    --no-plot
