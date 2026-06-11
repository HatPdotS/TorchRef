#!/bin/bash
#SBATCH --job-name=iso_aniso_cpu
#SBATCH --partition=day
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=%x_%j.out

set -euo pipefail

REPO_ROOT="/das/work/p17/p17490/Peter/Library/work_trees_torchref/comparison_SFcalc"
BENCH_DIR="${REPO_ROOT}/paper/figure3_performance/SF_calc_comparison"
PYTHON="${REPO_ROOT}/.dev/bin/python"

cd "${BENCH_DIR}"

export TORCHREF_NUM_THREADS=16
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

echo "=== iso vs aniso SF benchmark — CPU (16 threads) ==="
echo "Host: $(hostname)"
echo

# FFT (electron-density) path: iso separable vs new aniso box kernel — both structures.
"${PYTHON}" bench_iso_vs_aniso.py \
    --device cpu \
    --methods sf_fft \
    --structures 1DAW 4BX9 \
    --n-iterations 10 \
    --n-warmup 3

# Direct summation is much heavier on CPU; benchmark the small structure only.
"${PYTHON}" bench_iso_vs_aniso.py \
    --device cpu \
    --methods sf_ds \
    --structures 1DAW \
    --n-iterations 5 \
    --n-warmup 2
