#!/bin/bash
#SBATCH --job-name=iso_aniso_gpu
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --output=%x_%j.out

set -euo pipefail

REPO_ROOT="/das/work/p17/p17490/Peter/Library/work_trees_torchref/comparison_SFcalc"
BENCH_DIR="${REPO_ROOT}/paper/figure3_performance/SF_calc_comparison"
PYTHON="${REPO_ROOT}/.dev/bin/python"

cd "${BENCH_DIR}"

export TORCHREF_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "=== iso vs aniso SF benchmark — GPU ==="
echo "Host: $(hostname)"
nvidia-smi | head -15 || true
echo

"${PYTHON}" bench_iso_vs_aniso.py \
    --device cuda \
    --methods sf_fft sf_ds \
    --structures 1DAW 4BX9 \
    --n-iterations 20 \
    --n-warmup 5
