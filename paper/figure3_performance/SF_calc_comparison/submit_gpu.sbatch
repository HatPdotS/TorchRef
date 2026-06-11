#!/bin/bash
#SBATCH --job-name=sf_cmp_gpu
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
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

echo "=== SF comparison — GPU ==="
echo "Host: $(hostname)"
nvidia-smi | head -20 || true
echo "Output dir: ${OUTPUT_DIR:-auto}"
echo

"${PYTHON}" run_benchmarks.py \
    --device cuda \
    --n-iterations 10 \
    --n-warmup 3 \
    ${OUTPUT_DIR:+--output-dir "${OUTPUT_DIR}"} \
    --no-plot
