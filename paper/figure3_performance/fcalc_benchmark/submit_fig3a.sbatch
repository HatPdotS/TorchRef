#!/bin/bash
#SBATCH --job-name=fig3a_fcalc
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=%x_%j.out

set -euo pipefail

REPO_ROOT="/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev"
BENCH_DIR="${REPO_ROOT}/paper/figure3_performance/fcalc_benchmark"
DATA_DIR="${REPO_ROOT}/paper/figure3_performance/data/fcalc"
PYTHON="${REPO_ROOT}/.dev/bin/python"

cd "${BENCH_DIR}"

echo "=== Fig3a fcalc thread scaling ==="
echo "Host: $(hostname)"
nvidia-smi | head -20 || true
echo "Output dir: ${DATA_DIR}"
echo

# All 10 test structures (auto-discovered from tests/files/{pdb,mtz}/).
"${PYTHON}" benchmark_thread_scaling.py --output_dir "${DATA_DIR}" --timeout 1200
