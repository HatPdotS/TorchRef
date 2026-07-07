#!/bin/bash
#SBATCH --job-name=fig3b_refcycle
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --output=%x_%j.out

set -euo pipefail

REPO_ROOT="/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev"
BENCH_DIR="${REPO_ROOT}/paper/figure3_performance/refinement_cycle_benchmark"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${REPO_ROOT}/paper/figure3_performance/data/refinement_cycle/results_${TS}"
PYTHON="${REPO_ROOT}/.dev/bin/python"

cd "${BENCH_DIR}"

echo "=== Fig3b refinement-cycle thread scaling ==="
echo "Host: $(hostname)"
nvidia-smi | head -20 || true
echo "Output dir: ${OUT_DIR}"
echo

# All 10 test structures (auto-discovered from tests/files/{pdb,mtz}/). The
# per-target breakdown on large structures (5BOV/2DQ6) is the long pole, so a
# generous per-worker timeout is given.
"${PYTHON}" benchmark_thread_scaling.py --output_dir "${OUT_DIR}" --timeout 3600
