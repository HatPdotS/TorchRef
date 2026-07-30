#!/bin/bash
#SBATCH --job-name=torchref-gpu-tests
#SBATCH --output=tests/scripts/logs/gpu_test_%j.out
#SBATCH --error=tests/scripts/logs/gpu_test_%j.err
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# torchref GPU Test Suite - SLURM Submission Script
# ==================================================
#
# Usage:
#   sbatch tests/scripts/submit_gpu_tests.sbatch
#
# This script runs all GPU-marked tests on a GPU node.

set -e

# Navigate to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Create log directory if it doesn't exist
mkdir -p tests/scripts/logs

echo "=========================================="
echo "torchref GPU Test Suite"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Repository: ${REPO_ROOT}"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules
module load anaconda

# Activate conda environment
conda activate /das/work/p17/p17490/CONDA/torchref

# Check GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "=========================================="
echo "Running GPU tests..."
echo "=========================================="

# Run GPU tests.
#
# ``gpu or cuda`` rather than ``gpu`` alone: a mark expression *deselects* at collection
# time, so ``-m "gpu"`` never even collected the `cuda`-marked tests -- which is most of the
# accelerator coverage, including every direct-call kernel leg and the Triton provenance and
# gate tests. They were silently absent from this job rather than failing in it.
#
# ``mps`` is deliberately not included: this is a CUDA cluster, and those tests are skipped
# by ``pytest_collection_modifyitems`` on a host without Metal anyway.
python -m pytest tests/ \
    -m "gpu or cuda" \
    --cov=torchref \
    --cov-report=term-missing \
    -v \
    --tb=short

EXIT_CODE=$?

echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "=========================================="

exit ${EXIT_CODE}
