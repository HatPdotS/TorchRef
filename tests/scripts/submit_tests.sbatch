#!/bin/bash
#SBATCH --job-name=torchref-tests
#SBATCH --output=tests/scripts/logs/test_%j.out
#SBATCH --error=tests/scripts/logs/test_%j.err
#SBATCH --partition=day
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# torchref Test Suite - SLURM Submission Script
# ==============================================
#
# Usage:
#   sbatch tests/scripts/submit_tests.sbatch [test_type]
#
# Arguments:
#   test_type: unit, integration, all (default: unit)
#
# Examples:
#   sbatch tests/scripts/submit_tests.sbatch unit
#   sbatch tests/scripts/submit_tests.sbatch integration
#   sbatch tests/scripts/submit_tests.sbatch all

set -e

# Get test type from argument (default: unit)
TEST_TYPE=${1:-unit}

# Navigate to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Create log directory if it doesn't exist
mkdir -p tests/scripts/logs

echo "=========================================="
echo "torchref Test Suite"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Test Type: ${TEST_TYPE}"
echo "Repository: ${REPO_ROOT}"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules
module load anaconda

# Activate conda environment
conda activate /das/work/p17/p17490/CONDA/torchref

# Run tests based on type
case ${TEST_TYPE} in
    unit)
        echo "Running unit tests..."
        python -m pytest tests/unit \
            -m "unit and not gpu" \
            --cov=torchref \
            --cov-report=term-missing \
            --cov-report=html:tests/scripts/logs/coverage_html \
            -v \
            --tb=short
        ;;
    
    integration)
        echo "Running integration tests..."
        python -m pytest tests/integration \
            -m "integration and not gpu" \
            --cov=torchref \
            --cov-report=term-missing \
            -v \
            --tb=short
        ;;
    
    all)
        echo "Running all tests..."
        python -m pytest tests/ \
            -m "not gpu" \
            --cov=torchref \
            --cov-report=term-missing \
            --cov-report=html:tests/scripts/logs/coverage_html \
            -v \
            --tb=short
        ;;
    
    *)
        echo "Unknown test type: ${TEST_TYPE}"
        echo "Valid options: unit, integration, all"
        exit 1
        ;;
esac

EXIT_CODE=$?

echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "=========================================="

exit ${EXIT_CODE}
