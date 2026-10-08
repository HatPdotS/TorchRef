#!/bin/bash
# Quick interactive test runner for torchref
# Run this with: srun -c 8 -p day -t 1-00:00:00 tests/scripts/run_tests.sh [options]
#
# Usage:
#   ./tests/scripts/run_tests.sh              # Run every test this host supports
#   ./tests/scripts/run_tests.sh unit         # Run unit tests only
#   ./tests/scripts/run_tests.sh integration  # Run integration tests only
#   ./tests/scripts/run_tests.sh -k "test_name" # Run specific test
#   ./tests/scripts/run_tests.sh --cov        # Run with coverage

set -e

# Navigate to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Load modules
module load anaconda 2>/dev/null || true

# Activate conda environment
conda activate /das/work/p17/p17490/CONDA/torchref

# Parse arguments
TEST_PATH="tests/"
PYTEST_ARGS="-v --tb=short"

for arg in "$@"; do
    case $arg in
        unit)
            TEST_PATH="tests/unit"
            ;;
        integration)
            TEST_PATH="tests/integration"
            ;;
        --cov)
            PYTEST_ARGS="${PYTEST_ARGS} --cov=torchref --cov-report=term-missing"
            ;;
        -k*)
            PYTEST_ARGS="${PYTEST_ARGS} ${arg}"
            ;;
        *)
            # Pass through any other arguments to pytest
            PYTEST_ARGS="${PYTEST_ARGS} ${arg}"
            ;;
    esac
done

echo "Running tests: ${TEST_PATH}"
echo "Arguments: ${PYTEST_ARGS}"
echo "=========================================="

# Run pytest
python -m pytest ${TEST_PATH} ${PYTEST_ARGS}
