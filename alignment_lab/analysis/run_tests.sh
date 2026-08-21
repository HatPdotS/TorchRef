#!/bin/bash
# Run the test suite on a compute node. The login node is shared and slow enough
# that a cold import alone can take minutes; and scripts a job needs must live on
# /das, not in a node-local /tmp scratch directory the compute node cannot see.
#
#   sbatch --partition=hour --time=00:55:00 --cpus-per-task=8 --mem=32G \
#          --constraint=cpu_epyc9335 alignment_lab/analysis/run_tests.sh
#   sbatch ... alignment_lab/analysis/run_tests.sh --run-slow
#SBATCH --job-name=frf_tests
#SBATCH --output=alignment_lab/slurm/%x_%j.out
#SBATCH --error=alignment_lab/slurm/%x_%j.err
set -uo pipefail
REPO="${FRF_TEST_REPO:-/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement}"
PY="$REPO/.dev/bin/python"
[ -x "$PY" ] || PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export TORCHREF_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$TORCHREF_NUM_THREADS" MKL_NUM_THREADS="$TORCHREF_NUM_THREADS"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
mkdir -p alignment_lab/slurm
echo "host=$(hostname) sha=$(git -C "$REPO" rev-parse --short HEAD) threads=$TORCHREF_NUM_THREADS"
"$PY" -m pytest tests/unit alignment_lab/tests -q "$@"
rc=$?
echo "exit_code=$rc"
exit "$rc"
