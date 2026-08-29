#!/bin/bash
# Deliverable 1 gate: the Chebyshev extraction must be inert.
#SBATCH --job-name=scalegate
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export TORCHREF_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$PY" -m pytest -c tests/pytest.ini tests/unit/scaling tests/unit/refinement --run-slow -q 2>&1 | tail -8
echo "PYTEST_RC=${PIPESTATUS[0]}"
