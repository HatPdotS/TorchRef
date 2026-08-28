#!/bin/bash
#SBATCH --job-name=seamgate
#SBATCH --output=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.out
#SBATCH --error=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement/alignment_lab/slurm/%x_%j.err
#SBATCH --partition=hour
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --constraint=cpu_epyc9335
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
# Single-threaded: the FRF peak list only reproduces bit-for-bit at one thread
# (~5e-8 score noise reorders 12 of 500 peaks on 3GR5 otherwise), and this gate
# is about bit-identity.
export TORCHREF_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "== pytest =="
"$PY" -m pytest tests/unit -q 2>&1 | tail -12
echo "PYTEST_RC=${PIPESTATUS[0]}"
echo "== seam identity =="
"$PY" -u alignment_lab/analysis/seam_identity.py 2>/dev/null | grep -E "^(case|SEAM|[0-9A-Z]{4} )"
echo "RC=$?"
