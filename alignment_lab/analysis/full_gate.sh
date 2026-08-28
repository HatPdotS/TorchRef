#!/bin/bash
#SBATCH --job-name=fullgate
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
"$PY" -m pytest tests/unit -q 2>&1 | tail -14
echo "PYTEST_RC=${PIPESTATUS[0]}"
echo "== seam identity =="
"$PY" -u alignment_lab/analysis/seam_identity.py 2>/dev/null | grep -E "SEAM|conv|case|^ *[0-9A-Z]"
