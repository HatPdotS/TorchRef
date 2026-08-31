#!/bin/bash
#SBATCH --job-name=refgate
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
echo "== import smoke =="
"$PY" -m pytest -c tests/pytest.ini tests/unit/test_imports_smoke.py --run-slow -q 2>&1 | tail -5
echo "SMOKE_RC=${PIPESTATUS[0]}"
echo "== alignment + frf_separate + scaling =="
"$PY" -m pytest -c tests/pytest.ini tests/unit/alignment tests/unit/frf_separate tests/unit/scaling --run-slow -q 2>&1 | tail -20
echo "SCOPED_RC=${PIPESTATUS[0]}"
