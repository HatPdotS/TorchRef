#!/bin/bash
# Gate for routing the rescore's epsilon through SpaceGroup.epsilon(friedel=False).
# This CHANGES the LLG on every centric reflection, so the unit suite is necessary
# but not sufficient -- the rescore panel is the real test.
#SBATCH --job-name=epsgate
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
cd "$REPO"; export PYTHONPATH="$REPO"
export TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "host=$(hostname)"
L=alignment_lab/slurm/epsgate_tests_$SLURM_JOB_ID.log
"$PY" -m pytest tests/unit/symmetry tests/unit/alignment tests/unit/frf_separate tests/unit/model -q > "$L" 2>&1
rc=$?; echo "=== TESTS rc=$rc ==="; tail -12 "$L"; grep "^FAILED" "$L" | head -8
