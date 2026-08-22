#!/bin/bash
# Test gate for the build_grid change. rc is captured immediately after pytest,
# not after a pipe: $? on a pipeline reads the LAST command, which silently
# reports success for a failed test run.
#SBATCH --job-name=frf_tests
#SBATCH --output=alignment_lab/slurm/%x_%j.out
#SBATCH --error=alignment_lab/slurm/%x_%j.err
set -uo pipefail
REPO=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/alignement
PY=/das/work/units/LBR-FEL/p17490/Peter/Library/work_trees_torchref/dev/.dev/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" TORCHREF_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
mkdir -p alignment_lab/slurm
echo "host=$(hostname)"

echo "###### fast: model + alignment + frf_separate"
"$PY" -m pytest -q --no-header -p no:cacheprovider \
    tests/unit/model tests/unit/alignment tests/unit/frf_separate \
    > alignment_lab/slurm/_t_fast.log 2>&1
rc_fast=$?
tail -4 alignment_lab/slurm/_t_fast.log
echo "rc_fast=$rc_fast"

echo "###### slow-included: alignment + frf_separate"
"$PY" -m pytest -q --no-header -p no:cacheprovider --run-slow \
    tests/unit/alignment tests/unit/frf_separate \
    > alignment_lab/slurm/_t_slow.log 2>&1
rc_slow=$?
tail -4 alignment_lab/slurm/_t_slow.log
echo "rc_slow=$rc_slow"

echo "###### failures, if any"
grep -E "^(FAILED|ERROR)" alignment_lab/slurm/_t_fast.log alignment_lab/slurm/_t_slow.log || echo "  none"
echo "done"
